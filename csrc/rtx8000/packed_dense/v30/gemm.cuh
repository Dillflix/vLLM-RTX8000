// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include "mixed_b_iterator.cuh"
#include "staged_mma.cuh"

namespace glm_lossless_cutlass_v30 {

template <int BM, int BN, bool Packed, int FixedWidth = 0, bool Full64 = false>
struct Types {
  using H = cutlass::half_t;
  using Device = cutlass::gemm::device::Gemm<
      H, cutlass::layout::RowMajor, H, cutlass::layout::ColumnMajor, H,
      cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<BM, BN, 32>,
      cutlass::gemm::GemmShape<32, (BN == 64 ? 32 : 64), 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<H, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
  using OriginalKernel = typename Device::GemmKernel;
  using OriginalMma = typename OriginalKernel::Mma;
  using MixedMma = cutlass::gemm::threadblock::MmaPipelined<
      typename OriginalMma::Shape, typename OriginalMma::IteratorA,
      typename OriginalMma::SmemIteratorA,
      MixedBIterator<typename OriginalMma::IteratorB, FixedWidth, Full64>,
      typename OriginalMma::SmemIteratorB, typename OriginalMma::ElementC,
      typename OriginalMma::LayoutC, typename OriginalMma::Policy>;
  using Mma =
      typename cutlass::platform::conditional<Packed, StagedMma<MixedMma>,
                                              OriginalMma>::type;
  using Kernel =
      cutlass::gemm::kernel::Gemm<Mma, typename OriginalKernel::Epilogue,
                                  typename OriginalKernel::ThreadblockSwizzle,
                                  false>;
  static constexpr int kShared = sizeof(typename Kernel::SharedStorage);
  static_assert(Kernel::kThreadCount <= 256);
  static_assert(kShared <= 64 * 1024);
  inline static bool configured = false;

  // Private single-device harness initialization, always outside measured
  // calls. stats:
  // registers,threads,dynamicshared,activeCTAs/SM,maxoptinshared,SMcount.
  static int configure(int* stats) {
    if (!stats) return -1;
    int device, major, minor, maximum, sms;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return int(status);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (major != 7 || minor != 5) return -2;
    cudaDeviceGetAttribute(&maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (kShared > maximum) return -3;
    if constexpr (kShared > 48 * 1024) {
      status = cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kShared);
      if (status != cudaSuccess) return int(status);
    }
    cudaFuncAttributes attributes;
    status = cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>);
    if (status != cudaSuccess) return int(status);
    int active;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, cutlass::Kernel<Kernel>, Kernel::kThreadCount, kShared);
    if (status != cudaSuccess) return int(status);
    stats[0] = attributes.numRegs;
    stats[1] = Kernel::kThreadCount;
    stats[2] = kShared;
    stats[3] = active;
    stats[4] = maximum;
    stats[5] = sms;
    configured = true;
    return 0;
  }
};

template <int BM, int BN, bool Packed, int FixedWidth = 0, bool Full64 = false>
int launch(void const* x, void const* weight, void const* metadata, void* out,
           int m, int n, int k, void* stream_pointer) {
  if (m < 0 || n <= 0 || k <= 0 || k % 64 || n % 8 || (Packed && !metadata) ||
      !x || !weight || !out || reinterpret_cast<uintptr_t>(x) % 16 ||
      reinterpret_cast<uintptr_t>(weight) % 16 ||
      reinterpret_cast<uintptr_t>(out) % 16)
    return -1;
  if (!m) return 0;
  using T = Types<BM, BN, Packed>;
  if (!T::configured) return -4;
  using Kernel = typename T::Kernel;
  using H = typename T::H;
  using Swizzle = typename Kernel::ThreadblockSwizzle;
  auto problem = cutlass::gemm::GemmCoord(m, n, k);
  auto tiled = Swizzle().get_tiled_shape(problem, {BM, BN, 32}, 1);
  typename Kernel::Params params(
      problem, tiled, {const_cast<H*>(static_cast<H const*>(x)), k},
      {const_cast<H*>(static_cast<H const*>(weight)), k},
      {static_cast<H*>(out), n}, {static_cast<H*>(out), n}, {1.0f, 0.0f});
  if constexpr (Packed)
    params.params_B.metadata = static_cast<uint64_t const*>(metadata);
  constexpr int shared = sizeof(typename Kernel::SharedStorage);
  static_assert(shared <= 48 * 1024);
  dim3 grid = Swizzle().get_grid_shape(tiled);
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, shared,
         static_cast<cudaStream_t>(stream_pointer)>>>(params);
  return int(cudaGetLastError());
}

}  // namespace glm_lossless_cutlass_v30
