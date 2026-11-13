// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include "mixed_b_iterator.cuh"
#include "staged_mma.cuh"

namespace glm_packed_splitk_v2 {

template <int BM, int BN, bool Packed>
struct Types {
  using H = cutlass::half_t;
  using Device = cutlass::gemm::device::Gemm<
      H, cutlass::layout::RowMajor, H, cutlass::layout::ColumnMajor, float,
      cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<BM, BN, 32>,
      cutlass::gemm::GemmShape<(BM == 16 ? 16 : 32), (BN / 2), 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
  using OriginalKernel = typename Device::GemmKernel;
  using OriginalMma = typename OriginalKernel::Mma;
  using MixedMma = cutlass::gemm::threadblock::MmaPipelined<
      typename OriginalMma::Shape, typename OriginalMma::IteratorA,
      typename OriginalMma::SmemIteratorA,
      MixedBIterator<typename OriginalMma::IteratorB>,
      typename OriginalMma::SmemIteratorB, typename OriginalMma::ElementC,
      typename OriginalMma::LayoutC, typename OriginalMma::Policy>;
  using Mma =
      typename cutlass::platform::conditional<Packed, StagedMma<MixedMma>,
                                              OriginalMma>::type;
  using BaseKernel =
      cutlass::gemm::kernel::Gemm<Mma, typename OriginalKernel::Epilogue,
                                  typename OriginalKernel::ThreadblockSwizzle,
                                  false>;
  // Independent FP32 output planes, no serial semaphore or FP16 intermediate.
  struct Kernel : BaseKernel {
    using Params = typename BaseKernel::Params;
    using SharedStorage = typename BaseKernel::SharedStorage;
    CUTLASS_DEVICE void operator()(Params const& params,
                                   SharedStorage& shared) {
      Params local = params;
      int64_t offset = int64_t(blockIdx.z) * params.problem_size.m() *
                       params.problem_size.n();
      local.ref_C.add_pointer_offset(offset);
      local.ref_D.add_pointer_offset(offset);
      BaseKernel::operator()(local, shared);
    }
  };

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

static __global__ void reduce_partials(float const* partials, __half* out,
                                       int count, int splits) {
  int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    float value = partials[i];
    for (int s = 1; s < splits; ++s) value += partials[int64_t(s) * count + i];
    out[i] = __float2half_rn(value);
  }
}

template <int BM, int BN, bool Packed>
int launch(void const* x, void const* weight, void const* metadata, void* out,
           void* workspace, int m, int n, int k, int requested_splits,
           void* stream_pointer) {
  if (m < 0 || m > 512 || n <= 0 || n > 65536 || k <= 0 || k % 64 || n % 8 ||
      requested_splits < 1 || requested_splits > 8 || (Packed && !metadata) ||
      !x || !weight || !out || !workspace ||
      reinterpret_cast<uintptr_t>(x) % 16 ||
      reinterpret_cast<uintptr_t>(weight) % 16 ||
      reinterpret_cast<uintptr_t>(workspace) % 16 ||
      reinterpret_cast<uintptr_t>(out) % 16)
    return -1;
  if (!m) return 0;
  using T = Types<BM, BN, Packed>;
  if (!T::configured) return -4;
  using Kernel = typename T::Kernel;
  using H = typename T::H;
  using Swizzle = typename Kernel::ThreadblockSwizzle;
  // Avoid empty slices for short or uneven K. Caller reserves requested*m*n
  // floats.
  int tiles = k / 32;
  int chunk_tiles = (tiles + requested_splits - 1) / requested_splits;
  int splits = (tiles + chunk_tiles - 1) / chunk_tiles;
  auto problem = cutlass::gemm::GemmCoord(m, n, k);
  auto tiled = Swizzle().get_tiled_shape(problem, {BM, BN, 32}, splits);
  typename Kernel::Params params(
      problem, tiled, {const_cast<H*>(static_cast<H const*>(x)), k},
      {const_cast<H*>(static_cast<H const*>(weight)), k},
      {static_cast<float*>(workspace), n}, {static_cast<float*>(workspace), n},
      {1.0f, 0.0f});
  if constexpr (Packed)
    params.params_B.metadata = static_cast<uint64_t const*>(metadata);
  constexpr int shared = sizeof(typename Kernel::SharedStorage);
  static_assert(shared <= 48 * 1024);
  auto stream = static_cast<cudaStream_t>(stream_pointer);
  dim3 grid = Swizzle().get_grid_shape(tiled);
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, shared, stream>>>(params);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  reduce_partials<<<(m * n + 255) / 256, 256, 0, stream>>>(
      static_cast<float const*>(workspace), static_cast<__half*>(out), m * n,
      splits);
  return int(cudaGetLastError());
}

}  // namespace glm_packed_splitk_v2
