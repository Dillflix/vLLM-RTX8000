// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gemm.cuh"
namespace glm_lossless_cutlass_v26 {
template <int BM, int BN>
struct RepresentationKernel {
  using K12 = typename Types<BM, BN, true, 12>::Kernel;
  using K16 = typename Types<BM, BN, true, 16>::Kernel;
  using F12 = typename Types<BM, BN, true, 12, true>::Kernel;
  using F16 = typename Types<BM, BN, true, 16, true>::Kernel;
  struct Params {
    typename K12::Params p12;
    typename K16::Params p16;
    typename F12::Params f12;
    typename F16::Params f16;
  };
  union SharedStorage {
    typename K12::SharedStorage s12;
    typename K16::SharedStorage s16;
    typename F12::SharedStorage f12;
    typename F16::SharedStorage f16;
  };
  static constexpr int kThreadCount = K12::kThreadCount;
  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& storage) {
    typename K12::ThreadblockSwizzle swizzle;
    auto coord = swizzle.get_tile_offset(params.p12.swizzle_log_tile);
    // The swizzled grid can include padding CTAs. Do not read metadata for
    // them.
    if (coord.m() >= params.p12.grid_tiled_shape.m() ||
        coord.n() >= params.p12.grid_tiled_shape.n())
      return;
    int group = coord.n() * BN / 256;
    int width = int(params.p12.params_B.metadata[group * 2 + 1]);
    if ((coord.n() + 1) * BN <= params.p12.problem_size.n()) {
      if (width == 12) {
        F12 op;
        op(params.f12, storage.f12);
      } else {
        F16 op;
        op(params.f16, storage.f16);
      }
      return;
    }
    if (width == 12) {
      K12 op;
      op(params.p12, storage.s12);
    } else {
      K16 op;
      op(params.p16, storage.s16);
    }
  }
};
template <int BM, int BN>
struct Specialized {
  using Kernel = RepresentationKernel<BM, BN>;
  static constexpr int kShared = sizeof(typename Kernel::SharedStorage);
  inline static bool configured = false;
  static int configure(int* stats) {
    if (!stats) return -1;
    int device, major, minor, maximum, sms;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return int(error);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (major != 7 || minor != 5) return -2;
    cudaDeviceGetAttribute(&maximum, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    static_assert(kShared <= 48 * 1024);
    cudaFuncAttributes attributes;
    error = cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>);
    if (error != cudaSuccess) return int(error);
    int active;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, cutlass::Kernel<Kernel>, Kernel::kThreadCount, kShared);
    if (error != cudaSuccess) return int(error);
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
template <int BM, int BN>
int launch_specialized(void const* x, void const* weight, void const* metadata,
                       void* out, int m, int n, int k, void* stream) {
  if (m < 0 || n <= 0 || k <= 0 || k % 64 || n % 8 || !metadata || !x ||
      !weight || !out || reinterpret_cast<uintptr_t>(x) % 16 ||
      reinterpret_cast<uintptr_t>(weight) % 16 ||
      reinterpret_cast<uintptr_t>(out) % 16)
    return -1;
  if (!m) return 0;
  using S = Specialized<BM, BN>;
  if (!S::configured) return -4;
  using Kernel = typename S::Kernel;
  using K12 = typename Kernel::K12;
  using K16 = typename Kernel::K16;
  using F12 = typename Kernel::F12;
  using F16 = typename Kernel::F16;
  using H = cutlass::half_t;
  using Swizzle = typename K12::ThreadblockSwizzle;
  auto problem = cutlass::gemm::GemmCoord(m, n, k);
  auto tiled = Swizzle().get_tiled_shape(problem, {BM, BN, 32}, 1);
  typename K12::Params p12(
      problem, tiled, {const_cast<H*>(static_cast<H const*>(x)), k},
      {const_cast<H*>(static_cast<H const*>(weight)), k},
      {static_cast<H*>(out), n}, {static_cast<H*>(out), n}, {1.0f, 0.0f});
  typename K16::Params p16(
      problem, tiled, {const_cast<H*>(static_cast<H const*>(x)), k},
      {const_cast<H*>(static_cast<H const*>(weight)), k},
      {static_cast<H*>(out), n}, {static_cast<H*>(out), n}, {1.0f, 0.0f});
  typename F12::Params f12(
      problem, tiled, {const_cast<H*>(static_cast<H const*>(x)), k},
      {const_cast<H*>(static_cast<H const*>(weight)), k},
      {static_cast<H*>(out), n}, {static_cast<H*>(out), n}, {1.0f, 0.0f});
  typename F16::Params f16(
      problem, tiled, {const_cast<H*>(static_cast<H const*>(x)), k},
      {const_cast<H*>(static_cast<H const*>(weight)), k},
      {static_cast<H*>(out), n}, {static_cast<H*>(out), n}, {1.0f, 0.0f});
  p12.params_B.metadata = static_cast<uint64_t const*>(metadata);
  p16.params_B.metadata = static_cast<uint64_t const*>(metadata);
  f12.params_B.metadata = static_cast<uint64_t const*>(metadata);
  f16.params_B.metadata = static_cast<uint64_t const*>(metadata);
  typename Kernel::Params params{p12, p16, f12, f16};
  dim3 grid = Swizzle().get_grid_shape(tiled);
  cutlass::Kernel<Kernel><<<grid, Kernel::kThreadCount, S::kShared,
                            static_cast<cudaStream_t>(stream)>>>(params);
  return int(cudaGetLastError());
}
}  // namespace glm_lossless_cutlass_v26
