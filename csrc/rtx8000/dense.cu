// SPDX-License-Identifier: Apache-2.0
#include "dense_gemv_kernel.cuh"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

extern "C" int dense75_mma_launch(const void*, const void*, void*, int, int,
                                  int, int, void*);

#define RTX_DENSE_GEMV_EXTRA(M)                                               \
  extern "C" int dense75_gemv_m##M(const void*, const void*, void*, int, int, \
                                   int, void*);
#include "dense_gemv_extra_variants.inc"
#undef RTX_DENSE_GEMV_EXTRA

template <int BN, int BK, int WN>
int tensor_gemm(const void* x, const void* w, void* y, int M, int N, int K,
                cudaStream_t stream) {
  using H = cutlass::half_t;
  using G = cutlass::gemm::device::Gemm<
      H, cutlass::layout::RowMajor, H, cutlass::layout::ColumnMajor, H,
      cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75, cutlass::gemm::GemmShape<32, BN, BK>,
      cutlass::gemm::GemmShape<32, WN, BK>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<H, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;
  typename G::Arguments args(
      {M, N, K}, {static_cast<const H*>(x), K}, {static_cast<const H*>(w), K},
      {static_cast<const H*>(y), N}, {static_cast<H*>(y), N}, {1.0f, 0.0f});
  G op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return int(status);
  return int(op(args, nullptr, stream));
}

extern "C" int dense75_launch(const void* x, const void* w, void* y, int M,
                              int N, int K, int variant, void* stream_ptr) {
  if (variant >= 4)
    return dense75_mma_launch(x, w, y, M, N, K, variant, stream_ptr);
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  if (variant == 2) return tensor_gemm<64, 64, 32>(x, w, y, M, N, K, stream);
  if (variant == 3) return tensor_gemm<128, 64, 32>(x, w, y, M, N, K, stream);
  const int threads = variant == 0 ? 128 : 256;
  // Eight-element vector loads require all rows to have aligned length.
  if (K % 8) return -2;
#define CASE_M(m)                                                             \
  case m:                                                                     \
    gemv<m><<<(N + threads / 32 - 1) / (threads / 32), threads, 0, stream>>>( \
        static_cast<const half*>(x), static_cast<const half*>(w),             \
        static_cast<half*>(y), N, K);                                         \
    break
  switch (M) {
#define RTX_DENSE_GEMV_EXTRA(m) \
  case m:                       \
    return dense75_gemv_m##m(x, w, y, N, K, variant, stream_ptr);
#include "dense_gemv_extra_variants.inc"
#undef RTX_DENSE_GEMV_EXTRA
    CASE_M(1);
    CASE_M(2);
    CASE_M(4);
    CASE_M(5);
    CASE_M(8);
    CASE_M(12);
    CASE_M(16);
    CASE_M(20);
    default:
      return -1;
  }
#undef CASE_M
  return int(cudaGetLastError());
}
