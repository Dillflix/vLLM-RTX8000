// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

extern "C" int dense75_mma_launch(const void*, const void*, void*, int, int,
                                  int, int, void*);

// Weight rows are contiguous; each warp produces a column of up to 20 tokens.
// Keep FP32 accumulation and the original FP16 weights.
template <int M>
__global__ void gemv(const half* __restrict__ x, const half* __restrict__ w,
                     half* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (n >= N) return;
  float sums[M] = {};
  for (int k = lane * 8; k < K; k += 256) {
    const int4 packed_w = *reinterpret_cast<const int4*>(w + n * K + k);
    const half* wh = reinterpret_cast<const half*>(&packed_w);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const int4 packed_x = *reinterpret_cast<const int4*>(x + m * K + k);
      const half* xh = reinterpret_cast<const half*>(&packed_x);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sums[m] = fmaf(__half2float(wh[i]), __half2float(xh[i]), sums[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int step = 16; step; step /= 2)
      sums[m] += __shfl_down_sync(0xffffffff, sums[m], step);
    if (!lane) y[m * N + n] = __float2half(sums[m]);
  }
}

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
