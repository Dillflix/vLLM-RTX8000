// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace glm_packed_splitk_v2 {
template <class Parent>
class StagedMma : public Parent {
 public:
  using Parent::Parent;
  using IteratorA = typename Parent::IteratorA;
  using IteratorB = typename Parent::IteratorB;
  using FragmentA = typename Parent::FragmentA;
  using FragmentB = typename Parent::FragmentB;
  using FragmentC = typename Parent::FragmentC;
  using Operator = typename Parent::Operator;
  static_assert(Parent::kWarpGemmIterations == 4);

  CUTLASS_DEVICE void operator()(int iterations, FragmentC& accum,
                                 IteratorA iterator_A, IteratorB iterator_B,
                                 FragmentC const& src) {
    this->prologue(iterator_A, iterator_B, iterations);
    typename IteratorB::CompressedFragment current_B;
    typename IteratorB::CompressedFragment future_B;
    iterator_B.clear_mask(iterations <= 1);
    iterator_B.load_compressed(current_B);
    ++iterator_B;
    iterator_B.clear_mask(iterations <= 2);
    this->gmem_wait();
    accum = src;
    typename Operator::FragmentA warp_A[2];
    typename Operator::FragmentB warp_B[2];
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_A[0]);
    this->warp_tile_iterator_B_.load(warp_B[0]);
    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    FragmentA tb_A;
    FragmentB tb_B;
    iterator_A.clear_mask(iterations <= 1);
    CUTLASS_GEMM_LOOP
    for (; iterations > 0; --iterations) {
      CUTLASS_PRAGMA_UNROLL
      for (int step = 0; step < 4; ++step) {
        if (step == 3) {
          iterator_B.template unpack<0, IteratorB::kVectors>(current_B, tb_B);
          this->smem_iterator_A_.store(this->transform_A_(tb_A));
          this->smem_iterator_B_.store(this->transform_B_(tb_B));
          current_B = future_B;
          this->gmem_wait();
          this->advance_smem_stages();
        }
        this->warp_tile_iterator_A_.set_kgroup_index((step + 1) % 4);
        this->warp_tile_iterator_B_.set_kgroup_index((step + 1) % 4);
        this->warp_tile_iterator_A_.load(warp_A[(step + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_B[(step + 1) % 2]);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
        if (step == 0) {
          tb_A.clear();
          iterator_A.load(tb_A);
          ++iterator_A;
          iterator_B.load_compressed(future_B);
          ++iterator_B;
          iterator_A.clear_mask(iterations <= 2);
          iterator_B.clear_mask(iterations <= 3);
        }
        this->warp_mma(accum, warp_A[step % 2], warp_B[step % 2], accum);
      }
    }
  }
};
}  // namespace glm_packed_splitk_v2
