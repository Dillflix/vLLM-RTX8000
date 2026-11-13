// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>

namespace glm_lossless_cutlass_v29 {
template <class OriginalIterator, int FixedWidth = 0, bool Full64 = false>
class MixedBIterator {
 public:
  using Shape = typename OriginalIterator::Shape;
  using Element = typename OriginalIterator::Element;
  using Layout = typename OriginalIterator::Layout;
  using TensorRef = typename OriginalIterator::TensorRef;
  using TensorCoord = typename OriginalIterator::TensorCoord;
  using ThreadMap = typename OriginalIterator::ThreadMap;
  using Fragment = typename OriginalIterator::Fragment;
  using AccessType = typename OriginalIterator::AccessType;
  static int const kAdvanceRank = 0;
  static_assert(ThreadMap::kElementsPerAccess == 8 &&
                AccessType::kElements == 8);
  static_assert(Shape::kRow == 32 && ThreadMap::Iterations::kContiguous == 1);
  static_assert(Shape::kColumn <= 256 && 256 % Shape::kColumn == 0);
  struct Params {
    uint64_t const* metadata = nullptr;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE explicit Params(Layout const&) {}
  };

 private:
  static constexpr int S = ThreadMap::Iterations::kStrided;
  uint8_t const* low_[S];
  int tile_bytes_[S], high_delta_[S];
  bool valid_n_[S];
  int k_, local_k_, extent_k_, width_;
  bool enabled_ = true;
  CUTLASS_DEVICE static uint16_t decode(uint32_t code) {
    uint32_t payload = code & 2047;
    int shift = max(0, min(3, int(payload >> 7) - 1));
    return uint16_t(((payload - (uint32_t(shift) << 7)) << shift) |
                    ((code & 2048) << 4));
  }

 public:
  CUTLASS_DEVICE MixedBIterator(Params const& params, Element* pointer,
                                TensorCoord extent, int thread_id,
                                TensorCoord offset, int const* = nullptr) {
    auto initial = ThreadMap::initial_offset(thread_id);
    k_ = offset.row() + initial.contiguous();
    local_k_ = k_ & 31;
    extent_k_ = extent.row();
    // Each BN=128/256 CTA remains inside a single 256-row representation group.
    int first_n = offset.column() + initial.strided();
    int group = offset.column() / 256;
    uint64_t group_offset = params.metadata[group * 2];
    width_ = FixedWidth ? FixedWidth : int(params.metadata[group * 2 + 1]);
    auto* payload = reinterpret_cast<uint8_t const*>(pointer) + group_offset;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
      int n = first_n + s * ThreadMap::Delta::kStrided;
      int tile_n = n & ~63;
      int rows = Full64 ? 64 : min(64, extent.column() - tile_n);
      valid_n_[s] = Full64 || n < extent.column();
      // Invalid tail lanes never dereference these pointers.
      rows = max(0, rows);
      uint64_t before =
          uint64_t(tile_n & 255) * extent_k_ + uint64_t(k_ & ~31) * rows;
      low_[s] = payload + before * width_ / 8 +
                (n & 63) * (width_ == 16 ? 64 : 16) +
                (width_ == 16 ? local_k_ * 2 : local_k_ / 2);
      tile_bytes_[s] = rows * 32 * width_ / 8;
      high_delta_[s] = rows * 16;
    }
  }
  CUTLASS_DEVICE MixedBIterator& operator++() {
    k_ += 32;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) low_[s] += tile_bytes_[s];
    return *this;
  }
  CUTLASS_DEVICE void clear_mask(bool enable = true) {
    if (enable) enabled_ = false;
  }
  struct CompressedFragment {
    uint4 words[S];
  };
  CUTLASS_DEVICE void load_compressed(CompressedFragment& raw) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < S; ++s) {
      uint4 value = {0, 0, 0, 0};
      if (enabled_ && k_ + 7 < extent_k_ && valid_n_[s]) {
        if (width_ == 16) {
          value = *reinterpret_cast<uint4 const*>(low_[s]);
        } else {
          auto* base = low_[s];
          value.x = *reinterpret_cast<uint32_t const*>(base);
          value.y = *reinterpret_cast<uint32_t const*>(base + high_delta_[s]);
          value.z =
              *reinterpret_cast<uint32_t const*>(base + 2 * high_delta_[s]);
        }
      }
      raw.words[s] = value;
    }
  }
  template <int Begin, int End>
  CUTLASS_DEVICE void unpack(CompressedFragment const& raw,
                             Fragment& fragment) {
    auto* dst = reinterpret_cast<uint32_t*>(&fragment);
    CUTLASS_PRAGMA_UNROLL
    for (int s = Begin; s < End; ++s) {
      uint4 value = raw.words[s];
      if (width_ == 16) {
        dst[s * 4] = value.x;
        dst[s * 4 + 1] = value.y;
        dst[s * 4 + 2] = value.z;
        dst[s * 4 + 3] = value.w;
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int pair = 0; pair < 4; ++pair) {
          uint32_t source = pair % 2 ? value.y : value.x;
          uint32_t encoded;
          if (pair < 2) {
            encoded = source & 0xbff8bff8u;
          } else {
            uint32_t low = __byte_perm(value.z, 0u, pair == 2 ? 0x4140 : 0x4342)
                           << 3;
            encoded = low | ((source & 0x00070007u) << 11) |
                      ((source & 0x40004000u) << 1);
          }
          uint32_t decoded;
          asm("mul.rn.f16x2 %0, %1, %2;"
              : "=r"(decoded)
              : "r"(encoded), "r"(0x30003000u));
          dst[s * 4 + pair] = decoded;
        }
      }
    }
  }
  static constexpr int kSplit = S / 2;
  static constexpr int kVectors = S;
  CUTLASS_DEVICE void load(Fragment& fragment) {
    CompressedFragment raw;
    load_compressed(raw);
    unpack<0, S>(raw, fragment);
  }
};
}  // namespace glm_lossless_cutlass_v29
