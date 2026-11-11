# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an isolated paired-column Marlin prototype without editing its source."""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = (
    "859b4dc5d242c0a7b1799680eca44c13a5fbfc319738d29c429b357d7b6d7d54"
)


def paired_column(column: int) -> int:
    return column // 128 * 64 + (column % 128 // 64) * 320 + column % 64


def audit_layout() -> dict:
    # The existing repacker independently packs each16x64 tile. Pairing whole
    # 64-column tiles preserves every within-tile bit permutation.
    mapped = [paired_column(i) for i in range(640)]
    assert sorted(mapped) == list(range(640))
    for tile in range(5):
        for lane in range(128):
            original = tile * 128 + lane
            address = original + (lane // 64) * 256 - tile * 64
            assert address == paired_column(original)
            assert address // 64 == (tile if lane < 64 else tile + 5)
            assert address % 64 == lane % 64
        for vector in range(16):
            original = tile * 16 + vector
            address = original + (vector // 8) * 32 - tile * 8
            for half in range(8):
                assert address * 8 + half == paired_column(original * 8 + half)
    return {
        "weight_layout": "independent16x64 Marlin packed tiles",
        "verified_weight_columns": 640,
        "verified_scale_elements": 640,
        "weight_or_scale_requantization": False,
        "precision": "unchanged local MMA, FP32 split-K reduction, FP16 rounding",
        "traffic_saved_bytes_at_2048": 2 * 2048 * 10 * 640 * 2,
    }


def generate(source: Path, output: Path) -> dict:
    text = source.read_text()
    original_hash = hashlib.sha256(text.encode()).hexdigest()
    if original_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("Changed Marlin template requires reviewing the fusion patch")
    assert text.count("&B[b_gl_idx]") == 1
    assert text.count("&scales_ptr[s_gl_rd]") == 3
    text = text.replace(
        "&B[b_gl_idx]",
        "&B[b_gl_idx + (threadIdx.x / 64) * 256 - slice_col * 64]",
    )
    text = text.replace(
        "&scales_ptr[s_gl_rd]",
        "&scales_ptr[s_gl_rd + (threadIdx.x / 8) * 32 - slice_col * 8]",
    )
    marker = (
        "  constexpr int moe_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);"
    )
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        marker
        + """
  static_assert(a_type_id == vllm::kFloat16.id() &&
                b_type_id == vllm::kU8B128.id() &&
                c_type_id == vllm::kFloat16.id() &&
                s_type_id == vllm::kFloat16.id());
  static_assert(threads == 128 && thread_n_blocks == 8 &&
                thread_k_blocks == 4 && !m_block_size_8 && stages == 2 &&
                group_blocks == -1 && !is_zp_float);
""",
    )
    start = text.index(
        "  #pragma unroll\n    for (int i = 0;\n         i < div_ceil",
        text.index("  auto write_result ="),
    )
    end = text.index("\n    __syncthreads();\n  };", start)
    text = (
        text[:start]
        + """
  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      int row = c_gl_wr / c_gl_stride;
      constexpr int half_tile_vectors = thread_n_blocks;
      int vector_col = threadIdx.x % (2 * thread_n_blocks);
      if (row < block_num_valid_tokens && vector_col < half_tile_vectors) {
        int64_t sorted_row = sh_block_sorted_ids[row];
        int4 result;
        half* out_half = reinterpret_cast<half*>(&result);
        const half* gate = reinterpret_cast<const half*>(&sh_red[c_sh_rd]);
        const half* up = reinterpret_cast<const half*>(
            &sh_red[c_sh_rd + half_tile_vectors]);
  #pragma unroll
        for (int j = 0; j < 8; j++) {
          // Preserve both original rounding points: GEMM output is already
          // FP16 in sh_red, then SiLU is rounded before multiplication by up.
          const float g = __half2float(gate[j]);
          const half activated = __float2half_rn(g / (1.0f + expf(-g)));
          out_half[j] = __float2half_rn(
              __half2float(activated) * __half2float(up[j]));
        }
        C[sorted_row * (prob_n / 16) + slice_col * half_tile_vectors +
          vector_col] = result;
      }
      c_gl_wr += c_gl_wr_delta;
      c_sh_rd += c_sh_rd_delta;
    }
"""
        + text[end:]
    )
    # The split-K reduction, lock protocol, MMA and FP16 conversion code have
    # not been rewritten. Only paired reads and the final-completion store vary.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return {
        "source": str(source),
        "source_sha256": original_hash,
        "generated": str(output),
        "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "layout_audit": audit_layout(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.manifest.write_text(json.dumps(generate(args.source, args.output), indent=2))
