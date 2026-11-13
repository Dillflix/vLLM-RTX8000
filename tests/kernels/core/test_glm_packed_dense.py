# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing preserves every source bit, including raw16 exception groups."""

import numpy as np
import pytest

from vllm.models.glm5next.nvidia.ops.packed_dense_codec import (
    pack_mixed_tiles,
    unpack_mixed_tiles,
)


@pytest.mark.parametrize("shape", [(64, 64), (288, 128), (520, 192), (544, 256)])
def test_packed_dense_preserves_all_codes_raw_exceptions_and_n_tails(shape):
    rng = np.random.default_rng(183331)
    codes = rng.integers(0, 4096, size=shape, dtype=np.uint32)
    codes.flat[:4096] = np.arange(4096)
    magnitude = codes & 2047
    shift = np.clip((magnitude >> 7).astype(np.int32) - 1, 0, 3).astype(np.uint32)
    words = (((magnitude - (shift << 7)) << shift) | ((codes & 2048) << 4)).astype(
        np.uint16
    )
    if shape[0] > 256:
        words[256:] = rng.integers(0, 65536, size=words[256:].shape, dtype=np.uint16)
        words[256, :6] = [0, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0xFE55]
    payload, metadata = pack_mixed_tiles(words)
    assert np.array_equal(unpack_mixed_tiles(payload, metadata, shape), words)
    assert metadata[0, 1] == 12
    if shape[0] > 256:
        assert np.all(metadata[1:, 1] == 16)
    expected_bytes = sum(
        min(256, shape[0] - i * 256) * shape[1] * int(width) // 8
        for i, (_, width) in enumerate(metadata)
    )
    assert payload.nbytes == expected_bytes


@pytest.mark.parametrize("shape", [(0, 64), (64, 0), (64, 63)])
def test_packed_dense_rejects_empty_or_unaligned_weights(shape):
    with pytest.raises(ValueError):
        pack_mixed_tiles(np.empty(shape, dtype=np.uint16))
