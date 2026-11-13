# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rearrange exact12-bit codes into coalesced low-byte/high-nibble planes."""

import numpy as np


def row12_to_planes(rows):
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] % 3:
        raise ValueError("Expected contiguous rowwise12-bit storage")
    triplets = np.ascontiguousarray(rows).reshape(-1, 3)
    low = np.empty(triplets.shape[0] * 2, dtype=np.uint8)
    low[0::2] = triplets[:, 0]
    low[1::2] = (triplets[:, 1] >> 4) | ((triplets[:, 2] & 15) << 4)
    high = (triplets[:, 1] & 15) | (triplets[:, 2] & 240)
    return low, high


def planes_to_row12(low, high, shape):
    if low.dtype != np.uint8 or high.dtype != np.uint8 or low.size != high.size * 2:
        raise ValueError("Invalid low-byte/high-nibble planes")
    low, high = low.reshape(-1), high.reshape(-1)
    out = np.empty((high.size, 3), dtype=np.uint8)
    out[:, 0] = low[0::2]
    out[:, 1] = (high & 15) | ((low[1::2] & 15) << 4)
    out[:, 2] = (low[1::2] >> 4) | (high & 240)
    return out.reshape(shape)


def decode_planes(low, high):
    low, high = low.reshape(-1).astype(np.uint32), high.reshape(-1).astype(np.uint32)
    codes = np.empty(low.size, dtype=np.uint32)
    codes[0::2] = low[0::2] | ((high & 15) << 8)
    codes[1::2] = low[1::2] | ((high >> 4) << 8)
    payload = codes & 2047
    shift = np.clip((payload.astype(np.int32) >> 7) - 1, 0, 3).astype(np.uint32)
    return (((payload - (shift << 7)) << shift) | ((codes & 2048) << 4)).astype(
        np.uint16
    )
