# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact 12-bit storage: adjacent low/high planes in each N64 by K64 tile."""

import numpy as np

from .arithmetic12_planes import decode_planes, planes_to_row12, row12_to_planes

TILE_N = 64
TILE_K = 64


def _shape(rows):
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] % 96:
        raise ValueError("Expected rowwise 12-bit bytes with K divisible by 64")
    return rows.shape[0], rows.shape[1] * 2 // 3


def row12_to_tileplanes(rows):
    n, k = _shape(rows)
    output = np.empty(rows.size, dtype=np.uint8)
    for first_n in range(0, n, TILE_N):
        r = min(TILE_N, n - first_n)
        low, high = row12_to_planes(rows[first_n : first_n + r])
        low = low.reshape(r, k // TILE_K, TILE_K).transpose(1, 0, 2)
        high = high.reshape(r, k // TILE_K, TILE_K // 2).transpose(1, 0, 2)
        tiles = output[first_n * k * 3 // 2 : (first_n + r) * k * 3 // 2]
        tiles = tiles.reshape(k // TILE_K, r * TILE_K * 3 // 2)
        tiles[:, : r * TILE_K] = low.reshape(k // TILE_K, -1)
        tiles[:, r * TILE_K :] = high.reshape(k // TILE_K, -1)
    return output


def tileplanes_to_row12(packed, shape):
    n, row_bytes = shape
    if packed.dtype != np.uint8 or packed.ndim != 1 or packed.size != n * row_bytes:
        raise ValueError("Wrong tile-plane storage size or dtype")
    output = np.empty(shape, dtype=np.uint8)
    _, k = _shape(output)
    for first_n in range(0, n, TILE_N):
        r = min(TILE_N, n - first_n)
        tiles = packed[first_n * k * 3 // 2 : (first_n + r) * k * 3 // 2]
        tiles = tiles.reshape(k // TILE_K, r * TILE_K * 3 // 2)
        low = tiles[:, : r * TILE_K].reshape(k // TILE_K, r, TILE_K)
        high = tiles[:, r * TILE_K :].reshape(k // TILE_K, r, TILE_K // 2)
        low = low.transpose(1, 0, 2).copy().reshape(-1)
        high = high.transpose(1, 0, 2).copy().reshape(-1)
        output[first_n : first_n + r] = planes_to_row12(low, high, (r, row_bytes))
    return output


def native_vector_words(packed, n, k, first_n, first_k, vector):
    """Independent Python spelling of the exact native pointer arithmetic."""
    r = min(TILE_N, n - first_n)
    local = vector * 16
    if local >= r * TILE_K:
        raise ValueError("Padded N lanes have no encoded storage")
    base = (first_n * k + first_k * r) * 3 // 2
    low = base + local
    high = base + r * TILE_K + local // 2
    return decode_planes(packed[low : low + 16], packed[high : high + 8])
