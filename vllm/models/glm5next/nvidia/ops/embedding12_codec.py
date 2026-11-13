# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-preserving12-bit dictionary storage; no quantization or numeric casting."""

import numpy as np


def make_dictionary(bitpatterns: np.ndarray) -> np.ndarray:
    if bitpatterns.dtype != np.uint16:
        raise TypeError("Dictionary input must contain exact16-bit words")
    present = np.zeros(65536, dtype=np.bool_)
    flat = bitpatterns.reshape(-1)
    for start in range(0, flat.size, 1024 * 1024):
        present[flat[start : start + 1024 * 1024]] = True
    dictionary = np.flatnonzero(present).astype(np.uint16)
    if not 0 < dictionary.size <= 4096:
        raise ValueError("Exact12-bit storage requires1..4096 distinct bitpatterns")
    return dictionary


def pack_rows(bitpatterns: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    """Pack adjacent12-bit indices as low8/pairednibbles/high8 per pair."""
    if bitpatterns.dtype != np.uint16 or dictionary.dtype != np.uint16:
        raise TypeError("Inputs must be exactuint16 bitpatterns")
    if bitpatterns.ndim != 2 or bitpatterns.shape[1] % 2:
        raise ValueError("Embedding rows must have an even width")
    if dictionary.ndim != 1 or not 0 < dictionary.size <= 4096:
        raise ValueError("Dictionary must have1..4096 entries")
    if np.unique(dictionary).size != dictionary.size:
        raise ValueError("Dictionary contains duplicate entries")
    lookup = np.full(65536, 65535, dtype=np.uint16)
    lookup[dictionary] = np.arange(dictionary.size, dtype=np.uint16)
    codes = lookup[bitpatterns]
    if np.any(codes >= dictionary.size):
        raise ValueError("Dictionary does not represent every input bitpattern")
    even, odd = codes[:, 0::2], codes[:, 1::2]
    result = np.empty((*even.shape, 3), dtype=np.uint8)
    result[..., 0] = even & 255
    result[..., 1] = (even >> 8) | ((odd & 15) << 4)
    result[..., 2] = odd >> 4
    return result.reshape(bitpatterns.shape[0], -1)


def unpack_rows(packed: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    if packed.dtype != np.uint8 or dictionary.dtype != np.uint16:
        raise TypeError("Packed bytes and dictionary have invalid dtypes")
    if packed.ndim != 2 or packed.shape[1] % 3:
        raise ValueError("Packed row size must be a multiple of3")
    if dictionary.ndim != 1 or not 0 < dictionary.size <= 4096:
        raise ValueError("Dictionary must have1..4096 entries")
    triples = packed.reshape(packed.shape[0], -1, 3).astype(np.uint16)
    codes = np.empty((packed.shape[0], triples.shape[1] * 2), dtype=np.uint16)
    codes[:, 0::2] = triples[..., 0] | ((triples[..., 1] & 15) << 8)
    codes[:, 1::2] = (triples[..., 1] >> 4) | (triples[..., 2] << 4)
    if np.any(codes >= dictionary.size):
        raise ValueError("Packed data references an absent dictionary entry")
    return dictionary[codes]
