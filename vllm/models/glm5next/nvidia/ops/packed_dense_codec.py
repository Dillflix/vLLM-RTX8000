# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless 12-bit pairs in N64/K32 tiles; raw16 exception groups retained."""

import numpy as np

from .arithmetic12_codec import pack_rows
from .arithmetic12_planes import row12_to_planes


def _codes_to_words(codes):
    c = codes.astype(np.uint32)
    mag = c & 2047
    shift = np.clip((mag >> 7).astype(np.int32) - 1, 0, 3).astype(np.uint32)
    return (((mag - (shift << 7)) << shift) | ((c & 2048) << 4)).astype(np.uint16)


def pack_mixed_tiles(words):
    if words.dtype != np.uint16 or words.ndim != 2 or words.shape[1] % 64:
        raise ValueError("Expected nonempty FP16 rows, K divisible by64")
    n, k = words.shape
    if min(n, k) <= 0:
        raise ValueError("Empty matrix")
    chunks, records, offset = [], [], 0
    for start in range(0, n, 256):
        group = words[start : start + 256]
        try:
            row12 = pack_rows(group, 12)
        except ValueError:
            width, codes = 16, None
        else:
            width = 12
            low, high = row12_to_planes(row12)
            low, high = low.astype(np.uint32), high.astype(np.uint32)
            codes = np.empty(low.size, dtype=np.uint32)
            codes[::2] = low[::2] | ((high & 15) << 8)
            codes[1::2] = low[1::2] | ((high >> 4) << 8)
            codes = codes.reshape(group.shape)
        records.append((offset, width))
        for first_n in range(0, len(group), 64):
            r = min(64, len(group) - first_n)
            if width == 16:
                tiled = group[first_n : first_n + r].reshape(r, k // 32, 32)
                block = np.ascontiguousarray(tiled.transpose(1, 0, 2))
                data = block.view(np.uint8).reshape(-1)
            else:
                assert codes is not None
                c = codes[first_n : first_n + r].reshape(r, k // 32, 4, 8)
                c = c.transpose(1, 0, 2, 3)
                e = ((c & 2047) << 3) | ((c & 2048) << 4)
                a = e[..., 0] | (e[..., 1] << 16)
                b = e[..., 2] | (e[..., 3] << 16)
                for dest, i in ((a, 4), (b, 6)):
                    dest |= (c[..., i] >> 8) & 7
                    dest |= ((c[..., i + 1] >> 8) & 7) << 16
                    dest |= ((c[..., i] >> 11) & 1) << 14
                    dest |= ((c[..., i + 1] >> 11) & 1) << 30
                d = sum((c[..., i] & 255) << (8 * (i - 4)) for i in range(4, 8))
                packed = np.stack([a, b, d], axis=1).astype("<u4")
                data = np.ascontiguousarray(packed).view(np.uint8).reshape(-1)
            chunks.append(data)
            offset += data.size
    return np.concatenate(chunks), np.asarray(records, dtype=np.uint64)


def unpack_mixed_tiles(data, records, shape):
    n, k = shape
    out = np.empty(shape, dtype=np.uint16)
    expected = 0
    assert data.dtype == np.uint8 and data.ndim == 1
    assert records.dtype == np.uint64 and records.shape == ((n + 255) // 256, 2)
    for group, (offset, width) in enumerate(records.tolist()):
        assert offset == expected and width in (12, 16)
        for first_n in range(group * 256, min(n, group * 256 + 256), 64):
            r = min(64, n - first_n)
            size = r * k * width // 8
            raw = data[expected : expected + size]
            if width == 16:
                values = raw.copy().view("<u2").reshape(k // 32, r, 32)
                result = values.transpose(1, 0, 2).reshape(r, k)
            else:
                planes = raw.copy().view("<u4").reshape(k // 32, 3, r, 4)
                a, b, d = planes[:, 0], planes[:, 1], planes[:, 2]
                c = np.empty((k // 32, r, 4, 8), dtype=np.uint32)
                for value, i in ((a, 0), (b, 2)):
                    c[..., i] = ((value & 0x3FF8) >> 3) | ((value & 0x8000) >> 4)
                    c[..., i + 1] = ((value & 0x3FF80000) >> 19) | (
                        (value & 0x80000000) >> 20
                    )
                for value, i in ((a, 4), (b, 6)):
                    c[..., i] = (
                        ((d >> (8 * (i - 4))) & 255)
                        | ((value & 7) << 8)
                        | ((value & 0x4000) >> 3)
                    )
                    c[..., i + 1] = (
                        ((d >> (8 * (i - 3))) & 255)
                        | ((value & 0x70000) >> 8)
                        | ((value & 0x40000000) >> 19)
                    )
                result = _codes_to_words(c).transpose(1, 0, 2, 3).reshape(r, k)
            out[first_n : first_n + r] = result
            expected += size
    assert expected == data.size
    return out
