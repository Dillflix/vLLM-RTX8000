# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lossless CPU codec for exact BF16-origin words already served as FP16.

No values are rounded. Unsupported FP16 words and insufficient widths reject.
"""

import numpy as np


def payload_to_words(payload):
    payload = np.asarray(payload, dtype=np.uint32)
    shift = np.clip((payload.astype(np.int32) >> 7) - 1, 0, 3).astype(np.uint32)
    return (payload - (shift << 7)) << shift


def words_to_codes(words, width):
    if width not in (12, 13) or words.dtype != np.uint16:
        raise ValueError("Expected uint16 words and a12/13-bit storage width")
    raw = words.astype(np.uint32) & 32767
    payload = np.where(
        raw < 256,
        raw,
        np.where(
            raw < 512,
            256 + (raw - 256) // 2,
            np.where(raw < 1024, 384 + (raw - 512) // 4, 384 + raw // 8),
        ),
    )
    if np.any(payload >= (1 << (width - 1))):
        raise ValueError("Values exceed the selected lossless code width")
    if not np.array_equal(payload_to_words(payload), raw):
        raise ValueError("The supplied FP16 words are not exactly representable")
    return payload | ((words.astype(np.uint32) >> 15) << (width - 1))


def codes_to_words(codes, width):
    if width not in (12, 13):
        raise ValueError("Expected12/13-bit storage width")
    payload = codes.astype(np.uint32) & ((1 << (width - 1)) - 1)
    sign = (codes.astype(np.uint32) >> (width - 1)) << 15
    return (payload_to_words(payload) | sign).astype(np.uint16)


def pack_rows(words, width):
    if words.ndim != 2 or words.shape[1] % 8:
        raise ValueError("Rows must contain a multiple of eight values")
    codes = words_to_codes(words, width)
    if width == 12:
        even, odd = codes[:, 0::2], codes[:, 1::2]
        packed = np.empty((*even.shape, 3), dtype=np.uint8)
        packed[:, :, 0] = even & 255
        packed[:, :, 1] = (even >> 8) | ((odd & 15) << 4)
        packed[:, :, 2] = odd >> 4
    else:
        values = codes.reshape(-1, 8).astype(np.uint64)
        low = (
            values[:, 0]
            | (values[:, 1] << 13)
            | (values[:, 2] << 26)
            | (values[:, 3] << 39)
            | (values[:, 4] << 52)
        )
        high = (
            (values[:, 4] >> 12)
            | (values[:, 5] << 1)
            | (values[:, 6] << 14)
            | (values[:, 7] << 27)
        )
        packed = np.empty((len(values), 13), dtype=np.uint8)
        packed[:, :8] = low.astype("<u8", copy=False).view(np.uint8).reshape(-1, 8)
        packed[:, 8:] = (
            high.astype("<u8", copy=False).view(np.uint8).reshape(-1, 8)[:, :5]
        )
    return packed.reshape(words.shape[0], words.shape[1] * width // 8)


def unpack_rows(packed, width):
    if packed.dtype != np.uint8 or packed.ndim != 2 or (packed.shape[1] * 8) % width:
        raise ValueError("Invalid packed row geometry")
    n, row_bytes = packed.shape
    k = row_bytes * 8 // width
    if k % 8:
        raise ValueError("Rows must contain a multiple of eight values")
    if width == 12:
        chunks = packed.reshape(n, k // 2, 3).astype(np.uint32)
        codes = np.empty((n, k), dtype=np.uint32)
        codes[:, 0::2] = chunks[:, :, 0] | ((chunks[:, :, 1] & 15) << 8)
        codes[:, 1::2] = (chunks[:, :, 1] >> 4) | (chunks[:, :, 2] << 4)
    elif width == 13:
        chunks = packed.reshape(-1, 13)
        low = chunks[:, :8].copy().view("<u8").reshape(-1)
        high_bytes = np.zeros((len(chunks), 8), dtype=np.uint8)
        high_bytes[:, :5] = chunks[:, 8:]
        high = high_bytes.view("<u8").reshape(-1)
        codes = np.empty((len(chunks), 8), dtype=np.uint32)
        for i in range(4):
            codes[:, i] = (low >> (i * 13)) & 8191
        codes[:, 4] = ((low >> 52) | (high << 12)) & 8191
        for i in range(5, 8):
            codes[:, i] = (high >> (1 + (i - 5) * 13)) & 8191
        codes = codes.reshape(n, k)
    else:
        raise ValueError("Expected12/13-bit storage width")
    return codes_to_words(codes, width)


def pack_groups(words, group_rows=256):
    if group_rows % 128 or group_rows < 128:
        raise ValueError("Groups must align to both qualified N64/N128 tiles")
    packed, offsets, widths = [], [], []
    offset = 0
    for start in range(0, len(words), group_rows):
        block = words[start : start + group_rows]
        try:
            data = pack_rows(block, 12)
            width = 12
        except ValueError:
            data = pack_rows(block, 13)
            width = 13
        assert np.array_equal(unpack_rows(data, width), block)
        offsets.append(offset)
        widths.append(width)
        packed.append(data.reshape(-1))
        offset += data.size
    return (
        np.concatenate(packed),
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(widths, dtype=np.uint8),
    )


def pack_raw_exception_groups(words, group_rows=256):
    """Use arithmetic12 where exact; keep every other group as rawFP16 bits."""
    if words.dtype != np.uint16 or words.ndim != 2 or words.shape[1] % 8:
        raise ValueError("Expected FP16 word rows aligned to eight values")
    if group_rows < 128 or group_rows % 128:
        raise ValueError("Groups must align to N64/N128 tiles")
    chunks, records = [], []
    offset = 0
    for start in range(0, len(words), group_rows):
        block = words[start : start + group_rows]
        try:
            chunk = pack_rows(block, 12)
            width = 12
            assert np.array_equal(unpack_rows(chunk, 12), block)
        except ValueError:
            chunk = np.ascontiguousarray(block, dtype="<u2").view(np.uint8)
            width = 16
            assert np.array_equal(chunk.view("<u2"), block)
        records.append((offset, width))
        chunks.append(chunk.reshape(-1))
        offset += chunk.size
    return np.concatenate(chunks), np.asarray(records, dtype=np.uint64)


def unpack_raw_exception_groups(packed, metadata, shape, group_rows=256):
    n, k = shape
    if metadata.shape != ((n + group_rows - 1) // group_rows, 2):
        raise ValueError("Metadata must describe every output-row group")
    output = np.empty(shape, dtype=np.uint16)
    for group, (offset, width) in enumerate(metadata.tolist()):
        start = group * group_rows
        rows = min(group_rows, n - start)
        size = rows * k * width // 8
        values = packed[offset : offset + size].reshape(rows, k * width // 8)
        if width == 12:
            output[start : start + rows] = unpack_rows(values, width)
        elif width == 16:
            output[start : start + rows] = values.copy().view("<u2")
        else:
            raise ValueError("Unexpected group width")
    return output
