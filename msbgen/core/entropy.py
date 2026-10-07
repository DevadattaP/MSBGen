"""Lossless adaptive coding for top-k bit planes.

The coder uses the same four causal neighbours as ``analysis.context_entropy``.
Each context starts with a Laplace prior (one count for each symbol) and adapts
as pixels are encoded. The framed stream stores a four-byte length before every
plane so the planes can be decoded without external offsets.
"""

from __future__ import annotations

import struct

import numpy as np

FULL = (1 << 32) - 1
HALF = 1 << 31
FIRST_QUARTER = 1 << 30
THIRD_QUARTER = 3 * FIRST_QUARTER


def _context(plane: np.ndarray, y: int, x: int) -> int:
    height, width = plane.shape
    west = int(plane[y, x - 1]) if x else 0
    north_west = int(plane[y - 1, x - 1]) if y and x else 0
    north = int(plane[y - 1, x]) if y else 0
    north_east = int(plane[y - 1, x + 1]) if y and x + 1 < width else 0
    del height  # shape validation is handled by the public functions
    return west | (north_west << 1) | (north << 2) | (north_east << 3)


def _encode_plane(plane: np.ndarray) -> bytes:
    low, high = 0, FULL
    bits: list[int] = []
    pending = 0

    def emit(bit: int) -> None:
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0

    counts = np.ones((16, 2), dtype=np.uint32)
    for y, x in np.ndindex(plane.shape):
        context = _context(plane, y, x)
        bit = int(plane[y, x])
        zero, one = (int(counts[context, 0]), int(counts[context, 1]))
        total = zero + one
        cumulative_low = zero if bit else 0
        cumulative_high = cumulative_low + (one if bit else zero)
        interval = high - low + 1
        high = low + (interval * cumulative_high // total) - 1
        low = low + (interval * cumulative_low // total)
        counts[context, bit] += 1
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= FIRST_QUARTER and high < THIRD_QUARTER:
                pending += 1
                low -= FIRST_QUARTER
                high -= FIRST_QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) + 1
    pending += 1
    emit(0 if low < FIRST_QUARTER else 1)
    output = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        output[index // 8] |= bit << (7 - index % 8)
    return bytes(output)


def _decode_plane(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    if not data:
        raise ValueError("encoded plane is too short")
    bit_index = 0

    def read_bit() -> int:
        nonlocal bit_index
        if bit_index >= len(data) * 8:
            return 0
        bit = (data[bit_index // 8] >> (7 - bit_index % 8)) & 1
        bit_index += 1
        return bit

    low, high = 0, FULL
    code = 0
    for _ in range(32):
        code = (code << 1) | read_bit()
    plane = np.zeros(shape, dtype=np.uint8)
    counts = np.ones((16, 2), dtype=np.uint32)
    for y, x in np.ndindex(shape):
        context = _context(plane, y, x)
        zero, one = (int(counts[context, 0]), int(counts[context, 1]))
        total = zero + one
        interval = high - low + 1
        value = ((code - low + 1) * total - 1) // interval
        bit = 0 if value < zero else 1
        cumulative_low = zero if bit else 0
        cumulative_high = cumulative_low + (one if bit else zero)
        high = low + (interval * cumulative_high // total) - 1
        low = low + (interval * cumulative_low // total)
        plane[y, x] = bit
        counts[context, bit] += 1
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                code -= HALF
                low -= HALF
                high -= HALF
            elif low >= FIRST_QUARTER and high < THIRD_QUARTER:
                code -= FIRST_QUARTER
                low -= FIRST_QUARTER
                high -= FIRST_QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) + 1
            code = (code << 1) | read_bit()
    return plane


def encode_top_planes(image: np.ndarray, k: int, bit_depth: int | None = None) -> bytes:
    """Encode the top k planes and return a self-delimiting lossless payload."""
    from msbgen.core.bitplanes import infer_bit_depth, slice_bitplanes

    depth = infer_bit_depth(image) if bit_depth is None else bit_depth
    if not 1 <= k < depth:
        raise ValueError(f"k must be in 1..{depth - 1}, got {k}")
    planes = slice_bitplanes(image, depth)
    chunks = []
    for channel in range(planes.shape[0]):
        for bit in range(depth - 1, depth - k - 1, -1):
            encoded = _encode_plane(planes[channel, bit])
            chunks.append(struct.pack(">I", len(encoded)) + encoded)
    return b"".join(chunks)


def decode_top_planes(data: bytes, shape: tuple[int, ...], k: int, bit_depth: int = 8) -> np.ndarray:
    """Decode a payload from ``encode_top_planes`` into the transmitted top-bit image."""
    if len(shape) not in (2, 3):
        raise ValueError(f"expected (H, W) or (H, W, C), got {shape}")
    height, width = shape[:2]
    channels = 1 if len(shape) == 2 else shape[2]
    if not 1 <= k < bit_depth:
        raise ValueError(f"k must be in 1..{bit_depth - 1}, got {k}")
    planes = np.zeros((channels, bit_depth, height, width), dtype=np.uint8)
    offset = 0
    for channel in range(channels):
        for bit in range(bit_depth - 1, bit_depth - k - 1, -1):
            if offset + 4 > len(data):
                raise ValueError("encoded payload ended before all plane lengths were decoded")
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            end = offset + length
            if end > len(data):
                raise ValueError("encoded payload ended inside a plane")
            planes[channel, bit] = _decode_plane(data[offset:end], (height, width))
            offset = end
    if offset != len(data):
        raise ValueError("encoded payload has trailing bytes")
    from msbgen.core.bitplanes import reconstruct_from_planes

    decoded = reconstruct_from_planes(planes)
    return decoded[..., 0] if len(shape) == 2 else decoded