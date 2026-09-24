"""Bit-plane decomposition of integer images.

For an n-bit image I (n = 8 for most images, 16 for high-bit-depth ones), each
pixel value of each channel decomposes as

    I = sum_{k=0}^{n-1} 2^k * B_k,    B_k in {0, 1}

where B_{n-1} is the most-significant bit-plane (MSB) and B_0 the least (LSB).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DTYPE_BIT_DEPTH = {np.dtype(bool): 1, np.dtype(np.uint8): 8, np.dtype(np.uint16): 16}


def infer_bit_depth(image: np.ndarray) -> int:
    """Bit depth implied by the array dtype (bool → 1, uint8 → 8, uint16 → 16)."""
    try:
        return DTYPE_BIT_DEPTH[image.dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {image.dtype}; expected bool, uint8 or uint16") from None


def slice_bitplanes(image: np.ndarray, bit_depth: int | None = None) -> np.ndarray:
    """Split an integer image into its binary bit-planes.

    Args:
        image: bool / uint8 / uint16 array of shape (H, W) or (H, W, C), any number of
            channels (gray, gray+alpha, RGB, RGBA, CMYK, ...).
        bit_depth: number of planes per channel. Defaults to the dtype's width.

    Returns:
        uint8 array of shape (C, bit_depth, H, W) with values in {0, 1}.
        Index [c, k] holds bit-plane B_k of channel c (k = bit_depth - 1 is the MSB).
        A 2-D input is treated as a single channel (C=1).
    """
    depth = infer_bit_depth(image) if bit_depth is None else bit_depth
    if not 1 <= depth <= infer_bit_depth(image):
        raise ValueError(f"bit_depth {depth} does not fit dtype {image.dtype}")
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError(f"expected (H, W) or (H, W, C) array, got shape {image.shape}")

    channels_first = np.moveaxis(image.astype(np.uint16), -1, 0)  # (C, H, W)
    shifts = np.arange(depth, dtype=np.uint16).reshape(1, depth, 1, 1)
    return ((channels_first[:, np.newaxis, :, :] >> shifts) & 1).astype(np.uint8)


def reconstruct_from_planes(planes: np.ndarray, bits: Iterable[int] | None = None) -> np.ndarray:
    """Recombine bit-planes into an image: sum of 2^k * B_k over the selected bits.

    Args:
        planes: array of shape (C, n, H, W) as returned by `slice_bitplanes`.
        bits: bit indices to keep (e.g. [7] for the MSB of an 8-bit image). Defaults to all.

    Returns:
        Array of shape (H, W, C): uint8 if n <= 8, otherwise uint16.
    """
    depth = planes.shape[1]
    selected = range(depth) if bits is None else list(bits)
    out = np.zeros((planes.shape[0], planes.shape[2], planes.shape[3]), dtype=np.uint32)
    for k in selected:
        out += planes[:, k].astype(np.uint32) << k
    dtype = np.uint8 if depth <= 8 else np.uint16
    return np.moveaxis(out.astype(dtype), 0, -1)


def combine_channel_planes(planes: np.ndarray, bit: int) -> np.ndarray:
    """Stack bit-plane B_k of every channel into one multi-channel binary image.

    Args:
        planes: array of shape (C, n, H, W) as returned by `slice_bitplanes`.
        bit: bit index k.

    Returns:
        uint8 array of shape (H, W, C) with values in {0, 1}. For RGB each pixel is one
        of 8 colours telling which channels have bit k set (e.g. (1, 1, 0) = R and G).
    """
    return np.moveaxis(planes[:, bit], 0, -1)


def binary_entropy(p: float) -> float:
    """Shannon entropy (bits) of a Bernoulli(p) source, i.e. an i.i.d. bound for one plane."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
