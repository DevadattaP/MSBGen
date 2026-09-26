"""Reconstructors: anything that turns the transmitted top-k bits back into an image.

The receiver only has the k most-significant planes. A `Reconstructor` therefore
never sees the original image: it gets `top`, the image with every bit below the
top k set to zero (equivalently, Σ_{top k} 2^j B_j), and returns an estimate Î.

The constant fills below are the no-model baselines; trained networks implement the
same interface (see `msbgen.models.ModelReconstructor`), so every method is scored
by the same `evaluate` function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from msbgen.core.bitplanes import infer_bit_depth
from msbgen.core.metrics import mae, psnr, ssim


def keep_top_bits(image: np.ndarray, k: int, bit_depth: int | None = None) -> np.ndarray:
    """What the receiver has: `image` with all but the top k of n bits zeroed (same dtype)."""
    n = infer_bit_depth(image) if bit_depth is None else bit_depth
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    drop = n - k
    return ((image.astype(np.uint32) >> drop) << drop).astype(image.dtype)


class Reconstructor(ABC):
    """Estimate the full image from its top k bit-planes."""

    name: str  # short id used in tables / APIs, e.g. "mid"
    label: str  # human-readable, e.g. "Midpoint fill"

    def supports(self, k: int, bit_depth: int, channels: int) -> bool:
        return True

    @abstractmethod
    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        """Args:
            top: (H, W) or (H, W, C) integer image holding only the top k bits.
        Returns:
            float64 estimate of the same shape, in [0, 2^n - 1].
        """


class ZeroFill(Reconstructor):
    name, label = "zero", "Zero fill"

    def reconstruct(self, top, k, bit_depth):
        return top.astype(np.float64)


class MidpointFill(Reconstructor):
    """Missing bits at the centre of their range: 128·B7 + 63.5 for n = 8, k = 1."""

    name, label = "mid", "Midpoint fill"

    def reconstruct(self, top, k, bit_depth):
        return top.astype(np.float64) + ((1 << (bit_depth - k)) - 1) / 2


FILL_RECONSTRUCTORS: dict[str, Reconstructor] = {r.name: r for r in (ZeroFill(), MidpointFill())}


def round_estimate(estimate: np.ndarray, bit_depth: int) -> np.ndarray:
    """Round half up and clip to the valid range (what you would store or display)."""
    rounded = np.clip(np.floor(estimate + 0.5), 0, (1 << bit_depth) - 1)
    return rounded.astype(np.uint16 if bit_depth > 8 else np.uint8)


def msb_consistency(reference: np.ndarray, estimate: np.ndarray, k: int, bit_depth: int) -> float:
    """Fraction of pixels whose rounded estimate keeps the transmitted top k bits.

    1.0 means the reconstruction never contradicts what the receiver was sent.
    """
    rounded = round_estimate(estimate, bit_depth)
    same = keep_top_bits(rounded, k, bit_depth) == keep_top_bits(reference, k, bit_depth)
    return float(same.mean())


def evaluate(
    reference: np.ndarray, estimate: np.ndarray, k: int, bit_depth: int, with_ssim: bool = True
) -> dict[str, float | None]:
    """Score an estimate against the original: PSNR, SSIM, MAE and MSB consistency."""
    data_range = float((1 << bit_depth) - 1)
    return {
        "psnr": psnr(reference, estimate, data_range),
        "ssim": ssim(reference, estimate, data_range) if with_ssim else None,
        "mae": mae(reference, estimate),
        "msb_consistency": msb_consistency(reference, estimate, k, bit_depth),
    }


def reconstruct_and_evaluate(
    reconstructor: Reconstructor, image: np.ndarray, k: int, bit_depth: int | None = None, with_ssim: bool = True
) -> tuple[np.ndarray, dict[str, float | None]]:
    """Give `reconstructor` only the top k bits of `image`, then score its output."""
    n = infer_bit_depth(image) if bit_depth is None else bit_depth
    estimate = reconstructor.reconstruct(keep_top_bits(image, k, n), k, n)
    return estimate, evaluate(image, estimate, k, n, with_ssim)
