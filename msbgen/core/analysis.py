"""Information content of bit-planes: entropy, correlations and k-MSB reconstruction.

Answers the Milestone "bit-plane analysis" questions for a single image:

* How many bits/pixel does each plane B_k carry? Two estimates:
  - i.i.d. entropy H(B_k) (ignores all structure), and
  - context entropy H(B_k | W, NW, N, NE): the empirical conditional entropy given
    the 4 causal neighbours in the same plane. It approximates what a context-model
    binary coder (JBIG-style) would achieve for that plane on its own.
* How structured is each plane? Spatial correlation between neighbouring pixels.
* How related are the planes to each other? Pearson (phi) correlation between every
  pair of planes, across channels too.
* How good is an image made from only the top k planes? PSNR / SSIM / MAE for
  - zero fill:     Î = Σ_{top k} 2^j B_j                  (truncation)
  - midpoint fill: Î = Σ_{top k} 2^j B_j + (2^(n-k) - 1)/2 (MSE-optimal if the missing
    bits are uniform; for n = 8, k = 1 this is the baseline 128·B7 + 63.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from msbgen.core.bitplanes import binary_entropy, infer_bit_depth, slice_bitplanes
from msbgen.core.metrics import mae, psnr, ssim
from msbgen.core.reconstruction import FILL_RECONSTRUCTORS, keep_top_bits

FILLS = ("zero", "mid")
CORRELATION_CHUNK = 1 << 20  # pixels per chunk; bounds memory for the plane×plane matrix


@dataclass
class PlaneStats:
    channel_index: int
    bit: int
    ones_ratio: float
    entropy: float  # i.i.d. bound, bits/pixel
    context_entropy: float  # H(B | W, NW, N, NE), bits/pixel
    spatial_correlation: float | None  # mean phi with right and lower neighbour; None if constant


@dataclass
class ReconstructionStats:
    k: int  # number of most-significant planes kept
    bits_per_pixel: int  # raw cost: k planes × channels
    coded_bpp: float  # Σ context entropy of the kept planes (each coded on its own)
    psnr: dict[str, float]  # fill → dB (inf when lossless)
    ssim: dict[str, float | None]
    mae: dict[str, float]


@dataclass
class ImageAnalysis:
    bit_depth: int
    channels: int
    planes: list[PlaneStats]  # ordered by channel, then MSB → LSB
    correlation: np.ndarray  # (C·n, C·n) in the same order; NaN where a plane is constant
    reconstructions: list[ReconstructionStats] = field(default_factory=list)  # k = 1 … n


def context_entropy(plane: np.ndarray) -> float:
    """Empirical H(B | W, NW, N, NE) of a (H, W) {0,1} plane, in bits/pixel.

    Neighbours outside the image count as 0. With only 16 contexts the plug-in
    estimate is close to the true conditional entropy for any realistic image size.
    """
    b = plane.astype(np.uint8)
    p = np.pad(b, ((1, 0), (1, 1)))
    ctx = p[1:, :-2] | (p[:-1, :-2] << 1) | (p[:-1, 1:-1] << 2) | (p[:-1, 2:] << 3)
    counts = np.bincount((ctx * 2 + b).ravel(), minlength=32).reshape(16, 2)
    totals = counts.sum(axis=1)
    n = totals.sum()
    return float(
        sum(t / n * binary_entropy(c[1] / t) for c, t in zip(counts, totals, strict=True) if t)
    )


def _phi(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation of two {0,1} arrays; None if either is constant or empty."""
    if x.size == 0:
        return None
    px, py = x.mean(), y.mean()
    denom = px * (1 - px) * py * (1 - py)
    if denom <= 0:
        return None
    pxy = np.count_nonzero(x & y) / x.size
    return float(np.clip((pxy - px * py) / np.sqrt(denom), -1.0, 1.0))


def spatial_correlation(plane: np.ndarray) -> float | None:
    """Mean phi between each pixel and its right and lower neighbours."""
    b = plane.astype(bool)
    values = [v for v in (_phi(b[:, :-1], b[:, 1:]), _phi(b[:-1, :], b[1:, :])) if v is not None]
    return float(np.mean(values)) if values else None


def plane_correlation(planes: np.ndarray) -> np.ndarray:
    """Pearson (phi) correlation between all planes.

    Args:
        planes: (C, n, H, W) array from `slice_bitplanes`.

    Returns:
        (C·n, C·n) float array ordered by channel, then MSB → LSB. Entries involving a
        constant plane are NaN. Computed exactly in pixel chunks.
    """
    c, n = planes.shape[:2]
    flat = planes[:, ::-1].reshape(c * n, -1)  # MSB first within each channel
    total = flat.shape[1]
    joint = np.zeros((c * n, c * n), dtype=np.float64)
    for start in range(0, total, CORRELATION_CHUNK):
        chunk = flat[:, start : start + CORRELATION_CHUNK].astype(np.float32)
        joint += chunk @ chunk.T  # exact: chunk counts stay well below 2^24
    joint /= total
    p = np.diag(joint).copy()
    std = np.sqrt(p * (1 - p))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (joint - np.outer(p, p)) / np.outer(std, std)
    corr[np.outer(std, std) == 0] = np.nan
    return np.clip(corr, -1.0, 1.0)


def msb_reconstruction(image: np.ndarray, k: int, fill: str = "mid", bit_depth: int | None = None) -> np.ndarray:
    """Image rebuilt from its top k bit-planes, as float64 (not rounded).

    Args:
        image: bool / uint8 / uint16 array (H, W) or (H, W, C).
        k: number of most-significant planes to keep, 1 … n.
        fill: "zero" (missing bits = 0) or "mid" (missing bits at the centre of their range).
        bit_depth: n; defaults to the dtype's width.
    """
    n = infer_bit_depth(image) if bit_depth is None else bit_depth
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
    return FILL_RECONSTRUCTORS[fill].reconstruct(keep_top_bits(image, k, n), k, n)


def analyze_image(image: np.ndarray, bit_depth: int | None = None, with_ssim: bool = True) -> ImageAnalysis:
    """Run the full bit-plane analysis on one integer image (H, W) or (H, W, C)."""
    n = infer_bit_depth(image) if bit_depth is None else bit_depth
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    planes = slice_bitplanes(image, n)
    c = planes.shape[0]

    stats: list[PlaneStats] = []
    for ch in range(c):
        for k in reversed(range(n)):
            plane = planes[ch, k]
            ones = float(plane.mean())
            stats.append(
                PlaneStats(
                    channel_index=ch,
                    bit=k,
                    ones_ratio=ones,
                    entropy=binary_entropy(ones),
                    context_entropy=context_entropy(plane),
                    spatial_correlation=spatial_correlation(plane),
                )
            )

    ctx_by_bit = np.zeros(n)
    for s in stats:
        ctx_by_bit[s.bit] += s.context_entropy
    data_range = float((1 << n) - 1)

    reconstructions = []
    for k in range(1, n + 1):
        scores: dict[str, dict] = {"psnr": {}, "ssim": {}, "mae": {}}
        for fill in FILLS:
            estimate = msb_reconstruction(image, k, fill, n)
            scores["psnr"][fill] = psnr(image, estimate, data_range)
            scores["ssim"][fill] = ssim(image, estimate, data_range) if with_ssim else None
            scores["mae"][fill] = mae(image, estimate)
        reconstructions.append(
            ReconstructionStats(
                k=k,
                bits_per_pixel=k * c,
                coded_bpp=float(ctx_by_bit[n - k :].sum()),
                **scores,
            )
        )

    return ImageAnalysis(
        bit_depth=n,
        channels=c,
        planes=stats,
        correlation=plane_correlation(planes),
        reconstructions=reconstructions,
    )
