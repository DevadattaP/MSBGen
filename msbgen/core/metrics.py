"""Full-reference image quality metrics (PSNR, SSIM, MAE) on integer-range images.

All functions take images as arrays of shape (H, W) or (H, W, C) holding values in
[0, data_range] (uint8, uint16 or float — float lets a reconstruction such as
128*B7 + 63.5 be scored without rounding).
"""

from __future__ import annotations

import cv2
import numpy as np

# Wang et al. (2004) constants, matching skimage's `structural_similarity` with
# gaussian_weights=True, sigma=1.5, use_sample_covariance=False.
SSIM_SIGMA = 1.5
SSIM_WIN = 11  # 2 * int(3.5 * sigma + 0.5) + 1
SSIM_K1, SSIM_K2 = 0.01, 0.03


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    diff = reference.astype(np.float64) - estimate.astype(np.float64)
    return float(np.mean(diff * diff))


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(reference.astype(np.float64) - estimate.astype(np.float64))))


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
    """Peak signal-to-noise ratio in dB; `inf` for identical images."""
    err = mse(reference, estimate)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / err))


def ssim(reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float | None:
    """Mean structural similarity (Gaussian window, σ = 1.5), averaged over channels.

    Returns None when the image is smaller than the 11×11 window.
    """
    if min(reference.shape[:2]) < SSIM_WIN:
        return None
    x = reference.astype(np.float64)
    y = estimate.astype(np.float64)
    if x.ndim == 2:
        x, y = x[..., np.newaxis], y[..., np.newaxis]
    return float(np.mean([_ssim_channel(x[..., c], y[..., c], data_range) for c in range(x.shape[2])]))


def _ssim_channel(x: np.ndarray, y: np.ndarray, data_range: float) -> float:
    def blur(a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(a, (SSIM_WIN, SSIM_WIN), SSIM_SIGMA, borderType=cv2.BORDER_REFLECT)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x * mu_x
    var_y = blur(y * y) - mu_y * mu_y
    cov = blur(x * y) - mu_x * mu_y
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    pad = (SSIM_WIN - 1) // 2  # ignore the border, where the window hangs off the image
    return float(s[pad:-pad, pad:-pad].mean())
