"""Full-reference image quality metrics (PSNR, SSIM, MAE) on integer-range images,
plus the set-level KID and the patch extraction it is computed on.

All per-image functions take images as arrays of shape (H, W) or (H, W, C) holding
values in [0, data_range] (uint8, uint16 or float — float lets a reconstruction such
as 128*B7 + 63.5 be scored without rounding).
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


def image_patches(image: np.ndarray, size: int) -> np.ndarray:
    """Square patches (N, size, size[, C]) for set-level metrics such as KID.

    A grid of non-overlapping patches plus the same grid shifted by size/2 in both
    directions (the HiFiC patch-FID protocol), so a 768×512 Kodak image gives
    6·4 + 5·3 = 39 patches of 128. Empty (0, size, size[, C]) if the image is smaller.
    """
    h, w = image.shape[:2]
    corners = [(y, x) for off in (0, size // 2)
               for y in range(off, h - size + 1, size) for x in range(off, w - size + 1, size)]
    if not corners:
        return np.empty((0, size, size, *image.shape[2:]), dtype=image.dtype)
    return np.stack([image[y:y + size, x:x + size] for y, x in corners])


def kid_subset_size(n_x: int, n_y: int, subset_size: int = 1000) -> int:
    """Subset size `kid` uses: at most half of the smaller set, so subsets really differ."""
    return min(subset_size, n_x // 2, n_y // 2)


def kid(x_features: np.ndarray, y_features: np.ndarray, subsets: int = 100, subset_size: int = 1000,
        seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance (Bińkowski et al. 2018): (mean, std) over random subsets.

    The unbiased MMD² between two feature sets (N, D) with the cubic polynomial kernel
    k(a, b) = (a·b / D + 1)³. Unlike FID it has no bias towards small sets, so a few
    hundred patches give a usable estimate. Lower is better, 0 = same distribution
    (and it can come out slightly negative). Usually reported ×10³.

    Subsets hold at most half of the smaller set (see `kid_subset_size`): with all of
    it, every subset would be the same and the std would be a meaningless 0. The std is
    that of a subset-sized estimate, so it is a conservative error bar for the mean.
    """
    x = np.asarray(x_features, dtype=np.float64)
    y = np.asarray(y_features, dtype=np.float64)
    m = kid_subset_size(len(x), len(y), subset_size)
    if m < 2:
        raise ValueError(f"KID needs at least 4 features per set, got {len(x)} and {len(y)}")
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        k_aa = (a @ a.T / d + 1) ** 3
        k_bb = (b @ b.T / d + 1) ** 3
        k_ab = (a @ b.T / d + 1) ** 3
        within = (k_aa.sum() - np.trace(k_aa) + k_bb.sum() - np.trace(k_bb)) / (m * (m - 1))
        scores.append(within - 2 * k_ab.mean())
    return float(np.mean(scores)), float(np.std(scores))
