"""Deep-feature image metrics: LPIPS and DISTS per image, Inception features for KID.

LPIPS (Zhang et al., CVPR 2018), via the `lpips` package. Lower is better: 0 =
identical, roughly 0.1 = small visible differences, 0.4+ = clearly different.
Unlike PSNR/SSIM it compares deep features of a pretrained network (AlexNet), so
it tracks human judgements of blur and texture better.

DISTS (Ding et al., TPAMI 2020), via `piq`. Lower is better, in [0, 1]. It compares
VGG feature *statistics* (means and correlations) as well as structure, so a texture
that is plausible but not pixel-aligned with the original is penalised much less
than by LPIPS. That makes it the per-image metric that can reward a generative
sample over a blurry regression output.

InceptionFeatures (via `piq`): 2048-d pool features of the FID Inception network,
for the set-level KID (`msbgen.core.metrics.kid`).

Grayscale is replicated to 3 channels everywhere. Weights download on first use:
AlexNet (~230 MB) for LPIPS, VGG16 (~530 MB) plus DISTS weights for DISTS, the FID
Inception (~100 MB) for KID.
"""

from __future__ import annotations

import numpy as np
import torch


def _default_device(device: str | torch.device | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _to_batch(images: np.ndarray, data_range: float, signed: bool, batched: bool = False) -> torch.Tensor:
    """One image (H, W) / (H, W, C), or with `batched` (N, H, W) / (N, H, W, C), in
    [0, data_range] → (N, 3, H, W) in [0, 1], or [-1, 1] if `signed`."""
    a = np.clip(np.asarray(images, dtype=np.float32), 0, data_range) / data_range
    if not batched:
        a = a[None]
    t = torch.from_numpy(a)
    t = t[:, None] if t.ndim == 3 else t.permute(0, 3, 1, 2)
    if t.shape[1] == 1:
        t = t.expand(-1, 3, -1, -1)
    return t * 2 - 1 if signed else t


def _require_piq(metric: str):
    try:
        import piq
    except ImportError as exc:
        raise ImportError(f"{metric} needs the `piq` package: pip install piq") from exc
    return piq


class LPIPS:
    def __init__(self, device: str | torch.device | None = None, net: str = "alex"):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS needs the `lpips` package: pip install lpips") from exc
        self.device = _default_device(device)
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
        """Images (H, W) or (H, W, C) in [0, data_range]."""
        ref = _to_batch(reference, data_range, signed=True).to(self.device)
        est = _to_batch(estimate, data_range, signed=True).to(self.device)
        return float(self.model(ref, est).item())


class DISTS:
    def __init__(self, device: str | torch.device | None = None):
        piq = _require_piq("DISTS")
        self.device = _default_device(device)
        self.model = piq.DISTS(reduction="none").to(self.device).eval()

    @torch.no_grad()
    def __call__(self, reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
        """Images (H, W) or (H, W, C) in [0, data_range]. Like the reference implementation,
        images whose short side exceeds 256 are downscaled to 256 first."""
        ref = _to_batch(reference, data_range, signed=False).to(self.device)
        est = _to_batch(estimate, data_range, signed=False).to(self.device)
        return float(self.model(est, ref).item())


class InceptionFeatures:
    def __init__(self, device: str | torch.device | None = None, batch_size: int = 64):
        InceptionV3 = _require_piq("KID").feature_extractors.InceptionV3  # noqa: N806
        self.device = _default_device(device)
        self.batch_size = batch_size
        # Inputs in [0, 1], resized to 299 inside the network, as for FID.
        self.model = InceptionV3(resize_input=True, normalize_input=True).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: np.ndarray, data_range: float) -> np.ndarray:
        """Batch (N, H, W) or (N, H, W, C) in [0, data_range] → features (N, 2048), float32."""
        if len(images) == 0:
            return np.empty((0, 2048), dtype=np.float32)
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = _to_batch(images[i:i + self.batch_size], data_range, signed=False, batched=True).to(self.device)
            feats.append(self.model(batch)[0].flatten(1).cpu().numpy())
        return np.concatenate(feats).astype(np.float32)
