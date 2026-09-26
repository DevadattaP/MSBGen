"""LPIPS perceptual distance (Zhang et al., CVPR 2018), via the `lpips` package.

Lower is better: 0 = identical, roughly 0.1 = small visible differences, 0.4+ =
clearly different. Unlike PSNR/SSIM it compares deep features of a pretrained
network (AlexNet), so it tracks human judgements of blur and texture better.

The first use downloads the AlexNet weights through torchvision (~230 MB).
"""

from __future__ import annotations

import numpy as np
import torch


class LPIPS:
    def __init__(self, device: str | torch.device | None = None, net: str = "alex"):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS needs the `lpips` package: pip install lpips") from exc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
        """Images (H, W) or (H, W, C) in [0, data_range]; grayscale is replicated to 3 channels."""
        def prep(a: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(a, dtype=np.float32) / data_range * 2 - 1)
            t = t[None] if t.ndim == 2 else t.permute(2, 0, 1)
            if t.shape[0] == 1:
                t = t.expand(3, -1, -1)
            return t[None].to(self.device)

        return float(self.model(prep(reference), prep(np.clip(estimate, 0, data_range))).item())
