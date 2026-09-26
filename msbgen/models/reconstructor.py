"""A trained MSBNet as a `Reconstructor`, with tiled inference for large images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from msbgen.core.reconstruction import Reconstructor
from msbgen.models.checkpoint import load_model


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelReconstructor(Reconstructor):
    """Runs a checkpoint on the top-k image.

    Images larger than `tile` pixels on a side are processed in overlapping tiles:
    each tile is run with `margin` extra context on every side and only its centre is
    kept, so there are no visible seams and memory stays bounded.
    """

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None,
                 name: str | None = None, tile: int = 512, margin: int = 64):
        self.path = Path(checkpoint)
        self.device = torch.device(device) if device else default_device()
        self.model, payload = load_model(self.path, self.device)
        self.config = self.model.config
        self.step = payload["step"]
        self.metrics = payload.get("metrics", {})
        self.name = name or self.path.parent.name
        arch = "U-Net" if self.config.arch == "unet" else "CNN"
        self.label = f"{arch} ({self.config.output})"
        self.tile, self.margin = tile, margin

    def supports(self, k: int, bit_depth: int, channels: int) -> bool:
        c = self.config
        return (k, bit_depth, channels) == (c.k, c.bit_depth, c.channels)

    @torch.no_grad()
    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        if not self.supports(k, bit_depth, 1 if top.ndim == 2 else top.shape[-1]):
            raise ValueError(f"{self.name} was trained for k={self.config.k}, {self.config.bit_depth}-bit, "
                             f"{self.config.channels} channel(s)")
        squeeze = top.ndim == 2
        x = torch.from_numpy(np.ascontiguousarray(top.astype(np.float32)))
        x = x[None, None] if squeeze else x.permute(2, 0, 1)[None]
        out = self._run_tiled(x)[0]  # (C, H, W) in [0, 1]
        est = out.double().numpy() * self.model.max_value
        return est[0] if squeeze else np.moveaxis(est, 0, -1)

    def _run_tiled(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if max(h, w) <= self.tile:
            return self.model(x.to(self.device)).cpu()
        out = torch.empty_like(x)
        t, m = self.tile, self.margin
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                ya, xa = max(y0 - m, 0), max(x0 - m, 0)
                yb, xb = min(y1 + m, h), min(x1 + m, w)
                pred = self.model(x[..., ya:yb, xa:xb].to(self.device)).cpu()
                out[..., y0:y1, x0:x1] = pred[..., y0 - ya : y1 - ya, x0 - xa : x1 - xa]
        return out


def discover_checkpoints(root: Path, filename: str = "best.pt") -> list[Path]:
    """`<root>/<run>/best.pt` files, newest first."""
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
