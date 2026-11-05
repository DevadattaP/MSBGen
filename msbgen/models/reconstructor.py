"""Trained checkpoints as `Reconstructor`s, with tiled inference for large images.

    load_reconstructor(path)   → ModelReconstructor (regression), DiffusionReconstructor
                                 or FlowReconstructor
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from msbgen.core.reconstruction import Reconstructor
from msbgen.models.checkpoint import GENERATIVE_KINDS, load_model, read_checkpoint
from msbgen.models.bitdepth_recovery import BitplaneResidualCNN


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_tiled(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, tile: int, margin: int) -> torch.Tensor:
    """Apply a fully convolutional `fn` to x (B, C, H, W) in tiles, keeping memory bounded.

    Each tile is run with `margin` extra context on every side and only its centre is
    kept, so there are no visible seams. The output stays on the device `fn` returns it on.
    """
    h, w = x.shape[-2:]
    if max(h, w) <= tile:
        return fn(x)
    out = None
    m = margin
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(y0 - m, 0), max(x0 - m, 0)
            yb, xb = min(y1 + m, h), min(x1 + m, w)
            pred = fn(x[..., ya:yb, xa:xb])
            if out is None:
                out = pred.new_empty((*pred.shape[:-2], h, w))
            out[..., y0:y1, x0:x1] = pred[..., y0 - ya : y1 - ya, x0 - xa : x1 - xa]
    return out


def _to_tensor(top: np.ndarray) -> torch.Tensor:
    """(H, W) or (H, W, C) → (1, C, H, W) float32."""
    x = torch.from_numpy(np.ascontiguousarray(top.astype(np.float32)))
    return x[None, None] if top.ndim == 2 else x.permute(2, 0, 1)[None]


def _to_estimate(out: torch.Tensor, max_value: float, like: np.ndarray) -> np.ndarray:
    """(1, C, H, W) in [0, 1] → float64 estimate shaped like `like`."""
    est = out[0].cpu().double().numpy() * max_value
    return est[0] if like.ndim == 2 else np.moveaxis(est, 0, -1)


class _CheckpointReconstructor(Reconstructor):
    def __init__(self, checkpoint: Path, device: str | torch.device | None, name: str | None,
                 tile: int, margin: int):
        self.path = Path(checkpoint)
        self.device = torch.device(device) if device else default_device()
        self.model, payload = load_model(self.path, self.device)
        self.config = self.model.config
        self.step = payload["step"]
        self.metrics = payload.get("metrics", {})
        self.name = name or self.path.parent.name
        self.tile, self.margin = tile, margin

    def supports(self, k: int, bit_depth: int, channels: int) -> bool:
        c = self.config
        return (k, bit_depth, channels) == (c.k, c.bit_depth, c.channels)

    def _check(self, top: np.ndarray, k: int, bit_depth: int) -> None:
        if not self.supports(k, bit_depth, 1 if top.ndim == 2 else top.shape[-1]):
            raise ValueError(f"{self.name} was trained for k={self.config.k}, {self.config.bit_depth}-bit, "
                             f"{self.config.channels} channel(s)")


class ModelReconstructor(_CheckpointReconstructor):
    """Runs a regression checkpoint (CNN / U-Net) on the top-k image."""

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None,
                 name: str | None = None, tile: int = 512, margin: int = 64,
                 on_step: Callable[[], object] | None = None):
        super().__init__(checkpoint, device, name, tile, margin)
        self.on_step = on_step  # called after every tile, e.g. a thermal-guard check
        arch = {"unet": "U-Net", "vit": "ViT"}.get(self.config.arch, "CNN")
        self.label = f"{arch} ({self.config.output})"

    @torch.no_grad()
    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        self._check(top, k, bit_depth)

        def run(t: torch.Tensor) -> torch.Tensor:
            out = self.model(t.to(self.device)).cpu()
            if self.on_step is not None:
                self.on_step()
            return out

        out = run_tiled(run, _to_tensor(top), self.tile, self.margin)
        return _to_estimate(out, self.model.max_value, top)


class _SamplingReconstructor(_CheckpointReconstructor):
    """Samples a generative checkpoint: Î = top + R̂, R̂ ~ p(R | top).

    `reconstruct` draws with `seed`; `sample(..., seed)` draws with any seed. The noise
    comes from a CPU generator, so a given seed gives the same image on CPU and GPU.

    Images larger than `tile` are tiled at every sampling step (the network is applied
    per tile, the sampler state covers the whole image). The network's GroupNorm layers
    normalise over their whole input, so tiles see slightly different statistics than a
    full pass; the default tile (1024) keeps Kodak-sized images in one piece.
    """

    stochastic = True

    def __init__(self, checkpoint: Path, device: str | torch.device | None, name: str | None, tile: int,
                 margin: int, steps: int | None, seed: int, on_step: Callable[[], object] | None):
        super().__init__(checkpoint, device, name, tile, margin)
        self.on_step = on_step  # called after every network evaluation, e.g. a thermal-guard check
        self.steps = steps or self.config.sample_steps
        self.seed = seed

    @property
    def sampler(self) -> dict:
        """The sampler settings, as recorded by `evaluate`."""
        return {"sample_steps": self.steps}

    def _sampler_kwargs(self) -> dict:
        return {}

    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        return self.sample(top, k, bit_depth, self.seed)

    @torch.no_grad()
    def sample(self, top: np.ndarray, k: int, bit_depth: int, seed: int) -> np.ndarray:
        self._check(top, k, bit_depth)
        x = _to_tensor(top).to(self.device)
        c = x.shape[1]

        def predict(x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            return run_tiled(lambda inp: self.model.predict(inp[:, :c], t[:1], inp[:, c:]),
                             torch.cat([x_t, cond], dim=1), self.tile, self.margin)

        gen = torch.Generator().manual_seed(seed)
        out = self.model.sample(x, steps=self.steps, generator=gen, predict=predict, on_step=self.on_step,
                                **self._sampler_kwargs())
        return _to_estimate(out, self.model.max_value, top)


class DiffusionReconstructor(_SamplingReconstructor):
    """A diffusion checkpoint, sampled with DDIM (`steps`, `eta`)."""

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None,
                 name: str | None = None, tile: int = 1024, margin: int = 64,
                 steps: int | None = None, eta: float | None = None, seed: int = 0,
                 on_step: Callable[[], object] | None = None):
        super().__init__(checkpoint, device, name, tile, margin, steps, seed, on_step)
        self.eta = self.config.eta if eta is None else eta
        self.label = "Diffusion (residual)"

    @property
    def sampler(self) -> dict:
        return {"sample_steps": self.steps, "eta": self.eta}

    def _sampler_kwargs(self) -> dict:
        return {"eta": self.eta}


class FlowReconstructor(_SamplingReconstructor):
    """A flow-matching checkpoint, sampled by integrating its ODE (`steps`, `solver`)."""

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None,
                 name: str | None = None, tile: int = 1024, margin: int = 64,
                 steps: int | None = None, solver: str | None = None, seed: int = 0,
                 on_step: Callable[[], object] | None = None):
        super().__init__(checkpoint, device, name, tile, margin, steps, seed, on_step)
        self.solver = solver or self.config.solver
        self.label = "Flow matching (residual)"

    @property
    def sampler(self) -> dict:
        return {"sample_steps": self.steps, "solver": self.solver}

    def _sampler_kwargs(self) -> dict:
        return {"solver": self.solver}


class NormalizingFlowReconstructor(_CheckpointReconstructor):
    """Samples a conditional normalizing-flow checkpoint in fixed-size patches."""

    stochastic = True

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None,
                 name: str | None = None, on_step: Callable[[], object] | None = None):
        super().__init__(checkpoint, device, name, tile=512, margin=0)
        self.tile = self.config.crop_size
        self.on_step = on_step
        self.label = "Normalizing flow (residual)"

    @property
    def sampler(self) -> dict:
        return {"patch_size": self.config.crop_size}

    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        return self.sample(top, k, bit_depth, seed=0)

    @torch.no_grad()
    def sample(self, top: np.ndarray, k: int, bit_depth: int, seed: int) -> np.ndarray:
        self._check(top, k, bit_depth)
        x = _to_tensor(top).to(self.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        out = self.model.sample(x, generator=generator, on_step=self.on_step)
        return _to_estimate(out, self.model.max_value, top)


class HybridReconstructor(_CheckpointReconstructor):
    """A hybrid checkpoint (U-Net + flow matching + diffusion refinement), run on the whole image.

    `steps` = flow-matching steps (0: stage 1 only), `refine_steps` = refiner steps (0: no stage 3).
    """

    def __init__(self, checkpoint: Path, device: str | torch.device | None = None, name: str | None = None,
                 steps: int | None = None, refine_steps: int | None = None, refine_t: float | None = None,
                 eta: float | None = None, seed: int = 0, on_step: Callable[[], object] | None = None):
        super().__init__(checkpoint, device, name, tile=1 << 30, margin=0)
        c = self.config
        self.steps = c.sample_steps if steps is None else steps
        self.refine_steps = c.refine_steps if refine_steps is None else refine_steps
        self.refine_t = c.refine_t if refine_t is None else refine_t
        self.eta = c.eta if eta is None else eta
        self.seed, self.on_step = seed, on_step
        self.stochastic = self.steps > 0 or self.refine_steps > 0
        s1 = "ViT" if c.unet_arch == "vit" else "U-Net"
        self.label = ((f"Hybrid ({s1} only)" if self.steps == 0 else f"Hybrid ({s1} + flow)")
                      if self.refine_steps == 0 else f"Hybrid ({s1} + flow + diffusion)")

    @property
    def sampler(self) -> dict:
        return {"sample_steps": self.steps, "refine_steps": self.refine_steps, "refine_t": self.refine_t,
                "eta": self.eta}

    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        return self.sample(top, k, bit_depth, self.seed)

    @torch.no_grad()
    def sample(self, top: np.ndarray, k: int, bit_depth: int, seed: int) -> np.ndarray:
        self._check(top, k, bit_depth)
        gen = torch.Generator().manual_seed(seed)
        out = self.model.sample(_to_tensor(top).to(self.device), steps=self.steps, refine_steps=self.refine_steps,
                                refine_t=self.refine_t, eta=self.eta, generator=gen, on_step=self.on_step)
        return _to_estimate(out, self.model.max_value, top)


class BitplaneReconstructor(Reconstructor):
    """The paper-style sequential bitplane CNNs (`runs/bitdepth-*/bit_NN.pt`): top k bits -> 8 bits (gray or RGB)."""

    def __init__(self, folder: Path, device: str | torch.device | None = None, name: str | None = None,
                 on_step: Callable[[], object] | None = None):
        self.folder = Path(folder)
        self.on_step = on_step  # called after every bit-plane network
        self.device = torch.device(device) if device else default_device()
        self.name = name or self.folder.name
        self.label = "Sequential bitplane CNN"

    def _path(self, bits: int) -> Path:
        return self.folder / f"bit_{bits:02d}.pt"

    def supports(self, k: int, bit_depth: int, channels: int) -> bool:
        if not all(self._path(b).exists() for b in range(k, bit_depth)):
            return False
        for bits in range(k, bit_depth):
            payload = torch.load(self._path(bits), map_location="cpu", weights_only=True)
            model_channels = payload["config"]["channels"]
            del payload
            if model_channels != channels:
                return False
        return True

    @torch.no_grad()
    def reconstruct(self, top: np.ndarray, k: int, bit_depth: int) -> np.ndarray:
        maximum = float((1 << bit_depth) - 1)
        current = (_to_tensor(top) / maximum).to(self.device)
        for bits in range(k, bit_depth):
            model = None
            payload = None
            try:
                payload = torch.load(self._path(bits), map_location="cpu", weights_only=True)
                model = BitplaneResidualCNN(**payload["config"])
                model.load_state_dict(payload["state_dict"])
                del payload
                payload = None
                model = model.to(self.device).eval()
                weight = float(1 << (bit_depth - bits - 1)) / maximum
                current = current + torch.sigmoid(model(current)) * weight
            finally:
                del model, payload
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            if self.on_step is not None:
                self.on_step()
        return _to_estimate(current.clamp(0, 1), maximum, top)


SAMPLER_OPTIONS = {"diffusion": ("steps", "eta"), "flow": ("steps", "solver"), "normalizing_flow": (),
                   "hybrid": ("steps", "eta", "refine_steps", "refine_t")}


def load_reconstructor(checkpoint: Path, device: str | torch.device | None = None, **kwargs) -> _CheckpointReconstructor:
    """The right `Reconstructor` for a checkpoint.

    Generative models accept `steps`, `seed`, `on_step`, plus `eta` (diffusion) or
    `solver` (flow matching); options that don't apply to the checkpoint's kind are ignored.
    """
    kind = read_checkpoint(checkpoint)["model_kind"]
    sampler_keys = {"steps", "eta", "solver", "seed", "on_step", "refine_steps", "refine_t"}
    keep = set(SAMPLER_OPTIONS.get(kind, ())) | {"on_step"} | ({"seed"} if kind in GENERATIVE_KINDS else set())
    kwargs = {key: v for key, v in kwargs.items() if key not in sampler_keys or key in keep}
    if kind == "diffusion":
        return DiffusionReconstructor(checkpoint, device, **kwargs)
    if kind == "flow":
        return FlowReconstructor(checkpoint, device, **kwargs)
    if kind == "normalizing_flow":
        return NormalizingFlowReconstructor(checkpoint, device, **kwargs)
    if kind == "hybrid":
        return HybridReconstructor(checkpoint, device, **kwargs)
    return ModelReconstructor(checkpoint, device, **kwargs)


def discover_checkpoints(root: Path, filename: str = "best.pt") -> list[Path]:
    """`<root>/<run>/best.pt` files, newest first."""
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
