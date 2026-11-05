"""Hybrid decoder: residual U-Net + flow matching + diffusion refinement (PLAN Milestone 6).

Three stages, all working on the scaled residual r ∈ [-1, 1] of the missing low bits
(see `msbgen.models.generative`), so the output never contradicts the transmitted planes:

    1. residual U-Net   r0 = E[r | top]              deterministic, best PSNR
    2. flow matching    δ ~ p(r − r0 | top, r0)      few Euler steps; r1 = r0 + δ  (coarse sample)
    3. diffusion        r ~ p(r | top, r0, r1)       truncated DDIM (SDEdit style): r1 is noised to
                                                     level `refine_t` and denoised in a few steps

Stage 1 keeps the regression model's distortion, stage 2 adds the perceptual texture
flow matching gave on its own, stage 3 cleans up high-frequency detail of that sample.
The stages are trained jointly (or with the U-Net loaded from a regression checkpoint
and frozen); the heads see r0.detach(), the refiner is trained on real stage-2 samples.
`refine_steps = 0` / `steps = 0` switch stage 3 / stages 2 and 3 off for ablations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import torch
from torch import nn

from msbgen.models.diffusion import cosine_schedule
from msbgen.models.generative import rand, randn
from msbgen.models.msb_net import ModelConfig, MSBNet, encode_planes, pad_to_multiple
from msbgen.models.nets import TimeUNet


@dataclass
class HybridConfig:
    kind: ClassVar[str] = "hybrid"
    output: ClassVar[str] = "residual"

    k: int = 1
    bit_depth: int = 8
    channels: int = 1
    unet_arch: str = "unet"  # stage-1 backbone: "unet" or "vit"
    unet_args: dict = field(default_factory=lambda: {"width": 24, "levels": 4})  # stage 1
    arch_args: dict = field(default_factory=lambda: {"width": 32, "levels": 4, "blocks": 1})  # stages 2 and 3
    sample_steps: int = 3  # default flow-matching (Euler) steps; 0 = stage 1 only
    refine_steps: int = 2  # default DDIM steps of the refiner; 0 = no refinement
    refine_t: float = 0.5  # noise level the coarse sample is pushed to before refinement, also the max training t
    eta: float = 0.0  # DDIM eta of the refiner
    train_coarse_steps: int = 2  # Euler steps used to make the coarse samples the refiner trains on
    unet_weight: float = 1.0  # weight of the stage-1 MSE (only when the U-Net is trainable)

    def __post_init__(self):
        if not 1 <= self.k < self.bit_depth:
            raise ValueError(f"k must be in 1..{self.bit_depth - 1}, got {self.k}")
        if self.sample_steps < 0 or self.refine_steps < 0 or self.train_coarse_steps < 1:
            raise ValueError("sample_steps / refine_steps must be >= 0 and train_coarse_steps >= 1")
        if not 0 < self.refine_t <= 1:
            raise ValueError(f"refine_t must be in (0, 1], got {self.refine_t}")
        if not 0 <= self.eta <= 1:
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")

    @property
    def arch(self) -> str:
        return f"{self.unet_arch}+tunet"

    def to_dict(self) -> dict:
        return asdict(self)


class MSBHybrid(nn.Module):
    kind = "hybrid"

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        c, k = config.channels, config.k
        self.unet = MSBNet(ModelConfig(arch=config.unet_arch, output="residual", k=k, bit_depth=config.bit_depth,
                                       channels=c, arch_args=config.unet_args))
        self.flow = TimeUNet(c + c * k + c, c, **config.arch_args)  # [x_t, planes, r0]
        self.refiner = TimeUNet(c + c * k + 2 * c, c, **config.arch_args)  # [x_t, planes, r0, r1]
        self.max_value = float((1 << config.bit_depth) - 1)
        self.residual_max = float((1 << (config.bit_depth - k)) - 1)

    # ----- data <-> residual (same conventions as ResidualGenerator) -----

    def keep_top(self, x: torch.Tensor) -> torch.Tensor:
        drop = self.config.bit_depth - self.config.k
        return ((x.to(torch.int32) >> drop) << drop).float()

    def to_residual(self, x: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        return (x.float() - top) / self.residual_max * 2 - 1

    def from_residual(self, r: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        return (top + (r.clamp(-1, 1) + 1) / 2 * self.residual_max) / self.max_value

    def condition(self, top: torch.Tensor) -> torch.Tensor:
        return encode_planes(top, self.config.k, self.config.bit_depth)

    def load_unet(self, path: Path) -> None:
        """Initialise stage 1 from a residual U-Net regression checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        have = ckpt.get("model_config", {})
        want = self.unet.config.to_dict()
        if ckpt.get("model_kind", "regression") != "regression" or have != want:
            raise ValueError(f"{path} is not a matching residual U-Net checkpoint: {have} vs {want}")
        self.unet.load_state_dict(ckpt["state_dict"])

    @property
    def unet_trainable(self) -> bool:
        return any(p.requires_grad for p in self.unet.parameters())

    # ----- stages -----

    def unet_residual(self, top: torch.Tensor) -> torch.Tensor:
        """Stage 1: r0 ∈ [-1, 1]."""
        est = self.unet(top) * self.max_value  # top + R̂
        return (est - top) / self.residual_max * 2 - 1

    @staticmethod
    def _run(net: TimeUNet, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h, w = x_t.shape[-2:]
        inp = pad_to_multiple(torch.cat([x_t, cond], dim=1), net.multiple)
        return net(inp, t)[..., :h, :w]

    @torch.no_grad()
    def _coarse(self, cond: torch.Tensor, r0: torch.Tensor, steps: int, generator: torch.Generator | None,
                on_step: Callable[[], object] | None = None) -> torch.Tensor:
        """Stage 2: r1 = r0 + δ, δ integrated from noise with `steps` Euler steps (steps = 0: r1 = r0)."""
        if steps == 0:
            return r0
        c = torch.cat([cond, r0], dim=1)
        x = randn(r0.shape, generator, r0.device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=r0.device)
        for i in range(steps):
            v = self._run(self.flow, x, ts[i].expand(x.shape[0]), c)
            if on_step is not None:
                on_step()
            x = x + (ts[i + 1] - ts[i]) * v
        return (r0 + x).clamp(-1, 1)

    # ----- training -----

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None, reduction: str = "mean") -> torch.Tensor:
        """Sum of the stage losses for original images x (B, C, H, W), integer levels."""
        cfg = self.config
        top = self.keep_top(x)
        r = self.to_residual(x, top)
        cond = self.condition(top)
        r0 = self.unet_residual(top)
        r0d = r0.detach()
        b = x.shape[0]
        total = torch.zeros(b, device=x.device)
        if self.unet_trainable:
            total = total + cfg.unet_weight * (r0 - r).square().flatten(1).mean(1)

        # stage 2: velocity of the straight path from δ = r − r0 to noise
        delta = r - r0d
        t = rand(b, generator, x.device)
        eps = randn(r.shape, generator, x.device)
        tb = t[:, None, None, None]
        v = self._run(self.flow, (1 - tb) * delta + tb * eps, t, torch.cat([cond, r0d], dim=1))
        total = total + (v - (eps - delta)).square().flatten(1).mean(1)

        # stage 3: v-prediction of the truncated diffusion, conditioned on a real stage-2 sample
        if cfg.refine_steps > 0:
            r1 = self._coarse(cond, r0d, cfg.train_coarse_steps, generator)
            t2 = rand(b, generator, x.device) * cfg.refine_t
            eps2 = randn(r.shape, generator, x.device)
            alpha, sigma = (u[:, None, None, None] for u in cosine_schedule(t2))
            v2 = self._run(self.refiner, alpha * r + sigma * eps2, t2, torch.cat([cond, r0d, r1], dim=1))
            total = total + (v2 - (alpha * eps2 - sigma * r)).square().flatten(1).mean(1)
        return total.mean() if reduction == "mean" else total

    # ----- inference -----

    @torch.no_grad()
    def sample(self, top: torch.Tensor, steps: int | None = None, refine_steps: int | None = None,
               refine_t: float | None = None, eta: float | None = None, generator: torch.Generator | None = None,
               predict=None, on_step: Callable[[], object] | None = None) -> torch.Tensor:
        """Draw Î / (2^n - 1) given the top-k image (B, C, H, W). `predict` is unused (no tiling)."""
        cfg = self.config
        steps = cfg.sample_steps if steps is None else steps
        refine_steps = cfg.refine_steps if refine_steps is None else refine_steps
        t0 = cfg.refine_t if refine_t is None else refine_t
        eta = cfg.eta if eta is None else eta
        top = top.float()
        cond = self.condition(top)
        r0 = self.unet_residual(top)
        if on_step is not None:
            on_step()
        r1 = self._coarse(cond, r0, steps, generator, on_step)
        if refine_steps == 0:
            return self.from_residual(r1, top)

        full = torch.cat([cond, r0, r1], dim=1)
        ts = torch.linspace(t0, 0.0, refine_steps + 1, device=top.device)
        alphas, sigmas = cosine_schedule(ts)
        x = alphas[0] * r1 + sigmas[0] * randn(r1.shape, generator, top.device)
        for i in range(refine_steps):
            a_t, s_t, a_s, s_s = alphas[i], sigmas[i], alphas[i + 1], sigmas[i + 1]
            v = self._run(self.refiner, x, ts[i].expand(x.shape[0]), full)
            if on_step is not None:
                on_step()
            x0 = (a_t * x - s_t * v).clamp(-1, 1)
            if i == refine_steps - 1:
                x = x0
                break
            e = (x - a_t * x0) / s_t
            c = eta * torch.sqrt((s_s**2 / s_t**2) * (1 - a_t**2 / a_s**2)).clamp_min(0) if eta else 0.0
            x = a_s * x0 + torch.sqrt((s_s**2 - c**2).clamp_min(0)) * e
            if eta:
                x = x + c * randn(x.shape, generator, x.device)
        return self.from_residual(x, top)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
