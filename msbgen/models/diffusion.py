"""Conditional diffusion on the residual: a generative model of p(R | B7) (PLAN Milestone 3).

The receiver knows the top k bits, so the model only generates the missing part

    R = I - top ∈ [0, 2^(n-k) - 1],   scaled to r = 2R / (2^(n-k) - 1) - 1 ∈ [-1, 1],

and the reconstruction is Î = top + R̂. Like the residual U-Net, it can never
contradict the transmitted planes (MSB consistency = 1).

Forward process (continuous time t ∈ [0, 1], cosine schedule):

    x_t = α_t r + σ_t ε,   α_t² + σ_t² = 1,   α_0 = 1, α_1 = 0

The network sees [x_t, B7 planes] and t, and predicts v = α_t ε − σ_t r
("v-prediction", well behaved at both ends of the schedule). Sampling uses DDIM:
eta = 0 is deterministic given the starting noise z, eta = 1 adds fresh noise at
every step (DDPM-like). Different z give different, equally plausible residuals:
that is the point of a generative model here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, ClassVar

import torch
from torch import nn

from msbgen.models.msb_net import encode_planes, pad_to_multiple
from msbgen.models.nets import TimeUNet

Predictor = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # (x_t, t, cond) → v


@dataclass
class DiffusionConfig:
    kind: ClassVar[str] = "diffusion"
    arch: ClassVar[str] = "tunet"  # time-conditioned U-Net (for listings; there is only one)
    output: ClassVar[str] = "residual"  # always generates R, never the whole image

    k: int = 1
    bit_depth: int = 8
    channels: int = 1
    arch_args: dict = field(default_factory=lambda: {"width": 32, "levels": 4, "blocks": 1})
    sample_steps: int = 50  # default number of DDIM steps at inference
    eta: float = 0.0  # default DDIM eta: 0 = deterministic given z, 1 = ancestral (DDPM-like)

    def __post_init__(self):
        if not 1 <= self.k < self.bit_depth:
            raise ValueError(f"k must be in 1..{self.bit_depth - 1}, got {self.k}")
        if self.sample_steps < 1:
            raise ValueError("sample_steps must be at least 1")
        if not 0 <= self.eta <= 1:
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")

    def to_dict(self) -> dict:
        return asdict(self)


def cosine_schedule(t: torch.Tensor, s: float = 0.008) -> tuple[torch.Tensor, torch.Tensor]:
    """(α_t, σ_t) of the cosine schedule (Nichol & Dhariwal 2021), continuous t ∈ [0, 1]."""
    f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2)  # noqa: E731
    alpha = (f(t) / f(torch.zeros_like(t))).clamp(0.0, 1.0)
    return alpha, (1 - alpha**2).clamp_min(0.0).sqrt()


def _randn(shape, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    """Gaussian noise from `generator` (on its own device), moved to `device`.

    A CPU generator makes samples identical on CPU and GPU and independent of tiling.
    """
    gen_device = generator.device if generator is not None else device
    return torch.randn(shape, generator=generator, device=gen_device).to(device)


class MSBDiffusion(nn.Module):
    kind = "diffusion"

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.backbone = TimeUNet(c + c * config.k, c, **config.arch_args)
        self.max_value = float((1 << config.bit_depth) - 1)
        self.residual_max = float((1 << (config.bit_depth - config.k)) - 1)

    # ----- data ↔ residual -----

    def keep_top(self, x: torch.Tensor) -> torch.Tensor:
        drop = self.config.bit_depth - self.config.k
        return ((x.to(torch.int32) >> drop) << drop).float()

    def to_residual(self, x: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        return (x.float() - top) / self.residual_max * 2 - 1

    def from_residual(self, r: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        """Residual in [-1, 1] → Î / (2^n - 1) in [0, 1]."""
        return (top + (r.clamp(-1, 1) + 1) / 2 * self.residual_max) / self.max_value

    def condition(self, top: torch.Tensor) -> torch.Tensor:
        return encode_planes(top, self.config.k, self.config.bit_depth)

    # ----- network -----

    def predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """v-prediction for a batch: x_t (B, C, H, W), t (B,), cond (B, C·k, H, W). Any H, W."""
        h, w = x_t.shape[-2:]
        inp = pad_to_multiple(torch.cat([x_t, cond], dim=1), self.backbone.multiple)
        return self.backbone(inp, t)[..., :h, :w]

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None, reduction: str = "mean") -> torch.Tensor:
        """v-prediction MSE for a batch of original images x (B, C, H, W), integer levels.

        With a seeded `generator`, t and ε are fixed, so the loss is repeatable (validation).
        """
        top = self.keep_top(x)
        r0 = self.to_residual(x, top)
        b = x.shape[0]
        gen_device = generator.device if generator is not None else x.device
        t = torch.rand(b, generator=generator, device=gen_device).to(x.device)
        eps = _randn(r0.shape, generator, x.device)
        alpha, sigma = (v[:, None, None, None] for v in cosine_schedule(t))
        v_pred = self.predict(alpha * r0 + sigma * eps, t, self.condition(top))
        per_item = (v_pred - (alpha * eps - sigma * r0)).square().flatten(1).mean(1)
        return per_item.mean() if reduction == "mean" else per_item

    @torch.no_grad()
    def sample(self, top: torch.Tensor, steps: int | None = None, eta: float | None = None,
               generator: torch.Generator | None = None, predict: Predictor | None = None,
               on_step: Callable[[], object] | None = None) -> torch.Tensor:
        """Draw Î / (2^n - 1) given the top-k image (B, C, H, W) with DDIM.

        `predict` replaces `self.predict`, e.g. to run the network tile by tile.
        `on_step` is called after every denoising step. Sampling is minutes of sustained
        GPU load, so callers pass a thermal-guard check here (see msbgen.utils.thermal).
        """
        steps = steps or self.config.sample_steps
        eta = self.config.eta if eta is None else eta
        predict = predict or self.predict
        top = top.float()
        cond = self.condition(top)
        x = _randn(top.shape, generator, top.device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=top.device)
        alphas, sigmas = cosine_schedule(ts)
        for i in range(steps):
            a_t, s_t, a_s, s_s = alphas[i], sigmas[i], alphas[i + 1], sigmas[i + 1]
            v = predict(x, ts[i].expand(x.shape[0]), cond)
            if on_step is not None:
                on_step()
            x0 = (a_t * x - s_t * v).clamp(-1, 1)
            if i == steps - 1:
                x = x0
                break
            eps = (x - a_t * x0) / s_t
            c = eta * torch.sqrt((s_s**2 / s_t**2) * (1 - a_t**2 / a_s**2)).clamp_min(0) if eta else 0.0
            x = a_s * x0 + torch.sqrt((s_s**2 - c**2).clamp_min(0)) * eps
            if eta:
                x = x + c * _randn(x.shape, generator, x.device)
        return self.from_residual(x, top)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
