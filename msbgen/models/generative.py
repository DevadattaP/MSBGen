"""What the generative models of p(R | B7) share: the residual, the conditioning and the network.

The receiver knows the top k bits, so a generative model only generates the missing part

    R = I - top ∈ [0, 2^(n-k) - 1],   scaled to r = 2R / (2^(n-k) - 1) - 1 ∈ [-1, 1],

and the reconstruction is Î = top + R̂. Like the residual U-Net, it can never
contradict the transmitted planes (MSB consistency = 1). The network is a
time-conditioned U-Net that sees [x_t, top-k planes] and t ∈ [0, 1]; what x_t, t and
its output mean is up to the subclass (diffusion: noise level and v; flow matching:
flow time and velocity).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from msbgen.models.msb_net import encode_planes, pad_to_multiple
from msbgen.models.nets import TimeUNet

Predictor = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # (x_t, t, cond) → output


def randn(shape, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    """Gaussian noise from `generator` (on its own device), moved to `device`.

    A CPU generator makes samples identical on CPU and GPU and independent of tiling.
    """
    gen_device = generator.device if generator is not None else device
    return torch.randn(shape, generator=generator, device=gen_device).to(device)


def rand(n: int, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    """n uniform times in [0, 1) from `generator`, moved to `device`."""
    gen_device = generator.device if generator is not None else device
    return torch.rand(n, generator=generator, device=gen_device).to(device)


class ResidualGenerator(nn.Module):
    """Base class: `config` needs k, bit_depth, channels and arch_args (for TimeUNet)."""

    kind = "generative"

    def __init__(self, config):
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
        """Network output for a batch: x_t (B, C, H, W), t (B,), cond (B, C·k, H, W). Any H, W."""
        h, w = x_t.shape[-2:]
        inp = pad_to_multiple(torch.cat([x_t, cond], dim=1), self.backbone.multiple)
        return self.backbone(inp, t)[..., :h, :w]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
