"""MSBNet: a backbone plus the input / output conventions for top-k reconstruction.

Input:  the k transmitted planes of each channel, B_{n-1} … B_{n-k}, as ±0.5.
Output: Î in [0, 1] (i.e. divided by 2^n - 1), in one of two modes:

* "direct"   Î = sigmoid(z)                                  (PLAN baselines 2 and 3)
* "residual" Î = (top + sigmoid(z) · (2^(n-k) - 1)) / (2^n - 1)  (PLAN baseline 4)

In residual mode the network only predicts the missing low bits R ∈ [0, 2^(n-k) - 1],
so the transmitted planes are always reproduced exactly (MSB consistency = 1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from msbgen.models.nets import ARCHITECTURES

OUTPUT_MODES = ("direct", "residual")


@dataclass
class ModelConfig:
    kind: ClassVar[str] = "regression"

    arch: str = "unet"  # "cnn" or "unet"
    output: str = "residual"  # "direct" or "residual"
    k: int = 1  # number of transmitted planes
    bit_depth: int = 8
    channels: int = 1  # 1 = grayscale
    arch_args: dict = field(default_factory=dict)  # e.g. {"width": 32, "levels": 4}

    def __post_init__(self):
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {list(ARCHITECTURES)}, got {self.arch!r}")
        if self.output not in OUTPUT_MODES:
            raise ValueError(f"output must be one of {OUTPUT_MODES}, got {self.output!r}")
        if not 1 <= self.k < self.bit_depth:
            raise ValueError(f"k must be in 1..{self.bit_depth - 1}, got {self.k}")

    def to_dict(self) -> dict:
        return asdict(self)


def encode_planes(top: torch.Tensor, k: int, bit_depth: int) -> torch.Tensor:
    """(B, C, H, W) integer-valued top-k image → (B, C·k, H, W) planes as ±0.5, MSB first."""
    top = top.to(torch.int32)
    planes = [((top >> (bit_depth - 1 - i)) & 1) for i in range(k)]
    return torch.stack(planes, dim=2).flatten(1, 2).float() - 0.5


def pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    """Pad H and W at the bottom / right up to a multiple of m (reflect if possible)."""
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % m, (-w) % m
    if not (pad_h or pad_w):
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if min(h, w) > max(pad_h, pad_w) else "replicate")


class MSBNet(nn.Module):
    kind = "regression"  # stored in checkpoints; see msbgen.models.diffusion for "diffusion"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.backbone = ARCHITECTURES[config.arch](c * config.k, c, **config.arch_args)
        self.max_value = float((1 << config.bit_depth) - 1)
        self.residual_max = float((1 << (config.bit_depth - config.k)) - 1)

    def encode_input(self, top: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) integer-valued top-k image → (B, C·k, H, W) planes as ±0.5."""
        return encode_planes(top, self.config.k, self.config.bit_depth)

    def forward(self, top: torch.Tensor) -> torch.Tensor:
        """top: (B, C, H, W) with only the top k bits set (any integer or float dtype).

        Returns Î / (2^n - 1), shape (B, C, H, W), values in [0, 1].
        """
        x = self.encode_input(top)
        h, w = x.shape[-2:]
        z = self.backbone(pad_to_multiple(x, getattr(self.backbone, "multiple", 1)))[..., :h, :w]
        if self.config.output == "direct":
            return torch.sigmoid(z)
        return (top.float() + torch.sigmoid(z) * self.residual_max) / self.max_value

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
