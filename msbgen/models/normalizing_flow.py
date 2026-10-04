"""Conditional multi-scale Glow model for the missing image residual."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock
from torch import nn

from msbgen.models.generative import randn
from msbgen.models.msb_net import encode_planes, pad_to_multiple


@dataclass
class NormalizingFlowConfig:
    kind: ClassVar[str] = "normalizing_flow"
    arch: ClassVar[str] = "conditional-glow"
    output: ClassVar[str] = "residual"

    k: int = 1
    bit_depth: int = 8
    channels: int = 1
    crop_size: int = 128
    scales: int = 4
    blocks_per_scale: int = 4
    condition_width: int = 32
    hidden_width: int = 64

    def __post_init__(self):
        if not 1 <= self.k < self.bit_depth:
            raise ValueError(f"k must be in 1..{self.bit_depth - 1}, got {self.k}")
        if self.crop_size < 2 ** self.scales or self.crop_size % (2 ** self.scales):
            raise ValueError("crop_size must be divisible by 2**scales")
        if self.scales < 1 or self.blocks_per_scale < 1:
            raise ValueError("scales and blocks_per_scale must be at least 1")
        if min(self.channels, self.condition_width, self.hidden_width) < 1:
            raise ValueError("channels and widths must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def _coupling_subnet(channels_in: int, channels_out: int, hidden_width: int) -> nn.Module:
    final = nn.Conv2d(hidden_width, channels_out, 3, padding=1)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return nn.Sequential(
        nn.Conv2d(channels_in, hidden_width, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(hidden_width, hidden_width, 3, padding=1),
        nn.SiLU(),
        final,
    )


class ConditionPyramid(nn.Module):
    """Encode the known bit planes at the spatial scale used by each flow stage."""

    def __init__(self, in_channels: int, width: int, scales: int):
        super().__init__()
        layers = []
        for scale in range(scales):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels if scale == 0 else width, width, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.SiLU(),
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, planes: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = planes
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class MSBNormalizingFlow(nn.Module):
    """Model the dequantized residual distribution conditioned on the top-k planes.

    Each input is factored into fixed-size patches because FrEIA's invertible blocks
    validate their spatial dimensions. Patches are conditionally independent under
    this model; this keeps training and full-image inference on the same event size.
    """

    kind = "normalizing_flow"

    def __init__(self, config: NormalizingFlowConfig):
        super().__init__()
        self.config = config
        self.max_value = float((1 << config.bit_depth) - 1)
        self.residual_levels = float(1 << (config.bit_depth - config.k))
        self.residual_max = self.residual_levels - 1
        self.condition_encoder = ConditionPyramid(config.channels * config.k, config.condition_width,
                                                  config.scales)
        self.flows = nn.ModuleList()
        for scale in range(config.scales):
            channels = config.channels * (2 ** (scale + 2))
            size = config.crop_size // (2 ** (scale + 1))
            flow = SequenceINN(channels, size, size)
            for _ in range(config.blocks_per_scale):
                flow.append(
                    AllInOneBlock,
                    cond=0,
                    cond_shape=(config.condition_width, size, size),
                    subnet_constructor=lambda cin, cout, hidden=config.hidden_width:
                        _coupling_subnet(cin, cout, hidden),
                    affine_clamping=1.9,
                    learned_householder_permutation=4,
                )
            self.flows.append(flow)

    def keep_top(self, x: torch.Tensor) -> torch.Tensor:
        drop = self.config.bit_depth - self.config.k
        return ((x.to(torch.int32) >> drop) << drop).float()

    def condition(self, top: torch.Tensor) -> list[torch.Tensor]:
        planes = encode_planes(top, self.config.k, self.config.bit_depth)
        return self.condition_encoder(planes)

    def _tiles(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        batch, channels, height, width = x.shape
        tile = self.config.crop_size
        x = pad_to_multiple(x, tile)
        padded_height, padded_width = x.shape[-2:]
        rows, cols = padded_height // tile, padded_width // tile
        tiles = x.unfold(2, tile, tile).unfold(3, tile, tile)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(batch * rows * cols, channels, tile, tile)
        return tiles, (batch, channels, height, width, padded_height, padded_width, rows, cols)

    @staticmethod
    def _untile(tiles: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        batch, channels, height, width, padded_height, padded_width, rows, cols = shape
        tile = tiles.shape[-1]
        x = tiles.reshape(batch, rows, cols, channels, tile, tile)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, padded_height, padded_width)
        return x[..., :height, :width]

    def log_prob(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Dequantization-based residual NLL in bits/dimension, one value per image."""
        tiles, shape = self._tiles(x.float())
        top = self.keep_top(tiles)
        residual = tiles - top
        gen_device = generator.device if generator is not None else residual.device
        uniform = torch.rand(residual.shape, generator=generator, device=gen_device).to(residual.device)
        unit = ((residual + uniform) / self.residual_levels).clamp(1e-6, 1 - 1e-6)
        logit_jacobian = (-unit.log() - torch.log1p(-unit)).flatten(1).sum(1)
        value = torch.logit(unit)
        conditions = self.condition(top)
        latents = []
        logdet = torch.zeros(value.shape[0], device=value.device, dtype=value.dtype)
        for scale, flow in enumerate(self.flows):
            value = F.pixel_unshuffle(value, 2)
            value, stage_logdet = flow(value, c=[conditions[scale]])
            logdet = logdet + stage_logdet
            if scale < len(self.flows) - 1:
                latent, value = value.chunk(2, dim=1)
                latents.append(latent)
            else:
                latents.append(value)
        log_base = sum(-0.5 * (z.square() + math.log(2 * math.pi)).flatten(1).sum(1) for z in latents)
        dimensions = tiles.shape[1] * tiles.shape[2] * tiles.shape[3]
        nll = -log_base - logdet - logit_jacobian + dimensions * math.log(self.residual_levels)
        tile_bpd = nll / (dimensions * math.log(2))
        batch, _, _, _, _, _, rows, cols = shape
        return tile_bpd.reshape(batch, rows * cols).mean(1)

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None,
             reduction: str = "mean") -> torch.Tensor:
        values = self.log_prob(x, generator)
        return values.mean() if reduction == "mean" else values

    @torch.no_grad()
    def sample(self, top: torch.Tensor, steps: int | None = None, generator: torch.Generator | None = None,
               on_step: Callable[[], object] | None = None) -> torch.Tensor:
        del steps  # Keep the common generative-model interface; this flow has no solver steps.
        tiles, shape = self._tiles(top.float())
        top_tiles = self.keep_top(tiles)
        conditions = self.condition(top_tiles)
        batch = tiles.shape[0]
        latents = []
        for scale in range(self.config.scales):
            channels = self.config.channels * (2 ** (scale + 2))
            size = self.config.crop_size // (2 ** (scale + 1))
            latent_channels = channels if scale == self.config.scales - 1 else channels // 2
            latents.append(randn((batch, latent_channels, size, size), generator, tiles.device))

        value = latents[-1]
        for scale in reversed(range(self.config.scales)):
            if scale < self.config.scales - 1:
                value = F.pixel_shuffle(value, 2)
                value = torch.cat((latents[scale], value), dim=1)
            value, _ = self.flows[scale](value, c=[conditions[scale]], rev=True, jac=False)
            if on_step is not None:
                on_step()
        value = F.pixel_shuffle(value, 2)
        residual = (torch.sigmoid(value) * self.residual_levels - 0.5).clamp(0, self.residual_max)
        estimate = (top_tiles + residual) / self.max_value
        return self._untile(estimate, shape)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())