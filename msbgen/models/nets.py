"""Backbones: a plain CNN (PLAN baseline 2) and a U-Net (baselines 3 and 4).

Both map (B, in_ch, H, W) → (B, out_ch, H, W) logits and are fully convolutional,
so a model trained on 128×128 crops runs on any image size.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """`depth` 3×3 conv layers at constant width: a purely local, pixelwise regressor.

    Its receptive field is 2·depth + 1 pixels, which bounds how much context it can use.
    """

    multiple = 1  # input size needs no padding

    def __init__(self, in_ch: int, out_ch: int, width: int = 64, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, out_ch, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UNet(nn.Module):
    """Encoder–decoder with skip connections.

    `levels` downsamplings (×2 each) give a large receptive field, so the network can
    use scene-level context from B7 (shapes, gradients) rather than only neighbours.
    Channels double per level: width, 2·width, … up to 2^levels · width at the bottom.
    """

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, levels: int = 4):
        super().__init__()
        self.multiple = 2**levels  # H and W must be divisible by this
        chans = [width * 2**i for i in range(levels + 1)]
        self.inc = conv_block(in_ch, chans[0])
        self.down = nn.ModuleList(conv_block(chans[i], chans[i + 1]) for i in range(levels))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(levels)))
        self.dec = nn.ModuleList(conv_block(2 * chans[i], chans[i]) for i in reversed(range(levels)))
        self.out = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.down:
            skips.append(down(self.pool(skips[-1])))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec, strict=True):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        return self.out(h)


ARCHITECTURES = {"cnn": SimpleCNN, "unet": UNet}
