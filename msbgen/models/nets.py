"""Backbones: a plain CNN (PLAN baseline 2), a U-Net (baselines 3 and 4) and a
time-conditioned U-Net for the generative models (Milestone 3 onwards).

All map (B, in_ch, H, W) → (B, out_ch, H, W) and are fully convolutional, so a model
trained on 128×128 crops runs on any image size.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.heads = heads
        self.norm1, self.qkv, self.proj = nn.LayerNorm(dim), nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


def sincos_2d(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding, (h·w, dim); defined for any grid size."""
    q = dim // 4
    freqs = torch.exp(-math.log(10_000) * torch.arange(q, device=device, dtype=torch.float32) / q)
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * freqs[None]
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * freqs[None]
    ys, xs = ys[:, None, :].expand(h, w, q), xs[None, :, :].expand(h, w, q)
    return torch.cat([ys.sin(), ys.cos(), xs.sin(), xs.cos()], dim=-1).reshape(h * w, dim)


class ViT(nn.Module):
    """Vision Transformer: patch tokens with global self-attention, plus a small conv branch.

    Patches (`patch`×`patch`) are embedded, mixed by `depth` transformer blocks, and
    unpatchified with a linear head. A 3×3 conv branch on the raw planes is fused with the
    unpatchified features so the output has pixel-level detail and no patch-grid artefacts.
    Position embeddings are fixed sin-cos, so any image size works (padded to a multiple of `patch`).
    """

    def __init__(self, in_ch: int, out_ch: int, patch: int = 8, dim: int = 192, depth: int = 6, heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        if dim % 4 or dim % heads:
            raise ValueError("dim must be divisible by 4 and by heads")
        self.multiple, self.patch, self.dim = patch, patch, dim
        self.embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.blocks = nn.ModuleList(ViTBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 16 * patch * patch)
        self.local = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, out_ch, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.patch
        gh, gw = h // p, w // p
        t = self.embed(x).flatten(2).transpose(1, 2) + sincos_2d(gh, gw, self.dim, x.device).to(x.dtype)
        for block in self.blocks:
            t = block(t)
        t = self.head(self.norm(t))  # (B, N, 16·p²)
        t = t.reshape(b, gh, gw, 16, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 16, h, w)
        return self.fuse(torch.cat([t, self.local(x)], dim=1))


ARCHITECTURES = {"cnn": SimpleCNN, "unet": UNet, "vit": ViT}


# ---------- time-conditioned U-Net (generative models) ----------


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of continuous times t ∈ [0, 1], shape (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    """GroupNorm → SiLU → conv, twice, with the time embedding added in between."""

    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.norm1, self.conv1 = _norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2, self.conv2 = _norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TimeUNet(nn.Module):
    """U-Net whose blocks are conditioned on a time t ∈ [0, 1] (noise level / flow time).

    `levels` downsamplings give levels + 1 resolutions; resolution i has
    width · 2^min(i, 2) channels (e.g. 32, 64, 128, 128) and `blocks` ResBlocks on the
    way down and on the way up. No attention, so memory stays linear in the image area
    and full-resolution sampling works on a small GPU.
    """

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, levels: int = 3, blocks: int = 2):
        super().__init__()
        self.multiple = 2**levels
        chans = [width * 2 ** min(i, 2) for i in range(levels + 1)]
        self.tdim = tdim = 4 * width
        self.time_mlp = nn.Sequential(nn.Linear(width, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.width = width
        self.inc = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        c = chans[0]
        for i, ch in enumerate(chans):
            self.down_blocks.append(nn.ModuleList(ResBlock(c if j == 0 else ch, ch, tdim) for j in range(blocks)))
            c = ch
            if i < levels:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        self.mid = nn.ModuleList([ResBlock(c, c, tdim), ResBlock(c, c, tdim)])
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in reversed(range(levels + 1)):
            ch = chans[i]
            self.up_blocks.append(nn.ModuleList(ResBlock((c + ch) if j == 0 else ch, ch, tdim) for j in range(blocks)))
            c = ch
            if i > 0:
                self.upsample.append(nn.Conv2d(ch, chans[i - 1], 3, padding=1))
                c = chans[i - 1]
        self.out_norm = _norm(chans[0])
        self.out = nn.Conv2d(chans[0], out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)  # start by predicting 0
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.width))
        h = self.inc(x)
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        for block in self.mid:
            h = block(h, temb)
        for i, blocks in enumerate(self.up_blocks):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, temb)
            if i < len(self.upsample):
                h = self.upsample[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(h)))
