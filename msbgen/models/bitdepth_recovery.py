"""PyTorch reproduction of bitplane-wise bit-depth recovery.

The implementation follows Punnappurath and Brown (arXiv:2005.01091): one
independent residual CNN is trained for every missing bitplane, with BCE loss,
and the models are applied sequentially at inference time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from msbgen.utils.thermal import ThermalGuard, ThermalLimits


def quantize(image: torch.Tensor, bits: int, bit_depth: int) -> torch.Tensor:
    """Zero-pad an image quantized to ``bits`` while retaining full-depth scale."""
    if not 1 <= bits <= bit_depth:
        raise ValueError(f"bits must be in 1..{bit_depth}, got {bits}")
    levels = 1 << bit_depth
    step = 1 << (bit_depth - bits)
    values = torch.round(image * (levels - 1)) if image.is_floating_point() else image
    return torch.floor(values / step) * step / (levels - 1)


def next_bit_target(image: torch.Tensor, input_bits: int, bit_depth: int) -> torch.Tensor:
    """Return the binary plane immediately below ``input_bits`` as ``NCHW`` floats."""
    if not image.is_floating_point():
        values = image.to(torch.int64)
    else:
        values = torch.round(image * ((1 << bit_depth) - 1)).to(torch.int64)
    shift = bit_depth - input_bits - 1
    if shift < 0:
        raise ValueError("input_bits must be smaller than bit_depth")
    return ((values >> shift) & 1).float()


class BitplaneResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        # Keras momentum=0.0 copies each batch's statistics; PyTorch's equivalent is 1.0.
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-4, momentum=1.0)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-4, momentum=1.0)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn1(x)
        residual = self.conv1(residual)
        residual = torch.relu(self.bn2(residual))
        residual = self.conv2(residual)
        return x + residual


class BitplaneResidualCNN(nn.Module):
    """The paper's DnCNN-style classifier for one missing bitplane."""

    def __init__(self, channels: int = 3, depth: int = 4, filters: int = 64):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.config = {"channels": channels, "depth": depth, "filters": filters}
        self.input = nn.Conv2d(channels, filters, 3, padding=1)
        self.blocks = nn.Sequential(*(BitplaneResidualBlock(filters) for _ in range(depth)))
        self.output_bn = nn.BatchNorm2d(filters, eps=1e-4, momentum=1.0)
        self.output = nn.Conv2d(filters, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits; use sigmoid for probabilities and BCEWithLogitsLoss for training."""
        x = self.input(x)
        x = self.blocks(x)
        return self.output(self.output_bn(x))


class BitplaneDataset(Dataset):
    """Non-overlapping paper-style patches for one input bit depth.

    Images may be ``HWC``/``HW`` NumPy arrays or ``CHW``/``HW`` tensors, stored
    either as integer values or floating-point values in ``[0, 1]``.
    """

    def __init__(self, images: Sequence[np.ndarray | torch.Tensor], input_bits: int,
                 bit_depth: int = 8, patch_size: int = 48, augment: bool = False):
        if not images:
            raise ValueError("images must not be empty")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.images = [self._as_chw(image, bit_depth) for image in images]
        self.input_bits, self.bit_depth = input_bits, bit_depth
        self.patch_size, self.augment = patch_size, augment
        self.index: list[tuple[int, int, int]] = []
        for image_index, image in enumerate(self.images):
            _, height, width = image.shape
            self.index.extend((image_index, y, x) for y in range(0, height - patch_size + 1, patch_size)
                              for x in range(0, width - patch_size + 1, patch_size))
        if not self.index:
            raise ValueError("no complete patches found")

    @staticmethod
    def _as_chw(image: np.ndarray | torch.Tensor, bit_depth: int) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.ndim == 2:
            tensor = tensor[None]
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3:
            raise ValueError(f"expected HW, HWC, or CHW image, got {tuple(tensor.shape)}")
        return tensor  # integers stay compact; `__getitem__` scales the patch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index, y, x = self.index[item]
        image = self.images[image_index]
        patch = image[:, y : y + self.patch_size, x : x + self.patch_size]
        patch = patch.float()
        if not image.is_floating_point():
            patch = patch / ((1 << self.bit_depth) - 1)
        patch = patch.clamp(0, 1)
        if self.augment:
            generator = torch.Generator().manual_seed(item)
            if torch.rand((), generator=generator) < 0.5:
                patch = patch.flip(-1)
            patch = torch.rot90(patch, int(torch.randint(4, (), generator=generator)), (-2, -1))
        return quantize(patch, self.input_bits, self.bit_depth), next_bit_target(patch, self.input_bits, self.bit_depth)


@dataclass
class BitplaneTrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    lr_drop_epoch: int = 16
    lr_drop_factor: float = 5.0
    workers: int = 0
    device: str = "cpu"
    thermal_guard: bool = True
    thermal: ThermalLimits = field(default_factory=ThermalLimits)


def train_bitplane(model: BitplaneResidualCNN, train_data: Dataset,
                   config: BitplaneTrainConfig | None = None,
                   val_data: Dataset | None = None) -> list[dict[str, float]]:
    """Train one plane model and return epoch metrics."""
    config = config or BitplaneTrainConfig()
    device = torch.device(config.device)
    model.to(device)
    guard = ThermalGuard(config.thermal, enabled=config.thermal_guard,
                         gpu_index=device.index or 0 if device.type == "cuda" else 0)
    print(guard.describe())
    loader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.workers,
                        pin_memory=device.type == "cuda", persistent_workers=config.workers > 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        if epoch == config.lr_drop_epoch + 1:
            for group in optimizer.param_groups:
                group["lr"] /= config.lr_drop_factor
        model.train()
        train_loss = 0.0
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
            guard.check()
        record = {"epoch": float(epoch), "loss": train_loss / count, "lr": optimizer.param_groups[0]["lr"]}
        if val_data is not None:
            model.eval()
            val_loss, val_count = 0.0, 0
            with torch.no_grad():
                for inputs, targets in DataLoader(
                    val_data, config.batch_size, num_workers=config.workers,
                    pin_memory=device.type == "cuda", persistent_workers=config.workers > 0,
                ):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    val_loss += nn.functional.binary_cross_entropy_with_logits(model(inputs), targets).item() * inputs.shape[0]
                    val_count += inputs.shape[0]
            record["val_loss"] = val_loss / val_count
        history.append(record)
        validation = f"  val_loss {record['val_loss']:.5f}" if "val_loss" in record else ""
        thermal = ""
        if guard.gpu_temp is not None:
            thermal += f"  GPU {guard.gpu_temp:.0f}C"
        if guard.cpu_temp is not None:
            thermal += f"  CPU {guard.cpu_temp:.0f}C"
        print(f"epoch {epoch:>3}/{config.epochs}  loss {record['loss']:.5f}  "
              f"lr {record['lr']:.2e}{validation}{thermal}", flush=True)
    return history


@torch.no_grad()
def recover_bitplanes(image: torch.Tensor, models: Sequence[nn.Module], input_bits: int,
                      bit_depth: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Sequentially recover missing planes; returns a normalized full-depth image."""
    if len(models) != bit_depth - input_bits:
        raise ValueError("one model is required for every missing bitplane")
    current = quantize(image.to(device), input_bits, bit_depth)
    maximum = float((1 << bit_depth) - 1)
    for offset, model in enumerate(models):
        bits = input_bits + offset
        model = model.to(device).eval()
        probabilities = torch.sigmoid(model(current))
        weight = float(1 << (bit_depth - bits - 1)) / maximum
        current = current + probabilities * weight
    return current.clamp(0, 1)
