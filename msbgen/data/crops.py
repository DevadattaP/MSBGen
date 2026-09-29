"""PyTorch datasets of grayscale crops from `prepare_data` output (`*.npy`, uint8 (H, W)).

Images are memory-mapped, so only the cropped pixels are read from disk; an 800-image
DIV2K set does not need to fit in RAM. The datasets return raw uint8 crops; the model
itself derives the transmitted planes, so the same data serves any k.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def list_arrays(folder: Path) -> list[Path]:
    """The .npy files of a prepared dataset (manifest order when present)."""
    manifest = folder / "manifest.json"
    if manifest.exists():
        files = [folder / f["file"] for f in json.loads(manifest.read_text())["files"]]
    else:
        files = sorted(folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy images in {folder}; run `python -m msbgen.cli.prepare_data` first")
    return files


class RandomCrops(Dataset):
    """Random `crop`×`crop` patches with random flips / 90° rotations.

    The dataset has `samples` items; item i is fully determined by (seed, i). Images are
    chosen with probability proportional to their area, so every pixel is equally likely
    to be seen.
    """

    def __init__(self, folders: list[Path], crop: int = 128, samples: int = 16_000, augment: bool = True, seed: int = 0):
        self.files = [f for folder in folders for f in list_arrays(Path(folder))]
        shapes = [np.load(f, mmap_mode="r").shape for f in self.files]
        keep = [i for i, s in enumerate(shapes) if min(s) >= crop]
        if not keep:
            raise ValueError(f"no image is at least {crop}×{crop}")
        self.files = [self.files[i] for i in keep]
        areas = np.array([shapes[i][0] * shapes[i][1] for i in keep], dtype=np.float64)
        self.weights = areas / areas.sum()
        self.total_pixels = int(areas.sum())  # for expressing progress in epochs
        self.crop, self.samples, self.augment, self.seed = crop, samples, augment, seed
        self._arrays: dict[int, np.ndarray] = {}  # per-worker mmap cache

    def __len__(self) -> int:
        return self.samples

    def _array(self, i: int) -> np.ndarray:
        if i not in self._arrays:
            self._arrays[i] = np.load(self.files[i], mmap_mode="r")
        return self._arrays[i]

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng((self.seed, index))
        img = self._array(int(rng.choice(len(self.files), p=self.weights)))
        y = int(rng.integers(0, img.shape[0] - self.crop + 1))
        x = int(rng.integers(0, img.shape[1] - self.crop + 1))
        patch = np.asarray(img[y : y + self.crop, x : x + self.crop])
        if self.augment:
            patch = np.rot90(patch, int(rng.integers(4)))
            if rng.integers(2):
                patch = patch[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(patch))[None]  # (1, crop, crop) uint8


class CenterCrops(Dataset):
    """One deterministic centre crop per image, for fast and repeatable validation."""

    def __init__(self, folders: list[Path], crop: int = 256, limit: int | None = None):
        self.files = [f for folder in folders for f in list_arrays(Path(folder))][:limit]
        self.crop = crop

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = np.load(self.files[index], mmap_mode="r")
        c = min(self.crop, *img.shape)
        y, x = (img.shape[0] - c) // 2, (img.shape[1] - c) // 2
        return torch.from_numpy(np.ascontiguousarray(img[y : y + c, x : x + c]))[None]
