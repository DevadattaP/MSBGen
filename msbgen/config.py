"""Application settings (override with environment variables prefixed MSBGEN_)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "web" / "static"


@dataclass(frozen=True)
class Settings:
    max_upload_mb: float = float(os.getenv("MSBGEN_MAX_UPLOAD_MB", "25"))
    max_pixels: int = int(os.getenv("MSBGEN_MAX_PIXELS", str(4096 * 4096)))
    # Trained checkpoints are discovered as <models_dir>/<run>/best.pt.
    models_dir: Path = Path(os.getenv("MSBGEN_MODELS_DIR", "runs"))
    # Sequential bitplane CNNs are discovered as <bitdepth_models_dir>/bitdepth-*/bit_NN.pt.
    bitdepth_models_dir: Path = Path(
        os.getenv("MSBGEN_BITDEPTH_MODELS_DIR", os.getenv("MSBGEN_MODELS_DIR", "runs"))
    )
    # Optional JSON mapping of color mode and k to ordered checkpoint paths.
    model_registry_file: Path | None = Path(os.environ["MSBGEN_MODEL_REGISTRY"]) if os.getenv("MSBGEN_MODEL_REGISTRY") else None
    device: str | None = os.getenv("MSBGEN_DEVICE") or None  # default: cuda if available
    # Inference pauses at these temperatures and resumes 8 (GPU) / 10 (CPU) degrees lower.
    gpu_max_c: float = float(os.getenv("MSBGEN_GPU_MAX_C", "80"))
    cpu_max_c: float = float(os.getenv("MSBGEN_CPU_MAX_C", "90"))
    # Saved test-set evaluations (msbgen.cli.evaluate output) are read from <eval_dir>/*/summary.json.
    eval_dir: Path = Path(os.getenv("MSBGEN_EVAL_DIR", "outputs/eval"))
    # LPIPS / DISTS per reconstruction (needs `lpips` / `piq`); skipped above perceptual_max_pixels.
    perceptual: bool = os.getenv("MSBGEN_PERCEPTUAL", "1") not in ("0", "false", "no")
    perceptual_max_pixels: int = int(os.getenv("MSBGEN_PERCEPTUAL_MAX_PIXELS", str(2048 * 2048)))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


settings = Settings()
