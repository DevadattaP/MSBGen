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

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


settings = Settings()
