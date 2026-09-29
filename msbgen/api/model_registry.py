"""Trained models available to the web app.

Checkpoints are found as `<settings.models_dir>/<run>/best.pt` on every call, so a
newly finished training run shows up without restarting the server. Loaded models
are cached per (path, modification time). PyTorch is imported lazily: without it,
the app still works and simply lists no models.

Diffusion models check a thermal guard (default limits, as in training) after every
denoising step: a full-resolution sample is several seconds of full GPU load, and
repeated requests could otherwise overheat a laptop. When it is too hot, the request
waits until the GPU and CPU have cooled.
"""

from __future__ import annotations

import threading
from pathlib import Path

from msbgen.config import settings

_cache: dict[Path, tuple[float, object]] = {}
_lock = threading.Lock()  # one inference at a time keeps GPU memory bounded
_guard = None  # ThermalGuard, created on first use


def _thermal_check() -> None:
    global _guard
    if _guard is None:
        from msbgen.utils.thermal import ThermalGuard

        _guard = ThermalGuard()
    _guard.check()


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def list_models() -> list:
    """Loaded reconstructors (regression or diffusion) for every checkpoint, newest first."""
    if not torch_available():
        return []
    from msbgen.models import discover_checkpoints, load_reconstructor

    models = []
    with _lock:
        for path in discover_checkpoints(settings.models_dir):
            mtime = path.stat().st_mtime
            cached = _cache.get(path)
            if cached is None or cached[0] != mtime:
                try:
                    cached = (mtime, load_reconstructor(path, device=settings.device, on_step=lambda: _thermal_check()))
                except Exception:  # noqa: BLE001 - a broken checkpoint must not break the app
                    continue
                _cache[path] = cached
            models.append(cached[1])
    return models


def inference_lock() -> threading.Lock:
    return _lock
