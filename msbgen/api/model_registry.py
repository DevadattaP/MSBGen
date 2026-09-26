"""Trained models available to the web app.

Checkpoints are found as `<settings.models_dir>/<run>/best.pt` on every call, so a
newly finished training run shows up without restarting the server. Loaded models
are cached per (path, modification time). PyTorch is imported lazily: without it,
the app still works and simply lists no models.
"""

from __future__ import annotations

import threading
from pathlib import Path

from msbgen.config import settings

_cache: dict[Path, tuple[float, object]] = {}
_lock = threading.Lock()  # one inference at a time keeps GPU memory bounded


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def list_models() -> list:
    """Loaded `ModelReconstructor`s for every checkpoint, newest first."""
    if not torch_available():
        return []
    from msbgen.models import ModelReconstructor, discover_checkpoints

    models = []
    with _lock:
        for path in discover_checkpoints(settings.models_dir):
            mtime = path.stat().st_mtime
            cached = _cache.get(path)
            if cached is None or cached[0] != mtime:
                try:
                    cached = (mtime, ModelReconstructor(path, device=settings.device))
                except Exception:  # noqa: BLE001 - a broken checkpoint must not break the app
                    continue
                _cache[path] = cached
            models.append(cached[1])
    return models


def inference_lock() -> threading.Lock:
    return _lock
