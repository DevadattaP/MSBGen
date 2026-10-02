"""Trained models available to the web app.

Checkpoints are found as `<settings.models_dir>/<run>/best.pt` on every call, so a
newly finished training run shows up without restarting the server. Loaded models
are cached per (path, modification time). PyTorch is imported lazily: without it,
the app still works and simply lists no models.

Generative models (diffusion, flow matching) check a thermal guard (default limits, as
in training) after every sampling step: a full-resolution sample is several seconds of
full GPU load, and
repeated requests could otherwise overheat a laptop. When it is too hot, the request
waits until the GPU and CPU have cooled.

The perceptual metrics (LPIPS, DISTS) are networks too: they are loaded on first use,
once, and run under the same lock. A metric whose package or weights are missing is
left out (its score is null) instead of failing the request.
"""

from __future__ import annotations

import threading
from pathlib import Path

from msbgen.config import settings

_cache: dict[Path, tuple[float, object]] = {}
_lock = threading.Lock()  # one inference at a time keeps GPU memory bounded
_guard = None  # ThermalGuard, created on first use
_perceptual: dict | None = None  # name → metric(reference, estimate, data_range), loaded on first use


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
    """Loaded reconstructors (regression, diffusion or flow) for every checkpoint, newest first."""
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


def perceptual_metrics() -> dict:
    """{"lpips": fn, "dists": fn} for the metrics that could be loaded; {} when disabled."""
    global _perceptual
    if not settings.perceptual or not torch_available():
        return {}
    with _lock:
        if _perceptual is None:
            from msbgen.models import perceptual

            _perceptual = {}
            for name, cls in (("lpips", perceptual.LPIPS), ("dists", perceptual.DISTS)):
                try:
                    _perceptual[name] = cls(settings.device)
                except Exception:  # noqa: BLE001 - missing package or weights: just leave it out
                    continue
    return _perceptual


def inference_lock() -> threading.Lock:
    return _lock
