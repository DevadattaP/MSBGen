"""Trained models available to the web app.

An optional JSON registry selects ordered model paths by color mode and k. Without
one, checkpoints are discovered under `<settings.models_dir>` for grayscale and
`<settings.models_dir>/rgb` for RGB. PyTorch is imported lazily.

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

import json
import threading
from functools import lru_cache
from pathlib import Path

from msbgen.config import settings

_lock = threading.Lock()  # one inference at a time keeps GPU memory bounded
_guard = None  # ThermalGuard, created on first use
_perceptual: dict | None = None  # name → metric(reference, estimate, data_range), loaded on first use


def _thermal_guard():
    global _guard
    if _guard is None:
        from msbgen.utils.thermal import ThermalGuard, ThermalLimits

        _guard = ThermalGuard(ThermalLimits(
            gpu_max_c=settings.gpu_max_c, gpu_resume_c=settings.gpu_max_c - 8,
            cpu_max_c=settings.cpu_max_c, cpu_resume_c=settings.cpu_max_c - 10,
            check_every_s=1.0,
        ))
    return _guard


def thermal_check(force: bool = False) -> None:
    """Block while the GPU or CPU is too hot; `force` skips the sensor-read rate limit."""
    _thermal_guard().check(force=force)


def paused_seconds() -> float:
    return _guard.paused_s if _guard is not None else 0.0


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _registry_data() -> dict | None:
    path = settings.model_registry_file
    if path is None:
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model registry {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Model registry {path} must contain a JSON object.")
    return data


def _configured_candidates(color_mode: str, k: int, data: dict) -> list[tuple[str, Path]]:
    color_key = "grayscale" if color_mode == "L" else "rgb"
    mode_entries = data.get(color_key, {})
    if not isinstance(mode_entries, dict):
        raise ValueError(f"Model registry entry {color_key} must be an object keyed by k.")
    entries = mode_entries.get(str(k), [])
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError(f"Model registry entry {color_key}.{k} must be a list of paths.")
    root = settings.model_registry_file.parent
    candidates = []
    for item in entries:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        kind = "checkpoint" if path.suffix == ".pt" else "bitdepth"
        candidates.append((kind, path))
    return candidates


def model_paths(color_mode: str | None = None) -> list[Path]:
    """Checkpoint paths selected by mode, or all configured paths for the model list."""
    if not torch_available():
        return []
    from msbgen.models import discover_checkpoints

    registry = _registry_data()
    if registry is not None:
        modes = (color_mode,) if color_mode else ("L", "RGB")
        paths = []
        for mode in modes:
            for k in (1, 2, 3):
                paths.extend(path for kind, path in _configured_candidates(mode, k, registry) if kind == "checkpoint")
        return list(dict.fromkeys(paths))

    roots = (settings.models_dir, settings.models_dir / "rgb") if color_mode is None else (
        settings.models_dir if color_mode == "L" else settings.models_dir / "rgb",
    )
    return list(dict.fromkeys(path for root in roots for path in discover_checkpoints(root)))


def bitdepth_model_dirs(color_mode: str | None = None) -> list[Path]:
    """Sequential CNN folders such as ``runs/bitdepth-d4``."""
    roots = (settings.bitdepth_models_dir, settings.bitdepth_models_dir / "rgb") if color_mode is None else (
        settings.bitdepth_models_dir if color_mode == "L" else settings.bitdepth_models_dir / "rgb",
    )
    folders = (folder for root in dict.fromkeys(roots) if root.is_dir() for folder in root.glob("bitdepth-*")
               if folder.is_dir() and any(folder.glob("bit_*.pt")))
    return sorted(folders, key=lambda folder: folder.stat().st_mtime, reverse=True)


def model_candidates(color_mode: str, k: int) -> list[tuple[str, Path]]:
    """Ordered candidate models for the selected mode and transmitted bit count."""
    registry = _registry_data()
    if registry is not None:
        return _configured_candidates(color_mode, k, registry)
    checkpoints = [("checkpoint", path) for path in model_paths(color_mode)]
    bitdepth = [("bitdepth", path) for path in bitdepth_model_dirs(color_mode)]
    return [*checkpoints, *bitdepth]


def load_model(path: Path, device: str | None = None):
    from msbgen.models import load_reconstructor

    return load_reconstructor(path, device=device or settings.device, on_step=thermal_check)


@lru_cache(maxsize=1024)
def _is_compatible(kind: str, path: Path, mtime: float, k: int, n: int, channels: int) -> bool:
    try:
        if kind == "checkpoint":
            from msbgen.models import read_checkpoint

            cfg = read_checkpoint(path)["model_config"]
            return (cfg["k"], cfg["bit_depth"], cfg["channels"]) == (k, n, channels)
        from msbgen.models.reconstructor import BitplaneReconstructor

        return BitplaneReconstructor(path, device="cpu").supports(k, n, channels)
    except Exception:  # noqa: BLE001 - unreadable models are simply not offered
        return False


def compatible_candidates(color_mode: str, k: int, n: int, channels: int) -> list[tuple[str, Path]]:
    """Ordered candidates that can run at this mode and k (cached per file modification time)."""
    found = []
    for kind, path in model_candidates(color_mode, k):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if _is_compatible(kind, path, mtime, k, n, channels):
            found.append((kind, path))
    return found


def cuda_requested() -> bool:
    import torch

    return torch.device(settings.device).type == "cuda" if settings.device else torch.cuda.is_available()


def is_cuda_oom(error: Exception) -> bool:
    import torch

    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if oom_type and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or cuda_requested())


def release_model_memory() -> None:
    """Drop unused CUDA allocations between model runs."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
