"""
msbgen/utils/gpu_guard.py
=========================
Thermal guard for GPU training on laptops without a usable power/thermal ceiling.

The development laptop's GeForce MX350 reports `temperature.gpu` through nvidia-smi
but not `power.draw` / `power.limit`, so `nvidia-smi -pl <watts>` can't cap it. Its
own slowdown point is 100 °C, but the laptop shut itself down at ~95 °C during a
baseline training run. Training therefore has to keep the chip cooler itself.

`wait_for_safe_temp()` blocks once `temperature.gpu` reaches `max_temp_c`, re-checking
every `poll_interval_s`, and only returns once it has dropped to `resume_temp_c`. The
lower resume threshold (hysteresis) stops it flapping on/off at the boundary. The
training loop calls it through `msbgen.utils.thermal.ThermalGuard`, which rate-limits
the checks and saves a checkpoint before pausing.

No-ops cleanly wherever nvidia-smi isn't available (CPU-only machines, no NVIDIA GPU,
PATH issues): `get_gpu_temp()` returns None and every check is skipped.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_gpu_temp(index: int = 0) -> float | None:
    """Current temperature of GPU `index` in °C, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_safe_temp(
    max_temp_c: float = 80.0,
    resume_temp_c: float = 72.0,
    poll_interval_s: float = 5.0,
    index: int = 0,
    log: Callable[[str], None] = _stderr,
) -> float:
    """Block while the GPU is at/above `max_temp_c`, until it cools to `resume_temp_c`.

    Returns the seconds spent waiting (0 when the GPU is below the limit or has no sensor).
    """
    temp = get_gpu_temp(index)
    if temp is None or temp < max_temp_c:
        return 0.0

    log(f"GPU at {temp:.0f}°C ≥ {max_temp_c:.0f}°C: pausing to cool down")
    waited = 0.0
    while temp is not None and temp > resume_temp_c:
        time.sleep(poll_interval_s)
        waited += poll_interval_s
        temp = get_gpu_temp(index)
        if temp is not None and waited % 30 < poll_interval_s:
            log(f"  still cooling: GPU {temp:.0f}°C (target ≤ {resume_temp_c:.0f}°C, waited {waited:.0f}s)")
    log(f"GPU cooled to {temp if temp is not None else '?'}°C: resuming (paused {waited:.0f}s)")
    return waited
