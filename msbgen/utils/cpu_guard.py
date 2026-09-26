"""
msbgen/utils/cpu_guard.py
=========================
CPU counterpart to `msbgen/utils/gpu_guard.py`.

During training the CPU is busy too: the DataLoader workers read and augment crops,
and SSIM runs on the CPU during validation. On a laptop the CPU and GPU usually share
one heat-pipe and fan, so a hot CPU also heats the GPU (and vice versa). Sustained
load can reach thermal throttling or, worse, an uncontrolled shutdown.

`wait_for_safe_temp()` blocks once the CPU package reaches `max_temp_c` and returns
once it has cooled to `resume_temp_c`. It uses psutil's temperature sensors
(`coretemp` "Package id 0" on Intel, `k10temp` "Tctl" on AMD, ...) and no-ops cleanly
if none are exposed: `get_cpu_temp()` returns None and every check is skipped.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

SENSOR_CHIPS = ("coretemp", "k10temp", "zenpower", "cpu_thermal")
PACKAGE_LABELS = ("Package id 0", "Tctl", "")


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_cpu_temp() -> float | None:
    """Current CPU package temperature in °C, or None if no suitable sensor is exposed."""
    try:
        import psutil

        temps = psutil.sensors_temperatures()
    except (ImportError, AttributeError, OSError):
        return None
    for chip in SENSOR_CHIPS:
        entries = temps.get(chip)
        if entries:
            for e in entries:
                if e.label in PACKAGE_LABELS:
                    return float(e.current)
            return float(entries[0].current)
    return None


def wait_for_safe_temp(
    max_temp_c: float = 90.0,
    resume_temp_c: float = 80.0,
    poll_interval_s: float = 5.0,
    log: Callable[[str], None] = _stderr,
) -> float:
    """Block while the CPU is at/above `max_temp_c`, until it cools to `resume_temp_c`.

    Returns the seconds spent waiting (0 when the CPU is below the limit or has no sensor).
    """
    temp = get_cpu_temp()
    if temp is None or temp < max_temp_c:
        return 0.0

    log(f"CPU at {temp:.0f}°C ≥ {max_temp_c:.0f}°C: pausing to cool down")
    waited = 0.0
    while temp is not None and temp > resume_temp_c:
        time.sleep(poll_interval_s)
        waited += poll_interval_s
        temp = get_cpu_temp()
        if temp is not None and waited % 30 < poll_interval_s:
            log(f"  still cooling: CPU {temp:.0f}°C (target ≤ {resume_temp_c:.0f}°C, waited {waited:.0f}s)")
    log(f"CPU cooled to {temp if temp is not None else '?'}°C: resuming (paused {waited:.0f}s)")
    return waited
