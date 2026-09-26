"""Rate-limited GPU + CPU thermal guard for long-running loops (training, evaluation).

    guard = ThermalGuard(ThermalLimits())
    for batch in loader:
        ...                                   # one training step
        guard.check(before_pause=save_ckpt)   # cheap: reads sensors at most every 5 s

A training step takes a fraction of a second, so reading the sensors every step would
waste time (nvidia-smi is a subprocess). `check()` reads them at most every
`check_every_s` seconds. When either chip reaches its `max` temperature it calls
`before_pause` (training saves `last.pt` there, so even a hard shutdown loses almost
nothing), then blocks until the GPU and the CPU have both cooled to their `resume`
temperature. The whole machine idles during the pause, which cools it fastest.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from msbgen.utils import cpu_guard, gpu_guard


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ThermalLimits:
    gpu_max_c: float = 80.0  # pause at or above …
    gpu_resume_c: float = 72.0  # … resume at or below
    cpu_max_c: float = 90.0
    cpu_resume_c: float = 80.0
    check_every_s: float = 5.0  # how often check() actually reads the sensors
    poll_s: float = 5.0  # sensor polling interval while paused

    def __post_init__(self):
        if self.gpu_resume_c >= self.gpu_max_c or self.cpu_resume_c >= self.cpu_max_c:
            raise ValueError("resume temperatures must be below the max temperatures")


class ThermalGuard:
    def __init__(self, limits: ThermalLimits | None = None, enabled: bool = True, gpu_index: int = 0,
                 log: Callable[[str], None] = _stderr):
        self.limits = limits or ThermalLimits()
        self.enabled = enabled
        self.gpu_index = gpu_index
        self.log = log
        self.gpu_temp: float | None = None
        self.cpu_temp: float | None = None
        self.paused_s = 0.0  # total time spent cooling down
        self.pauses = 0
        self._last_check = -float("inf")
        if enabled:
            self._read()

    def _read(self) -> None:
        self.gpu_temp = gpu_guard.get_gpu_temp(self.gpu_index)
        self.cpu_temp = cpu_guard.get_cpu_temp()
        self._last_check = time.monotonic()

    def describe(self) -> str:
        if not self.enabled:
            return "thermal guard: off"
        lim = self.limits

        def part(name: str, temp: float | None, hi: float, lo: float) -> str:
            reading = f"{temp:.0f}°C" if temp is not None else "no sensor (not guarded)"
            return f"{name} {reading}, pause ≥ {hi:.0f}°C / resume ≤ {lo:.0f}°C"

        return ("thermal guard: " + part("GPU", self.gpu_temp, lim.gpu_max_c, lim.gpu_resume_c)
                + "; " + part("CPU", self.cpu_temp, lim.cpu_max_c, lim.cpu_resume_c))

    def too_hot(self) -> bool:
        lim = self.limits
        return ((self.gpu_temp is not None and self.gpu_temp >= lim.gpu_max_c)
                or (self.cpu_temp is not None and self.cpu_temp >= lim.cpu_max_c))

    def check(self, before_pause: Callable[[], None] | None = None, force: bool = False) -> float:
        """Read the sensors (at most every `check_every_s`, unless `force`) and pause if too hot.

        Returns the seconds spent paused (0 if no pause was needed).
        """
        if not self.enabled or (not force and time.monotonic() - self._last_check < self.limits.check_every_s):
            return 0.0
        self._read()
        if not self.too_hot():
            return 0.0

        if before_pause is not None:
            before_pause()
        lim = self.limits
        waited = 0.0
        # Wait for both chips; loop in case one heats up again while waiting for the other.
        while True:
            waited += gpu_guard.wait_for_safe_temp(lim.gpu_max_c, lim.gpu_resume_c, lim.poll_s, self.gpu_index, self.log)
            waited += cpu_guard.wait_for_safe_temp(lim.cpu_max_c, lim.cpu_resume_c, lim.poll_s, self.log)
            self._read()
            if not self.too_hot():
                break
        self.paused_s += waited
        self.pauses += 1
        return waited

    def status(self) -> dict:
        """Latest readings and totals, for logging."""
        return {"gpu_temp": self.gpu_temp, "cpu_temp": self.cpu_temp, "paused_s": round(self.paused_s, 1)}
