import pytest

from msbgen.utils import ThermalGuard, ThermalLimits, cpu_guard, gpu_guard


class FakeSensor:
    """Returns the queued readings one per call, then keeps repeating the last one."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    return ThermalLimits(check_every_s=0, poll_s=1)


def test_limits_need_hysteresis():
    with pytest.raises(ValueError):
        ThermalLimits(gpu_max_c=80, gpu_resume_c=80)


def test_no_pause_when_cool(monkeypatch, fast):
    monkeypatch.setattr(gpu_guard, "get_gpu_temp", FakeSensor(60.0))
    monkeypatch.setattr(cpu_guard, "get_cpu_temp", FakeSensor(50.0))
    saved = []
    guard = ThermalGuard(fast, log=lambda m: None)
    assert guard.check(before_pause=lambda: saved.append(1)) == 0
    assert saved == [] and guard.pauses == 0
    assert guard.status() == {"gpu_temp": 60.0, "cpu_temp": 50.0, "paused_s": 0.0}


def test_pauses_until_gpu_cools(monkeypatch, fast):
    # init read, check read, wait: first read (hot), then cooling 78 → 74 → 71 (≤ 72 resumes), final re-read
    monkeypatch.setattr(gpu_guard, "get_gpu_temp", FakeSensor(60.0, 85.0, 85.0, 78.0, 74.0, 71.0, 70.0))
    monkeypatch.setattr(cpu_guard, "get_cpu_temp", FakeSensor(50.0))
    saved, logs = [], []
    guard = ThermalGuard(fast, log=logs.append)
    paused = guard.check(before_pause=lambda: saved.append(1))
    assert saved == [1]  # checkpoint written before pausing
    assert paused == 3.0 and guard.paused_s == 3.0 and guard.pauses == 1
    assert any("pausing" in m for m in logs) and any("resuming" in m for m in logs)


def test_cpu_hot_also_pauses(monkeypatch, fast):
    monkeypatch.setattr(gpu_guard, "get_gpu_temp", FakeSensor(60.0))
    monkeypatch.setattr(cpu_guard, "get_cpu_temp", FakeSensor(50.0, 95.0, 95.0, 85.0, 79.0, 70.0))
    guard = ThermalGuard(fast, log=lambda m: None)
    assert guard.check() == 2.0


def test_rate_limited(monkeypatch):
    sensor = FakeSensor(60.0)
    monkeypatch.setattr(gpu_guard, "get_gpu_temp", sensor)
    monkeypatch.setattr(cpu_guard, "get_cpu_temp", FakeSensor(None))
    guard = ThermalGuard(ThermalLimits(check_every_s=60), log=lambda m: None)
    for _ in range(100):
        guard.check()
    assert sensor.calls == 1  # only the initial read; the next check is due in 60 s
    guard.check(force=True)
    assert sensor.calls == 2


def test_no_sensors_is_a_noop(monkeypatch, fast):
    monkeypatch.setattr(gpu_guard, "get_gpu_temp", FakeSensor(None))
    monkeypatch.setattr(cpu_guard, "get_cpu_temp", FakeSensor(None))
    guard = ThermalGuard(fast, log=lambda m: None)
    assert guard.check() == 0 and "no sensor" in guard.describe()
    assert ThermalGuard(enabled=False).check() == 0


def test_real_sensors_do_not_crash():
    # Whatever this machine has (or not), reading must return a float or None.
    for temp in (gpu_guard.get_gpu_temp(), cpu_guard.get_cpu_temp()):
        assert temp is None or isinstance(temp, float)
