# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPU runs work: the probe, its backoff while the device recovers, and the point at which waiting stops."""

from __future__ import annotations

import asyncio

from kernelforge.evaluation import device_health
from kernelforge.evaluation.device_faults import DEVICE_HANG, DEVICE_UNRESPONSIVE, DeviceFault
from kernelforge.evaluation.device_health import probe_device, wait_for_device_recovery

_STILL_DOWN = DeviceFault(DEVICE_UNRESPONSIVE, "hipErrorNoDevice")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_the_wait_backs_off_between_probes_until_the_gpu_runs_work_again():
    answers = iter([_STILL_DOWN, _STILL_DOWN, None])
    clock = _Clock()

    async def probe():
        return next(answers)

    recovered = asyncio.run(wait_for_device_recovery(timeout_sec=900, probe=probe, sleep=clock.sleep, clock=clock))

    assert recovered is True
    assert clock.sleeps == [5.0, 10.0]


def test_the_wait_gives_up_at_its_limit_without_sleeping_past_it():
    clock = _Clock()

    async def probe():
        return _STILL_DOWN

    recovered = asyncio.run(wait_for_device_recovery(timeout_sec=100, probe=probe, sleep=clock.sleep, clock=clock))

    assert recovered is False
    assert clock.sleeps == [5.0, 10.0, 20.0, 40.0, 25.0]


def test_a_probe_whose_kernel_completes_reports_no_fault(monkeypatch):
    monkeypatch.setattr(device_health, "_PROBE", "raise SystemExit(0)")

    assert asyncio.run(probe_device(30.0)) is None


def test_a_probe_that_fails_reports_its_last_line_as_an_unresponsive_device(monkeypatch):
    monkeypatch.setattr(device_health, "_PROBE", "print('loading'); raise SystemExit('hipErrorNoDevice')")

    assert asyncio.run(probe_device(30.0)) == DeviceFault(DEVICE_UNRESPONSIVE, "hipErrorNoDevice")


def test_a_probe_that_names_a_hang_reports_the_hang(monkeypatch):
    line = "HW Exception by GPU node-3 (Agent handle: 0x1) reason :GPU Hang"
    monkeypatch.setattr(device_health, "_PROBE", f"print({line!r}); raise SystemExit(1)")

    fault = asyncio.run(probe_device(30.0))

    assert (fault.kind, fault.node, fault.line) == (DEVICE_HANG, 3, line)


def test_a_probe_that_does_not_finish_in_time_reports_an_unresponsive_device(monkeypatch):
    monkeypatch.setattr(device_health, "_PROBE", "import time; time.sleep(30)")

    fault = asyncio.run(probe_device(0.5))

    assert fault.kind == DEVICE_UNRESPONSIVE and fault.node is None
    assert "did not complete within" in fault.line
