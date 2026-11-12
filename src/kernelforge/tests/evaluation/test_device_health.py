# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Waiting for a reset GPU to run work again: the probe, its backoff, and the point at which waiting stops."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.evaluation import device_health
from kernelforge.evaluation.device_health import probe_device, wait_for_device_recovery


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
    answers = iter([False, False, True])
    clock = _Clock()

    async def probe():
        return next(answers)

    recovered = asyncio.run(wait_for_device_recovery(timeout_sec=900, probe=probe, sleep=clock.sleep, clock=clock))

    assert recovered is True
    assert clock.sleeps == [5.0, 10.0]


def test_the_wait_gives_up_at_its_limit_without_sleeping_past_it():
    clock = _Clock()

    async def probe():
        return False

    recovered = asyncio.run(wait_for_device_recovery(timeout_sec=100, probe=probe, sleep=clock.sleep, clock=clock))

    assert recovered is False
    assert clock.sleeps == [5.0, 10.0, 20.0, 40.0, 25.0]


@pytest.mark.parametrize(
    "program, timeout_sec, healthy",
    [
        ("raise SystemExit(0)", 30.0, True),
        ("raise SystemExit(1)", 30.0, False),
        ("import time; time.sleep(30)", 0.5, False),
    ],
)
def test_the_probe_reports_whether_its_child_process_ran_to_completion(monkeypatch, program, timeout_sec, healthy):
    monkeypatch.setattr(device_health, "_PROBE", program)

    assert asyncio.run(probe_device(timeout_sec)) is healthy
