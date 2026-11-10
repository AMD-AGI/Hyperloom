# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``ray status`` probe that times out says nothing about the head: it must not tear a healthy head down."""

from __future__ import annotations

import subprocess

import pytest

from hyperloom.orchestrator.actions.executors import _ray_runtime as ray_runtime

#: How long a healthy head on an R9700 host takes to answer ``ray status``; above the 5 s default probe timeout.
SLOW_HEAD_LATENCY_SEC = 6.0


class _Completed:
    def __init__(self, returncode: int):
        self.returncode = returncode


def _install_ray_cli(monkeypatch, *, status_latency: float, status_rc: int = 0) -> list[tuple[str, ...]]:
    """Stub the ``ray`` CLI: ``ray status`` answers after ``status_latency`` seconds, anything slower than its timeout
    raises ``TimeoutExpired``. Returns the list of ``ray <verb>`` invocations seen.
    """
    seen: list[tuple[str, ...]] = []

    def _run(cmd, **kwargs):
        seen.append(tuple(cmd[:2]))
        if cmd[:2] == ["ray", "status"]:
            if status_latency > kwargs["timeout"]:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return _Completed(status_rc)
        return _Completed(0)

    monkeypatch.setattr(ray_runtime.subprocess, "run", _run)
    monkeypatch.delenv("HYPERLOOM_RAY_STATUS_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("HYPERLOOM_RAY_READY_TIMEOUT_SEC", raising=False)
    return seen


def test_slow_healthy_head_is_neither_stopped_nor_restarted(monkeypatch):
    seen = _install_ray_cli(monkeypatch, status_latency=SLOW_HEAD_LATENCY_SEC)

    ray_runtime.ensure_ray_cluster()

    assert ("ray", "stop") not in seen
    assert ("ray", "start") not in seen


def test_head_that_never_answers_raises_without_restarting(monkeypatch):
    seen = _install_ray_cli(monkeypatch, status_latency=float("inf"))
    monkeypatch.setenv("HYPERLOOM_RAY_READY_TIMEOUT_SEC", "0.05")

    with pytest.raises(RuntimeError, match="did not answer `ray status`"):
        ray_runtime.ensure_ray_cluster()

    assert ("ray", "stop") not in seen
    assert ("ray", "start") not in seen


def test_head_that_is_down_is_still_restarted(monkeypatch):
    seen = _install_ray_cli(monkeypatch, status_latency=0.0, status_rc=1)
    monkeypatch.setenv("HYPERLOOM_RAY_READY_TIMEOUT_SEC", "0.05")

    with pytest.raises(RuntimeError, match="cluster is not reachable"):
        ray_runtime.ensure_ray_cluster()

    assert ("ray", "stop") in seen
    assert ("ray", "start") in seen


def test_new_head_is_waited_for_until_it_answers(monkeypatch):
    answers = iter([ray_runtime.RayProbe.DOWN, ray_runtime.RayProbe.UNKNOWN, ray_runtime.RayProbe.UP])
    monkeypatch.setattr(ray_runtime, "ray_probe", lambda timeout=None: next(answers))
    monkeypatch.setattr(ray_runtime.time, "sleep", lambda _seconds: None)

    assert ray_runtime._wait_for_ray_head() is ray_runtime.RayProbe.UP


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_Completed(0), ray_runtime.RayProbe.UP),
        (_Completed(1), ray_runtime.RayProbe.DOWN),
        (subprocess.TimeoutExpired(["ray", "status"], 5.0), ray_runtime.RayProbe.UNKNOWN),
        (FileNotFoundError("ray"), ray_runtime.RayProbe.DOWN),
    ],
)
def test_ray_probe_tells_the_outcomes_apart(monkeypatch, outcome, expected):
    def _run(cmd, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(ray_runtime.subprocess, "run", _run)

    assert ray_runtime.ray_probe() is expected


def test_routing_probes_the_driver_connect_of_a_head_that_timed_out(monkeypatch):
    ray_runtime.forget_ray_head_verdict()
    monkeypatch.setattr(ray_runtime, "ray_probe", lambda timeout=None: ray_runtime.RayProbe.UNKNOWN)
    probed: list[str] = []

    def _probe_and_report(address):
        probed.append(address)
        return ray_runtime.RayConnectProbe(True, "connected")

    monkeypatch.setattr(ray_runtime, "_probe_and_report", _probe_and_report)

    try:
        assert ray_runtime.ray_head_answers_driver() is True
    finally:
        ray_runtime.forget_ray_head_verdict()

    assert probed
