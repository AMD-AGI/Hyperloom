# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The driver connect probe: a Ray head that does not answer a driver is found in a child process, not in-process.

In-process, Ray 2.44's driver registration holds the GIL while it waits for the raylet, so a head that never answers
freezes the whole process with no timeout able to fire. Every case here runs the real probe, a child interpreter
importing the stand-in ``ray`` package of :class:`FakeRayHead`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _ray_backend as rb
from hyperloom.orchestrator.actions.executors import _ray_runtime as rr
from hyperloom.orchestrator.actions.executors import _ray_serving as rs

from .conftest import FakeRayHead

_PROBE_BOUND_SEC = 2.0
# What a SIGKILL'd group and the reaping of its leader may take on a loaded host, on top of the bound.
_KILL_SLACK_SEC = 10.0


def _running(pid: int) -> bool:
    """Whether ``pid`` is a live process; a zombie is not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == rr.__name__ and r.levelno == logging.WARNING]


@pytest.fixture
def silent_head(fake_ray_head: FakeRayHead, monkeypatch: pytest.MonkeyPatch) -> FakeRayHead:
    """A head that accepts a driver and never registers it, probed under a short bound."""
    fake_ray_head.answers("silent")
    monkeypatch.setenv("HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC", str(_PROBE_BOUND_SEC))
    return fake_ray_head


def test_a_silent_head_sends_the_work_down_the_local_path_within_the_bound(
    silent_head: FakeRayHead, caplog: pytest.LogCaptureFixture
):
    """The first routing question returns after the bound, and the answer is the one RAY_EXEC=0 gives."""
    caplog.set_level(logging.INFO, logger=rr.__name__)
    started = time.monotonic()

    enabled = rb.ray_exec_enabled()

    elapsed = time.monotonic() - started
    assert enabled is False
    assert _PROBE_BOUND_SEC <= elapsed < _PROBE_BOUND_SEC + _KILL_SLACK_SEC
    assert rs.maybe_serving_lease(num_gpus=1) is None
    assert rs.maybe_gpu_specialist_lease(num_gpus=1) is None
    assert rb.serving_slot_busy() is False
    [warning] = _warnings(caplog)
    assert f"did not connect within {_PROBE_BOUND_SEC:g}s and was killed" in warning
    assert "HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC" in warning
    assert "ray stop --force" in warning and "INFERENCE_OPTIMIZER_RAY_EXEC=0" in warning
    assert os.getpid() not in silent_head.connects()


def test_a_silent_probe_leaves_no_process_behind(silent_head: FakeRayHead):
    """The probe's whole process group goes, including what its driver started before it hung."""
    probe = rr.probe_ray_connect("auto", _PROBE_BOUND_SEC)

    assert probe.answered is False
    pids = dict(silent_head.calls())
    assert set(pids) == {"init", "daemon"}, "the stand-in driver did not get as far as starting its process"
    for kind, pid in pids.items():
        assert _gone_within(pid, _KILL_SLACK_SEC), f"the probe's {kind} (pid {pid}) outlived it"


def test_the_verdict_is_reached_once_per_process(silent_head: FakeRayHead, caplog: pytest.LogCaptureFixture):
    """Later rounds ask again and again; only the first one pays for the probe, and it warns once."""
    caplog.set_level(logging.INFO, logger=rr.__name__)

    answers = [rb.ray_exec_enabled() for _ in range(3)]
    started = time.monotonic()
    rs.maybe_serving_lease(num_gpus=1)

    assert answers == [False, False, False]
    assert time.monotonic() - started < 1.0
    assert len(silent_head.connects()) == 1
    assert len(_warnings(caplog)) == 1


def test_a_head_that_refuses_the_driver_is_not_connected_to(
    fake_ray_head: FakeRayHead, caplog: pytest.LogCaptureFixture
):
    """A probe that exits non-zero is the same verdict as one that hangs, and its error reaches the log."""
    fake_ray_head.answers("raises")
    caplog.set_level(logging.INFO, logger=rr.__name__)

    assert rb.ray_exec_enabled() is False

    [warning] = _warnings(caplog)
    assert "exited 1" in warning and "the raylet refused the driver" in warning
    assert rs.maybe_serving_lease(num_gpus=1) is None


def test_a_healthy_head_is_connected_to_in_process_as_before(
    fake_ray_head: FakeRayHead, caplog: pytest.LogCaptureFixture
):
    """The probe connects and disconnects once in its child; the lease then connects this process, unprobed."""
    caplog.set_level(logging.INFO, logger=rr.__name__)
    fake_ray_head.import_in_process()

    assert rb.ray_exec_enabled() is True
    lease = rs.maybe_serving_lease(num_gpus=1)
    assert isinstance(lease, rs.ServingLease)
    rb.RayExecutionBackend().ensure()

    [probe_pid, in_process_pid] = fake_ray_head.connects()
    assert probe_pid != os.getpid() and in_process_pid == os.getpid()
    assert ("shutdown", probe_pid) in fake_ray_head.calls()
    assert _warnings(caplog) == []


def test_a_version_mismatch_is_left_to_the_in_process_repair(fake_ray_head: FakeRayHead):
    """A foreign head answers the driver with a mismatch, which ``quiet_ray_init`` repairs by replacing the head."""
    fake_ray_head.answers("mismatch")

    assert rb.ray_exec_enabled() is True


class TestAHeadThisProcessStarts:
    """With no head running when work is routed, the probe runs against the one started for the connect."""

    @pytest.fixture(autouse=True)
    def _no_head_yet(self, fake_ray_head: FakeRayHead, monkeypatch: pytest.MonkeyPatch) -> None:
        status = iter([False])
        monkeypatch.setattr(rr, "ray_status_ok", lambda: next(status, True))
        monkeypatch.setattr(rr, "ensure_ray_cluster", lambda num_gpus=None: None)

    def test_routing_does_not_probe_a_head_that_is_not_there(self, fake_ray_head: FakeRayHead):
        assert [rb.ray_exec_enabled() for _ in range(2)] == [True, True]
        assert fake_ray_head.connects() == []

    def test_a_silent_head_is_found_before_the_connect_and_not_connected_to(
        self, fake_ray_head: FakeRayHead, monkeypatch: pytest.MonkeyPatch
    ):
        fake_ray_head.answers("silent")
        monkeypatch.setenv("HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC", str(_PROBE_BOUND_SEC))
        fake_ray_head.import_in_process()
        assert rb.ray_exec_enabled() is True

        with pytest.raises(rr.RayHeadUnresponsiveError, match="did not connect within"):
            rb.RayExecutionBackend().ensure()

        assert os.getpid() not in fake_ray_head.connects()
        assert rb.ray_exec_enabled() is False

    def test_a_healthy_head_is_probed_then_connected_to(self, fake_ray_head: FakeRayHead):
        fake_ray_head.import_in_process()
        assert rb.ray_exec_enabled() is True

        rb.RayExecutionBackend().ensure()

        [probe_pid, in_process_pid] = fake_ray_head.connects()
        assert probe_pid != os.getpid() and in_process_pid == os.getpid()
