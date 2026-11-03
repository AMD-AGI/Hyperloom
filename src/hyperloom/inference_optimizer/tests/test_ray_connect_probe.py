# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The driver connect probe: a Ray head that does not answer a driver is found in a child process, not in-process.

In-process, Ray 2.44's driver registration holds the GIL while it waits for the raylet, so a head that never answers
freezes the whole process with no timeout able to fire. Every case here runs the real probe, a child interpreter
importing the stand-in ``ray`` package of :class:`FakeRayHead`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
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
# How long a ``gilhold`` connect freezes the process it runs in, and the watchdog gap that reads as such a freeze.
_GIL_HOLD_SEC = 10
_FROZEN_SEC = 5.0


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


def _group_members(pgid: int) -> list[int]:
    """The live processes, zombies aside, whose process group is ``pgid``."""
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            fields = Path(f"/proc/{entry}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            members.append(int(entry))
    return members


class _Watchdog:
    """A thread that needs the GIL every 20 ms: the longest gap between its ticks is how long the process froze."""

    def __init__(self) -> None:
        self.longest_gap = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._tick, name="freeze-watchdog", daemon=True)

    def _tick(self) -> None:
        last = time.monotonic()
        while True:
            stopped = self._stop.wait(0.02)
            now = time.monotonic()
            self.longest_gap = max(self.longest_gap, now - last)
            last = now
            if stopped:
                return

    def __enter__(self) -> "_Watchdog":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()


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


@pytest.fixture
def wedgeable_head(fake_ray_head: FakeRayHead, monkeypatch: pytest.MonkeyPatch) -> FakeRayHead:
    """A head this process's backend probes and connects to; ``answers("gilhold")`` then wedges it.

    A wedged head holds a driver's ``ray.init`` with the GIL for ``_GIL_HOLD_SEC``, so a connect made in-process
    freezes this process for that long instead of forever.
    """
    monkeypatch.setenv("HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC", str(_PROBE_BOUND_SEC))
    monkeypatch.setenv("FAKE_RAY_GIL_HOLD_SEC", str(_GIL_HOLD_SEC))
    monkeypatch.setattr(rr, "ensure_fd_limit", lambda: (0, 0))
    monkeypatch.setattr(rb, "_BACKEND", rb.RayExecutionBackend())
    fake_ray_head.import_in_process()
    return fake_ray_head


class TestAReconnectProbesTheHeadAgain:
    """A head that answered once can stop answering; every in-process reconnect after that is probed first."""

    @pytest.fixture(autouse=True)
    def _connected(self, wedgeable_head: FakeRayHead, caplog: pytest.LogCaptureFixture) -> None:
        assert rb.ray_exec_enabled() is True
        rb.get_ray_backend().ensure()
        assert wedgeable_head.connects()[-1] == os.getpid()
        caplog.set_level(logging.INFO, logger=rr.__name__)

    @staticmethod
    def _assert_fell_back(head: FakeRayHead, caplog: pytest.LogCaptureFixture, watchdog: _Watchdog) -> None:
        assert watchdog.longest_gap < _FROZEN_SEC, f"the process froze for {watchdog.longest_gap:.1f}s"
        [_probe, in_process, reprobe] = head.connects()
        assert in_process == os.getpid() and reprobe != os.getpid()
        [warning] = _warnings(caplog)
        assert f"did not connect within {_PROBE_BOUND_SEC:g}s and was killed" in warning
        assert rb.ray_exec_enabled() is False
        assert rs.maybe_serving_lease(num_gpus=1) is None
        assert len(head.connects()) == 3, "the verdict on the wedged head is kept"

    def test_after_the_backend_is_marked_unhealthy(self, wedgeable_head: FakeRayHead, caplog):
        """An actor died (its raylet reaped after an OOM, say) and the head wedged; the lease's next ensure()."""
        wedgeable_head.answers("gilhold")
        rb.mark_ray_backend_unhealthy()

        with _Watchdog() as watchdog, pytest.raises(rr.RayHeadUnresponsiveError, match="did not connect within"):
            rb.get_ray_backend().ensure()

        self._assert_fell_back(wedgeable_head, caplog, watchdog)

    def test_after_a_local_head_is_restarted(self, wedgeable_head: FakeRayHead, caplog):
        """The serving_slot repair replaces the head, and the one it starts never answers."""
        wedgeable_head.answers("gilhold")

        with _Watchdog() as watchdog, pytest.raises(rr.RayHeadUnresponsiveError, match="did not connect within"):
            rb.get_ray_backend().restart_local_head()

        assert ["stop --force"] == [c for c in wedgeable_head.cli_calls() if c.startswith("stop")]
        assert [c for c in wedgeable_head.cli_calls() if c.startswith("start --head")]
        self._assert_fell_back(wedgeable_head, caplog, watchdog)


def test_the_head_that_replaces_a_mismatched_one_is_probed_before_the_retry(
    wedgeable_head: FakeRayHead, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """The probe counts a version mismatch as answered; the head started to repair it is probed before the retry."""
    caplog.set_level(logging.INFO, logger=rr.__name__)
    wedgeable_head.answers("mismatch")
    restart = rr.force_restart_local_cluster

    def restart_onto_a_wedged_head(num_gpus=None):
        restart(num_gpus=num_gpus)
        wedgeable_head.answers("gilhold")

    monkeypatch.setattr(rr, "force_restart_local_cluster", restart_onto_a_wedged_head)
    assert rb.ray_exec_enabled() is True

    with _Watchdog() as watchdog, pytest.raises(rr.RayHeadUnresponsiveError, match="did not connect within"):
        rb.get_ray_backend().ensure()

    assert watchdog.longest_gap < _FROZEN_SEC, f"the process froze for {watchdog.longest_gap:.1f}s"
    [_probe, mismatched, reprobe] = wedgeable_head.connects()
    assert mismatched == os.getpid() and reprobe != os.getpid()
    assert [c for c in wedgeable_head.cli_calls() if c.startswith("start --head")]
    [warning] = _warnings(caplog)
    assert "did not connect within" in warning
    assert rb.ray_exec_enabled() is False


#: A process that probes a head with no bound of its own, and turns SIGTERM into ``SystemExit`` as a handler would.
_PROBING_PARENT = """
import signal, sys
from hyperloom.orchestrator.actions.executors import _ray_runtime as rr
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
rr.probe_ray_connect("auto", 3600.0)
"""


def _start_probing_parent(head: FakeRayHead) -> tuple[subprocess.Popen, dict[str, int]]:
    """Start :data:`_PROBING_PARENT` against ``head``; return it once its probe's driver is waiting on the head."""
    src_root = Path(rr.__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.environ["PYTHONPATH"], str(src_root)]))
    parent = subprocess.Popen(
        [sys.executable, "-c", _PROBING_PARENT],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 60
    while "daemon" not in dict(head.calls()):
        if parent.poll() is not None or time.monotonic() > deadline:
            parent.kill()
            parent.wait()
            pytest.fail(f"the probe never reached the head (parent exit {parent.returncode}, calls {head.calls()})")
        time.sleep(0.05)
    return parent, dict(head.calls())


@pytest.mark.parametrize(
    "signum", [signal.SIGINT, signal.SIGTERM], ids=["KeyboardInterrupt", "SystemExit-from-a-SIGTERM-handler"]
)
def test_a_probe_interrupted_mid_wait_takes_its_whole_group_with_it(fake_ray_head: FakeRayHead, signum: int):
    """However the wait ends, the probe's driver and what it started are killed and nothing of its group remains."""
    fake_ray_head.answers("gilhold")
    parent, pids = _start_probing_parent(fake_ray_head)

    os.kill(parent.pid, signum)

    assert parent.wait(timeout=_KILL_SLACK_SEC) != 0
    for kind, pid in pids.items():
        assert _gone_within(pid, _KILL_SLACK_SEC), f"the probe's {kind} (pid {pid}) outlived the interrupted wait"
    assert _group_members(pids["init"]) == []


def test_a_probe_whose_parent_is_killed_outright_dies_with_it(fake_ray_head: FakeRayHead):
    """A parent that cannot unwind (SIGKILL, or a SIGTERM it does not handle) still leaves no driver on the head."""
    fake_ray_head.answers("gilhold")
    parent, pids = _start_probing_parent(fake_ray_head)

    parent.kill()
    parent.wait()

    try:
        assert _gone_within(pids["init"], _KILL_SLACK_SEC), "the probe's driver outlived the process that started it"
    finally:
        # Only on failure is there anything here to kill: the probe, recognised by its source.
        with contextlib.suppress(OSError):
            if b"ray.init(address=sys.argv[1]" in Path(f"/proc/{pids['init']}/cmdline").read_bytes():
                os.kill(pids["init"], signal.SIGKILL)
