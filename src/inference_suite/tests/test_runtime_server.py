# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the round's server lifecycle: own, boot and reuse, the reuse ownership checks, the ``server.log`` range
a boot or reuse round copies, cancel reaping the whole tree, and what ``stop`` reaps and leaves behind."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_common.proctree import proc_identity, running
from inference_suite.contracts import (
    Deadlines,
    FailedStep,
    Lifecycle,
    LifecycleMode,
    LogRange,
    ProfileConfig,
    RunLayout,
    RunPaths,
    SuiteConfig,
)
from inference_suite.frameworks import Framework
from inference_suite.runtime import process, server, watch
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import Reaped, ServerPidfile
from inference_suite.runtime.server import Server, ServerLaunch, ServerLifecycle, stop
from inference_suite.runtime.watch import Stop
from inference_suite.tests.test_config_contract import _config

pytestmark = pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="needs Linux procfs")

_DIGEST = "sha256:" + "a" * 64
_READY_LINE = "Application startup complete"
_FAKE_SERVER = textwrap.dedent(
    """
    import http.server, os, subprocess, sys, time
    port, behaviour, worker_file = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(worker_file, "w") as fh:
        fh.write(str(worker.pid))
    print("loading weights", flush=True)
    if behaviour == "exit":
        print("RuntimeError: Engine core initialization failed", flush=True)
        sys.exit(3)
    if behaviour == "silent":
        time.sleep(600)

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", port), Health)
    print("%s", flush=True)
    httpd.serve_forever()
    """
    % _READY_LINE
)


@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    monkeypatch.setattr(watch, "POLL_SEC", 0.05)
    monkeypatch.setattr(server, "POLL_SEC", 0.05)


@pytest.fixture(autouse=True)
def _free_gpus(monkeypatch):
    monkeypatch.setattr(server, "unavailable_gpus", lambda gpus: None)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Rig:
    """One fake framework server's launch, port and the files a test reads back."""

    def __init__(self, tmp_path: Path, behaviour: str = "serve") -> None:
        self.tmp = tmp_path
        self.port = _free_port()
        script = tmp_path / "fake_server.py"
        script.write_text(_FAKE_SERVER, encoding="utf-8")
        self.worker_file = tmp_path / f"worker-{behaviour}.pid"
        self.launch = ServerLaunch(
            argv=(sys.executable, str(script), str(self.port), behaviour, str(self.worker_file)),
            env={"PATH": os.environ.get("PATH", "")},
            framework=Framework.SGLANG,
            digest=_DIGEST,
        )

    def config(self, run: str, mode: LifecycleMode = LifecycleMode.OWN, **deadlines: float) -> SuiteConfig:
        base = _config()
        limits = {"boot_sec": 60, "round_sec": 120, "eval_sec": 60, "silence_sec": 60, **deadlines}
        return replace(
            base,
            server=replace(base.server, port=self.port, env_inherit=()),
            lifecycle=Lifecycle(mode, None if mode is LifecycleMode.OWN else str(self.pid_dir)),
            deadlines=Deadlines(**limits),
            paths=RunPaths(str(self.tmp / run), base.paths.tool_cache, base.paths.tool_manifest_sha256),
            profile=ProfileConfig(enabled=False),
        )

    @property
    def pid_dir(self) -> Path:
        return self.tmp / "pids"

    def worker_pid(self) -> int:
        return int(self.worker_file.read_text(encoding="utf-8"))

    def lifecycle(self, config: SuiteConfig, cancel=lambda: "", launch: ServerLaunch | None = None) -> ServerLifecycle:
        return ServerLifecycle(config, launch or self.launch, PhaseClock(), cancel)


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_own_round_tears_its_server_down_and_deletes_its_pidfile(tmp_path):
    rig = _Rig(tmp_path)
    clock = PhaseClock()
    lifecycle = ServerLifecycle(rig.config("r1"), rig.launch, clock, lambda: "")
    acquired = lifecycle.acquire()
    assert isinstance(acquired, Server) and not acquired.reused
    layout = RunLayout(tmp_path / "r1")
    record = ServerPidfile.read(layout.server_pid)
    assert record is not None and record.port == rig.port and record.launch_digest == _DIGEST
    assert record.pgid == record.pid and record.leader_alive
    assert json.loads(layout.server_pid.read_text())["framework"] == "sglang"
    assert layout.read_ready_stamp() is not None
    assert acquired.check(time.monotonic()) is None

    released = lifecycle.release()

    assert released == server.Released(None, None)
    assert not layout.server_pid.exists()
    assert _gone(record.pid) and _gone(rig.worker_pid())
    assert _READY_LINE in layout.server_log.read_text()
    assert [mark.phase for mark in clock.marks] == [Phase.SPAWN, Phase.READY, Phase.TEARDOWN]


def test_the_server_starts_in_the_run_dir_not_the_callers_directory(tmp_path, monkeypatch):
    # ``python -m`` puts the current directory first on the import path; the caller's may hold a framework checkout.
    rig = _Rig(tmp_path)
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    lifecycle = rig.lifecycle(rig.config("r1"))
    assert isinstance(lifecycle.acquire(), Server)
    try:
        record = ServerPidfile.read(RunLayout(tmp_path / "r1").server_pid)
        assert record is not None
        assert os.readlink(f"/proc/{record.pid}/cwd") == str(tmp_path / "r1")
    finally:
        lifecycle.release()


def test_boot_then_reuse_rounds_each_copy_their_range_of_the_live_log(tmp_path):
    rig = _Rig(tmp_path)
    boot = rig.lifecycle(rig.config("boot", LifecycleMode.BOOT))
    assert isinstance(boot.acquire(), Server)
    booted = boot.release()
    live_log = rig.pid_dir / "server.log"
    boot_end = live_log.stat().st_size
    assert booted == server.Released(LogRange(0, boot_end), None)
    assert (tmp_path / "boot" / "server.log").read_bytes() == live_log.read_bytes()
    record = ServerPidfile.read(rig.pid_dir / "server.pid")
    assert record is not None and record.leader_alive

    reuse = rig.lifecycle(rig.config("reuse", LifecycleMode.REUSE))
    acquired = reuse.acquire()
    assert isinstance(acquired, Server) and acquired.reused
    with live_log.open("a") as log:
        log.write("Decode batch. gen throughput (token/s): 100.0\n")
    reused = reuse.release()

    assert reused == server.Released(LogRange(boot_end, live_log.stat().st_size), None)
    assert (tmp_path / "reuse" / "server.log").read_text() == "Decode batch. gen throughput (token/s): 100.0\n"
    assert RunLayout(tmp_path / "reuse").read_ready_stamp().boot_sec == 0
    assert record.leader_alive

    outcome = stop(rig.pid_dir)
    assert record.pid in outcome.reaped and rig.worker_pid() in outcome.reaped and not outcome.survivors
    assert not (rig.pid_dir / "server.pid").exists()


@pytest.fixture
def booted(tmp_path):
    rig = _Rig(tmp_path)
    boot = rig.lifecycle(rig.config("boot", LifecycleMode.BOOT))
    assert isinstance(boot.acquire(), Server)
    boot.release()
    yield rig
    stop(rig.pid_dir)


def _rewrite_pidfile(rig: _Rig, **changes) -> None:
    path = rig.pid_dir / "server.pid"
    record = ServerPidfile.read(path)
    replace(record, **changes).write(path)


def _reuse_refusal(rig: _Rig, run: str, launch: ServerLaunch | None = None) -> Stop:
    lifecycle = rig.lifecycle(rig.config(run, LifecycleMode.REUSE), launch=launch)
    refused = lifecycle.acquire()
    assert isinstance(refused, Stop) and refused.step is FailedStep.REUSE_NOT_OWNED
    released = lifecycle.release()
    assert released.log_range is not None and released.log_range.start == released.log_range.end
    assert (rig.tmp / run / "server.log").read_bytes() == b""
    return refused


def test_reuse_needs_the_pidfile(booted):
    (booted.pid_dir / "server.pid").rename(booted.pid_dir / "elsewhere.pid")
    assert "records no server" in _reuse_refusal(booted, "r").detail
    (booted.pid_dir / "elsewhere.pid").rename(booted.pid_dir / "server.pid")


def test_reuse_needs_the_recorded_incarnation(booted):
    record = ServerPidfile.read(booted.pid_dir / "server.pid")
    _rewrite_pidfile(booted, start_time=record.start_time + 1)
    assert "no longer the process" in _reuse_refusal(booted, "r").detail
    _rewrite_pidfile(booted, start_time=record.start_time)


def test_reuse_needs_the_listener_in_the_recorded_group(booted, tmp_path):
    record = ServerPidfile.read(booted.pid_dir / "server.pid")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
    try:
        start_time = proc_identity(other.pid)[1]
        _rewrite_pidfile(booted, pid=other.pid, pgid=other.pid, start_time=start_time)
        assert "does not hold the listener" in _reuse_refusal(booted, "r").detail
    finally:
        other.kill()
        other.wait()
        replace(record).write(booted.pid_dir / "server.pid")


def test_reuse_needs_the_same_launch(booted):
    other = replace(booted.launch, digest="sha256:" + "b" * 64)
    assert "this round's is" in _reuse_refusal(booted, "r", launch=other).detail
    assert isinstance(booted.lifecycle(booted.config("ok", LifecycleMode.REUSE)).acquire(), Server)


def test_cancel_during_boot_reaps_the_whole_tree(tmp_path):
    rig = _Rig(tmp_path, "silent")
    lifecycle = rig.lifecycle(rig.config("r1"), cancel=lambda: "operator stop" if rig.worker_file.exists() else "")

    stopped = lifecycle.acquire()

    assert stopped == Stop(FailedStep.CANCELLED, "operator stop")
    assert _gone(rig.worker_pid())
    assert not (tmp_path / "r1" / "server.pid").exists()
    assert lifecycle.release() == server.Released(None, None)


def test_a_server_that_exits_before_ready_names_why(tmp_path):
    rig = _Rig(tmp_path, "exit")
    stopped = rig.lifecycle(rig.config("r1")).acquire()
    assert isinstance(stopped, Stop) and stopped.step is FailedStep.SERVER_EXITED_BEFORE_READY
    assert "Engine core initialization failed" in stopped.detail
    assert _gone(rig.worker_pid())


def test_a_taken_port_is_refused_before_spawning(tmp_path):
    rig = _Rig(tmp_path)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", rig.port))
        sock.listen()
        stopped = rig.lifecycle(rig.config("r1")).acquire()
    assert isinstance(stopped, Stop) and stopped.step is FailedStep.PORT_IN_USE
    assert not rig.worker_file.exists()


@pytest.mark.parametrize("mode", [LifecycleMode.OWN, LifecycleMode.BOOT])
def test_gpus_that_are_not_free_are_refused_before_spawning(tmp_path, monkeypatch, mode):
    rig = _Rig(tmp_path)
    asked: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        server, "unavailable_gpus", lambda gpus: asked.append(gpus) or "GPU 0 (amd-smi card 0): held by pid 9 (vllm)"
    )
    config = rig.config("r1", mode)

    stopped = rig.lifecycle(config).acquire()

    assert isinstance(stopped, Stop) and stopped.step is FailedStep.GPUS_UNAVAILABLE
    assert stopped.detail == "GPU 0 (amd-smi card 0): held by pid 9 (vllm)"
    assert asked == [config.server.gpus]
    assert not rig.worker_file.exists() and not (rig.pid_dir / "server.pid").exists()


def test_reuse_and_external_rounds_start_no_server_so_check_no_gpus(booted, monkeypatch):
    monkeypatch.setattr(server, "unavailable_gpus", lambda gpus: pytest.fail("checked the GPUs of a found server"))
    external = replace(booted.config("r-external"), lifecycle=Lifecycle(LifecycleMode.EXTERNAL))
    for config in (booted.config("r-reuse", LifecycleMode.REUSE), external):
        lifecycle = booted.lifecycle(config)
        assert isinstance(lifecycle.acquire(), Server)
        lifecycle.release()


def test_stop_reaps_the_workers_by_run_mark_once_the_leader_is_gone(booted):
    record = ServerPidfile.read(booted.pid_dir / "server.pid")
    worker = booted.worker_pid()
    os.kill(record.pid, signal.SIGKILL)
    assert _gone(record.pid) and running(worker)

    outcome = stop(booted.pid_dir)

    assert worker in outcome.reaped and record.pid not in outcome.reaped
    assert _gone(worker) and not (booted.pid_dir / "server.pid").exists()


def test_stop_keeps_the_pidfile_while_a_member_outlives_sigkill(booted, monkeypatch):
    real_reap = server.reap
    monkeypatch.setattr(server, "reap", lambda roots, mark: Reaped(real_reap(roots, mark).reaped, (4242,)))
    assert stop(booted.pid_dir).survivors == (4242,)
    assert (booted.pid_dir / "server.pid").exists()


def test_own_teardown_with_a_survivor_keeps_the_pidfile_and_says_so(tmp_path, monkeypatch):
    rig = _Rig(tmp_path)
    lifecycle = rig.lifecycle(rig.config("r1"))
    assert isinstance(lifecycle.acquire(), Server)
    real_reap = process.reap
    monkeypatch.setattr(process, "reap", lambda roots, mark: Reaped(real_reap(roots, mark).reaped, (4242,)))

    released = lifecycle.release()

    assert released.teardown is not None and released.teardown.step is FailedStep.TEARDOWN_INCOMPLETE
    assert (tmp_path / "r1" / "server.pid").exists()
    monkeypatch.undo()
    assert not stop(tmp_path / "r1").survivors


def test_reap_waits_out_a_gpu_rank_that_takes_seconds_to_exit_after_sigkill(tmp_path, monkeypatch):
    from hyperloom_common import proctree

    class _Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock, killed_at = _Clock(), []
    child = process.spawn([sys.executable, "-c", "import time; time.sleep(600)"], dict(os.environ), tmp_path / "log")
    real_signal_tree = proctree.signal_tree

    def signal_tree(tree, sig):
        if sig == signal.SIGKILL:
            killed_at.append(clock.now)
            real_signal_tree(tree, sig)

    def running(pid):
        return not killed_at or clock.now < killed_at[0] + 8.0

    monkeypatch.setattr(proctree, "time", clock)
    monkeypatch.setattr(proctree, "signal_tree", signal_tree)
    monkeypatch.setattr(proctree, "running", running)
    monkeypatch.setattr(process, "running", running)

    outcome = child.reap()

    assert child.proc.pid in outcome.reaped and outcome.survivors == ()


def test_stop_without_a_pidfile_reaps_nothing(tmp_path):
    assert stop(tmp_path) == Reaped((), ())
