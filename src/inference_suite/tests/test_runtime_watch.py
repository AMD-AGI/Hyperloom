# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the round's stop gates, its phase marks, ``server.pid`` and the reap of a child's whole tree."""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hyperloom_common.proctree import running
from inference_suite.contracts import Deadlines, FailedStep, RunLayout
from inference_suite.frameworks import Framework
from inference_suite.runtime import watch
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import PidfileError, ServerPidfile, spawn
from inference_suite.runtime.watch import Exited, RoundGate, ServerWatch, Stop, wait_child

_DEADLINES = Deadlines(boot_sec=100, round_sec=1000, eval_sec=60, silence_sec=50, pre_ready_silence_sec=30)
_BUILD_LINE = "[aiter] [pid=7 pname=x] start build [module_cktile_gemm] under /tmp/jit\n"


def _gate(cancel=lambda: "", deadlines: Deadlines = _DEADLINES) -> RoundGate:
    return RoundGate(deadlines, cancel, started=0.0)


def test_round_gate_checks_cancel_then_absolute_deadline_then_budget():
    assert _gate().check(10.0) is None
    assert _gate(lambda: "operator").check(10.0) == Stop(FailedStep.CANCELLED, "operator")
    past = Deadlines(boot_sec=1, round_sec=1000, eval_sec=1, silence_sec=1, deadline_unix=int(time.time()) - 1)
    assert _gate(deadlines=past).check(10.0).step is FailedStep.DEADLINE_EXHAUSTED
    assert _gate().check(1000.0).step is FailedStep.ROUND_TIMEOUT


def _watch(tmp_path: Path, text: str = "") -> tuple[ServerWatch, Path]:
    log = tmp_path / "server.log"
    log.write_text(text, encoding="utf-8")
    return ServerWatch(log, 0, _DEADLINES, now=0.0), log


def test_an_aiter_build_is_activity_and_does_not_count_against_the_boot(tmp_path):
    server_watch, log = _watch(tmp_path, "loading\n" + _BUILD_LINE)
    server_watch.advance(1.0)
    # 200s quiet inside a cktile build: neither silent nor over the 100s boot budget.
    assert server_watch.before_ready(201.0, None) is None
    with log.open("a") as fh:
        fh.write("build done\n")
    server_watch.advance(201.0)
    assert server_watch.before_ready(230.0, None) is None
    with log.open("a") as fh:
        fh.write("capturing graphs\n")
    server_watch.advance(280.0)
    assert server_watch.before_ready(299.0, None) is None
    assert server_watch.before_ready(300.0, None).step is FailedStep.SERVER_BOOT_TIMEOUT


def test_pre_ready_silence_and_exit_end_the_boot(tmp_path):
    server_watch, _ = _watch(tmp_path, "loading\n")
    server_watch.advance(0.0)
    assert server_watch.before_ready(31.0, None).step is FailedStep.SERVER_SILENT_BEFORE_READY
    assert server_watch.before_ready(1.0, 1).step is FailedStep.SERVER_EXITED_BEFORE_READY


def test_live_server_gates(tmp_path):
    server_watch, log = _watch(tmp_path, "Application startup complete\n")
    server_watch.advance(0.0)
    assert server_watch.ready_logged
    server_watch.mark_ready(10.0)
    assert server_watch.stalled(59.0) is None
    server_watch.note_output(40.0)
    assert server_watch.stalled(89.0) is None
    assert server_watch.stalled(91.0).step is FailedStep.SERVER_STALLED
    with log.open("a") as fh:
        fh.write("RuntimeError: EngineCore encountered a fatal error\n")
    server_watch.advance(92.0)
    died = server_watch.died()
    assert died.step is FailedStep.SERVER_DIED and "EngineCore encountered a fatal error" in died.detail


_TREE = textwrap.dedent(
    """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
    open(sys.argv[1], "w").write(str(child.pid))
    time.sleep(600)
    """
)


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_cancel_reaps_the_child_and_what_it_detached(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "POLL_SEC", 0.05)
    script, grandchild_file = tmp_path / "tree.py", tmp_path / "grandchild.pid"
    script.write_text(_TREE, encoding="utf-8")
    child = spawn((sys.executable, str(script), str(grandchild_file)), {}, tmp_path / "client" / "stdout.log")
    gate = RoundGate(_DEADLINES, lambda: "stop" if grandchild_file.exists() else "", time.monotonic())

    assert wait_child(child, gate) == Stop(FailedStep.CANCELLED, "stop")

    assert child.proc.returncode is not None
    assert _wait_gone(int(grandchild_file.read_text()))


def test_a_child_exit_is_its_returncode_and_its_leftovers_are_reaped(tmp_path):
    script, grandchild_file = tmp_path / "tree.py", tmp_path / "grandchild.pid"
    script.write_text(_TREE.replace("time.sleep(600)\n", "", 1).rstrip() + "\nraise SystemExit(4)\n", encoding="utf-8")
    child = spawn((sys.executable, str(script), str(grandchild_file)), {}, tmp_path / "out.log", tmp_path / "err.log")
    assert wait_child(child, RoundGate(_DEADLINES, lambda: "", time.monotonic())) == Exited(4)
    assert _wait_gone(int(grandchild_file.read_text()))


def test_live_check_stops_the_wait(tmp_path):
    child = spawn((sys.executable, "-c", "import time; time.sleep(600)"), {}, tmp_path / "out.log")
    died = Stop(FailedStep.SERVER_DIED, "gone")
    assert wait_child(child, RoundGate(_DEADLINES, lambda: "", time.monotonic()), lambda now: died) == died
    assert child.proc.returncode is not None


def test_phase_clock_keeps_each_first_mark_tells_listeners_and_writes_the_marks(tmp_path):
    heard: list[tuple[Phase, float]] = []
    clock = PhaseClock([lambda phase, mono: heard.append((phase, mono))])
    clock.mark(Phase.SPAWN, 1.0)
    clock.mark(Phase.READY, 2.0)
    clock.mark(Phase.READY, 3.0)
    layout = RunLayout(tmp_path)
    clock.write(layout)

    assert heard == [(Phase.SPAWN, 1.0), (Phase.READY, 2.0)]
    marks = json.loads(layout.phase_marks.read_text())["phase_marks"]
    assert [(mark["phase"], mark["mono"]) for mark in marks] == [("spawn", 1.0), ("ready", 2.0)]
    assert all(set(mark) == {"phase", "ts", "mono"} for mark in marks)


_RECORD = ServerPidfile(
    pid=10, pgid=10, start_time=5, run_mark="m", port=30000, framework=Framework.VLLM, launch_digest="sha256:x"
)


def test_pidfile_round_trips_and_refuses_what_it_did_not_write(tmp_path):
    path = tmp_path / "server.pid"
    _RECORD.write(path)
    assert ServerPidfile.read(path) == _RECORD
    assert set(json.loads(path.read_text())) == {
        "pid",
        "pgid",
        "start_time",
        "run_mark",
        "port",
        "framework",
        "launch_digest",
    }
    assert ServerPidfile.read(tmp_path / "absent.pid") is None
    path.write_text(json.dumps({**_RECORD.to_document(), "cmdline": "sglang"}))
    with pytest.raises(PidfileError, match="unknown key"):
        ServerPidfile.read(path)
    path.write_text("1234 1234\n")
    with pytest.raises(PidfileError, match="not JSON"):
        ServerPidfile.read(path)
