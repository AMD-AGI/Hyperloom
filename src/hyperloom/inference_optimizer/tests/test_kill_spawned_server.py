# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``_subprocess_kill.kill_my_spawned_server`` and the BaselineExecutor integration.

Covers the no-op / already-exited cases, the same-session-group refusal guard,
SIGTERM→grace→SIGKILL ordering, and grandchild reaping.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors import _subprocess_kill
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    RoundWatch,
    kill_my_spawned_server,
    run_with_session_kill,
    session_deadline_to_remaining_sec,
    session_remaining_to_deadline_sec,
)
from inference_suite.server_log import death_excerpt


class _FakeCapture:
    """A capture whose child wrote ``out`` and ``err`` and nothing the watchdog could see arrive."""

    last_activity_at = None

    def __init__(self, out: str = "", err: str = "") -> None:
        self._captured = (out, err)

    def note_output(self) -> None:
        pass

    def finish(self) -> tuple[str, str]:
        return self._captured


def test_exited_process_wins_over_expired_gates():
    class Proc:
        def poll(self):
            return 7

    out = _subprocess_kill._communicate_with_watchdog(
        Proc(),
        _subprocess_kill._WatchedRound(RoundWatch(timeout=0, session_deadline_sec=-1)),
        capture=_FakeCapture("finished"),
    )
    assert out == ("finished", "")


def test_capture_partial_utf8_records_bytes_before_newline():
    from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; os.write(1,b'\\xe2'); time.sleep(.4); os.write(1,b'\\x82\\xac'); time.sleep(.4)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    capture = sk._StreamCapture(proc)
    capture.start()
    try:
        deadline = time.monotonic() + 0.35
        while capture.last_activity_at is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert capture.last_activity_at is not None
        assert proc.poll() is None
        proc.wait(timeout=5)
        stdout, stderr = capture.finish()
        assert stdout == "€"
        assert stderr == ""
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


@pytest.mark.parametrize(
    "name", ["INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC", "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC"]
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "bad", ""])
def test_benchmark_timeouts_reject_invalid_overrides(name, value):
    from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk

    with pytest.raises(ValueError, match=name):
        sk.resolve_benchmark_timeouts({name: value})


def test_benchmark_timeouts_have_one_finite_positive_policy():
    from hyperloom.orchestrator.actions.executors import _subprocess_kill as sk

    assert sk.resolve_benchmark_timeouts({}) == (600.0, 7800.0)
    assert sk.resolve_benchmark_timeouts({"INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC": "42.5"}) == (600.0, 42.5)


def test_hard_timeout_preserves_partial_captured_output():
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run_with_session_kill(
            [sys.executable, "-c", "import os,time; os.write(1,b'partial'); os.write(2,b'error'); time.sleep(20)"],
            watch=RoundWatch(timeout=0.4),
        )
    assert error.value.stdout == "partial"
    assert error.value.stderr == "error"


def test_kill_my_spawned_server_handles_none():
    """Plain no-op when given None so callers can use it in ``finally:`` unguarded."""
    kill_my_spawned_server(None)  # must not raise


def test_kill_my_spawned_server_handles_already_exited():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    proc.wait(timeout=10)
    kill_my_spawned_server(proc)


def test_kill_my_spawned_server_refuses_own_session_group(caplog):
    """Defensive guard: the helper must NOT killpg the parent's own session group."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with caplog.at_level("ERROR"):
            kill_my_spawned_server(proc, grace_seconds=0.5)
        assert proc.poll() is None, (
            "helper killed a process in the parent's own session — that would take down the Coordinator in production"
        )
        assert any("refusing to killpg own session" in rec.message for rec in caplog.records), (
            "expected an ERROR log line about same-pgid refusal"
        )
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_kill_my_spawned_server_sigterm_then_sigkill_for_ignorer():
    """A child that traps SIGTERM is still reaped via SIGKILL after the grace window."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            ("import signal, time;\nsignal.signal(signal.SIGTERM, signal.SIG_IGN);\ntime.sleep(60)\n"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Let the child install its SIGTERM handler before we signal.
    time.sleep(0.3)
    start = time.monotonic()
    kill_my_spawned_server(proc, grace_seconds=1.0)
    elapsed = time.monotonic() - start
    assert proc.poll() is not None
    assert elapsed < 5.0, f"kill_my_spawned_server hung for {elapsed:.2f}s"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires Linux process groups")
def test_completed_warmup_keeps_its_persistent_server(tmp_path):
    """The lifecycle owner, not a completed warmup wrapper, decides when to stop the server."""
    pidfile = tmp_path / "server.pid"
    server_code = "import time; time.sleep(60)"
    wrapper_code = (
        "import pathlib, subprocess, sys\n"
        "server = subprocess.Popen([sys.executable, '-c', sys.argv[2]], "
        "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "published = False\n"
        "try:\n"
        "    pidfile = pathlib.Path(sys.argv[1])\n"
        "    pending = pidfile.with_suffix('.tmp')\n"
        "    pending.write_text(str(server.pid))\n"
        "    pending.replace(pidfile)\n"
        "    published = True\n"
        "finally:\n"
        "    if not published:\n"
        "        server.kill()\n"
        "        server.wait()\n"
    )
    server_pid = None
    try:
        result = run_with_session_kill(
            [sys.executable, "-c", wrapper_code, str(pidfile), server_code], watch=RoundWatch(timeout=10)
        )
        assert result.returncode == 0
        server_pid = int(pidfile.read_text())
        assert os.getpgid(server_pid) == server_pid
        os.kill(server_pid, 0)
    finally:
        if server_pid is None:
            try:
                server_pid = int(pidfile.read_text())
            except (OSError, ValueError):
                # Failed publication is cleaned up by the wrapper itself.
                pass
        if server_pid is not None:
            try:
                os.killpg(server_pid, signal.SIGKILL)
            except ProcessLookupError:
                # The test server may have exited before cleanup reached its group.
                pass


def test_kill_my_spawned_server_reaps_grandchildren():
    """A child that spawns a grandchild leaves no surviving descendant after the helper returns."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, sys, time;\n"
                "# Write our pgid + grandchild PID to disk for the test to read.\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    # Grandchild: pretend to be a long-running server.\n"
                "    time.sleep(120)\n"
                "    sys.exit(0)\n"
                "open(sys.argv[1], 'w').write(str(pid))\n"
                "time.sleep(120)\n"
            ),
            "/tmp/hyperloom_test_grandchild.pid",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file = Path("/tmp/hyperloom_test_grandchild.pid")
    try:
        deadline = time.monotonic() + 5.0
        grandchild_pid: int | None = None
        while time.monotonic() < deadline:
            if pid_file.exists():
                txt = pid_file.read_text().strip()
                if txt:
                    grandchild_pid = int(txt)
                    break
            time.sleep(0.05)
        assert grandchild_pid is not None, "parent never wrote grandchild pid"

        os.kill(grandchild_pid, 0)  # raises if gone

        kill_my_spawned_server(proc, grace_seconds=1.5)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
    finally:
        try:
            pid_file.unlink()
        except FileNotFoundError:
            # PID file already gone; nothing to clean up.
            pass
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                # Process already exited; nothing to signal.
                pass


def _make_fake_round_command(
    tmp_path: Path,
    *,
    mode: str,
) -> tuple[Path, Path]:
    """Build a round command stand-in; returns (script_path, sentinel_file)."""
    script = tmp_path / "fake_round.py"
    sentinel = tmp_path / "leaked_grandchild.pid"
    workspace = tmp_path / "out" / "benchmark_fake_20260101_000000"

    if mode == "succeed_then_leak":
        body = f"""
import json, os, pathlib, sys, time
ws = pathlib.Path({str(workspace)!r})
ws.mkdir(parents=True, exist_ok=True)
(ws / "benchmark_report.json").write_text(json.dumps({{
    "output_throughput": 12.3,
    "completed": 42,
}}))
pid = os.fork()
if pid == 0:
    time.sleep(120)
    sys.exit(0)
pathlib.Path({str(sentinel)!r}).write_text(str(pid))
sys.exit(0)
"""
    elif mode == "timeout":
        body = f"""
import os, pathlib, sys, time
pid = os.fork()
if pid == 0:
    time.sleep(120)
    sys.exit(0)
pathlib.Path({str(sentinel)!r}).write_text(str(pid))
time.sleep(120)
"""
    else:
        raise ValueError(mode)

    script.write_text(body)
    return script, sentinel


@pytest.mark.asyncio
async def test_baseline_executor_kills_grandchild_on_timeout(tmp_path, monkeypatch):
    """A leaked grandchild must be dead by the time the executor returns after its timeout fires."""
    script, sentinel = _make_fake_round_command(tmp_path, mode="timeout")
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        grandchild_pid: int | None = None
        while time.monotonic() < deadline:
            if sentinel.exists():
                txt = sentinel.read_text().strip()
                if txt:
                    grandchild_pid = int(txt)
                    break
            time.sleep(0.05)
        assert grandchild_pid is not None
        os.kill(grandchild_pid, 0)

        kill_my_spawned_server(proc, grace_seconds=1.5)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                # Process already exited; nothing to signal.
                pass


class TestSessionDeadline:
    """The session budget is a separate channel from the soft deadline.

    The soft deadline answers "is this variant abnormally slow", which is why it
    retires when the accuracy eval starts. The session budget answers "is the run
    out of time", which no phase boundary changes.
    """

    def test_expired_session_budget_reaps_the_tree_with_its_own_sentinel(self):
        start = time.monotonic()
        cp = run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            watch=RoundWatch(timeout=60, session_deadline_sec=time.monotonic() - 1.0),
        )
        elapsed = time.monotonic() - start
        assert cp.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
        assert elapsed < 10.0, f"session-deadline path took {elapsed:.2f}s"

    def test_a_budget_with_room_left_leaves_the_child_alone(self):
        start = time.monotonic()
        cp = run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            watch=RoundWatch(timeout=60, session_deadline_sec=time.monotonic() + 3600.0),
        )
        elapsed = time.monotonic() - start
        assert cp.returncode == 0
        assert elapsed >= 1.5, f"child was cut short at {elapsed:.2f}s"

    def test_no_session_deadline_keeps_the_previous_behaviour(self):
        cp = run_with_session_kill(
            [sys.executable, "-c", "print('done')"],
            watch=RoundWatch(timeout=30, session_deadline_sec=None),
        )
        assert cp.returncode == 0
        assert "done" in (cp.stdout or "")


class TestAnOrchestratorCancelReachesTheChild:
    """The last defence has to stop the child, not just the coroutine above it.

    The executor blocks in a worker thread, so cancelling its task frees the
    lanes and the GPU lease while the benchmark is still running. The cancel
    scope is the channel the thread checks, at the poll it already runs.
    """

    def test_a_cancel_raised_before_the_call_reaps_the_tree(self):
        scope = CancelScope()
        scope.cancel(reason="shutdown_requested")
        start = time.monotonic()
        with use_cancel_scope(scope):
            cp = run_with_session_kill(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                watch=RoundWatch(timeout=60),
            )
        elapsed = time.monotonic() - start
        assert cp.returncode == ORCHESTRATOR_CANCELLED_RETURNCODE
        assert elapsed < 10.0, f"cancel path took {elapsed:.2f}s"

    def test_a_cancel_that_arrives_mid_run_still_reaches_it(self):
        """The interesting case: nothing is wrong when the child is launched."""
        scope = CancelScope()
        timer = threading.Timer(0.5, lambda: scope.cancel(reason="session_time_exhausted"))
        timer.start()
        start = time.monotonic()
        try:
            with use_cancel_scope(scope):
                cp = run_with_session_kill(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    watch=RoundWatch(timeout=60),
                )
        finally:
            timer.cancel()
        elapsed = time.monotonic() - start
        assert cp.returncode == ORCHESTRATOR_CANCELLED_RETURNCODE
        assert elapsed < 10.0, f"mid-run cancel took {elapsed:.2f}s"

    def test_a_scope_nobody_cancelled_leaves_the_child_alone(self):
        """The channel must cost nothing on the path every healthy round takes."""
        with use_cancel_scope(CancelScope()):
            cp = run_with_session_kill(
                [sys.executable, "-c", "import time; time.sleep(1); print('done')"],
                watch=RoundWatch(timeout=60),
            )
        assert cp.returncode == 0
        assert "done" in (cp.stdout or "")

    def test_a_spent_budget_keeps_its_own_attribution(self):
        """Both are true at once whenever the budget is what triggered the cancel.

        The budget is a fact about the run and the cancel is only the dispatcher
        acting on it, so the ledger gets the cause, not the mechanism.
        """
        scope = CancelScope()
        scope.cancel(reason="session_time_exhausted")
        with use_cancel_scope(scope):
            cp = run_with_session_kill(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                watch=RoundWatch(timeout=60, session_deadline_sec=time.monotonic() - 1.0),
            )
        assert cp.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE

    def test_the_call_registers_as_a_listener_while_the_child_lives(self):
        """The canceller waits only for work that can hear it, so this is load-bearing."""
        scope = CancelScope()
        seen: list[bool] = []
        timer = threading.Timer(
            0.5,
            lambda: (seen.append(scope.has_listeners), scope.cancel(reason="test")),
        )
        timer.start()
        try:
            with use_cancel_scope(scope):
                run_with_session_kill(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    watch=RoundWatch(timeout=60),
                )
        finally:
            timer.cancel()
        assert seen == [True]
        assert not scope.has_listeners


class TestSessionDeadlineCrossesAProcessBoundary:
    """A ``time.monotonic()`` instant is only meaningful in the process that read it.

    Handing the absolute deadline to a Ray worker would name an instant on the
    worker's own clock, whose origin is unrelated -- an immediate kill or one
    that never fires, both silently. Only a duration survives the trip.
    """

    def test_an_unbounded_budget_stays_unbounded_in_both_directions(self):
        assert session_deadline_to_remaining_sec(None) is None
        assert session_remaining_to_deadline_sec(None) is None

    def test_a_deadline_becomes_the_seconds_it_has_left(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        assert session_deadline_to_remaining_sec(1300.0) == pytest.approx(300.0)

    def test_a_spent_budget_crosses_as_a_non_positive_duration(self, monkeypatch):
        """Not floored at zero: the receiver must reap, not read it as "no budget given"."""
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        assert session_deadline_to_remaining_sec(940.0) == pytest.approx(-60.0)

    def test_the_duration_re_anchors_onto_the_reading_clock(self, monkeypatch):
        """The same duration names a different instant on each side; that is the point."""
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        remaining = session_deadline_to_remaining_sec(1300.0)
        monkeypatch.setattr(time, "monotonic", lambda: 5_000_000.0)
        assert session_remaining_to_deadline_sec(remaining) == pytest.approx(5_000_300.0)

    def test_the_round_trip_is_the_identity_within_one_clock(self, monkeypatch):
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
        assert session_remaining_to_deadline_sec(session_deadline_to_remaining_sec(1234.5)) == pytest.approx(1234.5)


def _sentinel_returncodes() -> dict[int, set[str]]:
    """Map every sentinel returncode to the qualified names that claim it."""
    from hyperloom.orchestrator.actions.executors import _ray_serving, _subprocess_kill

    assigned: dict[int, set[str]] = {}
    for module in (_subprocess_kill, _ray_serving):
        short = module.__name__.rsplit(".", 1)[-1]
        for name, value in vars(module).items():
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            if not (name.endswith(("_RETURNCODE", "_RC"))):
                continue
            assigned.setdefault(value, set()).add(f"{short}.{name}")
    return assigned


def test_every_sentinel_returncode_names_exactly_one_cause():
    """A sentinel shared by two causes makes attribution a coin flip.

    The codes are handed out in more than one module and all arrive at their
    consumer as a plain ``returncode``, so a new one can quietly reuse a number
    already taken. That is how the session-budget code first landed on the Ray
    actor-died number, which would have had every actor death read as a spent
    budget and taught the ledger the wrong thing about both.
    """
    assigned = _sentinel_returncodes()

    collisions = {code: sorted(names) for code, names in assigned.items() if len(names) > 1}
    assert not collisions, f"sentinel return codes collide: {collisions}"
    assert SESSION_TIME_EXHAUSTED_RETURNCODE in assigned


def test_run_with_session_kill_streams_child_output_to_parent(capsys):
    """Captured child output is also mirrored immediately to parent streams."""
    code = "import sys\nprint('child-out', flush=True)\nprint('child-err', file=sys.stderr, flush=True)\n"
    cp = run_with_session_kill([sys.executable, "-c", code], watch=RoundWatch(timeout=10))

    captured = capsys.readouterr()
    assert cp.returncode == 0
    assert "child-out" in (cp.stdout or "")
    assert "child-err" in (cp.stderr or "")
    assert "child-out" in captured.out
    assert "child-err" in captured.err


def test_run_with_session_kill_reports_each_line_of_child_output():
    """The liveness callback fires while the child runs, once per line it emits."""
    code = "import sys, time\nfor i in range(3):\n    print(i, flush=True)\n    time.sleep(0.05)\n"
    lines: list[float] = []

    cp = run_with_session_kill(
        [sys.executable, "-c", code],
        on_output=lambda: lines.append(time.monotonic()),
        watch=RoundWatch(timeout=10),
    )

    assert cp.returncode == 0
    assert len(lines) >= 3


def test_run_with_session_kill_legacy_timeout_still_raises():
    """With ``soft_deadline_sec`` None, a child exceeding the hard ``timeout`` still raises ``TimeoutExpired``."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            watch=RoundWatch(timeout=1),
        )


# Server-liveness watchdog


def test_death_excerpt_surfaces_root_cause(tmp_path):
    """The excerpt helper returns the engine/worker-init root-cause line (with a
    little context) for the failure classifier; a healthy / missing log returns ``None``."""
    log_path = tmp_path / "server.log"
    assert death_excerpt(str(log_path)) is None  # missing → None
    log_path.write_text("INFO loading shards 50%\nINFO graph capture\n")
    assert death_excerpt(str(log_path)) is None  # healthy → None
    log_path.write_text(
        "(APIServer pid=16160)   File '.../vllm/v1/engine/utils.py', line 1057, "
        "in wait_for_engine_startup\n"
        "(APIServer pid=16160)     raise RuntimeError(\n"
        "(APIServer pid=16160) RuntimeError: Engine core initialization failed. "
        "See root cause above. Failed core proc(s): {}\n"
    )
    excerpt = death_excerpt(str(log_path))
    assert excerpt is not None
    assert "Engine core initialization failed" in excerpt


def test_death_excerpt_surfaces_config_validation_arch_miss(tmp_path):
    """A config-validation-stage failure (brand-new checkpoint ``model_type``
    unknown to the installed transformers/vLLM) dies BEFORE the engine starts and
    must still be surfaced as a fatal excerpt. Without this the enablement failure
    classifier only sees the round's ``subprocess_nonzero`` stdout tail, classifies
    ``unknown``, and never seeds the ``pip install -U transformers`` bridge —
    starving every enablement round of the real root cause (DeepSeek-V4 repro)."""
    from hyperloom.common.failure_signature import classify_failure

    log_path = tmp_path / "server.log"
    # A healthy INFO banner naming architectures must NOT trip the markers.
    log_path.write_text("INFO [registry] Model architectures ['Qwen3ForCausalLM'] loaded\n")
    assert death_excerpt(str(log_path)) is None
    # The real DeepSeek-V4 config-validation failure.
    log_path.write_text(
        "(APIServer pid=1046234) Traceback (most recent call last):\n"
        "(APIServer pid=1046234) pydantic_core._pydantic_core.ValidationError: "
        "1 validation error for ModelConfig\n"
        "(APIServer pid=1046234)   Value error, The checkpoint you are trying to "
        "load has model type `deepseek_v4` but Transformers does not recognize "
        "this architecture.\n"
    )
    excerpt = death_excerpt(str(log_path))
    assert excerpt is not None
    assert "does not recognize this architecture" in excerpt
    # The extracted excerpt must classify as missing_model_arch (not unknown),
    # which is what seeds the deterministic pip-install enablement bridge.
    sig = classify_failure(excerpt)
    assert sig.kind == "missing_model_arch"
    assert sig.offending_symbol == "deepseek_v4"


def _engine_death_log(tmp_path, *, downstream_errors: int = 4000):
    """A server that served, then died: the cause, then the cascade it caused.

    Mirrors the shape a real run produces -- a multi-megabyte log whose last
    megabyte is one downstream error per rejected request, with the scheduler
    state dumped just above the fatal line.
    """
    log = tmp_path / "server.log"
    lines = [
        "(APIServer pid=1) INFO:     Application startup complete.",
        "(EngineCore pid=2) ERROR [dump_input.py:79] Dumping scheduler output: " + "x=1, " * 6000,
        "(EngineCore pid=2) ERROR [core.py:1138] EngineCore encountered a fatal error.",
        "(EngineCore pid=2) ERROR [core.py:1138] Traceback (most recent call last):",
    ]
    lines += [f'(EngineCore pid=2) ERROR [core.py:1138]   File "/vllm/x{i}.py", line {i}, in step' for i in range(22)]
    lines.append(
        "(EngineCore pid=2) ERROR [core.py:1138] RuntimeError: Worker failed with error "
        "'HIP out of memory. Tried to allocate 10.54 GiB. GPU 0 has a total capacity of 255.98 GiB'"
    )
    lines += [
        "(APIServer pid=1) ERROR [serving.py:448] vllm.v1.engine.exceptions.EngineDeadError: "
        "EngineCore encountered an issue."
    ]
    lines += [
        '(APIServer pid=1) INFO:     127.0.0.1:5 - "POST /v1/completions HTTP/1.1" 500 Internal Server Error'
    ] * downstream_errors
    log.write_text("\n".join(lines), encoding="utf-8")
    return log


def test_a_server_that_dies_after_serving_yields_a_classifiable_excerpt(tmp_path):
    """The defect this closes: no excerpt at all, so the failure read as unknown.

    The fatal markers only named pre-serving bootstrap failures, and the search
    only read the log's tail -- which an engine that dies mid-serving fills with
    one downstream error per rejected request. The cause sits at the head of
    that cascade, so nothing was ever returned and the specialist was handed
    ``failure_kind: unknown``.
    """
    from hyperloom.common.failure_signature import classify_failure

    excerpt = death_excerpt(str(_engine_death_log(tmp_path)))

    assert excerpt is not None
    assert "EngineCore encountered a fatal error" in excerpt
    assert "HIP out of memory" in excerpt
    assert classify_failure(excerpt).kind == "resource_constraint"


def test_the_excerpt_leads_with_the_marker_not_the_scheduler_dump(tmp_path):
    """vLLM dumps its whole scheduler state on the line above the fatal error."""
    excerpt = death_excerpt(str(_engine_death_log(tmp_path)))

    assert excerpt is not None
    assert excerpt.splitlines()[0].endswith("EngineCore encountered a fatal error.")
    assert "Dumping scheduler output" not in excerpt


def test_the_cause_outranks_the_consequence_it_triggered(tmp_path):
    """`EngineDeadError` is what the API server saw; the OOM is why."""
    excerpt = death_excerpt(str(_engine_death_log(tmp_path)))

    assert excerpt is not None
    assert "HIP out of memory" in excerpt
    lines = excerpt.splitlines()
    oom = next(i for i, line in enumerate(lines) if "HIP out of memory" in line)
    dead = next((i for i, line in enumerate(lines) if "EngineDeadError" in line), len(lines))
    assert oom < dead, "the cause must reach the classifier ahead of the consequence"


def test_a_log_with_no_fatal_marker_still_yields_nothing(tmp_path):
    """Unchanged: a healthy log must not produce a death excerpt."""
    log = tmp_path / "server.log"
    log.write_text("(APIServer pid=1) INFO:     Application startup complete.\n" * 50, encoding="utf-8")

    assert death_excerpt(str(log)) is None


def test_a_bootstrap_wrapper_keeps_the_root_cause_above_it(tmp_path):
    """The legacy markers wrap a cause that sits *above* them.

    "Engine core initialization failed ... See root cause above" names nothing
    a classifier can act on; the exception on the preceding line does. Extracting
    only the marker and what follows -- right for a post-startup engine death --
    would drop that cause and classify the failure as unknown.
    """
    from hyperloom.common.failure_signature import classify_failure

    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "(VllmWorker pid=3) INFO starting engine",
                "(VllmWorker pid=3) ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
                "(EngineCore pid=2) ERROR RuntimeError: Engine core initialization failed. "
                "See root cause above. Failed core proc(s): {}",
            ]
        ),
        encoding="utf-8",
    )

    excerpt = death_excerpt(str(log))

    assert excerpt is not None
    assert "out of memory" in excerpt.lower(), "the cause above the wrapper must survive"
    assert classify_failure(excerpt).kind == "resource_constraint"


def test_a_death_past_any_head_bound_and_outside_the_tail_is_still_found(tmp_path):
    """Streaming, not sampling: a bounded head read leaves the middle unsearched.

    A server healthy for a long while, then dead, then flooding the log with
    downstream errors puts its fatal marker beyond any fixed head window and
    before any fixed tail window.
    """
    from hyperloom.common.failure_signature import classify_failure

    log = tmp_path / "server.log"
    filler = "(APIServer pid=1) INFO:     healthy request served fine and produced ordinary output\n"
    with log.open("w", encoding="utf-8") as fh:
        written = 0
        while written < 17 * 1024 * 1024:  # past a 16 MiB head bound
            fh.write(filler)
            written += len(filler)
        fh.write("(EngineCore pid=2) ERROR [core.py:1138] EngineCore encountered a fatal error.\n")
        fh.write(
            "(EngineCore pid=2) ERROR [core.py:1138] RuntimeError: Worker failed with error "
            "'HIP out of memory. Tried to allocate 10.54 GiB'\n"
        )
        trailing = '(APIServer pid=1) INFO:     "POST /v1/completions HTTP/1.1" 500 Internal Server Error\n'
        for _ in range(4000):  # past a 64 KiB tail window
            fh.write(trailing)

    excerpt = death_excerpt(str(log))

    assert excerpt is not None
    assert "HIP out of memory" in excerpt
    assert classify_failure(excerpt).kind == "resource_constraint"


def test_a_legacy_marker_that_contains_a_fatal_one_keeps_legacy_handling(tmp_path):
    """``EngineDeadError`` is a substring of ``AsyncEngineDeadError``.

    Classing by substring without ordering sends a legacy line down the
    post-startup path, which keeps no leading context -- and for these markers
    the actionable cause is exactly the line above.
    """
    from hyperloom.common.failure_signature import classify_failure

    for legacy in ("AsyncEngineDeadError", "raise EngineDeadError"):
        log = tmp_path / f"server-{legacy.replace(' ', '_')}.log"
        log.write_text(
            "\n".join(
                [
                    "(VllmWorker pid=3) ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
                    f"(APIServer pid=1) ERROR {legacy}: the engine is gone",
                ]
            ),
            encoding="utf-8",
        )

        excerpt = death_excerpt(str(log))

        assert excerpt is not None, legacy
        assert "out of memory" in excerpt.lower(), f"{legacy} must keep the cause above it"
        assert classify_failure(excerpt).kind == "resource_constraint", legacy
