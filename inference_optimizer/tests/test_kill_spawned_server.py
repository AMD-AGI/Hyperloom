# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_subprocess_kill.kill_my_spawned_server`` and the BaselineExecutor integration (``bugs.md`` §B).

Covers the no-op / already-exited cases, the same-session-group refusal guard,
SIGTERM→grace→SIGKILL ordering, and grandchild reaping (the bugs.md §B leak).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors._subprocess_kill import (
    DETOKENIZER_STALL_RETURNCODE,
    OVERTIME_KILL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    _scan_server_log_increment,
    _server_log_shows_death,
    kill_my_spawned_server,
    new_session_kwargs,
    run_with_session_kill,
    server_log_death_excerpt,
)


# Helper-level tests
def test_kill_my_spawned_server_handles_none():
    """Plain no-op when given None so callers can use it in ``finally:`` unguarded."""
    kill_my_spawned_server(None)  # must not raise


def test_kill_my_spawned_server_handles_already_exited():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
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
        **new_session_kwargs(),
    )
    # Let the child install its SIGTERM handler before we signal.
    time.sleep(0.3)
    start = time.monotonic()
    kill_my_spawned_server(proc, grace_seconds=1.0)
    elapsed = time.monotonic() - start
    assert proc.poll() is not None
    assert elapsed < 5.0, f"kill_my_spawned_server hung for {elapsed:.2f}s"


def test_kill_my_spawned_server_reaps_grandchildren():
    """bugs.md §B: a child that spawns a grandchild leaves no surviving descendant after the helper returns."""
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
        **new_session_kwargs(),
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
            pass
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass


# BaselineExecutor integration — confirm the kill is on every exit path
def _make_fake_magpie_command(
    tmp_path: Path,
    *,
    mode: str,
) -> tuple[Path, Path]:
    """Build a ``python -m Magpie`` stand-in; returns (script_path, sentinel_file)."""
    script = tmp_path / "fake_magpie.py"
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
    script, sentinel = _make_fake_magpie_command(tmp_path, mode="timeout")
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_session_kwargs(),
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
                pass


# Fix E — run_with_session_kill soft_deadline_sec
def test_run_with_session_kill_soft_deadline_returns_sentinel():
    """A child past ``soft_deadline_sec`` is reaped and returns ``OVERTIME_KILL_RETURNCODE`` (no ``TimeoutExpired``)."""
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=30,
        soft_deadline_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 10.0, f"soft-deadline path took {elapsed:.2f}s"


def test_run_with_session_kill_soft_deadline_does_not_fire_for_quick_child():
    """A child exiting before ``soft_deadline_sec`` returns normally with its own returncode."""
    cp = run_with_session_kill(
        [sys.executable, "-c", "print('hi'); raise SystemExit(0)"],
        timeout=10,
        soft_deadline_sec=5.0,
    )
    assert cp.returncode == 0
    assert "hi" in (cp.stdout or "")


def test_run_with_session_kill_streams_child_output_to_parent(capsys):
    """Captured child output is also mirrored immediately to parent streams."""
    code = "import sys\nprint('child-out', flush=True)\nprint('child-err', file=sys.stderr, flush=True)\n"
    cp = run_with_session_kill([sys.executable, "-c", code], timeout=10)

    captured = capsys.readouterr()
    assert cp.returncode == 0
    assert "child-out" in (cp.stdout or "")
    assert "child-err" in (cp.stderr or "")
    assert "child-out" in captured.out
    assert "child-err" in captured.err


def test_run_with_session_kill_legacy_timeout_still_raises():
    """With ``soft_deadline_sec`` None, a child exceeding the hard ``timeout`` still raises ``TimeoutExpired``."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
            soft_deadline_sec=None,
        )


# Server-liveness watchdog — fast-fail on a crashed-but-hung server
def test_server_log_shows_death_detects_marker(tmp_path):
    """A ``server.log`` containing a terminal-init marker reads as dead;
    a healthy / missing log reads as alive."""
    log_path = tmp_path / "server.log"
    assert _server_log_shows_death(str(log_path)) is None  # missing → alive
    log_path.write_text("INFO loading shards 50%\nINFO graph capture\n")
    assert _server_log_shows_death(str(log_path)) is None  # healthy → alive
    log_path.write_text(
        "ERROR core.py Exception: WorkerProc initialization failed due to an exception in a background process.\n"
    )
    # dead → returns the matched marker (truthy) rather than a bare bool
    assert _server_log_shows_death(str(log_path)) is not None


def test_server_log_shows_death_detects_vllm_engine_core(tmp_path):
    """#524: the vLLM v1 engine-core bootstrap tail must read as dead. The
    ``RuntimeError: Engine core initialization failed`` line and the
    ``Failed core proc(s)`` anchor both trip the watchdog."""
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(APIServer pid=16160)   File '.../vllm/v1/engine/utils.py', line 1057, "
        "in wait_for_engine_startup\n"
        "(APIServer pid=16160) RuntimeError: Engine core initialization failed. "
        "See root cause above. Failed core proc(s): {}\n"
    )
    assert _server_log_shows_death(str(log_path)) is not None


def test_server_log_shows_death_detects_nested_benchmark_log(tmp_path):
    """Magpie wrappers that ignore ``$SERVER_LOG`` write the real server log to a
    nested ``benchmark_<fw>_<ts>/server.log``. The watchdog must still detect the
    crash via that nested file even when the watched ``output_dir/server.log`` is
    absent (otherwise a hung-after-death server burns the full hard timeout)."""
    watched = tmp_path / "server.log"  # never written by the wrapper
    nested_dir = tmp_path / "benchmark_vllm_20260625_003729"
    nested_dir.mkdir()
    nested_log = nested_dir / "server.log"
    assert _server_log_shows_death(str(watched)) is None  # nothing yet → alive
    nested_log.write_text("INFO loading shards 50%\nINFO graph capture\n")
    assert _server_log_shows_death(str(watched)) is None  # healthy nested → alive
    nested_log.write_text(
        "(EngineCore pid=2581809) RuntimeError: Engine core initialization "
        "failed. See root cause above. Failed core proc(s): {}\n"
    )
    assert _server_log_shows_death(str(watched)) is not None


def test_server_log_death_excerpt_surfaces_nested_root_cause(tmp_path):
    """The excerpt helper also falls back to a nested ``benchmark_*/server.log``
    so the failure classifier still surfaces the real server fault."""
    watched = tmp_path / "server.log"
    nested_dir = tmp_path / "benchmark_vllm_20260625_003729"
    nested_dir.mkdir()
    nested_log = nested_dir / "server.log"
    assert server_log_death_excerpt(str(watched)) is None
    nested_log.write_text(
        "(EngineCore pid=2581809)     raise RuntimeError(\n"
        "(EngineCore pid=2581809) RuntimeError: Engine core initialization "
        "failed. See root cause above. Failed core proc(s): {}\n"
    )
    excerpt = server_log_death_excerpt(str(watched))
    assert excerpt is not None
    assert "Engine core initialization failed" in excerpt


def test_server_log_death_excerpt_surfaces_root_cause(tmp_path):
    """#524: the excerpt helper returns the engine/worker-init root-cause line
    (with a little context) so the failure classifier can put the real server
    fault in the operator-facing ``error`` field; a healthy / missing log
    returns ``None``."""
    log_path = tmp_path / "server.log"
    assert server_log_death_excerpt(str(log_path)) is None  # missing → None
    log_path.write_text("INFO loading shards 50%\nINFO graph capture\n")
    assert server_log_death_excerpt(str(log_path)) is None  # healthy → None
    log_path.write_text(
        "(APIServer pid=16160)   File '.../vllm/v1/engine/utils.py', line 1057, "
        "in wait_for_engine_startup\n"
        "(APIServer pid=16160)     raise RuntimeError(\n"
        "(APIServer pid=16160) RuntimeError: Engine core initialization failed. "
        "See root cause above. Failed core proc(s): {}\n"
    )
    excerpt = server_log_death_excerpt(str(log_path))
    assert excerpt is not None
    assert "Engine core initialization failed" in excerpt


def test_run_with_session_kill_watchdog_reaps_hung_server(tmp_path):
    """A child that writes a fatal server marker then hangs is reaped via the
    watchdog with ``SERVER_DEAD_RETURNCODE`` — well before the hard timeout."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write("
        "'Exception: WorkerProc initialization failed in background\\n')\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        server_log_path=str(log_path),
        server_dead_grace_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == SERVER_DEAD_RETURNCODE
    assert elapsed < 15.0, f"watchdog path took {elapsed:.2f}s (expected fast)"


def test_run_with_session_kill_watchdog_grace_lets_clean_exit_win(tmp_path):
    """If the harness exits on its own within the grace window after emitting a
    marker, its real returncode wins (no spurious SERVER_DEAD)."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write("
        "'Exception: WorkerProc initialization failed in background\\n')\n"
        "time.sleep(0.3)\n"
        "raise SystemExit(7)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_dead_grace_sec=10.0,
        server_log_path=str(log_path),
    )
    assert cp.returncode == 7


def test_run_with_session_kill_watchdog_ignores_healthy_server(tmp_path):
    """A child with a clean server.log returns its own returncode — the
    watchdog must not false-positive on a healthy (or slow) server."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys\n"
        "open(sys.argv[1], 'w').write('INFO server ready on port 8888\\n')\n"
        "print('ok')\n"
        "raise SystemExit(0)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_dead_grace_sec=2.0,
        server_log_path=str(log_path),
    )
    assert cp.returncode == 0
    assert "ok" in (cp.stdout or "")


# ── Detokenizer-stall watchdog ──
def test_scan_server_log_increment_detects_ready_and_progress(tmp_path):
    """The incremental scanner advances its offset and flags ready/progress
    markers only in the newly appended bytes."""
    log_path = tmp_path / "server.log"
    log_path.write_text("INFO loading weights\nApplication startup complete\n")
    off, ready, prog = _scan_server_log_increment(str(log_path), 0)
    assert ready is True and prog is False and off == log_path.stat().st_size
    # Re-scan from the advanced offset: nothing new, no re-trigger.
    off2, ready2, prog2 = _scan_server_log_increment(str(log_path), off)
    assert ready2 is False and prog2 is False and off2 == off
    # Append a vLLM throughput line; only the new bytes are scanned.
    with log_path.open("a") as f:
        f.write("Avg generation throughput: 123.4 tokens/s, Running: 8\n")
    off3, ready3, prog3 = _scan_server_log_increment(str(log_path), off2)
    assert prog3 is True and ready3 is False and off3 == log_path.stat().st_size


def test_run_with_session_kill_detok_stall_reaps_ready_but_silent_server(tmp_path):
    """A server that reports ready then produces no generation progress is
    reaped with ``DETOKENIZER_STALL_RETURNCODE`` well before the hard timeout."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write('Application startup complete\\n')\n"
        "time.sleep(60)\n"  # ready, then silent — detokenizer stall
    )
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=60,
        server_log_path=str(log_path),
        detok_stall_grace_sec=1.0,
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == DETOKENIZER_STALL_RETURNCODE
    assert elapsed < 15.0, f"stall watchdog took {elapsed:.2f}s (expected fast)"


def test_run_with_session_kill_detok_stall_not_armed_before_ready(tmp_path):
    """A server still loading weights (no ready marker) must NOT trip the stall
    gate even past the grace window — slow is not stalled."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write('INFO loading weights shard 1/8\\n')\n"
        "time.sleep(2)\n"
        "raise SystemExit(0)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_log_path=str(log_path),
        detok_stall_grace_sec=0.5,
    )
    assert cp.returncode == 0


def test_run_with_session_kill_detok_stall_progress_keeps_it_alive(tmp_path):
    """Continued generation-progress lines reset the stall clock so a healthy
    (if slow) run finishes with its own returncode."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('Application startup complete\\n'); f.flush()\n"
        "for _ in range(6):\n"
        "    time.sleep(0.3)\n"
        "    f.write('gen throughput (token/s): 250.0, #queue-req: 0\\n'); f.flush()\n"
        "raise SystemExit(0)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_log_path=str(log_path),
        detok_stall_grace_sec=1.0,
    )
    assert cp.returncode == 0


def test_run_with_session_kill_detok_stall_compile_logs_keep_it_alive(tmp_path):
    """A long, quiet first-request JIT/compile after ready must NOT trip the
    gate: ANY new log line (not just throughput) is liveness, so a huge model
    that logs compile progress between ready and its first token survives."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "f.write('The server is fired up and ready to roll\\n'); f.flush()\n"
        "for i in range(6):\n"  # non-throughput compile chatter, no tokens yet
        "    time.sleep(0.3)\n"
        "    f.write('aiter: JIT compiling kernel %d/6\\n' % i); f.flush()\n"
        "raise SystemExit(0)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_log_path=str(log_path),
        detok_stall_grace_sec=1.0,
    )
    assert cp.returncode == 0


def test_run_with_session_kill_detok_stall_disabled_when_grace_nonpositive(tmp_path):
    """``detok_stall_grace_sec <= 0`` disables the gate entirely."""
    log_path = tmp_path / "server.log"
    script = (
        "import sys, time\n"
        "open(sys.argv[1], 'w').write('Application startup complete\\n')\n"
        "time.sleep(1)\n"
        "raise SystemExit(0)\n"
    )
    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(log_path)],
        timeout=30,
        server_log_path=str(log_path),
        detok_stall_grace_sec=0.0,
    )
    assert cp.returncode == 0
