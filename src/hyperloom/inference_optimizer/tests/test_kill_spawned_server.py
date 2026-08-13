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

from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    DETOKENIZER_STALL_RETURNCODE,
    OVERTIME_KILL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    _scan_logs_increment,
    _scan_server_log_increment,
    _server_log_shows_death,
    kill_my_spawned_server,
    new_session_kwargs,
    run_with_session_kill,
    server_log_death_excerpt,
)


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
            # PID file already gone; nothing to clean up.
            pass
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                # Process already exited; nothing to signal.
                pass


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
                # Process already exited; nothing to signal.
                pass


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


def test_run_with_session_kill_eval_start_marker_retires_soft_deadline(tmp_path):
    """Once the accuracy eval announces itself the soft deadline stops applying:
    the deadline bounds the throughput phase, and its anchor excludes eval."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Application startup complete\nHYPERLOOM_EVAL_START\n")
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(4)"],
        timeout=30,
        soft_deadline_sec=1.0,
        server_log_path=str(log_path),
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == 0
    assert elapsed >= 3.5, f"child was cut short at {elapsed:.2f}s"


def test_run_with_session_kill_soft_deadline_still_fires_without_eval_marker(tmp_path):
    """Without the eval marker the deadline keeps its teeth — a genuinely slow
    throughput phase is still reaped."""
    log_path = tmp_path / "server.log"
    log_path.write_text("Application startup complete\n")
    start = time.monotonic()
    cp = run_with_session_kill(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=30,
        soft_deadline_sec=1.0,
        server_log_path=str(log_path),
    )
    elapsed = time.monotonic() - start
    assert cp.returncode == OVERTIME_KILL_RETURNCODE
    assert elapsed < 10.0, f"soft-deadline path took {elapsed:.2f}s"


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
            timeout=60,
            session_deadline_sec=time.monotonic() - 1.0,
        )
        elapsed = time.monotonic() - start
        assert cp.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
        assert cp.returncode != OVERTIME_KILL_RETURNCODE, (
            "a budget kill must not share the overtime code, which asserts the variant is slow"
        )
        assert elapsed < 10.0, f"session-deadline path took {elapsed:.2f}s"

    def test_eval_start_does_not_retire_the_session_budget(self, tmp_path):
        """The marker that retires the soft deadline must not retire this one.

        An accuracy eval that starts one minute before the run is out of time
        still has to stop; this is the whole reason the two are separate channels.
        """
        log_path = tmp_path / "server.log"
        log_path.write_text("Application startup complete\nHYPERLOOM_EVAL_START\n")
        start = time.monotonic()
        cp = run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=60,
            soft_deadline_sec=1.0,
            server_log_path=str(log_path),
            session_deadline_sec=time.monotonic() + 1.5,
        )
        elapsed = time.monotonic() - start
        assert cp.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
        assert elapsed < 10.0, f"session budget was not enforced during eval ({elapsed:.2f}s)"

    def test_a_budget_with_room_left_leaves_the_child_alone(self):
        start = time.monotonic()
        cp = run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            timeout=60,
            session_deadline_sec=time.monotonic() + 3600.0,
        )
        elapsed = time.monotonic() - start
        assert cp.returncode == 0
        assert elapsed >= 1.5, f"child was cut short at {elapsed:.2f}s"

    def test_no_session_deadline_keeps_the_previous_behaviour(self):
        cp = run_with_session_kill(
            [sys.executable, "-c", "print('done')"],
            timeout=30,
            session_deadline_sec=None,
        )
        assert cp.returncode == 0
        assert "done" in (cp.stdout or "")


def _sentinel_returncodes() -> dict[int, set[str]]:
    """Map every sentinel returncode to the qualified names that claim it."""
    from hyperloom.orchestrator.actions.executors import _ray_serving, _subprocess_kill

    assigned: dict[int, set[str]] = {}
    for module in (_subprocess_kill, _ray_serving):
        short = module.__name__.rsplit(".", 1)[-1]
        for name, value in vars(module).items():
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            if not (name.endswith("_RETURNCODE") or name.endswith("_RC")):
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


def test_an_actor_timeout_is_not_recorded_as_a_failed_agentx_preflight():
    """The two causes that share ``_run_magpie``'s return channel stay apart.

    ``_run_magpie`` returns ``AGENTX_PREFLIGHT_RETURNCODE`` when the execution
    boundary fails preflight and, a few lines on, whatever the serving lease's
    actor returned -- including ``_ACTOR_TIMEOUT_RC``. Callers see one
    ``returncode`` either way, so the two sharing a number (which they did) is
    enough to have a hung actor blamed on a missing aiperf.
    """
    from hyperloom.orchestrator.actions.executors import _ray_serving, _subprocess_kill

    assert _ray_serving._ACTOR_TIMEOUT_RC != _subprocess_kill.AGENTX_PREFLIGHT_RETURNCODE


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


def test_run_with_session_kill_reports_each_line_of_child_output():
    """The liveness callback fires while the child runs, once per line it emits."""
    code = "import sys, time\nfor i in range(3):\n    print(i, flush=True)\n    time.sleep(0.05)\n"
    lines: list[float] = []

    cp = run_with_session_kill(
        [sys.executable, "-c", code],
        timeout=10,
        on_output=lambda: lines.append(time.monotonic()),
    )

    assert cp.returncode == 0
    assert len(lines) == 3


def _appends_until_stopped(path: Path, line: str, stop: threading.Event) -> threading.Thread:
    """Start a writer that appends ``line`` to ``path`` until ``stop`` is set.

    Stands in for a writer that is provably not the child under test: the
    inference server, which keeps logging while its benchmark client is wedged.

    Args:
        path (Path): Log file to append to.
        line (str): Line written each round, newline included.
        stop (threading.Event): Set by the caller to end the writer.

    Returns:
        threading.Thread: The started daemon writer.
    """

    def _write() -> None:
        with path.open("a") as fh:
            while not stop.wait(0.05):
                fh.write(line)
                fh.flush()

    writer = threading.Thread(target=_write, daemon=True)
    writer.start()
    return writer


@pytest.mark.parametrize(
    ("appended_line", "reports_liveness"),
    [
        ('INFO:     127.0.0.1:0 - "GET /health HTTP/1.1" 200 OK\n', False),
        ("Avg generation throughput: 0.0 tokens/s, Running: 0 reqs\n", False),
        ("Avg generation throughput: 123.4 tokens/s, Running: 8 reqs\n", True),
    ],
    ids=[
        "an_access_log_line",
        "an_idle_engines_throughput_line",
        "a_generation_throughput_line",
    ],
)
def test_run_with_session_kill_reports_a_silent_child_alive_only_on_real_progress(
    tmp_path,
    appended_line: str,
    reports_liveness: bool,
):
    """A log that grew is not the child talking; a log that shows tokens flowing is.

    All three lines are written by the same third party, so growth alone cannot
    tell them apart — and one of them is the access line vLLM and sglang emit
    per request, including the health probe the robustness agent issues on its
    own tick. Counting those as the child's output closes a loop where the
    monitor's probe manufactures the evidence that suppresses its own stall
    accusation, and turns the heartbeat into the bare timer it documents itself
    as never being. A throughput line is different in kind — whoever logged it,
    tokens were being produced during the interval — but only if it carries a
    rate: some vLLM builds keep printing the stats line at ``0.0 tokens/s`` on
    an idle engine, and an engine goes idle precisely when the client that was
    driving it wedges, so the zero-rate line is the shape this failure actually
    takes in production.
    """
    log_path = tmp_path / "server.log"
    log_path.write_text("Application startup complete\n")
    stop = threading.Event()
    writer = _appends_until_stopped(log_path, appended_line, stop)
    reported: list[int] = []
    try:
        cp = run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            timeout=30,
            server_log_path=str(log_path),
            detok_stall_grace_sec=30.0,
            on_output=lambda: reported.append(1),
        )
    finally:
        stop.set()
        writer.join(timeout=5.0)

    assert cp.returncode == 0
    assert bool(reported) is reports_liveness, (
        f"a child that printed nothing was reported alive {len(reported)} times "
        f"by {appended_line.strip()!r}"
    )


def test_run_with_session_kill_reports_the_output_a_child_redirected_to_disk(tmp_path):
    """A round whose body writes only to ``benchmark_stderr.log`` is still working.

    The scriptable and bypass paths run the customer body with its stderr
    redirected there rather than into the parent's pipe, and a long phase of one
    — a client that logs its request counter but produces no server throughput
    line yet — would otherwise have nothing left to report liveness with.
    """
    bench = tmp_path / "benchmark_atom_20260731_085850"
    bench.mkdir(parents=True)
    (bench / "server.log").write_text("Application startup complete\n")
    script = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'a')\n"
        "for i in range(6):\n"
        "    time.sleep(0.2)\n"
        "    f.write('bench: request %d done\\n' % i); f.flush()\n"
    )
    reported: list[int] = []

    cp = run_with_session_kill(
        [sys.executable, "-c", script, str(bench / "benchmark_stderr.log")],
        timeout=30,
        server_log_path=str(tmp_path / "server.log"),
        detok_stall_grace_sec=30.0,
        on_output=lambda: reported.append(1),
    )

    assert cp.returncode == 0
    assert reported, "a child talking only through its redirected stderr was never reported alive"


def test_run_with_session_kill_survives_a_broken_liveness_callback():
    """Reporting is best-effort; a raising callback must not eat child output."""

    def _boom() -> None:
        raise RuntimeError("callback is broken")

    cp = run_with_session_kill(
        [sys.executable, "-c", "print('still-captured', flush=True)"],
        timeout=10,
        on_output=_boom,
    )

    assert cp.returncode == 0
    assert "still-captured" in (cp.stdout or "")


def test_run_with_session_kill_legacy_timeout_still_raises():
    """With ``soft_deadline_sec`` None, a child exceeding the hard ``timeout`` still raises ``TimeoutExpired``."""
    with pytest.raises(subprocess.TimeoutExpired):
        run_with_session_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
            soft_deadline_sec=None,
        )


# Server-liveness watchdog
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
    """The vLLM v1 engine-core bootstrap tail must read as dead."""
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
    crash via that nested file even when the watched ``output_dir/server.log`` is absent."""
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
    """The excerpt helper returns the engine/worker-init root-cause line (with a
    little context) for the failure classifier; a healthy / missing log returns ``None``."""
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


def test_server_log_death_excerpt_surfaces_config_validation_arch_miss(tmp_path):
    """A config-validation-stage failure (brand-new checkpoint ``model_type``
    unknown to the installed transformers/vLLM) dies BEFORE the engine starts and
    must still be surfaced as a fatal excerpt. Without this the enablement failure
    classifier only sees Magpie's ``subprocess_nonzero`` stdout tail, classifies
    ``unknown``, and never seeds the ``pip install -U transformers`` bridge —
    starving every enablement round of the real root cause (DeepSeek-V4 repro)."""
    from hyperloom.agents.framework.enablement import classify_failure

    log_path = tmp_path / "server.log"
    # A healthy INFO banner naming architectures must NOT trip the markers.
    log_path.write_text("INFO [registry] Model architectures ['Qwen3ForCausalLM'] loaded\n")
    assert server_log_death_excerpt(str(log_path)) is None
    # The real DeepSeek-V4 config-validation failure.
    log_path.write_text(
        "(APIServer pid=1046234) Traceback (most recent call last):\n"
        "(APIServer pid=1046234) pydantic_core._pydantic_core.ValidationError: "
        "1 validation error for ModelConfig\n"
        "(APIServer pid=1046234)   Value error, The checkpoint you are trying to "
        "load has model type `deepseek_v4` but Transformers does not recognize "
        "this architecture.\n"
    )
    excerpt = server_log_death_excerpt(str(log_path))
    assert excerpt is not None
    assert "does not recognize this architecture" in excerpt
    # The extracted excerpt must classify as missing_model_arch (not unknown),
    # which is what seeds the deterministic pip-install enablement bridge.
    sig = classify_failure(excerpt)
    assert sig.kind == "missing_model_arch"
    assert sig.offending_symbol == "deepseek_v4"


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
    off, ready, prog, ev = _scan_server_log_increment(str(log_path), 0)
    assert ready is True and prog is False and ev is False and off == log_path.stat().st_size
    # Re-scan from the advanced offset: nothing new, no re-trigger.
    off2, ready2, prog2, ev2 = _scan_server_log_increment(str(log_path), off)
    assert ready2 is False and prog2 is False and ev2 is False and off2 == off
    # Append a vLLM throughput line; only the new bytes are scanned.
    with log_path.open("a") as f:
        f.write("Avg generation throughput: 123.4 tokens/s, Running: 8\n")
    off3, ready3, prog3, ev3 = _scan_server_log_increment(str(log_path), off2)
    assert prog3 is True and ready3 is False and ev3 is False and off3 == log_path.stat().st_size
    # The eval-start marker is reported independently of ready/progress.
    with log_path.open("a") as f:
        f.write("HYPERLOOM_EVAL_START\n")
    off4, ready4, prog4, ev4 = _scan_server_log_increment(str(log_path), off3)
    assert ev4 is True and ready4 is False and prog4 is False and off4 == log_path.stat().st_size
    # An idle engine keeps printing the same line with no rate on it: the value
    # is the progress signal, not the marker.
    with log_path.open("a") as f:
        f.write("Avg generation throughput: 0.0 tokens/s, Running: 0 reqs\n")
    off5, _ready5, prog5, _ev5 = _scan_server_log_increment(str(log_path), off4)
    assert prog5 is False and off5 == log_path.stat().st_size


def test_scan_logs_increment_reads_nested_stderr_for_eval_start(tmp_path):
    """The real Magpie layout: the caller passes ``<output_dir>/server.log``,
    which does not exist, while the engine log and the eval-start marker live in
    a ``benchmark_*/`` subdir -- the marker only ever reaching stderr."""
    output_dir = tmp_path / "measure_round"
    bench = output_dir / "benchmark_atom_20260731_085850"
    bench.mkdir(parents=True)
    (bench / "server.log").write_text("Application startup complete\n")
    (bench / "benchmark_stderr.log").write_text("running benchmark\n")
    passed = str(output_dir / "server.log")
    assert not Path(passed).exists()

    offsets: dict[str, int] = {}
    first = _scan_logs_increment(passed, offsets)
    assert first.saw_ready is True and first.saw_eval_start is False and first.grew is True

    # The marker lands in stderr, never in server.log.
    with (bench / "benchmark_stderr.log").open("a") as f:
        f.write("HYPERLOOM_EVAL_START\n")
    second = _scan_logs_increment(passed, offsets)
    assert second.saw_eval_start is True and second.saw_ready is False and second.grew is True

    # Nothing new appended: no re-trigger, offsets stay put.
    third = _scan_logs_increment(passed, offsets)
    assert third.saw_eval_start is False and third.grew is False


def test_scan_logs_increment_tells_the_childs_own_log_from_the_servers(tmp_path):
    """Only one of the resolved logs is written by the process being waited on.

    ``server.log`` is the inference server's; ``benchmark_stderr.log`` is where
    Magpie redirects the benchmark body's own stderr, so the parent's pipe stays
    empty for the whole round and that file is the only place the child's own
    output shows up. Liveness that cannot tell them apart either vouches for a
    wedged client or leaves a working one unable to report.
    """
    bench = tmp_path / "benchmark_atom_20260731_085850"
    bench.mkdir(parents=True)
    server_log = bench / "server.log"
    child_log = bench / "benchmark_stderr.log"
    server_log.write_text("Application startup complete\n")
    child_log.write_text("running benchmark\n")
    passed = str(tmp_path / "server.log")
    offsets: dict[str, int] = {}
    _scan_logs_increment(passed, offsets)

    with server_log.open("a") as f:
        f.write('INFO:     127.0.0.1:0 - "GET /health HTTP/1.1" 200 OK\n')
    server_only = _scan_logs_increment(passed, offsets)
    assert server_only.grew is True and server_only.child_spoke is False

    with child_log.open("a") as f:
        f.write("bench: 128/2000 requests done\n")
    child_only = _scan_logs_increment(passed, offsets)
    assert child_only.grew is True and child_only.child_spoke is True


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
