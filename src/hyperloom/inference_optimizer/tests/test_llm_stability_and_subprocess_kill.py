# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Covers the LLM-transport stability env helper and the process-group kill in ``_run_subprocess`` that reaps a hung grandchild instead of orphaning it."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles._llm_stability_env import (
    DEFAULT_API_TIMEOUT_MS,
    apply_llm_stability_env,
)
from hyperloom.orchestrator.kernel.request_handlers import _run_subprocess, _tool_label
from hyperloom.orchestrator.trace.task_progress import progress_scope

from .conftest import chatty_child


def test_apply_llm_stability_env_sets_defaults():
    env: dict[str, str] = {}
    apply_llm_stability_env(env)
    # API_TIMEOUT_MS is opt-in: some clients treat it as a total request timeout that can kill a legitimate long
    # streaming response.
    assert "API_TIMEOUT_MS" not in env
    assert DEFAULT_API_TIMEOUT_MS == "300000"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTOUPDATER"] == "1"


def test_apply_llm_stability_env_respects_operator_override():
    env = {"API_TIMEOUT_MS": "60000"}
    apply_llm_stability_env(env)
    # Must not clobber an operator-set value.
    assert env["API_TIMEOUT_MS"] == "60000"
    assert env["DISABLE_AUTOUPDATER"] == "1"


def test_apply_llm_stability_env_custom_timeout():
    env: dict[str, str] = {}
    apply_llm_stability_env(env, api_timeout_ms="120000")
    assert env["API_TIMEOUT_MS"] == "120000"


async def test_run_subprocess_returns_output_normally():
    rc, stdout, stderr = await _run_subprocess(
        [sys.executable, "-c", "print('hello-stdout')"],
        timeout_sec=30,
    )
    assert rc == 0
    assert "hello-stdout" in stdout


_ECHO_UNBUFFERED = [sys.executable, "-c", "import os; print(os.environ['PYTHONUNBUFFERED'])"]


async def test_run_subprocess_unbuffers_its_child(monkeypatch):
    """A block-buffered child would look dead between flushes."""
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    rc, stdout, _stderr = await _run_subprocess(_ECHO_UNBUFFERED, timeout_sec=30)
    assert rc == 0
    assert stdout.strip() == "1"


async def test_run_subprocess_leaves_an_operator_chosen_buffering_alone(monkeypatch):
    """Defaulting is help; overriding an explicit setting is a surprise."""
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    rc, stdout, _stderr = await _run_subprocess(_ECHO_UNBUFFERED, timeout_sec=30)
    assert rc == 0
    assert stdout.strip() == "0"


async def test_run_subprocess_reports_activity_for_its_child_output(monkeypatch):
    """The heartbeat above it reports only when this tally moves."""
    from hyperloom.orchestrator.actions.executors import _subprocess_kill

    counted: list[int] = []
    real = _subprocess_kill.run_with_session_kill

    def _spy(cmd, **kwargs):
        calls = 0
        reported = kwargs.pop("on_output")

        def _count() -> None:
            nonlocal calls
            calls += 1
            reported()

        try:
            return real(cmd, on_output=_count, **kwargs)
        finally:
            counted.append(calls)

    monkeypatch.setattr(_subprocess_kill, "run_with_session_kill", _spy)
    rc, stdout, _stderr = await _run_subprocess(
        [sys.executable, "-c", "print('a')\nprint('b')"],
        timeout_sec=30,
    )

    assert rc == 0
    assert stdout.splitlines() == ["a", "b"]
    assert len(counted) == 1
    assert counted[0] > 0


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
@pytest.mark.parametrize("text", [True, False], ids=["text", "bytes"])
def test_stream_capture_tees_partial_output_before_child_exit(monkeypatch, stream_name, text):
    """A partial UTF-8 codepoint counts as activity without blocking the visible prefix."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import _StreamCapture

    mirrored = threading.Event()
    activity = threading.Event()
    sink_base = io.StringIO if text else io.BytesIO

    class Sink(sink_base):
        def write(self, data):
            written = super().write(data)
            mirrored.set()
            return written

    sink = Sink()
    monkeypatch.setattr(sys, stream_name, sink if text else SimpleNamespace(buffer=sink))
    descriptor = 1 if stream_name == "stdout" else 2
    script = (
        f"import os,sys; os.write({descriptor}, b'A\\xe2'); "
        "sys.stdin.buffer.read(1); "
        f"os.write({descriptor}, b'\\x82\\xacB')"
    )
    with subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        capture = _StreamCapture(proc, text=text, on_output=activity.set)
        capture.start()
        try:
            assert activity.wait(5), "partial bytes never registered as output activity"
            assert mirrored.wait(5), "output without a newline was not mirrored before EOF"
            assert proc.poll() is None, "the child must still be waiting for the parent"
            assert capture.last_activity_at is not None
            assert sink.getvalue() == ("A" if text else b"A\xe2")
            proc.stdin.write(b"x")
            proc.stdin.flush()
            assert proc.wait(timeout=5) == 0
            stdout, stderr = capture.finish()
            expected = "A€B" if text else b"A\xe2\x82\xacB"
            assert (stdout if stream_name == "stdout" else stderr) == expected
            assert (stderr if stream_name == "stdout" else stdout) == ("" if text else b"")
            assert sink.getvalue() == expected
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            capture.finish()


async def test_a_kernel_tool_keeps_reporting_while_its_child_works(monkeypatch, progress_cadence):
    """A trace analysis blocks for the better part of an hour behind one ``await``."""
    from hyperloom.orchestrator.actions.executors import _subprocess_kill

    def _done(cmd, **_kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        _subprocess_kill,
        "run_with_session_kill",
        chatty_child(progress_cadence, _done, blocks_for_s=600.0, line_every_s=30.0),
    )

    with progress_scope(progress_cadence.sink()):
        rc, _stdout, _stderr = await _run_subprocess([sys.executable, "-c", "pass"], timeout_sec=30)

    assert rc == 0
    running = [note for note in progress_cadence.notes if note["status"] == "running"]
    assert len(running) >= 3
    assert all(note["output_lines"] > 0 for note in running)
    assert progress_cadence.widest_silence() <= 150.0


def test_a_tool_is_named_after_the_script_it_runs():
    """``kernel_tool:tracelens_analysis`` is what an operator has to recognize."""
    assert _tool_label(["python3", "/opt/tools/tracelens_analysis.py", "--x"]) == "tracelens_analysis"
    assert _tool_label(["ls", "-l"]) == "ls"
    assert _tool_label([]) == "subprocess"


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
async def test_run_subprocess_kills_grandchild_on_timeout(tmp_path):
    """A timed-out child that spawned a long-lived grandchild must have the grandchild reaped too (process-group kill), not orphaned."""
    pidfile = tmp_path / "grandchild.pid"
    # Parent spawns a grandchild `sleep 300`, records its pid, then blocks.
    script = (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen(['sleep', '300'])\n"
        "open(sys.argv[1], 'w').write(str(gc.pid))\n"
        "time.sleep(300)\n"
    )

    with pytest.raises(Exception) as excinfo:
        await _run_subprocess(
            [sys.executable, "-c", script, str(pidfile)],
            timeout_sec=2,
        )
    # A hard timeout surfaces as TimeoutExpired.
    assert isinstance(excinfo.value, subprocess.TimeoutExpired)

    assert pidfile.exists(), "grandchild never recorded its pid"
    gc_pid = int(pidfile.read_text().strip())

    # Poll briefly for the grandchild to be reaped by the process-group kill.
    deadline = time.monotonic() + 10.0
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            alive = False
            break
        except PermissionError:
            alive = False
            break
        time.sleep(0.1)

    if alive:
        # Best-effort cleanup so a regression doesn't leak a 300s sleep.
        try:
            os.kill(gc_pid, signal.SIGKILL)
        except OSError:
            # Process may already have disappeared; nothing to clean up.
            pass
        pytest.fail(f"grandchild pid={gc_pid} survived the timeout reap (orphaned)")
