# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Covers the LLM-transport stability env helper and the process-group kill in
``_run_subprocess`` that reaps a hung grandchild instead of orphaning it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

import pytest

from hyperloom.common import llm_attribution
from hyperloom.orchestrator.roles._llm_stability_env import (
    DEFAULT_API_TIMEOUT_MS,
    apply_llm_stability_env,
)
from hyperloom.orchestrator.kernel.request_handlers import _run_subprocess, _tool_label
from hyperloom.orchestrator.trace.task_progress import progress_scope

from .conftest import chatty_child, suppression_window_s


@pytest.fixture(autouse=True)
def _reset_published_phase():
    """The phase is a module global; leaving one set would label later tests."""
    yield
    llm_attribution.set_current_phase("")


def test_apply_llm_stability_env_sets_defaults():
    env: dict[str, str] = {}
    apply_llm_stability_env(env)
    # API_TIMEOUT_MS is opt-in: some clients treat it as a total request timeout
    # that can kill a legitimate long streaming response.
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
        ["python3", "-c", "print('hello-stdout')"],
        timeout_sec=30,
    )
    assert rc == 0
    assert "hello-stdout" in stdout


_ECHO_UNBUFFERED = ["python3", "-c", "import os; print(os.environ['PYTHONUNBUFFERED'])"]


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


_ECHO_TAGS = [
    "python3",
    "-c",
    "import os; print(os.environ.get('ANTHROPIC_CUSTOM_HEADERS', '(unset)'))",
]


async def test_run_subprocess_carries_the_phase_into_its_child(monkeypatch):
    """The kernel tools spend a phase's money and cannot name the phase.

    It is a module global of the orchestrator, so the child interpreter starts
    with none; without this the tools that do name themselves still reach the
    gateway unplaceable in the run.
    """
    monkeypatch.setenv(llm_attribution.ATTRIBUTION_ENV, "litellm")
    monkeypatch.setenv(llm_attribution.CLAW_SESSION_ID_ENV, "claw-abc")
    monkeypatch.setenv(llm_attribution.ANTHROPIC_CUSTOM_HEADERS_ENV, "Ocp-Apim-Subscription-Key: secret")
    llm_attribution.set_current_phase("KERNEL_AGENT")

    rc, stdout, _stderr = await _run_subprocess(_ECHO_TAGS, timeout_sec=30)

    assert rc == 0
    assert "phase=KERNEL_AGENT" in stdout
    assert "session=claw-abc" in stdout
    # The operator's own header has to survive being joined, not be replaced.
    assert "Ocp-Apim-Subscription-Key" in stdout


async def test_run_subprocess_leaves_an_unconfigured_deployment_alone(monkeypatch):
    """No gateway selected means the child sees the env it always did."""
    monkeypatch.delenv(llm_attribution.ATTRIBUTION_ENV, raising=False)
    monkeypatch.delenv(llm_attribution.ANTHROPIC_CUSTOM_HEADERS_ENV, raising=False)
    llm_attribution.set_current_phase("KERNEL_AGENT")

    rc, stdout, _stderr = await _run_subprocess(_ECHO_TAGS, timeout_sec=30)

    assert rc == 0
    assert stdout.strip() == "(unset)"


async def test_run_subprocess_counts_the_lines_its_child_emits(monkeypatch):
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
    await _run_subprocess(
        ["python3", "-c", "print('a')\nprint('b')"],
        timeout_sec=30,
    )

    assert counted == [2]


async def test_a_kernel_tool_keeps_reporting_while_its_child_works(monkeypatch, progress_cadence):
    """A trace analysis blocks for the better part of an hour behind one ``await``.

    Bounding the gap between notes is what a dropped liveness callback fails;
    asserting that a callback was passed is not. The child is faked rather than
    spawned so the timeline is the simulated one — that a real child's lines
    reach ``on_output`` is covered by
    ``test_run_subprocess_counts_the_lines_its_child_emits``.
    """
    from hyperloom.orchestrator.actions.executors import _subprocess_kill

    def _done(cmd, **_kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        _subprocess_kill,
        "run_with_session_kill",
        chatty_child(progress_cadence, _done, blocks_for_s=600.0, line_every_s=30.0),
    )

    with progress_scope(progress_cadence.sink()):
        rc, _stdout, _stderr = await _run_subprocess(["python3", "-c", "pass"], timeout_sec=30)

    assert rc == 0
    assert progress_cadence.widest_silence() < suppression_window_s()


def test_a_tool_is_named_after_the_script_it_runs():
    """``kernel_tool:tracelens_analysis`` is what an operator has to recognize."""
    assert _tool_label(["python3", "/opt/tools/tracelens_analysis.py", "--x"]) == "tracelens_analysis"
    assert _tool_label(["ls", "-l"]) == "ls"
    assert _tool_label([]) == "subprocess"


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
async def test_run_subprocess_kills_grandchild_on_timeout(tmp_path):
    """A timed-out child that spawned a long-lived grandchild must have the
    grandchild reaped too (process-group kill), not orphaned."""
    pidfile = tmp_path / "grandchild.pid"
    # Parent spawns a grandchild `sleep 300`, records its pid, then blocks. The
    # grandchild shares the parent's process group and must die with it on reap.
    script = (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen(['sleep', '300'])\n"
        "open(sys.argv[1], 'w').write(str(gc.pid))\n"
        "time.sleep(300)\n"
    )

    with pytest.raises(Exception) as excinfo:
        await _run_subprocess(
            ["python3", "-c", script, str(pidfile)],
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
