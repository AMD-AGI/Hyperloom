# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Covers the LLM-transport stability env helper and the process-group kill in
``_run_subprocess`` that reaps a hung grandchild instead of orphaning it.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from hyperloom.orchestrator.roles._llm_stability_env import (
    DEFAULT_API_TIMEOUT_MS,
    apply_llm_stability_env,
)
from hyperloom.orchestrator.kernel.request_handlers import _run_subprocess


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
    import subprocess as _sp

    assert isinstance(excinfo.value, _sp.TimeoutExpired)

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
