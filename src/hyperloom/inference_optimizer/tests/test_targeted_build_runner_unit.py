# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the off-loop targeted-build runner (S2).

Uses real short-lived subprocesses (fake ``build_command`` argv) plus an
injectable monotonic clock to exercise the two-phase timeout kill deterministically.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction
from hyperloom.orchestrator.framework.targeted_build import (
    default_budget_sec,
    kill_build_pgroup,
    poll_build,
    spawn_build,
)


def _action(cmd, **kw):
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", build_command=tuple(cmd))
    base.update(kw)
    return TargetedBuildAction(**base)


def _wait_terminal(handle, *, now=None, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = poll_build(handle, now=now) if now else poll_build(handle)
        if res is not None:
            return res
        time.sleep(0.05)
    raise AssertionError("build did not reach a terminal result in time")


def test_build_success(tmp_path):
    action = _action([sys.executable, "-c", "print('ok')"])
    handle = spawn_build(action, attempt_root=str(tmp_path / "a1"))
    res = _wait_terminal(handle)
    assert res.ok is True
    assert res.failure_class == "ok"
    assert res.attempt_root == str(tmp_path / "a1")
    # The promoted runtime carries the per-attempt JIT dir.
    assert res.runtime.runtime_env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"].endswith("aiter_jit")


def test_build_nonzero_is_compile_error(tmp_path):
    action = _action([sys.executable, "-c", "import sys; sys.exit(3)"])
    handle = spawn_build(action, attempt_root=str(tmp_path / "a2"))
    res = _wait_terminal(handle)
    assert res.ok is False
    assert res.failure_class == "compile_error"
    assert "3" in res.error


def test_per_attempt_jit_dir_env_and_log(tmp_path):
    action = _action(
        [sys.executable, "-c", "import os; print(os.environ['INFERENCE_OPTIMIZER_AITER_JIT_DIR'])"]
    )
    root = tmp_path / "a3"
    handle = spawn_build(action, attempt_root=str(root))
    _wait_terminal(handle)
    assert handle.aiter_jit_dir == str(root / "aiter_jit")
    assert (root / "aiter_jit").is_dir()
    log_text = (root / "build.log").read_text()
    assert str(root / "aiter_jit") in log_text


def test_empty_build_command_raises(tmp_path):
    action = _action([])
    with pytest.raises(ValueError):
        spawn_build(action, attempt_root=str(tmp_path / "a4"))


def test_timeout_two_phase_kill(tmp_path):
    """A long build past its deadline gets SIGTERM then SIGKILL; result=timeout."""
    action = _action([sys.executable, "-c", "import time; time.sleep(600)"])
    handle = spawn_build(action, attempt_root=str(tmp_path / "a5"))
    base = handle.deadline - default_budget_sec("aiter")

    # Before the deadline: still running.
    assert poll_build(handle, now=lambda: base + 1.0) is None
    assert handle.sigterm_at == 0.0

    # At the deadline: SIGTERM fired, sigterm_at recorded, still not terminal.
    at_deadline = handle.deadline
    assert poll_build(handle, now=lambda: at_deadline) is None
    assert handle.sigterm_at == at_deadline

    # SIGTERM alone should stop a plain `sleep`; wait for the terminal result.
    res = _wait_terminal(handle, now=lambda: at_deadline + 0.1)
    assert res.ok is False
    assert res.failure_class == "timeout"


def test_timeout_sigkill_escalation_kills_stubborn_child(tmp_path):
    """A process that ignores SIGTERM is SIGKILLed after the grace window."""
    # Install SIG_IGN, flush a readiness marker, THEN sleep — so the test only
    # triggers the deadline once the handler is provably in place (no race).
    prog = (
        "import signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "print('ready', flush=True); time.sleep(600)"
    )
    action = _action([sys.executable, "-c", prog])
    root = tmp_path / "a6"
    handle = spawn_build(action, attempt_root=str(root))
    at_deadline = handle.deadline

    # Wait until the child has installed the SIGTERM-ignore handler.
    log_path = root / "build.log"
    for _ in range(100):
        if log_path.exists() and "ready" in log_path.read_text():
            break
        time.sleep(0.05)
    else:
        raise AssertionError("stubborn child never signalled readiness")

    # Deadline -> SIGTERM (ignored by the child).
    assert poll_build(handle, now=lambda: at_deadline) is None
    time.sleep(0.3)
    assert handle.proc.poll() is None  # survived SIGTERM

    # Past the grace window -> SIGKILL escalation.
    poll_build(handle, now=lambda: at_deadline + 6.0)
    res = _wait_terminal(handle, now=lambda: at_deadline + 6.0)
    assert res.failure_class == "timeout"
    # Process group is gone.
    with pytest.raises(ProcessLookupError):
        os.killpg(handle.pgid, 0)


def test_kill_build_pgroup_noop_on_bad_pgid():
    # Must never raise on a dead/invalid pgid.
    kill_build_pgroup(0)
    kill_build_pgroup(-1)
    kill_build_pgroup(2**31, sig=signal.SIGKILL)
