# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the off-loop targeted-build runner.

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
    classify_build_exit,
    ensure_build_dead,
    kill_build_pgroup,
    spawn_build,
)


def _action(cmd, **kw):
    base = dict(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe", build_command=tuple(cmd))
    base.update(kw)
    return TargetedBuildAction(**base)


def _wait_terminal(handle, *, timeout=10.0):
    rc = handle.proc.wait(timeout=timeout)
    return classify_build_exit(handle, rc)


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


def test_ensure_build_dead_reaps_a_running_group(tmp_path):
    """A live build is SIGKILLed and confirmed, so the sentinel can be dropped."""
    action = _action([sys.executable, "-c", "import time; time.sleep(600)"])
    handle = spawn_build(action, attempt_root=str(tmp_path / "a5"))
    assert handle.proc.poll() is None

    assert ensure_build_dead(handle) is True
    with pytest.raises(ProcessLookupError):
        os.killpg(handle.pgid, 0)


def test_ensure_build_dead_kills_a_child_that_ignores_sigterm(tmp_path):
    """No SIGTERM grace to sit out: the teardown goes straight to SIGKILL."""
    prog = (
        "import signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "print('ready', flush=True); time.sleep(600)"
    )
    action = _action([sys.executable, "-c", prog])
    root = tmp_path / "a6"
    handle = spawn_build(action, attempt_root=str(root))

    log_path = root / "build.log"
    for _ in range(100):
        if log_path.exists() and "ready" in log_path.read_text():
            break
        time.sleep(0.05)
    else:
        raise AssertionError("stubborn child never signalled readiness")

    assert ensure_build_dead(handle) is True
    with pytest.raises(ProcessLookupError):
        os.killpg(handle.pgid, 0)


def test_ensure_build_dead_on_an_already_exited_build(tmp_path):
    action = _action([sys.executable, "-c", "print('done')"])
    handle = spawn_build(action, attempt_root=str(tmp_path / "a7"))
    handle.proc.wait(timeout=10.0)

    assert ensure_build_dead(handle) is True


def test_kill_build_pgroup_noop_on_bad_pgid():
    # Must never raise on a dead/invalid pgid.
    kill_build_pgroup(0)
    kill_build_pgroup(-1)
    kill_build_pgroup(2**31, sig=signal.SIGKILL)
