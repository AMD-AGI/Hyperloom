# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resume recovery for an in-flight targeted build (S2).

A detached compile cannot survive a coordinator restart; the resume pass must
kill the orphaned process group, GC the attempt dir, sweep its jit locks, fail
the row, and clear the sentinel.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _silent_plan() -> ScriptedPlan:
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

    return ScriptedPlan(
        turns=[], default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
    )


def _build_backends() -> dict[str, Backend]:
    return {
        name: MockBackend(_silent_plan(), name=name)
        for name in ("orchestration", "kernel_agent", "critic", "robustness")
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


@pytest.mark.asyncio
async def test_resume_kills_orphan_and_clears_sentinel(coord: Coordinator):
    coord._resumed_from["is_resume"] = True
    # A real detached process group standing in for the orphaned build.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True
    )
    pgid = os.getpgid(proc.pid)

    attempt_root = Path(coord.session_dir) / "enablement" / "builds" / "t-orphan"
    jit_dir = attempt_root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    stale_lock = jit_dir / "lock"
    stale_lock.write_text("")
    # Age the lock past the mtime gate so the dead-path sweep removes it.
    old = time.time() - 3600
    os.utime(stale_lock, (old, old))

    # Enqueue a matching row and mark it running (the crash left it in flight).
    from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="fp4_moe")
    task, _ = await coord.tasks.create_or_return_existing(
        kind="targeted_build", params=action.to_state(), idempotency_key="k-orphan", requires_lanes=["build_lane"]
    )
    await coord.tasks.transition(task.task_id, "running")

    coord.shared_state.pending_targeted_build = {
        "task_id": task.task_id,
        "pid": proc.pid,
        "pgid": pgid,
        "attempt_root": str(attempt_root),
        "aiter_jit_dir": str(jit_dir),
    }

    report = await coord._resume_consistency_pass()

    fix = next(
        f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "reclaimed_pending_targeted_build"
    )
    assert fix["task_id"] == task.task_id
    # Sentinel cleared, failure recorded, row failed.
    assert coord.shared_state.pending_targeted_build == {}
    assert coord.shared_state.enablement_last_build_failure["failure_class"] == "timeout"
    reclaimed = await coord.tasks.get(task.task_id)
    assert reclaimed.state == "failed"
    # Attempt dir removed and stale jit lock swept.
    assert not attempt_root.exists()
    # Orphan process group killed.
    for _ in range(40):
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


@pytest.mark.asyncio
async def test_resume_no_pending_targeted_build_is_noop(coord: Coordinator):
    coord._resumed_from["is_resume"] = True
    coord.shared_state.pending_targeted_build = {}
    report = await coord._resume_consistency_pass()
    assert not any(
        isinstance(f, dict) and f.get("kind") == "reclaimed_pending_targeted_build"
        for f in report["fixes"]
    )
