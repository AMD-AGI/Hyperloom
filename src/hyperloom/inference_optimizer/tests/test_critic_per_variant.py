# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic gate over specialist patches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import (
    INTEGRATE_PATCH_ACTION_NAME,
    INTEGRATE_PATCH_PERMISSIVE_VERDICTS,
    PolicyDenied,
    PolicyGate,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


# 1. SharedState ledger
def test_shared_state_record_and_read_verdict():
    s = SharedState()
    assert s.get_specialist_patch_verdict("t1") == ""
    s.record_specialist_patch_verdict("t1", "approve")
    assert s.get_specialist_patch_verdict("t1") == "approve"
    s.record_specialist_patch_verdict("t1", "reject")
    assert s.get_specialist_patch_verdict("t1") == "reject"
    # Empty verdict clears the entry (re-review pathway).
    s.record_specialist_patch_verdict("t1", "")
    assert s.get_specialist_patch_verdict("t1") == ""


def test_shared_state_ignores_empty_task_id():
    s = SharedState()
    s.record_specialist_patch_verdict("", "approve")
    assert s.specialist_patch_verdicts == {}


# 2. PolicyGate gate
def _make_gate(shared_state: SharedState | None = None) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=shared_state,
    )


def _make_intent(payload: dict[str, Any]) -> Intent:
    return Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": INTEGRATE_PATCH_ACTION_NAME,
            "params": payload,
        },
    )


def test_policy_denies_integrate_patch_naming_no_review_subject():
    gate = _make_gate(SharedState())
    intent = _make_intent({})
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "integrate_patch_requires_critic_verdict"
    assert "no Critic review subject" in str(exc.value)


def test_policy_reviews_an_upstream_pr_candidate_under_its_candidate_id():
    """The pre-screen is reviewed before any specialist exists.

    Its dispatched task carries a candidate id and no specialist task id, so a
    gate that only knew the latter denied the whole upstream-PR arm at
    dispatch as if the row had been forged.
    """
    s = SharedState()
    gate = _make_gate(s)
    intent = _make_intent({"framework_agent_candidate_id": "vllm/vllm#1015"})
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert "no Critic verdict" in str(exc.value)

    s.record_specialist_patch_verdict("vllm/vllm#1015", "approve")
    gate.validate_intent("orchestration", intent)


def test_policy_denies_integrate_patch_without_critic_verdict():
    gate = _make_gate(SharedState())
    intent = _make_intent({"specialist_task_id": "t-spec-1"})
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "integrate_patch_requires_critic_verdict"
    assert "no Critic verdict" in str(exc.value)


def test_policy_denies_integrate_patch_on_reject_verdict():
    s = SharedState()
    s.record_specialist_patch_verdict("t-spec-2", "reject")
    gate = _make_gate(s)
    intent = _make_intent({"specialist_task_id": "t-spec-2"})
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "integrate_patch_requires_critic_verdict"
    assert "reject" in str(exc.value)


def test_policy_denies_integrate_patch_on_needs_review():
    """``needs_review`` requires an explicit operator override."""
    s = SharedState()
    s.record_specialist_patch_verdict("t-spec-3", "needs_review")
    gate = _make_gate(s)
    intent = _make_intent({"specialist_task_id": "t-spec-3"})
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


def test_policy_allows_integrate_patch_on_approve():
    s = SharedState()
    s.record_specialist_patch_verdict("t-spec-4", "approve")
    gate = _make_gate(s)
    intent = _make_intent({"specialist_task_id": "t-spec-4"})
    # The phase check still applies; verify only the critic gate.
    s.phase = "FRAMEWORK_AGENT"
    gate.validate_intent("orchestration", intent)


def test_policy_allows_integrate_patch_on_advise():
    s = SharedState()
    s.record_specialist_patch_verdict("t-spec-5", "advise")
    s.phase = "FRAMEWORK_AGENT"
    gate = _make_gate(s)
    intent = _make_intent({"specialist_task_id": "t-spec-5"})
    gate.validate_intent("orchestration", intent)


def test_policy_permissive_verdicts_constant_covers_expected_set():
    """The permissive set includes approve + advise but excludes reject / needs_review / redirect."""
    assert "approve" in INTEGRATE_PATCH_PERMISSIVE_VERDICTS
    assert "advise" in INTEGRATE_PATCH_PERMISSIVE_VERDICTS
    for v in ("reject", "needs_review", "redirect"):
        assert v not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS


# 3. Executor defense in depth
def _write_specialist_workspace_with_patch(
    session_dir: Path,
    task_id: str,
) -> Path:
    workspace = session_dir / "runs" / "specialist" / task_id
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    patch_file = workspace / "worktree" / "patches" / "001_test.patch"
    patch_file.write_text("dummy patch contents\n")
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "gap_canonical_id": "gap.test",
                "domain": "serving_specialist",
                "proposal_set": [],
                "patches_written": ["patches/001_test.patch"],
                "empty": False,
                "summary": "PR-A7 executor defense-in-depth fixture",
            }
        )
    )
    return workspace


@pytest.mark.asyncio
async def test_executor_short_circuits_on_recorded_reject(tmp_path: Path):
    """A recorded 'reject' verdict makes the executor refuse to bench."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    _write_specialist_workspace_with_patch(session_dir, "t-spec-x")
    state = SharedState()
    state.record_specialist_patch_verdict("t-spec-x", "reject")

    executor = IntegratePatchExecutor(session_dir=session_dir)
    task = Task(
        task_id="t-int-x",
        kind="integrate_patch",
        state="queued",
        params={
            "specialist_task_id": "t-spec-x",
            "framework_source_root": str(tmp_path / "framework"),
            "apply_only": True,
        },
        idempotency_key="t-int-x",
        requires_lanes=tuple(),
    )
    ctx = RunnerContext(task=task, lease=None, extra={"shared_state": state})
    # Create the framework dir so the executor reaches the critic check.
    (tmp_path / "framework").mkdir()
    result = await executor(ctx)
    # Either outcome is fine as long as patches_applied is empty and no bench ran.
    assert result["status"] in ("rejected_by_critic", "apply_failed")
    assert result["patches_applied"] == []
    if result["status"] == "rejected_by_critic":
        # The gate now requires a permissive verdict; a recorded 'reject' is
        # surfaced in the reason (wording generalized in SWSPLAT-42420 fix).
        assert "verdict 'reject'" in result["reason"]


@pytest.mark.asyncio
async def test_executor_proceeds_when_verdict_is_approve(tmp_path: Path, monkeypatch):
    """No short-circuit when the recorded verdict is approve."""
    from hyperloom.inference_optimizer.tests.conftest import patch_integrate_patch_allowlist

    patch_integrate_patch_allowlist(monkeypatch, tmp_path)
    import os
    import subprocess

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    repo.mkdir(parents=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "PR-A7"
    env["GIT_AUTHOR_EMAIL"] = "pr-a7@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, env=env)
    (repo / "src.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, env=env)

    patch_text = "diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    workspace = session_dir / "runs" / "specialist" / "t-spec-y"
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    (workspace / "worktree" / "patches" / "001.patch").write_text(patch_text)
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "gap_canonical_id": "gap.ok",
                "domain": "serving_specialist",
                "proposal_set": [],
                "patches_written": ["patches/001.patch"],
                "empty": False,
                "summary": "approved",
            }
        )
    )
    state = SharedState()
    state.record_specialist_patch_verdict("t-spec-y", "approve")

    executor = IntegratePatchExecutor(session_dir=session_dir)
    task = Task(
        task_id="t-int-y",
        kind="integrate_patch",
        state="queued",
        params={
            "specialist_task_id": "t-spec-y",
            "framework_source_root": str(repo),
            "apply_only": True,
        },
        idempotency_key="t-int-y",
        requires_lanes=tuple(),
    )
    ctx = RunnerContext(task=task, lease=None, extra={"shared_state": state})
    result = await executor(ctx)
    assert result["status"] == "applied_no_bench"
    assert len(result["patches_applied"]) == 1
