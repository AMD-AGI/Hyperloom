# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch-time PolicyGate replay for queued task rows (F013 plan B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session import paths
from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry


def _gate(tmp_path: Path, monkeypatch, *, strict_phase: bool = False) -> tuple[PolicyGate, Path]:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    state = SharedState.load_or_init(sd)
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=state,
        strict_paths=True,
        strict_phase=strict_phase,
    )
    return gate, sd


def _runner_with_policy(tmp_path: Path, monkeypatch, *, shared_state: object | None = None) -> SubAgentRunner:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "coord.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tasks = TaskRegistry(db)
    state = shared_state if shared_state is not None else SharedState.load_or_init(sd)
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=state,
        strict_paths=True,
    )
    return SubAgentRunner(
        locks,
        tasks,
        session_dir=sd,
        shared_state=state,
        policy=gate,
    )


@pytest.mark.asyncio
async def test_dispatched_integrate_patch_without_critic_verdict_fails(tmp_path, monkeypatch):
    """Forged queued integrate_patch rows must fail dispatch policy replay."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("integrate_patch", _stub)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={
            "specialist_task_id": "evil0",
            "apply_only": True,
        },
        idempotency_key="forged-integrate",
    )
    res = await sub.run_task(task)
    assert res.state == "failed"
    assert "no Critic verdict on record" in (res.error or "")
    assert executed["ran"] is False
    updated = await sub.tasks.get(task.task_id)
    assert updated.state == "cancelled"
    assert updated.attempts == 0


@pytest.mark.asyncio
async def test_dispatched_integrate_patch_outside_allowlist_fails(tmp_path, monkeypatch):
    """framework_source_root outside the source allowlist is denied at dispatch."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.record_specialist_patch_verdict("evil0", "approve")

    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("integrate_patch", _stub)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={
            "specialist_task_id": "evil0",
            "framework_source_root": "/root",
            "apply_only": True,
        },
        idempotency_key="forged-root-override",
    )
    res = await sub.run_task(task)
    assert res.state == "failed"
    assert "framework_source_root'='/root'" in (res.error or "")
    assert executed["ran"] is False


@pytest.mark.asyncio
async def test_dispatched_internal_roofline_passes_delegate_gates(tmp_path, monkeypatch):
    """Coordinator-internal actions skip LLM delegate gates but still dispatch."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("roofline", _stub)
    task = await sub.tasks.create(
        kind="roofline",
        params={"reason": "prelude_bootstrap"},
        idempotency_key="internal-roofline",
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_dispatched_recover_rejected_for_orchestration_source(tmp_path, monkeypatch):
    """Robustness-only delegates cannot be forged as orchestration queued tasks."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("recover", _stub)
    task = await sub.tasks.create(
        kind="recover",
        params={"reason": "forged", "evidence": {"kind": "test"}},
        idempotency_key="forged-recover",
    )
    res = await sub.run_task(task)
    assert res.state == "failed"
    assert "cannot delegate action='recover'" in (res.error or "")
    assert executed["ran"] is False


def test_validate_dispatched_task_unit_integrate_patch_gate(tmp_path):
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=tmp_path,
        strict_paths=True,
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task(
            "integrate_patch",
            {"specialist_task_id": "x", "framework_source_root": "/root"},
        )
    assert exc.value.rule == "source_file_not_allowlisted"


def test_validate_dispatched_task_accepts_integrate_patch_with_verdict(tmp_path, monkeypatch):
    gate, _sd = _gate(tmp_path, monkeypatch)
    state = gate.shared_state
    assert isinstance(state, SharedState)
    state.record_specialist_patch_verdict("spec-1", "approve")
    gate.validate_dispatched_task(
        "integrate_patch",
        {"specialist_task_id": "spec-1", "apply_only": True},
    )


def test_validate_dispatched_task_rejects_missing_specialist_task_id(tmp_path, monkeypatch):
    gate, _sd = _gate(tmp_path, monkeypatch)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task("integrate_patch", {"apply_only": True})
    assert exc.value.rule == "integrate_patch_requires_critic_verdict"


def test_validate_dispatched_task_internal_profile_skips_delegate_body(tmp_path, monkeypatch):
    gate, _sd = _gate(tmp_path, monkeypatch)
    gate.validate_dispatched_task("profile", {"reason": "watermark_refresh"})


def test_validate_dispatched_task_skips_phase_incompatible(tmp_path, monkeypatch):
    gate, _sd = _gate(tmp_path, monkeypatch, strict_phase=True)
    state = gate.shared_state
    assert isinstance(state, SharedState)
    state.phase = "CLOSE"
    gate.validate_dispatched_task("baseline", {})

    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "baseline", "params": {}},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "phase_incompatible"


@pytest.mark.asyncio
async def test_dispatched_integrate_patch_with_verdict_passes(tmp_path, monkeypatch):
    sub = _runner_with_policy(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.record_specialist_patch_verdict("spec-ok", "advise")
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("integrate_patch", _stub)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-ok", "apply_only": True},
        idempotency_key="legit-integrate",
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_dispatched_internal_framework_agent_passes(tmp_path, monkeypatch):
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("framework_agent", _stub)
    task = await sub.tasks.create(
        kind="framework_agent",
        params={"candidate": {"repo": "x/y", "pr_number": 1}},
        idempotency_key="internal-framework-agent",
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_dispatched_kernel_owned_action_rejected(tmp_path, monkeypatch):
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("kernel_opt", _stub)
    task = await sub.tasks.create(
        kind="kernel_opt",
        params={},
        idempotency_key="forged-kernel-opt",
    )
    res = await sub.run_task(task)
    assert res.state == "failed"
    assert "owned by the Kernel-agent" in (res.error or "")
    assert executed["ran"] is False


@pytest.mark.asyncio
async def test_runner_without_policy_skips_dispatch_validation(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "coord.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tasks = TaskRegistry(db)
    sub = SubAgentRunner(locks, tasks, session_dir=sd, policy=None)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("integrate_patch", _stub)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "no-gate", "apply_only": True},
        idempotency_key="no-policy-gate",
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True
