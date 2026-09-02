# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch-time PolicyGate replay for queued task rows (F013 plan B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session import paths
from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry


def _warm_replay_dispatch_params(
    session_dir: Path,
    framework_root: Path,
    *,
    patch_target: str = "vllm/v1/attention/ops/prefix_prefill.py",
    patch_path: Path | None = None,
) -> tuple[dict, Path, Path]:
    target = framework_root / "vllm/v1/attention/ops/prefix_prefill.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("original\n", encoding="utf-8")
    if patch_path is None:
        patch_path = session_dir / "runtime/remote_recipe/files/kernel/rewrite/patches/warm.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(
        "\n".join(
            [
                f"diff --git a/{patch_target} b/{patch_target}",
                f"--- a/{patch_target}",
                f"+++ b/{patch_target}",
                "@@ -1 +1 @@",
                "-original",
                "+patched",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return (
        {
            "warm_kernel_plan": [
                {
                    "resolved_patch_targets": [str(target)],
                    "patch_path": str(patch_path),
                }
            ],
            "warm_kernel_apply_results": [{"target_file": str(target)}],
        },
        target,
        patch_path,
    )


def _gate(tmp_path: Path, monkeypatch) -> tuple[PolicyGate, Path]:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    state = SharedState.load_or_init(sd)
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=state,
        strict_paths=True,
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
async def test_dispatched_internal_conc_sweep_passes_auto_enqueue_singleton(tmp_path, monkeypatch):
    """Auto-enqueued conc_sweep must not re-run LLM singleton validation at dispatch."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.phase = "SWEEP"
    state.phase_history = [
        {
            "to_phase": "SWEEP",
            "evidence": {"auto_conc_sweep_task_id": "internal-conc_sweep-phase_entry"},
        }
    ]
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("conc_sweep", _stub)
    task = await sub.tasks.create(
        kind="conc_sweep",
        params={"reason": "phase_entry"},
        idempotency_key="internal-conc-sweep",
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_dispatched_recover_executes_when_queued(tmp_path, monkeypatch):
    """A recover task that passed ingress validation must reach its executor at dispatch."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("recover", _stub)
    task = await sub.tasks.create(
        kind="recover",
        params={"reason": "gpu_memory_leaked", "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]}},
        idempotency_key="queued-recover",
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


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
    assert exc.value.rule == "source_file_outside_trusted_scope"


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


def test_dispatched_warm_replay_accepts_verified_framework_target(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    params, _target, _patch = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    gate.validate_dispatched_task("replay_warm_recipe", params)


def test_warm_replay_target_exception_is_not_shared_with_other_actions(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    params, _target, _patch = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task("profile", params)
    assert exc.value.rule == "path_outside_session_dir"


def test_dispatched_warm_replay_rejects_patch_outside_kb_download(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    outside_patch = tmp_path.parent / f"{tmp_path.name}-outside.patch"
    params, _target, _patch = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
        patch_path=outside_patch,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task("replay_warm_recipe", params)
    assert exc.value.rule == "warm_replay_patch_outside_kb_download"


def test_dispatched_warm_replay_rejects_patch_target_mismatch(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    params, _target, _patch = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
        patch_target="vllm/v1/attention/ops/different.py",
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task("replay_warm_recipe", params)
    assert exc.value.rule == "warm_replay_patch_target_mismatch"


def test_dispatched_warm_replay_rejects_undeclared_extra_patch_target(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    params, _target, patch_path = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
    )
    with patch_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "diff --git a/vllm/other.py b/vllm/other.py",
                    "--- a/vllm/other.py",
                    "+++ b/vllm/other.py",
                    "@@ -1 +1 @@",
                    "-old",
                    "+new",
                    "",
                ]
            )
        )
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task("replay_warm_recipe", params)
    assert exc.value.rule == "warm_replay_patch_target_mismatch"


def test_dispatched_warm_replay_accepts_declared_multi_file_targets(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    params, target, patch_path = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
    )
    added = framework_root / "vllm/v1/attention/ops/new_fused_ops.py"
    with patch_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "diff --git a/vllm/v1/attention/ops/new_fused_ops.py b/vllm/v1/attention/ops/new_fused_ops.py",
                    "--- /dev/null",
                    "+++ b/vllm/v1/attention/ops/new_fused_ops.py",
                    "@@ -0,0 +1 @@",
                    "+new",
                    "",
                ]
            )
        )
    params["warm_kernel_plan"][0]["resolved_patch_targets"] = [
        str(target),
        str(added),
    ]
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    gate.validate_dispatched_task("replay_warm_recipe", params)


def test_dispatched_warm_replay_rejects_framework_symlink_escape(
    tmp_path,
    monkeypatch,
):
    gate, session_dir = _gate(tmp_path, monkeypatch)
    framework_root = tmp_path.parent / f"{tmp_path.name}-framework"
    escaped_root = tmp_path.parent / f"{tmp_path.name}-escaped"
    escaped_target = escaped_root / "v1/attention/ops/prefix_prefill.py"
    escaped_target.parent.mkdir(parents=True, exist_ok=True)
    escaped_target.write_text("original\n", encoding="utf-8")
    framework_root.mkdir(parents=True, exist_ok=True)
    (framework_root / "vllm").symlink_to(escaped_root, target_is_directory=True)
    params, _target, _patch = _warm_replay_dispatch_params(
        session_dir,
        framework_root,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.resolve_session_framework_root",
        lambda: str(framework_root),
    )

    with pytest.raises(PolicyDenied) as exc:
        gate.validate_dispatched_task("replay_warm_recipe", params)
    assert exc.value.rule == "warm_replay_target_outside_framework_roots"


@pytest.mark.asyncio
async def test_dispatched_tracked_enablement_revalidation_bypasses_baseline_singleton(tmp_path, monkeypatch):
    sub = _runner_with_policy(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.baseline_tput = 1000.0
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("baseline", _stub)
    task = await sub.tasks.create(
        kind="baseline",
        params={"reason": "enablement_eval_revalidation"},
        idempotency_key="enablement-revalidation",
    )
    state.enablement.revalidation_task_id = task.task_id

    result = await sub.run_task(task)

    assert result.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio




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
async def test_dispatched_integrate_patch_resume_with_persisted_verdict_passes(tmp_path, monkeypatch):
    """Queued integrate_patch survives resume when Critic verdicts round-trip in state.json."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db_path = tmp_path / "coord.db"

    state = SharedState.load_or_init(sd)
    state.record_specialist_patch_verdict("spec-resume", "approve")
    state.save(sd)

    db = SqliteConnection(db_path)
    tasks = TaskRegistry(db)
    task = await tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-resume", "apply_only": True},
        idempotency_key="resume-integrate",
    )
    assert task.state == "queued"
    task_id = task.task_id

    resumed_state = SharedState.load_or_init(sd)
    assert resumed_state.get_specialist_patch_verdict("spec-resume") == "approve"

    db2 = SqliteConnection(db_path)
    locks2 = ResourceLockManager(SqliteLeaseBackend(db2))
    tasks2 = TaskRegistry(db2)
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=resumed_state,
        strict_paths=True,
    )
    sub = SubAgentRunner(
        locks2,
        tasks2,
        session_dir=sd,
        shared_state=resumed_state,
        policy=gate,
    )
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("integrate_patch", _stub)
    queued = await tasks2.get(task_id)
    assert queued.state == "queued"

    res = await sub.run_task(queued)
    assert res.state == "succeeded"
    assert executed["ran"] is True
    updated = await tasks2.get(task_id)
    assert updated.state == "succeeded"


class _ReconcileCoordStub:
    """Minimal coordinator shell for DispatcherCollaborator reconcile tests."""

    def __init__(self, *, sub: SubAgentRunner, tasks: TaskRegistry, shared_state: SharedState) -> None:
        self.sub = sub
        self.tasks = tasks
        self.shared_state = shared_state


@pytest.mark.asyncio
async def test_reconcile_cancelled_integrate_patch_when_verdict_restored(tmp_path, monkeypatch):
    """Dispatch-time policy cancel is re-queued once Critic verdicts are restored."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-reconcile", "apply_only": True},
        idempotency_key="approved-prop-reconcile",
    )
    res = await sub.run_task(task)
    assert res.state == "failed"
    cancelled = await sub.tasks.get(task.task_id)
    assert cancelled.state == "cancelled"

    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.record_specialist_patch_verdict("spec-reconcile", "approve")
    assert sub.policy is not None
    sub.policy.shared_state = state

    disp = DispatcherCollaborator(_ReconcileCoordStub(sub=sub, tasks=sub.tasks, shared_state=state))
    created = await disp._reconcile_cancelled_policy_denied_integrate_tasks()
    assert len(created) == 1
    queued = await sub.tasks.queued()
    assert len(queued) == 1
    assert queued[0].idempotency_key == "approved-prop-reconcile-reconcile1"
    assert queued[0].params.get("specialist_task_id") == "spec-reconcile"


@pytest.mark.asyncio
async def test_reconcile_does_not_spawn_second_child_after_first_succeeds(tmp_path, monkeypatch):
    """A succeeded reconcile child must not trigger reconcile2 on later pump passes."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-once", "apply_only": True},
        idempotency_key="approved-prop-once",
    )
    await sub.run_task(task)

    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.record_specialist_patch_verdict("spec-once", "approve")
    assert sub.policy is not None
    sub.policy.shared_state = state

    disp = DispatcherCollaborator(_ReconcileCoordStub(sub=sub, tasks=sub.tasks, shared_state=state))
    created = await disp._reconcile_cancelled_policy_denied_integrate_tasks()
    assert len(created) == 1
    child = await sub.tasks.get(created[0])
    await sub.tasks.transition(child.task_id, "running")
    await sub.tasks.transition(child.task_id, "succeeded", evidence={"status": "ok"})

    assert await disp._reconcile_cancelled_policy_denied_integrate_tasks() == []
    rows = await sub.tasks.db.fetchall(
        "SELECT idempotency_key FROM tasks WHERE kind='integrate_patch' AND idempotency_key LIKE ?",
        ("approved-prop-once-reconcile%",),
    )
    assert len(rows) == 1
    assert rows[0]["idempotency_key"] == "approved-prop-once-reconcile1"


@pytest.mark.asyncio
async def test_reconcile_skips_when_verdict_still_missing(tmp_path, monkeypatch):
    sub = _runner_with_policy(tmp_path, monkeypatch)
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": "spec-no-verdict", "apply_only": True},
        idempotency_key="approved-prop-no-verdict",
    )
    await sub.run_task(task)
    disp = DispatcherCollaborator(
        _ReconcileCoordStub(
            sub=sub,
            tasks=sub.tasks,
            shared_state=sub.shared_state,
        )
    )
    assert await disp._reconcile_cancelled_policy_denied_integrate_tasks() == []
    assert await sub.tasks.queued() == []


@pytest.mark.asyncio
async def test_reconcile_skips_non_critic_policy_denials(tmp_path, monkeypatch):
    sub = _runner_with_policy(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.record_specialist_patch_verdict("spec-root", "approve")
    task = await sub.tasks.create(
        kind="integrate_patch",
        params={
            "specialist_task_id": "spec-root",
            "framework_source_root": "/root",
            "apply_only": True,
        },
        idempotency_key="approved-prop-bad-root",
    )
    await sub.run_task(task)
    disp = DispatcherCollaborator(_ReconcileCoordStub(sub=sub, tasks=sub.tasks, shared_state=state))
    assert await disp._reconcile_cancelled_policy_denied_integrate_tasks() == []


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


@pytest.mark.asyncio
async def test_killed_running_task_keeps_its_result(tmp_path, monkeypatch):
    """A task cancelled mid-flight still returns its executor result."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    task = await sub.tasks.create(
        kind="report",
        params={},
        idempotency_key="k-kill-midflight",
    )

    async def _stub(ctx):
        # Simulate a kill landing while the executor is still running.
        await sub.tasks.transition(ctx.task.task_id, "cancelled", evidence={"reason": "killed"})
        return {"produced": "work"}

    sub.register_executor("report", _stub)

    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result == {"produced": "work"}
    updated = await sub.tasks.get(task.task_id)
    assert updated.state == "cancelled"






