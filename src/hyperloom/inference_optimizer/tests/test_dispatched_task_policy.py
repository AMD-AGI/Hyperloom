# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatch-time PolicyGate replay for queued task rows (F013 plan B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session import paths
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.policy.projection import ResourceFacts
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec
from hyperloom.orchestrator.tests.mock_backend import MockBackend, ScriptedPlan


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
        resources=ResourceFacts(),
    )
    return gate, sd


def _project(gate: PolicyGate, **fields) -> None:
    """Re-read the gate's resource facts, as a coordinator tick would.

    Args:
        gate: The gate whose facts are re-read.
        fields: Facts no SharedState holds -- a bring-up round is the round
            store's, not the session's.
    """
    gate.resources.update(gate.shared_state)
    for name, value in fields.items():
        setattr(gate.resources, name, value)


_ORIGIN = {"phase": "EXPLORE", "macro_cycle": 0, "tick": 0}


def _runner_with_policy(tmp_path: Path, monkeypatch, *, shared_state: object | None = None) -> SubAgentRunner:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "coord.db")
    locks = SqliteLeaseBackend(db)
    tasks = TaskRegistry(db, dispatch_origin_provider=lambda: _ORIGIN)
    state = shared_state if shared_state is not None else SharedState.load_or_init(sd)
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=state,
        resources=ResourceFacts(),
    )
    sub = SubAgentRunner(locks, tasks, session_dir=sd)
    sub.shared_state = state
    sub.policy = gate
    return sub


@pytest.mark.asyncio
async def test_a_queued_row_the_gate_refuses_at_dispatch_is_cancelled_and_never_runs(tmp_path, monkeypatch):
    """A queued row is replayed through the gate before it runs: a baseline another round's machine denies is cancelled."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("baseline", _stub)
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(kind="baseline", params={}, idempotency_key="second-bring-up"),
    )
    _project(sub.policy, excluding_round_id="round-7", excluding_round_holder="spec-abc")

    res = await sub.run_task(task)

    assert res.state == "cancelled"
    assert res.error_class == "policy_enablement_round_in_flight"
    assert "spec-abc" in (res.error or "")
    assert executed["ran"] is False
    updated = await sub.tasks.get(task.task_id)
    assert updated.state == "cancelled"
    terminal = updated.history[-1]["evidence"]
    assert terminal["rule"] == "enablement_round_in_flight"
    assert terminal["outcome"]["error_class"] == "policy_enablement_round_in_flight"


@pytest.mark.asyncio
async def test_dispatch_denied_baseline_is_not_charged_to_baseline_failures(session_dir, monkeypatch):
    """A baseline refused at dispatch measured nothing, so repeated refusals cannot stop the run as baseline_failed."""
    silent = ScriptedPlan(
        turns=[], default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
    )
    coord = Coordinator(session_dir, backends={"orchestration": MockBackend(silent, name="orchestration")})
    coord.shared_state.phase = "PRELUDE"

    def _deny(kind, params, *, task_id=""):
        raise PolicyDenied("bring-up round holds the machine", rule="enablement_round_in_flight")

    monkeypatch.setattr(coord.policy, "validate_dispatched_task", _deny)
    for i in range(3):
        task, _ = await coord.tasks.create_or_return_existing(
            TaskSpec(kind="baseline", params={}, idempotency_key=f"denied-baseline-{i}")
        )
        result = await coord.sub.run_task(task)
        await coord.dispatcher.reap_dispatched_task(task, result)
        assert result.state == "cancelled"
        assert result.error_class == "policy_enablement_round_in_flight"
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"

    state = coord.shared_state
    assert state.baseline_failure_streak == 0
    assert state.baseline_total_failures == 0
    assert state.stop_reason == ""
    assert not state.last_action_failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "params", "state_fields"),
    [
        pytest.param("roofline", {"reason": "prelude_bootstrap"}, {}, id="internal_roofline"),
        pytest.param(
            "recover",
            {"reason": "gpu_memory_leaked", "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]}},
            {},
            id="recover_admitted_at_ingress",
        ),
        pytest.param(
            "framework_agent", {"candidate": {"repo": "x/y", "pr_number": 1}}, {}, id="internal_framework_agent"
        ),
        pytest.param(
            "integrate_patch", {"specialist_task_id": "spec-ok", "apply_only": True}, {}, id="integrate_patch"
        ),
    ],
)
async def test_dispatched_admitted_row_passes_delegate_gates(tmp_path, monkeypatch, kind, params, state_fields):
    """A queued row the Coordinator or ingress already admitted reaches its executor past the delegate gates."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    for name, value in state_fields.items():
        setattr(sub.shared_state, name, value)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor(kind, _stub)
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(kind=kind, params=params, idempotency_key=f"queued-{kind}"),
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


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
async def test_dispatched_revalidation_holding_its_own_round_runs(tmp_path, monkeypatch):
    """The revalidation baseline executes rather than being cancelled at dispatch.

    A revalidation only exists once a KEEP measured throughput, so the anchor
    arm fires on exactly the session state a revalidation runs in, and a
    denial there cancels the row instead of deferring it. What admits it is the
    round it opened: the store names this task the holder, and the holder is not
    refused by its own acquire.
    """
    sub = _runner_with_policy(tmp_path, monkeypatch)
    state = sub.shared_state
    assert isinstance(state, SharedState)
    state.baseline_tput = 1000.0
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("baseline", _stub)
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(
            kind="baseline",
            params={"reason": "enablement_eval_revalidation"},
            idempotency_key="enablement-revalidation",
        ),
    )
    _project(sub.policy, excluding_round_id="revalidation-1", excluding_round_holder=task.task_id)

    result = await sub.run_task(task)

    assert result.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_runner_without_policy_skips_dispatch_validation(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "coord.db")
    locks = SqliteLeaseBackend(db)
    tasks = TaskRegistry(db)
    sub = SubAgentRunner(locks, tasks, session_dir=sd)
    executed = {"ran": False}

    async def _stub(_ctx) -> dict:
        executed["ran"] = True
        return {"status": "ok"}

    sub.register_executor("integrate_patch", _stub)
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(
            kind="integrate_patch",
            params={"specialist_task_id": "no-gate", "apply_only": True},
            idempotency_key="no-policy-gate",
        ),
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert executed["ran"] is True


@pytest.mark.asyncio
async def test_killed_running_task_keeps_its_result(tmp_path, monkeypatch):
    """A task cancelled mid-flight still returns its executor result."""
    sub = _runner_with_policy(tmp_path, monkeypatch)
    task, _ = await sub.tasks.create_or_return_existing(
        TaskSpec(
            kind="report",
            params={},
            idempotency_key="k-kill-midflight",
        ),
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


def test_a_round_holding_the_machine_denies_baseline(tmp_path, monkeypatch):
    gate, _ = _gate(tmp_path, monkeypatch)
    _project(gate, excluding_round_id="round-7", excluding_round_holder="spec-abc")
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "baseline", "params": {}},
    )
    with pytest.raises(PolicyDenied) as exc_info:
        gate.validate_intent("orchestration", intent)
    assert exc_info.value.rule == "enablement_round_in_flight"
    assert "spec-abc" in str(exc_info.value)


def test_no_round_in_flight_allows_baseline(tmp_path, monkeypatch):
    """Nothing holds the machine, so the rule has nothing to refuse."""
    gate, _ = _gate(tmp_path, monkeypatch)
    _project(gate)
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "baseline", "params": {}},
    )
    gate.validate_intent("orchestration", intent)


def test_an_established_anchor_alone_does_not_refuse_a_baseline(tmp_path, monkeypatch):
    """Re-measuring a reference the run already has is a workflow opinion.

    The rule is about the cards and ports a live round is using, not about how
    many baselines a session may take, so a session that already has its figure
    can still take another one while nothing holds the machine.
    """
    gate, _ = _gate(tmp_path, monkeypatch)
    gate.shared_state.baseline_tput = 1000.0
    _project(gate)

    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "baseline", "params": {}},
        ),
    )
