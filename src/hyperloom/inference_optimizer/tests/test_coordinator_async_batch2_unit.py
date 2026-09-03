# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Batch 2 coverage for Coordinator: synchronous context readers, the
no-progress circuit-breaker signal, resume replay, orchestration-conversation
reset, and lifecycle teardown (stop / Recipe KB T4 safety net)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def test_stale_delegated_method_raises_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    coord = Coordinator.__new__(Coordinator)
    stale_name = "_stale_delegated_for_test"
    monkeypatch.setitem(Coordinator._DELEGATED, stale_name, "phase_kernel")

    with pytest.raises(AttributeError, match="does not define"):
        getattr(coord, stale_name)


def _build_backends() -> dict[str, Backend]:
    return {name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "critic", "robustness")}


def test_delegated_missing_attr_raises_attribute_error_not_recursion(monkeypatch) -> None:
    coord = object.__new__(Coordinator)
    coord.__dict__["dummy_owner"] = object()
    monkeypatch.setitem(Coordinator._DELEGATED, "_deleted_delegate", "dummy_owner")

    with pytest.raises(AttributeError, match="delegates '_deleted_delegate'"):
        getattr(coord, "_deleted_delegate")


@pytest.mark.asyncio
async def test_resume_rolls_back_recipe_checkout_and_kernel(
    coord: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restores: list[tuple[str, str]] = []
    kernel_restores: list[dict] = []
    import hyperloom.orchestrator.actions.executors.baseline as baseline_module
    import hyperloom.orchestrator.kernel.request_handlers as kernel_handlers

    monkeypatch.setattr(
        baseline_module,
        "_revert_patches",
        lambda target, sha, manifest=None: restores.append((target, sha)) or {"ok": True, "errors": []},
    )
    monkeypatch.setattr(
        kernel_handlers,
        "_maybe_revert_kernel_patch",
        lambda result: kernel_restores.append(result) or {"status": "ok"},
    )
    task = await coord.tasks.create(
        kind="replay_warm_recipe",
        params={},
        idempotency_key="resume-warm",
    )
    coord.shared_state.warm_replay_pending = {
        "task_id": task.task_id,
        "recipe_patch_target": "/mirror",
        "recipe_patch_pre_sha": "mirror-sha",
        "recipe_patch_snapshot_manifest": {"manifest_path": "/mirror.json"},
        "kernel_apply_results": [{"manifest_path": "/tmp/m"}],
    }
    report = {"fixes": [], "warnings": []}

    await coord.writeback._resume_recover_pending_warm_replay(report)

    assert restores == [("/mirror", "mirror-sha")]
    assert kernel_restores == [{"manifest_path": "/tmp/m"}]
    assert coord.shared_state.warm_replay_pending == {}
    assert report["fixes"][0]["kind"] == "recovered_pending_warm_replay"
    assert report["fixes"][0]["task_state"] == "cancelled"
    assert (await coord.tasks.get(task.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_resume_retains_pending_recipe_target_without_manifest(
    coord: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel_restores: list[dict] = []
    import hyperloom.orchestrator.kernel.request_handlers as kernel_handlers

    monkeypatch.setattr(
        kernel_handlers,
        "_maybe_revert_kernel_patch",
        lambda result: kernel_restores.append(result) or {"status": "ok"},
    )
    coord.shared_state.warm_replay_pending = {
        "task_id": "warm-recipe-unarmed",
        "recipe_patch_target": "/mirror",
        "recipe_patch_pre_sha": "sha",
        "kernel_apply_results": [{"manifest_path": "/tmp/kernel"}],
    }
    report = {"fixes": [], "warnings": []}

    await coord.writeback._resume_recover_pending_warm_replay(report)

    assert coord.shared_state.warm_replay_pending["status"] == "rollback_failed"
    assert coord.shared_state.warm_replay_pending["rollback_errors"] == ["recipe:/mirror:missing_snapshot_manifest"]
    assert report["warnings"][0]["kind"] == "resume_warm_rollback_failed"
    assert report["fixes"] == []
    assert kernel_restores == [{"manifest_path": "/tmp/kernel"}]
    assert coord.shared_state.stop_reason == "warm_replay_rollback_failed"


@pytest.mark.asyncio
async def test_resume_retains_pending_when_any_restore_fails(
    coord: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hyperloom.orchestrator.actions.executors.baseline as baseline_module

    monkeypatch.setattr(
        baseline_module,
        "_revert_patches",
        lambda *_args: {"ok": False, "errors": ["restore failed"]},
    )
    coord.shared_state.warm_replay_pending = {
        "task_id": "warm-failed",
        "recipe_patch_target": "/mirror",
        "recipe_patch_pre_sha": "sha",
        "recipe_patch_snapshot_manifest": {"manifest_path": "/mirror.json"},
        "kernel_apply_results": [],
    }
    report = {"fixes": [], "warnings": []}

    await coord.writeback._resume_recover_pending_warm_replay(report)

    assert coord.shared_state.warm_replay_pending["status"] == "rollback_failed"
    assert coord.shared_state.warm_replay_pending["rollback_errors"] == ["restore failed"]
    assert report["warnings"][0]["kind"] == "resume_warm_rollback_failed"
    assert report["fixes"] == []


def test_collective_resume_gate_is_delegated_to_kernel_phase() -> None:
    """Writeback resume must resolve the Collective gate."""
    assert Coordinator._DELEGATED.get("_collective_required_before_kernel_opt") == "phase_kernel"


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


def test_every_delegated_name_resolves_on_its_collaborator(coord: Coordinator) -> None:
    """A map entry naming a method its collaborator never defined is a crash at first call, not at import.

    A field run lost every EXPLORE variant-failure record to exactly that: the
    entry was there, the method was not, and ``__getattr__`` raised only once
    the reap loop reached for it.
    """
    unresolved = []
    for name in Coordinator._DELEGATED:
        try:
            getattr(coord, name)
        except AttributeError as exc:
            unresolved.append(f"{name}: {exc}")
    assert unresolved == []


# -- _context_inbox_reader --------------------------------------------------
def test_context_inbox_reader_empty(coord: Coordinator) -> None:
    out = coord._context_inbox_reader()
    assert out == "(no inbox events)"


def test_trace_mcp_setup_persists_diagnostics(coord: Coordinator) -> None:
    backend = SimpleNamespace(
        model="claude-test",
        get_mcp_setup_diagnostic=lambda: {
            "sdk_name": "claude_agent_sdk",
            "emit_intent": {"registered": True},
        },
    )

    coord._trace_mcp_setup(agent_name="orchestration", backend=backend)

    setup = json.loads((coord.session_dir / "agents" / "orchestration" / "mcp_setup.json").read_text())
    assert setup["emit_intent"]["registered"] is True


@pytest.mark.asyncio
async def test_context_inbox_reader_with_events(coord: Coordinator) -> None:
    await coord.bus.append_and_seq(Message.new("kernel_agent", "orchestration", "heartbeat", {"body_md": "hi"}))
    out = coord._context_inbox_reader()
    assert "(no inbox events)" not in out
    assert isinstance(out, str)


# -- _context_recent_outcomes_reader ----------------------------------------
def test_recent_outcomes_reader_empty(coord: Coordinator) -> None:
    assert coord._context_recent_outcomes_reader() == "(no recent outcomes)"


@pytest.mark.asyncio
async def test_recent_outcomes_reader_with_rows(coord: Coordinator) -> None:
    await coord.bus.append_and_seq(
        Message.new("kernel_agent", "*", "delegated_result", {"action_name": "explore", "status": "succeeded"})
    )
    out = coord._context_recent_outcomes_reader(top_k=4)
    assert "Recent action outcomes" in out


def test_recent_outcomes_reader_clamps_top_k(coord: Coordinator) -> None:
    assert isinstance(coord._context_recent_outcomes_reader(top_k=999), str)
    assert isinstance(coord._context_recent_outcomes_reader(top_k=0), str)


# -- _reset_orchestration_conversation --------------------------------------
def test_reset_orchestration_conversation_clears_seed(coord: Coordinator) -> None:
    coord._orchestration_seeded = True
    coord._reset_orchestration_conversation()
    assert coord._orchestration_seeded is False


def test_reset_orchestration_conversation_invokes_backend_hook(coord: Coordinator) -> None:
    calls: list[int] = []
    backend = coord.backends["orchestration"]
    backend.reset_conversation = lambda: calls.append(1)  # type: ignore[attr-defined]
    coord._orchestration_seeded = True
    coord._reset_orchestration_conversation()
    assert calls == [1]
    assert coord._orchestration_seeded is False


def test_reset_orchestration_conversation_swallows_hook_error(coord: Coordinator) -> None:
    def _boom() -> None:
        raise RuntimeError("nope")

    coord.backends["orchestration"].reset_conversation = _boom  # type: ignore[attr-defined]
    coord._orchestration_seeded = True
    coord._reset_orchestration_conversation()
    assert coord._orchestration_seeded is False


@pytest.mark.asyncio
async def test_resume_consistency_marks_unvalidated_keeps(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "v1",
            "candidate_extra_server_args": "--a 1",
            "extra_envs": {"A": "1"},
            "tput": 110.0,
        },
        {
            "action": "integrate_patch",
            "variant_name": "p1",
            "candidate_extra_server_args": "--b 2",
            "extra_envs": {"B": "2"},
            "tput": 120.0,
        },
    ]
    coord.shared_state.cumulative_gain_validated_stack_len = 1
    coord.shared_state.current_best = {"extra_server_args": "--a 1 --b 2", "extra_envs": {"A": "1", "B": "2"}}

    report = await coord._resume_consistency_pass()

    warning_kinds = {w["kind"] for w in report["warnings"]}
    assert "resume_unvalidated_keeps" in warning_kinds
    assert coord.shared_state.resume_pending_revalidation is True
    assert any(isinstance(f, dict) and f.get("kind") == "queued_resume_stack_rebench" for f in report["fixes"])


@pytest.mark.asyncio
async def test_resume_consistency_leaves_current_best_alone(coord: Coordinator) -> None:
    """Resume must not rewrite the config; a stack replay loses ablated envs."""
    coord._resumed_from["is_resume"] = True
    coord.shared_state.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "v1",
            "candidate_extra_server_args": "--a 1",
            "extra_envs": {"OLD": "1"},
            "tput": 110.0,
        },
        {
            "action": "explore",
            "variant_name": "v2",
            "candidate_extra_server_args": "--b 2",
            "extra_envs": {"NEW": "1"},
            "unset_envs": ["OLD"],
            "tput": 120.0,
        },
    ]
    coord.shared_state.cumulative_gain_validated_stack_len = 2
    coord.shared_state.current_best = {"extra_server_args": "--b 2", "extra_envs": {"NEW": "1"}}

    report = await coord._resume_consistency_pass()

    assert coord.shared_state.current_best["extra_envs"] == {"NEW": "1"}
    assert coord.shared_state.current_best["extra_server_args"] == "--b 2"
    assert not any(
        isinstance(w, dict) and w.get("kind") == "resume_inconsistent_current_best" for w in report["warnings"]
    )
    assert "rebuilt_current_best_config_from_stack" not in report["fixes"]


@pytest.mark.asyncio
async def test_resume_restores_promoted_inferencex_checkout(
    coord: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    active = tmp_path / "active-inferencex"
    active.mkdir()
    coord._resumed_from["is_resume"] = True
    coord.shared_state.active_inferencex_path = str(active)
    # setenv, not delenv: delenv of an absent name arms no undo, so the value
    # the resume pass exports below would leak into every later test.
    monkeypatch.setenv("INFERENCEX_PATH", "")

    await coord._resume_consistency_pass()

    assert os.environ["INFERENCEX_PATH"] == str(active)


@pytest.mark.asyncio
async def test_resume_consistency_replays_orphaned_integrate_keep(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "ti-orphan",
                "kind": "integrate_patch",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "specialist_task_id": "spec-orphan",
                    "output_throughput": 123.0,
                    "source_phase": "FRAMEWORK_AGENT",
                    "domain": "serving_specialist",
                    "provenance": "specialist:serving_specialist",
                    "framework_agent_authoring": True,
                    "source_manifest": ("/session/optimization_stack/src/spec-orphan/manifest.json"),
                    "target_files": ["vllm/model.py"],
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    replay = next(f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "replayed_orphaned_keep")
    assert replay["orphan_kind"] == "integrate_patch"
    assert replay["variant"] == "spec-orphan"
    assert coord.shared_state.optimization_stack[-1]["action"] == "integrate_patch"
    assert coord.shared_state.optimization_stack[-1]["variant_name"] == "spec-orphan"
    assert coord.shared_state.optimization_stack[-1]["source_phase"] == "FRAMEWORK_AGENT"
    assert coord.shared_state.optimization_stack[-1]["provenance"] == ("specialist:serving_specialist")
    assert coord.shared_state.optimization_stack[-1]["source_manifest"] == (
        "/session/optimization_stack/src/spec-orphan/manifest.json"
    )
    assert coord.shared_state.optimization_stack[-1]["target_files"] == ["vllm/model.py"]
    assert coord.shared_state.resume_pending_revalidation is True


@pytest.mark.asyncio
async def test_resume_consistency_replays_pending_integrate_keep(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.pending_integrate = {"task_id": "ti-pending", "specialist_task_id": "spec-pending"}
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "ti-pending",
                "kind": "integrate_patch",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "specialist_task_id": "spec-pending",
                    "output_throughput": 125.0,
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    replay = next(f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "replayed_pending_integrate")
    assert replay["task_id"] == "ti-pending"
    assert replay["appended"] is True
    assert coord.shared_state.pending_integrate == {}
    assert coord.shared_state.optimization_stack[-1]["variant_name"] == "spec-pending"
    assert "source_phase" not in coord.shared_state.optimization_stack[-1]
    assert "domain" not in coord.shared_state.optimization_stack[-1]
    assert "framework_agent_authoring" not in coord.shared_state.optimization_stack[-1]
    assert coord.shared_state.optimization_stack[-1]["recipe_publishable"] is False
    assert coord.shared_state.resume_pending_revalidation is True


@pytest.mark.asyncio
async def test_resume_consistency_rolls_back_pending_integrate(coord: Coordinator, monkeypatch) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.pending_integrate = {
        "task_id": "ti-roll",
        "framework_source_root": "/tmp/framework",
        "patches": ["/tmp/p.diff"],
    }
    monkeypatch.setattr(
        coord.writeback,
        "_resume_rollback_pending_integrate",
        lambda pending: {"reversed": list(pending["patches"]), "failed": []},
    )

    report = await coord._resume_consistency_pass()

    rolled = next(f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "rolled_back_pending_integrate")
    assert rolled["task_id"] == "ti-roll"
    assert rolled["reversed"] == ["/tmp/p.diff"]
    assert coord.shared_state.pending_integrate == {}


@pytest.mark.asyncio
async def test_resume_consistency_clears_stale_pending_integrate(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.pending_integrate = {"task_id": "ti-stale"}

    report = await coord._resume_consistency_pass()

    cleared = next(f for f in report["fixes"] if isinstance(f, dict) and f["kind"] == "cleared_stale_pending_integrate")
    assert cleared["task_id"] == "ti-stale"
    assert coord.shared_state.pending_integrate == {}


@pytest.mark.asyncio
async def test_resume_consistency_keeps_sentinel_when_event_scan_fails(
    coord: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable event log must not be treated as 'no KEEP exists'."""
    coord._resumed_from["is_resume"] = True
    sentinel = {
        "task_id": "ti-unreadable",
        "framework_source_root": "/tmp/framework",
        "patches": ["/tmp/p.diff"],
    }
    coord.shared_state.pending_integrate = dict(sentinel)

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr(coord.bus, "tail", _boom)

    rolled_back: list[dict] = []
    monkeypatch.setattr(
        coord.writeback,
        "_resume_rollback_pending_integrate",
        lambda pending: rolled_back.append(pending) or {"reversed": [], "failed": []},
    )

    report = await coord._resume_consistency_pass()

    assert rolled_back == []
    assert coord.shared_state.pending_integrate == sentinel
    warning = next(w for w in report["warnings"] if w.get("kind") == "pending_integrate_scan_failed")
    assert warning["task_id"] == "ti-unreadable"
    assert not any(
        isinstance(f, dict) and f.get("kind") in {"rolled_back_pending_integrate", "cleared_stale_pending_integrate"}
        for f in report["fixes"]
    )


@pytest.mark.asyncio
async def test_resume_consistency_discards_orphaned_integrate_keep_missing_workspace(
    coord: Coordinator,
    tmp_path: Path,
) -> None:
    coord._resumed_from["is_resume"] = True
    missing_workspace = tmp_path / "missing-workspace"
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "ti-missing",
                "kind": "integrate_patch",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "specialist_task_id": "spec-missing",
                    "output_throughput": 125.0,
                    "workspace": str(missing_workspace),
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    discarded = next(w for w in report["warnings"] if w["kind"] == "orphaned_keep_discarded")
    assert discarded["orphan_kind"] == "integrate_patch"
    assert discarded["variant"] == "spec-missing"
    assert coord.shared_state.optimization_stack == []


@pytest.mark.asyncio
async def test_resume_consistency_discards_orphan_when_workspace_missing(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "ti-gone",
                "kind": "integrate_patch",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "specialist_task_id": "spec-gone",
                    "output_throughput": 123.0,
                    "workspace": "/nonexistent/path/spec-gone",
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    assert any(w.get("kind") == "orphaned_keep_discarded" for w in report["warnings"])
    assert not any(
        isinstance(e, dict) and e.get("variant_name") == "spec-gone" for e in coord.shared_state.optimization_stack
    )


@pytest.mark.asyncio
async def test_resume_consistency_explore_orphan_alerts_not_replayed(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "te-orphan",
                "kind": "explore",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "best_variant": {"name": "ev-1", "extra_envs": {"A": "1"}},
                    "output_throughput": 123.0,
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    assert any(w.get("kind") == "orphaned_keep" and w.get("orphan_kind") == "explore" for w in report["warnings"])
    assert not any(
        isinstance(e, dict) and e.get("variant_name") == "ev-1" for e in coord.shared_state.optimization_stack
    )


@pytest.mark.asyncio
async def test_resume_consistency_framework_keep_in_stack_is_not_orphaned(coord: Coordinator) -> None:
    """A landed framework KEEP reconciles against its own stack entry.

    The stack records the ``framework`` family label plus the canonical
    candidate key, while the event log records the ``framework_agent`` task
    kind. Comparing the two without translating flagged every landed KEEP as
    an orphan on every single resume.
    """
    coord._resumed_from["is_resume"] = True
    coord.shared_state.optimization_stack = [
        {
            "action": "framework",
            "variant_name": "https://example.com/pull/7",
            "candidate_extra_server_args": "",
            "extra_envs": {},
            "tput": 130.0,
        }
    ]
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "tf-landed",
                "kind": "framework_agent",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "candidate": {"pr_url": "https://example.com/pull/7", "ref": "PR:7"},
                    "output_throughput": 130.0,
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    assert not [
        w for w in report["warnings"] if w.get("kind") == "orphaned_keep" and w.get("orphan_kind") == "framework_agent"
    ]


@pytest.mark.asyncio
async def test_resume_consistency_framework_keep_absent_from_stack_still_alerts(coord: Coordinator) -> None:
    """The reconciliation fix must not suppress a genuinely missing framework KEEP."""
    coord._resumed_from["is_resume"] = True
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "tf-orphan",
                "kind": "framework_agent",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "candidate": {"pr_url": "https://example.com/pull/8", "ref": "PR:8"},
                    "output_throughput": 140.0,
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    orphan = next(
        w for w in report["warnings"] if w.get("kind") == "orphaned_keep" and w.get("orphan_kind") == "framework_agent"
    )
    assert orphan["variant"] == "https://example.com/pull/8"


@pytest.mark.asyncio
async def test_resume_consistency_replays_pending_integrate_with_kept_result(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.pending_integrate = {"task_id": "ti-half", "specialist_task_id": "spec-half"}
    await coord.bus.append_and_seq(
        Message.new(
            "coordinator",
            "*",
            "delegated_result",
            {
                "task_id": "ti-half",
                "kind": "integrate_patch",
                "state": "succeeded",
                "result": {
                    "status": "kept",
                    "specialist_task_id": "spec-half",
                    "output_throughput": 130.0,
                    "config_changes_applied": {"BAR": "2"},
                },
            },
        )
    )

    report = await coord._resume_consistency_pass()

    assert any(isinstance(f, dict) and f.get("kind") == "replayed_pending_integrate" for f in report["fixes"])
    assert coord.shared_state.pending_integrate == {}
    assert any(
        isinstance(e, dict) and e.get("variant_name") == "spec-half" for e in coord.shared_state.optimization_stack
    )


@pytest.mark.asyncio
async def test_resume_consistency_rolls_back_pending_integrate_without_kept(coord: Coordinator, monkeypatch) -> None:
    import hyperloom.orchestrator.actions.executors.integrate_patch as ip

    reversed_calls: list[str] = []

    def _fake_reverse(root, patch):
        reversed_calls.append(str(patch))
        return True, ""

    monkeypatch.setattr(ip, "_git_apply_reverse", _fake_reverse)
    coord._resumed_from["is_resume"] = True
    coord.shared_state.pending_integrate = {
        "task_id": "ti-crash",
        "specialist_task_id": "spec-crash",
        "framework_source_root": "/tmp/fw",
        "patches": ["/tmp/fw/p1.diff"],
    }

    report = await coord._resume_consistency_pass()

    assert reversed_calls == ["/tmp/fw/p1.diff"]
    assert any(isinstance(f, dict) and f.get("kind") == "rolled_back_pending_integrate" for f in report["fixes"])
    assert coord.shared_state.pending_integrate == {}


@pytest.mark.asyncio
async def test_resume_consistency_clears_stale_pending_integrate_with_specialist_id(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.pending_integrate = {"task_id": "ti-stale", "specialist_task_id": "spec-stale"}

    report = await coord._resume_consistency_pass()

    assert any(isinstance(f, dict) and f.get("kind") == "cleared_stale_pending_integrate" for f in report["fixes"])
    assert coord.shared_state.pending_integrate == {}


@pytest.mark.asyncio
async def test_resume_consistency_enqueues_stack_rebench_for_unvalidated(coord: Coordinator) -> None:
    coord._resumed_from["is_resume"] = True
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "v1",
            "candidate_extra_server_args": "--a 1",
            "extra_envs": {"A": "1"},
            "tput": 110.0,
        }
    ]
    coord.shared_state.cumulative_gain_validated_stack_len = 0
    # The lift writes both together, so a stack always has a config behind it.
    coord.shared_state.current_best = {
        "action": "explore",
        "variant_name": "v1",
        "tput": 110.0,
        "extra_server_args": "--a 1",
        "extra_envs": {"A": "1"},
    }

    report = await coord._resume_consistency_pass()

    assert coord.shared_state.resume_pending_revalidation is True
    queued = await coord.tasks.queued()
    assert any(t.kind == "explore" and t.params.get("source") == "resume_stack_revalidate" for t in queued)
    assert any(isinstance(f, dict) and f.get("kind") == "queued_resume_stack_rebench" for f in report["fixes"])


@pytest.mark.asyncio
async def test_resume_stack_revalidate_promote_clears_flag_and_sets_watermark(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "v1", "candidate_extra_server_args": "--a 1", "tput": 110.0}
    ]
    coord.shared_state.cumulative_gain_validated_stack_len = 0
    task = SimpleNamespace(task_id="tr-1", params={"source": "resume_stack_revalidate"})
    await coord._promote_to_shared_state(
        "explore",
        {"winners": [], "best_variant": None, "output_throughput": 121.0},
        task=task,
    )

    assert coord.shared_state.resume_pending_revalidation is False
    assert coord.shared_state.cumulative_gain_validated_stack_len == 1
    assert coord.shared_state.cumulative_gain_validated == pytest.approx(21.0)


@pytest.mark.asyncio
async def test_resume_revalidate_failed_rebench_keeps_flag_set(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "v1", "candidate_extra_server_args": "--a 1", "tput": 110.0}
    ]
    coord.shared_state.cumulative_gain_validated_stack_len = 0
    task = SimpleNamespace(task_id="tr-fail", params={"source": "resume_stack_revalidate"})
    await coord._promote_to_shared_state(
        "explore",
        {"winners": [], "best_variant": None, "output_throughput": 0.0},
        task=task,
    )

    assert coord.shared_state.resume_pending_revalidation is True
    assert coord.shared_state.cumulative_gain_validated_stack_len == 0


@pytest.mark.asyncio
async def test_resume_reverify_best_promote_clears_flag(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "v1", "candidate_extra_server_args": "--a 1", "tput": 110.0}
    ]
    coord.shared_state.cumulative_gain_validated_stack_len = 0
    task = SimpleNamespace(task_id="rb-1", params={"source": "resume_reverify_best"})
    await coord._promote_to_shared_state(
        "explore",
        {"winners": [], "best_variant": None, "output_throughput": 118.0},
        task=task,
    )

    assert coord.shared_state.resume_pending_revalidation is False
    assert coord.shared_state.cumulative_gain_validated_stack_len == 1


@pytest.mark.asyncio
async def test_integrate_patch_keep_promotes_stack_and_clears_pending(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.pending_integrate = {"task_id": "ti-1"}
    task = SimpleNamespace(task_id="ti-1", params={})
    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "specialist_task_id": "spec-1",
            "output_throughput": 112.0,
            "delta_pct": 12.0,
            "accuracy_pass": True,
            "config_changes_applied": {"X": "1"},
            "patches_applied": ["p.diff"],
            "patches_reverted": [],
            "workspace": "/tmp/integrate",
        },
        task=task,
    )

    assert coord.shared_state.pending_integrate == {}
    assert coord.shared_state.current_best["action"] == "integrate_patch"
    assert coord.shared_state.optimization_stack[-1]["variant_name"] == "spec-1"
    assert coord.shared_state.cumulative_gain_validated == pytest.approx(12.0)
    assert coord.shared_state.cumulative_gain_validated_stack_len == len(coord.shared_state.optimization_stack)


@pytest.mark.asyncio
async def test_reactor_pass_records_context_tokens(coord: Coordinator) -> None:
    backend = MockBackend(
        ScriptedPlan(
            turns=[
                MockTurn(
                    intents=[_heartbeat()],
                    raw_text="ok",
                    metadata={
                        "input_tokens": 10,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 30,
                        "context_tokens_peak": 45,
                    },
                )
            ]
        ),
        name="orchestration",
    )
    backend.conversational = True  # type: ignore[attr-defined]
    coord.backends["orchestration"] = backend
    await coord._reactor_pass("orchestration")
    assert coord._checkpoint_tracker.context_tokens_now == 45


@pytest.mark.asyncio
async def test_reactor_pass_ignores_call_cumulative_token_counters(coord: Coordinator) -> None:
    backend = MockBackend(
        ScriptedPlan(
            turns=[
                MockTurn(
                    intents=[_heartbeat()],
                    raw_text="ok",
                    metadata={
                        "input_tokens": 6,
                        "cache_read_input_tokens": 75_448,
                        "cache_creation_input_tokens": 154_099,
                    },
                )
            ]
        ),
        name="orchestration",
    )
    backend.conversational = True  # type: ignore[attr-defined]
    coord.backends["orchestration"] = backend
    await coord._reactor_pass("orchestration")
    assert coord._checkpoint_tracker.context_tokens_now == 0
    assert coord._checkpoint_tracker.chars_since_last > 0


@pytest.mark.asyncio
async def test_reactor_pass_chars_fallback_without_token_metadata(coord: Coordinator) -> None:
    backend = MockBackend(
        ScriptedPlan(turns=[MockTurn(intents=[_heartbeat()], raw_text="raw-reply", metadata={})]),
        name="orchestration",
    )
    backend.conversational = True  # type: ignore[attr-defined]
    coord.backends["orchestration"] = backend
    coord._checkpoint_tracker.chars_since_last = 0
    await coord._reactor_pass("orchestration")
    assert coord._checkpoint_tracker.context_tokens_now == 0
    assert coord._checkpoint_tracker.chars_since_last > len("raw-reply")


# -- _conversation_progress_signal ------------------------------------------
def test_progress_signal_first_call_seeds_marker(coord: Coordinator) -> None:
    coord._progress_marker = {}
    sig = coord._conversation_progress_signal()
    assert sig["ticks_without_progress"] == 0
    assert sig["severity"] == "ok"


def test_progress_signal_detects_progress(coord: Coordinator) -> None:
    coord._progress_marker = {}
    coord._conversation_progress_signal()  # seed
    coord.shared_state.tick = 5
    coord.shared_state.cumulative_gain_validated = 10.0
    sig = coord._conversation_progress_signal()
    assert sig["last_progress_tick"] == 5
    assert sig["severity"] == "ok"


def test_progress_signal_flags_stall(coord: Coordinator) -> None:
    coord._progress_marker = {}
    coord._no_progress_threshold = 2
    coord._conversation_progress_signal()  # seed at tick 0
    coord.shared_state.tick = 10
    sig = coord._conversation_progress_signal()
    assert sig["ticks_without_progress"] >= 2
    assert sig["severity"] == "high"


# -- replay_for_resume ------------------------------------------------------
@pytest.mark.asyncio
async def test_replay_for_resume_rebuilds_undecided_proposals(coord: Coordinator) -> None:
    p1 = Message.new("kernel_agent", "orchestration", "proposal", {"action_name": "explore", "predicted_gain_pct": 3.0})
    await coord.bus.append_and_seq(p1)
    p2 = Message.new(
        "kernel_agent", "orchestration", "proposal", {"action_name": "baseline", "predicted_gain_pct": 1.0}
    )
    await coord.bus.append_and_seq(p2)
    await coord.bus.append_and_seq(
        Message.new(
            "critic", "orchestration", "review_verdict", {"target_proposal_msg_id": p2.msg_id, "verdict": "approve"}
        )
    )
    out = await coord.replay_for_resume()
    assert out["pending_restored"] == 1
    assert p1.msg_id in coord.state.pending_proposals
    assert p2.msg_id not in coord.state.pending_proposals


@pytest.mark.asyncio
async def test_replay_for_resume_verdict_map_backcompat(coord: Coordinator) -> None:
    p1 = Message.new("kernel_agent", "orchestration", "proposal", {"action_name": "explore"})
    await coord.bus.append_and_seq(p1)
    await coord.bus.append_and_seq(
        Message.new(
            "critic",
            "orchestration",
            "review_verdict",
            {"target_proposal_msg_id": p1.msg_id, "verdict_map": {"x": "ok"}},
        )
    )
    out = await coord.replay_for_resume()
    assert out["verdicts_seen"] == 1
    assert p1.msg_id not in coord.state.pending_proposals


# -- _context_analysis_reader fallback (path read) --------------------------
def test_context_analysis_reader_path_fallback_on_format_error(
    coord: Coordinator,
    tmp_path,
    monkeypatch,
) -> None:
    md = tmp_path / "analysis.md"
    md.write_text("# roofline snapshot\n", encoding="utf-8")
    coord.shared_state.last_trace_analyze = {"analysis_md_path": str(md)}

    def _boom() -> str:
        raise RuntimeError("format failed")

    monkeypatch.setattr(coord.shared_state, "_format_analysis_md_full", _boom)
    out = coord._context_analysis_reader()
    assert "roofline snapshot" in out


def test_context_analysis_reader_unreadable_path(
    coord: Coordinator,
    monkeypatch,
) -> None:
    coord.shared_state.last_trace_analyze = {"analysis_md_path": "/nonexistent/dir/analysis.md"}
    monkeypatch.setattr(
        coord.shared_state,
        "_format_analysis_md_full",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    out = coord._context_analysis_reader()
    assert "unreadable" in out or "no analysis.md" in out


# -- _recipe_kb_t4_hook + stop -------------------------------------------------
@pytest.mark.asyncio
async def test_recipe_kb_t4_hook_noop_without_kb(coord: Coordinator) -> None:
    coord.recipe_kb = None
    await coord._recipe_kb_t4_hook()


@pytest.mark.asyncio
async def test_stop_cancels_and_closes(coord: Coordinator) -> None:
    await coord.stop()
    assert coord._stop.is_set()


# -- _pump_dispatcher_once --------------------------------------------------
def _sub_result(task_id: str, *, state: str = "succeeded", result=None, error=None):
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

    return SubAgentResult(task_id=task_id, state=state, result=result if result is not None else {}, error=error)


@pytest.mark.asyncio
async def test_pump_dispatcher_noop_when_empty(coord: Coordinator) -> None:
    await coord._pump_dispatcher_once()


@pytest.mark.asyncio
async def test_pump_dispatcher_explore_promotes(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.baseline_tput = 800.0
    task = await coord.tasks.create(
        kind="explore",
        params={},
        idempotency_key="disp-explore",
    )

    async def fake_run(t, **kw):
        return _sub_result(
            t.task_id,
            result={
                "status": "succeeded",
                "winners": [{"name": "v0", "extra_server_args": "--tp 1"}],
                "best_variant": {"name": "v0", "extra_server_args": "--tp 1"},
                "output_throughput": 900.0,
                "round_id": "r1",
                "losers": [],
                "skipped_dup": [],
            },
        )

    monkeypatch.setattr(coord.sub, "run_task", fake_run)
    await coord._pump_dispatcher_once()
    tail = await coord.bus.tail(topic="delegated_result", n=10)
    assert any(m.payload.get("task_id") == task.task_id for m in tail)


@pytest.mark.asyncio
async def test_pump_dispatcher_specialist_bookkeeping(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "0")
    await coord.tasks.create(
        kind="specialist",
        params={"domain": "kernel_agent"},
        idempotency_key="disp-spec",
    )

    async def fake_run(t, **kw):
        return _sub_result(
            t.task_id,
            result={
                "status": "succeeded",
                "specialist_done": {"patches_written": []},
            },
        )

    monkeypatch.setattr(coord.sub, "run_task", fake_run)
    await coord._pump_dispatcher_once()
    tail = await coord.bus.tail(topic="delegated_result", n=10)
    assert any(m.payload.get("kind") == "specialist" for m in tail)


@pytest.mark.asyncio
async def test_pump_dispatcher_absorbs_spawn_exception(coord: Coordinator, monkeypatch) -> None:
    await coord.tasks.create(
        kind="explore",
        params={},
        idempotency_key="disp-boom",
    )

    async def fake_run(t, **kw):
        raise RuntimeError("spawn boom")

    monkeypatch.setattr(coord.sub, "run_task", fake_run)
    await coord._pump_dispatcher_once()


# -- _maybe_checkpoint_orchestration (taken path) ---------------------------
class _FakeRunResult:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text


def _make_conversational(coord: Coordinator, *, raw_text: str | None = None) -> None:
    backend = coord.backends["orchestration"]
    backend.conversational = True  # type: ignore[attr-defined]
    backend.reset_conversation = lambda: None  # type: ignore[attr-defined]
    # Well-formed checkpoint reply exercises the compaction "taken" path; raw_text= exercises the degenerate path.
    reply = (
        raw_text
        if raw_text is not None
        else (
            '```json\n{"current_plan": "tune MoE", "hypotheses": ["h1"], '
            '"tried_and_why": ["explored attention backends"], "pending": ["p1"], '
            '"learnings": ["l1"]}\n```'
        )
    )

    async def _run(**kw):
        return _FakeRunResult(reply)

    backend.run = _run  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_checkpoint_disabled_returns_false(coord: Coordinator) -> None:
    coord._checkpoint_enabled = False
    assert await coord._maybe_checkpoint_orchestration(tick=1) is False


@pytest.mark.asyncio
async def test_checkpoint_policy_declines(coord: Coordinator) -> None:
    import time

    _make_conversational(coord)
    coord._checkpoint_enabled = True
    coord._orchestration_seeded = True
    coord._run_started_monotonic = time.monotonic()

    class _Policy:
        def should_checkpoint(self, **kw):
            return False

    coord._checkpoint_policy = _Policy()
    assert await coord._maybe_checkpoint_orchestration(tick=5) is False


@pytest.mark.asyncio
async def test_checkpoint_taken_compacts_memory(coord: Coordinator) -> None:
    import time

    _make_conversational(coord)
    coord._checkpoint_enabled = True
    coord._orchestration_seeded = True
    coord._run_started_monotonic = time.monotonic()

    class _Policy:
        def should_checkpoint(self, **kw):
            return True

    coord._checkpoint_policy = _Policy()
    took = await coord._maybe_checkpoint_orchestration(tick=12, phase_changed=True)
    assert took is True
    assert coord._orchestration_seeded is False
    assert coord.shared_state.orchestration_memory
    assert len(coord.shared_state.orchestration_memory_history) == 1


@pytest.mark.asyncio
async def test_checkpoint_history_ring_caps_at_ten(coord: Coordinator) -> None:
    import time

    coord._checkpoint_enabled = True
    coord._run_started_monotonic = time.monotonic()
    _always_checkpoint(coord)
    backend = coord.backends["orchestration"]
    backend.conversational = True  # type: ignore[attr-defined]
    backend.reset_conversation = lambda: None  # type: ignore[attr-defined]

    for i in range(12):

        async def _run(**kw):
            return _FakeRunResult(
                f'```json\n{{"current_plan": "plan {i}", "hypotheses": ["h{i}"], '
                f'"tried_and_why": [], "pending": [], "learnings": []}}\n```'
            )

        backend.run = _run  # type: ignore[assignment]
        coord._orchestration_seeded = True
        assert await coord._maybe_checkpoint_orchestration(tick=i + 1) is True

    hist = coord.shared_state.orchestration_memory_history
    assert len(hist) == 10
    assert hist[0]["current_plan"] == "plan 2"
    assert hist[-1]["current_plan"] == "plan 11"


def test_checkpoint_policy_context_fraction_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CTX_SOFT_FRACTION", "0.5")
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker

    seed_target_analysis_marker(sd)
    backends = _build_backends()
    backends["orchestration"].model = "claude-opus-4-8"  # type: ignore[attr-defined]
    c = Coordinator(sd, backends=backends)
    assert c._checkpoint_policy.context_token_soft == 100_000


def test_orchestration_memory_rollback_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ORCH_MEMORY_ROLLBACK", "2")
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker

    seed_target_analysis_marker(sd)
    state = SharedState.load_or_init(sd)
    state.orchestration_memory = {"current_plan": "latest"}
    state.orchestration_memory_history = [
        {"current_plan": "older"},
        {"current_plan": "middle"},
        {"current_plan": "latest"},
    ]
    state.save(sd)

    c = Coordinator(sd, backends=_build_backends())
    assert c.shared_state.orchestration_memory == {"current_plan": "middle"}
    assert "current_plan: middle" in c._orchestration_seed_memory


def _always_checkpoint(coord: Coordinator) -> None:
    class _Policy:
        def should_checkpoint(self, **kw):
            return True

    coord._checkpoint_policy = _Policy()


@pytest.mark.asyncio
async def test_checkpoint_degenerate_non_hard_skips_and_preserves(coord: Coordinator) -> None:
    import time

    # Non-JSON reply → degenerate; not near the window → skip compaction.
    _make_conversational(coord, raw_text="just prose, no JSON object here")
    coord._checkpoint_enabled = True
    coord._orchestration_seeded = True
    coord._run_started_monotonic = time.monotonic()
    coord.shared_state.orchestration_memory = {"current_plan": "keep me"}
    _always_checkpoint(coord)

    took = await coord._maybe_checkpoint_orchestration(tick=7)
    assert took is False
    # conversation NOT reset and prior memory untouched
    assert coord._orchestration_seeded is True
    assert coord.shared_state.orchestration_memory == {"current_plan": "keep me"}
    assert coord._consec_degenerate_ckpt == 1
    assert coord._checkpoint_tracker.last_tick == 7
    assert coord._checkpoint_tracker.chars_since_last == 0


@pytest.mark.asyncio
async def test_checkpoint_degenerate_three_times_emits_medium_observation(coord: Coordinator) -> None:
    import time

    _make_conversational(coord, raw_text="just prose, no JSON object here")
    coord._checkpoint_enabled = True
    coord._orchestration_seeded = True
    coord._run_started_monotonic = time.monotonic()
    _always_checkpoint(coord)

    for tick in (1, 2, 3):
        assert await coord._maybe_checkpoint_orchestration(tick=tick) is False

    rows = await coord.bus.tail(topic="observation", n=20)
    degraded = [m.payload for m in rows if m.payload.get("kind") == "orchestration_checkpoint_degraded"]
    assert any(p.get("severity") == "medium" and p.get("consecutive") == 3 for p in degraded)


@pytest.mark.asyncio
async def test_checkpoint_failure_resets_tracker(coord: Coordinator) -> None:
    import time

    _make_conversational(coord)
    coord._checkpoint_enabled = True
    coord._orchestration_seeded = True
    coord._run_started_monotonic = time.monotonic()
    coord._checkpoint_tracker.chars_since_last = 999
    _always_checkpoint(coord)

    async def _boom(**kw):
        raise RuntimeError("backend down")

    coord.backends["orchestration"].run = _boom  # type: ignore[assignment]
    took = await coord._maybe_checkpoint_orchestration(tick=30)
    assert took is False
    # tracker reset even on failure → no checkpoint storm next tick
    assert coord._checkpoint_tracker.chars_since_last == 0


# -- _compose_prompt advisory + telemetry append paths ----------------------
@pytest.mark.asyncio
async def test_compose_prompt_orchestration_all_advisory_blocks(
    coord: Coordinator,
    monkeypatch,
) -> None:
    ss = coord.shared_state
    monkeypatch.setattr(ss, "to_policy_denial_summary", lambda top_k=6: "DENIAL")
    monkeypatch.setattr(ss, "to_warm_start_summary", lambda: "WARM-BLOCK")
    monkeypatch.setattr(ss, "to_gaps_summary", lambda: "GAPS-BLOCK")
    monkeypatch.setattr(ss, "to_proposal_scores_summary", lambda: "SCORES-BLOCK")
    monkeypatch.setattr(ss, "to_intervention_mix_summary", lambda: "MIX-BLOCK")
    monkeypatch.setattr(coord.conversation, "_target_gap_advisory_block", lambda: "GAP-BLOCK")
    monkeypatch.setattr(coord.conversation, "_priors_match_advisory_block", lambda: "PRIORS-BLOCK")
    monkeypatch.setattr(coord.conversation, "_plateau_advisory_block", lambda: "PLATEAU-BLOCK")
    monkeypatch.setattr(coord.conversation, "_research_scout_seed_block", lambda: "HINTS-BLOCK")

    out = await coord._compose_prompt("orchestration")
    for token in (
        "DENIAL",
        "WARM-BLOCK",
        "GAPS-BLOCK",
        "HINTS-BLOCK",
        "GAP-BLOCK",
        "SCORES-BLOCK",
        "PRIORS-BLOCK",
        "MIX-BLOCK",
        "PLATEAU-BLOCK",
    ):
        assert token in out


def test_research_scout_seed_block_keeps_findings_and_questions_only(coord: Coordinator) -> None:
    from hyperloom.orchestrator.knowledge import research_hints

    research_hints.append_hints(
        coord.session_dir,
        [
            {"what": "hint one", "source": "https://example.test/one"},
            {"what": "hint two", "source": "https://example.test/two"},
        ],
    )
    coord.shared_state.specialist_rounds = [
        {
            "domain": "research_scout_specialist",
            "proposal_set": [
                {
                    "name": "first",
                    "extra_envs": {"FIRST": "1"},
                    "source_evidence": ["https://example.test/one"],
                }
            ],
            "residual_questions": ["question one"],
        },
        {
            "domain": "research_scout_specialist",
            "proposal_set": [
                {
                    "name": "second",
                    "extra_args": "--second",
                    "source_evidence": ["https://example.test/two"],
                }
            ],
            "residual_questions": ["question two"],
        },
        {
            "domain": "serving_specialist",
            "proposal_set": [{"name": "ignore-me"}],
            "residual_questions": ["ignore me"],
        },
    ]

    block = coord.conversation._research_scout_seed_block()

    assert "hint one" in block
    assert "hint two" in block
    assert "question one" in block
    assert "question two" in block
    assert "ignore-me" not in block
    # Proposals moved to the shared untested-proposal queue, which also drops
    # the ones already benched; rendering them here as well would double them.
    assert "Untested executable proposals" not in block
    assert '"name": "first"' not in block
    assert '"name": "second"' not in block


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_advisory_blocks_raise(
    coord: Coordinator,
    monkeypatch,
) -> None:
    ss = coord.shared_state

    def _boom(*a, **k):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(ss, "to_phase_status_summary", _boom)
    monkeypatch.setattr(ss, "to_warm_start_summary", _boom)
    monkeypatch.setattr(ss, "to_gaps_summary", _boom)
    monkeypatch.setattr(ss, "to_proposal_scores_summary", _boom)
    monkeypatch.setattr(ss, "to_intervention_mix_summary", _boom)
    monkeypatch.setattr(coord.conversation, "_target_gap_advisory_block", _boom)
    monkeypatch.setattr(coord.conversation, "_priors_match_advisory_block", _boom)
    monkeypatch.setattr(coord.conversation, "_plateau_advisory_block", _boom)
    monkeypatch.setattr(coord.conversation, "_research_scout_seed_block", _boom)

    # Every advisory failure is swallowed; the prompt still renders.
    out = await coord._compose_prompt("orchestration")
    assert "SESSION_DIR=" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_telemetry_raises(
    coord: Coordinator,
    monkeypatch,
) -> None:
    def _boom(*a, **k):
        raise RuntimeError("telemetry failed")

    monkeypatch.setattr(coord.shared_state, "to_phase_budget_telemetry", _boom)
    monkeypatch.setattr(coord.conversation, "_conversation_progress_signal", _boom)
    out = await coord._compose_prompt("robustness")
    # Every advisory failure is swallowed; the prompt still renders.
    assert "SESSION_DIR=" in out


@pytest.mark.asyncio
async def test_promote_baseline_no_warmup_parses_materialized(
    coord: Coordinator,
    monkeypatch,
) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"  # skip cascade
    import hyperloom.orchestrator.loop.writeback as mod

    monkeypatch.setattr(mod, "_parse_baseline_workload_extra", lambda path: {"isl": 256})
    await coord._promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,  # no warmup_round_tput -> else branch
            "materialized_config": "/tmp/run.yaml",
        },
    )
    assert coord.shared_state.baseline_tput == 1000.0
    assert coord.shared_state.baseline_workload_extra == {"isl": 256}


@pytest.mark.asyncio
async def test_promote_baseline_materialized_parse_raises(
    coord: Coordinator,
    monkeypatch,
) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    import hyperloom.orchestrator.loop.writeback as mod

    def _boom(path):
        raise RuntimeError("parse failed")

    monkeypatch.setattr(mod, "_parse_baseline_workload_extra", _boom)
    await coord._promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "materialized_config": "/tmp/run.yaml",
        },
    )
    assert coord.shared_state.baseline_config_path == "/tmp/run.yaml"


@pytest.mark.asyncio
async def test_promote_profile_skipped_clears_pending(coord: Coordinator) -> None:
    task = _ptask("prof-1", "profile")
    coord.shared_state.auto_roofline_pending_task_id = "prof-1"
    await coord._promote_to_shared_state(
        "profile",
        {"status": "skipped", "error_class": "x"},
        task=task,
    )
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_promote_profile_succeeded_reanchors(coord: Coordinator) -> None:
    task = _ptask("prof-2", "profile")
    coord.shared_state.auto_roofline_pending_task_id = "prof-2"
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.cumulative_gain_validated = 10.0
    await coord._promote_to_shared_state(
        "profile",
        {"status": "succeeded", "main_trace_path": "/tmp/t.json", "output_throughput": 880.0},
        task=task,
    )
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_promote_roofline_succeeded_clears_pending(coord: Coordinator) -> None:
    task = _ptask("roof-1", "roofline")
    coord.shared_state.auto_roofline_pending_task_id = "roof-1"
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.cumulative_gain_validated = 5.0
    await coord._promote_to_shared_state(
        "roofline",
        {"status": "succeeded"},
        task=task,
    )
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_promote_roofline_skipped_clears_pending(coord: Coordinator) -> None:
    task = _ptask("roof-2", "roofline")
    coord.shared_state.auto_roofline_pending_task_id = "roof-2"
    await coord._promote_to_shared_state(
        "roofline",
        {"status": "skipped"},
        task=task,
    )
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_promote_explore_discovered_flags_and_bad_winner(
    coord: Coordinator,
) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state(
        "explore",
        {
            "explore_search_update": {"round_id": "r1"},
            "discovered_flags_update": {
                "framework": "sglang",
                "backend_flags": ["--x"],
                "param_flags": [],
                "source_path": "/tmp/p",
                "discovery_error": "parse glitch",
            },
            "winners": ["not-a-dict"],
            "round_id": "r1",
        },
    )
    assert coord.shared_state.discovered_flags_error == "parse glitch"


@pytest.mark.asyncio
async def test_promote_conc_sweep_records(coord: Coordinator) -> None:
    task = _ptask("cs-1", "conc_sweep")
    await coord._promote_to_shared_state(
        "conc_sweep",
        {
            "status": "succeeded",
            "was_skipped": False,
            "summary": {"best_speedup": 1.2, "best_conc": 64, "successful_pairs": 3},
            "report_json_path": "/tmp/cs.json",
        },
        task=task,
    )


@pytest.mark.asyncio
async def test_unpromotable_conc_sweep_records_failed_terminal_state(coord: Coordinator) -> None:
    from hyperloom.orchestrator.phases.machine_state import exit_normal_sweep

    task = _ptask("cs-failed", "conc_sweep")
    await coord._handle_unpromotable_result(
        task,
        {
            "status": "failed",
            "budget_exhausted": False,
            "summary": {"successful_pairs": 0},
            "report_json_path": "/tmp/cs-failed.json",
        },
    )

    assert coord.shared_state.last_conc_sweep["status"] == "failed"
    assert coord.shared_state.last_conc_sweep["budget_exhausted"] is False
    result = exit_normal_sweep(coord.shared_state)
    assert result is not None
    reason, evidence = result
    assert reason == "sweep_failed"
    assert evidence["sweep_status"] == "failed"


@pytest.mark.asyncio
async def test_budget_limited_conc_sweep_skip_records_done(coord: Coordinator) -> None:
    from hyperloom.orchestrator.phases.machine_state import exit_normal_sweep

    task = _ptask("cs-budget-skip", "conc_sweep")
    await coord._promote_to_shared_state(
        "conc_sweep",
        {
            "status": "skipped",
            "was_skipped": True,
            "skip_reason": "budget_exhausted_no_successful_pairs",
            "budget_exhausted": True,
            "summary": {"successful_pairs": 0},
            "report_json_path": "/tmp/cs-budget-skip.json",
        },
        task=task,
    )

    assert coord.shared_state.last_conc_sweep["status"] == "skipped"
    assert coord.shared_state.last_conc_sweep["budget_exhausted"] is True
    result = exit_normal_sweep(coord.shared_state)
    assert result is not None
    reason, evidence = result
    assert reason == "sweep_done"
    assert evidence["sweep_status"] == "skipped"


# -- _promote_to_shared_state additional branches ---------------------------
def _ptask(tid: str, kind: str):
    from hyperloom.orchestrator.state.task_registry import Task

    return Task(task_id=tid, kind=kind, state="running", params={}, idempotency_key=f"{tid}-k")


# -- specialist visibility contract -----------------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_has_no_specialist_status_block(coord: Coordinator) -> None:
    """No periodic specialist block: it can never observe a live specialist.

    The prompt renders only between blocking actions, so a running specialist
    is structurally absent from it. Reporting "none running" there would
    manufacture a false belief; in-flight work reaches the agent via
    ``specialist_progress`` observations and ``get_running_tasks`` instead.
    """
    spec = await coord.tasks.create(
        kind="specialist",
        params={"domain": "serving_specialist"},
        idempotency_key="visible-spec",
    )
    await coord.tasks.transition(spec.task_id, "running")
    for agent in ("orchestration", "robustness"):
        out = await coord._compose_prompt(agent)
        assert "Specialist health" not in out
        assert "stale" not in out.lower()


@pytest.mark.asyncio
async def test_running_tasks_reader_sees_live_specialist(coord: Coordinator) -> None:
    """``get_running_tasks`` is the on-demand path and is not turn-bound."""
    spec = await coord.tasks.create(
        kind="specialist",
        params={"domain": "serving_specialist", "gap_canonical_id": "gap.xyz"},
        idempotency_key="queryable-spec",
    )
    await coord.tasks.transition(spec.task_id, "running")
    out = coord.conversation._context_running_tasks_reader()
    assert spec.task_id in out
    assert "serving_specialist" in out


# -- _fan_out_specialist_wave (valid entries) -------------------------------
@pytest.mark.asyncio
async def test_fan_out_wave_dispatches_valid_task(coord: Coordinator, monkeypatch) -> None:
    seen: list[dict] = []

    async def _fake_delegate(source, intent):
        seen.append(dict(intent.payload.get("params") or {}))

    monkeypatch.setattr(coord, "_handle_delegate", _fake_delegate)
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"idempotency_key": "wave", "action_name": "specialist"},
    )
    await coord._fan_out_specialist_wave(
        "orchestration",
        intent,
        {
            "domain": "kernel_agent",
            "tasks": [
                {"task_description": "scout fused moe", "task_summary": "moe", "mode": "patch", "lane": "gpu"},
            ],
        },
    )
    assert len(seen) == 1
    assert seen[0]["scope"] == "freeform"
    assert seen[0]["task_description"] == "scout fused moe"
    assert seen[0]["mode"] == "patch"


# -- _warm_specialist_params (rich state) -----------------------------------
@pytest.mark.asyncio
async def test_warm_specialist_params_rich_context(coord: Coordinator, monkeypatch) -> None:
    state = coord.shared_state
    state.framework = "sglang"
    state.stack_fingerprint_meta = {"sglang": "0.4.1"}
    state.model_name = "llama"
    state.gpu_type = "mi300x"
    state.last_trace_analyze = {
        "analysis_md_text": "roofline body",
        "analysis_md_path": "/tmp/a.md",
        "roofline_snapshot_id": "snap-1",
        "hot_kernels_top15": [{"name": "gemm"}],
    }
    monkeypatch.setattr(
        state,
        "find_gap",
        lambda cid: {
            "symptom": "mem bound",
            "layer": "attention",
            "domain_hint": "kernel_agent",
            "severity": "high",
            "attempts": [{"r": 1}],
        },
    )
    monkeypatch.setattr(coord.conversation, "_target_gap_advisory_block", lambda: "GAP-NOTES")
    from hyperloom.orchestrator.knowledge import research_hints as rh

    monkeypatch.setattr(rh, "summarise_for_prompt", lambda sd: "HINTS-TEXT")
    from hyperloom.orchestrator.state import shared_state as ss_mod

    monkeypatch.setattr(ss_mod, "render_model_arch_compact", lambda a: "ARCH-NOTES")
    from hyperloom.orchestrator.framework import paths as fp

    monkeypatch.setattr(fp, "resolve_source_file_allowlist", lambda: ["/src/root"])

    params: dict = {"domain": "kernel_agent", "gap_canonical_id": "g1"}
    await coord._warm_specialist_params(params)
    assert params["framework_version"] == "0.4.1"
    assert params["target_gap_notes"] == "GAP-NOTES"
    assert params["research_hints"] == "HINTS-TEXT"
    assert params["arch_notes"] == "ARCH-NOTES"
    assert params["framework_source_roots"] == ["/src/root"]
    assert params["gap_symptom"] == "mem bound"
    assert "roofline_evidence" in params


# -- _record_fact_per_task (recipe KB amend path) ------------------------------
@pytest.mark.asyncio
async def test_record_fact_per_task_writes_lesson(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    coord.recipe_kb = object()  # non-None -> KB amend path
    coord.shared_state.model_name = "llama"
    coord.shared_state.gpu_type = "mi300x"
    amends: list[dict] = []
    monkeypatch.setattr(coord.proposals, "_kb_amend_recipe", lambda **k: amends.append(k))
    task = Task(task_id="fact-keep", kind="explore", state="succeeded", params={}, idempotency_key="fk")
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess",
        result_dict={"gain_pct": 6.0, "output_throughput": 950.0},
        kept=True,
    )
    assert amends and "append_lesson" in amends[0]


@pytest.mark.asyncio
async def test_record_fact_per_task_writes_pitfall(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    coord.recipe_kb = object()
    amends: list[dict] = []
    monkeypatch.setattr(coord.proposals, "_kb_amend_recipe", lambda **k: amends.append(k))
    monkeypatch.setattr(coord.writeback, "_pitfall_severity_for", lambda rd: "high")
    task = Task(task_id="fact-revert", kind="integrate_patch", state="failed", params={}, idempotency_key="fr")
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess",
        result_dict={"error_class": "oom", "reason": "bad"},
        kept=False,
    )
    assert amends and "append_pitfall" in amends[0]


# -- _plateau_advisory_block (triggered) ------------------------------------
@pytest.mark.asyncio
async def test_plateau_advisory_reports_the_config_arm_alone_as_not_a_plateau(coord: Coordinator, monkeypatch) -> None:
    """One dry arm is not a plateau: the phase stays open on the other lever."""
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = ps.PHASE_FRAMEWORK_AGENT
    monkeypatch.setattr(
        ps, "compute_plateau_explore", lambda *a, **k: (True, {"recent_keep_gain_pct": 0.1, "empty_streak": 3})
    )
    monkeypatch.setattr(ps, "source_arm_plateaued", lambda *a, **k: (False, {}))
    out = coord._plateau_advisory_block()
    assert "OPTIMIZE config arm plateaued" in out
    assert "Only one arm is dry" in out


@pytest.mark.asyncio
async def test_plateau_advisory_reports_the_source_arm_alone_as_not_a_plateau(coord: Coordinator, monkeypatch) -> None:
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = ps.PHASE_FRAMEWORK_AGENT
    monkeypatch.setattr(ps, "compute_plateau_explore", lambda *a, **k: (False, {}))
    monkeypatch.setattr(
        ps,
        "source_arm_plateaued",
        lambda *a, **k: (True, {"source_consecutive_no_keep": 3, "source_candidates_exhausted": True}),
    )
    out = coord._plateau_advisory_block()
    assert "OPTIMIZE source arm plateaued" in out
    assert "Only one arm is dry" in out


@pytest.mark.asyncio
async def test_plateau_advisory_both_arms_dry_states_the_advance(coord: Coordinator, monkeypatch) -> None:
    """Both arms dry is the condition the phase actually leaves on."""
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = ps.PHASE_FRAMEWORK_AGENT
    monkeypatch.setattr(
        ps, "compute_plateau_explore", lambda *a, **k: (True, {"recent_keep_gain_pct": 0.1, "empty_streak": 3})
    )
    monkeypatch.setattr(ps, "source_arm_plateaued", lambda *a, **k: (True, {"source_consecutive_no_keep": 3}))
    out = coord._plateau_advisory_block()
    assert "OPTIMIZE config arm plateaued" in out
    assert "OPTIMIZE source arm plateaued" in out
    assert "Only one arm is dry" not in out
    assert "KERNEL_AGENT" in out


@pytest.mark.asyncio
async def test_plateau_advisory_kernel_triggered(coord: Coordinator, monkeypatch) -> None:
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = ps.PHASE_KERNEL_AGENT
    monkeypatch.setattr(ps, "compute_plateau_kernel", lambda *a, **k: (True, {"revert_streak": 4}))
    out = coord._plateau_advisory_block()
    assert "KERNEL_AGENT plateau detected" in out


@pytest.mark.asyncio
# -- _record_specialist_result ----------------------------------------------
@pytest.mark.asyncio
async def test_record_specialist_result_with_proposals(coord: Coordinator) -> None:
    task = _ptask("rec-spec-1", "specialist")
    await coord._record_specialist_result(
        task=task,
        done_payload={
            "domain": "kernel_agent",
            "gap_canonical_id": "g1",
            "proposal_set": [{"name": "fuse-moe"}],
            "summary": "found one",
            "confidence": 0.8,
        },
        source="specialist:rec-spec-1",
    )
    last = coord.shared_state.last_specialist
    assert last.get("task_id") == "rec-spec-1"


@pytest.mark.asyncio
async def test_record_specialist_result_no_dead_research_evidence_log(
    coord: Coordinator,
    caplog,
) -> None:
    """Successful specialist recording must not emit the
    research-evidence failure log."""
    import logging

    task = _ptask("rec-spec-dead", "specialist")
    with caplog.at_level(logging.ERROR):
        await coord._record_specialist_result(
            task=task,
            done_payload={
                "domain": "kernel_agent",
                "proposal_set": [{"name": "p1"}],
            },
            source="specialist:rec-spec-dead",
        )
    assert not any("research evidence aggregation failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_record_specialist_result_research_scout(coord: Coordinator, monkeypatch) -> None:
    task = _ptask("rec-spec-2", "specialist")
    harvested: list[dict] = []

    async def harvest(done_payload):
        harvested.append(done_payload)

    monkeypatch.setattr(coord, "_harvest_research_scout", harvest)
    await coord._record_specialist_result(
        task=task,
        done_payload={
            "domain": "research_scout_specialist",
            "proposal_set": [],
            "empty": True,
            "research": {"hints": {}},
        },
        source="specialist:rec-spec-2",
    )
    assert harvested


@pytest.mark.asyncio
async def test_record_specialist_result_with_scorer(coord: Coordinator) -> None:
    class _Scorer:
        async def score(self, *, gap, proposals):
            return {"models": ["m1"], "ranking": [0]}

    coord._proposal_scorer = _Scorer()
    task = _ptask("rec-spec-3", "specialist")
    await coord._record_specialist_result(
        task=task,
        done_payload={
            "domain": "kernel_agent",
            "proposal_set": [{"name": "p1"}],
        },
        source="specialist:rec-spec-3",
    )


# -- finalize_recipe_and_journal (KB path) ---------------------------
class _FakeLocal:
    def get_recipe(self, *, canonical_id):
        return {
            "best_throughput": 0.0,
            "sessions": [],
            "kernel_optimizations": [],
            "stack_fingerprint": {},
        }


class _FakeRecipeKB:
    def __init__(self) -> None:
        self.local = _FakeLocal()


@pytest.mark.asyncio
async def test_recipe_kb_finalize_skips_without_model(coord: Coordinator) -> None:
    coord.recipe_kb = _FakeRecipeKB()
    coord.shared_state.model_name = ""  # missing model -> skip update_recipe
    coord.shared_state.gpu_type = "mi300x"
    coord.finalize_recipe_and_journal()


@pytest.mark.asyncio
async def test_recipe_kb_finalize_amends_recipe(coord: Coordinator, monkeypatch) -> None:
    coord.recipe_kb = _FakeRecipeKB()
    coord.shared_state.model_name = "llama"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.cumulative_gain_validated = 12.0
    coord.shared_state.current_best = {"tput": 950.0}
    amends: list[dict] = []
    monkeypatch.setattr(coord.proposals, "_kb_amend_recipe", lambda **k: amends.append(k))
    coord.finalize_recipe_and_journal()
    assert amends and "recipe_overrides" in amends[0]


# -- _run_action_now_sync ---------------------------------------------------
def test_run_action_now_sync_disabled(coord: Coordinator) -> None:
    coord._inline_fast_actions_enabled = False
    out = coord._run_action_now_sync("report")
    assert "disabled" in out


def test_run_action_now_sync_requires_name(coord: Coordinator) -> None:
    coord._inline_fast_actions_enabled = True
    assert "action_name required" in coord._run_action_now_sync("")


def test_run_action_now_sync_not_whitelisted(coord: Coordinator, monkeypatch) -> None:
    coord._inline_fast_actions_enabled = True
    monkeypatch.setattr(coord.dispatcher, "_inline_action_whitelist", lambda: {"report"})
    out = coord._run_action_now_sync("explore")
    assert "not inline-eligible" in out


def test_run_action_now_sync_no_loop(coord: Coordinator, monkeypatch) -> None:
    coord._inline_fast_actions_enabled = True
    monkeypatch.setattr(coord.dispatcher, "_inline_action_whitelist", lambda: {"report"})
    coord._coordinator_loop = None
    out = coord._run_action_now_sync("report")
    assert "coordinator loop not running" in out


# -- _handle_intent routing -------------------------------------------------
@pytest.mark.asyncio
async def test_handle_intent_policy_denied(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.policy.gate import PolicyDenied

    recorded: list = []

    def _deny(source, intent):
        raise PolicyDenied("nope")

    monkeypatch.setattr(coord.policy, "validate_intent", _deny)

    async def _rec(source, intent, denied):
        recorded.append(denied)

    monkeypatch.setattr(coord.writeback, "_record_policy_denied", _rec)
    await coord._handle_intent("orchestration", _heartbeat())
    assert recorded


@pytest.mark.asyncio
async def test_handle_intent_handler_exception_is_recorded(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.policy, "validate_intent", lambda s, i: None)

    async def _boom(source, intent):
        raise RuntimeError("handler boom")

    monkeypatch.setattr(coord, "_handle_send_message", _boom)
    await coord._handle_intent("orchestration", _heartbeat())


@pytest.mark.asyncio
async def test_handle_intent_routes_rare_types(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.policy, "validate_intent", lambda s, i: None)
    seen: list[str] = []

    routes = {
        IntentType.PRUNE_BRANCH: "_handle_prune_branch",
        IntentType.ALERT: "_handle_alert",
        IntentType.UPDATE_STATE: "_handle_update_state",
    }
    for it, attr in routes.items():

        async def _h(source, intent, _n=attr):
            seen.append(_n)

        monkeypatch.setattr(coord, attr, _h)
    for it in routes:
        await coord._handle_intent("orchestration", Intent(type=it, payload={}))
    assert len(seen) == len(routes)


# -- _advance_phase_if_needed -----------------------------------------------
@pytest.mark.asyncio
async def test_advance_phase_noop_when_already_there(coord: Coordinator, monkeypatch) -> None:
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = "FRAMEWORK_AGENT"
    monkeypatch.setattr(ps, "compute_next_phase", lambda *a, **k: ("FRAMEWORK_AGENT", "x", {}))

    async def _scout():
        return None

    monkeypatch.setattr(coord.phase_internal, "_maybe_enqueue_explore_research_scout", _scout)
    await coord._advance_phase_if_needed()


@pytest.mark.asyncio
async def test_advance_phase_escalation_transition(coord: Coordinator, monkeypatch) -> None:
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = "PRELUDE"
    monkeypatch.setattr(
        ps,
        "compute_next_phase",
        lambda *a, **k: ("FRAMEWORK_AGENT", "robustness_escalated", {"evidence": "llm_escalation"}),
    )

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert (coord.shared_state.phase or "").upper() == "FRAMEWORK_AGENT"


@pytest.mark.asyncio
async def test_advance_phase_terminal_sets_stop_reason(coord: Coordinator, monkeypatch) -> None:
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = "SWEEP"
    coord.shared_state.stop_reason = ""
    monkeypatch.setattr(
        ps, "compute_next_phase", lambda *a, **k: (ps.PHASE_CLOSE, "target_reached", {"terminal": True})
    )

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert coord.shared_state.stop_reason == "target_reached"


@pytest.mark.asyncio
async def test_advance_phase_hint_survives_arrival_at_its_consumer(coord: Coordinator, monkeypatch) -> None:
    """A hint set during PRELUDE must survive PRELUDE -> FRAMEWORK_AGENT.

    ``exit_normal_optimize`` is the hint's only consumer and it runs in
    FRAMEWORK_AGENT, so discarding on the transition that arrives there drops
    the hint on the doorstep of the rule that reads it.
    """
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = "PRELUDE"
    coord.shared_state.pending_escalate_hint = "skip_to_kernel"
    monkeypatch.setattr(ps, "compute_next_phase", lambda *a, **k: ("FRAMEWORK_AGENT", "prelude_done", {}))

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert (coord.shared_state.phase or "").upper() == "FRAMEWORK_AGENT"
    assert coord.shared_state.pending_escalate_hint == "skip_to_kernel"


@pytest.mark.asyncio
async def test_advance_phase_hint_discarded_when_not_headed_to_its_consumer(coord: Coordinator, monkeypatch) -> None:
    """A pending hint is genuinely stale once the target is not the phase whose
    exit rule reads it -- it can never reach that check again -- so this is the
    one case the unrelated-transition cleanup should still clear it.

    A discard is not a consumption: it must land in last_discarded_escalate_hint,
    not last_consumed_escalate_hint, which specifically means "this hint drove
    a transition" and this one never did.
    """
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.pending_escalate_hint = "skip_to_kernel"
    monkeypatch.setattr(ps, "compute_next_phase", lambda *a, **k: ("SWEEP", "some_other_reason", {}))

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert (coord.shared_state.phase or "").upper() == "SWEEP"
    assert coord.shared_state.pending_escalate_hint == ""
    assert coord.shared_state.last_discarded_escalate_hint == "skip_to_kernel"
    assert coord.shared_state.last_discarded_escalate_hint_ts
    assert coord.shared_state.last_consumed_escalate_hint == ""


@pytest.mark.asyncio
async def test_advance_phase_hint_consumed_when_it_drove_the_transition(coord: Coordinator, monkeypatch) -> None:
    """The complementary case: a hint-driven transition must record consumption,
    not a discard, so the two are distinguishable in the breakdown.
    """
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.pending_escalate_hint = "skip_to_kernel"
    monkeypatch.setattr(
        ps,
        "compute_next_phase",
        lambda *a, **k: ("KERNEL_AGENT", "skip_to_kernel", {"hint": "skip_to_kernel"}),
    )

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord.phase_machine, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert (coord.shared_state.phase or "").upper() == "KERNEL_AGENT"
    assert coord.shared_state.pending_escalate_hint == ""
    assert coord.shared_state.last_consumed_escalate_hint == "skip_to_kernel"
    assert coord.shared_state.last_consumed_escalate_hint_ts
    assert coord.shared_state.last_discarded_escalate_hint == ""


# -- _materialize_approved_proposal -----------------------------------------
def _pending(action_name: str, payload: dict, msg_id: str = "prop-1"):
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    return PendingProposal(
        proposal_msg_id=msg_id,
        from_agent="orchestration",
        action_name=action_name,
        predicted_gain_pct=3.0,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_direct_integrate_proposal_inherits_specialist_owner(
    coord: Coordinator,
    monkeypatch,
) -> None:
    specialist = await coord.tasks.create(
        kind="specialist",
        params={
            "source_phase": "FRAMEWORK_AGENT",
            "domain": "serving_specialist",
            "gap_layer": "framework",
        },
        idempotency_key="owner-source",
    )
    monkeypatch.setattr(
        coord.router,
        "_admission_denial_for_action",
        lambda _action: None,
    )
    await coord._handle_propose_action(
        "orchestration",
        Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={
                "action_name": "integrate_patch",
                "params": {"specialist_task_id": specialist.task_id},
            },
        ),
    )

    pending = next(iter(coord.state.pending_proposals.values()))
    params = pending.payload["params"]
    assert params["source_phase"] == "FRAMEWORK_AGENT"
    assert params["domain"] == "serving_specialist"
    assert params["gap_layer"] == "framework"


def test_specialist_owner_is_frozen_at_creation_outside_agent_phases(
    coord: Coordinator,
) -> None:
    coord.shared_state.phase = "KERNEL_AGENT"
    explore_params = {
        "domain": "serving_specialist",
        "gap_layer": "perf_explore",
    }
    framework_params = {
        "domain": "serving_specialist",
        "gap_layer": "framework",
    }

    assert coord.router._stamp_specialist_owner(explore_params) == "EXPLORE"
    assert explore_params["source_phase"] == "EXPLORE"
    assert coord.router._stamp_specialist_owner(framework_params) == "FRAMEWORK_AGENT"
    assert framework_params["source_phase"] == "FRAMEWORK_AGENT"


def test_forward_integrate_source_has_no_current_phase_fallback() -> None:
    from hyperloom.orchestrator.phases.explore import _forward_integrate_source

    forwarded: dict = {}
    _forward_integrate_source({}, forwarded)
    assert "source_phase" not in forwarded


@pytest.mark.asyncio
async def test_materialize_explore_filters_grid(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    pending = _pending(
        "explore",
        {
            "params": {
                "grid": [
                    {"name": "v0"},
                    {"name": "v1"},
                    "non-dict-slot",
                ]
            }
        },
    )
    await coord._materialize_approved_proposal(
        pending,
        approved_variant_names={"v0"},
    )
    tail = await coord.bus.tail(topic="decision", n=10)
    assert any(m.payload.get("kind") == "approved_proposal" for m in tail)


@pytest.mark.asyncio
async def test_materialize_integrate_patch_rejects_missing_owner(
    coord: Coordinator,
) -> None:
    coord.shared_state.baseline_tput = 800.0
    pending = _pending(
        "integrate_patch",
        {"params": {"specialist_task_id": "missing-specialist"}},
        msg_id="prop-ownerless",
    )
    coord.state.pending_proposals[pending.proposal_msg_id] = pending

    await coord._materialize_approved_proposal(pending)

    assert not [task for task in await coord.tasks.queued() if task.kind == "integrate_patch"]
    assert pending.proposal_msg_id not in coord.state.pending_proposals
    assert coord.shared_state.get_specialist_patch_verdict("missing-specialist") == "owner_missing"
    observations = await coord.bus.tail(topic="observation", n=10)
    assert any(message.payload.get("reason") == "integrate_patch_owner_missing" for message in observations)


@pytest.mark.asyncio
async def test_materialize_sweep_stamps_base(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.current_best = {"tput": 900.0, "extra_server_args": "--tp 1"}
    coord.shared_state.baseline_config_path = "/tmp/base.yaml"
    pending = _pending("sweep", {"params": {}}, msg_id="prop-sweep")
    await coord._materialize_approved_proposal(pending)
    task = await coord.tasks.get((await coord.tasks.queued())[0].task_id)
    assert task.kind == "sweep"


@pytest.mark.asyncio
async def test_materialize_explore_seeds_cumulative_env_base(coord: Coordinator) -> None:
    # Regression: explore must inherit current_best.extra_envs as its env base,
    # else the accepted stack's envs collapse to the last variant's delta.
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.current_best = {
        "tput": 900.0,
        "extra_server_args": "--kv-cache-dtype fp8",
        "extra_envs": {"VLLM_ROCM_USE_AITER_MHA": "1", "HIP_FORCE_DEV_KERNARG": "1"},
    }
    pending = _pending("explore", {"params": {"grid": [{"name": "v0"}]}}, msg_id="prop-env")
    await coord._materialize_approved_proposal(pending)
    task = await coord.tasks.get((await coord.tasks.queued())[0].task_id)
    assert task.params["base_extra_args"] == "--kv-cache-dtype fp8"
    assert task.params["base_extra_envs"] == {
        "VLLM_ROCM_USE_AITER_MHA": "1",
        "HIP_FORCE_DEV_KERNARG": "1",
    }


@pytest.mark.asyncio
async def test_materialize_duplicate_idempotency_skips(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    pending = _pending("profile", {"params": {}}, msg_id="prop-dup")
    await coord._materialize_approved_proposal(pending)
    await coord._materialize_approved_proposal(pending)


@pytest.mark.asyncio
async def test_materialize_baseline_ignores_params_outside_fingerprint(coord: Coordinator) -> None:
    await coord._materialize_approved_proposal(_pending("baseline", {"params": {}}, msg_id="prop-b0"))
    await coord._materialize_approved_proposal(
        _pending("baseline", {"params": {"tag": "x"}}, msg_id="prop-b1"),
    )
    queued = [t for t in await coord.tasks.queued() if t.kind == "baseline"]
    assert len(queued) == 1
    tail = await coord.bus.tail(topic="observation", n=20)
    assert any(m.payload.get("reason") == "duplicate_proposal_content" for m in tail)


@pytest.mark.asyncio
async def test_materialize_baseline_distinct_envs_queue_separately(coord: Coordinator) -> None:
    await coord._materialize_approved_proposal(_pending("baseline", {"params": {}}, msg_id="prop-e0"))
    await coord._materialize_approved_proposal(
        _pending("baseline", {"params": {"extra_envs": {"VLLM_ROCM_USE_AITER_MOE": "0"}}}, msg_id="prop-e1"),
    )
    queued = [t for t in await coord.tasks.queued() if t.kind == "baseline"]
    assert len(queued) == 2


@pytest.mark.asyncio
async def test_materialize_requeues_same_content_after_terminal_twin(coord: Coordinator) -> None:
    await coord._materialize_approved_proposal(_pending("baseline", {"params": {}}, msg_id="prop-t0"))
    first = [t for t in await coord.tasks.queued() if t.kind == "baseline"][0]
    await coord.tasks.transition(first.task_id, "running")
    await coord.tasks.transition(first.task_id, "failed")
    await coord._materialize_approved_proposal(_pending("baseline", {"params": {}}, msg_id="prop-t1"))
    queued = [t for t in await coord.tasks.queued() if t.kind == "baseline"]
    assert len(queued) == 1
    assert queued[0].task_id != first.task_id


# -- _handle_delegate branches ----------------------------------------------
def _delegate(action_name: str, key: str, params=None) -> Intent:
    payload = {"action_name": action_name, "params": params or {}, "idempotency_key": key}
    return Intent(type=IntentType.DELEGATE, payload=payload)


@pytest.mark.asyncio
async def test_handle_delegate_pruned_advisory(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.baseline_tput = 800.0
    monkeypatch.setattr(coord.shared_state, "is_pruned", lambda a: True)
    monkeypatch.setattr(coord.dispatcher, "_sequence_denial_for_action", lambda a: None)
    await coord._handle_delegate("orchestration", _delegate("explore", "d-pruned"))
    assert await coord.tasks.queued()


@pytest.mark.asyncio
async def test_handle_delegate_sequence_denied(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.policy.gate import PolicyDenied

    monkeypatch.setattr(
        coord.dispatcher,
        "_sequence_denial_for_action",
        lambda a: PolicyDenied(
            "blocked",
            rule="exec_order",
            hint="wait",
        ),
    )
    recorded: list = []

    async def _rec(source, intent, denied, action_name=None):
        recorded.append(denied)

    monkeypatch.setattr(coord.writeback, "_record_policy_denied", _rec)
    await coord._handle_delegate("orchestration", _delegate("explore", "d-seq"))
    assert recorded


@pytest.mark.asyncio
async def test_handle_delegate_duplicate_running_denied(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.baseline_tput = 800.0
    monkeypatch.setattr(coord.dispatcher, "_sequence_denial_for_action", lambda a: None)
    await coord._handle_delegate("orchestration", _delegate("explore", "d-same"))
    recorded: list = []

    async def _rec(source, intent, denied, action_name=None):
        recorded.append(denied)

    monkeypatch.setattr(coord.writeback, "_record_policy_denied", _rec)
    # Same key while the first task is still queued (non-terminal) -> denied.
    await coord._handle_delegate("orchestration", _delegate("explore", "d-same"))
    assert recorded


# -- _maybe_autosubmit_specialist_patches early returns ---------------------
def _make_real_patch(coord: Coordinator, sid: str) -> None:
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    wt = runs_dir(coord.session_dir, "specialist", sid) / "worktree"
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "kernel.py").write_text("# patched\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_autosubmit_returns_when_verdict_exists(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    sid = "spec-verdict"
    _make_real_patch(coord, sid)
    monkeypatch.setattr(coord.shared_state, "get_specialist_patch_verdict", lambda s: {"verdict": "approve"})
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="kv1")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": ["kernel.py"]},
    )
    assert len(coord.state.pending_proposals) == n_before


@pytest.mark.asyncio
async def test_autosubmit_returns_when_review_in_flight(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    sid = "spec-inflight"
    _make_real_patch(coord, sid)
    coord.state.pending_proposals["existing"] = PendingProposal(
        proposal_msg_id="existing",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"params": {"specialist_task_id": sid}},
    )
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="kv2")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": ["kernel.py"]},
    )
    assert len(coord.state.pending_proposals) == n_before


# -- _promote_warm_replay branches ------------------------------------------
def _warm_task():
    from hyperloom.orchestrator.state.task_registry import Task

    return Task(
        task_id="warm-x",
        kind="replay_warm_recipe",
        state="running",
        params={"extra_envs": {"HSA_FORCE": "1"}, "baseline_tput_anchor": 800.0},
        idempotency_key="warm-k",
    )


def test_promote_warm_replay_already_pushed(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.optimization_stack = [{"action": "replay_warm_recipe"}]
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0},
        task=_warm_task(),
    )
    n = sum(
        1
        for e in coord.shared_state.optimization_stack
        if isinstance(e, dict) and e.get("action") == "replay_warm_recipe"
    )
    assert n == 1


# -- finalize_recipe_and_journal (rich existing row merge) -----------
class _FakeLocalRich:
    def get_recipe(self, *, canonical_id):
        return {
            "best_throughput": 100.0,
            "sessions": [{"session_id": "other-session"}],
            "kernel_optimizations": [{"kernel_id": "k-old"}],
            "stack_fingerprint": {"sglang": "0.1"},
        }


class _FakeRecipeKBRich:
    def __init__(self) -> None:
        self.local = _FakeLocalRich()

    def get_authoritative_recipe(self, *, canonical_id):
        return self.local.get_recipe(canonical_id=canonical_id)


@pytest.mark.asyncio
async def test_recipe_kb_finalize_merges_existing_row(coord: Coordinator, monkeypatch) -> None:
    coord.recipe_kb = _FakeRecipeKBRich()
    coord.shared_state.model_name = "llama"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.cumulative_gain_validated = 15.0
    coord.shared_state.current_best = {"tput": 999.0}
    amends: list[dict] = []
    monkeypatch.setattr(coord.proposals, "_kb_amend_recipe", lambda **k: amends.append(k))
    coord.finalize_recipe_and_journal()
    assert amends
    overrides = amends[0]["recipe_overrides"]
    assert any(s.get("session_id") == "other-session" for s in overrides["sessions"])


# -- _on_enter_close 7-step sequencer ---------------------------------------
@pytest.mark.asyncio
async def test_on_enter_close_runs_full_sequence(coord: Coordinator, monkeypatch) -> None:
    async def _fake_run(task, **kw):
        from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

        return SubAgentResult(task_id=task.task_id, state="succeeded", result={}, error=None)

    monkeypatch.setattr(coord.sub, "run_task", _fake_run)
    await coord._on_enter_close(from_phase="SWEEP")
    assert coord.shared_state.close_sequence_done is True
    assert coord.shared_state.stop_reason


# -- _pump_framework_agent_phase -----------------------------------------------
def _enter_framework(coord: Coordinator) -> None:
    import hyperloom.orchestrator.phases.machine_state as ps

    coord.shared_state.phase = ps.PHASE_FRAMEWORK_AGENT
    coord.shared_state.framework_agent_phase_done = False


@pytest.mark.asyncio
async def test_pump_framework_agent_wrong_phase_noop(coord: Coordinator) -> None:
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    await coord._pump_framework_agent_phase()


@pytest.mark.asyncio
async def test_pump_framework_agent_phase_done_noop(coord: Coordinator) -> None:
    _enter_framework(coord)
    coord.shared_state.framework_agent_phase_done = True
    await coord._pump_framework_agent_phase()


@pytest.mark.asyncio
async def test_pump_framework_agent_skips_when_task_inflight(coord: Coordinator) -> None:
    _enter_framework(coord)
    await coord.tasks.create(kind="framework_agent", params={}, idempotency_key="fpr-inflight")
    await coord._pump_framework_agent_phase()


@pytest.mark.asyncio
async def test_pump_framework_agent_discover_empty_marks_done(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.framework import client as _fa_client

    _enter_framework(coord)
    # Arm disabled: discovery exhaustion falls back to the historical exit
    # (the enabled arm pivots to local exploration instead — covered separately).
    coord.shared_state.framework_local_explore_enabled = False
    coord.shared_state.framework_agent_discover_failures = 0
    # Discovery has spent its retry budget, so the upstream lane declines and
    # the tick reaches the terminal rung.
    coord.shared_state.framework_agent_empty_discoveries = _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
    monkeypatch.setattr(coord.phase_framework, "_select_next_framework_agent_candidate", lambda: None)
    monkeypatch.setattr(coord.phase_framework, "_record_framework_agent_phase_done", lambda **k: None)
    await coord._pump_framework_agent_phase()
    assert coord.shared_state.framework_agent_phase_done is True


@pytest.mark.asyncio
async def test_pump_framework_agent_submits_candidate_proposal(coord: Coordinator, monkeypatch) -> None:
    """The pump submits the candidate as a proposal instead of enqueuing inline."""
    _enter_framework(coord)
    candidate = {
        "candidate_id": "c1",
        "pr_url": "https://example.com/pr/1",
        "batch_id": "b1",
        "route": "direct_framework",
    }
    monkeypatch.setattr(
        coord.phase_framework,
        "_select_next_framework_agent_candidate",
        lambda: candidate,
    )

    await coord._pump_framework_agent_phase()

    pendings = [p for p in coord.state.pending_proposals.values() if p.action_name == "integrate_patch"]
    assert len(pendings) == 1
    payload = pendings[0].payload
    assert payload["framework_agent_candidate_id"] == "c1"
    assert payload["audit_step"] == "direct_framework"
    queued = await coord.tasks.queued()
    assert not [t for t in queued if getattr(t, "kind", "") == "integrate_patch"]


@pytest.mark.asyncio
async def test_pump_framework_agent_dedup_does_not_resubmit(coord: Coordinator, monkeypatch) -> None:
    """A candidate already awaiting its verdict is not re-submitted on the next tick."""
    _enter_framework(coord)

    candidate = {"candidate_id": "c1", "batch_id": "b1", "route": "direct_framework"}
    monkeypatch.setattr(
        coord.phase_framework,
        "_select_next_framework_agent_candidate",
        lambda: candidate,
    )

    await coord._pump_framework_agent_phase()
    await coord._pump_framework_agent_phase()
    pendings = [p for p in coord.state.pending_proposals.values() if p.action_name == "integrate_patch"]
    assert len(pendings) == 1


@pytest.mark.asyncio
async def test_framework_agent_reject_records_critic_denied(coord: Coordinator) -> None:
    """A reject verdict on a framework_agent candidate proposal writes a critic_denied progress row."""
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    pending = PendingProposal(
        proposal_msg_id="m1",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"framework_agent_candidate_id": "c1", "batch_id": "b1"},
    )
    coord.state.pending_proposals["m1"] = pending
    await coord._handle_single_verdict(
        source="critic",
        pending=pending,
        verdict="reject",
        reasoning="unsafe",
    )
    prog = coord.shared_state.framework_agent_phase_progress
    assert any(p.get("status") == "critic_denied" and p.get("candidate_id") == "c1" for p in prog)


@pytest.mark.asyncio
async def test_framework_agent_approve_routes_to_enqueue(coord: Coordinator, monkeypatch) -> None:
    """An approve verdict routes a ``direct_framework`` candidate to the raw-diff enqueue helper."""
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    enq: list = []

    async def _enqueue(cand):
        enq.append(cand)

    monkeypatch.setattr(coord.phase_framework, "_enqueue_framework_agent_task", _enqueue)
    pending = PendingProposal(
        proposal_msg_id="m2",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={
            "framework_agent_candidate_id": "c2",
            "batch_id": "b2",
            "candidate": {"candidate_id": "c2", "batch_id": "b2"},
            "audit_step": "direct_framework",
        },
    )
    coord.state.pending_proposals["m2"] = pending
    await coord._handle_single_verdict(
        source="critic",
        pending=pending,
        verdict="approve",
        reasoning="ok",
    )
    assert enq


# -- _session_integrated_kernel_patch (post-opt roofline gate) ---------------
@pytest.mark.parametrize(
    "action",
    ["integrate", "integrate_patch", "gemm_tuning", "geak_e2e"],
)
def test_post_opt_roofline_gate_true_for_kernel_level_actions(coord: Coordinator, action: str) -> None:
    """Any kernel-level optimization gates the post-opt roofline on."""
    coord.shared_state.optimization_stack = [{"action": action}]
    assert coord._session_integrated_kernel_patch() is True


def test_post_opt_roofline_gate_false_for_param_search_only(coord: Coordinator) -> None:
    """Pure param-search does not trigger the extra profile."""
    coord.shared_state.optimization_stack = [{"action": "explore"}, {"action": "sweep"}]
    assert coord._session_integrated_kernel_patch() is False


def test_post_opt_roofline_gate_false_for_empty_stack(coord: Coordinator) -> None:
    coord.shared_state.optimization_stack = []
    assert coord._session_integrated_kernel_patch() is False


def test_post_opt_roofline_gate_ignores_non_dict_entries(coord: Coordinator) -> None:
    """Malformed (non-dict) stack entries are skipped without raising."""
    coord.shared_state.optimization_stack = ["bad", {"action": "gemm_tuning"}]
    assert coord._session_integrated_kernel_patch() is True


@pytest.mark.asyncio
async def test_run_action_now_sync_on_loop_thread_emits_audit(coord: Coordinator, monkeypatch, caplog) -> None:
    # Defensive audit (log-only): invoking the run_action_now sync bridge on
    # the coordinator loop thread must emit a log-only audit.
    # run_coroutine_threadsafe is stubbed so the test never actually blocks.
    import asyncio
    import logging

    coord._inline_fast_actions_enabled = True
    monkeypatch.setattr(coord.dispatcher, "_inline_action_whitelist", lambda: {"report"})
    coord._coordinator_loop = asyncio.get_running_loop()

    class _ImmediateFuture:
        def result(self, timeout=None):
            return "(stubbed inline result)"

    def _fake_schedule(coro, loop):
        coro.close()
        return _ImmediateFuture()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_schedule)

    with caplog.at_level(logging.WARNING, logger="hyperloom.orchestrator.loop.dispatcher"):
        out = coord._run_action_now_sync("report")

    assert any("run_action_now:" in r.getMessage() for r in caplog.records)
    assert "stubbed inline result" in out
