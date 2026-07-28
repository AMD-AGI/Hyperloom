# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``WritebackCollaborator._promote_to_shared_state``:
per-task_kind state writes, audit rows, and sweep/conc_sweep early-return."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.state.shared_state import _AUDIT_ACTIONS
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.state.task_registry import Task


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel_agent": MockBackend(silent, name="kernel_agent"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }


def _coord(session_dir: Path) -> Coordinator:
    return Coordinator(session_dir, backends=_silent_backends())


def _task(kind: str, *, task_id: str = "t1", params: dict | None = None) -> Task:
    return Task(
        task_id=task_id,
        kind=kind,
        state="running",
        params=params or {},
        idempotency_key=f"{kind}-{task_id}",
    )


def _count_record_attempt(coord: Coordinator, monkeypatch) -> list[dict]:
    """Spy that records every record_action_attempt call's kwargs, forwarding to the real impl."""
    calls: list[dict] = []
    real = coord.shared_state.record_action_attempt

    def spy(*args, **kwargs):
        # record_action_attempt is called as (action=..., ...) keyword in prod code.
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(coord.shared_state, "record_action_attempt", spy)
    return calls


# ---------------------------------------------------------------------------
# GAP 1: sweep / conc_sweep early-return double-track — each records + saves +
# returns on its own, so the unified tail record_action_attempt must not re-fire.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_sweep_records_once_and_returns_before_tail(session_dir, monkeypatch):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.conc_sweep_enabled = False
    calls = _count_record_attempt(coord, monkeypatch)

    await coord._promote_to_shared_state(
        "sweep",
        {
            "status": "succeeded",
            "pareto_front": [1, 2, 3],
            "grid_size": 7,
            "output_throughput": 200.0,
        },
        task=_task("sweep"),
    )

    # Exactly ONE record_action_attempt (the in-branch one), never the tail.
    assert len(calls) == 1
    assert calls[0]["action"] == "sweep"
    assert calls[0]["decision"] == "discarded"
    assert calls[0]["status"] == "succeeded"
    # The audit row landed in sweep_attempts; the tail segment never appended a 2nd.
    assert len(s.sweep_attempts) == 1
    # record_sweep ran (discovery bookkeeping) after the audit row.
    assert s.last_sweep  # non-empty snapshot written by record_sweep


@pytest.mark.asyncio
async def test_promote_conc_sweep_records_once_and_returns_before_tail(session_dir, monkeypatch):
    coord = _coord(session_dir)
    s = coord.shared_state
    calls = _count_record_attempt(coord, monkeypatch)

    await coord._promote_to_shared_state(
        "conc_sweep",
        {
            "status": "succeeded",
            "summary": {"best_speedup": 1.3, "best_conc": 8},
        },
        task=_task("conc_sweep"),
    )

    # conc_sweep is NOT in _AUDIT_ACTIONS, so the in-branch record_action_attempt
    # is a no-op recorder, and the tail also skips it. Exactly one CALL, zero effect.
    assert "conc_sweep" not in _AUDIT_ACTIONS
    assert len(calls) == 1
    assert calls[0]["action"] == "conc_sweep"
    assert calls[0]["decision"] == "discarded"
    # No conc_sweep_attempts ledger exists; record_conc_sweep wrote last_conc_sweep.
    assert not hasattr(s, "conc_sweep_attempts")
    assert s.last_conc_sweep.get("status") == "succeeded"
    assert s.last_conc_sweep.get("summary", {}).get("best_speedup") == 1.3


# ---------------------------------------------------------------------------
# GAP 2: changed / audit convergence for baseline / profile / explore / roofline.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_baseline_writes_state_and_audit(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state

    await coord._promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 100.0,
            "warmup_round_tput": 80.0,
            "accuracy": 0.9,
            "subprocess_runtime_sec": 30.0,
        },
        task=_task("baseline"),
    )

    # Hot-measure contract: baseline_tput/hot from tput, cold from warmup anchor.
    assert s.baseline_tput == 100.0
    assert s.baseline_hot_tput == 100.0
    assert s.baseline_cold_tput == 80.0
    assert s.baseline_accuracy == 0.9
    assert s.baseline_runtime_sec == 30.0
    assert s.current_best["action"] == "baseline"
    assert s.current_best["tput"] == 100.0
    # Audit row: promoted with key_metric = output_throughput.
    assert s.last_baseline["decision"] == "promoted"
    assert s.last_baseline["status"] == "succeeded"
    assert s.last_baseline["key_metric"] == 100.0


@pytest.mark.asyncio
async def test_promote_profile_writes_state_and_audit(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "profile",
        {
            "status": "succeeded",
            "main_trace_path": "/tmp/trace.json.gz",
            "output_throughput": 150.0,
        },
        task=_task(
            "profile",
            params={
                "base_extra_args": "--mem-fraction-static=0.9",
                "framework": "vllm",
                "precision": "fp8",
                "model_path": "/models/qwen",
                "tp": 1,
                "conc": 64,
                "isl": 1024,
                "osl": 1024,
                "max_model_len": 4096,
            },
        ),
    )

    assert s.last_profile_trace == "/tmp/trace.json.gz"
    assert s.last_profile_status == "succeeded"
    assert s.last_profile_args == "--mem-fraction-static=0.9"
    assert s.last_profile_workload == s.profile_workload_context(
        {
            "base_extra_args": "--mem-fraction-static=0.9",
            "framework": "vllm",
            "precision": "fp8",
            "model_path": "/models/qwen",
            "tp": 1,
            "conc": 64,
            "isl": 1024,
            "osl": 1024,
            "max_model_len": 4096,
        }
    )
    # +1% rule met (150 vs 100): current_best re-lifted to profile.
    assert s.current_best["action"] == "profile"
    assert s.current_best["tput"] == 150.0
    # Audit row.
    assert s.last_profile["decision"] == "promoted"
    assert s.last_profile["status"] == "succeeded"
    assert s.last_profile["extras"]["trace_path"] == "/tmp/trace.json.gz"
    assert s.last_profile["extras"]["profile_args"] == "--mem-fraction-static=0.9"


@pytest.mark.asyncio
async def test_promote_profile_without_task_uses_shared_state_workload(session_dir):
    coord = _coord(session_dir)
    state = coord.shared_state
    state.framework = "vllm"
    state.precision = "fp8"
    state.model_path = "/models/qwen"
    state.tp = 1
    state.conc = 64
    state.isl = 1024
    state.osl = 1024
    state.max_model_len = 4096
    state.current_best = {
        "extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR": "1"},
    }

    await coord._promote_to_shared_state(
        "profile",
        {
            "status": "succeeded",
            "main_trace_path": "/tmp/trace.json.gz",
            "output_throughput": 100.0,
        },
        task=None,
    )

    assert state.last_profile_workload == state.current_profile_workload_context()


@pytest.mark.asyncio
async def test_promote_explore_promoted_writes_state_and_audit(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    winner = {"name": "v1", "fingerprint": "abc", "tput": 130.0}

    await coord._promote_to_shared_state(
        "explore",
        {
            "explore_search_update": {},
            "winners": [winner],
            "round_id": "r1",
            "best_variant": winner,
            "output_throughput": 130.0,
            "best_gain_pct": 30.0,
        },
        task=_task("explore", params={"gap_canonical_id": "g1"}),
    )

    assert s.current_best["action"] == "explore"
    assert s.current_best["tput"] == 130.0
    accepted = s.explore_search.get("accepted") if isinstance(s.explore_search, dict) else None
    assert isinstance(accepted, list) and len(accepted) == 1
    # Audit row: promoted; extras carry the round + winner stats.
    assert s.last_explore["decision"] == "promoted"
    assert s.last_explore["status"] == "succeeded"
    assert s.last_explore["extras"]["round_id"] == "r1"
    assert s.last_explore["extras"]["winners_count"] == 1


@pytest.mark.asyncio
async def test_promote_roofline_succeeded_writes_audit(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_trace_analyze = {"roofline_snapshot_id": 5, "analysis_md_path": "/tmp/a.md"}

    await coord._promote_to_shared_state(
        "roofline",
        {
            "status": "succeeded",
            "snapshot_id": 5,
            "last_profile_trace": "/tmp/trace.gz",
        },
        task=_task("roofline"),
    )

    # roofline resets its failure streak on a succeeded snapshot.
    assert s.roofline_failure_streak == 0
    # Audit row: promoted; snapshot_id taken from last_trace_analyze snapshot.
    assert s.last_roofline["decision"] == "promoted"
    assert s.last_roofline["status"] == "succeeded"
    assert s.last_roofline["extras"]["snapshot_id"] == 5


# ---------------------------------------------------------------------------
# GAP 3: successful profile with a trace clears the stale trace_analyze cache.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_profile_with_trace_clears_last_trace_analyze(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_trace_analyze = {"stale": True, "roofline_snapshot_id": 9}

    await coord._promote_to_shared_state(
        "profile",
        {
            "status": "succeeded",
            "main_trace_path": "/tmp/trace.json.gz",
            "output_throughput": 150.0,
        },
        task=_task("profile", params={"base_extra_args": "--foo"}),
    )

    assert s.last_trace_analyze == {}


# ---------------------------------------------------------------------------
# GAP 4: profile "skipped" arm audits as skipped and clears the pending roofline
# task, without touching current_best / last_profile_trace.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_profile_skipped_audits_and_clears_pending(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.auto_roofline_pending_task_id = "t1"

    await coord._promote_to_shared_state(
        "profile",
        {
            "status": "skipped",
            "error_class": "no_profiler",
            "error": "profiler disabled",
        },
        task=_task("profile", task_id="t1"),
    )

    # Pending roofline task cleared; audit row is a skipped verdict.
    assert s.auto_roofline_pending_task_id == ""
    assert s.last_profile["decision"] == "skipped"
    assert s.last_profile["extras"]["error_class"] == "no_profiler"
    # Skipped arm never promotes current_best.
    assert not s.current_best or s.current_best.get("action") != "profile"


# ---------------------------------------------------------------------------
# GAP 5: integrate_patch KEEP lifts current_best and clears pending_integrate;
# integrate_patch is NOT in _AUDIT_ACTIONS so no last_integrate_patch is written.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_integrate_patch_kept_lifts_and_clears_pending(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.pending_integrate = {"task_id": "t1"}

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 140.0,
            "specialist_task_id": "spec-1",
            "delta_pct": 40.0,
            "extra_server_args_applied": "--kv-cache-dtype fp8",
            "extra_envs_applied": {"FOO": "1"},
            "workspace": "/w",
        },
        task=_task("integrate_patch", task_id="t1"),
    )

    assert s.current_best["action"] == "integrate_patch"
    assert s.current_best["tput"] == 140.0
    assert s.current_best["extra_server_args"] == "--kv-cache-dtype fp8"
    assert s.current_best["extra_envs"] == {"FOO": "1"}
    # pending_integrate sentinel cleared after the outcome is observed.
    assert s.pending_integrate == {}
    # Not an audited action: no last_integrate_patch attribute is created.
    assert not hasattr(s, "last_integrate_patch")


@pytest.mark.asyncio
async def test_promote_integrate_patch_reverted_keeps_current_best(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.current_best = {"action": "baseline", "tput": 100.0}

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "reverted",
            "output_throughput": 90.0,
            "specialist_task_id": "spec-2",
        },
        task=_task("integrate_patch", task_id="t2"),
    )

    # A reverted patch never lifts current_best.
    assert s.current_best["action"] == "baseline"
    assert s.current_best["tput"] == 100.0


# ---------------------------------------------------------------------------
# GAP 6: framework_agent appends a progress row every time and lifts on KEEP.
# framework_agent is NOT in _AUDIT_ACTIONS.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_framework_agent_kept_lifts_and_records_progress(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "framework_agent",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "delta_pct": 30.0,
            "batch_id": "b1",
            "candidate": {"pr_url": "https://x/pull/1", "ref": "PR:1"},
            "workspace": "/w",
        },
        task=_task("framework_agent", task_id="t1"),
    )

    # One progress row appended; KEEP lifted current_best to the framework arm.
    assert isinstance(s.framework_agent_phase_progress, list)
    assert len(s.framework_agent_phase_progress) == 1
    row = s.framework_agent_phase_progress[0]
    assert row["status"] == "kept"
    assert row["kept"] is True
    assert row["batch_id"] == "b1"
    assert s.current_best["action"] == "framework"
    assert s.current_best["tput"] == 130.0
    assert not hasattr(s, "last_framework_agent")


@pytest.mark.asyncio
async def test_promote_framework_agent_failed_records_progress_no_lift(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.current_best = {"action": "baseline", "tput": 100.0}

    await coord._promote_to_shared_state(
        "framework_agent",
        {
            "status": "reverted",
            "output_throughput": 80.0,
            "delta_pct": -20.0,
            "batch_id": "b1",
            "candidate": {"pr_url": "https://x/pull/2", "ref": "PR:2"},
        },
        task=_task("framework_agent", task_id="t2"),
    )

    assert len(s.framework_agent_phase_progress) == 1
    assert s.framework_agent_phase_progress[0]["kept"] is False
    # No lift on a non-kept candidate.
    assert s.current_best["action"] == "baseline"


# ---------------------------------------------------------------------------
# GAP 7: replay_warm_recipe routes through _promote_warm_replay (self-saves) and
# never sets outcome.changed, so the unified tail neither audits nor re-saves.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_replay_warm_recipe_routes_and_skips_tail(session_dir, monkeypatch):
    coord = _coord(session_dir)
    calls = _count_record_attempt(coord, monkeypatch)

    warm_calls: list[dict] = []

    def _spy_warm(result, *, task=None):
        warm_calls.append({"result": result, "task": task})

    # _promote_warm_replay lives on the writeback collaborator; also stub the
    # deferred PRELUDE analysis enqueue so the test stays hermetic.
    monkeypatch.setattr(coord.writeback, "_promote_warm_replay", _spy_warm)

    async def _noop_prelude(*a, **k):
        return None

    monkeypatch.setattr(
        coord.writeback,
        "_maybe_enqueue_prelude_initial_analysis_after_baseline",
        _noop_prelude,
    )

    await coord._promote_to_shared_state(
        "replay_warm_recipe",
        {"status": "succeeded", "output_throughput": 120.0},
        task=_task("replay_warm_recipe", task_id="t1"),
    )

    # The dedicated warm-replay promote path ran exactly once with the result.
    assert len(warm_calls) == 1
    assert warm_calls[0]["result"]["output_throughput"] == 120.0
    # replay_warm_recipe is not audited by the unified tail.
    assert all(c["action"] != "replay_warm_recipe" for c in calls)


# ---------------------------------------------------------------------------
# GAP 8: roofline failure (status != succeeded/skipped) bumps the failure streak
# and audits as discarded (roofline IS an audited action).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_roofline_failed_bumps_streak_and_audits_discarded(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.roofline_failure_streak = 2
    s.auto_roofline_pending_task_id = "t1"

    await coord._promote_to_shared_state(
        "roofline",
        {
            "status": "failed",
            "phase": "trace_analyze",
            "error_class": "tracelens_error",
            "error": "boom",
        },
        task=_task("roofline", task_id="t1"),
    )

    # Streak incremented; pending pointer cleared.
    assert s.roofline_failure_streak == 3
    assert s.auto_roofline_pending_task_id == ""
    # Audit row: discarded, with the failure context in extras.
    assert s.last_roofline["decision"] == "discarded"
    assert s.last_roofline["status"] == "succeeded"  # record_action_attempt stamps the attempt status
    assert s.last_roofline["extras"]["error_class"] == "tracelens_error"
    assert s.last_roofline["extras"]["phase"] == "trace_analyze"


# ---------------------------------------------------------------------------
# GAP 9: explore resume_stack_revalidate (native, non-GEAK) with a valid tput
# clears resume_pending_revalidation and does NOT promote a variant.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_explore_resume_revalidate_clears_pending(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.current_best = {"action": "explore", "tput": 130.0}
    s.resume_pending_revalidation = True

    await coord._promote_to_shared_state(
        "explore",
        {
            "explore_search_update": {},
            "winners": [],  # revalidation confirms the stack, never adds a variant
            "round_id": "rv1",
            "output_throughput": 128.0,
        },
        task=_task(
            "explore",
            task_id="t1",
            params={"source": "resume_stack_revalidate"},
        ),
    )

    # A valid rebench clears the pending flag; current_best is not re-promoted.
    assert s.resume_pending_revalidation is False
    assert s.current_best["action"] == "explore"
    assert s.current_best["tput"] == 130.0


@pytest.mark.asyncio
async def test_promote_explore_resume_revalidate_keeps_pending_on_empty_rebench(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.resume_pending_revalidation = True

    await coord._promote_to_shared_state(
        "explore",
        {
            "explore_search_update": {},
            "winners": [],
            "round_id": "rv2",
            "output_throughput": None,  # failed/empty rebench
        },
        task=_task(
            "explore",
            task_id="t2",
            params={"source": "resume_stack_revalidate"},
        ),
    )

    # No valid measurement -> the flag stays set so reports keep warning.
    assert s.resume_pending_revalidation is True


# ---------------------------------------------------------------------------
# GAP 10: every _PROMOTE_HANDLERS value resolves to a callable on the class,
# so a typo or unregistered handler is caught at test time, not at runtime.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "task_kind,handler_name",
    list(WritebackCollaborator._PROMOTE_HANDLERS.items()),
)
def test_promote_handlers_are_callable(task_kind, handler_name):
    handler = getattr(WritebackCollaborator, handler_name, None)
    assert callable(handler), f"{task_kind!r} -> {handler_name!r} is not a callable on WritebackCollaborator"


# ---------------------------------------------------------------------------
# Env preservation across layers and source_snapshot propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integrate_keep_preserves_prior_explore_envs(session_dir):
    """An artifact-only integrate KEEP must not erase envs from the explore layer."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1083.0
    s.current_best = {
        "action": "explore",
        "tput": 4616.0,
        "extra_server_args": "--no-scheduler-reserve-full-isl",
        "extra_envs": {"VLLM_ROCM_USE_AITER_MOE": "0"},
    }

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 4700.0,
            "specialist_task_id": "spec-keep",
            "config_changes_applied": {},
        },
        task=_task("integrate_patch", task_id="t-keep"),
    )

    assert s.current_best["extra_envs"].get("VLLM_ROCM_USE_AITER_MOE") == "0", (
        "explore env must survive artifact-only integrate KEEP"
    )


def test_lift_applies_unset_envs_before_new_envs(session_dir):
    coord = _coord(session_dir)
    coord.shared_state.current_best = {
        "action": "explore",
        "tput": 1000.0,
        "extra_server_args": "",
        "extra_envs": {"KEEP": "old", "DROP": "old", "RESTORE": "old"},
    }

    coord._lift_to_current_best(
        "explore",
        1100.0,
        {
            "name": "env-update",
            "extra_server_args": "",
            "extra_envs": {"KEEP": "new", "RESTORE": "new"},
            "unset_envs": ["DROP", "RESTORE"],
        },
    )

    assert coord.shared_state.current_best["extra_envs"] == {
        "KEEP": "new",
        "RESTORE": "new",
    }


@pytest.mark.asyncio
async def test_lift_copies_source_snapshot_into_stack_entry(session_dir):
    """source_snapshot/framework_root/base_sha from the lift bv reach the stack entry."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.current_best = {"action": "baseline", "tput": 1000.0, "extra_server_args": "", "extra_envs": {}}

    coord._lift_to_current_best(
        "integrate_patch",
        1500.0,
        {
            "name": "patch-1",
            "candidate_extra_server_args": "",
            "extra_envs": {},
            "tput": 1500.0,
            "scope": "source_patch",
            "source_snapshot": "/session/optimization_stack/src/abc123",
            "framework_root": "/opt/vllm",
            "base_sha": "deadbeef",
        },
    )

    top = s.optimization_stack[-1]
    assert top.get("source_snapshot") == "/session/optimization_stack/src/abc123"
    assert top.get("framework_root") == "/opt/vllm"
    assert top.get("base_sha") == "deadbeef"
