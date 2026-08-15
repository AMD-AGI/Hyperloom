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

    # Hot-measure contract: baseline_tput is the hot round, not the cold warmup.
    assert s.baseline_tput == 100.0
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
    s.current_best = {
        "action": "explore",
        "engine": "sglang",
        "tput": 100.0,
        "extra_server_args": "--attention-backend aiter",
        "extra_envs": {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tmp/tuned.csv"},
    }

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
    assert s.current_best["engine"] == "sglang"
    assert s.current_best["extra_server_args"] == "--attention-backend aiter"
    assert s.current_best["extra_envs"] == {
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tmp/tuned.csv"
    }
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


@pytest.mark.asyncio
async def test_roofline_with_an_analysis_anchors_the_watermark(session_dir):
    """A roofline that produced an analysis costs the next one a 10% climb."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.cumulative_gain_validated = 75.0
    s.last_roofline_tput = 0.0
    s.last_trace_analyze = {
        "roofline_snapshot_id": 5,
        "analysis_md_path": "/tmp/a.md",
        "analysis_md_text": "# roofline\nattention is 64.8% of GPU time\n",
    }

    await coord._promote_to_shared_state(
        "roofline",
        {"status": "succeeded", "snapshot_id": 5},
        task=_task("roofline"),
    )

    assert s.last_roofline_tput == 175.0


@pytest.mark.asyncio
async def test_roofline_without_an_analysis_leaves_the_watermark_armed(session_dir):
    """An empty analysis must not buy a cycle of silence.

    The anchor is what stops the watermark firing again until throughput climbs
    another 10%. A roofline that recorded nothing once anchored anyway, so the
    specialist kept reading "(none — no fresh roofline snapshot has been
    recorded yet)" while the anchor insisted one had been taken there, and the
    only thing that could have lifted throughput past the anchor was the
    evidence the empty snapshot was standing in for.
    """
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.cumulative_gain_validated = 75.0
    s.last_roofline_tput = 0.0
    s.last_trace_analyze = {"roofline_snapshot_id": 5, "analysis_md_text": ""}

    await coord._promote_to_shared_state(
        "roofline",
        {"status": "succeeded", "snapshot_id": 5},
        task=_task("roofline"),
    )

    assert s.last_roofline_tput == 0.0


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
async def test_integrate_patch_preserves_proposal_owner_across_phase_change(
    session_dir,
):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.phase = "KERNEL_AGENT"

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 110.0,
            "specialist_task_id": "spec-framework",
            "delta_pct": 10.0,
            "extra_server_args_applied": "--quantization fp8_per_channel",
            "workspace": "/w",
        },
        task=_task(
            "integrate_patch",
            task_id="t-cross-phase",
            params={
                "specialist_task_id": "spec-framework",
                "source_phase": "FRAMEWORK_AGENT",
                "domain": "serving_specialist",
                "provenance": "specialist:serving_specialist",
                "gap_canonical_id": "gap.framework.fp8",
                "gap_layer": "framework",
                "framework_agent_authoring": True,
            },
        ),
    )

    entry = s.optimization_stack[0]
    assert entry["source_phase"] == "FRAMEWORK_AGENT"
    assert entry["domain"] == "serving_specialist"
    assert entry["provenance"] == "specialist:serving_specialist"
    assert entry["gap_canonical_id"] == "gap.framework.fp8"
    assert entry["framework_agent_authoring"] is True


@pytest.mark.asyncio
async def test_prebaseline_enablement_patch_is_config_only_not_gain(session_dir):
    """A patch required to establish baseline stays reproducible but has no gain."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 0.0
    s.pending_integrate = {"task_id": "t-enable"}

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "enablement": True,
            "output_throughput": 140.0,
            "specialist_task_id": "spec-enable",
            "extra_server_args_applied": "--mem-fraction-static 0.95",
            "workspace": "/w",
        },
        task=_task(
            "integrate_patch",
            task_id="t-enable",
            params={"enablement": True},
        ),
    )

    assert len(s.optimization_stack) == 1
    entry = s.optimization_stack[0]
    assert entry["action"] == "integrate_patch"
    assert entry["baseline_enablement"] is True
    assert entry["attribution_eligible"] is False
    assert s.gain_per_stack_entry == [None]
    assert s.cumulative_gain == 0.0
    assert s.cumulative_gain_validated == 0.0
    assert s.pending_integrate == {}


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
    s.phase = "KERNEL_AGENT"

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
    assert s.optimization_stack[-1]["source_phase"] == "FRAMEWORK_AGENT"
    assert s.optimization_stack[-1]["provenance"] == "framework_agent"
    # The stack variant must be the canonical candidate key, undecorated, so
    # resume can reconcile it against the recorded framework_agent KEEP.
    assert s.optimization_stack[-1]["variant_name"] == "https://x/pull/1"
    assert not hasattr(s, "last_framework_agent")


@pytest.mark.asyncio
async def test_promote_framework_agent_kept_without_a_candidate_key_is_skipped_loudly(
    session_dir,
    caplog,
):
    """A KEEP with no identity advances current_best but leaves no stack entry.

    Promote used to name the stack entry ``f"framework:{cand_id}"``, which is
    truthy even when the key is empty, so a candidate carrying no candidate_id,
    pr_url or ref was stacked under the bare name ``"framework:"``. The
    undecorated key is falsy, so ``_lift_to_current_best``'s guard now skips the
    append instead.

    Only the append: current_best and cumulative_gain are set unconditionally
    further down, so the win still counts — it just is not recorded as a step
    anything can later reconcile, dedupe or replay. Pinned because that split is
    easy to misread in either direction, and because a KEEP that leaves no trace
    in the stack has to at least leave one in the log.
    """
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.current_best = {"action": "baseline", "tput": 100.0}

    with caplog.at_level("WARNING"):
        await coord._promote_to_shared_state(
            "framework_agent",
            {
                "status": "kept",
                "output_throughput": 130.0,
                "delta_pct": 30.0,
                "batch_id": "b1",
                # No candidate_id, no pr_url, no ref — and no task params to
                # recover one from either.
                "candidate": {},
            },
            task=_task("framework_agent", task_id="t-nameless"),
        )

    # The win lands, ...
    assert s.current_best["action"] == "framework"
    assert s.current_best["tput"] == 130.0
    # ... but nothing in the stack says how it was reached.
    assert not [e for e in s.optimization_stack if str(e.get("action")) == "framework"]
    assert "no candidate key" in caplog.text
    # The outcome is still recorded, so the pump does not re-select it.
    assert len(s.framework_agent_phase_progress) == 1
    assert s.framework_agent_phase_progress[0]["status"] == "kept"


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


def test_lift_is_the_only_writer_so_an_ablated_env_stays_gone(session_dir):
    """A later winner that drops an inherited env must not see it come back.

    Every writer routes through the lift, so the ablation survives on
    current_best without anything replaying the stack to reconstruct it.
    """
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    coord._lift_to_current_best(
        "explore",
        1100.0,
        {"name": "adds-env", "extra_server_args": "--flag-a 1", "extra_envs": {"SGLANG_OLD": "1"}},
    )
    coord._lift_to_current_best(
        "explore",
        1200.0,
        {
            "name": "drops-env",
            "extra_server_args": "--flag-a 1",
            "extra_envs": {"SGLANG_NEW": "1"},
            "unset_envs": ["SGLANG_OLD"],
        },
    )

    assert s.current_best["extra_envs"] == {"SGLANG_NEW": "1"}
    assert [e["variant_name"] for e in s.optimization_stack] == ["adds-env", "drops-env"]


def test_lift_refuses_a_winner_that_does_not_beat_the_anchor(session_dir):
    """A measurement below current_best must leave config and stack untouched."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    coord._lift_to_current_best(
        "explore",
        1500.0,
        {"name": "good", "extra_server_args": "--flag-a 1", "extra_envs": {"A": "1"}},
    )

    assert (
        coord._lift_to_current_best(
            "gemm_tuning",
            1100.0,
            {"name": "worse", "extra_server_args": "--flag-b 2", "extra_envs": {"B": "2"}},
        )
        is False
    )
    assert s.current_best["tput"] == 1500.0
    assert s.current_best["extra_envs"] == {"A": "1"}
    assert [e["variant_name"] for e in s.optimization_stack] == ["good"]


def test_lift_keeps_entry_extra_off_current_best(session_dir):
    """Artifact and provenance handles belong to the stack entry, not the config."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    coord._lift_to_current_best(
        "gemm_tuning",
        1200.0,
        {"name": "geak_a8w8", "extra_server_args": "", "extra_envs": {"AITER_CONFIG": "/tuned.csv"}},
        entry_extra={"tuned_file": "/tuned.csv", "backend": "geak", "empty": "", "absent": None},
    )

    entry = s.optimization_stack[-1]
    assert entry["tuned_file"] == "/tuned.csv"
    assert entry["backend"] == "geak"
    # Empty values are not stamped, so a missing handle stays missing.
    assert "empty" not in entry
    assert "absent" not in entry
    assert "tuned_file" not in s.current_best
    assert "backend" not in s.current_best


def test_lift_carries_the_active_overlay_forward(session_dir):
    """An authored-kernel overlay outlives the KEEP that built it."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    coord._lift_to_current_best(
        "geak_e2e",
        1200.0,
        {"name": "geak", "extra_server_args": "", "extra_envs": {}, "final_overlay": "/overlay/build"},
    )
    assert s.current_best["final_overlay"] == "/overlay/build"
    assert s.optimization_stack[-1]["final_overlay"] == "/overlay/build"

    coord._lift_to_current_best(
        "explore",
        1300.0,
        {"name": "flags-only", "extra_server_args": "--flag-a 1", "extra_envs": {}},
    )
    assert s.current_best["final_overlay"] == "/overlay/build"


@pytest.mark.asyncio
async def test_lift_copies_source_snapshot_into_stack_entry(session_dir):
    """Source snapshot manifest and changed files reach the stack entry."""
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
            "source_manifest": "/session/optimization_stack/src/abc123/manifest.json",
            "target_files": ["vllm/model_executor/layers/quantization/foo.py"],
            "framework_root": "/opt/vllm",
            "base_sha": "deadbeef",
        },
    )

    top = s.optimization_stack[-1]
    assert top.get("source_snapshot") == "/session/optimization_stack/src/abc123"
    assert top.get("source_manifest") == (
        "/session/optimization_stack/src/abc123/manifest.json"
    )
    assert top.get("target_files") == [
        "vllm/model_executor/layers/quantization/foo.py"
    ]
    assert top.get("framework_root") == "/opt/vllm"
    assert top.get("base_sha") == "deadbeef"


@pytest.mark.asyncio
async def test_drain_cancels_queued_baselines_but_spares_revalidation(session_dir):
    """A succeeded baseline drains its backlog; the enablement revalidation survives."""
    from hyperloom.orchestrator.actions.executors._accuracy_gate import (
        ENABLEMENT_REVALIDATION_REASON,
    )

    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 2195.86

    stale_a = await coord.tasks.create(kind="baseline", params={}, idempotency_key="bl-a")
    stale_b = await coord.tasks.create(kind="baseline", params={"tag": "x"}, idempotency_key="bl-b")
    reval = await coord.tasks.create(
        kind="baseline",
        params={"reason": ENABLEMENT_REVALIDATION_REASON},
        idempotency_key="bl-reval",
    )
    other = await coord.tasks.create(kind="explore", params={}, idempotency_key="ex-a")

    cancelled = await coord._drain_queued_baselines(reason="baseline_established")

    assert set(cancelled) == {stale_a.task_id, stale_b.task_id}
    assert (await coord.tasks.get(reval.task_id)).state == "queued"
    assert (await coord.tasks.get(other.task_id)).state == "queued"
    assert (await coord.tasks.get(stale_a.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_drain_spares_the_tracked_revalidation_task_id(session_dir):
    """The tracked id is honoured even when params carry no revalidation reason."""
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 1000.0
    reval = await coord.tasks.create(kind="baseline", params={}, idempotency_key="bl-tracked")
    coord.shared_state.enablement.revalidation_task_id = reval.task_id

    assert await coord._drain_queued_baselines(reason="baseline_established") == []
    assert (await coord.tasks.get(reval.task_id)).state == "queued"


@pytest.mark.asyncio
async def test_promote_baseline_drains_the_backlog(session_dir):
    """The drain is wired into promotion, not just available as a helper."""
    coord = _coord(session_dir)
    stale = await coord.tasks.create(kind="baseline", params={}, idempotency_key="bl-stale")

    await coord._promote_to_shared_state(
        "baseline",
        {"status": "succeeded", "output_throughput": 2185.95},
        task=_task("baseline", task_id="t-first"),
    )

    assert coord.shared_state.baseline_tput == 2185.95
    assert (await coord.tasks.get(stale.task_id)).state == "cancelled"


def test_lift_refuses_winner_that_does_not_beat_current_best(session_dir):
    """current_best never moves down, even for a winner its executor called a KEEP."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 2195.86
    s.current_best = {
        "action": "replay_warm_recipe",
        "tput": 2358.80,
        "extra_server_args": "--enable-aiter-allreduce-fusion",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }
    s.optimization_stack = [{"action": "replay_warm_recipe", "variant_name": "warm_replay"}]
    s.gain_per_stack_entry = [7.908]

    lifted = coord._lift_to_current_best(
        "explore",
        2355.46,
        {
            "name": "minimax-fused-swiglu+moe-combine",
            "candidate_extra_server_args": "--trust-remote-code",
            "extra_envs": {"SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE": "1"},
            "tput": 2355.46,
        },
    )

    assert lifted is False
    assert s.current_best["tput"] == 2358.80
    assert len(s.optimization_stack) == 1
    assert s.gain_per_stack_entry == [7.908]


def test_lift_refuses_winner_below_baseline_when_stack_is_empty(session_dir):
    """Before any validated layer the baseline is the anchor, and it holds too."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    lifted = coord._lift_to_current_best(
        "explore",
        900.0,
        {"name": "regression", "candidate_extra_server_args": "--slow", "extra_envs": {}},
    )

    assert lifted is False
    assert not s.current_best
    assert s.optimization_stack == []


def test_lift_accepts_winner_that_beats_current_best(session_dir):
    """The guard only blocks regressions; a genuine win still lifts."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.current_best = {"action": "baseline", "tput": 1000.0, "extra_server_args": "", "extra_envs": {}}

    lifted = coord._lift_to_current_best(
        "explore",
        1100.0,
        {"name": "real-win", "candidate_extra_server_args": "--fast", "extra_envs": {}},
    )

    assert lifted is True
    assert s.current_best["tput"] == 1100.0
    assert s.optimization_stack[-1]["variant_name"] == "real-win"


def test_lift_does_not_double_append_same_fingerprint(session_dir):
    """A renamed variant with the same content fingerprint must not add a second stack entry."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    fp = "shared_fp_abc123"
    s.optimization_stack = [
        {"action": "explore", "variant_name": "original", "fingerprint": fp, "tput": 1100.0}
    ]
    s.current_best = {"action": "explore", "tput": 1100.0, "extra_server_args": "--fast", "extra_envs": {}}

    lifted = coord._lift_to_current_best(
        "explore",
        1200.0,
        {"name": "renamed", "fingerprint": fp, "candidate_extra_server_args": "--fast", "extra_envs": {}},
    )

    # current_best refreshed but stack not duplicated.
    assert lifted is True
    assert s.current_best["tput"] == 1200.0
    assert len(s.optimization_stack) == 1
    assert s.optimization_stack[0]["variant_name"] == "original"


def test_lift_at_or_below_anchor_does_not_modify_stack(session_dir):
    """An accepted rerun that does not beat the live anchor leaves current_best unchanged."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    fp = "shared_fp_rerun"
    s.optimization_stack = [
        {"action": "explore", "variant_name": "prior", "fingerprint": fp, "tput": 1100.0}
    ]
    s.current_best = {"action": "explore", "tput": 1100.0, "extra_server_args": "--fast", "extra_envs": {}}

    lifted = coord._lift_to_current_best(
        "explore",
        1050.0,  # below current anchor of 1100
        {"name": "prior_rerun", "fingerprint": fp, "candidate_extra_server_args": "--fast", "extra_envs": {}},
    )

    assert lifted is False
    assert s.current_best["tput"] == 1100.0
    assert len(s.optimization_stack) == 1


async def test_integrate_keep_carries_the_stack_env_layer(session_dir):
    """A kernel integrate publishes args and envs from the same config.

    Writing ``current_best`` without ``extra_envs`` published a config whose args
    and envs came from different layers, and every dispatch site seeded from it.
    """
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.current_best = {
        "action": "explore",
        "tput": 1000.0,
        "extra_server_args": "--kv-cache-dtype fp8_e4m3",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }

    await coord.writeback._record_integrate_keep(
        {"new_tput": 1200.0, "kernel_id": "k001", "integration_id": "i1"},
    )

    assert s.current_best["extra_envs"] == {"VLLM_ROCM_USE_AITER": "1"}
    assert s.current_best["extra_server_args"] == "--kv-cache-dtype fp8_e4m3"


async def test_integrate_keep_lets_a_tuning_env_delta_win(session_dir):
    """A forge-GEMM KEEP ships ``result['extra_envs']``; it must survive the promote."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.current_best = {"action": "explore", "tput": 1000.0, "extra_envs": {"KEEP_ME": "1", "TUNED": "old"}}

    await coord.writeback._record_integrate_keep(
        {"new_tput": 1200.0, "kernel_id": "k002", "extra_envs": {"TUNED": "new"}},
    )

    assert s.current_best["extra_envs"] == {"KEEP_ME": "1", "TUNED": "new"}
