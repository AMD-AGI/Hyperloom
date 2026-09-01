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
from hyperloom.inference_optimizer.breakdown.agent_ownership import LEVER_CONFIG
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop import writeback as wb
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator, _is_patch_column_keep
from hyperloom.orchestrator.knowledge.remote_recipe._vendor.kb_store_client import (
    KnowledgeSections,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import has_new_keep
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
    # A profiler-on measurement never moves current_best, however high it reads.
    assert s.current_best["action"] == "explore"
    assert s.current_best["tput"] == 100.0
    assert s.cumulative_gain_validated == 0.0
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
async def test_promote_integrate_patch_marks_a_refused_keep(session_dir):
    """A KEEP measured below the live anchor is not adopted, and must not journal as one."""
    from hyperloom.orchestrator.state.optimization_journal import (
        OUTCOME_NO_PROMOTE,
        derive_journal_outcome,
    )

    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.current_best = {"action": "explore", "tput": 200.0, "extra_server_args": "", "extra_envs": {}}

    result = {
        "status": "kept",
        "output_throughput": 140.0,
        "specialist_task_id": "spec-1",
        "delta_pct": 40.0,
    }
    await coord._promote_to_shared_state("integrate_patch", result, task=_task("integrate_patch", task_id="t1"))

    assert s.current_best["tput"] == 200.0
    assert derive_journal_outcome("integrate_patch", result, promotable=True) == OUTCOME_NO_PROMOTE


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
@pytest.mark.parametrize("source_phase", ["EXPLORE", "FRAMEWORK_AGENT"])
async def test_integrate_keep_stages_patch_for_proposal_owner(session_dir, tmp_path, monkeypatch, source_phase):
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    patch = tmp_path / f"{source_phase.lower()}.diff"
    patch.write_bytes(f"{source_phase} bytes".encode())
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 120.0,
            "specialist_task_id": f"spec-{source_phase.lower()}",
            "source_phase": source_phase,
            "patches_applied": [str(patch)],
        },
        task=_task(
            "integrate_patch",
            params={
                "specialist_task_id": f"spec-{source_phase.lower()}",
                "source_phase": source_phase,
            },
        ),
    )

    staged = KnowledgeSections(draft).staged("patch")
    ref = f"patch/overlays/000000/00-{source_phase.lower()}.patch"
    assert staged.knowledge["patches"] == [ref]
    assert (draft / "files" / ref).read_bytes() == patch.read_bytes()
    assert KnowledgeSections(draft).staged("explore") is None
    # Explore and framework KEEPs share the one patch column, so both record the
    # same owner marker rather than the old per-column explore/framework label.
    assert coord.shared_state.optimization_stack[-1]["kb_required_owner"] == "PATCH"


@pytest.mark.asyncio
async def test_a_config_lever_keep_stages_under_the_configuration_section(session_dir, tmp_path, monkeypatch):
    """A KEEP that touched nothing on disk belongs to the configuration lever.

    The section it stages into is the other half of the routing an authored
    diff exercises: reading the phase instead would file both under one owner.
    """
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 120.0,
            "lever_kind": LEVER_CONFIG,
            "source_phase": "EXPLORE",
            "extra_server_args": "--page-size 32",
            "patches_applied": [],
        },
        task=_task("integrate_patch", params={"source_phase": "EXPLORE"}),
    )

    stack = coord.shared_state.optimization_stack
    assert stack and stack[-1]["lever_kind"] == LEVER_CONFIG
    assert _is_patch_column_keep({"source_phase": "EXPLORE"}, {"lever_kind": LEVER_CONFIG}) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["reverted", "kept_inert"])
async def test_integrate_nonpromotion_never_stages_patch(session_dir, tmp_path, monkeypatch, status):
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    patch = tmp_path / "not-kept.patch"
    patch.write_bytes(b"do not stage")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": status,
            "output_throughput": 90.0,
            "source_phase": "EXPLORE",
            "patches_applied": [str(patch)],
        },
        task=_task(
            "integrate_patch",
            params={"source_phase": "EXPLORE"},
        ),
    )

    assert KnowledgeSections(draft).sections() == []


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
    assert entry["recipe_publishable"] is False
    assert s.gain_per_stack_entry == [None]
    assert s.cumulative_gain_validated == 0.0
    assert s.pending_integrate == {}


@pytest.mark.asyncio
async def test_postbaseline_enablement_config_is_not_recipe_publishable(
    session_dir,
):
    coord = _coord(session_dir)
    state = coord.shared_state
    state.baseline_tput = 100.0
    state.current_best = {"action": "baseline", "tput": 100.0}

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "enablement": True,
            "output_throughput": 110.0,
            "specialist_task_id": "spec-enable-late",
            "extra_envs_applied": {"SGLANG_ENABLEMENT_ONLY": "1"},
        },
        task=_task(
            "integrate_patch",
            task_id="t-enable-late",
            params={"enablement": True, "source_phase": "FRAMEWORK_AGENT"},
        ),
    )

    assert state.optimization_stack[-1]["recipe_publishable"] is False
    # recipe_publishable is a config-layer filter applied inside
    # build_publishable_recipe_config; has_new_keep counts an enablement
    # KEEP as "new work" so the KB write proceeds and publishes patches.
    assert has_new_keep(state) is True


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
# GAP 6: an upstream-PR integrate_patch lifts current_best on KEEP. The
# candidate's progress row is the dispatcher's authored-outcome bridge, not
# this promote (see test_framework_agent_authoring).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_framework_agent_kept_lifts_and_records_progress(session_dir):
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.phase = "KERNEL_AGENT"

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "delta_pct": 30.0,
            "specialist_task_id": "https://x/pull/1",
            "workspace": "/w",
            "source_phase": "FRAMEWORK_AGENT",
        },
        task=_task(
            "integrate_patch",
            task_id="t1",
            params={
                "framework_agent_candidate_id": "https://x/pull/1",
                "batch_id": "b1",
                "patch_source": "upstream_pr",
                "source_phase": "FRAMEWORK_AGENT",
            },
        ),
    )

    assert s.current_best["action"] == "integrate_patch"
    assert s.current_best["tput"] == 130.0
    assert s.optimization_stack[-1]["source_phase"] == "FRAMEWORK_AGENT"
    # The stack variant must be the canonical candidate key, undecorated, so
    # resume can reconcile it against the recorded KEEP.
    assert s.optimization_stack[-1]["variant_name"] == "https://x/pull/1"


@pytest.mark.asyncio
async def test_framework_agent_keep_stages_returned_raw_patch(session_dir, tmp_path, monkeypatch):
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    patch = tmp_path / "pr-7.patch"
    patch.write_bytes(b"raw framework diff")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "specialist_task_id": "https://x/pull/7",
            "patches_applied": [str(patch)],
        },
        task=_task(
            "integrate_patch",
            params={"patch_source": "upstream_pr", "source_phase": "FRAMEWORK_AGENT"},
        ),
    )

    staged = KnowledgeSections(draft).staged("patch")
    ref = "patch/overlays/000000/00-pr-7.patch"
    assert staged.knowledge["patches"] == [ref]
    assert (draft / "files" / ref).read_bytes() == patch.read_bytes()
    row = staged.knowledge["provenance"][0]
    assert row["stack_index"] == 0
    assert row["base_sha"] == ""
    assert row["complete"] is True
    assert row["artifacts_outside_root"] == 0
    assert row["realized"] is False
    # No snapshot ran, so the delivered patch is the only absolute origin known.
    assert row["host_origin"] == {"sources": [str(patch)]}


@pytest.mark.asyncio
async def test_realized_diff_replaces_the_delivered_patch(session_dir, tmp_path, monkeypatch):
    """The realized diff is what landed, so publishing both would apply it twice."""
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    delivered = tmp_path / "delivered.patch"
    delivered.write_bytes(b"as delivered")
    realized = tmp_path / "snapshot" / "realized.patch"
    realized.parent.mkdir()
    realized.write_bytes(b"as landed")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "specialist_task_id": "spec-realized",
            "patches_applied": [str(delivered)],
            "source_realized_patch": str(realized),
            "base_sha": "abc123",
            "source_snapshot_complete": True,
            "source_artifacts_outside_root": 2,
            "framework_root": "/sglang",
            "source_snapshot": str(realized.parent),
        },
        task=_task("integrate_patch", params={"source_phase": "FRAMEWORK_AGENT"}),
    )

    staged = KnowledgeSections(draft).staged("patch")
    ref = "patch/overlays/000000/00-realized.patch"
    assert staged.knowledge["patches"] == [ref]
    assert (draft / "files" / ref).read_bytes() == b"as landed"
    row = staged.knowledge["provenance"][0]
    assert row["realized"] is True
    assert row["base_sha"] == "abc123"
    assert row["artifacts_outside_root"] == 2
    # Where the KEEP came from has to survive the handoff, not just the result,
    # and it lands on the ref so overlays from two trees stay distinguishable.
    assert row["host_origin"]["apply_roots"] == {ref: "/sglang"}
    assert row["host_origin"]["snapshot"] == str(realized.parent)
    assert row["host_origin"]["sources"] == [str(realized)]


@pytest.mark.asyncio
async def test_delivered_patch_is_the_fallback_when_no_realized_diff(session_dir, tmp_path, monkeypatch):
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    delivered = tmp_path / "delivered.patch"
    delivered.write_bytes(b"as delivered")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "specialist_task_id": "spec-fallback",
            "patches_applied": [str(delivered)],
            # A non-git tree harvests no realized diff.
            "source_realized_patch": "",
        },
        task=_task("integrate_patch", params={"source_phase": "FRAMEWORK_AGENT"}),
    )

    staged = KnowledgeSections(draft).staged("patch")
    assert staged.knowledge["patches"] == ["patch/overlays/000000/00-delivered.patch"]
    assert staged.knowledge["provenance"][0]["realized"] is False


@pytest.mark.asyncio
async def test_explicit_empty_patches_applied_never_scans_stale_workspace(session_dir, tmp_path, monkeypatch):
    draft = tmp_path / "kb-draft"
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "rejected.diff").write_bytes(b"rejected stale bytes")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "candidate": {"pr_url": "https://x/pull/8"},
            "patches_applied": [],
            "workspace": str(workspace),
        },
        task=_task("integrate_patch"),
    )

    # Final config comes from current_best at CLOSE; an explicit empty patch
    # list neither scans stale workspace files nor creates an owner section.
    assert KnowledgeSections(draft).staged("framework") is None
    assert coord.shared_state.kb_stage_outbox == []
    # A config-only KEEP must not mark a required patch owner; otherwise CLOSE
    # would reject the record for missing required section staging.
    assert "kb_required_owner" not in coord.shared_state.optimization_stack[-1]


@pytest.mark.asyncio
async def test_local_mode_without_remote_draft_does_not_enqueue_outbox(session_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("KB_DRAFT_DIR", raising=False)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    patch = tmp_path / "accepted.patch"
    patch.write_bytes(b"accepted")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "candidate": {"pr_url": "https://x/pull/10"},
            "patches_applied": [str(patch)],
        },
        task=_task("integrate_patch"),
    )

    assert coord.shared_state.kb_stage_outbox == []
    assert "kb_required_owner" not in coord.shared_state.optimization_stack[0]


@pytest.mark.asyncio
async def test_keep_kb_hook_runs_only_after_authoritative_save(session_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("KB_DRAFT_DIR", str(tmp_path / "draft"))
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    patch = tmp_path / "pr.patch"
    patch.write_bytes(b"raw diff")
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 100.0
    events: list[str] = []
    real_save = coord.shared_state.save

    def _save(*args, **kwargs):
        events.append("save")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(coord.shared_state, "save", _save)
    monkeypatch.setattr(
        coord.writeback,
        "_stage_agent_keep",
        lambda **_kwargs: events.append("stage") or True,
    )

    await coord._promote_to_shared_state(
        "integrate_patch",
        {
            "status": "kept",
            "output_throughput": 130.0,
            "specialist_task_id": "https://x/pull/9",
            "patches_applied": [str(patch)],
        },
        task=_task("integrate_patch", params={"source_phase": "FRAMEWORK_AGENT"}),
    )

    assert events[0:2] == ["save", "stage"]
    assert coord.shared_state.kb_stage_outbox == []


def test_outbox_drain_acknowledges_only_confirmed_success(
    session_dir,
    monkeypatch,
):
    coord = _coord(session_dir)
    patch = session_dir / "accepted.patch"
    patch.write_text("patch", encoding="utf-8")
    row = {
        "id": "FRAMEWORK_AGENT:0",
        "owner": "FRAMEWORK_AGENT",
        "stack_index": 0,
        "include_patches": True,
        "patch_sources": [str(patch)],
        "missing_patch_sources": [],
    }
    coord.shared_state.kb_stage_outbox = [row]
    monkeypatch.setattr(
        coord.writeback,
        "_stage_agent_keep",
        lambda **_kwargs: False,
    )

    coord.writeback._drain_agent_keep_outbox()

    assert coord.shared_state.kb_stage_outbox == [row]


def test_outbox_dead_letters_missing_patch_without_blocking_close(
    session_dir,
) -> None:
    coord = _coord(session_dir)
    coord.shared_state.optimization_stack = [
        {
            "action": "framework",
            "kb_required_owner": "FRAMEWORK_AGENT",
        }
    ]
    row = {
        "id": "FRAMEWORK_AGENT:0",
        "owner": "FRAMEWORK_AGENT",
        "stack_index": 0,
        "include_patches": True,
        "patch_sources": [str(session_dir / "gone.patch")],
        "missing_patch_sources": [],
    }
    coord.shared_state.kb_stage_outbox = [row]

    coord.writeback._drain_agent_keep_outbox()

    assert coord.shared_state.kb_stage_outbox == []
    assert coord.shared_state.kb_stage_dead_letter[0]["id"] == row["id"]
    assert coord.shared_state.kb_stage_dead_letter[0]["reason"] == ("patch_source_missing")
    assert "kb_required_owner" not in (coord.shared_state.optimization_stack[0])


@pytest.mark.asyncio
async def test_resume_settles_state_before_draining_kb_outbox(
    session_dir,
    monkeypatch,
):
    """The outbox drains after the recovery pass, from the durable config.

    Resume no longer rebuilds ``current_best`` from an ``optimization_stack``
    replay: the lift writes both together, so the replay could only ever
    reintroduce an env a later ablation removed. What still has to hold is the
    ordering — recovery settles and saves, and only then does the outbox stage
    whatever ``current_best`` durably holds.
    """
    coord = _coord(session_dir)
    coord._resumed_from = {"is_resume": True}
    coord.shared_state.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "winner",
            "candidate_extra_server_args": "--new",
            "extra_envs": {"NEW_ENV": "1"},
            "tput": 120.0,
        }
    ]
    coord.shared_state.current_best = {
        "extra_server_args": "--new",
        "extra_envs": {"NEW_ENV": "1"},
        "tput": 120.0,
    }
    coord.shared_state.cumulative_gain_validated_stack_len = 1
    coord.shared_state.kb_stage_outbox = [
        {
            "id": "EXPLORE:0",
            "owner": "EXPLORE",
            "stack_index": 0,
            "include_patches": False,
            "patch_sources": [],
            "missing_patch_sources": [],
        }
    ]

    async def _noop(*_args, **_kwargs):
        return None

    for name in (
        "_resume_recover_pending_integrate",
        "_resume_recover_pending_targeted_build",
        "_resume_recover_pending_warm_replay",
        "_resume_recover_pending_revalidation",
        "_resume_recover_orphaned_keeps",
        "_record_observation",
    ):
        monkeypatch.setattr(coord.writeback, name, _noop)

    events: list[str] = []
    staged: list[dict] = []

    def _save(_session_dir):
        events.append("save")

    def _stage(**_kwargs):
        events.append("stage")
        staged.append(dict(coord.shared_state.current_best))
        return True

    monkeypatch.setattr(coord.shared_state, "save", _save)
    monkeypatch.setattr(coord.writeback, "_stage_agent_keep", _stage)

    await coord.writeback._resume_consistency_pass()

    assert events.index("save") < events.index("stage")
    assert staged[0]["extra_server_args"] == "--new"
    assert staged[0]["extra_envs"] == {"NEW_ENV": "1"}
    assert coord.shared_state.kb_stage_outbox == []


@pytest.mark.asyncio
@pytest.mark.asyncio
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


def test_lift_persists_recipe_delta_separately_from_runtime_config(session_dir):
    coord = _coord(session_dir)
    coord.shared_state.current_best = {
        "action": "baseline",
        "tput": 1000.0,
        "extra_server_args": "--hld-only",
        "extra_envs": {"SGLANG_ENABLEMENT_ONLY": "1"},
    }

    coord._lift_to_current_best(
        "explore",
        1100.0,
        {
            "name": "optimized",
            "candidate_extra_server_args": "--page-size 64",
            "candidate_extra_envs": {"VLLM_OPTIMIZED": "1"},
            "recipe_delta": {
                "extra_server_args": "--page-size 64",
                "extra_envs": {"VLLM_OPTIMIZED": "1"},
                "remove_args": [],
                "unset_envs": [],
                "args_mode": "append",
            },
            "extra_server_args": "--hld-only --page-size 64",
            "extra_envs": {
                "SGLANG_ENABLEMENT_ONLY": "1",
                "VLLM_OPTIMIZED": "1",
            },
        },
    )

    top = coord.shared_state.optimization_stack[-1]
    assert top["recipe_delta"] == {
        "extra_server_args": "--page-size 64",
        "extra_envs": {"VLLM_OPTIMIZED": "1"},
        "remove_args": [],
        "unset_envs": [],
        "args_mode": "append",
    }
    assert "--hld-only" in coord.shared_state.current_best["extra_server_args"]
    assert "--hld-only" not in top["recipe_delta"]["extra_server_args"]


def test_lift_is_the_only_writer_so_an_ablated_env_stays_gone(session_dir):
    """A later winner that drops an inherited env must not see it come back."""
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


def test_lift_strips_a_harness_flag_inherited_from_the_previous_current_best(session_dir):
    """Issue #1192: the flag came back through the previous current_best re-merge."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 6137.0
    s.current_best = {
        "action": "replay_warm_recipe",
        "tput": 6165.0,
        "extra_server_args": "--no-enable-prefix-caching",
        "extra_envs": {},
    }

    coord._lift_to_current_best(
        "explore",
        8063.0,
        {
            "name": "aiter-fp8-kv-cache",
            "candidate_extra_server_args": "--kv-cache-dtype fp8",
            "extra_server_args": "--kv-cache-dtype fp8",
            "effective_extra_server_args": "--no-enable-prefix-caching --kv-cache-dtype fp8",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        },
    )

    assert s.current_best["extra_server_args"] == "--kv-cache-dtype fp8"
    assert s.current_best["effective_extra_server_args"] == "--kv-cache-dtype fp8"
    top = s.optimization_stack[-1]
    assert top["extra_server_args"] == "--kv-cache-dtype fp8"
    assert top["candidate_extra_server_args"] == "--kv-cache-dtype fp8"


def test_lift_strips_a_harness_flag_a_winner_proposed_directly(session_dir):
    """A winner whose own delta is the harness flag publishes no serving change."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.current_best = {
        "action": "explore",
        "tput": 1100.0,
        "extra_server_args": "--max-num-seqs 256",
        "extra_envs": {},
    }

    coord._lift_to_current_best(
        "explore",
        1200.0,
        {
            "name": "no-prefix-cache",
            "candidate_extra_server_args": "--no-enable-prefix-caching",
            "extra_server_args": "--max-num-seqs 256 --no-enable-prefix-caching",
            "extra_envs": {},
        },
    )

    assert s.current_best["extra_server_args"] == "--max-num-seqs 256"
    assert s.optimization_stack[-1]["candidate_extra_server_args"] == ""


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

    lifted = coord._lift_to_current_best(
        "gemm_tuning",
        1100.0,
        {"name": "worse", "extra_server_args": "--flag-b 2", "extra_envs": {"B": "2"}},
    )

    assert lifted is False
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
    assert "empty" not in entry
    assert "absent" not in entry
    assert "tuned_file" not in s.current_best
    assert "backend" not in s.current_best


def test_env_spec_reports_the_config_current_best_was_measured_on(session_dir):
    """The GEAK handoff must describe current_best, ablations included."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    coord._lift_to_current_best(
        "explore",
        1100.0,
        {"name": "v1", "extra_server_args": "--flag-a 1", "extra_envs": {"OLD": "1"}},
    )
    coord._lift_to_current_best(
        "explore",
        1200.0,
        {
            "name": "v2",
            "extra_server_args": "--flag-a 1",
            "extra_envs": {"NEW": "1"},
            "unset_envs": ["OLD"],
            "final_overlay": "/overlay/build",
        },
    )

    spec = coord.build_env_spec()

    assert spec["config"]["extra_envs"] == {"NEW": "1"}
    assert spec["config"]["extra_server_args"] == "--flag-a 1"
    assert spec["overlay_pythonpath"] == "/overlay/build"


def test_env_spec_routes_a_flag_stored_under_extra_envs_back_into_args(session_dir):
    """A ``-``-prefixed env key is a server arg; exporting it would drop it."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0
    coord._lift_to_current_best(
        "integrate_patch",
        1200.0,
        {
            "name": "patch-1",
            "extra_server_args": "--flag-a 1",
            "extra_envs": {"REAL_ENV": "1", "--compilation-config": "3"},
        },
    )

    spec = coord.build_env_spec()

    assert spec["config"]["extra_envs"] == {"REAL_ENV": "1"}
    args = spec["config"]["extra_server_args"].split()
    assert args[args.index("--compilation-config") + 1] == "3"
    assert "--flag-a" in args


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
    assert top.get("source_manifest") == ("/session/optimization_stack/src/abc123/manifest.json")
    assert top.get("target_files") == ["vllm/model_executor/layers/quantization/foo.py"]
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
    s.optimization_stack = [{"action": "explore", "variant_name": "original", "fingerprint": fp, "tput": 1100.0}]
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
    s.optimization_stack = [{"action": "explore", "variant_name": "prior", "fingerprint": fp, "tput": 1100.0}]
    s.current_best = {"action": "explore", "tput": 1100.0, "extra_server_args": "--fast", "extra_envs": {}}

    lifted = coord._lift_to_current_best(
        "explore",
        1050.0,  # below current anchor of 1100
        {"name": "prior_rerun", "fingerprint": fp, "candidate_extra_server_args": "--fast", "extra_envs": {}},
    )

    assert lifted is False
    assert s.current_best["tput"] == 1100.0
    assert len(s.optimization_stack) == 1


@pytest.mark.asyncio
async def test_promote_explore_two_winners_produce_two_stack_entries(session_dir):
    """Every winner applied in a round must get its own optimization_stack entry."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    # Winner A: gains 10 %, measured tput 1100.  Winner B is applied on top:
    # gains another 10 % over the new 1100 base, measured tput 1210.
    winner_a = {
        "name": "w-a",
        "fingerprint": "fp_a",
        "tput": 1100.0,
        "candidate_extra_server_args": "--flag-a 1",
        "extra_server_args": "--flag-a 1",
        "extra_envs": {},
        "gain_pct": 10.0,
    }
    winner_b = {
        "name": "w-b",
        "fingerprint": "fp_b",
        "tput": 1210.0,
        "candidate_extra_server_args": "--flag-b 2",
        "extra_server_args": "--flag-a 1 --flag-b 2",
        "extra_envs": {},
        "gain_pct": 10.0,
    }

    await coord._promote_to_shared_state(
        "explore",
        {
            "explore_search_update": {},
            "winners": [winner_a, winner_b],
            "round_id": "r1",
            "best_variant": winner_a,
            "output_throughput": 1210.0,
            "best_gain_pct": 10.0,
        },
        task=_task("explore", params={"gap_canonical_id": "g1"}),
    )

    assert len(s.optimization_stack) == 2
    assert s.optimization_stack[0]["variant_name"] == "w-a"
    assert s.optimization_stack[0]["tput"] == 1100.0
    assert s.optimization_stack[1]["variant_name"] == "w-b"
    assert s.optimization_stack[1]["tput"] == 1210.0
    assert s.current_best["tput"] == 1210.0
    # gain_per_stack_entry must be index-aligned with optimization_stack.
    assert len(s.gain_per_stack_entry) == 2


@pytest.mark.asyncio
async def test_promote_explore_multi_winner_dedup_skips_already_stacked(session_dir):
    """A winner whose fingerprint is already in the stack must not add a duplicate."""
    coord = _coord(session_dir)
    s = coord.shared_state
    s.baseline_tput = 1000.0

    existing_fp = "fp_existing"
    s.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "prior",
            "fingerprint": existing_fp,
            "tput": 1100.0,
            "extra_server_args": "--flag-a 1",
            "candidate_extra_server_args": "--flag-a 1",
            "extra_envs": {},
        }
    ]
    s.current_best = {"action": "explore", "tput": 1100.0, "extra_server_args": "--flag-a 1", "extra_envs": {}}

    winner_new = {
        "name": "w-new",
        "fingerprint": "fp_new",
        "tput": 1210.0,
        "candidate_extra_server_args": "--flag-b 2",
        "extra_server_args": "--flag-a 1 --flag-b 2",
        "extra_envs": {},
        "gain_pct": 10.0,
    }
    winner_dup = {
        "name": "prior-renamed",
        "fingerprint": existing_fp,
        "tput": 1300.0,
        "candidate_extra_server_args": "--flag-a 1",
        "extra_server_args": "--flag-a 1",
        "extra_envs": {},
        "gain_pct": 5.0,
    }

    await coord._promote_to_shared_state(
        "explore",
        {
            "explore_search_update": {},
            "winners": [winner_new, winner_dup],
            "round_id": "r2",
            "best_variant": winner_new,
            "output_throughput": 1300.0,
            "best_gain_pct": 10.0,
        },
        task=_task("explore", task_id="t2", params={"gap_canonical_id": "g1"}),
    )

    # Only the new winner's entry is appended; the duplicate fingerprint is skipped.
    assert len(s.optimization_stack) == 2
    assert s.optimization_stack[1]["variant_name"] == "w-new"


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


# ── source-layer handles: executor result → stack entry → env_spec ──────────


def _keep_result(tmp_path: Path, *, import_root: str, complete: bool = True) -> dict:
    """An integrate_patch KEEP result with a materialized snapshot on disk."""
    snapshot = tmp_path / "snap"
    (snapshot / "files" / import_root).mkdir(parents=True)
    return {
        "source_snapshot": str(snapshot),
        "source_manifest": str(snapshot / "manifest.json"),
        "target_files": ["python/sglang/srt/server.py"],
        "framework_root": "/sgl-workspace/sglang",
        "base_sha": "abc123",
        "source_import_root": import_root,
        "source_snapshot_complete": complete,
    }


def test_source_layer_handles_carry_every_field_the_stack_entry_needs(tmp_path):
    """A lift path that forwards a subset silently degrades the GEAK overlay."""
    handles = wb._source_layer_handles(_keep_result(tmp_path, import_root="python"))

    assert handles["source_import_root"] == "python"
    assert handles["source_snapshot_complete"] is True
    assert handles["framework_root"] == "/sgl-workspace/sglang"
    assert handles["base_sha"] == "abc123"
    assert handles["target_files"] == ["python/sglang/srt/server.py"]
    assert handles["source_snapshot"].endswith("snap")


def test_source_layer_handles_omit_a_completeness_the_result_never_recorded():
    """Absent must stay absent so legacy entries still read their manifest."""
    handles = wb._source_layer_handles({"source_snapshot": "/snap"})

    assert "source_snapshot_complete" not in handles


def test_env_spec_hands_geak_the_import_root_not_the_snapshot_top(session_dir, tmp_path):
    """GEAK PYTHONPATHs this value; the snapshot top holds no importable module."""
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 1000.0
    result = _keep_result(tmp_path, import_root="python")

    coord._lift_to_current_best(
        "integrate_patch",
        1200.0,
        {"name": "patch-1", "scope": "source_patch", **wb._source_layer_handles(result)},
    )
    spec = coord.build_env_spec()

    (snapshot,) = spec["source_snapshots"]
    assert snapshot["snapshot_dir"] == str(tmp_path / "snap" / "files" / "python")
    assert snapshot["reproducible"] is True


def test_env_spec_overlay_stops_at_files_for_a_dist_packages_install(session_dir, tmp_path):
    """No import root means modules already start at the tree root."""
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 1000.0
    result = _keep_result(tmp_path, import_root="")

    coord._lift_to_current_best(
        "integrate_patch",
        1200.0,
        {"name": "patch-1", "scope": "source_patch", **wb._source_layer_handles(result)},
    )
    spec = coord.build_env_spec()

    (snapshot,) = spec["source_snapshots"]
    assert snapshot["snapshot_dir"] == str(tmp_path / "snap" / "files")


def test_env_spec_refuses_an_incomplete_snapshot(session_dir, tmp_path):
    """GEAK drops a non-reproducible entry, so False must survive the lift."""
    coord = _coord(session_dir)
    coord.shared_state.baseline_tput = 1000.0
    result = _keep_result(tmp_path, import_root="python", complete=False)

    coord._lift_to_current_best(
        "integrate_patch",
        1200.0,
        {"name": "patch-1", "scope": "source_patch", **wb._source_layer_handles(result)},
    )
    spec = coord.build_env_spec()

    (snapshot,) = spec["source_snapshots"]
    assert snapshot["reproducible"] is False
