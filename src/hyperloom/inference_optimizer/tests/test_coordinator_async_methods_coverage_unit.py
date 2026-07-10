# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for Coordinator async/stateful methods invoked directly against a
real (mock-backed) Coordinator: SharedState promotion across task kinds, prompt
composition per agent, advisory blocks, research-scout harvest, and the
orchestration checkpoint guard."""

from __future__ import annotations

import time

import pytest

from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {
        name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "kernel_agent", "critic", "robustness")
    }


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


# -- _promote_to_shared_state ----------------------------------------------
@pytest.mark.asyncio
async def test_promote_baseline_sets_anchor_and_current_best(coord: Coordinator) -> None:
    # Skip the heavy PRELUDE cascade by pre-marking a pending roofline task.
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    coord.shared_state.baseline_failure_streak = 2
    coord.shared_state.baseline_arg_error_streak = 1
    await coord._promote_to_shared_state(
        "baseline",
        {
            "output_throughput": 1000.0,
            "warmup_round_tput": 900.0,
            "accuracy": 0.95,
            "subprocess_runtime_sec": 120.0,
            "ttft_mean_ms": 100.0,
            "e2el_mean_ms": 2000.0,
            "tpot_mean_ms": 10.0,
            "workspace": "/tmp/ws",
        },
    )
    # Hot measure round is the conclusion baseline; cold warmup is audit-only.
    assert coord.shared_state.baseline_tput == 1000.0
    assert coord.shared_state.baseline_cold_tput == 900.0
    assert coord.shared_state.baseline_hot_tput == 1000.0
    assert coord.shared_state.baseline_failure_streak == 0
    assert coord.shared_state.baseline_arg_error_streak == 0
    assert coord.shared_state.current_best["action"] == "baseline"
    assert coord.shared_state.current_best["tput"] == 1000.0
    assert coord.shared_state.current_best["cold_tput"] == 900.0


@pytest.mark.asyncio
async def test_promote_baseline_non_dict_is_noop(coord: Coordinator) -> None:
    await coord._promote_to_shared_state("baseline", "not-a-dict")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_unpromotable_baseline_fast_arg_errors_stop_after_two(
    coord: Coordinator,
) -> None:
    task = Task(
        task_id="baseline-fast-arg",
        kind="baseline",
        state="running",
        params={"config_path": "baseline.yaml"},
        idempotency_key="baseline-fast-arg",
    )
    result = {
        "status": "failed",
        "error_class": "fast_exit_arg_error",
        "error": "ValueError: Unknown attention backend: ROCM_FLASH",
    }

    await coord._handle_unpromotable_result(task, result)
    assert coord.shared_state.baseline_arg_error_streak == 1
    assert coord.shared_state.stop_reason != "baseline_arg_error"

    await coord._handle_unpromotable_result(task, result)
    assert coord.shared_state.baseline_arg_error_streak == 2
    assert coord.shared_state.baseline_failure_streak == 0
    assert coord.shared_state.stop_reason == "baseline_arg_error"


@pytest.mark.asyncio
async def test_unpromotable_baseline_mixed_classes_stop_after_three_total(
    coord: Coordinator,
) -> None:
    """Mixed subprocess_nonzero + fast_exit_arg_error failures must still
    fast-fail once 3 total baseline failures accrue — neither per-class streak
    reaches its own threshold, so the combined backstop is what stops the run
    (otherwise the session burns the whole budget -> time_exhausted)."""
    def _task() -> Task:
        return Task(
            task_id="bl-mixed", kind="baseline", state="running",
            params={"config_path": "baseline.yaml"},
            idempotency_key="bl-mixed",
        )
    subproc = {"status": "failed", "error_class": "subprocess_nonzero",
               "error": "boom"}
    argerr = {"status": "failed", "error_class": "fast_exit_arg_error",
              "error": "bad arg"}

    await coord._handle_unpromotable_result(_task(), subproc)
    await coord._handle_unpromotable_result(_task(), argerr)
    assert coord.shared_state.stop_reason not in (
        "baseline_failed", "baseline_arg_error",
    )
    await coord._handle_unpromotable_result(_task(), subproc)
    # failure_streak=2 (<3) and arg_error_streak reset to 0 — neither per-class
    # threshold trips, but 3 total failures does.
    assert coord.shared_state.baseline_failure_streak == 2
    assert coord.shared_state.baseline_total_failures == 3
    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_promote_profile_succeeded_records_trace(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state(
        "profile",
        {
            "status": "succeeded",
            "main_trace_path": "/tmp/trace.json",
            "output_throughput": 820.0,
        },
    )
    assert coord.shared_state.last_profile_status == "succeeded"
    assert coord.shared_state.last_profile_trace == "/tmp/trace.json"


@pytest.mark.asyncio
async def test_promote_profile_failed_clears_trace(coord: Coordinator) -> None:
    await coord._promote_to_shared_state(
        "profile",
        {
            "status": "failed",
            "error_class": "no_trace_files",
        },
    )
    assert coord.shared_state.last_profile_status == "failed"


@pytest.mark.asyncio
async def test_promote_roofline_succeeded_and_skipped_and_failed(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state("roofline", {"status": "succeeded"})
    await coord._promote_to_shared_state("roofline", {"status": "skipped"})
    await coord._promote_to_shared_state(
        "roofline",
        {
            "status": "failed",
            "error_class": "boom",
            "phase": "trace",
        },
    )
    # failure streak bumped on the failed branch
    assert getattr(coord.shared_state, "roofline_failure_streak", 0) >= 1


@pytest.mark.asyncio
async def test_promote_explore_with_winner(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state(
        "explore",
        {
            "winners": [{"name": "v0", "extra_server_args": "--tp 1"}],
            "best_variant": {"name": "v0", "extra_server_args": "--tp 1"},
            "output_throughput": 900.0,
            "round_id": "r1",
            "losers": [],
            "skipped_dup": [],
        },
    )
    assert coord.shared_state.current_best.get("tput") == 900.0


@pytest.mark.asyncio
async def test_promote_framework_kept(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state(
        "framework_agent",
        {
            "status": "kept",
            "candidate": {"candidate_id": "c1", "pr_url": "http://x/1"},
            "batch_id": "b1",
            "delta_pct": 5.0,
            "output_throughput": 840.0,
            "workspace": "/tmp/ws",
        },
    )
    assert isinstance(coord.shared_state.framework_agent_phase_progress, list)
    assert coord.shared_state.framework_agent_phase_progress[-1]["kept"] is True


# -- _compose_prompt -------------------------------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_orchestration_with_time_budget(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = time.monotonic() + 600.0
    coord.shared_state.max_minutes = 60
    out = await coord._compose_prompt("orchestration")
    assert "SESSION_DIR=" in out
    assert "Mission progress" in out
    assert "Time budget" in out


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_deadline_imminent_warning(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = time.monotonic() + 60.0  # < 5 min remaining
    coord.shared_state.max_minutes = 60
    coord.shared_state.closing_phase = False
    out = await coord._compose_prompt("orchestration")
    assert "< 5 min remaining" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_and_kernel(coord: Coordinator) -> None:
    coord._run_started_monotonic = time.monotonic() - 60.0
    coord._run_deadline = time.monotonic() + 600.0
    coord.shared_state.max_minutes = 60
    out_rob = await coord._compose_prompt("robustness")
    out_k = await coord._compose_prompt("kernel_agent")
    assert "SESSION_DIR=" in out_rob
    assert "SESSION_DIR=" in out_k


# -- advisory blocks -------------------------------------------------------
def test_advisory_blocks_disabled_return_empty(coord: Coordinator) -> None:
    coord.shared_state.target_advisory_enabled = False
    assert coord._target_gap_advisory_block() == ""
    assert coord._current_primary_gap() is None


def test_plateau_advisory_block_no_signal(coord: Coordinator) -> None:
    # No plateau override active -> empty advisory.
    assert isinstance(coord._plateau_advisory_block(), str)


def test_priors_match_advisory_block_no_variants(coord: Coordinator) -> None:
    assert coord._priors_match_advisory_block() == ""


# -- _harvest_research_scout -----------------------------------------------
def test_harvest_research_scout_empty_and_populated(coord: Coordinator) -> None:
    coord._harvest_research_scout({})  # no 'research' block -> fail-soft no-op
    coord._harvest_research_scout(
        {
            "research": {
                "hints": {"what_to_try": ["aiter"]},
                "gaps": [],
            }
        }
    )


# -- _maybe_checkpoint_orchestration ---------------------------------------
@pytest.mark.asyncio
async def test_maybe_checkpoint_orchestration_non_conversational(coord: Coordinator) -> None:
    took = await coord._maybe_checkpoint_orchestration(tick=1, phase_changed=False)
    assert took is False


# -- _handle_escalate_strategy_change --------------------------------------
def _escalate(hint: str) -> Intent:
    return Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={"summary": "s", "next_action_hint": hint},
    )


@pytest.mark.asyncio
async def test_escalate_invalid_hint_broadcasts_only(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change("orchestration", _escalate("bogus"))
    # invalid hint isn't consumed
    assert coord.shared_state.last_consumed_escalate_hint != "bogus"


@pytest.mark.asyncio
async def test_escalate_extend_explore_budget(coord: Coordinator) -> None:
    from hyperloom.orchestrator.phases.machine_state import (
        ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
    )

    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate(ESCALATE_HINT_EXTEND_EXPLORE_BUDGET),
    )
    assert coord.shared_state.last_consumed_escalate_hint == ESCALATE_HINT_EXTEND_EXPLORE_BUDGET


@pytest.mark.asyncio
async def test_escalate_extend_kernel_budget(coord: Coordinator) -> None:
    from hyperloom.orchestrator.phases.machine_state import (
        ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
    )

    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate(ESCALATE_HINT_EXTEND_KERNEL_BUDGET),
    )
    assert coord.shared_state.last_consumed_escalate_hint == ESCALATE_HINT_EXTEND_KERNEL_BUDGET


@pytest.mark.asyncio
async def test_escalate_pause_specialist(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("pause_specialist_kernel"),
    )
    assert coord.shared_state.last_consumed_escalate_hint == "pause_specialist_kernel"


@pytest.mark.asyncio
async def test_escalate_skip_to_kernel_deferred(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_kernel"),
    )
    # deferred hint queued for the next compute_next_phase
    assert coord.shared_state.pending_escalate_hint == "skip_to_kernel"


@pytest.mark.asyncio
async def test_escalate_skip_to_close_suppressed_pre_enablement(coord: Coordinator) -> None:
    """Q2: skip_to_close is dropped while a not-yet-enabled run is still enabling."""
    coord.shared_state.phase = "PRELUDE"
    coord.shared_state.baseline_tput = 0.0
    coord.shared_state.enablement_succeeded = False
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_close"),
    )
    # The premature close hint is NOT queued -> the enablement loop keeps going.
    assert coord.shared_state.pending_escalate_hint != "skip_to_close"


@pytest.mark.asyncio
async def test_escalate_skip_to_close_allowed_after_enablement(coord: Coordinator) -> None:
    """skip_to_close is honored once a baseline exists (guard no longer active)."""
    coord.shared_state.phase = "EXPLORE"
    coord.shared_state.baseline_tput = 1234.0
    coord.shared_state.enablement_succeeded = True
    await coord._handle_escalate_strategy_change(
        "orchestration",
        _escalate("skip_to_close"),
    )
    assert coord.shared_state.pending_escalate_hint == "skip_to_close"


# -- _scan_stale_specialists -----------------------------------------------
@pytest.mark.asyncio
async def test_scan_stale_specialists_empty(coord: Coordinator) -> None:
    assert await coord._scan_stale_specialists() == []


# -- _maybe_autosubmit_specialist_patches ----------------------------------
@pytest.mark.asyncio
async def test_autosubmit_skipped_when_no_patches(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="spec-1", kind="specialist", state="running", params={}, idempotency_key="k1")
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": []},
    )  # empty list -> early return


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_files_missing(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="spec-2", kind="specialist", state="running", params={}, idempotency_key="k2")
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={"patches_written": ["ghost.py"]},
    )  # claimed file does not exist -> records skip observation, returns


@pytest.mark.asyncio
async def test_autosubmit_creates_proposal_for_real_file(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-3"
    spec_root = runs_dir(coord.session_dir, "specialist", sid)
    wt = spec_root / "worktree"
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "kernel.py").write_text("# patched\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="k3")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": ["kernel.py"],
            "proposal_set": [{"name": "fuse-moe"}],
        },
    )
    assert len(coord.state.pending_proposals) == n_before + 1


@pytest.mark.asyncio
async def test_autosubmit_creates_proposal_for_artifacts_only(coord: Coordinator) -> None:
    """A specialist with NO source patch but a non-diff tuned artifact
    (``artifacts_written`` with a real file in its worktree) is a routable
    deliverable: autosubmit must create an integrate_patch proposal so the
    artifact-install channel runs (regression aiter#4130: the tuned FMOE CSV was
    silently dropped because routing keyed only on ``patches_written``)."""
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-route"
    art_dir = runs_dir(coord.session_dir, "specialist", sid) / "worktree" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "tuned_fmoe.csv").write_text("cu_num,token\n304,16\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka1")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [
                {
                    "source": "artifacts/tuned_fmoe.csv",
                    "target": "configs/model_configs/qwen3_tuned_fmoe.csv",
                    "kind": "aiter_tuned_fmoe_csv",
                }
            ],
        },
    )
    assert len(coord.state.pending_proposals) == n_before + 1


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_artifact_source_outside_sandbox(
    coord: Coordinator, tmp_path
) -> None:
    """An ``artifacts_written`` entry whose ``source`` is an ABSOLUTE path
    OUTSIDE the specialist sandbox must NOT be routable: integrate_patch would
    reject it as ``source_outside_workspace``, so autosubmit must not burn a
    round creating a proposal for it (source-sandbox parity with
    ``_resolve_artifact_specs``)."""
    from hyperloom.orchestrator.state.task_registry import Task

    outside = tmp_path / "outside.csv"
    outside.write_text("x", encoding="utf-8")
    task = Task(
        task_id="spec-art-outside",
        kind="specialist",
        state="running",
        params={},
        idempotency_key="ka2",
    )
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [
                {
                    "source": str(outside),
                    "target": "configs/model_configs/x.csv",
                    "kind": "k",
                }
            ],
        },
    )
    assert len(coord.state.pending_proposals) == n_before


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_artifact_source_relative_escapes_sandbox(
    coord: Coordinator,
) -> None:
    """A RELATIVE artifact ``source`` that escapes the specialist sandbox via
    ``..`` must NOT be routable, even though ``(base / source)`` resolves (the
    OS follows ``..``) to a real file: integrate_patch rejects it as
    ``source_outside_workspace``, so autosubmit must not route it. Full sandbox
    parity for relative sources, not just absolute ones."""
    import os

    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    sid = "spec-art-escape"
    worktree = runs_dir(coord.session_dir, "specialist", sid) / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    outside = coord.session_dir / "escape.csv"
    outside.write_text("x", encoding="utf-8")
    rel_escape = os.path.relpath(outside, worktree)
    task = Task(task_id=sid, kind="specialist", state="running", params={}, idempotency_key="ka3")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "artifacts_written": [
                {"source": rel_escape, "target": "configs/model_configs/x.csv", "kind": "k"}
            ],
        },
    )
    assert len(coord.state.pending_proposals) == n_before


# -- _record_fact_per_task -------------------------------------------------
def test_record_fact_per_task_keep_and_revert(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="t-fact", kind="explore", state="succeeded", params={}, idempotency_key="kf")
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"gain_pct": 5.0, "output_throughput": 900.0},
        kept=True,
    )
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"error_class": "boom", "reason": "bad"},
        kept=False,
    )


# -- Journal no longer records a reverted patch as KEEP ----------------------
def test_record_fact_reverted_integrate_patch_journals_revert(coord: Coordinator) -> None:
    """Regression for the "fake KEEP" bug: a reverted integrate_patch reaches the
    fact hook with kept=True (``status != failed`` is promotable), yet the
    journal must record REVERT with the REAL measured delta (from delta_pct)."""
    from hyperloom.orchestrator.state.optimization_journal import (
        OUTCOME_REVERT,
    )
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(
        task_id="t-revert-fake-keep",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="t-revert-fake-keep",
    )
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        # The exact real-session signature: tput == baseline → delta_pct ~0,
        # executor returns "reverted", dispatcher marks it promotable.
        result_dict={
            "status": "reverted",
            "delta_pct": -0.44,
            "output_throughput": 0.440529,
            "reason": "throughput delta -0.44% < keep_threshold 1.00%",
        },
        kept=True,
    )
    entry = coord._ensure_journal().entries[-1]
    assert entry.outcome == OUTCOME_REVERT
    assert entry.gain_pct == -0.44  # real delta shown, not null
    assert entry.reason and "keep_threshold" in entry.reason


def test_record_fact_kept_integrate_patch_journals_keep(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.optimization_journal import OUTCOME_KEEP
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(
        task_id="t-real-keep",
        kind="integrate_patch",
        state="succeeded",
        params={},
        idempotency_key="t-real-keep",
    )
    coord._record_fact_per_task(
        task=task,
        source_session_id="sess-a",
        result_dict={"status": "kept", "delta_pct": 6.2, "output_throughput": 1100.0},
        kept=True,
    )
    entry = coord._ensure_journal().entries[-1]
    assert entry.outcome == OUTCOME_KEEP
    assert entry.gain_pct == 6.2


def test_is_promotable_result_unchanged_for_reverted_integrate_patch(coord: Coordinator) -> None:
    """Guard the key Problem-3 constraint: we must NOT change routing — a reverted
    integrate_patch stays promotable so it still runs the pending_integrate
    cleanup in _promote_to_shared_state (only the journal semantics changed)."""
    assert coord._is_promotable_result("integrate_patch", {"status": "reverted"}) is True
    assert coord._is_promotable_result("integrate_patch", {"status": "failed"}) is False


# -- _compose_prompt additional branches -----------------------------------
class _Obj:
    kind = "gain_pct"
    value = 20.0


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_gain_objective(coord: Coordinator) -> None:
    coord._current_objective = _Obj()
    coord.shared_state.cumulative_gain = 5.0
    await coord._compose_prompt("orchestration")
    # target_gap_pct = max(0, 20 - 5)
    assert coord.shared_state.target_gap_pct == 15.0


@pytest.mark.asyncio
async def test_compose_prompt_conversational_delta(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.conversation, "_orchestration_conversational", lambda: True)
    coord._orchestration_seeded = True  # DELTA turn -> push_full False
    out = await coord._compose_prompt("orchestration")
    assert "Context (pull on demand)" in out


@pytest.mark.asyncio
async def test_compose_prompt_conversational_seed_memory(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.conversation, "_orchestration_conversational", lambda: True)
    coord._orchestration_seeded = False  # SEED turn -> push_full True
    coord._orchestration_seed_memory = "=== recovered memory ==="
    out = await coord._compose_prompt("orchestration")
    assert "recovered memory" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_high_no_progress(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(
        coord.conversation,
        "_conversation_progress_signal",
        lambda: {
            "ticks_without_progress": 9,
            "threshold": 5,
            "severity": "high",
            "last_progress_tick": 1,
        },
    )
    out = await coord._compose_prompt("robustness")
    assert "no observable progress" in out


# -- _context_analysis_reader ----------------------------------------------
def test_context_analysis_reader(coord: Coordinator) -> None:
    out = coord._context_analysis_reader()
    assert isinstance(out, str)


def test_context_analysis_reader_fallback_path(coord: Coordinator, tmp_path) -> None:
    md = tmp_path / "analysis.md"
    md.write_text("# roofline\n", encoding="utf-8")
    coord.shared_state.last_trace_analyze = {"analysis_md_path": str(md)}
    # _format_analysis_md_full returns empty -> falls back to the path read
    coord.shared_state.analysis_md = ""
    out = coord._context_analysis_reader()
    assert isinstance(out, str)


# -- advisory blocks enabled paths -----------------------------------------
def test_target_gap_advisory_enabled(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.knowledge import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: {"name": "comp"})
    monkeypatch.setattr(rh, "gap_analysis", lambda *a, **k: {"primary_gap": "throughput"})
    monkeypatch.setattr(rh, "full_gap_summary", lambda g: "GAP-SUMMARY")
    coord.shared_state.target_advisory_enabled = True
    coord.shared_state.current_best = {"tput": 1000.0, "tpot_mean_ms": 5.0}
    coord.shared_state.tp = 1
    coord.shared_state.conc = 64
    assert coord._target_gap_advisory_block() == "GAP-SUMMARY"
    assert coord._current_primary_gap() == "throughput"


def test_target_gap_advisory_no_target(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.knowledge import research_hints as rh

    monkeypatch.setattr(rh, "load_competitor_target", lambda _sd: None)
    coord.shared_state.target_advisory_enabled = True
    assert coord._target_gap_advisory_block() == ""
    assert coord._current_primary_gap() is None


# -- _promote_warm_replay --------------------------------------------------
def test_promote_warm_replay_non_dict(coord: Coordinator) -> None:
    coord._promote_warm_replay("nope")  # type: ignore[arg-type]
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_failed_status(coord: Coordinator) -> None:
    coord._promote_warm_replay({"status": "failed", "error_class": "x", "error": "boom"})
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_invalid_tput(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 0.0
    coord._promote_warm_replay({"status": "succeeded", "output_throughput": 0.0})
    assert coord.shared_state.warm_replay_outcome["status"] == "failed"


def test_promote_warm_replay_reproduced_no_params(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0},
        task=None,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced_but_no_params"


def test_promote_warm_replay_reproduced_with_params(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    coord.shared_state.baseline_tput = 800.0
    task = Task(
        task_id="warm-1",
        kind="replay_warm_recipe",
        state="running",
        params={"extra_envs": {"A": "1"}, "baseline_tput_anchor": 800.0},
        idempotency_key="kw",
    )
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0},
        task=task,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"


# -- _maybe_auto_retry_specialist ------------------------------------------
def _spec_task(**params):
    from hyperloom.orchestrator.state.task_registry import Task

    return Task(task_id="spec-r", kind="specialist", state="running", params=params, idempotency_key="spec-r-key")


def _result(state="failed", result=None, error=None):
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

    return SubAgentResult(task_id="spec-r", state=state, result=result or {}, error=error)


@pytest.mark.asyncio
async def test_auto_retry_disabled_by_env(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "0")
    assert await coord._maybe_auto_retry_specialist(_spec_task(), _result()) is False


@pytest.mark.asyncio
async def test_auto_retry_not_eligible(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    res = _result(result={"runner_status": "empty_synthesised"}, error="no_output")
    assert await coord._maybe_auto_retry_specialist(_spec_task(), res) is False


@pytest.mark.asyncio
async def test_auto_retry_schedules_on_transient_failure(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    res = _result(result={"runner_status": "stale"}, error="timeout waiting")
    scheduled = await coord._maybe_auto_retry_specialist(_spec_task(), res)
    assert scheduled is True


@pytest.mark.asyncio
async def test_auto_retry_caps_attempts(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY_MAX", "1")
    res = _result(result={"runner_status": "stale"}, error="timeout")
    task = _spec_task(_auto_retry_attempt=1)  # already at cap
    assert await coord._maybe_auto_retry_specialist(task, res) is False


# -- _fan_out_specialist_wave (skip path) ----------------------------------
@pytest.mark.asyncio
async def test_fan_out_wave_skips_invalid_entries(coord: Coordinator, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        coord,
        "_handle_delegate",
        lambda *a, **k: called.append(a),
    )
    intent = Intent(type=IntentType.DELEGATE, payload={"idempotency_key": "w"})
    await coord._fan_out_specialist_wave(
        "orchestration",
        intent,
        {"tasks": ["not-a-dict", {}, {"task_description": "   "}]},
    )
    assert called == []  # every entry skipped, no delegate fired


# -- _warm_specialist_params -----------------------------------------------
@pytest.mark.asyncio
async def test_warm_specialist_params_fills_defaults(coord: Coordinator) -> None:
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.tp = 1
    coord.shared_state.conc = 64
    coord.shared_state.isl = 256
    coord.shared_state.osl = 256
    params: dict = {"domain": "kernel_agent"}
    await coord._warm_specialist_params(params)
    assert params["gpu_type"] == "mi300x"
    assert params["tp"] == 1
    assert "kb_subgraph" in params


# -- cortex_finalize_recipe_and_journal ------------------------------------
def test_cortex_finalize_recipe_and_journal_no_kb(coord: Coordinator) -> None:
    coord.shared_state.current_best = {"tput": 950.0}
    coord.shared_state.cumulative_gain_validated = 12.5
    # cortex_kb is None in the mock harness -> journal finalize then early return
    coord.cortex_finalize_recipe_and_journal()


# -- _record_fact_per_variant ----------------------------------------------
def test_record_fact_per_variant_keep_revert_skip(coord: Coordinator) -> None:
    from hyperloom.orchestrator.state.task_registry import Task

    task = Task(task_id="t-var", kind="explore", state="succeeded", params={}, idempotency_key="kv")
    # SKIPPED_DEDUP -> early return (no journal row)
    coord._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={"outcome": "SKIPPED_DEDUP", "variant_name": "v0"},
    )
    # KEEP path
    coord._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "v1",
            "metrics": {"gain_pct": 4.0, "output_throughput": 900.0},
            "variant": {"name": "v1"},
        },
    )
    # REVERT path with error_class/reason
    coord._record_fact_per_variant(
        task=task,
        source_session_id="s",
        variant_outcome={
            "outcome": "REVERT",
            "variant_name": "v2",
            "error_class": "regressed",
            "reason": "slower",
            "metrics": {},
        },
    )
