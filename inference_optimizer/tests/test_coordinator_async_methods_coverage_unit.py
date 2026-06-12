# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for Coordinator async/stateful methods invoked directly against a
real (mock-backed) Coordinator: SharedState promotion across task kinds, prompt
composition per agent, advisory blocks, research-scout harvest, and the
orchestration checkpoint guard."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker
    seed_target_analysis_marker(sd)
    return sd


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {
        name: MockBackend(_silent_plan(), name=name)
        for name in ("orchestration", "kernel", "critic", "robustness")
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
    await coord._promote_to_shared_state("baseline", {
        "output_throughput": 1000.0,
        "warmup_round_tput": 900.0,
        "accuracy": 0.95,
        "subprocess_runtime_sec": 120.0,
        "ttft_mean_ms": 100.0,
        "e2el_mean_ms": 2000.0,
        "tpot_mean_ms": 10.0,
        "workspace": "/tmp/ws",
    })
    # warmup anchor wins as the comparison baseline; hot number kept separately
    assert coord.shared_state.baseline_tput == 900.0
    assert coord.shared_state.baseline_hot_tput == 1000.0
    assert coord.shared_state.baseline_failure_streak == 0
    assert coord.shared_state.baseline_arg_error_streak == 0
    assert coord.shared_state.current_best["action"] == "baseline"


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
async def test_promote_profile_succeeded_records_trace(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state("profile", {
        "status": "succeeded",
        "main_trace_path": "/tmp/trace.json",
        "output_throughput": 820.0,
    })
    assert coord.shared_state.last_profile_status == "succeeded"
    assert coord.shared_state.last_profile_trace == "/tmp/trace.json"


@pytest.mark.asyncio
async def test_promote_profile_failed_clears_trace(coord: Coordinator) -> None:
    await coord._promote_to_shared_state("profile", {
        "status": "failed",
        "error_class": "no_trace_files",
    })
    assert coord.shared_state.last_profile_status == "failed"


@pytest.mark.asyncio
async def test_promote_roofline_succeeded_and_skipped_and_failed(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state("roofline", {"status": "succeeded"})
    await coord._promote_to_shared_state("roofline", {"status": "skipped"})
    await coord._promote_to_shared_state("roofline", {
        "status": "failed", "error_class": "boom", "phase": "trace",
    })
    # failure streak bumped on the failed branch
    assert getattr(coord.shared_state, "roofline_failure_streak", 0) >= 1


@pytest.mark.asyncio
async def test_promote_explore_with_winner(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state("explore", {
        "winners": [{"name": "v0", "extra_server_args": "--tp 1"}],
        "best_variant": {"name": "v0", "extra_server_args": "--tp 1"},
        "output_throughput": 900.0,
        "round_id": "r1",
        "losers": [],
        "skipped_dup": [],
    })
    assert coord.shared_state.current_best.get("tput") == 900.0


@pytest.mark.asyncio
async def test_promote_framework_pr_kept(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state("framework_pr", {
        "status": "kept",
        "candidate": {"candidate_id": "c1", "pr_url": "http://x/1"},
        "batch_id": "b1",
        "delta_pct": 5.0,
        "output_throughput": 840.0,
        "workspace": "/tmp/ws",
    })
    assert isinstance(coord.shared_state.framework_pr_phase_progress, list)
    assert coord.shared_state.framework_pr_phase_progress[-1]["kept"] is True


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
    out_k = await coord._compose_prompt("kernel")
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
    coord._harvest_research_scout({"research": {
        "hints": {"what_to_try": ["aiter"]},
        "gaps": [],
    }})


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
    from inference_optimizer.orchestrator.phase_state import (
        ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
    )
    await coord._handle_escalate_strategy_change(
        "orchestration", _escalate(ESCALATE_HINT_EXTEND_EXPLORE_BUDGET),
    )
    assert coord.shared_state.last_consumed_escalate_hint == ESCALATE_HINT_EXTEND_EXPLORE_BUDGET


@pytest.mark.asyncio
async def test_escalate_extend_kernel_budget(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.phase_state import (
        ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
    )
    await coord._handle_escalate_strategy_change(
        "orchestration", _escalate(ESCALATE_HINT_EXTEND_KERNEL_BUDGET),
    )
    assert coord.shared_state.last_consumed_escalate_hint == ESCALATE_HINT_EXTEND_KERNEL_BUDGET


@pytest.mark.asyncio
async def test_escalate_pause_specialist(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change(
        "orchestration", _escalate("pause_specialist_kernel"),
    )
    assert coord.shared_state.last_consumed_escalate_hint == "pause_specialist_kernel"


@pytest.mark.asyncio
async def test_escalate_skip_to_kernel_deferred(coord: Coordinator) -> None:
    await coord._handle_escalate_strategy_change(
        "orchestration", _escalate("skip_to_kernel"),
    )
    # deferred hint queued for the next compute_next_phase
    assert coord.shared_state.pending_escalate_hint == "skip_to_kernel"


# -- _scan_stale_specialists -----------------------------------------------
@pytest.mark.asyncio
async def test_scan_stale_specialists_empty(coord: Coordinator) -> None:
    assert await coord._scan_stale_specialists() == []


# -- _maybe_autosubmit_specialist_patches ----------------------------------
@pytest.mark.asyncio
async def test_autosubmit_skipped_when_no_patches(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    task = Task(task_id="spec-1", kind="specialist", state="running",
                params={}, idempotency_key="k1")
    await coord._maybe_autosubmit_specialist_patches(
        task=task, done_payload={"patches_written": []},
    )  # empty list -> early return


@pytest.mark.asyncio
async def test_autosubmit_skipped_when_files_missing(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    task = Task(task_id="spec-2", kind="specialist", state="running",
                params={}, idempotency_key="k2")
    await coord._maybe_autosubmit_specialist_patches(
        task=task, done_payload={"patches_written": ["ghost.py"]},
    )  # claimed file does not exist -> records skip observation, returns


@pytest.mark.asyncio
async def test_autosubmit_creates_proposal_for_real_file(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    from inference_optimizer.session_paths import runs_dir
    sid = "spec-3"
    spec_root = runs_dir(coord.session_dir, "specialist", sid)
    wt = spec_root / "worktree"
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "kernel.py").write_text("# patched\n", encoding="utf-8")
    task = Task(task_id=sid, kind="specialist", state="running",
                params={}, idempotency_key="k3")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task,
        done_payload={
            "patches_written": ["kernel.py"],
            "proposal_set": [{"name": "fuse-moe"}],
        },
    )
    assert len(coord.state.pending_proposals) == n_before + 1


# -- _record_fact_per_task -------------------------------------------------
def test_record_fact_per_task_keep_and_revert(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    task = Task(task_id="t-fact", kind="explore", state="succeeded",
                params={}, idempotency_key="kf")
    coord._record_fact_per_task(
        task=task, source_session_id="sess-a",
        result_dict={"gain_pct": 5.0, "output_throughput": 900.0}, kept=True,
    )
    coord._record_fact_per_task(
        task=task, source_session_id="sess-a",
        result_dict={"error_class": "boom", "reason": "bad"}, kept=False,
    )


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
    monkeypatch.setattr(coord, "_orchestration_conversational", lambda: True)
    coord._orchestration_seeded = True  # DELTA turn -> push_full False
    out = await coord._compose_prompt("orchestration")
    assert "Context (pull on demand)" in out


@pytest.mark.asyncio
async def test_compose_prompt_conversational_seed_memory(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord, "_orchestration_conversational", lambda: True)
    coord._orchestration_seeded = False  # SEED turn -> push_full True
    coord._orchestration_seed_memory = "=== recovered memory ==="
    out = await coord._compose_prompt("orchestration")
    assert "recovered memory" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_high_no_progress(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(
        coord, "_conversation_progress_signal",
        lambda: {
            "ticks_without_progress": 9, "threshold": 5,
            "severity": "high", "last_progress_tick": 1,
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
    from inference_optimizer.orchestrator import research_hints as rh
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
    from inference_optimizer.orchestrator import research_hints as rh
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
        {"status": "succeeded", "output_throughput": 900.0}, task=None,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced_but_no_params"


def test_promote_warm_replay_reproduced_with_params(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    coord.shared_state.baseline_tput = 800.0
    task = Task(task_id="warm-1", kind="replay_warm_recipe", state="running",
                params={"extra_envs": {"A": "1"},
                        "baseline_tput_anchor": 800.0},
                idempotency_key="kw")
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0}, task=task,
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"


# -- _maybe_auto_retry_specialist ------------------------------------------
def _spec_task(**params):
    from inference_optimizer.orchestrator.task_registry import Task
    return Task(task_id="spec-r", kind="specialist", state="running",
                params=params, idempotency_key="spec-r-key")


def _result(state="failed", result=None, error=None):
    from inference_optimizer.orchestrator.sub_agent_runner import SubAgentResult
    return SubAgentResult(task_id="spec-r", state=state,
                          result=result or {}, error=error)


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
        coord, "_handle_delegate",
        lambda *a, **k: called.append(a),
    )
    intent = Intent(type=IntentType.DELEGATE, payload={"idempotency_key": "w"})
    await coord._fan_out_specialist_wave(
        "orchestration", intent,
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
    params: dict = {"domain": "kernel"}
    await coord._warm_specialist_params(params)
    assert params["gpu_type"] == "mi300x"
    assert params["tp"] == 1
    assert "pr_feed" in params
    assert "kb_subgraph" in params


# -- cortex_finalize_recipe_and_journal ------------------------------------
def test_cortex_finalize_recipe_and_journal_no_kb(coord: Coordinator) -> None:
    coord.shared_state.current_best = {"tput": 950.0}
    coord.shared_state.cumulative_gain_validated = 12.5
    # cortex_kb is None in the mock harness -> journal finalize then early return
    coord.cortex_finalize_recipe_and_journal()


# -- _record_fact_per_variant ----------------------------------------------
def test_record_fact_per_variant_keep_revert_skip(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    task = Task(task_id="t-var", kind="explore", state="succeeded",
                params={}, idempotency_key="kv")
    # SKIPPED_DEDUP -> early return (no journal row)
    coord._record_fact_per_variant(
        task=task, source_session_id="s",
        variant_outcome={"outcome": "SKIPPED_DEDUP", "variant_name": "v0"},
    )
    # KEEP path
    coord._record_fact_per_variant(
        task=task, source_session_id="s",
        variant_outcome={
            "outcome": "KEEP", "variant_name": "v1",
            "metrics": {"gain_pct": 4.0, "output_throughput": 900.0},
            "variant": {"name": "v1"},
        },
    )
    # REVERT path with error_class/reason
    coord._record_fact_per_variant(
        task=task, source_session_id="s",
        variant_outcome={
            "outcome": "REVERT", "variant_name": "v2",
            "error_class": "regressed", "reason": "slower",
            "metrics": {},
        },
    )
