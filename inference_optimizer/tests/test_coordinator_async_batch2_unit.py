# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Batch 2 coverage for Coordinator: synchronous context readers, the
no-progress circuit-breaker signal, resume replay, orchestration-conversation
reset, and lifecycle teardown (stop / Cortex T4 safety net)."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.message_bus import Message
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


# -- _context_inbox_reader --------------------------------------------------
def test_context_inbox_reader_empty(coord: Coordinator) -> None:
    out = coord._context_inbox_reader()
    assert out == "(no inbox events)"


@pytest.mark.asyncio
async def test_context_inbox_reader_with_events(coord: Coordinator) -> None:
    await coord.bus.append_and_seq(
        Message.new("kernel", "orchestration", "heartbeat", {"body_md": "hi"})
    )
    out = coord._context_inbox_reader()
    assert "(no inbox events)" not in out
    assert isinstance(out, str)


# -- _context_recent_outcomes_reader ----------------------------------------
def test_recent_outcomes_reader_empty(coord: Coordinator) -> None:
    assert coord._context_recent_outcomes_reader() == "(no recent outcomes)"


@pytest.mark.asyncio
async def test_recent_outcomes_reader_with_rows(coord: Coordinator) -> None:
    await coord.bus.append_and_seq(
        Message.new("kernel", "*", "delegated_result",
                    {"action_name": "explore", "status": "succeeded"})
    )
    out = coord._context_recent_outcomes_reader(top_k=4)
    assert "Recent action outcomes" in out


def test_recent_outcomes_reader_clamps_top_k(coord: Coordinator) -> None:
    # Out-of-range / bad top_k falls back to the default without raising.
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
    coord._reset_orchestration_conversation()  # logged, not raised
    assert coord._orchestration_seeded is False


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
    coord.shared_state.cumulative_gain_validated = 10.0  # progressed
    sig = coord._conversation_progress_signal()
    assert sig["last_progress_tick"] == 5
    assert sig["severity"] == "ok"


def test_progress_signal_flags_stall(coord: Coordinator) -> None:
    coord._progress_marker = {}
    coord._no_progress_threshold = 2
    coord._conversation_progress_signal()  # seed at tick 0
    coord.shared_state.tick = 10  # no other field changed -> stalled
    sig = coord._conversation_progress_signal()
    assert sig["ticks_without_progress"] >= 2
    assert sig["severity"] == "high"


# -- replay_for_resume ------------------------------------------------------
@pytest.mark.asyncio
async def test_replay_for_resume_rebuilds_undecided_proposals(coord: Coordinator) -> None:
    p1 = Message.new("kernel", "orchestration", "proposal",
                     {"action_name": "explore", "predicted_gain_pct": 3.0})
    await coord.bus.append_and_seq(p1)
    p2 = Message.new("kernel", "orchestration", "proposal",
                     {"action_name": "baseline", "predicted_gain_pct": 1.0})
    await coord.bus.append_and_seq(p2)
    # A verdict that decides p2 only.
    await coord.bus.append_and_seq(
        Message.new("critic", "orchestration", "review_verdict",
                    {"target_proposal_msg_id": p2.msg_id, "verdict": "approve"})
    )
    out = await coord.replay_for_resume()
    assert out["pending_restored"] == 1
    assert p1.msg_id in coord.state.pending_proposals
    assert p2.msg_id not in coord.state.pending_proposals


@pytest.mark.asyncio
async def test_replay_for_resume_verdict_map_backcompat(coord: Coordinator) -> None:
    p1 = Message.new("kernel", "orchestration", "proposal",
                     {"action_name": "explore"})
    await coord.bus.append_and_seq(p1)
    # Legacy verdict event: no 'verdict' summary but a verdict_map dict.
    await coord.bus.append_and_seq(
        Message.new("critic", "orchestration", "review_verdict",
                    {"target_proposal_msg_id": p1.msg_id,
                     "verdict_map": {"x": "ok"}})
    )
    out = await coord.replay_for_resume()
    assert out["verdicts_seen"] == 1
    assert p1.msg_id not in coord.state.pending_proposals


# -- _format_analysis_md fallback (path read) -------------------------------
def test_context_analysis_reader_path_fallback_on_format_error(
    coord: Coordinator, tmp_path, monkeypatch,
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
    coord: Coordinator, monkeypatch,
) -> None:
    coord.shared_state.last_trace_analyze = {
        "analysis_md_path": "/nonexistent/dir/analysis.md"
    }
    monkeypatch.setattr(
        coord.shared_state, "_format_analysis_md_full",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    out = coord._context_analysis_reader()
    assert "unreadable" in out or "no analysis.md" in out


# -- _cortex_t4_hook + stop -------------------------------------------------
@pytest.mark.asyncio
async def test_cortex_t4_hook_noop_without_kb(coord: Coordinator) -> None:
    coord.cortex_kb = None
    await coord._cortex_t4_hook()  # early return, no raise


@pytest.mark.asyncio
async def test_stop_cancels_and_closes(coord: Coordinator) -> None:
    await coord.stop()
    assert coord._stop.is_set()


# -- _pump_dispatcher_once --------------------------------------------------
def _sub_result(task_id: str, *, state: str = "succeeded", result=None, error=None):
    from inference_optimizer.orchestrator.sub_agent_runner import SubAgentResult
    return SubAgentResult(task_id=task_id, state=state,
                          result=result if result is not None else {}, error=error)


@pytest.mark.asyncio
async def test_pump_dispatcher_noop_when_empty(coord: Coordinator) -> None:
    await coord._pump_dispatcher_once()  # nothing queued -> early return


@pytest.mark.asyncio
async def test_pump_dispatcher_explore_promotes(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.baseline_tput = 800.0
    task = await coord.tasks.create(
        kind="explore", params={}, idempotency_key="disp-explore",
    )

    async def fake_run(t, **kw):
        return _sub_result(t.task_id, result={
            "status": "succeeded",
            "winners": [{"name": "v0", "extra_server_args": "--tp 1"}],
            "best_variant": {"name": "v0", "extra_server_args": "--tp 1"},
            "output_throughput": 900.0, "round_id": "r1",
            "losers": [], "skipped_dup": [],
        })

    monkeypatch.setattr(coord.sub, "run_task", fake_run)
    await coord._pump_dispatcher_once()
    # delegated_result landed on the bus
    tail = await coord.bus.tail(topic="delegated_result", n=10)
    assert any(m.payload.get("task_id") == task.task_id for m in tail)


@pytest.mark.asyncio
async def test_pump_dispatcher_specialist_bookkeeping(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "0")
    await coord.tasks.create(
        kind="specialist", params={"domain": "kernel"},
        idempotency_key="disp-spec",
    )

    async def fake_run(t, **kw):
        return _sub_result(t.task_id, result={
            "status": "succeeded",
            "specialist_done": {"patches_written": []},
        })

    monkeypatch.setattr(coord.sub, "run_task", fake_run)
    await coord._pump_dispatcher_once()
    tail = await coord.bus.tail(topic="delegated_result", n=10)
    assert any(m.payload.get("kind") == "specialist" for m in tail)


@pytest.mark.asyncio
async def test_pump_dispatcher_absorbs_spawn_exception(coord: Coordinator, monkeypatch) -> None:
    await coord.tasks.create(
        kind="explore", params={}, idempotency_key="disp-boom",
    )

    async def fake_run(t, **kw):
        raise RuntimeError("spawn boom")

    monkeypatch.setattr(coord.sub, "run_task", fake_run)
    # Exception is gathered with return_exceptions and logged, not propagated.
    await coord._pump_dispatcher_once()


# -- _maybe_checkpoint_orchestration (taken path) ---------------------------
class _FakeRunResult:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text


def _make_conversational(coord: Coordinator) -> None:
    backend = coord.backends["orchestration"]
    backend.conversational = True  # type: ignore[attr-defined]
    backend.reset_conversation = lambda: None  # type: ignore[attr-defined]

    async def _run(**kw):
        return _FakeRunResult("SUMMARY: explored attention backends; next: tune MoE")

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
    # next turn re-seeds from the compacted memory
    assert coord._orchestration_seeded is False
    assert coord.shared_state.orchestration_memory


# -- _compose_prompt advisory + telemetry append paths ----------------------
@pytest.mark.asyncio
async def test_compose_prompt_orchestration_all_advisory_blocks(
    coord: Coordinator, monkeypatch,
) -> None:
    ss = coord.shared_state
    monkeypatch.setattr(ss, "to_policy_denial_summary", lambda top_k=6: "DENIAL")
    monkeypatch.setattr(ss, "to_warm_start_summary", lambda: "WARM-BLOCK")
    monkeypatch.setattr(ss, "to_gaps_summary", lambda: "GAPS-BLOCK")
    monkeypatch.setattr(ss, "to_proposal_scores_summary", lambda: "SCORES-BLOCK")
    monkeypatch.setattr(ss, "to_intervention_mix_summary", lambda: "MIX-BLOCK")
    monkeypatch.setattr(coord, "_target_gap_advisory_block", lambda: "GAP-BLOCK")
    monkeypatch.setattr(coord, "_priors_match_advisory_block", lambda: "PRIORS-BLOCK")
    monkeypatch.setattr(coord, "_plateau_advisory_block", lambda: "PLATEAU-BLOCK")
    from inference_optimizer.orchestrator import research_hints as rh
    monkeypatch.setattr(rh, "summarise_for_prompt", lambda sd: "HINTS-BLOCK")

    out = await coord._compose_prompt("orchestration")
    for token in ("DENIAL", "WARM-BLOCK", "GAPS-BLOCK", "HINTS-BLOCK",
                  "GAP-BLOCK", "SCORES-BLOCK", "PRIORS-BLOCK", "MIX-BLOCK",
                  "PLATEAU-BLOCK"):
        assert token in out


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_advisory_blocks_raise(
    coord: Coordinator, monkeypatch,
) -> None:
    ss = coord.shared_state

    def _boom(*a, **k):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(ss, "to_phase_status_summary", _boom)
    monkeypatch.setattr(ss, "to_warm_start_summary", _boom)
    monkeypatch.setattr(ss, "to_gaps_summary", _boom)
    monkeypatch.setattr(ss, "to_proposal_scores_summary", _boom)
    monkeypatch.setattr(ss, "to_intervention_mix_summary", _boom)
    monkeypatch.setattr(coord, "_target_gap_advisory_block", _boom)
    monkeypatch.setattr(coord, "_priors_match_advisory_block", _boom)
    monkeypatch.setattr(coord, "_plateau_advisory_block", _boom)
    from inference_optimizer.orchestrator import research_hints as rh
    monkeypatch.setattr(rh, "summarise_for_prompt", _boom)

    # Every advisory failure is swallowed; the prompt still renders.
    out = await coord._compose_prompt("orchestration")
    assert "SESSION_DIR=" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_telemetry_raises(
    coord: Coordinator, monkeypatch,
) -> None:
    def _boom(*a, **k):
        raise RuntimeError("telemetry failed")

    monkeypatch.setattr(coord.shared_state, "to_phase_budget_telemetry", _boom)

    async def _scan_boom():
        raise RuntimeError("scan failed")

    monkeypatch.setattr(coord, "_scan_stale_specialists", _scan_boom)
    monkeypatch.setattr(coord, "_conversation_progress_signal", _boom)
    out = await coord._compose_prompt("robustness")
    assert "Specialist health" in out


@pytest.mark.asyncio
async def test_compose_prompt_robustness_lists_stale_specialists(
    coord: Coordinator, monkeypatch,
) -> None:
    async def _stale():
        return [{"task_id": "spec-stale", "running_seconds": 999}]

    monkeypatch.setattr(coord, "_scan_stale_specialists", _stale)
    out = await coord._compose_prompt("robustness")
    assert "stale specialists" in out
    assert "spec-stale" in out


# -- _promote_to_shared_state additional branches ---------------------------
def _ptask(tid: str, kind: str):
    from inference_optimizer.orchestrator.task_registry import Task
    return Task(task_id=tid, kind=kind, state="running", params={},
                idempotency_key=f"{tid}-k")


@pytest.mark.asyncio
async def test_promote_baseline_no_warmup_parses_materialized(
    coord: Coordinator, monkeypatch,
) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"  # skip cascade
    import inference_optimizer.orchestrator.coordinator as mod
    monkeypatch.setattr(mod, "_parse_baseline_workload_extra",
                        lambda path: {"isl": 256})
    await coord._promote_to_shared_state("baseline", {
        "output_throughput": 1000.0,  # no warmup_round_tput -> else branch
        "materialized_config": "/tmp/run.yaml",
    })
    assert coord.shared_state.baseline_tput == 1000.0
    assert coord.shared_state.baseline_workload_extra == {"isl": 256}


@pytest.mark.asyncio
async def test_promote_baseline_materialized_parse_raises(
    coord: Coordinator, monkeypatch,
) -> None:
    coord.shared_state.auto_roofline_pending_task_id = "pending-x"
    import inference_optimizer.orchestrator.coordinator as mod

    def _boom(path):
        raise RuntimeError("parse failed")

    monkeypatch.setattr(mod, "_parse_baseline_workload_extra", _boom)
    await coord._promote_to_shared_state("baseline", {
        "output_throughput": 1000.0,
        "materialized_config": "/tmp/run.yaml",
    })
    assert coord.shared_state.baseline_config_path == "/tmp/run.yaml"


@pytest.mark.asyncio
async def test_promote_profile_skipped_clears_pending(coord: Coordinator) -> None:
    task = _ptask("prof-1", "profile")
    coord.shared_state.auto_roofline_pending_task_id = "prof-1"
    await coord._promote_to_shared_state(
        "profile", {"status": "skipped", "error_class": "x"}, task=task,
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
        {"status": "succeeded", "main_trace_path": "/tmp/t.json",
         "output_throughput": 880.0},
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
        "roofline", {"status": "succeeded"}, task=task,
    )
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_promote_roofline_skipped_clears_pending(coord: Coordinator) -> None:
    task = _ptask("roof-2", "roofline")
    coord.shared_state.auto_roofline_pending_task_id = "roof-2"
    await coord._promote_to_shared_state(
        "roofline", {"status": "skipped"}, task=task,
    )
    assert coord.shared_state.auto_roofline_pending_task_id == ""


@pytest.mark.asyncio
async def test_promote_explore_discovered_flags_and_bad_winner(
    coord: Coordinator,
) -> None:
    coord.shared_state.baseline_tput = 800.0
    await coord._promote_to_shared_state("explore", {
        "explore_search_update": {"round_id": "r1"},
        "discovered_flags_update": {
            "framework": "sglang",
            "backend_flags": ["--x"],
            "param_flags": [],
            "source_path": "/tmp/p",
            "discovery_error": "parse glitch",
        },
        "winners": ["not-a-dict"],  # skipped by the dict guard
        "round_id": "r1",
    })
    assert coord.shared_state.discovered_flags_error == "parse glitch"


@pytest.mark.asyncio
async def test_promote_framework_pr_updates_batch_max_gain(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.framework_pr_batches = [
        {"batch_id": "b1", "max_gain_pct_observed_in_batch": 1.0},
    ]
    await coord._promote_to_shared_state("framework_pr", {
        "status": "kept",
        "candidate": {"candidate_id": "c1", "pr_url": "http://x/1"},
        "batch_id": "b1",
        "delta_pct": 7.0,
        "output_throughput": 856.0,
    })
    assert coord.shared_state.framework_pr_batches[0][
        "max_gain_pct_observed_in_batch"] == 7.0


@pytest.mark.asyncio
async def test_promote_sweep_chains_conc_sweep(coord: Coordinator) -> None:
    coord.shared_state.conc_sweep_enabled = True
    await coord._promote_to_shared_state("sweep", {
        "status": "succeeded",
        "grid_size": 4,
        "pareto_front": [{"x": 1}],
    })


@pytest.mark.asyncio
async def test_promote_conc_sweep_records(coord: Coordinator) -> None:
    task = _ptask("cs-1", "conc_sweep")
    await coord._promote_to_shared_state("conc_sweep", {
        "status": "succeeded",
        "was_skipped": False,
        "summary": {"best_speedup": 1.2, "best_conc": 64, "successful_pairs": 3},
        "report_json_path": "/tmp/cs.json",
    }, task=task)


# -- _scan_stale_specialists (running rows) ---------------------------------
@pytest.mark.asyncio
async def test_scan_stale_specialists_flags_running(coord: Coordinator) -> None:
    coord._specialist_stale_sec = 0  # any running specialist is "stale"
    spec = await coord.tasks.create(
        kind="specialist", params={}, idempotency_key="stale-spec",
    )
    await coord.tasks.transition(spec.task_id, "running")
    # A non-specialist running task is skipped by the kind guard.
    other = await coord.tasks.create(
        kind="explore", params={}, idempotency_key="stale-other",
    )
    await coord.tasks.transition(other.task_id, "running")
    stale = await coord._scan_stale_specialists()
    assert any(r["task_id"] == spec.task_id for r in stale)
    assert all(r["task_id"] != other.task_id for r in stale)


# -- _fan_out_specialist_wave (valid entries) -------------------------------
@pytest.mark.asyncio
async def test_fan_out_wave_dispatches_valid_task(coord: Coordinator, monkeypatch) -> None:
    seen: list[dict] = []

    async def _fake_delegate(source, intent):
        seen.append(dict(intent.payload.get("params") or {}))

    monkeypatch.setattr(coord, "_handle_delegate", _fake_delegate)
    intent = Intent(type=IntentType.DELEGATE, payload={"idempotency_key": "wave"})
    await coord._fan_out_specialist_wave("orchestration", intent, {
        "domain": "kernel",
        "tasks": [
            {"task_description": "scout fused moe", "task_summary": "moe",
             "mode": "patch", "lane": "gpu"},
        ],
    })
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
    monkeypatch.setattr(state, "find_gap", lambda cid: {
        "symptom": "mem bound", "layer": "attention",
        "domain_hint": "kernel", "severity": "high",
        "attempts": [{"r": 1}],
    })
    monkeypatch.setattr(coord, "_target_gap_advisory_block", lambda: "GAP-NOTES")
    from inference_optimizer.orchestrator import research_hints as rh
    monkeypatch.setattr(rh, "summarise_for_prompt", lambda sd: "HINTS-TEXT")
    import inference_optimizer.orchestrator.shared_state as ss_mod
    monkeypatch.setattr(ss_mod, "render_model_arch_compact", lambda a: "ARCH-NOTES")
    from inference_optimizer.orchestrator import framework_paths as fp
    monkeypatch.setattr(fp, "resolve_source_file_allowlist", lambda: ["/src/root"])

    params: dict = {"domain": "kernel", "gap_canonical_id": "g1"}
    await coord._warm_specialist_params(params)
    assert params["framework_version"] == "0.4.1"
    assert params["target_gap_notes"] == "GAP-NOTES"
    assert params["research_hints"] == "HINTS-TEXT"
    assert params["arch_notes"] == "ARCH-NOTES"
    assert params["framework_source_roots"] == ["/src/root"]
    assert params["gap_symptom"] == "mem bound"
    assert "roofline_evidence" in params


# -- _record_fact_per_task (cortex KB path) ---------------------------------
@pytest.mark.asyncio
async def test_record_fact_per_task_writes_lesson(coord: Coordinator, monkeypatch) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    coord.cortex_kb = object()  # non-None -> KB amend path
    coord.shared_state.model_name = "llama"
    coord.shared_state.gpu_type = "mi300x"
    amends: list[dict] = []
    monkeypatch.setattr(coord, "_kb_amend_recipe", lambda **k: amends.append(k))
    task = Task(task_id="fact-keep", kind="explore", state="succeeded",
                params={}, idempotency_key="fk")
    coord._record_fact_per_task(
        task=task, source_session_id="sess",
        result_dict={"gain_pct": 6.0, "output_throughput": 950.0}, kept=True,
    )
    assert amends and "append_lesson" in amends[0]


@pytest.mark.asyncio
async def test_record_fact_per_task_writes_pitfall(coord: Coordinator, monkeypatch) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    coord.cortex_kb = object()
    amends: list[dict] = []
    monkeypatch.setattr(coord, "_kb_amend_recipe", lambda **k: amends.append(k))
    monkeypatch.setattr(coord, "_pitfall_severity_for", lambda rd: "high")
    task = Task(task_id="fact-revert", kind="integrate_patch", state="failed",
                params={}, idempotency_key="fr")
    coord._record_fact_per_task(
        task=task, source_session_id="sess",
        result_dict={"error_class": "oom", "reason": "bad"}, kept=False,
    )
    assert amends and "append_pitfall" in amends[0]


# -- _plateau_advisory_block (triggered) ------------------------------------
@pytest.mark.asyncio
async def test_plateau_advisory_explore_triggered(coord: Coordinator, monkeypatch) -> None:
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = ps.PHASE_EXPLORE
    monkeypatch.setattr(ps, "compute_plateau_explore",
                        lambda *a, **k: (True, {"recent_keep_gain_pct": 0.1,
                                                "empty_streak": 3}))
    out = coord._plateau_advisory_block()
    assert "EXPLORE plateau detected" in out
    # Cyclic mode (default): footer must state the deterministic EXPLORE→KERNEL
    # advance, not the stale "informational only" claim.
    assert "advances EXPLORE" in out and "KERNEL" in out
    assert "informational" not in out


@pytest.mark.asyncio
async def test_plateau_advisory_kernel_triggered(coord: Coordinator, monkeypatch) -> None:
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = ps.PHASE_KERNEL
    monkeypatch.setattr(ps, "compute_plateau_kernel",
                        lambda *a, **k: (True, {"revert_streak": 4}))
    out = coord._plateau_advisory_block()
    assert "KERNEL plateau detected" in out


@pytest.mark.asyncio
async def test_plateau_advisory_framework_pr_triggered(coord: Coordinator, monkeypatch) -> None:
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = ps.PHASE_FRAMEWORK_PR
    monkeypatch.setattr(ps, "compute_plateau_framework_pr",
                        lambda *a, **k: (True, {"lookback": 3,
                                                "batch_max_gains": [0.1]}))
    out = coord._plateau_advisory_block()
    assert "FRAMEWORK_PR plateau detected" in out


# -- _record_specialist_result ----------------------------------------------
@pytest.mark.asyncio
async def test_record_specialist_result_with_proposals(coord: Coordinator) -> None:
    task = _ptask("rec-spec-1", "specialist")
    await coord._record_specialist_result(
        task=task,
        done_payload={
            "domain": "kernel",
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
    coord: Coordinator, caplog,
) -> None:
    """Regression (#486 leftover): the deleted ``_aggregate_research_evidence``
    call raised AttributeError that was swallowed into a spammy error log on
    every specialist result. The dead call must be gone, so no such error is
    logged."""
    import logging

    task = _ptask("rec-spec-dead", "specialist")
    with caplog.at_level(logging.ERROR):
        await coord._record_specialist_result(
            task=task,
            done_payload={
                "domain": "kernel",
                "proposal_set": [{"name": "p1"}],
            },
            source="specialist:rec-spec-dead",
        )
    assert not any(
        "research-evidence aggregation failed" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_record_specialist_result_research_scout(coord: Coordinator, monkeypatch) -> None:
    task = _ptask("rec-spec-2", "specialist")
    harvested: list[dict] = []
    monkeypatch.setattr(coord, "_harvest_research_scout",
                        lambda dp: harvested.append(dp))
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
    assert harvested  # research-scout harvest fired


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
            "domain": "kernel",
            "proposal_set": [{"name": "p1"}],
        },
        source="specialist:rec-spec-3",
    )


# -- cortex_finalize_recipe_and_journal (KB path) ---------------------------
class _FakeLocal:
    def get_recipe(self, *, canonical_id):
        return {
            "best_throughput": 0.0,
            "sessions": [],
            "kernel_optimizations": [],
            "stack_fingerprint": {},
        }


class _FakeCortexKB:
    def __init__(self) -> None:
        self.local = _FakeLocal()


@pytest.mark.asyncio
async def test_cortex_finalize_skips_without_model(coord: Coordinator) -> None:
    coord.cortex_kb = _FakeCortexKB()
    coord.shared_state.model_name = ""  # missing model -> skip update_recipe
    coord.shared_state.gpu_type = "mi300x"
    coord.cortex_finalize_recipe_and_journal()  # no raise, early return


@pytest.mark.asyncio
async def test_cortex_finalize_amends_recipe(coord: Coordinator, monkeypatch) -> None:
    coord.cortex_kb = _FakeCortexKB()
    coord.shared_state.model_name = "llama"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.cumulative_gain_validated = 12.0
    coord.shared_state.current_best = {"tput": 950.0}
    amends: list[dict] = []
    monkeypatch.setattr(coord, "_kb_amend_recipe", lambda **k: amends.append(k))
    coord.cortex_finalize_recipe_and_journal()
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
    monkeypatch.setattr(coord, "_inline_action_whitelist", lambda: {"report"})
    out = coord._run_action_now_sync("explore")
    assert "not inline-eligible" in out


def test_run_action_now_sync_no_loop(coord: Coordinator, monkeypatch) -> None:
    coord._inline_fast_actions_enabled = True
    monkeypatch.setattr(coord, "_inline_action_whitelist", lambda: {"report"})
    coord._coordinator_loop = None
    out = coord._run_action_now_sync("report")
    assert "coordinator loop not running" in out


# -- _handle_intent routing -------------------------------------------------
@pytest.mark.asyncio
async def test_handle_intent_policy_denied(coord: Coordinator, monkeypatch) -> None:
    from inference_optimizer.orchestrator.policy import PolicyDenied
    recorded: list = []

    def _deny(source, intent):
        raise PolicyDenied("nope")

    monkeypatch.setattr(coord.policy, "validate_intent", _deny)

    async def _rec(source, intent, denied):
        recorded.append(denied)

    monkeypatch.setattr(coord, "_record_policy_denied", _rec)
    await coord._handle_intent("orchestration", _heartbeat())
    assert recorded


@pytest.mark.asyncio
async def test_handle_intent_handler_exception_is_recorded(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.policy, "validate_intent", lambda s, i: None)

    async def _boom(source, intent):
        raise RuntimeError("handler boom")

    monkeypatch.setattr(coord, "_handle_send_message", _boom)
    # SEND_MESSAGE handler raises -> exception is logged + recorded, not raised.
    await coord._handle_intent("orchestration", _heartbeat())


@pytest.mark.asyncio
async def test_handle_intent_routes_rare_types(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setattr(coord.policy, "validate_intent", lambda s, i: None)
    seen: list[str] = []

    async def _mk(name):
        async def _h(source, intent):
            seen.append(name)
        return _h

    routes = {
        IntentType.KILL_TASK: "_handle_kill_task",
        IntentType.PRUNE_BRANCH: "_handle_prune_branch",
        IntentType.FORCE_DISPATCH: "_handle_force_dispatch",
        IntentType.ALERT: "_handle_alert",
        IntentType.UPDATE_STATE: "_handle_update_state",
        IntentType.SPECIALIST_DONE: "_handle_specialist_done",
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
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = "EXPLORE"
    monkeypatch.setattr(ps, "compute_next_phase",
                        lambda *a, **k: ("EXPLORE", "x", {}))

    async def _scout():
        return None

    monkeypatch.setattr(coord, "_maybe_enqueue_explore_research_scout", _scout)
    await coord._advance_phase_if_needed()  # target == current -> early return


@pytest.mark.asyncio
async def test_advance_phase_escalation_transition(coord: Coordinator, monkeypatch) -> None:
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = "PRELUDE"
    monkeypatch.setattr(ps, "compute_next_phase",
                        lambda *a, **k: ("EXPLORE", "robustness_escalated",
                                         {"evidence": "llm_escalation"}))

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert (coord.shared_state.phase or "").upper() == "EXPLORE"


@pytest.mark.asyncio
async def test_advance_phase_terminal_sets_stop_reason(coord: Coordinator, monkeypatch) -> None:
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = "SWEEP"
    coord.shared_state.stop_reason = ""
    monkeypatch.setattr(ps, "compute_next_phase",
                        lambda *a, **k: (ps.PHASE_CLOSE, "target_reached",
                                         {"terminal": True}))

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(coord, "_on_phase_entered", _entered)
    await coord._advance_phase_if_needed()
    assert coord.shared_state.stop_reason == "target_reached"


# -- _materialize_approved_proposal -----------------------------------------
def _pending(action_name: str, payload: dict, msg_id: str = "prop-1"):
    from inference_optimizer.orchestrator.coordinator import PendingProposal
    return PendingProposal(
        proposal_msg_id=msg_id, from_agent="orchestration",
        action_name=action_name, predicted_gain_pct=3.0, payload=payload,
    )


@pytest.mark.asyncio
async def test_materialize_explore_filters_grid(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    pending = _pending("explore", {"params": {"grid": [
        {"name": "v0"}, {"name": "v1"}, "non-dict-slot",
    ]}})
    await coord._materialize_approved_proposal(
        pending, approved_variant_names={"v0"},
    )
    tail = await coord.bus.tail(topic="decision", n=10)
    assert any(m.payload.get("kind") == "approved_proposal" for m in tail)


@pytest.mark.asyncio
async def test_materialize_sweep_stamps_base(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.current_best = {"tput": 900.0, "extra_server_args": "--tp 1"}
    coord.shared_state.baseline_config_path = "/tmp/base.yaml"
    pending = _pending("sweep", {"params": {}}, msg_id="prop-sweep")
    await coord._materialize_approved_proposal(pending)
    task = await coord.tasks.get(
        (await coord.tasks.queued())[0].task_id
    )
    assert task.kind == "sweep"


@pytest.mark.asyncio
async def test_materialize_duplicate_idempotency_skips(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    pending = _pending("profile", {"params": {}}, msg_id="prop-dup")
    await coord._materialize_approved_proposal(pending)
    # Second call collides on idempotency key -> records skip observation.
    await coord._materialize_approved_proposal(pending)


# -- _handle_delegate branches ----------------------------------------------
def _delegate(action_name: str, key: str, params=None) -> Intent:
    payload = {"action_name": action_name, "params": params or {},
               "idempotency_key": key}
    return Intent(type=IntentType.DELEGATE, payload=payload)


@pytest.mark.asyncio
async def test_handle_delegate_pruned_advisory(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.baseline_tput = 800.0
    monkeypatch.setattr(coord.shared_state, "is_pruned", lambda a: True)
    monkeypatch.setattr(coord, "_sequence_denial_for_action", lambda a, proposed_params=None: None)
    await coord._handle_delegate("orchestration", _delegate("explore", "d-pruned"))
    # advisory observation recorded but the task is still queued
    assert (await coord.tasks.queued())


@pytest.mark.asyncio
async def test_handle_delegate_sequence_denied(coord: Coordinator, monkeypatch) -> None:
    from inference_optimizer.orchestrator.policy import PolicyDenied
    monkeypatch.setattr(
        coord, "_sequence_denial_for_action",
        lambda a, proposed_params=None: PolicyDenied(
            "blocked", rule="exec_order", hint="wait",
        ),
    )
    recorded: list = []

    async def _rec(source, intent, denied, action_name=None):
        recorded.append(denied)

    monkeypatch.setattr(coord, "_record_policy_denied", _rec)
    await coord._handle_delegate("orchestration", _delegate("explore", "d-seq"))
    assert recorded


@pytest.mark.asyncio
async def test_handle_delegate_duplicate_running_denied(coord: Coordinator, monkeypatch) -> None:
    coord.shared_state.baseline_tput = 800.0
    monkeypatch.setattr(coord, "_sequence_denial_for_action", lambda a, proposed_params=None: None)
    await coord._handle_delegate("orchestration", _delegate("explore", "d-same"))
    recorded: list = []

    async def _rec(source, intent, denied, action_name=None):
        recorded.append(denied)

    monkeypatch.setattr(coord, "_record_policy_denied", _rec)
    # Same key while the first task is still queued (non-terminal) -> denied.
    await coord._handle_delegate("orchestration", _delegate("explore", "d-same"))
    assert recorded


# -- _maybe_autosubmit_specialist_patches early returns ---------------------
def _make_real_patch(coord: Coordinator, sid: str) -> None:
    from inference_optimizer.session_paths import runs_dir
    wt = runs_dir(coord.session_dir, "specialist", sid) / "worktree"
    wt.mkdir(parents=True, exist_ok=True)
    (wt / "kernel.py").write_text("# patched\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_autosubmit_returns_when_verdict_exists(coord: Coordinator, monkeypatch) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    sid = "spec-verdict"
    _make_real_patch(coord, sid)
    monkeypatch.setattr(coord.shared_state, "get_specialist_patch_verdict",
                        lambda s: {"verdict": "approve"})
    task = Task(task_id=sid, kind="specialist", state="running",
                params={}, idempotency_key="kv1")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task, done_payload={"patches_written": ["kernel.py"]},
    )
    assert len(coord.state.pending_proposals) == n_before  # no new proposal


@pytest.mark.asyncio
async def test_autosubmit_returns_when_review_in_flight(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.task_registry import Task
    from inference_optimizer.orchestrator.coordinator import PendingProposal
    sid = "spec-inflight"
    _make_real_patch(coord, sid)
    coord.state.pending_proposals["existing"] = PendingProposal(
        proposal_msg_id="existing", from_agent="coordinator",
        action_name="integrate_patch", predicted_gain_pct=0.0,
        payload={"params": {"specialist_task_id": sid}},
    )
    task = Task(task_id=sid, kind="specialist", state="running",
                params={}, idempotency_key="kv2")
    n_before = len(coord.state.pending_proposals)
    await coord._maybe_autosubmit_specialist_patches(
        task=task, done_payload={"patches_written": ["kernel.py"]},
    )
    assert len(coord.state.pending_proposals) == n_before  # in-flight -> skip


# -- _promote_warm_replay branches ------------------------------------------
def _warm_task():
    from inference_optimizer.orchestrator.task_registry import Task
    return Task(task_id="warm-x", kind="replay_warm_recipe", state="running",
                params={"extra_envs": {"HSA_FORCE": "1"},
                        "baseline_tput_anchor": 800.0},
                idempotency_key="warm-k")


def test_promote_warm_replay_already_pushed(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.optimization_stack = [{"action": "replay_warm_recipe"}]
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 900.0}, task=_warm_task(),
    )
    # idempotency guard keeps a single warm_replay entry
    n = sum(1 for e in coord.shared_state.optimization_stack
            if isinstance(e, dict) and e.get("action") == "replay_warm_recipe")
    assert n == 1


def test_promote_warm_replay_drift(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 750.0},  # below baseline
        task=_warm_task(),
    )
    assert coord.shared_state.warm_replay_outcome["status"] == "drift"


def test_promote_warm_replay_below_historical_bar(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.warm_replay_outcome = {"expected_gain_pct": 20.0}
    coord._promote_warm_replay(
        {"status": "succeeded", "output_throughput": 810.0},  # +1.25% < 16% bar
        task=_warm_task(),
    )
    out = coord.shared_state.warm_replay_outcome
    assert out.get("below_historical_reproduce_pct") is True


# -- cortex_finalize_recipe_and_journal (rich existing row merge) -----------
class _FakeLocalRich:
    def get_recipe(self, *, canonical_id):
        return {
            "best_throughput": 100.0,
            "sessions": [{"session_id": "other-session"}],
            "kernel_optimizations": [{"kernel_id": "k-old"}],
            "stack_fingerprint": {"sglang": "0.1"},
        }


class _FakeCortexKBRich:
    def __init__(self) -> None:
        self.local = _FakeLocalRich()


@pytest.mark.asyncio
async def test_cortex_finalize_merges_existing_row(coord: Coordinator, monkeypatch) -> None:
    coord.cortex_kb = _FakeCortexKBRich()
    coord.shared_state.model_name = "llama"
    coord.shared_state.gpu_type = "mi300x"
    coord.shared_state.cumulative_gain_validated = 15.0
    coord.shared_state.current_best = {"tput": 999.0}
    amends: list[dict] = []
    monkeypatch.setattr(coord, "_kb_amend_recipe", lambda **k: amends.append(k))
    coord.cortex_finalize_recipe_and_journal()
    assert amends
    overrides = amends[0]["recipe_overrides"]
    # prior session + kernel optimization were merged forward
    assert any(s.get("session_id") == "other-session" for s in overrides["sessions"])


# -- _on_enter_close 5-step sequencer ---------------------------------------
@pytest.mark.asyncio
async def test_on_enter_close_runs_full_sequence(coord: Coordinator, monkeypatch) -> None:
    async def _fake_run(task, **kw):
        from inference_optimizer.orchestrator.sub_agent_runner import SubAgentResult
        return SubAgentResult(task_id=task.task_id, state="succeeded",
                              result={}, error=None)

    monkeypatch.setattr(coord.sub, "run_task", _fake_run)
    await coord._on_enter_close(from_phase="SWEEP")
    assert coord.shared_state.close_sequence_done is True
    assert coord.shared_state.stop_reason  # derived stop reason persisted


# -- _pump_framework_pr_phase -----------------------------------------------
def _enter_framework_pr(coord: Coordinator) -> None:
    import inference_optimizer.orchestrator.phase_state as ps
    coord.shared_state.phase = ps.PHASE_FRAMEWORK_PR
    coord.shared_state.framework_pr_phase_done = False


@pytest.mark.asyncio
async def test_pump_framework_pr_wrong_phase_noop(coord: Coordinator) -> None:
    coord.shared_state.phase = "EXPLORE"
    await coord._pump_framework_pr_phase()  # early return


@pytest.mark.asyncio
async def test_pump_framework_pr_phase_done_noop(coord: Coordinator) -> None:
    _enter_framework_pr(coord)
    coord.shared_state.framework_pr_phase_done = True
    await coord._pump_framework_pr_phase()  # early return


@pytest.mark.asyncio
async def test_pump_framework_pr_skips_when_task_inflight(coord: Coordinator) -> None:
    _enter_framework_pr(coord)
    await coord.tasks.create(kind="framework_pr", params={},
                             idempotency_key="fpr-inflight")
    await coord._pump_framework_pr_phase()  # a framework_pr task exists -> return


@pytest.mark.asyncio
async def test_pump_framework_pr_discover_empty_marks_done(coord: Coordinator, monkeypatch) -> None:
    _enter_framework_pr(coord)
    coord.shared_state.framework_pr_discover_failures = 0
    monkeypatch.setattr(coord, "_select_next_framework_pr_candidate", lambda: None)

    async def _disc():
        return False

    monkeypatch.setattr(coord, "_discover_next_framework_pr_batch", _disc)
    monkeypatch.setattr(coord, "_record_framework_pr_phase_done",
                        lambda **k: None)
    await coord._pump_framework_pr_phase()
    assert coord.shared_state.framework_pr_phase_done is True


@pytest.mark.asyncio
async def test_pump_framework_pr_critic_rejects(coord: Coordinator, monkeypatch) -> None:
    _enter_framework_pr(coord)
    monkeypatch.setattr(coord, "_select_next_framework_pr_candidate",
                        lambda: {"candidate_id": "c1", "batch_id": "b1"})

    async def _review(cand):
        return {"verdict": "reject", "rationale": "unsafe"}

    monkeypatch.setattr(coord, "_critic_review_framework_pr_candidate", _review)
    await coord._pump_framework_pr_phase()
    prog = coord.shared_state.framework_pr_phase_progress
    assert any(p.get("status") == "critic_denied" for p in prog)


@pytest.mark.asyncio
async def test_pump_framework_pr_approve_enqueues(coord: Coordinator, monkeypatch) -> None:
    _enter_framework_pr(coord)
    monkeypatch.setattr(coord, "_select_next_framework_pr_candidate",
                        lambda: {"candidate_id": "c2", "batch_id": "b2"})

    async def _review(cand):
        return {"verdict": "approve"}

    monkeypatch.setattr(coord, "_critic_review_framework_pr_candidate", _review)
    enq: list = []

    async def _enqueue(cand):
        enq.append(cand)

    monkeypatch.setattr(coord, "_enqueue_framework_pr_task", _enqueue)
    await coord._pump_framework_pr_phase()
    assert enq
