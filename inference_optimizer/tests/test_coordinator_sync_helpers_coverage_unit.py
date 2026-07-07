# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for Coordinator pure/sync helper methods.

Builds one Coordinator with mock backends and exercises the formatting / gap /
fact / tag helpers directly (both empty-guard and populated paths), avoiding the
async event loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType


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


# -- WS1: explicit specialist wall-clock budget ----------------------------
def test_specialist_wall_budget_base_no_macro_cycle(coord: Coordinator) -> None:
    # ≤24h runs keep macro_cycle == 0 → base lane values (cpu 10min / gpu 60min).
    coord.shared_state.macro_cycle = 0
    assert coord._specialist_wall_budget_sec(needs_gpu=False) == 10 * 60
    assert coord._specialist_wall_budget_sec(needs_gpu=True) == 60 * 60


def test_specialist_wall_budget_macro_cycle_amplifies(coord: Coordinator) -> None:
    coord.shared_state.macro_cycle = 1
    # cpu: 10 × (1+1) = 20min; gpu: 60 × 2 = 120min.
    assert coord._specialist_wall_budget_sec(needs_gpu=False) == 20 * 60
    assert coord._specialist_wall_budget_sec(needs_gpu=True) == 120 * 60


def test_specialist_wall_budget_caps_at_4h(coord: Coordinator) -> None:
    coord.shared_state.macro_cycle = 10
    # gpu: 60 × 11 = 660min → capped at 240min (4h); cpu: 10 × 11 = 110min.
    assert coord._specialist_wall_budget_sec(needs_gpu=True) == 240 * 60
    assert coord._specialist_wall_budget_sec(needs_gpu=False) == 110 * 60


# -- WS2: GPU lease TTL re-source + structured-finally release --------------
def test_gpu_lease_ttl_grace_over_wall_budget(coord: Coordinator) -> None:
    # TTL = wall_budget × (1 + grace); the lease must outlive the kill so cards
    # are never reclaimed mid-run (iron law kill ≤ gpu_lease TTL ≤ lane TTL).
    from hyperloom.orchestrator.bus.gpu_pool import GPU_LEASE_TTL_GRACE

    coord.shared_state.macro_cycle = 0
    budget = coord._specialist_wall_budget_sec(needs_gpu=True)  # 3600
    ttl = int(budget * (1.0 + GPU_LEASE_TTL_GRACE))
    assert ttl == int(3600 * 1.1)
    assert ttl >= budget  # kill ≤ lease TTL


def test_run_dispatched_releases_gpu_lease_on_success(coord: Coordinator) -> None:
    import asyncio

    released: list[object] = []

    class _Task:
        task_id = "tg-ok"

    async def _fake_run_task(task, *, prebound_lease=None, extra_context=None):
        return "RESULT"

    async def _fake_release(lease):
        released.append(lease)

    coord.sub.run_task = _fake_run_task
    coord.gpu_specialist_pool.release = _fake_release
    sentinel_lease = object()
    out = asyncio.run(
        coord._run_dispatched_with_gpu_release(
            _Task(), prebound_lease=None, extra_context={}, gpu_lease=sentinel_lease,
        )
    )
    assert out == "RESULT"
    assert released == [sentinel_lease]


def test_run_dispatched_releases_gpu_lease_on_exception(coord: Coordinator) -> None:
    import asyncio

    released: list[object] = []

    class _Task:
        task_id = "tg-boom"

    async def _boom(task, *, prebound_lease=None, extra_context=None):
        raise RuntimeError("subprocess crashed")

    async def _fake_release(lease):
        released.append(lease)

    coord.sub.run_task = _boom
    coord.gpu_specialist_pool.release = _fake_release
    sentinel_lease = object()
    with pytest.raises(RuntimeError, match="subprocess crashed"):
        asyncio.run(
            coord._run_dispatched_with_gpu_release(
                _Task(), prebound_lease=None, extra_context={}, gpu_lease=sentinel_lease,
            )
        )
    # C1: lease released via finally even though run_task raised.
    assert released == [sentinel_lease]


def test_run_dispatched_no_gpu_lease_is_noop(coord: Coordinator) -> None:
    import asyncio

    called: list[object] = []

    class _Task:
        task_id = "tc-cpu"

    async def _fake_run_task(task, *, prebound_lease=None, extra_context=None):
        return "CPU"

    async def _fake_release(lease):
        called.append(lease)

    coord.sub.run_task = _fake_run_task
    coord.gpu_specialist_pool.release = _fake_release
    out = asyncio.run(
        coord._run_dispatched_with_gpu_release(
            _Task(), prebound_lease=None, extra_context={}, gpu_lease=None,
        )
    )
    assert out == "CPU"
    assert called == []  # no GPU lease → release never called


# -- static / pure helpers -------------------------------------------------
def test_gap_layer_for_action(coord: Coordinator) -> None:
    assert coord._gap_layer_for_action("kernel_opt") == ("kernel_agent", "kernel_switch_specialist")
    assert coord._gap_layer_for_action("profile") == ("kernel_agent", "kernel_switch_specialist")
    assert coord._gap_layer_for_action("sweep") == ("framework", "serving_specialist")
    assert coord._gap_layer_for_action("baseline") == ("system", "system_specialist")
    assert coord._gap_layer_for_action("anything-else") == ("framework", "serving_specialist")


def test_task_id_from_specialist_source(coord: Coordinator) -> None:
    from hyperloom.orchestrator.loop.coordinator import SPECIALIST_FROM_AGENT_PREFIX

    assert coord._task_id_from_specialist_source("") == ""
    assert coord._task_id_from_specialist_source("kernel_agent") == ""
    assert (
        coord._task_id_from_specialist_source(
            f"{SPECIALIST_FROM_AGENT_PREFIX}abc",
        )
        == "abc"
    )


def test_pr_summary_to_dict(coord: Coordinator) -> None:
    class _PR:
        repo = "owner/name"
        number = 42
        title = "fix moe"
        url = "https://github.com/owner/name/pull/42"
        state = "open"
        labels = ["perf", "moe"]
        author = "alice"

    out = coord._pr_summary_to_dict(_PR())
    assert out["repo"] == "owner/name"
    assert out["number"] == 42
    assert out["labels"] == ["perf", "moe"]
    assert out["author"] == "alice"


def test_lanes_fit(coord: Coordinator) -> None:
    assert coord._lanes_fit(["gpu"], {"gpu": 0}, {"gpu": 1}) is True
    assert coord._lanes_fit(["gpu"], {"gpu": 1}, {"gpu": 1}) is False
    assert coord._lanes_fit(["gpu"], {}, {"gpu": 0}) is False


def test_pitfall_severity_for(coord: Coordinator) -> None:
    assert coord._pitfall_severity_for(None) is None
    assert coord._pitfall_severity_for({"error_class": "oom"}) is not None
    assert coord._pitfall_severity_for({"status": "crash"}) is not None
    assert coord._pitfall_severity_for({"gain_pct": -10.0}) is not None
    assert coord._pitfall_severity_for({"gain_pct": 2.0}) is None
    assert coord._pitfall_severity_for({"gain_pct": "bad"}) is None


def test_is_promotable_result(coord: Coordinator) -> None:
    assert coord._is_promotable_result("baseline", "not-a-dict") is False
    assert coord._is_promotable_result("sweep", {"status": "succeeded"}) is True
    assert coord._is_promotable_result("sweep", {"status": "failed"}) is False
    assert coord._is_promotable_result("replay_warm_recipe", {"status": "failed"}) is True
    assert coord._is_promotable_result("explore", {"status": "ok"}) is True
    assert coord._is_promotable_result("explore", {"status": "failed"}) is False


# -- phase / id helpers ----------------------------------------------------
def test_journal_entry_phase(coord: Coordinator) -> None:
    coord.shared_state.phase = ""
    assert coord._journal_entry_phase() == "UNKNOWN"
    coord.shared_state.phase = "explore"
    assert coord._journal_entry_phase() == "EXPLORE"


def test_source_session_id_prefers_cortex(coord: Coordinator) -> None:
    coord.shared_state.cortex_session_id = "cortex-99"
    assert coord._source_session_id() == "cortex-99"
    coord.shared_state.cortex_session_id = ""
    assert coord._source_session_id() == coord.session_dir.name


def test_kernel_and_explore_enabled(coord: Coordinator) -> None:
    coord.shared_state.kernel_enabled = True
    assert coord._kernel_enabled() == ("kernel_agent" in coord.role_registry)
    coord.shared_state.kernel_enabled = False
    assert coord._kernel_enabled() is False


def test_internal_analysis_kind(coord: Coordinator) -> None:
    coord.shared_state.enable_roofline = True
    assert coord._internal_analysis_kind() == "roofline"
    coord.shared_state.enable_roofline = False
    assert coord._internal_analysis_kind() == "profile"


# -- watermark / tput projection ------------------------------------------
def test_current_tput_from_validated_gain(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 0.0
    assert coord._current_tput_from_validated_gain() == 0.0
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 10.0
    assert coord._current_tput_from_validated_gain() == pytest.approx(110.0)


def test_needs_roofline_for_watermark_guards(coord: Coordinator) -> None:
    ss = coord.shared_state
    # pending roofline -> never re-arm
    ss.auto_roofline_pending_task_id = "task-1"
    assert coord._needs_roofline_for_watermark() is False
    # no last roofline, no failure streak -> bootstrap guard
    ss.auto_roofline_pending_task_id = ""
    ss.last_roofline_tput = 0.0
    ss.roofline_failure_streak = 0
    assert coord._needs_roofline_for_watermark() is False
    # crossing the watermark over last roofline
    ss.last_roofline_tput = 100.0
    ss.baseline_tput = 100.0
    ss.cumulative_gain_validated = 50.0  # cur=150 -> 1.5x >= 1.1
    assert coord._needs_roofline_for_watermark() is True


# -- gap extraction --------------------------------------------------------
def test_extract_gaps_from_baseline_empty(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 0.0
    assert coord._extract_gaps_from_baseline() == []


def test_extract_gaps_from_baseline_populated(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.baseline_tput = 100.0
    ss.target_gap_pct = 12.0
    ss.baseline_failure_streak = 2
    gaps = coord._extract_gaps_from_baseline()
    ids = {g["canonical_id"].split("#")[-1] for g in gaps}
    assert "throughput_below_target" in ids
    assert "baseline_unstable" in ids
    sev = {g["canonical_id"].split("#")[-1]: g["severity"] for g in gaps}
    assert sev["throughput_below_target"] == "high"  # 12% >= 10
    assert sev["baseline_unstable"] == "high"  # streak >= 2


def test_extract_gaps_from_attempts(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.baseline_tput = 100.0
    ss.last_action_failures = [
        {"action": "kernel_opt", "error_class": "oom", "variant_name": "v1"},
        {"action": "kernel_opt", "error_class": "oom", "variant_name": "v2"},
    ]
    ss.params_no_promote_streak = 6
    ss.explore_search = {"winners_history": []}
    gaps = coord._extract_gaps_from_attempts()
    cids = {g["canonical_id"] for g in gaps}
    # recurring failure folded into one gap with two attempts
    fail_gap = [g for g in gaps if "fail:kernel_opt:oom" in g["canonical_id"]][0]
    assert len(fail_gap["attempts"]) == 2
    # explore plateau gap fired at streak >= 6 -> high severity
    plateau = [g for g in gaps if g["canonical_id"].endswith("explore_plateau")][0]
    assert plateau["severity"] == "high"
    assert cids  # non-empty


# -- advisory blocks (empty-guard paths) ----------------------------------
def test_advisory_blocks_empty_by_default(coord: Coordinator) -> None:
    # no plateau / no competitor target / no proposals -> empty strings
    assert coord._plateau_advisory_block() == ""
    assert coord._target_gap_advisory_block() == ""
    assert coord._current_primary_gap() is None
    assert coord._priors_match_advisory_block() == ""
    assert coord._recent_proposed_variants() == []


def test_recent_proposed_variants_dedup(coord: Coordinator) -> None:
    coord.shared_state.specialist_rounds = [
        {"proposal_set": [{"name": "a"}, {"name": "b"}]},
        {"proposal_set": [{"name": "b"}, {"name": "c"}, "not-a-dict"]},
    ]
    out = coord._recent_proposed_variants()
    names = {v["name"] for v in out}
    assert names == {"a", "b", "c"}


# -- warm recipe + workload tags ------------------------------------------
def test_warm_recipe_proven_items(coord: Coordinator) -> None:
    coord.shared_state.warm_start_recipe = {}
    assert coord._warm_recipe_proven_items() == []
    coord.shared_state.warm_start_recipe = {
        "recipe": {
            "attrs": {
                "what_worked": [
                    {"name": "fp8", "source": "kb"},
                    {"name": "", "source": "skip"},
                    "not-a-dict",
                ]
            }
        },
    }
    out = coord._warm_recipe_proven_items()
    assert out == [{"name": "fp8", "source": "kb"}]


def test_collect_workload_tags(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("EP", raising=False)
    monkeypatch.delenv("PP", raising=False)
    ss = coord.shared_state
    ss.framework = "sglang"
    ss.model_class = "moe"
    ss.model_name = "Qwen3-32B"
    ss.precision = "fp8"
    ss.tp = 8
    ss.conc = 64
    tags = coord._collect_workload_tags()
    assert tags["framework"] == "sglang"
    assert tags["model_class"] == "moe"
    assert tags["tp"] == 8
    assert tags["conc"] == 64
    assert tags["precision"] == "fp8"


def test_build_kernel_optimizations_from_state(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.kernel_opt_attempts = {
        "k1": {
            "last_decision": "KEEP",
            "last_micro_speedup": 1.3,
            "last_source_file": "a.py",
            "last_artifact_path": "a.so",
        },
        "k2": {"last_decision": "REVERT", "last_micro_speedup": 1.1},
    }
    ss.kernel_integrate_attempts = {
        "i1": {"kernel_id": "k1", "last_decision": "KEEP", "best_gain_pct": 5.0, "attempts": [{"new_tput": 210.0}]},
    }
    out = coord._build_kernel_optimizations_from_state()
    assert len(out) == 1  # only the KEEP'd k1
    row = out[0]
    assert row["kernel_id"] == "k1"
    assert row["integrated"] is True
    assert row["e2e_gain_pct"] == 5.0
    assert row["e2e_tput"] == 210.0


def test_derive_close_stop_reason_default(coord: Coordinator) -> None:
    coord.shared_state.phase_history = []
    assert coord._derive_close_stop_reason() == "time_exhausted"


# -- sequence denial gates -------------------------------------------------
def test_sequence_denial_for_action(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.stop_reason = ""
    ss.baseline_tput = 0.0
    # non-sequence action -> never denied
    assert coord._sequence_denial_for_action("frobnicate") is None
    # baseline itself allowed pre-baseline
    assert coord._sequence_denial_for_action("baseline") is None
    # explore denied until baseline measured
    denied = coord._sequence_denial_for_action("explore")
    assert denied is not None and denied.rule == "execution_order"
    # once baseline measured -> allowed
    ss.baseline_tput = 100.0
    assert coord._sequence_denial_for_action("explore") is None


def test_sequence_denial_for_request(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.stop_reason = ""
    ss.baseline_tput = 0.0
    # non-kernel target -> not gated
    assert coord._sequence_denial_for_request("orchestration", "anything") is None
    # trace_analyze always allowed
    assert coord._sequence_denial_for_request("kernel_agent", "trace_analyze") is None
    # unknown handler kind -> not gated
    assert coord._sequence_denial_for_request("kernel_agent", "no_such_kind") is None


def test_skip_gemm_tuning_env(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", raising=False)
    assert coord._skip_gemm_tuning() is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "yes")
    assert coord._skip_gemm_tuning() is True


def test_gemm_tuning_required_before_kernel_opt(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", raising=False)
    monkeypatch.delenv("GEMM_TUNING_BACKEND", raising=False)  # default: forge
    ss = coord.shared_state
    ss.last_gemm_tuning = {}
    # forge backend: any precision on a supported framework is eligible — bf16/
    # fp16 dense must NOT be pre-filtered out (real e2e KEEPs include them).
    ss.framework = "sglang"
    ss.precision = "fp16"
    assert coord._gemm_tuning_required_before_kernel_opt() is True
    ss.precision = "bf16"
    assert coord._gemm_tuning_required_before_kernel_opt() is True
    # Unsupported framework -> not eligible.
    ss.framework = "trt-llm"
    assert coord._gemm_tuning_required_before_kernel_opt() is False
    # Supported framework + terminal status -> not required (already done).
    ss.framework = "sglang"
    ss.precision = "fp8"
    ss.last_gemm_tuning = {"status": "succeeded"}
    assert coord._gemm_tuning_required_before_kernel_opt() is False


def test_should_continue_kernel_after_gemm(coord: Coordinator) -> None:
    coord.shared_state.continue_kernel_after_gemm = False
    assert coord._should_continue_kernel_after_gemm() is False


# -- canonical id helpers --------------------------------------------------
def test_workload_canonical_id_and_anchor(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.model_name = "Qwen3-32B"
    ss.gpu_type = "mi300x"
    ss.framework = "sglang"
    ss.precision = "fp8"
    cid = coord._workload_canonical_id()
    assert cid.startswith("inference:")
    assert "mi300x" in cid
    # anchor delegates to the same derivation
    assert coord._gap_anchor_canonical_id() == cid


def test_resolve_issue_canonical_priority(coord: Coordinator) -> None:
    from hyperloom.orchestrator.loop.coordinator import PendingProposal

    pending = PendingProposal(
        proposal_msg_id="m1",
        from_agent="orchestration",
        action_name="explore",
        predicted_gain_pct=0.0,
        payload={"gap_canonical_id": "explicit-top"},
    )
    assert coord._resolve_issue_canonical(pending) == "explicit-top"
    # falls back to params then anchor
    pending2 = PendingProposal(
        proposal_msg_id="m2",
        from_agent="orchestration",
        action_name="explore",
        predicted_gain_pct=0.0,
        payload={"params": {"gap_canonical_id": "from-params"}},
    )
    assert coord._resolve_issue_canonical(pending2) == "from-params"


def test_target_analysis_baseline_exists(coord: Coordinator) -> None:
    # conftest seeds a target analysis marker; the json may or may not exist,
    # but the call must return a bool without raising.
    assert isinstance(coord._target_analysis_baseline_exists(), bool)


def test_kernel_opt_keep_pending(coord: Coordinator) -> None:
    # delegates to SharedState; with no pending keeps -> empty string
    assert coord._kernel_opt_keep_pending() == ""


# -- framework candidate selection -------------------------------------
def test_select_next_framework_agent_candidate(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.framework_agent_batches = []
    assert coord._select_next_framework_agent_candidate() is None
    ss.framework_agent_batches = [
        {
            "candidates": [
                {"candidate_id": "c1"},
                {"candidate_id": "c2"},
            ],
        }
    ]
    ss.framework_agent_phase_progress = [{"candidate_id": "c1"}]
    nxt = coord._select_next_framework_agent_candidate()
    assert nxt == {"candidate_id": "c2"}


def test_unprocessed_framework_agent_candidates(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.framework_agent_batches = [
        {
            "candidates": [
                {"candidate_id": "c1"},
                {"candidate_id": "c2"},
                {"candidate_id": "c3"},
            ],
        }
    ]
    ss.framework_agent_phase_progress = [{"candidate_id": "c1"}]
    out = coord._unprocessed_framework_agent_candidates()
    assert [c["candidate_id"] for c in out] == ["c2", "c3"]


def test_match_framework_agent_candidate_by_id_and_pr_number(coord: Coordinator) -> None:
    cands = [
        {"candidate_id": "https://github.com/o/r/pull/12", "pr_url": "https://github.com/o/r/pull/12", "pr_number": 12},
        {"candidate_id": "https://github.com/o/r/pull/34", "ref": "PR:34", "pr_number": 34},
    ]
    # exact candidate_id
    assert coord._match_framework_agent_candidate("https://github.com/o/r/pull/12", cands)["pr_number"] == 12
    # ref match
    assert coord._match_framework_agent_candidate("PR:34", cands)["pr_number"] == 34
    # bare PR number fallback
    assert coord._match_framework_agent_candidate("34", cands)["pr_number"] == 34
    # unknown
    assert coord._match_framework_agent_candidate("999", cands) is None
    assert coord._match_framework_agent_candidate("", cands) is None


async def test_select_best_framework_agent_candidate_falls_back_to_linear(
    coord: Coordinator, monkeypatch
) -> None:
    """With the ranker client unavailable, selection degrades to discovery order."""
    ss = coord.shared_state
    ss.framework_agent_batches = [
        {"candidates": [{"candidate_id": "c1"}, {"candidate_id": "c2"}]}
    ]
    ss.framework_agent_phase_progress = []
    # Force the ranker client to be unavailable.
    monkeypatch.setattr(coord.phase_framework, "_framework_agent_ranker_client", lambda: None)
    chosen = await coord._select_best_framework_agent_candidate()
    assert chosen == {"candidate_id": "c1"}


async def test_select_best_framework_agent_candidate_single(coord: Coordinator) -> None:
    """A single unprocessed candidate is returned without invoking the ranker."""
    ss = coord.shared_state
    ss.framework_agent_batches = [{"candidates": [{"candidate_id": "only"}]}]
    ss.framework_agent_phase_progress = []
    chosen = await coord._select_best_framework_agent_candidate()
    assert chosen == {"candidate_id": "only"}


async def test_select_best_framework_agent_candidate_uses_ranker_choice(
    coord: Coordinator, monkeypatch
) -> None:
    """When the ranker returns a candidate, it is used over discovery order."""
    ss = coord.shared_state
    ss.framework_agent_batches = [
        {"candidates": [{"candidate_id": "c1"}, {"candidate_id": "c2"}, {"candidate_id": "c3"}]}
    ]
    ss.framework_agent_phase_progress = []

    async def _fake_rank(cands):
        return cands[-1]  # pick the last → c3

    monkeypatch.setattr(coord.phase_framework, "_rank_framework_agent_candidates_llm", _fake_rank)
    chosen = await coord._select_best_framework_agent_candidate()
    assert chosen == {"candidate_id": "c3"}


def test_framework_known_candidate_ids(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.framework_agent_batches = [
        {"candidates": [{"candidate_id": "c1"}, {"pr_url": "u2"}]},
    ]
    ss.research_scout_seen_pr_ids = ["p3"]
    ids = coord._framework_known_candidate_ids()
    assert {"c1", "u2", "p3"}.issubset(ids)
    # tried refs reflects the same set
    assert set(coord._framework_tried_refs()) == ids


# -- module-level helpers --------------------------------------------------
def test_first_present() -> None:
    from hyperloom.orchestrator.loop.coordinator import _first_present

    assert _first_present({"a": 1, "b": 2}, ("x", "b", "a")) == 2
    assert _first_present({"a": None, "b": 5}, ("a", "b")) == 5
    assert _first_present({}, ("a",)) is None
    assert _first_present("not-a-dict", ("a",)) is None


def test_lifecycle_paths() -> None:
    from hyperloom.orchestrator.loop.coordinator import _lifecycle_paths

    assert _lifecycle_paths("not-a-dict") == {}
    out = _lifecycle_paths({"patch_path": "/a/p.diff", "workspace": "", "other": "x"})
    assert out == {"patch_path": "/a/p.diff"}


def test_format_inbox_event_variants() -> None:
    from hyperloom.orchestrator.loop.coordinator import _format_inbox_event
    from hyperloom.orchestrator.bus.message_bus import Message

    delegated = Message.new(
        "kernel_agent",
        "orchestration",
        "delegated_result",
        {
            "kind": "explore",
            "state": "succeeded",
            "result": {"status": "kept", "gain_pct": 5.0, "tput": 200.0, "kept": True},
        },
    )
    line = _format_inbox_event(delegated)
    assert "topic=delegated_result" in line
    assert "status=" in line and "kept=" in line

    verdict = Message.new(
        "critic",
        "orchestration",
        "review_verdict",
        {"target_proposal_msg_id": "m1", "verdict": "approve", "reasoning": "ok"},
    )
    assert "verdict='approve'" in _format_inbox_event(verdict)

    obs = Message.new(
        "coordinator",
        "orchestration",
        "observation",
        {"kind": "policy_denied"},
    )
    assert "kind='policy_denied'" in _format_inbox_event(obs)

    plain = Message.new("a", "b", "misc", {"x": 1})
    assert "payload=" in _format_inbox_event(plain)
