# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for Coordinator pure/sync helper methods.

Builds one Coordinator with mock backends and exercises the formatting / gap /
fact / tag helpers directly (both empty-guard and populated paths), avoiding the
async event loop."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
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


# -- static / pure helpers -------------------------------------------------
def test_gap_layer_for_action(coord: Coordinator) -> None:
    assert coord._gap_layer_for_action("kernel_opt") == ("kernel", "kernel_switch_specialist")
    assert coord._gap_layer_for_action("profile") == ("kernel", "kernel_switch_specialist")
    assert coord._gap_layer_for_action("sweep") == ("framework", "serving_specialist")
    assert coord._gap_layer_for_action("baseline") == ("system", "system_specialist")
    assert coord._gap_layer_for_action("anything-else") == ("framework", "serving_specialist")


def test_task_id_from_specialist_source(coord: Coordinator) -> None:
    from inference_optimizer.orchestrator.coordinator import SPECIALIST_FROM_AGENT_PREFIX
    assert coord._task_id_from_specialist_source("") == ""
    assert coord._task_id_from_specialist_source("kernel") == ""
    assert coord._task_id_from_specialist_source(
        f"{SPECIALIST_FROM_AGENT_PREFIX}abc",
    ) == "abc"


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
    assert coord._kernel_enabled() == ("kernel" in coord.role_registry)
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
        "recipe": {"attrs": {"what_worked": [
            {"name": "fp8", "source": "kb"},
            {"name": "", "source": "skip"},
            "not-a-dict",
        ]}},
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
        "k1": {"last_decision": "KEEP", "last_micro_speedup": 1.3,
               "last_source_file": "a.py", "last_artifact_path": "a.so"},
        "k2": {"last_decision": "REVERT", "last_micro_speedup": 1.1},
    }
    ss.kernel_integrate_attempts = {
        "i1": {"kernel_id": "k1", "last_decision": "KEEP", "best_gain_pct": 5.0,
               "attempts": [{"new_tput": 210.0}]},
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
    assert coord._sequence_denial_for_request("kernel", "trace_analyze") is None
    # unknown handler kind -> not gated
    assert coord._sequence_denial_for_request("kernel", "no_such_kind") is None


def test_skip_gemm_tuning_env(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", raising=False)
    assert coord._skip_gemm_tuning() is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "yes")
    assert coord._skip_gemm_tuning() is True


def test_gemm_tuning_required_before_kernel_opt(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", raising=False)
    ss = coord.shared_state
    # non fp8/sglang -> not required
    ss.precision = "fp16"
    ss.framework = "sglang"
    assert coord._gemm_tuning_required_before_kernel_opt() is False
    # fp8 + sglang + no terminal status -> required
    ss.precision = "fp8"
    ss.last_gemm_tuning = {}
    assert coord._gemm_tuning_required_before_kernel_opt() is True
    # terminal status -> not required
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
    from inference_optimizer.orchestrator.coordinator import PendingProposal
    pending = PendingProposal(
        proposal_msg_id="m1", from_agent="orchestration",
        action_name="explore", predicted_gain_pct=0.0,
        payload={"gap_canonical_id": "explicit-top"},
    )
    assert coord._resolve_issue_canonical(pending) == "explicit-top"
    # falls back to params then anchor
    pending2 = PendingProposal(
        proposal_msg_id="m2", from_agent="orchestration",
        action_name="explore", predicted_gain_pct=0.0,
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


# -- framework_pr candidate selection -------------------------------------
def test_select_next_framework_pr_candidate(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.framework_pr_batches = []
    assert coord._select_next_framework_pr_candidate() is None
    ss.framework_pr_batches = [{
        "candidates": [
            {"candidate_id": "c1"}, {"candidate_id": "c2"},
        ],
    }]
    ss.framework_pr_phase_progress = [{"candidate_id": "c1"}]
    nxt = coord._select_next_framework_pr_candidate()
    assert nxt == {"candidate_id": "c2"}


def test_framework_pr_known_candidate_ids(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.framework_pr_batches = [
        {"candidates": [{"candidate_id": "c1"}, {"pr_url": "u2"}]},
    ]
    ss.research_scout_seen_pr_ids = ["p3"]
    ids = coord._framework_pr_known_candidate_ids()
    assert {"c1", "u2", "p3"}.issubset(ids)
    # tried refs reflects the same set
    assert set(coord._framework_pr_tried_refs()) == ids


# -- module-level helpers --------------------------------------------------
def test_first_present() -> None:
    from inference_optimizer.orchestrator.coordinator import _first_present
    assert _first_present({"a": 1, "b": 2}, ("x", "b", "a")) == 2
    assert _first_present({"a": None, "b": 5}, ("a", "b")) == 5
    assert _first_present({}, ("a",)) is None
    assert _first_present("not-a-dict", ("a",)) is None


def test_lifecycle_paths() -> None:
    from inference_optimizer.orchestrator.coordinator import _lifecycle_paths
    assert _lifecycle_paths("not-a-dict") == {}
    out = _lifecycle_paths({"patch_path": "/a/p.diff", "workspace": "", "other": "x"})
    assert out == {"patch_path": "/a/p.diff"}


def test_format_inbox_event_variants() -> None:
    from inference_optimizer.orchestrator.coordinator import _format_inbox_event
    from inference_optimizer.orchestrator.message_bus import Message

    delegated = Message.new(
        "kernel", "orchestration", "delegated_result",
        {"kind": "explore", "state": "succeeded",
         "result": {"status": "kept", "gain_pct": 5.0, "tput": 200.0, "kept": True}},
    )
    line = _format_inbox_event(delegated)
    assert "topic=delegated_result" in line
    assert "status=" in line and "kept=" in line

    verdict = Message.new(
        "critic", "orchestration", "review_verdict",
        {"target_proposal_msg_id": "m1", "verdict": "approve", "reasoning": "ok"},
    )
    assert "verdict='approve'" in _format_inbox_event(verdict)

    obs = Message.new(
        "coordinator", "orchestration", "observation", {"kind": "policy_denied"},
    )
    assert "kind='policy_denied'" in _format_inbox_event(obs)

    plain = Message.new("a", "b", "misc", {"x": 1})
    assert "payload=" in _format_inbox_event(plain)
