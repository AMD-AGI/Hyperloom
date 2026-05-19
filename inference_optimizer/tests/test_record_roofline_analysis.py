"""Roofline-v2 C2: ``record_roofline_analysis`` schema contract tests.

These tests pin the cleaning / coercion contract every downstream
consumer relies on:

* The ``roofline`` action executor (C4) can hand in either a clean
  decision dict or a partially-malformed analyzer LLM response — the
  recorder must normalize to the documented schema either way so the
  prompt renderer (C5) never needs ``None``-guards or ``isinstance``
  branches when rendering the conclusion section.
* All schema keys are always present after a successful record.
* ``raw_llm_response`` is capped at 8 KB; the structured fields are
  the canonical signal.
* Empty / non-dict input degrades to ``last_roofline_analysis == {}``
  (the documented "not yet run / malformed" state).
"""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


# All schema keys the recorder is contractually required to populate.
_REQUIRED_KEYS = {
    "snapshot_id",
    "analyzed_at_iso",
    "analyzed_at_gain_pct",
    "based_on_analysis_md",
    "primary_bottleneck",
    "bottleneck_distribution",
    "suggested_prunes",
    "suggested_next_actions",
    "reprofile_recommended",
    "reprofile_reason",
    "raw_llm_response",
}


def _well_formed_result() -> dict:
    return {
        "snapshot_id": 2,
        "analyzed_at_iso": "2026-05-19T10:30:00+00:00",
        "analyzed_at_gain_pct": 3.2,
        "based_on_analysis_md": "/sessions/abc/select_kernels/analysis.md",
        "primary_bottleneck": "comm",
        "bottleneck_distribution": {
            "comm": 0.45, "compute": 0.30, "memory": 0.15, "idle": 0.10,
        },
        "suggested_prunes": [
            {"family": "kernel_opt",
             "reason": "compute saturated 92%, no reusable_native in top-5",
             "confidence": "high"},
            {"family": "deep_kernel_analysis",
             "reason": "comm >40% — kernel-level analysis won't help",
             "confidence": "medium"},
        ],
        "suggested_next_actions": [
            {"kind": "params",
             "rationale": "try comm-overlap flags",
             "priority": "high"},
            {"kind": "comm_optimization",
             "rationale": "rccl Allreduce is top kernel",
             "priority": "high"},
            {"kind": "backends",
             "rationale": "try moe_a2a_backend=deepep",
             "priority": "medium"},
        ],
        "reprofile_recommended": False,
        "reprofile_reason": "no gain delta to compare yet",
        "raw_llm_response": "Roofline analysis: comm-dominant...",
    }


def test_well_formed_result_round_trips():
    """Happy path: clean dict in → all fields preserved verbatim."""
    state = SharedState()
    state.record_roofline_analysis(_well_formed_result())

    cached = state.last_roofline_analysis
    assert set(cached.keys()) == _REQUIRED_KEYS
    assert cached["snapshot_id"] == 2
    assert cached["primary_bottleneck"] == "comm"
    assert cached["bottleneck_distribution"]["comm"] == 0.45
    assert len(cached["suggested_prunes"]) == 2
    assert cached["suggested_prunes"][0]["family"] == "kernel_opt"
    assert cached["suggested_prunes"][0]["confidence"] == "high"
    assert len(cached["suggested_next_actions"]) == 3
    assert cached["suggested_next_actions"][1]["kind"] == "comm_optimization"
    assert cached["reprofile_recommended"] is False
    assert cached["reprofile_reason"] == "no gain delta to compare yet"
    assert cached["raw_llm_response"].startswith("Roofline analysis")


def test_non_dict_result_is_ignored():
    """``None`` / non-dict short-circuits leaving cache untouched."""
    state = SharedState()
    assert state.last_roofline_analysis == {}
    state.record_roofline_analysis(None)  # type: ignore[arg-type]
    assert state.last_roofline_analysis == {}
    state.record_roofline_analysis("garbage")  # type: ignore[arg-type]
    assert state.last_roofline_analysis == {}


def test_empty_dict_yields_default_schema():
    """Empty dict → every required key present with safe default."""
    state = SharedState()
    state.record_roofline_analysis({})
    cached = state.last_roofline_analysis
    assert set(cached.keys()) == _REQUIRED_KEYS
    assert cached["snapshot_id"] == 0
    assert cached["analyzed_at_iso"]  # auto-filled via _now_iso
    assert cached["analyzed_at_gain_pct"] == 0.0
    assert cached["based_on_analysis_md"] == ""
    assert cached["primary_bottleneck"] == "unknown"
    assert cached["bottleneck_distribution"] == {}
    assert cached["suggested_prunes"] == []
    assert cached["suggested_next_actions"] == []
    assert cached["reprofile_recommended"] is False
    assert cached["reprofile_reason"] == ""
    assert cached["raw_llm_response"] == ""


def test_malformed_distribution_filtered():
    """Non-numeric distribution values silently dropped."""
    state = SharedState()
    state.record_roofline_analysis({
        "primary_bottleneck": "comm",
        "bottleneck_distribution": {
            "comm": 0.5,
            "compute": "not-a-number",
            "memory": None,
            "idle": 0.2,
        },
    })
    dist = state.last_roofline_analysis["bottleneck_distribution"]
    assert dist == {"comm": 0.5, "idle": 0.2}


def test_malformed_distribution_non_dict_yields_empty():
    """Distribution must be a dict; list/None/str → empty distribution."""
    state = SharedState()
    state.record_roofline_analysis({
        "primary_bottleneck": "comm",
        "bottleneck_distribution": [("comm", 0.5)],  # wrong type
    })
    assert state.last_roofline_analysis["bottleneck_distribution"] == {}


def test_advice_entries_missing_anchor_key_dropped():
    """Prune entry without ``family`` (anchor key) dropped silently.

    Next-action entry without ``kind`` likewise dropped. This pins the
    contract so the prompt renderer (C5) never has to render an entry
    with an empty action target.
    """
    state = SharedState()
    state.record_roofline_analysis({
        "suggested_prunes": [
            {"family": "kernel_opt", "reason": "x", "confidence": "high"},
            {"reason": "no family", "confidence": "high"},  # dropped
            "not-a-dict",                                   # dropped
            {"family": "", "reason": "empty family"},       # dropped
        ],
        "suggested_next_actions": [
            {"kind": "params", "rationale": "x", "priority": "high"},
            {"rationale": "no kind"},                       # dropped
        ],
    })
    cached = state.last_roofline_analysis
    assert len(cached["suggested_prunes"]) == 1
    assert cached["suggested_prunes"][0]["family"] == "kernel_opt"
    assert len(cached["suggested_next_actions"]) == 1
    assert cached["suggested_next_actions"][0]["kind"] == "params"


def test_advice_missing_secondary_keys_default_to_empty_string():
    """``reason`` / ``confidence`` missing → present as empty string,
    so C5's f-string renderer cannot KeyError on a malformed entry."""
    state = SharedState()
    state.record_roofline_analysis({
        "suggested_prunes": [
            {"family": "kernel_opt"},  # only anchor key present
        ],
    })
    entry = state.last_roofline_analysis["suggested_prunes"][0]
    assert entry == {"family": "kernel_opt", "reason": "", "confidence": ""}


def test_raw_llm_response_capped_at_8kb():
    """Raw response capped — structured fields are canonical, raw is
    forensic only."""
    state = SharedState()
    state.record_roofline_analysis({"raw_llm_response": "x" * 20000})
    cached = state.last_roofline_analysis["raw_llm_response"]
    assert len(cached) == 8192 + len("...[truncated]")
    assert cached.endswith("...[truncated]")


def test_reprofile_recommended_coerced_to_bool():
    """Truthy non-bool inputs become True (forgiving analyzer output)."""
    state = SharedState()
    state.record_roofline_analysis({"reprofile_recommended": "yes"})
    assert state.last_roofline_analysis["reprofile_recommended"] is True
    state.record_roofline_analysis({"reprofile_recommended": 0})
    assert state.last_roofline_analysis["reprofile_recommended"] is False


def test_snapshot_id_and_gain_coerce_safely():
    """String numerics coerce; truly malformed → 0 / 0.0 (not raised)."""
    state = SharedState()
    state.record_roofline_analysis({
        "snapshot_id": "3",
        "analyzed_at_gain_pct": "4.7",
    })
    cached = state.last_roofline_analysis
    assert cached["snapshot_id"] == 3
    assert cached["analyzed_at_gain_pct"] == 4.7

    state.record_roofline_analysis({
        "snapshot_id": "not-a-number",
        "analyzed_at_gain_pct": None,
    })
    cached = state.last_roofline_analysis
    assert cached["snapshot_id"] == 0
    assert cached["analyzed_at_gain_pct"] == 0.0


def test_recording_replaces_previous_cache():
    """``last_roofline_analysis`` is single-snapshot, not append."""
    state = SharedState()
    state.record_roofline_analysis(_well_formed_result())
    assert state.last_roofline_analysis["snapshot_id"] == 2

    state.record_roofline_analysis({"snapshot_id": 7})
    cached = state.last_roofline_analysis
    assert cached["snapshot_id"] == 7
    assert cached["primary_bottleneck"] == "unknown"  # not carried over
    assert cached["suggested_prunes"] == []           # not carried over


def test_default_attribute_starts_empty():
    """Fresh SharedState has the field present but empty."""
    state = SharedState()
    assert state.last_roofline_analysis == {}
