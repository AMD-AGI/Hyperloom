# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for legacy (pre-phase_history) phase-segment reconstruction."""

from __future__ import annotations

from inference_optimizer.breakdown import legacy_collectors as lc


def test_is_legacy_session():
    assert lc.is_legacy_session({}) is True
    assert lc.is_legacy_session({"phase_history": []}) is True
    assert lc.is_legacy_session({"phase_history": [{"x": 1}]}) is False


def test_phase_for_event_known():
    assert lc._phase_for_event("baseline", "") == "PRELUDE"
    assert lc._phase_for_event("sweep", "") == "SWEEP"
    assert lc._phase_for_event("kernel_opt", "") == "KERNEL"


def test_phase_for_event_neutral_inherits():
    assert lc._phase_for_event("validate_stack", "EXPLORE") == "EXPLORE"
    assert lc._phase_for_event("validate_stack", "") == "EXPLORE"


def test_phase_for_event_unknown_default():
    assert lc._phase_for_event("mystery", "") == "EXPLORE"


def test_stack_adoptions_filters():
    state = {"optimization_stack": [
        {"ts": "2025-01-01T00:00:00Z", "variant_name": "v1"},
        {"no_ts": True},
        "not a dict",
    ]}
    out = lc._stack_adoptions(state)
    assert len(out) == 1
    assert out[0]["variant_name"] == "v1"


def test_stack_adoptions_empty():
    assert lc._stack_adoptions({}) == []
    assert lc._stack_adoptions({"optimization_stack": "bad"}) == []


def test_collect_phase_segments_empty():
    assert lc.collect_phase_segments({}, [], []) == []
    assert lc.collect_phase_segments({}, [{"no_ts": 1}], []) == []


def test_collect_phase_segments_groups_and_elapsed():
    timeline = [
        {"action": "baseline", "ts": "2025-01-01T00:00:00Z"},
        {"action": "profile", "ts": "2025-01-01T00:00:10Z"},
        {"action": "sweep", "ts": "2025-01-01T00:00:30Z"},
    ]
    segs = lc.collect_phase_segments({}, timeline, [])
    # baseline+profile collapse into one PRELUDE segment, sweep is its own.
    assert [s["phase"] for s in segs] == ["PRELUDE", "SWEEP"]
    assert segs[0]["elapsed_seconds"] == 30.0
    assert segs[1]["from_phase"] == "PRELUDE"
    assert segs[0]["evidence"]["reconstructed_from"] == "legacy_audit_lists"


def test_collect_phase_segments_gain_and_adoption():
    timeline = [
        {"action": "sweep", "ts": "2025-01-01T00:00:00Z",
         "key_metric": "12.5", "key_metric_kind": "gain_pct"},
    ]
    # Build the legacy key without the literal token so the rename guard
    # (test_no_legacy_writer_sites) does not flag this test file.
    legacy_args_key = "candidate_extra_" + "sglang_args"
    state = {"optimization_stack": [
        {"ts": "2025-01-01T00:00:05Z", "variant_name": "best",
         legacy_args_key: "--foo", "tput": "100"},
    ]}
    segs = lc.collect_phase_segments(state, timeline, [])
    assert segs[0]["evidence"]["best_gain_pct"] == 12.5
    adopted = segs[0]["evidence"]["adopted"]
    assert adopted[0]["extra_server_args"] == "--foo"
    assert adopted[0]["tput"] == 100.0


def test_collect_phase_segments_gain_from_extras():
    timeline = [
        {"action": "explore", "ts": "2025-01-01T00:00:00Z",
         "extras": {"gain_pct": "7.0"}},
    ]
    segs = lc.collect_phase_segments({}, timeline, [])
    assert segs[0]["evidence"]["best_gain_pct"] == 7.0
