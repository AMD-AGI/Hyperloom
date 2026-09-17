# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``framework_agent`` event.

Two themes: the paths an arm-shaped projection could not represent (config proposals the
orchestration agent raised itself, seed-grid variants nobody proposed), and the recorded-versus-derived
line -- policy, plateau inputs and gate verdicts are facts of the moment they were read, so
re-deriving them at export returns values the phase never acted on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
    ARM_CONFIG,
    ARM_SOURCE,
    DISPOSITION_ATTEMPTED,
    DISPOSITION_DROPPED,
    DISPOSITION_PENDING,
    PLATEAU_PATH_ADVISORY,
    PLATEAU_PATH_EXIT,
    PRODUCER_ORCHESTRATION,
    PRODUCER_SEED_GRID,
    PRODUCER_SPECIALIST,
    ROLE_AUTHORING,
    ROLE_DISCOVERY,
    STEP_ATTEMPTED,
    STEP_AUTHORED,
    STEP_PROPOSED,
    STEP_REAUTHORED,
    STEP_REVIEWED,
    make_framework_recorder,
    producer_for_provenance,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "framework_agent"]


def _ext(session_dir: Path) -> dict[str, Any]:
    events = _events(session_dir)
    assert len(events) == 1, f"expected one framework event, got {len(events)}"
    return events[0]["ext"]


def _one(session_dir: Path) -> dict[str, Any]:
    events = _events(session_dir)
    assert len(events) == 1, f"expected one framework event, got {len(events)}"
    return events[0]


def test_orchestration_proposal_needs_no_run(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal(
        "p-llm-1",
        arm=ARM_CONFIG,
        producer=PRODUCER_ORCHESTRATION,
        lever_kind="llm_direct",
    )
    recorder.settle_proposal("p-llm-1", disposition=DISPOSITION_ATTEMPTED)
    recorder.finish(exit_reason="both_arms_plateaued")

    ext = _ext(_bound_session)
    assert ext["runs"] == []
    proposal = ext["proposals"][0]
    assert proposal["producer"] == PRODUCER_ORCHESTRATION
    # Absent rather than empty: there is no run to point at.
    assert "run_ref" not in proposal


def test_seed_grid_attempt_needs_no_proposal(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt(
        "a-seed-1",
        arm=ARM_CONFIG,
        provenance=PRODUCER_SEED_GRID,
        outcome="keep",
        measurement={"before_tput": 100.0, "after_tput": 112.0, "gain_pct": 12.0},
    )
    recorder.finish(exit_reason="budget")

    ext = _ext(_bound_session)
    assert ext["proposals"] == []
    attempt = ext["attempts"][0]
    assert "proposal_ref" not in attempt
    assert attempt["provenance"] == PRODUCER_SEED_GRID
    assert attempt["measurement"]["after_tput"] == 112.0


def test_authoring_run_is_both_a_step_and_a_run(_bound_session):
    recorder = make_framework_recorder(macro_cycle=1)
    recorder.record_run("r-disc-1", role=ROLE_DISCOVERY, arm=ARM_SOURCE, domain="attention", status="succeeded")
    recorder.record_proposal(
        "cand-7",
        arm=ARM_SOURCE,
        producer=PRODUCER_SPECIALIST,
        producer_ref="attention",
        run_ref="r-disc-1",
        source_ref="vllm#4821",
    )
    recorder.record_proposal_step("cand-7", step=STEP_PROPOSED)
    recorder.record_proposal_review("cand-7", verdict="approved", iteration=1)
    recorder.record_proposal_step("cand-7", step=STEP_REVIEWED, outcome="approved")
    recorder.record_run("r-auth-1", role=ROLE_AUTHORING, arm=ARM_SOURCE, status="succeeded")
    recorder.record_proposal_step("cand-7", step=STEP_AUTHORED, run_ref="r-auth-1", outcome="patch_ready")
    recorder.finish(exit_reason="both_arms_plateaued")

    ext = _ext(_bound_session)
    assert [row["run_id"] for row in ext["runs"]] == ["r-auth-1", "r-disc-1"]
    proposal = ext["proposals"][0]
    assert [step["step"] for step in proposal["lifecycle"]] == [STEP_PROPOSED, STEP_REVIEWED, STEP_AUTHORED]
    assert proposal["lifecycle"][2]["run_ref"] == "r-auth-1"
    # The authoring run produced no proposal of its own, and says so by holding an empty list.
    produced = {row["run_id"]: row["produced_ids"] for row in ext["runs"]}
    assert produced == {"r-disc-1": ["cand-7"], "r-auth-1": []}


def test_reauthoring_reads_as_repeated_steps(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal("cand-9", arm=ARM_SOURCE, producer=PRODUCER_SPECIALIST)
    recorder.record_proposal_step("cand-9", step=STEP_AUTHORED, run_ref="r-a1", outcome="apply_failed")
    recorder.record_proposal_step("cand-9", step=STEP_REAUTHORED, run_ref="r-a2", outcome="patch_ready")
    recorder.record_proposal_step("cand-9", step=STEP_ATTEMPTED, outcome="revert")
    recorder.finish(exit_reason="budget")

    lifecycle = _ext(_bound_session)["proposals"][0]["lifecycle"]
    assert [(step["step"], step["run_ref"]) for step in lifecycle] == [
        (STEP_AUTHORED, "r-a1"),
        (STEP_REAUTHORED, "r-a2"),
        (STEP_ATTEMPTED, ""),
    ]


def test_policy_is_recorded_not_scavenged(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_policy(
        keep_threshold_pct=3.0,
        variant_timeout_sec=1800,
        overtime_kill_ratio=1.5,
        config={"keep_gain_threshold_pct": 1.0, "empty_streak_threshold": 2, "lookback": 5},
        source={"no_keep_streak_threshold": 3, "authoring_enabled": True},
    )
    recorder.finish(exit_reason="budget")

    policy = _ext(_bound_session)["policy"]
    assert policy["keep_threshold_pct"] == 3.0
    assert policy[ARM_CONFIG]["empty_streak_threshold"] == 2
    assert policy[ARM_SOURCE]["authoring_enabled"] is True
    # Never resolved, and so reported as unresolved rather than as an export-time default.
    assert policy[ARM_SOURCE]["discovery_retry_limit"] is None


def test_plateau_snapshots_the_values_it_ruled_on(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_plateau(
        arm=ARM_CONFIG,
        path=PLATEAU_PATH_ADVISORY,
        triggered=False,
        inputs={"empty_streak": 1, "recent_keep_gain_pct": 4.0},
        thresholds={"empty_streak_threshold": 2},
    )
    recorder.record_attempt("a-1", arm=ARM_CONFIG, outcome="revert")
    recorder.record_plateau(
        arm=ARM_CONFIG,
        path=PLATEAU_PATH_EXIT,
        triggered=True,
        inputs={"empty_streak": 2, "recent_keep_gain_pct": 0.0},
        thresholds={"empty_streak_threshold": 2},
    )
    recorder.finish(exit_reason="both_arms_plateaued")

    plateau = _ext(_bound_session)["plateau"]
    assert [(row["path"], row["triggered"]) for row in plateau] == [
        (PLATEAU_PATH_ADVISORY, False),
        (PLATEAU_PATH_EXIT, True),
    ]
    assert plateau[0]["inputs"]["empty_streak"] == 1
    assert plateau[1]["inputs"]["empty_streak"] == 2


def test_plateau_order_survives_same_second(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    for arm in (ARM_SOURCE, ARM_CONFIG, ARM_SOURCE):
        recorder.record_plateau(arm=arm, path=PLATEAU_PATH_ADVISORY, triggered=False)
    recorder.finish(exit_reason="budget")

    assert [row["arm"] for row in _ext(_bound_session)["plateau"]] == [ARM_SOURCE, ARM_CONFIG, ARM_SOURCE]


def test_a_second_leg_does_not_overwrite_the_first(_bound_session):
    first = make_framework_recorder(macro_cycle=0)
    first.record_plateau(arm=ARM_CONFIG, path=PLATEAU_PATH_ADVISORY, triggered=False, inputs={"empty_streak": 1})
    first.record_proposal("p-1", arm=ARM_CONFIG, producer=PRODUCER_ORCHESTRATION)
    first.record_proposal_step("p-1", step=STEP_PROPOSED)
    first.record_attempt("a-1", arm=ARM_CONFIG)
    first.record_attempt_gate("a-1", "tput_valid", passed=True)

    second = make_framework_recorder(macro_cycle=0)
    second.record_plateau(arm=ARM_CONFIG, path=PLATEAU_PATH_EXIT, triggered=True, inputs={"empty_streak": 2})
    second.record_proposal_step("p-1", step=STEP_ATTEMPTED)
    second.record_attempt_gate("a-1", "keep_threshold", passed=False)
    second.finish(exit_reason="both_arms_plateaued")

    ext = _ext(_bound_session)
    assert [row["inputs"]["empty_streak"] for row in ext["plateau"]] == [1, 2]
    assert [step["step"] for step in ext["proposals"][0]["lifecycle"]] == [STEP_PROPOSED, STEP_ATTEMPTED]
    assert [gate["gate"] for gate in ext["attempts"][0]["gates"]] == ["tput_valid", "keep_threshold"]


def test_each_macro_cycle_is_its_own_event(_bound_session):
    first = make_framework_recorder(macro_cycle=0)
    first.record_proposal("p-cycle0", arm=ARM_CONFIG, producer=PRODUCER_ORCHESTRATION)
    first.record_attempt("a-cycle0", arm=ARM_CONFIG, outcome="REVERT")
    first.finish(exit_reason="optimize_no_more_leverage")

    second = make_framework_recorder(macro_cycle=1)
    second.record_proposal("p-cycle1", arm=ARM_CONFIG, producer=PRODUCER_ORCHESTRATION)
    second.record_attempt("a-cycle1", arm=ARM_CONFIG, outcome="KEEP")
    second.finish(exit_reason="optimize_budget_cap")

    events = _events(_bound_session)
    assert [event["ext"]["macro_cycle"] for event in events] == [0, 1]
    assert [[row["proposal_id"] for row in event["ext"]["proposals"]] for event in events] == [
        ["p-cycle0"],
        ["p-cycle1"],
    ]
    assert [[row["attempt_id"] for row in event["ext"]["attempts"]] for event in events] == [
        ["a-cycle0"],
        ["a-cycle1"],
    ]
    assert [event["ext"]["exit"]["reason"] for event in events] == [
        "optimize_no_more_leverage",
        "optimize_budget_cap",
    ]


def test_url_shaped_ids_survive_the_fragment_key(_bound_session):
    url = "https://github.com/vllm-project/vllm/pull/4821"
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal(url, arm=ARM_SOURCE, producer=PRODUCER_SPECIALIST, source_ref=url)
    recorder.record_proposal_step(url, step=STEP_PROPOSED)
    recorder.record_attempt(url + "#a1", arm=ARM_SOURCE, proposal_ref=url, outcome="revert")
    recorder.record_attempt_gate(url + "#a1", "keep_threshold", passed=False)
    recorder.settle_proposal(url, disposition=DISPOSITION_ATTEMPTED)
    recorder.finish(exit_reason="budget")

    ext = _ext(_bound_session)
    proposal = ext["proposals"][0]
    assert proposal["proposal_id"] == url
    assert proposal["attempt_refs"] == [url + "#a1"]
    assert [step["step"] for step in proposal["lifecycle"]] == [STEP_PROPOSED]
    assert proposal["terminal"]["disposition"] == DISPOSITION_ATTEMPTED
    assert [gate["gate"] for gate in ext["attempts"][0]["gates"]] == ["keep_threshold"]


def test_ids_differing_only_by_the_separator_stay_distinct(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    for ident in ("a:b", "a%3Ab", "a%253Ab"):
        recorder.record_proposal(ident, arm=ARM_SOURCE, producer=PRODUCER_SPECIALIST)
    recorder.finish(exit_reason="budget")

    ids = {row["proposal_id"] for row in _ext(_bound_session)["proposals"]}
    assert ids == {"a:b", "a%3Ab", "a%253Ab"}


def test_unreached_gate_writes_no_row(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt("a-1", arm=ARM_CONFIG, outcome="revert")
    recorder.record_attempt_gate("a-1", "tput_valid", passed=True, observed=95.0)
    recorder.record_attempt_gate("a-1", "keep_threshold", passed=False, observed=1.0, threshold=3.0)
    recorder.finish(exit_reason="budget")

    attempt = _ext(_bound_session)["attempts"][0]
    assert [gate["gate"] for gate in attempt["gates"]] == ["tput_valid", "keep_threshold"]
    assert attempt["blocked_by"] == "keep_threshold"


def test_re_ruling_a_gate_keeps_its_position(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt("a-1", arm=ARM_SOURCE, outcome="revert")
    recorder.record_attempt_gate("a-1", "accuracy", passed=None, reason="eval pending")
    recorder.record_attempt_gate("a-1", "keep_threshold", passed=True, observed=6.0, threshold=3.0)
    recorder.record_attempt_gate("a-1", "accuracy", passed=False, observed=0.71, threshold=0.80)
    recorder.finish(exit_reason="budget")

    attempt = _ext(_bound_session)["attempts"][0]
    assert [gate["gate"] for gate in attempt["gates"]] == ["accuracy", "keep_threshold"]
    assert attempt["gates"][0]["passed"] is False
    assert attempt["blocked_by"] == "accuracy"


def test_outright_failure_outranks_an_unresolved_gate(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt("a-1", arm=ARM_SOURCE, outcome="revert")
    recorder.record_attempt_gate("a-1", "accuracy", passed=None, reason="eval produced no score")
    recorder.record_attempt_gate("a-1", "keep_threshold", passed=False, observed=0.4, threshold=3.0)
    recorder.finish(exit_reason="budget")

    assert _ext(_bound_session)["attempts"][0]["blocked_by"] == "keep_threshold"


def test_unresolved_gate_blocks_when_nothing_else_failed(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt("a-1", arm=ARM_SOURCE, outcome="revert")
    recorder.record_attempt_gate("a-1", "accuracy", passed=None, reason="eval produced no score")
    recorder.finish(exit_reason="budget")

    assert _ext(_bound_session)["attempts"][0]["blocked_by"] == "accuracy"


def test_attempt_pins_the_pair_it_was_judged_on(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt(
        "a-1",
        arm=ARM_CONFIG,
        outcome="keep",
        decision="KEEP",
        adopted=True,
        attribution_eligible=True,
        measured_against={"throughput": 100.0, "extra_server_args": "--foo 1"},
        measurement={"before_tput": 100.0, "after_tput": 110.0, "gain_pct": 10.0},
        config_delta={"extra_server_args": "--foo 2"},
    )
    recorder.record_attempt(
        "a-2",
        arm=ARM_CONFIG,
        outcome="revert",
        adopted=False,
        attribution_eligible=False,
        measured_against={"throughput": 110.0},
        measurement={"before_tput": 110.0, "after_tput": 108.0, "gain_pct": -1.8},
    )
    recorder.finish(exit_reason="budget")

    attempts = {row["attempt_id"]: row for row in _ext(_bound_session)["attempts"]}
    assert attempts["a-1"]["measurement"]["before_tput"] == 100.0
    assert attempts["a-1"]["measured_against"]["extra_server_args"] == "--foo 1"
    assert attempts["a-1"]["config_delta"]["extra_server_args"] == "--foo 2"
    assert attempts["a-2"]["measurement"]["before_tput"] == 110.0
    assert attempts["a-1"]["adopted"] is True
    assert attempts["a-2"]["adopted"] is False


def test_keep_unstable_stays_distinct_from_revert(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt(
        "a-1",
        arm=ARM_CONFIG,
        outcome="keep_unstable",
        decision="KEEP_UNSTABLE",
        adopted=False,
        measurement={"gain_pct": 8.0},
    )
    recorder.finish(exit_reason="budget")

    attempt = _ext(_bound_session)["attempts"][0]
    assert attempt["outcome"] == "keep_unstable"
    assert attempt["decision"] == "KEEP_UNSTABLE"
    assert attempt["adopted"] is False
    assert attempt["measurement"]["gain_pct"] == 8.0


def test_attempt_refs_are_derived_from_the_attempts(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal("p-1", arm=ARM_CONFIG, producer=PRODUCER_ORCHESTRATION)
    recorder.record_attempt("a-1", arm=ARM_CONFIG, proposal_ref="p-1", outcome="revert")
    recorder.record_attempt("a-2", arm=ARM_CONFIG, proposal_ref="p-1", outcome="keep")
    recorder.settle_proposal("p-1", disposition=DISPOSITION_ATTEMPTED)
    recorder.finish(exit_reason="budget")

    assert _ext(_bound_session)["proposals"][0]["attempt_refs"] == ["a-1", "a-2"]


def test_dropped_proposal_keeps_its_reason(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal("cand-3", arm=ARM_SOURCE, producer=PRODUCER_SPECIALIST)
    recorder.record_proposal_review("cand-3", verdict="denied", reason="touches the serving loop")
    recorder.settle_proposal("cand-3", disposition=DISPOSITION_DROPPED, reason="critic_denied")
    recorder.finish(exit_reason="both_arms_plateaued")

    proposal = _ext(_bound_session)["proposals"][0]
    assert proposal["critic_review"]["verdict"] == "denied"
    assert proposal["terminal"] == {
        "disposition": DISPOSITION_DROPPED,
        "reason": "critic_denied",
        "settled_at": proposal["terminal"]["settled_at"],
    }
    assert proposal["attempt_refs"] == []


def test_entry_with_no_work_reads_as_skipped(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_policy(keep_threshold_pct=3.0)
    recorder.finish(exit_reason="no_levers_available")

    assert _one(_bound_session)["status"] == "skipped"


def test_all_runs_failing_is_not_a_success(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_run("r-1", role=ROLE_DISCOVERY, arm=ARM_SOURCE, status="failed")
    recorder.finish(exit_reason="both_arms_plateaued")

    assert _one(_bound_session)["status"] == "failed"


def test_failed_entry_records_its_own_failure(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_run("r-1", role=ROLE_DISCOVERY, arm=ARM_SOURCE, status="failed")
    recorder.finish(
        exit_reason="task_failed",
        failure={"failed_task_id": "r-1", "error_class": "WorktreeError", "error": "patch did not apply"},
    )

    event = _one(_bound_session)
    assert event["status"] == "failed"
    assert event["ext"]["failure"]["failed_task_id"] == "r-1"
    assert event["ext"]["failure"]["error_class"] == "WorktreeError"


def test_crash_closes_the_event_with_the_exception(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_attempt("a-1", arm=ARM_CONFIG, outcome="error")
    recorder.finish_crashed(RuntimeError("serving slot never freed"))

    event = _one(_bound_session)
    assert event["status"] == "failed"
    assert "serving slot never freed" in event["ext"]["failure"]["message"]


def test_killed_entry_is_recovered_as_interrupted(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal("p-1", arm=ARM_CONFIG, producer=PRODUCER_ORCHESTRATION)
    recorder.record_attempt("a-1", arm=ARM_CONFIG, proposal_ref="p-1")
    # No finish: the process died here.

    assert finalize_events(_bound_session)
    event = _one(_bound_session)
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    assert event["ext"]["proposals"][0].get("terminal") in (None, {})
    assert event["ext"]["attempts"][0]["attempt_id"] == "a-1"


def test_each_macro_cycle_gets_its_own_event(_bound_session):
    for cycle in (0, 1):
        recorder = make_framework_recorder(macro_cycle=cycle)
        recorder.record_attempt(f"a-{cycle}", arm=ARM_CONFIG, outcome="keep")
        recorder.finish(exit_reason="budget")

    events = _events(_bound_session)
    assert [event["ext"]["macro_cycle"] for event in events] == [0, 1]
    assert [event["ext"]["attempts"][0]["attempt_id"] for event in events] == ["a-0", "a-1"]


def test_pending_disposition_is_representable(_bound_session):
    recorder = make_framework_recorder(macro_cycle=0)
    recorder.record_proposal("p-1", arm=ARM_CONFIG, producer=PRODUCER_ORCHESTRATION)
    recorder.settle_proposal("p-1", disposition=DISPOSITION_PENDING, reason="budget_exhausted")
    recorder.finish(exit_reason="budget")

    assert _ext(_bound_session)["proposals"][0]["terminal"]["disposition"] == DISPOSITION_PENDING


@pytest.mark.parametrize(
    "label,producer,ref",
    [
        ("llm_direct", PRODUCER_ORCHESTRATION, ""),
        ("default_grid", PRODUCER_SEED_GRID, ""),
        ("specialist:attention", PRODUCER_SPECIALIST, "attention"),
        ("legacy:whatever", PRODUCER_ORCHESTRATION, ""),
        ("", PRODUCER_ORCHESTRATION, ""),
        (None, PRODUCER_ORCHESTRATION, ""),
    ],
)
def test_provenance_labels_map_onto_producers(label, producer, ref):
    assert producer_for_provenance(label) == (producer, ref)
