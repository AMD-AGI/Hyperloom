# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The direction ledger and the early stop it feeds: attribution, standings, the rule, and the audit."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop.dashboard import LiveStatus, publish_dashboard
from kernelforge.loop.direction_ledger import (
    ACTIVE,
    EXHAUSTED,
    INFEASIBLE,
    KEPT,
    OPEN,
    DirectionLedger,
    by_priority,
    direction_standings,
    keep_gains,
)
from kernelforge.loop.early_stop import EarlyStopPolicy, EarlyStopStanding, TimeWindow, early_stop_standing, time_window
from kernelforge.loop.operator_inbox import post_message
from kernelforge.loop.runner import IterationLoop
from kernelforge.orchestrator.contracts import CaseEvidence, OrchestrationContext
from kernelforge.orchestrator.direction_audit import parse_new_directions
from kernelforge.orchestrator.direction_claims import DirectionClaim, parse_direction_line


def _event(iteration: int, decision: str, best: float, direction: str = "", *, timed: bool = True) -> dict:
    """An iteration result; ``timed`` False is a REVERT_PERF whose bench failed, which carries no score."""
    event = {
        "type": "iteration_result",
        "iter": iteration,
        "decision": decision,
        "best_after_mean_case_speedup": best,
        "direction_id": direction or None,
    }
    if timed and decision in {"KEEP", "REVERT_PERF"}:
        event["mean_case_speedup"] = best
    return event


def _cost(iteration: int, hours: float) -> dict:
    return {"type": "round_cost", "iter": iteration, "total_sec": hours * 3600}


@pytest.mark.parametrize(
    ("plan", "claim"),
    [
        ("Direction: split-k\nTry split-K 4.", DirectionClaim(direction_id="split-k")),
        ("**Direction:** `split-k` — tune it", DirectionClaim(direction_id="split-k")),
        (
            "Direction: new | mainloop | Deeper LDS pipeline\n",
            DirectionClaim(new_title="Deeper LDS pipeline", new_category="mainloop"),
        ),
        ("Direction: new | mainloop", None),
        ("# Plan\nNo direction here.", None),
    ],
)
def test_a_plan_names_its_direction_on_an_opening_line(plan: str, claim) -> None:
    assert parse_direction_line(plan) == claim


def test_the_ledger_deduplicates_titles_and_keeps_ids_unique(tmp_path: Path) -> None:
    ledger = DirectionLedger(tmp_path)
    added = ledger.add(
        [{"title": "Split K"}, {"title": "split  k"}, {"title": "Split-K"}], source="survey", iteration=1
    )
    assert added == ["split-k", "split-k-2"]
    assert DirectionLedger(tmp_path).directions()[0].source == "survey"


def test_a_claim_resolves_to_an_existing_id_or_adds_the_new_direction(tmp_path: Path) -> None:
    ledger = DirectionLedger(tmp_path)
    ledger.add([{"title": "Split K"}], source="survey", iteration=1)
    assert ledger.resolve_claim(DirectionClaim(direction_id="split-k"), iteration=2) == "split-k"
    assert ledger.resolve_claim(DirectionClaim(direction_id="unknown"), iteration=2) == ""
    assert ledger.resolve_claim(DirectionClaim(new_title="Epilogue stores", new_category="epilogue"), iteration=2) == (
        "epilogue-stores"
    )
    assert ledger.resolve_claim(None, iteration=2) == ""


def test_standings_fold_from_the_events_that_name_each_direction(tmp_path: Path) -> None:
    ledger = DirectionLedger(tmp_path)
    ledger.add([{"title": t} for t in ("open", "active", "kept", "exhausted", "ruled out")], source="s", iteration=0)
    ledger.retire("ruled-out", reason="bitwise rule forbids atomics", iteration=3)
    results = [
        _event(1, "REVERT_PERF", 1.0, "active"),
        _event(2, "KEEP", 1.10, "kept"),
        _event(3, "API_ERROR", 1.10, "exhausted"),
        _event(4, "REVERT_PERF", 1.10, "exhausted"),
        _event(5, "REVERT_PERF", 1.10, "exhausted", timed=False),
        _event(6, "BUILD_FAILED", 1.10, "exhausted"),
    ]
    gains = keep_gains(results, 1.0)
    standings = {
        s.direction.id: s for s in direction_standings(ledger.directions(), results, exhaust_after=2, gains=gains)
    }
    assert {key: value.status for key, value in standings.items()} == {
        "open": OPEN,
        "active": ACTIVE,
        "kept": KEPT,
        "exhausted": EXHAUSTED,
        "ruled-out": INFEASIBLE,
    }
    # The API error and the failed bench tested nothing.
    assert standings["exhausted"].attempts == 2
    assert standings["kept"].gain == pytest.approx(0.10)


def test_the_window_reaches_back_over_hours_of_rounds_that_tested_something() -> None:
    results = [
        _event(1, "KEEP", 1.05),
        _event(2, "KEEP", 1.10),  # a queued lane of round 1: no round of its own
        _event(3, "REVERT_PERF", 1.10, timed=False),
        _event(4, "API_ERROR", 1.10),
        _event(5, "REVERT_PERF", 1.10),
    ]
    costs = [_cost(1, 2.0), _cost(3, 5.0), _cost(4, 5.0), _cost(5, 1.0)]
    gains = keep_gains(results, 1.0)
    window = time_window(results, costs, gains, 3.0)
    assert (window.hours, window.rounds) == (3.0, 2)
    assert window.gain == pytest.approx(0.10)
    assert time_window(results, costs, gains, 1.0).gain == 0.0
    assert time_window(results, costs, gains, 3.5) is None


def test_the_ledger_is_ordered_by_expected_gain_with_the_unestimated_and_the_spent_after(tmp_path: Path) -> None:
    ledger = DirectionLedger(tmp_path)
    ledger.add(
        [
            {"title": "low", "expected_gain_pct": 1},
            {"title": "unestimated"},
            {"title": "high", "expected_gain_pct": 6, "reason": "the profile shows idle CUs"},
            {"title": "spent", "expected_gain_pct": 9},
        ],
        source="survey",
        iteration=0,
    )
    results = [_event(1, "REVERT_PERF", 1.0, "spent"), _event(2, "REVERT_PERF", 1.0, "spent")]
    ordered = by_priority(direction_standings(ledger.directions(), results, exhaust_after=2))
    assert [s.direction.id for s in ordered] == ["high", "low", "unestimated", "spent"]
    assert ordered[0].to_view()["expected_gain_pct"] == 6.0
    assert ordered[0].to_view()["expected_gain_reason"] == "the profile shows idle CUs"


def _flat(tmp_path: Path) -> tuple[DirectionLedger, list[dict]]:
    """Four flat hours on two directions expected below the bar, both now mined out."""
    ledger = DirectionLedger(tmp_path)
    ledger.add([{"title": t, "expected_gain_pct": 1} for t in ("a", "b")], source="survey", iteration=0)
    events = []
    for iteration, direction in ((1, "a"), (2, "a"), (3, "b"), (4, "b")):
        events += [_cost(iteration, 1.0), _event(iteration, "REVERT_PERF", 1.0, direction)]
    return ledger, events


def _rule(ledger, events, *, queue_empty=True, inbox=False, **policy) -> EarlyStopStanding:
    return early_stop_standing(
        EarlyStopPolicy(**({"enabled": True} | policy)),
        ledger,
        events,
        start_score=1.0,
        lane_queue_empty=queue_empty,
        inbox_pending=inbox,
    )


def test_the_rule_asks_for_an_audit_before_it_stops(tmp_path: Path) -> None:
    ledger, events = _flat(tmp_path)
    first = _rule(ledger, events)
    assert first.criteria_met and first.audit_due and not first.stop
    ledger.record_audit(iteration=5, ledger_size=2, added=[])
    after_empty_audit = _rule(ledger, events)
    assert after_empty_audit.stop and not after_empty_audit.audit_due


def test_an_audit_that_found_directions_rearms_the_rule(tmp_path: Path) -> None:
    ledger, events = _flat(tmp_path)
    ledger.record_audit(iteration=5, ledger_size=2, added=["c"])
    assert _rule(ledger, events).audit_due and not _rule(ledger, events).stop
    # An audit against a smaller ledger no longer counts.
    ledger, events = _flat(tmp_path / "stale")
    ledger.record_audit(iteration=5, ledger_size=1, added=[])
    assert _rule(ledger, events).audit_due


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"min_gain": 0.0},
        {"window_hours": 5.0},
        {"queue_empty": False},
        {"inbox": True},
    ],
)
def test_any_unmet_criterion_holds_the_campaign(tmp_path: Path, overrides: dict) -> None:
    ledger, events = _flat(tmp_path)
    ledger.record_audit(iteration=5, ledger_size=2, added=[])
    standing = _rule(ledger, events, **overrides)
    assert not standing.stop and not standing.audit_due


def test_a_direction_still_worth_the_bar_holds_the_campaign_and_once_flat_is_where_plans_go(tmp_path: Path) -> None:
    ledger, events = _flat(tmp_path)
    ledger.add([{"title": "c", "expected_gain_pct": 5}], source="audit", iteration=5)
    standing = _rule(ledger, events)
    assert standing.promising == ("c",) and standing.focus == ("c",)
    assert not standing.criteria_met
    assert "1 directions still expected to gain 3% or more, or with no estimate" in standing.reasons
    # Before progress is flat the ledger only orders the plans.
    assert _rule(ledger, [*events, _cost(5, 1.0), _event(5, "KEEP", 1.05, "c")]).focus == ()


def test_a_direction_listed_without_an_estimate_holds_the_campaign_until_it_is_mined_out(tmp_path: Path) -> None:
    ledger, events = _flat(tmp_path)
    ledger.add([{"title": "d"}], source="planner", iteration=5)
    standing = _rule(ledger, events)
    assert standing.promising == ("d",) and standing.focus == ("d",) and not standing.criteria_met
    for iteration in (5, 6):
        events += [_cost(iteration, 1.0), _event(iteration, "REVERT_PERF", 1.0, "d")]
    assert _rule(ledger, events).criteria_met


def test_only_evidenced_directions_survive_a_reply(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("evidence")
    evidence = [str(tmp_path / "report.md")]
    reply = (
        "Reasoning.\n```json\n"
        + json.dumps(
            {
                "new_directions": [
                    {"title": "kept", "expected_gain_pct": 4, "evidence": evidence},
                    {"title": "dropped", "evidence": ["/no/such/file"]},
                    {"title": "", "evidence": evidence},
                ]
            }
        )
        + "\n```"
    )
    assert [entry["title"] for entry in parse_new_directions(reply, workspace=tmp_path)] == ["kept"]
    assert parse_new_directions("no json", workspace=tmp_path) == []


def _context(**overrides) -> OrchestrationContext:
    values = dict(
        analysis_commit="abc",
        workspace="/ws",
        gpu_target="gfx950",
        objective="speed",
        program_context="Optimize the kernel.",
        source_map_path="/ws/kernel.py",
        cases=(CaseEvidence(case_id="m=8", latency_ms=1.0),),
    )
    return OrchestrationContext(**(values | overrides))


def test_planners_see_the_ledger_and_the_rule_only_with_early_stop() -> None:
    assert "directions" not in _context().to_prompt_dict()
    payload = _context(direction_ledger=()).to_prompt_dict()["directions"]
    assert payload["ledger"] == [] and payload["rule"].startswith("Begin every plan") and "focus" not in payload
    focus = _context(direction_ledger=(), direction_focus=("c",)).to_prompt_dict()["directions"]["focus"]
    assert focus["ids"] == ["c"] and "every plan pursues one of them" in focus["rule"]


def _loop(tmp_path: Path, **overrides) -> SimpleNamespace:
    root = tmp_path / "forge_experiments"
    root.mkdir(exist_ok=True)
    events: list[dict] = []
    ledger = DirectionLedger(root)
    values = dict(
        ic=SimpleNamespace(
            workspace_dir=str(tmp_path), early_stop=EarlyStopPolicy(enabled=True), program_md="", max_rounds=0
        ),
        state_store=SimpleNamespace(root=root, append_event=events.append, read_events=lambda: list(events)),
        run_state=SimpleNamespace(search_start_mean_case_speedup=1.0),
        _ledger=lambda: ledger,
        _active_analysis_context=None,
        _direction_payload=lambda standing, mode: {"mode": mode},
        events=events,
    )
    return SimpleNamespace(**(values | overrides))


def test_a_candidate_is_attributed_to_the_direction_its_plan_names(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    loop._ledger().add([{"title": "Split K"}], source="survey", iteration=1)
    sink: dict = {}
    IterationLoop._attribute_candidate(loop, sink, "Direction: split-k\n...", 4, lane_id="2")
    assert sink == {"lane_id": "2", "direction_id": "split-k"}
    IterationLoop._attribute_candidate(loop, sink, "Direction: new | epilogue | Direct stores", 5)
    assert sink["direction_id"] == "direct-stores"


async def test_an_empty_audit_is_recorded_and_a_failed_one_is_not(tmp_path: Path) -> None:
    replies = [None, []]

    async def audit(mode, *, workspace, iteration, payload):
        assert mode == "audit" and payload["mode"] == "audit"
        return replies.pop(0)

    loop = _loop(tmp_path, _direction_audit_fn=audit)
    loop._early_stop_standing = lambda: SimpleNamespace(audit_due=True)
    await IterationLoop._audit_directions_if_due(loop, 7)
    assert loop._ledger().audits() == []
    await IterationLoop._audit_directions_if_due(loop, 8)
    assert loop._ledger().audits() == [{"iteration": 8, "ledger_size": 0, "added": []}]
    assert loop.events[-1]["type"] == "direction_audit"


def test_the_loop_stops_as_directions_exhausted(tmp_path: Path) -> None:
    standing = EarlyStopStanding((), TimeWindow(3.2, 4, 0.004), True, (), True, False, True, ())
    loop = _loop(
        tmp_path,
        _is_gate_met=lambda: False,
        _is_roofline_target_met=lambda: False,
        _is_budget_exhausted=lambda: False,
        _is_force_stopped=lambda: False,
        _early_stop_standing=lambda: standing,
        termination_reason="",
    )
    loop.run_state = SimpleNamespace(orchestration_circuit_state="CLOSED")
    assert IterationLoop._stop_condition_met(loop, 12) == (True, None)
    assert loop.termination_reason == "directions_exhausted"


def test_a_pending_operator_message_is_seen_by_the_rule(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _lane_queue=[])
    post_message(tmp_path, "keep going")
    assert "operator messages not yet read" in IterationLoop._early_stop_standing(loop).reasons


def test_the_dashboard_shows_the_ledger_by_expected_gain_and_the_rule(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    loop._ledger().add([{"title": "Split K", "expected_gain_pct": 4}], source="survey", iteration=1)
    root = tmp_path / "forge_experiments"
    (root / "events.jsonl").write_text(json.dumps(_event(1, "REVERT_PERF", 1.0, "split-k")) + "\n")
    # A render of an older campaign carries thresholds the policy no longer has.
    live = LiveStatus(status="running", early_stop={"enabled": True, "min_directions": 10})
    html = publish_dashboard(root, live).read_text()
    assert "Directions (1)" in html and "Split K" in html and "4.0%" in html
    assert "Early stop: the last 3 h of attempts gain under 2%" in html
    assert "Now: under 3 h of attempts so far." in html
