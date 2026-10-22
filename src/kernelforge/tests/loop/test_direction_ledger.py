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
    direction_standings,
    keep_gains,
)
from kernelforge.loop.early_stop import EarlyStopPolicy, EarlyStopStanding, early_stop_standing, window_gain
from kernelforge.loop.operator_inbox import post_message
from kernelforge.loop.runner import IterationLoop
from kernelforge.orchestrator.contracts import CaseEvidence, OrchestrationContext
from kernelforge.orchestrator.direction_audit import parse_new_directions
from kernelforge.orchestrator.direction_claims import DirectionClaim, parse_direction_line


def _event(iteration: int, decision: str, best: float, direction: str = "") -> dict:
    return {
        "type": "iteration_result",
        "iter": iteration,
        "decision": decision,
        "best_after_mean_case_speedup": best,
        "direction_id": direction or None,
    }


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
        _event(5, "BUILD_FAILED", 1.10, "exhausted"),
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
    assert standings["exhausted"].attempts == 2  # the API error tried nothing
    assert standings["kept"].gain == pytest.approx(0.10)


def test_the_window_gain_counts_only_measured_rounds() -> None:
    results = [
        _event(1, "KEEP", 1.10),
        _event(2, "BUILD_FAILED", 1.10),
        _event(3, "REVERT_PERF", 1.10),
        _event(4, "KEEP", 1.111),
    ]
    gains = keep_gains(results, 1.0)
    assert window_gain(results, gains, 2) == pytest.approx(0.01)
    assert window_gain(results, gains, 4) is None


def _standings(tmp_path: Path, statuses_exhausted: bool = True):
    ledger = DirectionLedger(tmp_path)
    ledger.add([{"title": "a"}, {"title": "b"}], source="s", iteration=0)
    results = [_event(1, "REVERT_PERF", 1.0, "a"), _event(2, "REVERT_PERF", 1.0, "a")]
    if statuses_exhausted:
        results += [_event(3, "REVERT_PERF", 1.0, "b"), _event(4, "REVERT_PERF", 1.0, "b")]
    return direction_standings(ledger.directions(), results, exhaust_after=2)


def _rule(standings, *, gain=0.0, audits=(), queue_empty=True, inbox=False, enabled=True, min_directions=2):
    policy = EarlyStopPolicy(enabled=enabled, min_directions=min_directions, window_rounds=2, min_gain=0.02)
    return early_stop_standing(
        policy, standings, gain=gain, audits=list(audits), lane_queue_empty=queue_empty, inbox_pending=inbox
    )


def test_the_rule_asks_for_an_audit_before_it_stops(tmp_path: Path) -> None:
    standings = _standings(tmp_path)
    first = _rule(standings)
    assert first.criteria_met and first.audit_due and not first.stop
    after_empty_audit = _rule(standings, audits=[{"iteration": 5, "ledger_size": 2, "added": []}])
    assert after_empty_audit.stop and not after_empty_audit.audit_due


def test_an_audit_that_found_directions_rearms_the_rule(tmp_path: Path) -> None:
    standings = _standings(tmp_path)
    found = [{"iteration": 5, "ledger_size": 2, "added": ["c"]}]
    assert _rule(standings, audits=found).audit_due and not _rule(standings, audits=found).stop
    # An audit against a smaller ledger no longer counts.
    stale = [{"iteration": 5, "ledger_size": 1, "added": []}]
    assert _rule(standings, audits=stale).audit_due


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"gain": 0.05},
        {"gain": None},
        {"queue_empty": False},
        {"inbox": True},
        {"min_directions": 3},
    ],
)
def test_any_unmet_criterion_holds_the_campaign(tmp_path: Path, overrides: dict) -> None:
    standing = _rule(_standings(tmp_path), audits=[{"iteration": 5, "ledger_size": 2, "added": []}], **overrides)
    assert not standing.stop and not standing.audit_due


def test_a_direction_not_yet_mined_out_holds_the_campaign(tmp_path: Path) -> None:
    standing = _rule(_standings(tmp_path, statuses_exhausted=False))
    assert not standing.criteria_met
    assert "1 directions not yet exhausted" in standing.reasons


def test_only_evidenced_directions_survive_an_audit_reply(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("evidence")
    reply = (
        "Reasoning.\n```json\n"
        + json.dumps(
            {
                "new_directions": [
                    {"title": "kept", "evidence": [str(tmp_path / "report.md")]},
                    {"title": "dropped", "evidence": ["/no/such/file"]},
                    {"title": "", "evidence": [str(tmp_path / "report.md")]},
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
    assert payload["ledger"] == [] and payload["rule"].startswith("Begin every plan")


def _loop(tmp_path: Path, **overrides) -> SimpleNamespace:
    root = tmp_path / "forge_experiments"
    root.mkdir(exist_ok=True)
    events: list[dict] = []
    ledger = DirectionLedger(root)
    values = dict(
        ic=SimpleNamespace(workspace_dir=str(tmp_path), early_stop=EarlyStopPolicy(enabled=True), program_md=""),
        state_store=SimpleNamespace(root=root, append_event=events.append, read_events=lambda: list(events)),
        run_state=SimpleNamespace(search_start_mean_case_speedup=1.0),
        _ledger=lambda: ledger,
        _active_analysis_context=None,
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
    loop._early_stop_standing = lambda: EarlyStopStanding(2, 2, 0.0, True, True, False, ())
    loop._direction_payload = lambda ledger, mode: {"mode": mode}
    await IterationLoop._audit_directions_if_due(loop, 7)
    assert loop._ledger().audits() == []
    await IterationLoop._audit_directions_if_due(loop, 8)
    assert loop._ledger().audits() == [{"iteration": 8, "ledger_size": 0, "added": []}]
    assert loop.events[-1]["type"] == "direction_audit"


def test_the_loop_stops_as_directions_exhausted(tmp_path: Path) -> None:
    loop = _loop(
        tmp_path,
        _is_gate_met=lambda: False,
        _is_roofline_target_met=lambda: False,
        _is_budget_exhausted=lambda: False,
        _is_force_stopped=lambda: False,
        _early_stop_standing=lambda: EarlyStopStanding(10, 10, 0.004, True, False, True, ()),
        termination_reason="",
    )
    loop.run_state = SimpleNamespace(orchestration_circuit_state="CLOSED")
    assert IterationLoop._stop_condition_met(loop, 12) == (True, None)
    assert loop.termination_reason == "directions_exhausted"


def test_a_pending_operator_message_is_seen_by_the_rule(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _lane_queue=[])
    loop._direction_standings = lambda ledger, results=None: IterationLoop._direction_standings(loop, ledger, results)
    loop._iteration_results = list
    post_message(tmp_path, "keep going")
    assert "operator messages not yet read" in IterationLoop._early_stop_standing(loop).reasons


def test_the_dashboard_shows_the_ledger_and_the_rule(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    loop._ledger().add([{"title": "Split K"}], source="survey", iteration=1)
    root = tmp_path / "forge_experiments"
    (root / "events.jsonl").write_text(json.dumps(_event(1, "REVERT_PERF", 1.0, "split-k")) + "\n")
    live = LiveStatus(status="running", early_stop={"enabled": True, "min_directions": 10})
    html = publish_dashboard(root, live).read_text()
    assert "Directions (1)" in html and "Split K" in html
    assert "1 directions known, 10 required" in html
