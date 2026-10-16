# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A measured winner promotion refuses is recorded on state and stated in the report."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import report as rp
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


def _coord(tmp_path: Path) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(baseline_tput=1000.0, current_best={"tput": 1000.0}, model_path="/models/m")
    return coord


def test_refused_lift_is_recorded_with_its_reason(tmp_path):
    coord = _coord(tmp_path)

    assert coord.writeback.lift_to_current_best("explore", 985.0, {"name": "v1", "output_throughput": 985.0}) is False

    assert coord.shared_state.optimization_stack == []
    (held,) = coord.shared_state.unpromoted_keeps
    assert (held["action"], held["reason"]) == ("explore", "no_lift")
    assert "985.0" in held["detail"] and held["ts"]


def test_a_promoted_winner_records_nothing(tmp_path):
    coord = _coord(tmp_path)

    assert coord.writeback.lift_to_current_best("explore", 1500.0, {"name": "v1", "output_throughput": 1500.0}) is True

    assert coord.shared_state.unpromoted_keeps == []


def test_refusals_survive_a_state_round_trip(tmp_path):
    coord = _coord(tmp_path)
    coord.writeback.lift_to_current_best("explore", 985.0, {"name": "v1", "output_throughput": 985.0})

    reloaded = SharedState.from_dict(coord.shared_state.to_dict())

    assert [e["reason"] for e in reloaded.unpromoted_keeps] == ["no_lift"]


def test_report_states_why_the_stack_is_empty():
    out = rp._format_completeness_annotations(
        {"unpromoted_keeps": [{"action": "explore", "reason": "guard_refused", "detail": "refused by p90"}]}
    )

    body = "\n".join(out)
    assert "refused promotion" in body
    assert "| `explore` | `guard_refused` | refused by p90 |" in body


def test_report_is_silent_without_refusals():
    assert rp._format_completeness_annotations({"unpromoted_keeps": []}) == []
