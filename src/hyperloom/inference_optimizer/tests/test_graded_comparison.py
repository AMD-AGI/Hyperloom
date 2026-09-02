# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grading chokepoint reads candidate and reference off ONE axis.

Total runs ~140x output on the agentic corpus, so these pin the two sides
together rather than either one alone.
"""

from __future__ import annotations

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT, GRADED_TOTAL, graded_axes_of
from hyperloom.orchestrator.state.shared_state import (
    resolve_graded_comparison,
    resolve_grading_anchor_tput,
)


# A measured AgentX round: prefill dominates, so the two axes cannot be confused.
_ANCHOR = {
    "tput": 183.44,
    "input_throughput": 25801.36,
    "output_throughput": 183.44,
    "total_throughput": 25984.80,
    "intvty_p90": 447.20,
}


class _State:
    """Minimal SharedState double: the attributes the resolvers read."""

    def __init__(self, *, current_best=None, baseline_perf=None, baseline_tput=0.0, framework="vllm"):
        self.current_best = current_best if current_best is not None else {}
        self.baseline_perf = baseline_perf if baseline_perf is not None else {}
        self.baseline_tput = baseline_tput
        self.framework = framework


def _agentx(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


def _synthetic(monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _full_measurement(*, total: float, output: float, intvty: float) -> dict[str, float]:
    return {
        "output_throughput": output,
        "input_throughput": total - output,
        "total_token_throughput": total,
        "intvty_p90": intvty,
    }


# --- both sides, one axis ---


def test_agentx_reads_both_sides_on_the_total_axis(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_TOTAL
    assert graded.candidate == pytest.approx(26500.0)
    assert graded.reference == pytest.approx(_ANCHOR["total_throughput"])
    assert graded.degrade_reason == ""


def test_a_candidate_without_the_graded_axes_degrades_both_sides_together(monkeypatch):
    """Dropping only the candidate would divide it by a ~140x larger reference."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, {"output_throughput": 190.0})

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(_ANCHOR["output_throughput"])
    assert graded.degrade_reason == "candidate_axes_missing"


def test_an_anchor_without_the_graded_axes_degrades_both_sides_together(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best={"tput": 183.44}, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(183.44)
    assert graded.degrade_reason


def test_a_synthetic_run_grades_output_against_output(monkeypatch):
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(_ANCHOR["output_throughput"])
    assert graded.degrade_reason == ""


def test_a_scriptable_framework_keeps_output_grading_under_agentx(monkeypatch):
    """A scriptable framework reports an image-quality gate, not a total axis."""
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0, framework="xdit")
    graded = resolve_graded_comparison(state, _full_measurement(total=26500.0, output=190.0, intvty=450.0))

    assert graded.objective == GRADED_OUTPUT


# --- the interactivity constraint travels with the objective ---


def test_an_interactivity_regression_is_vetoed_not_scored(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=40000.0, output=190.0, intvty=200.0))

    assert graded.graded_on_total
    assert graded.vetoed is True


def test_the_veto_never_fires_on_the_output_axis(monkeypatch):
    """The constraint belongs to the total objective."""
    _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)
    graded = resolve_graded_comparison(state, _full_measurement(total=40000.0, output=190.0, intvty=1.0))

    assert graded.vetoed is False


# --- grading against the session baseline ---


def test_cumulative_gain_reads_the_baseline_on_the_graded_axis(monkeypatch):
    _agentx(monkeypatch)
    state = _State(baseline_perf=_ANCHOR, baseline_tput=183.44)
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=26500.0, output=190.0, intvty=450.0),
        against_baseline=True,
    )

    assert graded.objective == GRADED_TOTAL
    assert graded.reference == pytest.approx(_ANCHOR["total_throughput"])


def test_cumulative_gain_falls_back_to_baseline_tput_together(monkeypatch):
    """A resumed session that never wrote ``baseline_perf`` still grades one axis."""
    _agentx(monkeypatch)
    state = _State(baseline_perf={}, baseline_tput=183.44)
    graded = resolve_graded_comparison(
        state,
        _full_measurement(total=26500.0, output=190.0, intvty=450.0),
        against_baseline=True,
    )

    assert graded.objective == GRADED_OUTPUT
    assert graded.candidate == pytest.approx(190.0)
    assert graded.reference == pytest.approx(183.44)
    assert graded.degrade_reason == "baseline_axes_missing"


# --- the anchor chokepoint stays on the output axis ---


@pytest.mark.parametrize("agentx", [True, False])
def test_the_anchor_chokepoint_is_the_output_axis_on_every_session(monkeypatch, agentx):
    """It seeds ``base_tput``, backs the drift check, and answers the two
    objective resolvers, whose targets are operator-supplied output figures.
    """
    _agentx(monkeypatch) if agentx else _synthetic(monkeypatch)
    state = _State(current_best=_ANCHOR, baseline_tput=180.0)

    assert resolve_grading_anchor_tput(state) == pytest.approx(_ANCHOR["tput"])


def test_the_anchor_falls_back_to_the_baseline_before_any_layer_lands(monkeypatch):
    _agentx(monkeypatch)
    state = _State(current_best={}, baseline_tput=180.0)

    assert resolve_grading_anchor_tput(state) == pytest.approx(180.0)


# --- a KEEP must not strip the axes off the anchor ---


def test_graded_axes_survive_a_winner_record(monkeypatch):
    """The defect this guards: total grading dies after the first KEEP.

    ``current_best`` is the next round's anchor. A winner record that carries
    no graded axes writes an anchor with none, `resolve_grading_anchor_perf`
    then answers ``current_best_axes_missing`` for the rest of the session, and
    the objective silently reverts to output throughput -- the exact grading
    this branch exists to replace.
    """
    _agentx(monkeypatch)
    measurement = _full_measurement(total=26500.0, output=190.0, intvty=450.0)
    winner = {"name": "v", "tput": 190.0, **graded_axes_of(measurement)}

    graded = resolve_graded_comparison(_State(current_best=winner, baseline_tput=180.0), measurement)
    assert graded.objective == GRADED_TOTAL
    assert graded.reference == pytest.approx(26500.0)


def test_graded_axes_of_omits_what_was_not_measured():
    assert graded_axes_of({"output_throughput": 190.0}) == {}
    assert graded_axes_of({"total_token_throughput": 26500.0}) == {"total_throughput": 26500.0}
    assert graded_axes_of(None) == {}
