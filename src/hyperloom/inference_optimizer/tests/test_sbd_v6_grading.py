# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the grading axis the SBD V6 document declares and the axes it publishes.

An AgentX replay is ranked on the slow-tail interactivity percentile with throughput held as a guard; a
synthetic run is ranked on output throughput alone. On the canonical corpus the two axes differ by roughly
two orders of magnitude, and every throughput field in the breakdown is the output axis by construction, so
without the declaration a consumer would sort one kind of session against the other and every number would
look plausible.

Two separate facts, deliberately not resolved from one another. ``metadata.grading`` is the axis the session
was configured for; ``outcome.validation.graded_on`` is the axis the run actually decided its last promotion
on, which differs whenever a comparison could not supply the axis pair.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.common.perf_metric import GRADED_AXIS_KEYS, GRADED_INTVTY, GRADED_OUTPUT
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_metadata, collect_v6_outcome
from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts, recorder_for, snapshot_metadata
from hyperloom.inference_optimizer.breakdown.recorder import stack_event
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import (
    PRODUCER as BASELINE_PRODUCER,
    baseline_event_id,
    make_baseline_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.breakdown.recorder.session_metadata import SECTION as METADATA_SECTION, _grading
from hyperloom.inference_optimizer.cli.bootstrap import seed_grading
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.state.shared_state import SharedState, resolved_grading

#: The AgentX axes a measured round carries, on the keys grading itself reads them from.
AGENTX_AXES: dict[str, Any] = {
    GRADED_INTVTY: 41.2,
    "total_throughput": 25978.0,
    "input_throughput": 25795.0,
    "tpot_p90_ms": 24.3,
}


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


@pytest.fixture(autouse=True)
def _no_ambient_grading(monkeypatch):
    """Clear the grading vars, so a test that does not set them is not reading the developer's shell."""
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _state(**fields: Any) -> SharedState:
    return SharedState(session_id="s-1", **fields)


# ---------------------------------------------------------------------------
# Resolving the axis at seed, where the run can still see its own configuration
# ---------------------------------------------------------------------------


def test_a_synthetic_session_is_seeded_on_the_output_axis():
    assert seed_grading("sglang", "synthetic")["objective"] == GRADED_OUTPUT


def test_an_agentx_session_is_seeded_on_the_interactivity_axis():
    assert seed_grading("sglang", "agentx")["objective"] == GRADED_INTVTY


def test_a_scriptable_framework_stays_on_output_even_under_agentx():
    # An image framework reports a quality gate, not an interactivity percentile, so the axis does not exist
    # for it to be ranked on.
    assert seed_grading("xdit", "agentx")["objective"] == GRADED_OUTPUT


def test_the_noise_band_is_captured_at_seed_rather_than_left_to_the_environment(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "3.5")

    assert seed_grading("sglang", "agentx")["noise_pct"] == 3.5


def test_an_explicit_metric_override_is_resolved_at_seed(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")

    assert seed_grading("sglang", "agentx")["objective"] == GRADED_OUTPUT


# ---------------------------------------------------------------------------
# What was recorded beats what the current process happens to hold
# ---------------------------------------------------------------------------


def test_the_recorded_axis_wins_over_a_shell_that_lost_the_variable(monkeypatch):
    # A resume is a new process. Deriving here would flip a session that graded on interactivity back to
    # output halfway through, and the KEEP/REVERT rule has to be the one the session started with.
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    state = _state(grading={"objective": GRADED_INTVTY, "noise_pct": 3.5}, benchmark_mode="synthetic")

    assert resolved_grading(state) == (True, 3.5)


def test_the_recorded_band_wins_over_the_ambient_default(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "5.0")
    state = _state(grading={"objective": GRADED_INTVTY, "noise_pct": 3.5})

    assert resolved_grading(state)[1] == 3.5


def test_a_session_predating_the_field_derives_and_reports_no_band():
    # The band that session applied was never recorded, and today's default is not evidence of it.
    state = _state(benchmark_mode="agentx", framework="sglang")

    assert resolved_grading(state) == (True, None)


# ---------------------------------------------------------------------------
# metadata.grading: the axis the session was configured for
# ---------------------------------------------------------------------------


def test_metadata_grading_declares_the_axis_and_the_guard_band():
    state = _state(
        benchmark_mode="agentx",
        grading={"objective": GRADED_INTVTY, "noise_pct": 3.5},
    )

    assert _grading(state) == {
        "benchmark_mode": "agentx",
        "objective": GRADED_INTVTY,
        "tput_guard": {"enabled": True, "noise_pct": 3.5},
    }


def test_metadata_grading_reports_no_guard_on_a_synthetic_session():
    state = _state(benchmark_mode="synthetic", grading=seed_grading("sglang", "synthetic"))

    block = _grading(state)
    assert block["objective"] == GRADED_OUTPUT
    assert block["tput_guard"]["enabled"] is False


def test_metadata_grading_names_the_mode_a_session_with_no_recorded_axis_ran():
    # ``benchmark_mode`` reaches ``reports/final.json`` but nothing else in the breakdown, so an AgentX
    # session would otherwise be unidentifiable in this document.
    assert _grading(_state(benchmark_mode="agentx", framework="sglang"))["benchmark_mode"] == "agentx"


def test_the_grading_block_reaches_the_exported_metadata(tmp_path):
    # The whole point of recording it: the collector projects nothing for this block, so it lands purely from
    # the spool and the export path reads no environment at all to produce it.
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state(benchmark_mode="agentx", grading={"objective": GRADED_INTVTY, "noise_pct": 3.5}))

    metadata = collect_v6_metadata(
        exported_at_utc="2026-09-01T02:00:05+00:00",
        session={"session_id": "s-1"},
        workload={"framework_name": "sglang"},
        model_info={},
        langfuse={"enabled": False},
        state={},
        warnings=[],
        recorded=assemble_parts(tmp_path)[METADATA_SECTION],
    )
    assert metadata["grading"] == {
        "benchmark_mode": "agentx",
        "objective": GRADED_INTVTY,
        "tput_guard": {"enabled": True, "noise_pct": 3.5},
    }


def test_an_unrecorded_noise_band_survives_the_export_as_a_null(tmp_path):
    # The singleton and the overlay both have to carry an explicit null through: today's default is not
    # evidence of the band a session that predates the field applied.
    rec = recorder_for(tmp_path, producer="coordinator")
    snapshot_metadata(rec, _state(benchmark_mode="agentx", framework="sglang"))

    recorded = assemble_parts(tmp_path)[METADATA_SECTION]
    assert recorded["grading"]["tput_guard"]["noise_pct"] is None


def test_metadata_grading_is_not_resolved_from_what_a_promotion_graded_on():
    # The lock on keeping the two facts separate. A session configured for interactivity whose comparisons
    # all degraded still asked for interactivity; reporting output here would erase the request, and
    # reporting interactivity in ``outcome`` would put that label on an output figure.
    state = _state(benchmark_mode="agentx", grading={"objective": GRADED_INTVTY, "noise_pct": 5.0})

    assert _grading(state)["objective"] == GRADED_INTVTY


# ---------------------------------------------------------------------------
# outcome: the axis a promotion was actually decided on, and the axes it measured
# ---------------------------------------------------------------------------


def _adopt(index: int, *, objective: str = GRADED_OUTPUT, degrade_reason: str = "") -> None:
    """Record one adoption the way the lift does."""
    stack_event.record_adoption(
        stack_index=index,
        entry={"action": "explore"},
        throughput_before=100.0 + index * 10.0,
        throughput_after=110.0 + index * 10.0,
        baseline_tput=100.0,
        objective=objective,
        degrade_reason=degrade_reason,
    )


def _outcome(session_dir, *, close: dict[str, Any] | None = None) -> dict[str, Any]:
    """The assembled ``outcome`` block over the recorded timeline."""
    stack_event.finish()
    return collect_v6_outcome(
        session={"stop_reason": "target_reached"},
        close=close or {},
        state={},
        timeline=read_timeline_events(session_dir),
    )


def test_outcome_reports_the_axis_the_settled_validation_was_measured_on(tmp_path):
    _adopt(0, objective=GRADED_INTVTY)
    stack_event.record_validation(
        stack_len=1,
        baseline_tput=38.0,
        validated_tput=41.2,
        validated_gain_pct=8.42,
        graded_objective=GRADED_INTVTY,
        measurement=AGENTX_AXES,
    )

    assert _outcome(tmp_path)["validation"]["graded_on"] == GRADED_INTVTY


def test_outcome_falls_back_to_the_adoption_axis_when_nothing_validated(tmp_path):
    # A session that adopted but never measured the whole stack has no settled row to read, and the axis its
    # adoptions were graded on is the only recorded answer.
    _adopt(0, objective=GRADED_INTVTY)

    assert _outcome(tmp_path)["validation"]["graded_on"] == GRADED_INTVTY


def test_the_final_gain_carries_the_same_axis_as_the_reconciliation(tmp_path):
    # One lock for both: the gain and the attribution are the same figure read twice, so a reader must never
    # find two axis labels on them.
    _adopt(0, objective=GRADED_INTVTY)
    stack_event.record_validation(
        stack_len=1,
        baseline_tput=38.0,
        validated_tput=41.2,
        validated_gain_pct=8.42,
        graded_objective=GRADED_INTVTY,
        measurement=AGENTX_AXES,
    )

    outcome = _outcome(tmp_path, close={"final_recipe": {"throughput": 183.0}})
    assert outcome["final"]["graded_on"] == outcome["validation"]["graded_on"] == GRADED_INTVTY


def test_the_settled_axes_are_published_beside_the_gain_they_produced(tmp_path):
    # Read off the validation row rather than ``current_best``: a revalidation moves the cumulative figure
    # without re-promoting the recipe, so ``current_best`` can be a different measurement entirely.
    stack_event.record_validation(
        stack_len=1,
        baseline_tput=38.0,
        validated_tput=41.2,
        validated_gain_pct=8.42,
        graded_objective=GRADED_INTVTY,
        measurement=AGENTX_AXES,
    )

    outcome = _outcome(tmp_path)
    assert outcome["validation"]["perf"] == AGENTX_AXES
    assert outcome["final"]["perf"] == AGENTX_AXES


def test_an_unmeasured_axis_is_an_explicit_null_rather_than_an_absent_key(tmp_path):
    # Absent would be indistinguishable from an axis the framework failed to report, and zero reads as
    # "measured, and it was zero".
    stack_event.record_validation(
        stack_len=1,
        baseline_tput=100.0,
        validated_tput=120.0,
        validated_gain_pct=20.0,
        graded_objective=GRADED_OUTPUT,
        measurement={"output_throughput": 120.0},
    )

    perf = _outcome(tmp_path)["validation"]["perf"]
    assert set(perf) == set(GRADED_AXIS_KEYS)
    assert all(value is None for value in perf.values())


def test_a_session_with_no_ledger_publishes_no_axis(tmp_path):
    outcome = collect_v6_outcome(session={"stop_reason": "signal"}, close={}, state={}, timeline=[])

    assert outcome["validation"]["graded_on"] is None
    assert all(value is None for value in outcome["validation"]["perf"].values())


def test_the_notes_name_adoptions_that_fell_off_the_configured_axis(tmp_path):
    # Their contributions sit in the same sum as the axis-graded ones, so the total is not single-axis and
    # the reader has to be told.
    _adopt(0, objective=GRADED_INTVTY)
    _adopt(1, objective=GRADED_OUTPUT, degrade_reason="candidate_axes_missing")
    _adopt(2, objective=GRADED_OUTPUT, degrade_reason="candidate_axes_missing")

    joined = " | ".join(_outcome(tmp_path)["validation"]["notes"])
    assert "2 adoption(s) were graded on the output axis" in joined
    assert "candidate_axes_missing" in joined


def test_a_fully_graded_ledger_reports_no_degrade_finding(tmp_path):
    # The whole and the parts agree here, so an empty list is the meaningful assertion: no finding at all,
    # rather than a degrade finding drowned out by a reconciliation complaint.
    _adopt(0, objective=GRADED_INTVTY)
    stack_event.record_validation(
        stack_len=1,
        baseline_tput=100.0,
        validated_tput=110.0,
        validated_gain_pct=10.0,
        graded_objective=GRADED_INTVTY,
        measurement=AGENTX_AXES,
    )

    assert _outcome(tmp_path)["validation"]["notes"] == []


# ---------------------------------------------------------------------------
# outcome.baseline: the axes the session was anchored on
# ---------------------------------------------------------------------------


def _record_baseline(**axes: Any) -> None:
    """Record an anchoring baseline the way a dispatched measurement does."""
    recorder = make_baseline_recorder(
        make_sink(baseline_event_id("prelude", 0), producer=BASELINE_PRODUCER),
        task_id="t-1",
        task_kind="baseline",
        reason="",
        framework="sglang",
        establishes_quality_ref=True,
        params={"config_path": "/cfg.yaml", "output_dir": "/w", "timeout_sec": 7800},
    )
    assert recorder is not None
    recorder.finish({"status": "succeeded", "output_throughput": 183.0, **axes})


def test_the_baseline_publishes_the_axes_the_session_was_anchored_on(tmp_path):
    # Recorded on the baseline round rather than read off ``state.baseline_perf`` at export, because this
    # block is already where ``outcome.baseline`` comes from and a second source is a second answer.
    _record_baseline(**AGENTX_AXES)

    baseline = _outcome(tmp_path)["baseline"]
    assert baseline["perf"] == AGENTX_AXES
    # The output axis keeps its own meaning beside them: this addition takes nothing away.
    assert baseline["throughput_tok_s_per_gpu"] == 183.0


def test_a_synthetic_baseline_publishes_four_nulls(tmp_path):
    _record_baseline()

    assert all(value is None for value in _outcome(tmp_path)["baseline"]["perf"].values())


def test_a_session_with_no_anchoring_baseline_still_publishes_the_axis_shape(tmp_path):
    assert set(_outcome(tmp_path)["baseline"]["perf"]) == set(GRADED_AXIS_KEYS)
