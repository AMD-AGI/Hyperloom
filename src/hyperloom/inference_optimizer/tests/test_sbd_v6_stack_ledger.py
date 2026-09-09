# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``stack`` event and the ``outcome.validation`` it feeds.

Most of what these tests pin is something the legacy ledger got wrong, and all
of it is the same mistake: figures that were facts at one moment were thrown
away and reconstructed later from whatever survived.

The throughput each adoption beat was never recorded, so the chain's before
figures were inferred by pairing up attempt rows. Step gains were computed on
each step's own anchor and then added, which is not how ratios compose, and the
drift that produced was published as ``unattributed_gain_pct`` -- as though it
were gain nobody could account for rather than an artifact of the arithmetic.
Eight guard counts existed to report where the three v4 streams the ledger was
rebuilt from disagreed with one another.

The tests below pin the arithmetic identity that replaces the first two
(contributions on one denominator sum to the chain total exactly, and the
residual is the anchor movement), and pin that the residual is zero on a
continuous chain -- which is the case the legacy sum could not produce.
"""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_outcome
from hyperloom.inference_optimizer.breakdown.recorder import stack_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import stack_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope

BASE = 100.0


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _ext() -> dict[str, Any]:
    """The ledger's assembled ``ext``, whether or not it has closed."""
    ext, _status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
    return ext


def _status() -> str:
    _ext_, status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
    return status


def _events(session_dir) -> list[dict[str, Any]]:
    """The ledger events on the timeline."""
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "stack"]


def _adopt(
    index: int,
    action: str,
    before: float | None,
    after: float | None,
    *,
    backend: str = "",
    baseline: float | None = BASE,
    **entry: Any,
) -> None:
    """Record one adoption the way the lift does."""
    stack_event.record_adoption(
        stack_index=index,
        entry={"action": action, "backend": backend, **entry},
        throughput_before=before,
        throughput_after=after,
        baseline_tput=baseline,
        objective="output_throughput",
    )


def _rows() -> list[dict[str, Any]]:
    return _ext()["adoptions"]["rows"]


# --------------------------------------------------------------------------
# The arithmetic: one denominator, and a residual that means something
# --------------------------------------------------------------------------


def test_contributions_sum_to_the_chain_total_on_a_continuous_chain():
    """A chain with no re-measured anchor has no unattributed gain at all.

    This is the case the legacy ledger could not produce. Summing 10/100,
    10/110 and 10/120 gives 26.5%, while the chain actually gained 30% -- and
    the 3.5% difference was published as gain nobody could attribute, on a
    chain where every step is accounted for. Measuring each step against the
    baseline the three share gives 10 + 10 + 10 = 30, exactly the chain total.
    """
    _adopt(0, "explore", 100.0, 110.0)
    _adopt(1, "explore", 110.0, 120.0)
    _adopt(2, "integrate", 120.0, 130.0)

    ext = _ext()
    assert ext["attributed_gain_pct"] == 30.0
    assert ext["chain_total_gain_pct"] == 30.0
    assert ext["unattributed_gain_pct"] == 0.0
    assert ext["guards"][stack_event.GUARD_CHAIN_BREAKS] == 0


def test_unattributed_gain_is_exactly_the_anchor_movement():
    """When the anchor moves between adoptions, the residual names how much.

    The second adoption's anchor is 115, not the 110 the first one measured:
    something re-measured the service in between. That 5 tok/s is real gain the
    chain carries and no adoption claims, and it is what ``unattributed`` is
    for -- so it should equal exactly the movement, 5/100 = 5%.
    """
    _adopt(0, "explore", 100.0, 110.0)
    _adopt(1, "explore", 115.0, 125.0)

    ext = _ext()
    assert ext["attributed_gain_pct"] == 20.0
    assert ext["chain_total_gain_pct"] == 25.0
    assert ext["unattributed_gain_pct"] == 5.0
    assert ext["guards"][stack_event.GUARD_CHAIN_BREAKS] == 1


def test_before_and_after_are_recorded_not_derived():
    """Both endpoints are on the row, so no reader has to pair rows up.

    ``throughput_before`` is the figure the lift refused to accept the winner
    without beating, and the ``current_best`` write immediately after the
    append overwrites it. Recording it here is the only chance.
    """
    _adopt(0, "explore", 100.0, 137.5)

    row = _rows()[0]
    assert row["throughput_before"] == 100.0
    assert row["throughput_after"] == 137.5
    assert row["baseline_tput"] == 100.0
    # The step's own gain over its anchor is kept as well: it is what the
    # promotion decision was made on, even though it is not what sums.
    assert row["local_gain_pct"] == 37.5
    assert row["contribution_pct"] == 37.5


def test_local_gain_and_contribution_diverge_once_the_anchor_leaves_the_baseline():
    """The two gain figures answer different questions and are both kept."""
    _adopt(0, "explore", 100.0, 200.0)
    _adopt(1, "explore", 200.0, 300.0)

    second = _rows()[1]
    # Over its own anchor the second adoption gained 50%...
    assert second["local_gain_pct"] == 50.0
    # ...and it contributed 100 points of baseline throughput, same as the first.
    assert second["contribution_pct"] == 100.0
    assert _ext()["attributed_gain_pct"] == 200.0


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def test_by_source_buckets_each_adoption_and_reports_empty_buckets():
    """Every bucket is present, so "earned nothing" reads differently from "not reported"."""
    _adopt(0, "replay_warm_recipe", 100.0, 110.0)
    _adopt(1, "explore", 110.0, 120.0)
    _adopt(2, "integrate", 120.0, 140.0, backend="forge")

    buckets = _ext()["adoptions"]["by_source"]
    assert buckets["warm_replay"] == {"count": 1, "total_gain_pct": 10.0, "unmeasured": 0}
    assert buckets["explore"] == {"count": 1, "total_gain_pct": 10.0, "unmeasured": 0}
    assert buckets["framework_agent"] == {"count": 0, "total_gain_pct": 0.0, "unmeasured": 0}
    assert buckets["kernel"]["count"] == 1
    assert buckets["kernel"]["total_gain_pct"] == 20.0
    # The buckets partition the ledger, so they sum to the attributed total.
    assert sum(b["total_gain_pct"] for b in buckets.values()) == _ext()["attributed_gain_pct"]


def test_kernel_bucket_splits_by_backend_and_folds_unrecognized_backends():
    """A kernel adoption with no backend is reported, not silently dropped.

    Which backend authored a kernel adoption is not always in scope at the
    lift: the integrate path carries a kernel id and no backend. Folding those
    into ``unattributed`` keeps the split summing to the kernel bucket, so the
    per-backend figures cannot quietly under-report it.
    """
    _adopt(0, "gemm_tuning", 100.0, 110.0, backend="geak")
    _adopt(1, "collective", 110.0, 120.0, backend="forge")
    _adopt(2, "integrate", 120.0, 130.0)

    kernel = _ext()["adoptions"]["by_source"]["kernel"]
    backends = kernel["by_backend"]
    assert backends["geak"]["total_gain_pct"] == 10.0
    assert backends["forge"]["total_gain_pct"] == 10.0
    assert backends["unattributed"] == {"count": 1, "total_gain_pct": 10.0, "unmeasured": 0}
    assert sum(b["total_gain_pct"] for b in backends.values()) == kernel["total_gain_pct"]


def test_an_unlisted_action_lands_in_unattributed_with_its_kind_intact():
    """A gain nobody claimed is a finding, so the raw action stays on the row."""
    _adopt(0, "some_future_kind", 100.0, 120.0)

    row = _rows()[0]
    assert row["source"] == stack_event.SOURCE_UNATTRIBUTED
    assert row["action"] == "some_future_kind"
    assert _ext()["adoptions"]["by_source"]["unattributed"]["total_gain_pct"] == 20.0


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_an_unmeasurable_adoption_is_counted_rather_than_treated_as_zero():
    """No measurement is a different fact from no gain, and the total says so."""
    _adopt(0, "explore", 100.0, 120.0)
    _adopt(1, "explore", None, None)

    ext = _ext()
    assert ext["guards"][stack_event.GUARD_UNMEASURED] == 1
    assert _rows()[1]["contribution_pct"] is None
    # The measurable adoption still contributes; the total is a floor, and the
    # guard is what tells a reader that.
    assert ext["attributed_gain_pct"] == 20.0
    assert ext["adoptions"]["by_source"]["explore"]["unmeasured"] == 1


def test_a_zero_baseline_makes_every_contribution_unmeasurable():
    """With no baseline there is no denominator, so nothing can be attributed."""
    _adopt(0, "explore", 100.0, 120.0, baseline=0.0)

    ext = _ext()
    assert ext["attributed_gain_pct"] == 0.0
    assert ext["guards"][stack_event.GUARD_UNMEASURED] == 1
    assert ext["unattributed_gain_pct"] is None


# --------------------------------------------------------------------------
# Reconciliation against the whole-stack measurement
# --------------------------------------------------------------------------


def test_reconciliation_gap_compares_the_ledger_to_the_measured_whole():
    """The parts and the whole are asked the same question, and may disagree."""
    _adopt(0, "explore", 100.0, 110.0)
    _adopt(1, "explore", 110.0, 120.0)
    stack_event.record_validation(
        stack_len=2,
        baseline_tput=100.0,
        validated_tput=118.0,
        validated_gain_pct=18.0,
        source="writeback",
        measurement_basis="e2e_rebench",
    )

    ext = _ext()
    assert ext["chain_total_gain_pct"] == 20.0
    assert ext["validated_total_gain_pct"] == 18.0
    # The stack measured 2 points worse as a whole than its steps claimed.
    assert ext["reconciliation_gap_pct"] == -2.0
    assert ext["validations"]["at_head"] is True


def test_a_re_validation_at_one_stack_length_supersedes_the_earlier_figure():
    """Re-measuring the same stack replaces the figure it revises.

    Keyed by stack length rather than appended, because two figures for one
    stack are not two facts -- the second is a correction of the first.
    """
    _adopt(0, "explore", 100.0, 120.0)
    for gain, tput in ((20.0, 120.0), (17.5, 117.5)):
        stack_event.record_validation(
            stack_len=1,
            baseline_tput=100.0,
            validated_tput=tput,
            validated_gain_pct=gain,
            source="writeback",
            measurement_basis="e2e_rebench",
        )

    ext = _ext()
    assert ext["validations"]["count"] == 1
    assert ext["validated_total_gain_pct"] == 17.5


def test_a_validation_behind_the_stack_head_is_reported_as_such():
    """The last adoptions were never measured as a whole, and the ledger says so.

    A total validated at stack length 1 is a claim about a one-entry stack. The
    session shipped two, so the figure does not cover what shipped.
    """
    _adopt(0, "explore", 100.0, 110.0)
    stack_event.record_validation(
        stack_len=1,
        baseline_tput=100.0,
        validated_tput=110.0,
        validated_gain_pct=10.0,
    )
    _adopt(1, "explore", 110.0, 130.0)

    assert _ext()["validations"]["at_head"] is False


def test_no_validation_leaves_the_ledger_with_nothing_to_reconcile_against():
    """Absent an independent figure, the gap is unknown rather than zero."""
    _adopt(0, "explore", 100.0, 120.0)

    ext = _ext()
    assert ext["validated_total_gain_pct"] is None
    assert ext["reconciliation_gap_pct"] is None


# --------------------------------------------------------------------------
# Event lifecycle
# --------------------------------------------------------------------------


def test_the_ledger_is_one_event_however_many_phases_adopt_onto_it(tmp_path):
    """Adoptions from four phases land on one chain, not four."""
    _adopt(0, "replay_warm_recipe", 100.0, 105.0, source_phase="PRELUDE")
    _adopt(1, "explore", 105.0, 115.0, source_phase="EXPLORE")
    _adopt(2, "integrate_patch", 115.0, 125.0, source_phase="FRAMEWORK_AGENT")
    _adopt(3, "integrate", 125.0, 140.0, source_phase="KERNEL_AGENT")
    stack_event.finish()

    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["id"] == "stack:0:stack"
    assert events[0]["ext"]["adoptions"]["count"] == 4
    assert [row["stack_index"] for row in events[0]["ext"]["adoptions"]["rows"]] == [0, 1, 2, 3]


def test_rows_are_ordered_by_stack_position_not_by_arrival():
    """The chain is an order, and the assembled rows carry it."""
    _adopt(2, "integrate", 120.0, 130.0)
    _adopt(0, "explore", 100.0, 110.0)
    _adopt(1, "explore", 110.0, 120.0)

    assert [row["stack_index"] for row in _rows()] == [0, 1, 2]


def test_a_session_that_kept_nothing_closes_a_skipped_ledger(tmp_path):
    """Zero adoptions is a result, and it is not the same as no ledger."""
    stack_event.finish()

    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == stack_event.STATUS_SKIPPED
    assert events[0]["ext"]["adoptions"]["count"] == 0


def test_a_run_that_never_reached_close_leaves_the_ledger_interrupted(tmp_path):
    """The reconciliation never ran, and the timeline should not pretend it did."""
    _adopt(0, "explore", 100.0, 120.0)
    finalize_events(tmp_path)

    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "interrupted"
    # Recovered from the fragments, so the figures survive the interruption.
    assert events[0]["ext"]["attributed_gain_pct"] == 20.0


def test_a_ledger_with_adoptions_but_no_measurable_total_is_degraded():
    """Something was kept and there is no figure for what it bought."""
    _adopt(0, "explore", None, None)

    assert _status() == stack_event.STATUS_DEGRADED


def test_recording_without_a_bound_session_is_a_no_op(tmp_path, monkeypatch):
    """An adoption outranks its own record; the lift must not fail on it."""
    from hyperloom.inference_optimizer.session import session_binding

    monkeypatch.setattr(session_binding, "bound_session_or_none", lambda: None)
    _adopt(0, "explore", 100.0, 120.0)
    stack_event.record_validation(stack_len=1, baseline_tput=100.0, validated_tput=120.0, validated_gain_pct=20.0)
    stack_event.finish()


# --------------------------------------------------------------------------
# outcome.validation reads the ledger
# --------------------------------------------------------------------------


def _outcome(session_dir) -> dict[str, Any]:
    """``outcome.validation`` over the recorded timeline."""
    stack_event.finish()
    return collect_v6_outcome(
        session={"stop_reason": "target_reached"},
        close={},
        state={},
        timeline=read_timeline_events(session_dir),
    )["validation"]


def test_outcome_validation_reads_the_ledger_rather_than_a_legacy_projection(tmp_path):
    """Every figure comes off the ``stack`` event; no ``optimizations`` involved.

    ``collect_v6_outcome`` no longer takes the section at all, which is the
    point: the coupling that blocked deleting it is gone by construction, not
    by convention.
    """
    _adopt(0, "replay_warm_recipe", 100.0, 110.0)
    _adopt(1, "integrate", 110.0, 130.0, backend="geak")
    stack_event.record_validation(stack_len=2, baseline_tput=100.0, validated_tput=130.0, validated_gain_pct=30.0)

    validation = _outcome(tmp_path)
    assert validation["attributed_gain_pct"] == 30.0
    assert validation["unattributed_gain_pct"] == 0.0
    assert validation["reconciliation_gap_pct"] == 0.0
    assert validation["validated_total_gain_pct"] == 30.0
    by_source = validation["attribution"]["by_source"]
    assert validation["attribution"]["available"] is True
    assert by_source["warm_replay"] == {"total_gain_pct": 10.0, "keep_count": 1, "unmeasured_keep_count": 0}
    assert by_source["kernel"]["by_backend"]["geak"]["total_gain_pct"] == 20.0
    assert by_source["kernel"]["by_backend"]["forge"]["keep_count"] == 0


def test_outcome_folds_explore_into_framework_agent(tmp_path):
    """V6 retired the Explore phase into Framework Agent; its gain follows."""
    _adopt(0, "explore", 100.0, 110.0)
    _adopt(1, "integrate_patch", 110.0, 120.0)

    by_source = _outcome(tmp_path)["attribution"]["by_source"]
    assert by_source["framework_agent"] == {
        "total_gain_pct": 20.0,
        "keep_count": 2,
        "unmeasured_keep_count": 0,
    }


def test_outcome_notes_are_findings_and_empty_when_the_ledger_reconciles(tmp_path):
    """The legacy notes were prose about how the ledger was built."""
    _adopt(0, "explore", 100.0, 110.0)
    stack_event.record_validation(stack_len=1, baseline_tput=100.0, validated_tput=110.0, validated_gain_pct=10.0)

    assert _outcome(tmp_path)["notes"] == []


def test_outcome_notes_name_the_anchor_movement_and_the_missing_measurement(tmp_path):
    """Each note is present only because the recorded rows show its condition."""
    _adopt(0, "explore", 100.0, 110.0)
    _adopt(1, "explore", 120.0, 130.0)
    _adopt(2, "explore", None, None)

    validation = _outcome(tmp_path)
    assert validation["guards"] == {"unmeasured": 1, "chain_breaks": 1}
    joined = " | ".join(validation["notes"])
    assert "lower bound" in joined
    assert "anchor moved" in joined
    assert "nothing to reconcile against" in joined


def test_outcome_marks_attribution_unavailable_without_a_ledger_event():
    """A session with no ``stack`` event never closed one, and cannot claim zero."""
    validation = collect_v6_outcome(
        session={"stop_reason": "signal"},
        close={},
        state={},
        timeline=[],
    )["validation"]

    assert validation["attribution"]["available"] is False
    assert validation["attribution"]["by_source"]["warm_replay"]["total_gain_pct"] is None
    assert validation["reconciliation_gap_pct"] is None
    assert validation["notes"] == []
