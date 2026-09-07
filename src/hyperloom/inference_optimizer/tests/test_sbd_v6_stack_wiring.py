# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stack ledger against the real production writers, not a stand-in.

``test_sbd_v6_stack_ledger.py`` pins what the ledger computes; these pin that
the orchestrator actually feeds it. The distinction matters here more than
usual, because the whole argument for recording adoptions is that
``throughput_before`` exists only inside ``_lift_to_current_best`` -- so a test
that hands the recorder a before figure of its own has assumed away the thing
being claimed.

The tests below drive the real ``WritebackCollaborator`` on a real
``SharedState`` and read what came out the other end, including the case the
lift refuses, which must leave no row at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import stack_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import stack_event_parts
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }


def _coord(session_dir: Path, *, baseline: float = 1000.0, anchor: float = 1000.0) -> Coordinator:
    """A coordinator anchored on a measured baseline, ready to be lifted."""
    coord = Coordinator(session_dir, backends=_silent_backends())
    coord.shared_state.baseline_tput = baseline
    coord.shared_state.current_best = {
        "action": "baseline",
        "tput": anchor,
        "extra_server_args": "",
        "extra_envs": {},
    }
    return coord


def _rows() -> list[dict[str, Any]]:
    """The adoption rows the run recorded."""
    ext, _status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
    return ext["adoptions"]["rows"]


def test_a_real_lift_records_the_anchor_it_beat(session_dir):
    """The figure the lift graded against lands on the row.

    This is the fact the legacy ledger had to reconstruct: it is read from
    ``current_best`` inside the lift and overwritten by the same call, so an
    export-time reader has only the final anchor and has to walk backwards
    through the attempt rows guessing which reading each step was graded on.
    """
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1000.0)

        assert coord._lift_to_current_best("explore", 1100.0, {"name": "page-size-64"}) is True

        rows = _rows()
        assert len(rows) == 1
        assert rows[0]["throughput_before"] == 1000.0
        assert rows[0]["throughput_after"] == 1100.0
        assert rows[0]["baseline_tput"] == 1000.0
        assert rows[0]["contribution_pct"] == 10.0
        assert rows[0]["stack_index"] == 0
        assert rows[0]["source"] == stack_event.SOURCE_EXPLORE


def test_the_row_index_matches_the_stack_the_lift_appended_to(session_dir):
    """``stack_index`` indexes ``optimization_stack`` directly, so it can be joined."""
    with session_scope(session_dir):
        coord = _coord(session_dir)
        coord._lift_to_current_best("explore", 1100.0, {"name": "first"})
        coord._lift_to_current_best("explore", 1200.0, {"name": "second"})

        rows = _rows()
        stack = coord.shared_state.optimization_stack
        assert [row["stack_index"] for row in rows] == [0, 1]
        for row in rows:
            assert stack[row["stack_index"]]["variant_name"] == row["variant_name"]


def test_a_chain_of_real_lifts_reconciles_against_its_own_baseline(session_dir):
    """Two real lifts, and the contributions add up to what the chain gained."""
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1000.0)
        coord._lift_to_current_best("explore", 1100.0, {"name": "first"})
        coord._lift_to_current_best("integrate", 1250.0, {"name": "second"})

        ext, status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
        assert ext["attributed_gain_pct"] == 25.0
        assert ext["chain_total_gain_pct"] == 25.0
        assert ext["unattributed_gain_pct"] == 0.0
        assert status == stack_event.STATUS_SUCCEEDED


def test_a_refused_lift_records_nothing(session_dir):
    """No append, no row. The ledger is the stack, not the attempts.

    A winner that does not beat the anchor is not an adoption, and the attempt
    is already on its own stage event. Recording it here would put a
    non-adoption in the chain and break the arithmetic that makes the ledger
    reconcile.
    """
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1200.0)

        assert coord._lift_to_current_best("explore", 1100.0, {"name": "loser"}) is False
        assert _rows() == []


def test_an_already_stacked_config_is_not_recorded_twice(session_dir):
    """A rerun of a stacked config re-anchors without re-earning its gain.

    The lift skips the append, keyed by ``(action, variant_name)``. Recording
    on the lift's return rather than on the append would have credited the same
    optimization twice, which is exactly the kind of double count the
    reconciliation exists to catch.
    """
    with session_scope(session_dir):
        coord = _coord(session_dir)
        coord._lift_to_current_best("explore", 1100.0, {"name": "same"})
        coord._lift_to_current_best("explore", 1200.0, {"name": "same"})

        assert len(coord.shared_state.optimization_stack) == 1
        assert len(_rows()) == 1


def test_the_lift_carries_the_backend_onto_the_row(session_dir):
    """Kernel adoptions reach the per-backend split through the stack entry.

    Which backend authored a kernel win is passed to the lift as
    ``entry_extra`` by the promoting lane, so the row can report it without the
    recorder knowing anything about kernel backends.
    """
    with session_scope(session_dir):
        coord = _coord(session_dir)
        coord._lift_to_current_best(
            "gemm_tuning",
            1150.0,
            {"name": "tuned-gemm"},
            entry_extra={"backend": "geak", "kernel_id": "k-1"},
        )

        row = _rows()[0]
        assert row["source"] == stack_event.SOURCE_KERNEL
        assert row["backend"] == "geak"
        assert row["kernel_id"] == "k-1"


def test_a_real_session_validation_records_the_whole_stack_figure(session_dir):
    """``_update_cumulative_gain_validated`` is the other production writer."""
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1000.0)
        coord._lift_to_current_best("explore", 1100.0, {"name": "first"})
        coord._update_cumulative_gain_validated(1100.0, {"output_throughput": 1100.0})

        ext, _status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
        settled = ext["validations"]["settled"]
        assert settled["stack_len"] == 1
        assert settled["validated_gain_pct"] == pytest.approx(10.0)
        assert ext["validations"]["at_head"] is True
        # The whole-stack figure and the ledger agree, which is the outcome
        # worth pinning: the two were measured independently.
        assert ext["reconciliation_gap_pct"] == pytest.approx(0.0)


def test_recording_failure_does_not_refuse_the_adoption(session_dir, monkeypatch):
    """The stack outranks its own record: a broken sink must not lose a KEEP."""
    with session_scope(session_dir):
        coord = _coord(session_dir)
        monkeypatch.setattr(stack_event, "_sink", lambda: (_ for _ in ()).throw(RuntimeError("spool down")))

        assert coord._lift_to_current_best("explore", 1100.0, {"name": "kept"}) is True
        assert len(coord.shared_state.optimization_stack) == 1
