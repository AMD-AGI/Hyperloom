# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stack ledger against the real production writers, not a stand-in.

``test_sbd_v6_stack_ledger.py`` pins what the ledger computes; these pin that the
orchestrator feeds it. ``throughput_before`` exists only inside
``_lift_to_current_best``, so a test that supplies its own has assumed the claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.perf_metric import GRADED_INTVTY
from hyperloom.inference_optimizer.breakdown.recorder import stack_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import stack_event_parts
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
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
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1000.0)
        coord._lift_to_current_best("explore", 1100.0, {"name": "first"})
        coord._lift_to_current_best("integrate", 1250.0, {"name": "second"})

        ext, status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
        assert ext["attributed_gain_pct"] == 25.0
        assert ext["chain_total_gain_pct"] == 25.0
        assert ext["unattributed_gain_pct"] == 0.0
        assert status == stack_event.STATUS_SUCCEEDED


def test_a_degraded_agentx_lift_is_refused(session_dir, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1000.0)
        coord.shared_state.benchmark_mode = "agentx"
        coord.shared_state.grading = {"objective": GRADED_INTVTY, "noise_pct": 5.0}
        coord.shared_state.current_best = {
            "action": "baseline",
            "tput": 1000.0,
            "output_throughput": 1000.0,
            "total_throughput": 20000.0,
            "extra_server_args": "",
            "extra_envs": {},
        }

        assert (
            coord._lift_to_current_best(
                "explore",
                1100.0,
                {
                    "name": "degraded-winner",
                    "output_throughput": 1100.0,
                    "total_throughput": 22000.0,
                    "e2e_norm_intvty_p90": 30.0,
                },
            )
            is False
        )
        assert _rows() == []


def test_a_refused_lift_records_nothing(session_dir):
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1200.0)

        assert coord._lift_to_current_best("explore", 1100.0, {"name": "loser"}) is False
        assert _rows() == []


def test_an_already_stacked_config_is_not_recorded_twice(session_dir):
    with session_scope(session_dir):
        coord = _coord(session_dir)
        coord._lift_to_current_best("explore", 1100.0, {"name": "same"})
        coord._lift_to_current_best("explore", 1200.0, {"name": "same"})

        assert len(coord.shared_state.optimization_stack) == 1
        assert len(_rows()) == 1


def test_the_lift_carries_the_backend_onto_the_row(session_dir):
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


@pytest.mark.parametrize(
    "accepted_config,operation_kind",
    [
        ({"flags": "--attention-backend aiter"}, "backend"),
        ({"flags": "--max-num-batched-tokens 4096"}, "param"),
        ({"env_map": {"VLLM_ROCM_USE_AITER": "1"}}, "env"),
    ],
)
@pytest.mark.parametrize("overlay_loaded", [None, False])
def test_geak_config_promotion_records_engine_without_kernel_credit(
    session_dir, accepted_config, operation_kind, overlay_loaded
):
    with session_scope(session_dir):
        coord = _coord(session_dir)
        result = {
            "status": "ok",
            "accepted_config": accepted_config,
            "accepted_kernels": ["unloaded_candidate"] if overlay_loaded is False else [],
        }

        assert coord._promote_geak_from_candidate(result, measured_tput=1100.0, overlay_loaded=overlay_loaded)

        entry = coord.shared_state.optimization_stack[0]
        assert entry["backend"] == "geak"
        assert entry["lever_kind"] == "config"
        assert entry["operation_kind"] == operation_kind
        assert not entry.get("accepted_kernels")
        ext, _status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
        row = ext["adoptions"]["rows"][0]
        assert row["source"] == stack_event.SOURCE_KERNEL
        assert row["backend"] == "geak"
        assert row["lever_kind"] == "config"
        assert row["operation_kind"] == operation_kind
        assert row["accepted_kernels"] == []
        assert row["kernel_id"] is None
        assert row["contribution_pct"] == pytest.approx(10.0)
        engines = ext["adoptions"]["by_source"][stack_event.SOURCE_KERNEL]["by_backend"]
        assert engines["geak"] == {"count": 1, "total_gain_pct": 10.0, "unmeasured": 0}
        assert engines[stack_event.SOURCE_UNATTRIBUTED]["count"] == 0


def test_geak_proven_kernel_promotion_retains_kernel_identity_and_engine(session_dir):
    with session_scope(session_dir):
        coord = _coord(session_dir)
        result = {"status": "ok", "accepted_kernels": ["loaded_candidate"]}

        assert coord._promote_geak_from_candidate(result, measured_tput=1100.0, overlay_loaded=True)

        row = _rows()[0]
        assert row["backend"] == "geak"
        assert row["lever_kind"] == "kernel"
        assert row["accepted_kernels"] == ["loaded_candidate"]


@pytest.mark.parametrize("measured", [1000.0, 950.0])
def test_refused_geak_config_promotion_records_no_adoption(session_dir, measured):
    with session_scope(session_dir):
        coord = _coord(session_dir)
        result = {"status": "ok", "accepted_config": {"flags": "--attention-backend aiter"}}

        assert not coord._promote_geak_from_candidate(result, measured_tput=measured)

        assert coord.shared_state.optimization_stack == []
        assert _rows() == []
        assert coord.shared_state.geak_result["revalidation_status"] == "no_promote"


def test_a_real_session_validation_records_the_whole_stack_figure(session_dir):
    with session_scope(session_dir):
        coord = _coord(session_dir, baseline=1000.0, anchor=1000.0)
        coord._lift_to_current_best("explore", 1100.0, {"name": "first"})
        coord._update_cumulative_gain_validated(1100.0, {"output_throughput": 1100.0})

        ext, _status = stack_event.assemble_stack_ext(stack_event_parts(), event=stack_event.stack_event_id())
        settled = ext["validations"]["settled"]
        assert settled["stack_len"] == 1
        assert settled["validated_gain_pct"] == pytest.approx(10.0)
        assert ext["validations"]["at_head"] is True
        # The whole-stack figure and the ledger were measured independently.
        assert ext["reconciliation_gap_pct"] == pytest.approx(0.0)


def test_a_spool_that_cannot_be_written_does_not_refuse_the_adoption(session_dir, monkeypatch):
    """The sink drops the row it could not write; the adoption still stands.

    Failed at the write itself rather than by making the writer raise: the sink
    is where recording is allowed to fail quietly, so a fault anywhere else is
    a defect and is meant to surface.
    """
    from hyperloom.inference_optimizer.breakdown.recorder import recorder as recorder_module

    with session_scope(session_dir):
        coord = _coord(session_dir)
        monkeypatch.setattr(
            recorder_module.Recorder,
            "record_upsert_item",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("spool down")),
        )

        assert coord._lift_to_current_best("explore", 1100.0, {"name": "kept"}) is True
        assert len(coord.shared_state.optimization_stack) == 1


def test_an_unreadable_spool_on_finish_does_not_raise(session_dir, monkeypatch):
    """``stack_event.finish`` claims Never raises; the close-time read must keep that."""
    from hyperloom.inference_optimizer.breakdown.recorder import stack_event

    with session_scope(session_dir):
        stack_event.record_adoption(
            stack_index=0,
            entry={"action": "explore", "variant_name": "kept"},
            throughput_before=1000.0,
            throughput_after=1100.0,
            baseline_tput=1000.0,
        )
        monkeypatch.setattr(
            "hyperloom.inference_optimizer.breakdown.recorder.assembler.event_parts",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("spool down")),
        )
        stack_event.finish()
