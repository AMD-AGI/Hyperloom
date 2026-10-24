# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``reports/kernel_optimization_summary.json``: a projection of the KERNEL timeline's candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    REBENCH_NO_PROMOTE,
    REBENCH_VALIDATED,
    ROUTE_FORGE,
    ROUTE_GEAK,
    make_kernel_recorder,
    record_integrate_verdict,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import (
    IntegrateVerdict,
    KernelRewriteRun,
    LaneRunFields,
)
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.kernel.attempt_summary import SCHEMA_VERSION, build_kernel_optimization_summary
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    with session_scope(tmp_path):
        yield tmp_path


def _state(*, top15: list[dict[str, Any]] | None = None) -> SharedState:
    state = SharedState(session_id="20260529T104050Z", model_name="test/model")
    state.last_trace_analyze = {"kernel_roofline_top15": list(top15 or [])}
    return state


def _top(kid: str, name: str, gpu_pct: float, efficiency: float = 40.0) -> dict[str, Any]:
    return {
        "kernel_id": kid,
        "name": name,
        "source_file": f"/repo/{name}.py",
        "gpu_pct": gpu_pct,
        "efficiency_percent": efficiency,
        "bound_type": "memory-bound",
    }


def _forge_visit(macro_cycle: int = 0):
    recorder = make_kernel_recorder(macro_cycle=macro_cycle, route=ROUTE_FORGE)
    recorder.begin(tput_before=1000.0)
    return recorder


def _rewrite(recorder, run_id: str, kernel_id: str, *, decision: str, status: str = "success", **fields: Any):
    recorder.record_kernel_rewrite(
        KernelRewriteRun(
            lane=LaneRunFields(run_id=run_id, status=status, micro_decision=decision, **fields),
            kernel_id=kernel_id,
            kernel_name=kernel_id,
        )
    )


def _geak_visit(*, rebench: str | None, specs: list[dict[str, Any]]):
    recorder = make_kernel_recorder(macro_cycle=1, route=ROUTE_GEAK)
    recorder.begin(tput_before=900.0)
    recorder.record_geak_claim({"self_reported_gain_pct": 9.0, "geak_status": "ok"}, specs=specs)
    if rebench is not None:
        recorder.record_geak_rebench_attempt(
            max_attempts=4,
            attempt_id="geak-rb-1",
            idempotency_key="geak-rb-1",
            task_id="geak-rb-1",
            base_tput=900.0,
            measured_tput=950.0,
            decision=rebench,
            status="settled",
        )
    recorder.finish(tput_after=950.0)


def test_a_session_with_no_kernel_visit_skips_every_lane(tmp_path: Path) -> None:
    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert out["schema_version"] == SCHEMA_VERSION
    assert out["kernel_opt_outcome"] == "skip"
    assert out["totals"]["attempted"] == 0
    assert out["by_kernel"] == []
    skipped = {"attempted": 0, "claimed": 0, "success": 0, "unvalidated": 0, "failed": 0, "outcome": "skip"}
    assert out["lane_totals"] == {"source_level": skipped, "geak": skipped, "gemm_tuning": skipped}
    assert out["top_takeaways"][0] == "No kernel candidates were dispatched by any lane."


def test_top_level_shape_is_stable(tmp_path: Path) -> None:
    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert set(out) == {
        "schema_version",
        "session_id",
        "model_name",
        "cumulative_gain_validated_pct",
        "kernel_opt_outcome",
        "totals",
        "failure_reason_breakdown",
        "field_glossary",
        "by_kernel",
        "top_takeaways",
        "lane_totals",
    }
    assert {"claimed", "outcome", "settled_by", "gain_pct"} <= set(out["field_glossary"])


def test_forge_candidates_carry_the_verdict_that_settled_them(tmp_path: Path) -> None:
    recorder = _forge_visit()
    _rewrite(recorder, "a-keep", "k001", decision="KEEP", integrate_ref="int-1")
    _rewrite(recorder, "a-revert", "k002", decision="REVERT")
    _rewrite(recorder, "a-crash", "k003", decision="", status="failed", error_class="compile_failed")
    record_integrate_verdict(
        macro_cycle=0, verdict=IntegrateVerdict(integration_id="int-1", kernel_id="k001", decision="KEEP", gain_pct=3.5)
    )
    recorder.finish(tput_after=1035.0)

    out = build_kernel_optimization_summary(_state(top15=[_top("k001", "fused_moe", 30.0)]), tmp_path)

    rows = {row["kernel_id"]: row for row in out["by_kernel"]}
    assert rows["k001"]["outcome"] == "adopted"
    assert rows["k001"]["settled_by"] == "integrate"
    assert rows["k001"]["gain_pct"] == 3.5
    assert rows["k001"]["gpu_pct"] == 30.0
    assert rows["k002"]["outcome"] == "rejected"
    assert rows["k003"]["outcome"] == "failed"
    assert out["totals"] == {
        "top_candidates": 1,
        "attempted": 3,
        "claimed": 1,
        "adopted": 1,
        "rejected": 1,
        "failed": 1,
        "needs_review": 0,
        "not_dispatched": 0,
    }
    assert out["failure_reason_breakdown"] == {"compile_failed": 1, "unclassified": 1}
    assert out["lane_totals"]["source_level"]["outcome"] == "success"
    assert out["kernel_opt_outcome"] == "success"


def test_geak_acceptances_count_as_claims_until_the_rebench_adopts_them(tmp_path: Path) -> None:
    specs = [{"short_name": "rms_norm", "op_kind": "norm", "e2e_delta_pct": 3.05}]
    _geak_visit(rebench=REBENCH_NO_PROMOTE, specs=specs)

    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert out["lane_totals"]["geak"] == {
        "attempted": 1,
        "claimed": 1,
        "success": 0,
        "unvalidated": 0,
        "failed": 1,
        "outcome": "fail",
    }
    assert out["by_kernel"][0]["settled_by"] == "rebench"


def test_a_geak_acceptance_the_rebench_validated_is_a_success(tmp_path: Path) -> None:
    _geak_visit(rebench=REBENCH_VALIDATED, specs=[{"short_name": "rms_norm", "op_kind": "norm"}])

    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert out["lane_totals"]["geak"]["success"] == 1
    assert out["kernel_opt_outcome"] == "success"


def test_an_acceptance_no_rebench_ruled_on_is_unvalidated(tmp_path: Path) -> None:
    _geak_visit(rebench=None, specs=[{"short_name": "rms_norm", "op_kind": "norm", "e2e_delta_pct": 12.0}])

    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert out["lane_totals"]["geak"]["success"] == 0
    assert out["lane_totals"]["geak"]["unvalidated"] == 1
    assert out["kernel_opt_outcome"] == "unvalidated"


def test_gemm_tuning_counts_in_its_own_lane(tmp_path: Path) -> None:
    recorder = _forge_visit()
    recorder.record_gemm_tuning_run(run_id="gemm-c0", status="ok", micro_decision="REVERT", tuner="hipblaslt")
    recorder.finish(tput_after=1000.0)

    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert out["lane_totals"]["gemm_tuning"]["attempted"] == 1
    assert out["lane_totals"]["source_level"]["attempted"] == 0
    assert out["top_takeaways"][0] == "No source-level kernel rewrites were attempted; other kernel lanes did run."


def test_candidates_from_every_visit_are_counted(tmp_path: Path) -> None:
    for cycle in (0, 1):
        recorder = _forge_visit(cycle)
        _rewrite(recorder, f"a-{cycle}", f"k00{cycle}", decision="REVERT")
        recorder.finish(tput_after=1000.0)

    out = build_kernel_optimization_summary(_state(), tmp_path)

    assert [row["macro_cycle"] for row in out["by_kernel"]] == [0, 1]
    assert out["totals"]["attempted"] == 2


def test_a_visit_still_in_flight_contributes_no_candidates(tmp_path: Path) -> None:
    _rewrite(_forge_visit(), "a-1", "k001", decision="KEEP")

    assert build_kernel_optimization_summary(_state(), tmp_path)["by_kernel"] == []


def test_a_recipe_id_joins_the_trace_row_by_operator_name(tmp_path: Path) -> None:
    recorder = _forge_visit()
    _rewrite(recorder, "a-1", "kernel:forge-loop:fused_moe:sglang:0.5.17:triton:mi355x", decision="KEEP")
    recorder.finish(tput_after=1000.0)

    out = build_kernel_optimization_summary(_state(top15=[_top("k004", "FusedMoeKernel", 22.0)]), tmp_path)

    assert out["by_kernel"][0]["trace_kernel_id"] == "k004"
    assert out["by_kernel"][0]["gpu_pct"] == 22.0


def test_the_hottest_kernel_nothing_was_adopted_for_is_the_missed_opportunity(tmp_path: Path) -> None:
    recorder = _forge_visit()
    _rewrite(recorder, "a-1", "k001", decision="KEEP")
    recorder.finish(tput_after=1000.0)
    top15 = [_top("k001", "fused_moe", 40.0), _top("k002", "rms_norm", 12.0, efficiency=8.0)]

    out = build_kernel_optimization_summary(_state(top15=top15), tmp_path)

    assert out["top_takeaways"][-1] == (
        "Highest-impact missed opportunity: rms_norm at 12.0% GPU time, 8.0% efficiency — no candidate for it was adopted."
    )


def test_geak_absent_backends_surface_as_a_takeaway(tmp_path: Path) -> None:
    state = _state()
    state.geak_result = {
        "status": "validated_no_win",
        "absent_backends": {"hipblaslt_bench_cli": "hipblaslt-bench not on PATH", "ck": "ckProfiler not on PATH"},
    }

    out = build_kernel_optimization_summary(state, tmp_path)

    assert out["top_takeaways"][-1].startswith("GEAK ran without these backends (unavailable): ck, hipblaslt_bench_cli")
