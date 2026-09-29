# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native totals keep their scoring value while displays use measured GPU counts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_outcome
from hyperloom.inference_optimizer.breakdown.exporter import write_minimal_final_report
from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import baseline_event_id, make_baseline_recorder
from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.breakdown.reporters._renderers import baseline, final
from hyperloom.inference_optimizer.breakdown.reporters.cross_section import build_global_facts
from hyperloom.inference_optimizer.cli.bootstrap import _print_final_summary
from hyperloom.inference_optimizer.performance_display import (
    format_measurement_metric,
    format_session_metric,
    throughput_fields,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.actions.executors import report
from hyperloom.orchestrator.state.shared_state import SharedState

AGGREGATE = 286.78509


def _native_state(gpu_count=4):
    perf = {} if gpu_count is None else {"agentx_gpu_count": gpu_count}
    return SharedState(
        session_id="native-display",
        framework="sglang",
        benchmark_mode="agentx",
        agentx_epoch=2,
        tp=4,
        ep=4,
        baseline_tput=AGGREGATE,
        baseline_perf=perf,
        current_best={"tput": AGGREGATE, "action": "baseline", **perf},
    )


@pytest.mark.parametrize("gpu_count,expected", [(4, "71.69627 tok/s/GPU"), (None, "286.78509 tok/s (aggregate)")])
def test_native_display_uses_physical_count_without_changing_measurements(gpu_count, expected, monkeypatch):
    state = _native_state(gpu_count)
    # TP and EP overlap on the same four GPUs; ambient topology must never override a recorded measurement.
    monkeypatch.setenv("TP", "16")
    monkeypatch.setenv("HYPERLOOM_AGENTX_GPU_COUNT", "16")
    result = {"output_throughput": AGGREGATE, "native_agentx_report": True, **state.baseline_perf}
    before = deepcopy(result)

    assert format_session_metric(state, state.baseline_tput, precision=5) == expected
    assert format_measurement_metric("sglang", result, precision=5) == expected
    assert result == before
    assert state.baseline_tput == AGGREGATE
    assert state.current_best["tput"] == AGGREGATE


@pytest.mark.parametrize("count", [0, -4, 1.5, True, "4", float("inf"), float("nan")])
def test_native_invalid_topology_remains_explicitly_aggregate(count):
    fields = throughput_fields(AGGREGATE, "sglang", native_agentx=True, gpu_count=count)
    assert fields["throughput_tok_s_per_gpu"] is None
    assert fields["throughput_tok_s"] == AGGREGATE
    assert fields["throughput_unit"] == "tok/s (aggregate)"


@pytest.mark.parametrize("mode,epoch", [("agentx", 1), ("synthetic", 0)])
def test_legacy_and_synthetic_display_keep_the_existing_throughput(mode, epoch):
    state = _native_state()
    state.benchmark_mode, state.agentx_epoch = mode, epoch
    assert format_session_metric(state, AGGREGATE, precision=2) == "286.79 tok/s/GPU"
    assert format_measurement_metric("sglang", {"output_throughput": AGGREGATE, "agentx_gpu_count": 4}) == (
        "286.8 tok/s/GPU"
    )


def test_scriptable_primary_metric_stays_latency():
    state = SharedState(session_id="image-display", framework="xdit", baseline_tput=2.0)
    assert format_session_metric(state, 2.0) == "500.0 ms"


@pytest.mark.parametrize("gpu_count,expected", [(4, "71.7 tok/s/GPU"), (None, "286.8 tok/s (aggregate)")])
def test_native_cli_mission_and_final_reports_share_correct_units(tmp_path, capsys, gpu_count, expected):
    state = _native_state(gpu_count)
    state.save(tmp_path)
    _print_final_summary(state, "time_exhausted")
    assert expected in capsys.readouterr().out
    assert expected in state.to_mission_summary()
    assert expected in state._format_current_best_for_mission()

    summary = report._build_summary_dict(state, {}, [])
    assert summary["baseline_tput"] == AGGREGATE
    assert summary["current_best"]["tput"] == AGGREGATE
    assert report._format_md(summary).count(expected) == 2
    emergency = write_minimal_final_report(tmp_path).read_text()
    expected_precise = "71.70 tok/s/GPU" if gpu_count else "286.79 tok/s (aggregate)"
    assert emergency.count(expected_precise) == 2


@pytest.mark.parametrize("gpu_count", [4, None])
def test_native_breakdown_records_and_renders_matching_units(tmp_path, gpu_count):
    state = _native_state(gpu_count)
    result = {
        "status": "succeeded",
        "output_throughput": AGGREGATE,
        "native_agentx_report": True,
        **state.baseline_perf,
    }
    with session_scope(tmp_path):
        recorder = make_baseline_recorder(
            make_sink(baseline_event_id("prelude", 0), producer="orchestrator"),
            task_id="baseline",
            task_kind="baseline",
            framework="sglang",
            establishes_quality_ref=True,
        )
        recorder.finish(result)
    timeline = read_timeline_events(tmp_path)
    timeline.append(
        {
            "type": "stack",
            "ext": {
                "validations": {"settled": {"perf": state.baseline_perf}, "at_head": True},
                "validated_total_gain_pct": 0.0,
            },
        }
    )
    outcome = collect_v6_outcome(
        session={"stop_reason": "time_exhausted"},
        close={"final_recipe": {"throughput": AGGREGATE}},
        state=state.to_dict(),
        timeline=timeline,
    )
    expected = "71.70 tok/s/GPU" if gpu_count else "286.79 tok/s (aggregate)"
    for key in ("baseline", "final"):
        measurement = outcome[key]
        assert measurement["throughput_tok_s"] == AGGREGATE
        if gpu_count:
            assert measurement["throughput_tok_s_per_gpu"] == pytest.approx(AGGREGATE / 4)
            assert measurement["perf"]["agentx_gpu_count"] == 4
        else:
            assert measurement["throughput_tok_s_per_gpu"] is None
    breakdown = {
        "metadata": {"task_config": {"framework_name": "sglang"}},
        "timeline": timeline,
        "outcome": outcome,
    }
    rendered = [baseline.render(breakdown), final.render(breakdown)]
    assert all(any(expected in fact for fact in section.key_facts) for section in rendered)
    assert expected in rendered[0].markdown_block  # The attempts table states each row's units too.
    assert build_global_facts(breakdown, rendered).headline.count(expected) == 2
    assert result["output_throughput"] == AGGREGATE


def test_breakdown_headline_does_not_equate_different_units_or_interactivity_gain():
    baseline_fields = throughput_fields(AGGREGATE, "sglang", native_agentx=True, gpu_count=4)
    final_fields = throughput_fields(AGGREGATE, "sglang", native_agentx=True)
    breakdown = {
        "metadata": {"task_config": {"framework_name": "sglang"}},
        "outcome": {
            "baseline": baseline_fields,
            "final": {**final_fields, "gain_pct": 3.0, "graded_on": "e2e_norm_intvty_p50"},
        },
    }
    headline = build_global_facts(breakdown, []).headline
    assert "71.70 tok/s/GPU" in headline
    assert "286.79 tok/s (aggregate)" in headline
    assert "= +3.00%" not in headline
    assert "validated gain: +3.00% (e2e_norm_intvty_p50)" in headline
