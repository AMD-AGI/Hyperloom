# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--max-power-w`` (total) and ``--max-per-gpu-power-w`` ride the latency budget's veto on every KEEP point."""

from __future__ import annotations

import argparse

import pytest

from hyperloom.common.perf_metric import (
    VERDICT_KEEP,
    VERDICT_REVERT,
    constraint_veto_reason,
    power_veto_reason,
)
from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.state.shared_state import SharedState, resolve_graded_comparison


class TestPredicate:
    def test_a_total_over_the_budget_is_vetoed(self):
        assert power_veto_reason({"4": 700.0, "5": 712.0}, total_budget_w=1400.0) == "power_budget_exceeded"

    def test_the_budget_itself_passes(self):
        assert power_veto_reason({"4": 700.0, "5": 700.0}, total_budget_w=1400.0) == ""

    def test_a_card_over_its_own_limit_is_vetoed_inside_the_total(self):
        reason = power_veto_reason({"4": 760.0, "5": 500.0}, total_budget_w=1400.0, per_gpu_budget_w={"4": 700.0})
        assert reason == "gpu_power_budget_exceeded"

    def test_an_unlisted_card_is_bound_only_by_the_total(self):
        assert power_veto_reason({"4": 600.0, "5": 790.0}, total_budget_w=1400.0, per_gpu_budget_w={"4": 700.0}) == ""

    def test_a_listed_card_that_did_not_serve_is_not_judged(self):
        assert power_veto_reason({"5": 650.0}, total_budget_w=1400.0, per_gpu_budget_w={"4": 300.0}) == ""

    def test_per_gpu_limits_alone_are_a_budget(self):
        assert power_veto_reason({"4": 760.0}, per_gpu_budget_w={"4": 700.0}) == "gpu_power_budget_exceeded"
        assert power_veto_reason({"4": 650.0}, per_gpu_budget_w={"4": 700.0}) == ""

    @pytest.mark.parametrize(
        "missing", [None, {}, 700.0, {"4": "700"}, {"4": float("nan")}, {"4": float("inf")}, {"4": True}]
    )
    def test_an_unmeasured_round_is_refused_not_admitted(self, missing):
        assert power_veto_reason(missing, total_budget_w=1400.0) == "power_unmeasured"

    @pytest.mark.parametrize("off", [0.0, None])
    def test_no_budget_vetoes_nothing(self, off):
        assert power_veto_reason(None, total_budget_w=off) == ""
        assert power_veto_reason(None, total_budget_w=off, per_gpu_budget_w={}) == ""

    def test_latency_and_power_share_one_channel(self):
        state = SharedState(latency_budget_ms=250.0, power_budget_w=700.0)
        over = {"4": 400.0, "5": 400.0}
        assert constraint_veto_reason({"e2el_mean_ms": 900.0, "gpu_power_by_gpu_w": over}, state) == (
            "latency_budget_exceeded"
        )
        assert constraint_veto_reason({"e2el_mean_ms": 100.0, "gpu_power_by_gpu_w": over}, state) == (
            "power_budget_exceeded"
        )

    def test_the_veto_reads_the_per_gpu_limits_from_state(self):
        state = SharedState(power_budget_w=1400.0, power_budget_per_gpu_w={"4": 600.0})
        assert constraint_veto_reason({"gpu_power_by_gpu_w": {"4": 650.0, "5": 650.0}}, state) == (
            "gpu_power_budget_exceeded"
        )

    def test_a_round_with_only_the_mean_counts_as_unmeasured(self):
        state = SharedState(power_budget_w=700.0)
        assert constraint_veto_reason({"gpu_power_avg_w": 300.0}, state) == "power_unmeasured"


class TestVerdict:
    def _state(self) -> SharedState:
        return SharedState(baseline_tput=1000.0, power_budget_w=1400.0)

    def test_a_throughput_win_over_the_power_budget_reverts(self):
        graded = resolve_graded_comparison(
            self._state(),
            {"output_throughput": 1200.0, "gpu_power_by_gpu_w": {"4": 712.0, "5": 712.0}},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_REVERT
        assert graded.veto_reason == "power_budget_exceeded"

    def test_the_same_win_inside_the_budget_keeps(self):
        graded = resolve_graded_comparison(
            self._state(),
            {"output_throughput": 1200.0, "gpu_power_by_gpu_w": {"4": 650.0, "5": 650.0}},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_KEEP

    def test_a_loss_carries_no_power_veto(self):
        graded = resolve_graded_comparison(
            self._state(),
            {"output_throughput": 800.0, "gpu_power_by_gpu_w": {"4": 812.0, "5": 812.0}},
            keep_threshold_pct=1.0,
        )
        assert graded.verdict == VERDICT_REVERT
        assert graded.veto_reason == ""


class TestMeasurementCarriesPower:
    def test_the_rounds_gpu_monitor_power_lands_on_the_measurement(self):
        from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

        report = {
            "success": True,
            "throughput": {"output_throughput": 1200.0},
            "gpu_monitor": [
                {"sample_count": 10, "power_watts": {"avg": 600.0, "max": 700.0}},
                {"sample_count": 30, "power_watts": {"avg": 700.0, "max": 760.0}},
            ],
        }
        measurement = extract_benchmark_measurement(report)
        assert measurement["gpu_power_avg_w"] == pytest.approx(675.0)
        assert measurement["gpu_power_by_gpu_w"] is None, "the report's mean does not say which card drew what"

    def test_no_telemetry_is_absent_not_zero(self):
        from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

        assert extract_benchmark_measurement({"success": True, "throughput": {}})["gpu_power_avg_w"] is None

    def test_variant_result_carries_it(self):
        v = VariantResult(
            name="v", extra_server_args="", extra_envs={}, status="succeeded", gpu_power_by_gpu_w={"4": 650.0}
        )
        assert v.to_dict()["gpu_power_by_gpu_w"] == {"4": 650.0}

    def test_the_integrate_lift_carries_it(self):
        from hyperloom.orchestrator.measurement.integrate_performance import integrate_measurement_fields

        v = VariantResult(
            name="v", extra_server_args="", extra_envs={}, status="succeeded", gpu_power_by_gpu_w={"4": 650.0}
        )
        assert integrate_measurement_fields(v.to_dict())["gpu_power_by_gpu_w"] == {"4": 650.0}


class TestIntegrateDecision:
    def test_the_shared_integrate_decision_reverts_an_over_budget_gain(self):
        from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

        performance = assess_integrate_performance(
            SharedState(baseline_tput=1000.0, power_budget_w=700.0),
            {"output_throughput": 1200.0, "gpu_power_by_gpu_w": {"4": 812.0}},
            base_tput=1000.0,
            keep_threshold_pct=1.0,
            stack_incremental_keep_threshold_pct=0.5,
        )
        assert performance.decision == "REVERT"
        assert performance.graded.veto_reason == "power_budget_exceeded"

    @pytest.mark.parametrize(("power_w", "decision"), [(812.0, "REVERT"), (None, "REVERT"), (650.0, "KEEP")])
    def test_a_stacked_layer_below_the_graded_bar_still_honours_the_power_budget(self, power_w, decision):
        """+0.7% on a stacked integrate KEEPs on the 0.5% stack bar while the 1.0% graded verdict carries no veto."""
        from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

        state = SharedState(baseline_tput=1000.0, power_budget_w=700.0)
        state.current_best = {"action": "integrate", "tput": 1000.0}
        state.optimization_stack = [{"action": "integrate"}]
        measurement = {"output_throughput": 1007.0}
        if power_w is not None:
            measurement["gpu_power_by_gpu_w"] = {"4": power_w}

        performance = assess_integrate_performance(
            state,
            measurement,
            base_tput=1000.0,
            keep_threshold_pct=1.0,
            stack_incremental_keep_threshold_pct=0.5,
        )
        assert performance.stack_positive_keep is True
        assert performance.graded.veto_reason == ""
        assert performance.decision == decision


class TestCli:
    def _parse(self, *argv: str) -> argparse.Namespace:
        from hyperloom.inference_optimizer.cli.parser import _build_parser

        return _build_parser().parse_args(["optimize", "--model", "/m", *argv])

    def test_the_flags_parse(self):
        args = self._parse("--max-power-w", "700", "--gpu-power-cap-w", "1000", "--gpu-perf-level", "determinism")
        assert (args.max_power_w, args.gpu_power_cap_w, args.gpu_perf_level) == (700.0, 1000.0, "determinism")

    @pytest.mark.parametrize("bad", ["700W", "0", "-5", "nan"])
    def test_an_unusable_ceiling_stops_the_launch(self, bad):
        with pytest.raises(SystemExit) as exc:
            self._parse("--max-power-w", bad)
        assert exc.value.code == 2

    def test_a_changed_budget_on_resume_is_refused(self):
        from hyperloom.inference_optimizer.cli.bootstrap import power_budget_resume_conflict

        assert "--max-power-w 600" in power_budget_resume_conflict(SharedState(power_budget_w=700.0), 600.0)
        assert power_budget_resume_conflict(SharedState(power_budget_w=700.0), None) == ""

    def test_per_gpu_limits_parse_into_a_map(self):
        args = self._parse(
            "--max-power-w", "2800", "--max-per-gpu-power-w", "4", "700", "--max-per-gpu-power-w", "5", "650"
        )
        assert args.max_per_gpu_power_w == {"4": 700.0, "5": 650.0}
        assert self._parse().max_per_gpu_power_w is None

    @pytest.mark.parametrize(
        "bad",
        [("x", "700"), ("-1", "700"), ("4", "0"), ("4", "700W"), ("4", "nan")],
        ids=["non-integer-id", "negative-id", "zero-watts", "unit-suffix", "nan"],
    )
    def test_an_unusable_per_gpu_limit_stops_the_launch(self, bad):
        with pytest.raises(SystemExit) as exc:
            self._parse("--max-per-gpu-power-w", *bad)
        assert exc.value.code == 2

    def test_a_repeated_gpu_stops_the_launch(self):
        with pytest.raises(SystemExit) as exc:
            self._parse("--max-per-gpu-power-w", "4", "700", "--max-per-gpu-power-w", "4", "650")
        assert exc.value.code == 2

    def test_changed_per_gpu_limits_on_resume_are_refused(self):
        from hyperloom.inference_optimizer.cli.bootstrap import per_gpu_power_budget_resume_conflict

        state = SharedState(power_budget_per_gpu_w={"4": 700.0})
        assert "GPU 4 700 W" in per_gpu_power_budget_resume_conflict(state, {"4": 650.0})
        assert "no per-GPU limits" in per_gpu_power_budget_resume_conflict(SharedState(), {"4": 650.0})
        assert per_gpu_power_budget_resume_conflict(state, {"4": 700.0}) == ""
        assert per_gpu_power_budget_resume_conflict(state, None) == ""


class TestPerGpuLaunchCheck:
    @staticmethod
    def _error(per_gpu, *, total_w=2800.0, available_gpus=frozenset({4, 5, 6, 7})):
        from hyperloom.inference_optimizer.cli.bootstrap import per_gpu_power_budget_error

        available = set(available_gpus) if available_gpus is not None else None
        return per_gpu_power_budget_error(per_gpu, total_w=total_w, available_gpus=available)

    def test_limits_inside_the_total_on_visible_cards_pass(self):
        assert self._error({"4": 700.0, "5": 700.0, "6": 700.0, "7": 700.0}) == ""
        assert self._error({"4": 900.0}, total_w=None) == "", "per-GPU limits alone need no total"
        assert self._error(None) == ""

    def test_limits_that_add_up_past_the_total_refuse(self):
        assert "add up to 2900 W, more than --max-power-w 2800 W" in self._error(
            {"4": 800.0, "5": 700.0, "6": 700.0, "7": 700.0}
        )

    def test_a_gpu_the_session_cannot_use_refuses(self):
        assert "GPU(s) 0, 1" in self._error({"0": 300.0, "1": 300.0, "4": 300.0})

    def test_an_unknown_gpu_set_refuses(self):
        assert "cannot be checked" in self._error({"4": 700.0}, available_gpus=None)


class TestUnmeasurableBudget:
    """A budget that no round could measure is refused up front instead of costing a baseline."""

    @staticmethod
    def _error(*, total_w=2800.0, per_gpu_w=None, nodes=1):
        from hyperloom.inference_optimizer.cli.bootstrap import power_budget_unmeasurable_error

        return power_budget_unmeasurable_error(total_w=total_w, per_gpu_w=per_gpu_w, nodes=nodes)

    @pytest.fixture
    def sampler_available(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _gpu_power

        monkeypatch.delenv(_gpu_power.GPU_POWER_ENV, raising=False)
        monkeypatch.setattr(_gpu_power.shutil, "which", lambda name: "/usr/bin/amd-smi")

    def test_a_measurable_budget_passes(self, sampler_available):
        assert self._error() == ""
        assert self._error(total_w=None, per_gpu_w={"4": 700.0}) == ""

    @pytest.mark.parametrize(("total_w", "per_gpu_w"), [(2800.0, None), (None, {"4": 700.0})])
    def test_a_multi_node_session_refuses_either_budget(self, sampler_available, total_w, per_gpu_w):
        assert "multi-node" in self._error(total_w=total_w, per_gpu_w=per_gpu_w, nodes=2)

    def test_sampling_turned_off_refuses(self, sampler_available, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GPU_POWER_SAMPLING", "0")
        assert "HYPERLOOM_GPU_POWER_SAMPLING=0" in self._error()

    def test_no_amd_smi_refuses(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _gpu_power

        monkeypatch.delenv(_gpu_power.GPU_POWER_ENV, raising=False)
        monkeypatch.setattr(_gpu_power.shutil, "which", lambda name: None)
        assert "amd-smi is not on PATH" in self._error(total_w=None, per_gpu_w={"4": 700.0})

    @pytest.mark.parametrize("off", [None, 0.0])
    def test_no_budget_needs_no_sampler(self, monkeypatch, off):
        monkeypatch.setenv("HYPERLOOM_GPU_POWER_SAMPLING", "0")
        assert self._error(total_w=off, per_gpu_w={}, nodes=2) == ""


def test_the_prompt_states_the_budget_and_the_cap():
    state = SharedState(
        power_budget_w=700.0,
        gpu_power_settings={"observed": {"0": {"power_cap_w": 1000.0}, "1": {"power_cap_w": 1000.0}}},
    )
    line = state.to_power_budget_summary()
    assert "\n" not in line
    assert "700 W total across the serving GPUs, cards capped at 1000 W" in line
    assert SharedState().to_power_budget_summary() == ""


def test_the_prompt_states_the_per_gpu_limits():
    line = SharedState(
        power_budget_w=2800.0, power_budget_per_gpu_w={"10": 650.0, "4": 700.0}
    ).to_power_budget_summary()
    assert line.startswith("2800 W total across the serving GPUs; per-GPU GPU 4 700 W, GPU 10 650 W:")
    assert "gpu_power_budget_exceeded" in line
    assert SharedState(power_budget_per_gpu_w={"4": 700.0}).to_power_budget_summary().startswith("per-GPU GPU 4 700 W:")


def test_the_stop_reason_is_in_the_closed_vocabulary():
    from hyperloom.orchestrator.phases.machine_state import is_valid_stop_reason

    assert is_valid_stop_reason("baseline_over_power_budget")
