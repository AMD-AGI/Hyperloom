# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--max-power-w`` rides the latency budget's veto; power settings are asserted and recorded, never set."""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest

from hyperloom.common.gpu_power_settings import (
    GpuPowerSettingsError,
    declared_setting_problems,
    read_gpu_power_settings,
)
from hyperloom.common.perf_metric import (
    VERDICT_KEEP,
    VERDICT_REVERT,
    constraint_veto_reason,
    power_veto_reason,
)
from hyperloom.orchestrator.actions.executors._grid_base import VariantResult
from hyperloom.orchestrator.state.shared_state import SharedState, resolve_graded_comparison


class TestPredicate:
    def test_over_the_ceiling_is_vetoed(self):
        assert power_veto_reason(812.0, 700.0) == "power_budget_exceeded"

    def test_the_ceiling_itself_passes(self):
        assert power_veto_reason(700.0, 700.0) == ""

    @pytest.mark.parametrize("missing", [None, "", "700", float("nan"), 0.0, True])
    def test_an_unmeasured_round_is_refused_not_admitted(self, missing):
        assert power_veto_reason(missing, 700.0) == "power_unmeasured"

    @pytest.mark.parametrize("off", [0.0, None])
    def test_no_budget_vetoes_nothing(self, off):
        assert power_veto_reason(None, off) == ""

    def test_latency_and_power_share_one_channel(self):
        state = SharedState(latency_budget_ms=250.0, power_budget_w=700.0)
        assert constraint_veto_reason({"e2el_mean_ms": 900.0, "gpu_power_avg_w": 800.0}, state) == (
            "latency_budget_exceeded"
        )
        assert constraint_veto_reason({"e2el_mean_ms": 100.0, "gpu_power_avg_w": 800.0}, state) == (
            "power_budget_exceeded"
        )


class TestVerdict:
    def _state(self) -> SharedState:
        return SharedState(baseline_tput=1000.0, power_budget_w=700.0)

    def test_a_throughput_win_over_the_power_ceiling_reverts(self):
        graded = resolve_graded_comparison(
            self._state(), {"output_throughput": 1200.0, "gpu_power_avg_w": 812.0}, keep_threshold_pct=1.0
        )
        assert graded.verdict == VERDICT_REVERT
        assert graded.veto_reason == "power_budget_exceeded"

    def test_the_same_win_inside_the_ceiling_keeps(self):
        graded = resolve_graded_comparison(
            self._state(), {"output_throughput": 1200.0, "gpu_power_avg_w": 650.0}, keep_threshold_pct=1.0
        )
        assert graded.verdict == VERDICT_KEEP

    def test_a_loss_carries_no_power_veto(self):
        graded = resolve_graded_comparison(
            self._state(), {"output_throughput": 800.0, "gpu_power_avg_w": 812.0}, keep_threshold_pct=1.0
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
        assert extract_benchmark_measurement(report)["gpu_power_avg_w"] == pytest.approx(675.0)

    def test_no_telemetry_is_absent_not_zero(self):
        from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement

        assert extract_benchmark_measurement({"success": True, "throughput": {}})["gpu_power_avg_w"] is None

    def test_variant_result_carries_it(self):
        v = VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded", gpu_power_avg_w=650.0)
        assert v.to_dict()["gpu_power_avg_w"] == 650.0

    def test_the_integrate_lift_carries_it(self):
        from hyperloom.orchestrator.measurement.integrate_performance import integrate_measurement_fields

        v = VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded", gpu_power_avg_w=650.0)
        assert integrate_measurement_fields(v.to_dict())["gpu_power_avg_w"] == 650.0


class TestIntegrateDecision:
    def test_the_shared_integrate_decision_reverts_an_over_budget_gain(self):
        from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

        performance = assess_integrate_performance(
            SharedState(baseline_tput=1000.0, power_budget_w=700.0),
            {"output_throughput": 1200.0, "gpu_power_avg_w": 812.0},
            base_tput=1000.0,
            keep_threshold_pct=1.0,
            stack_incremental_keep_threshold_pct=0.5,
        )
        assert performance.decision == "REVERT"
        assert performance.graded.veto_reason == "power_budget_exceeded"

    @pytest.mark.parametrize(("gpu_power_avg_w", "decision"), [(812.0, "REVERT"), (None, "REVERT"), (650.0, "KEEP")])
    def test_a_stacked_layer_below_the_graded_bar_still_honours_the_power_budget(self, gpu_power_avg_w, decision):
        """+0.7% on a stacked integrate KEEPs on the 0.5% stack bar while the 1.0% graded verdict carries no veto."""
        from hyperloom.orchestrator.measurement.integrate_performance import assess_integrate_performance

        state = SharedState(baseline_tput=1000.0, power_budget_w=700.0)
        state.current_best = {"action": "integrate", "tput": 1000.0}
        state.optimization_stack = [{"action": "integrate"}]
        measurement = {"output_throughput": 1007.0}
        if gpu_power_avg_w is not None:
            measurement["gpu_power_avg_w"] = gpu_power_avg_w

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


# --- reading and asserting the settings -----------------------------------------------------------------------------

_LIMIT = {
    "gpu_data": [
        {"gpu": 0, "limit": {"ppt0": {"socket_power_limit": {"value": 1400, "unit": "W"}}}},
        {"gpu": 1, "limit": {"ppt0": {"socket_power_limit": {"value": 1000, "unit": "W"}}}},
    ]
}
_PERF = {
    "gpu_data": [
        {"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"},
        {"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_DETERMINISM"},
    ]
}


def _fake_run(cmd, **_kwargs):
    payload = _LIMIT if "--limit" in cmd else _PERF
    return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")


def test_the_settings_are_read_from_amd_smi_json():
    settings = read_gpu_power_settings(run=_fake_run)
    assert settings == {
        0: {"power_cap_w": 1400.0, "perf_level": "auto"},
        1: {"power_cap_w": 1000.0, "perf_level": "determinism"},
    }


def test_a_failing_amd_smi_is_an_error_not_an_empty_reading():
    def _fail(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 2, "", "permission denied")

    with pytest.raises(GpuPowerSettingsError):
        read_gpu_power_settings(run=_fail)


def test_declared_settings_are_checked_per_card():
    settings = read_gpu_power_settings(run=_fake_run)
    assert declared_setting_problems(settings, power_cap_w=1000.0, gpus={1}) == []
    problems = declared_setting_problems(settings, power_cap_w=1000.0, perf_level="determinism")
    assert problems == [
        "GPU 0 power cap is 1400.0 W, declared 1000 W",
        "GPU 0 perf level is 'auto', declared 'determinism'",
    ]


class TestResolve:
    def _resolve(self, monkeypatch, **kw):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        kw.setdefault("nodes", 1)
        kw.setdefault("power_cap_w", None)
        kw.setdefault("perf_level", None)
        kw.setdefault("read", lambda: read_gpu_power_settings(run=_fake_run))
        return resolve_gpu_power_settings(**kw)

    def test_nothing_declared_records_what_the_cards_report(self, monkeypatch):
        record, error = self._resolve(monkeypatch)
        assert error == ""
        assert record["declared"] == {}
        assert record["observed"]["1"] == {"power_cap_w": 1000.0, "perf_level": "determinism"}

    def test_a_declared_cap_the_cards_are_not_at_refuses(self, monkeypatch):
        _, error = self._resolve(monkeypatch, power_cap_w=1000.0)
        assert "GPU 0 power cap is 1400.0 W" in error

    def test_only_the_cards_the_session_uses_are_checked(self, monkeypatch):
        from hyperloom.inference_optimizer.cli.bootstrap import resolve_gpu_power_settings

        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
        record, error = resolve_gpu_power_settings(
            power_cap_w=1000.0, perf_level="determinism", nodes=1, read=lambda: read_gpu_power_settings(run=_fake_run)
        )
        assert error == ""
        assert list(record["observed"]) == ["1"]

    def test_a_declaration_that_cannot_be_read_refuses(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi is not on PATH")

        _, error = self._resolve(monkeypatch, power_cap_w=1000.0, read=_unreadable)
        assert "cannot be checked" in error

    def test_no_declaration_and_no_amd_smi_records_nothing(self, monkeypatch):
        def _unreadable():
            raise GpuPowerSettingsError("amd-smi is not on PATH")

        assert self._resolve(monkeypatch, read=_unreadable) == ({}, "")

    def test_a_multi_node_declaration_refuses(self, monkeypatch):
        _, error = self._resolve(monkeypatch, power_cap_w=1000.0, nodes=2)
        assert "multi-node" in error


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


def test_the_prompt_states_the_budget_and_the_cap():
    state = SharedState(
        power_budget_w=700.0,
        gpu_power_settings={"observed": {"0": {"power_cap_w": 1000.0}, "1": {"power_cap_w": 1000.0}}},
    )
    line = state.to_power_budget_summary()
    assert "\n" not in line
    assert "700 W per-GPU mean power, cards capped at 1000 W" in line
    assert SharedState().to_power_budget_summary() == ""


def test_the_platform_fingerprint_records_the_settings(monkeypatch):
    from hyperloom.common.platform_probe import GPU_POWER_SETTINGS_ENV, platform_fingerprint

    monkeypatch.setenv(GPU_POWER_SETTINGS_ENV, json.dumps({"observed": {"0": {"power_cap_w": 1000.0}}}))
    record = platform_fingerprint("mi355x")
    if record.get("status") != "ok":
        pytest.skip("no host sysfs")
    assert record["gpu"]["power_settings"]["observed"]["0"]["power_cap_w"] == 1000.0


def test_the_stop_reason_is_in_the_closed_vocabulary():
    from hyperloom.orchestrator.phases.machine_state import is_valid_stop_reason

    assert is_valid_stop_reason("baseline_over_power_budget")
