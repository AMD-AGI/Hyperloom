# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``actions.executors._accuracy_gate``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.common.gain_math import (
    AGENTX_ERROR_RATE_THRESHOLD_PCT,
    DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
    accuracy_meets_floor,
    accuracy_passed,
)
from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag
from hyperloom.orchestrator.kernel import integrate_accuracy as integrate_accuracy


class TestParseEvalResults:
    def test_returns_error_when_no_results_dir(self, tmp_path):
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] is None
        assert "no results" in out["error"]

    def test_parses_gsm8k_score(self, tmp_path):
        eval_dir = tmp_path / "eval_001"
        eval_dir.mkdir()
        results = {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.81,
                    "acc,none": 0.79,
                },
            },
        }
        (eval_dir / "results.json").write_text(json.dumps(results))
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.81)
        assert out["task"] == "gsm8k"
        assert out["metric"] == "exact_match,strict-match"
        assert out["source_file"].endswith("results.json")

    @pytest.mark.parametrize(("recorded", "expected"), [("7043.486084267031", 7043.486084267031), (None, None)])
    def test_reports_lm_evals_own_runtime(self, tmp_path, recorded, expected):
        eval_dir = tmp_path / "eval_001"
        eval_dir.mkdir()
        results = {"results": {"gsm8k": {"exact_match,strict-match": 0.845}}}
        if recorded is not None:
            results["total_evaluation_time_seconds"] = recorded
        (eval_dir / "results.json").write_text(json.dumps(results))
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.845)
        assert out["eval_runtime_sec"] == (pytest.approx(expected) if expected is not None else None)

    def test_falls_back_to_secondary_metric(self, tmp_path):
        eval_dir = tmp_path / "eval_002"
        eval_dir.mkdir()
        results = {
            "results": {
                "gsm8k": {"acc,none": 0.62},
            },
        }
        (eval_dir / "results.json").write_text(json.dumps(results))
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.62)
        assert out["metric"] == "acc,none"

    def test_handles_unrecognized_metric(self, tmp_path):
        eval_dir = tmp_path / "eval_003"
        eval_dir.mkdir()
        (eval_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {"task": {"unknown_metric": 0.5}},
                }
            )
        )
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] is None
        assert "no recognized metric" in out["error"]

    def test_handles_malformed_json(self, tmp_path):
        eval_dir = tmp_path / "eval_004"
        eval_dir.mkdir()
        (eval_dir / "results.json").write_text("{this is not json")
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] is None
        assert "parse error" in out["error"]


class TestAccuracyPassed:
    def test_baseline_zero_skips_gate(self):
        assert accuracy_passed(0.0, 0.42) is True
        assert accuracy_passed(-0.1, 0.0) is True

    def test_within_default_threshold(self):
        assert accuracy_passed(0.80, 0.76) is True

    def test_outside_default_threshold(self):
        assert accuracy_passed(0.80, 0.70) is False

    def test_custom_threshold(self):
        assert accuracy_passed(0.80, 0.70, threshold=0.11) is True
        assert accuracy_passed(0.80, 0.68, threshold=0.10) is False


class TestRequireKernelAccuracyDefault:
    def test_required_by_default(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", raising=False)
        assert ag.require_kernel_accuracy_default() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
    def test_opt_out_spellings(self, monkeypatch, value):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", value)
        assert ag.require_kernel_accuracy_default() is False


class TestGradeIntegrateAccuracy:
    """The kernel integrate gate reads the re-baseline's own score (no re-measure)."""

    @staticmethod
    def _grade(monkeypatch, tmp_path, *, bench: dict, baseline: float):

        return integrate_accuracy.grade_integrate_accuracy(
            bench, reference=ag.AccuracyReference(baseline, ""), workspace=tmp_path
        )

    def test_accuracy_within_tolerance_does_not_block(self, monkeypatch, tmp_path):
        out = self._grade(monkeypatch, tmp_path, bench={"accuracy": 0.78}, baseline=0.80)
        assert out["accuracy_pass"] is True
        assert out["blocked"] is False
        assert out["degraded"] is False

    def test_regression_blocks_with_negative_verdict(self, monkeypatch, tmp_path):
        out = self._grade(monkeypatch, tmp_path, bench={"accuracy": 0.60}, baseline=0.80)
        assert out["accuracy_pass"] is False
        assert out["blocked"] is True
        assert out["accuracy"] == pytest.approx(0.60)

    def test_missing_score_with_known_baseline_blocks_without_verdict(self, monkeypatch, tmp_path):
        out = self._grade(monkeypatch, tmp_path, bench={}, baseline=0.80)
        assert out["accuracy_pass"] is None
        assert out["blocked"] is True
        assert out["accuracy"] is None

    def test_no_baseline_degrades_to_throughput_only(self, monkeypatch, tmp_path):
        out = self._grade(monkeypatch, tmp_path, bench={"accuracy": 0.60}, baseline=0.0)
        assert out["accuracy_pass"] is None
        assert out["blocked"] is False
        assert out["degraded"] is True

    def test_opt_out_never_blocks(self, monkeypatch, tmp_path):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "0")
        out = self._grade(monkeypatch, tmp_path, bench={}, baseline=0.80)
        assert out["blocked"] is False

    def test_falls_back_to_warmup_round_eval_output(self, monkeypatch, tmp_path):
        """The double-run evaluates in the warmup round only, so the result dict carries no accuracy and the score must be recovered from the workspace."""
        warmup = tmp_path / "warmup_round" / "benchmark_sglang_smoke"
        warmup.mkdir(parents=True)
        (warmup / "results_gsm8k.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.61}}}),
            encoding="utf-8",
        )
        out = self._grade(monkeypatch, tmp_path, bench={}, baseline=0.80)
        assert out["accuracy"] == pytest.approx(0.61)
        assert out["accuracy_pass"] is False
        assert out["blocked"] is True
        assert out["source_file"].endswith("results_gsm8k.json")

    def test_carries_eval_provenance(self, monkeypatch, tmp_path):
        bench = {
            "accuracy": 0.78,
            "accuracy_task": "gsm8k",
            "accuracy_metric": "exact_match,strict-match",
            "accuracy_source": "/ws/results_gsm8k.json",
        }
        out = self._grade(monkeypatch, tmp_path, bench=bench, baseline=0.80)
        assert out["task"] == "gsm8k"
        assert out["metric"] == "exact_match,strict-match"
        assert out["source_file"] == "/ws/results_gsm8k.json"


class TestEnablementReaders:
    def test_floor_default_rejects_a_collapsed_model(self):
        """The default must be strong enough to be the only correctness authority."""
        assert DEFAULT_ENABLEMENT_ACCURACY_FLOOR > 0.0
        assert not accuracy_meets_floor(0.00076, DEFAULT_ENABLEMENT_ACCURACY_FLOOR)

    def test_missing_mode_is_fail_closed(self):
        """A caller that cannot supply the field has not opted in."""
        assert ag.resolve_enablement_mode(None) == ag.ENABLEMENT_MODE_OFF
        assert ag.resolve_enablement_mode(SimpleNamespace()) == ag.ENABLEMENT_MODE_OFF

    def test_unknown_mode_collapses_to_off(self):
        state = SimpleNamespace(enablement_mode="sometimes")
        assert ag.resolve_enablement_mode(state) == ag.ENABLEMENT_MODE_OFF
        assert ag.launch_enablement_allowed(state) is False
        assert ag.eval_enablement_allowed(state) is False

    @pytest.mark.parametrize(
        "mode,launch,eval_",
        [
            ("off", False, False),
            ("launch", True, False),
            ("eval", False, True),
            ("all", True, True),
        ],
    )
    def test_mode_admits_the_matching_lane(self, mode, launch, eval_):
        state = SimpleNamespace(enablement_mode=mode)
        assert ag.launch_enablement_allowed(state) is launch
        assert ag.eval_enablement_allowed(state) is eval_


class TestAccuracyValidator:
    @pytest.mark.parametrize(
        "score,floor,expected",
        [
            (0.5, 0.2, True),
            (0.2, 0.2, True),
            (0.19, 0.2, False),
            (0.0, 0.0, False),
            (-0.1, 0.0, False),
            (None, 0.0, False),
            (True, 0.0, False),
            (float("nan"), 0.0, False),
            (float("inf"), 0.0, False),
            ("0.5", 0.2, False),
        ],
    )
    def test_meets_floor(self, score, floor, expected):
        assert accuracy_meets_floor(score, floor) is expected

    def test_classify(self):
        assert ag.classify_accuracy_failure(0.5, 0.2) is None
        assert ag.classify_accuracy_failure(None, 0.2) == ag.EVAL_KIND_ACCURACY_UNAVAILABLE
        assert ag.classify_accuracy_failure(float("nan"), 0.2) == ag.EVAL_KIND_ACCURACY_UNAVAILABLE
        assert ag.classify_accuracy_failure(0.1, 0.2) == ag.EVAL_KIND_ACCURACY_BELOW_FLOOR
        assert ag.classify_accuracy_failure(0.0, 0.0) == ag.EVAL_KIND_ACCURACY_BELOW_FLOOR


def _accuracy_round(slot, score):
    """Write the lm-eval result an AgentX slot's accuracy round leaves below ``<slot>/accuracy/eval``."""
    eval_dir = slot / ag.ACCURACY_DIR / "eval" / "model"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_2026.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}), encoding="utf-8"
    )


def _error_rate(slot, **fields):
    ws = slot / "benchmark_x"
    ws.mkdir()
    (ws / "inferencex_result.json").write_text(json.dumps(fields), encoding="utf-8")
    return slot


class TestTheAgentXAccuracyGate:
    """Outside MLPerf an AgentX slot is graded on the GSM8K score of its accuracy round, never on its error rate."""

    @pytest.fixture(autouse=True)
    def _native_backend(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)

    def test_the_slot_reads_the_score_its_accuracy_round_wrote(self, tmp_path):
        _accuracy_round(tmp_path, 0.945)
        out = ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")
        assert (out["accuracy"], out["task"], out["metric"]) == (0.945, "gsm8k", "exact_match,strict-match")

    def test_a_dropped_score_is_not_hidden_behind_a_clean_error_rate(self, tmp_path):
        """#155 c5075: a candidate at 0.000 against a 0.945 baseline passed while the gate read 1.0 off its error
        rate."""
        _error_rate(tmp_path, request_error_rate=0.0)
        _accuracy_round(tmp_path, 0.0)
        assert (
            ag.grade_accuracy(str(tmp_path), ag.AccuracyReference(0.945, ""), framework="vllm", benchmark_mode="agentx")
            is False
        )

    def test_a_slot_without_an_accuracy_round_has_no_accuracy(self, tmp_path):
        """The error rate no longer stands in for a score: a clean rate alone grades nothing."""
        _error_rate(tmp_path, request_error_rate=0.0)
        out = ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")
        assert out["accuracy"] is None

    def test_an_eval_outside_the_accuracy_round_is_not_read(self, tmp_path):
        (tmp_path / "results.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}), encoding="utf-8"
        )
        assert ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")["accuracy"] is None

    def test_the_round_outranks_the_scriptable_quality_gate(self, tmp_path):
        (tmp_path / "benchmark_report.json").write_text(
            json.dumps({"quality_gate": {"passed": False}}), encoding="utf-8"
        )
        _accuracy_round(tmp_path, 0.9)
        assert ag.parse_eval_results(tmp_path, framework="xdit", benchmark_mode="agentx")["accuracy"] == 0.9

    def test_a_synthetic_session_never_reaches_this_gate(self, tmp_path):
        """The mode is the only switch; the round's presence must not arm it."""
        _error_rate(tmp_path, request_error_rate=0.9)
        assert ag.parse_eval_results(tmp_path, framework="vllm")["accuracy"] is None


class TestTheMLPerfInlineGate:
    """The MLPerf harness keeps grading on its inline score, admitted by the error rate."""

    @pytest.fixture(autouse=True)
    def _mlperf_backend(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")

    def test_a_scored_run_inside_the_threshold_reads_its_inline_score(self, tmp_path):
        _error_rate(
            tmp_path, request_error_rate=AGENTX_ERROR_RATE_THRESHOLD_PCT, accuracy_score=0.8, accuracy_missing_turns=0
        )
        out = ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")
        assert (out["accuracy"], out["task"]) == (0.8, "mlperf_agentic_inline")

    @pytest.mark.parametrize(
        "fields",
        [{"request_error_rate": 25.0}, {}, {"request_error_rate": 0.0, "accuracy_missing_turns": 1}],
        ids=["past_threshold", "no_rate", "unscored_turn"],
    )
    def test_a_run_the_harness_could_not_score_drops_to_zero(self, tmp_path, fields):
        _error_rate(tmp_path, **{"accuracy_score": 0.8, "accuracy_missing_turns": 0, **fields})
        assert ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")["accuracy"] == 0.0

    def test_an_accuracy_round_is_not_read(self, tmp_path):
        _accuracy_round(tmp_path, 0.9)
        assert ag.parse_eval_results(tmp_path, framework="vllm", benchmark_mode="agentx")["accuracy"] == 0.0


class TestParseAgentXErrorRate:
    def test_reads_the_rate_from_the_result(self, tmp_path):
        (tmp_path / "inferencex_result.json").write_text(json.dumps({"request_error_rate": 0.02}), encoding="utf-8")
        assert ag.parse_agentx_error_rate(tmp_path) == pytest.approx(0.02)

    def test_reports_none_when_no_result_exists(self, tmp_path):
        assert ag.parse_agentx_error_rate(tmp_path) is None

    def test_reports_none_when_the_field_is_absent_or_not_a_number(self, tmp_path):
        (tmp_path / "inferencex_result.json").write_text(json.dumps({"request_error_rate": "n/a"}), encoding="utf-8")
        assert ag.parse_agentx_error_rate(tmp_path) is None

    def test_an_unreadable_result_reports_none_and_warns(self, tmp_path, caplog):
        """A parse failure must not read as a rate of zero."""
        (tmp_path / "inferencex_result.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert ag.parse_agentx_error_rate(tmp_path) is None
        assert "unreadable" in caplog.text

    def test_the_newest_result_wins(self, tmp_path):
        """Rounds land side by side; the decision round is the latest one."""
        import os

        old = tmp_path / "round_a"
        new = tmp_path / "round_b"
        for d, rate in ((old, 0.9), (new, 0.01)):
            d.mkdir()
            (d / "inferencex_result.json").write_text(json.dumps({"request_error_rate": rate}), encoding="utf-8")
        os.utime(old / "inferencex_result.json", (1, 1))
        assert ag.parse_agentx_error_rate(tmp_path) == pytest.approx(0.01)
