# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``action_executors._accuracy_gate``."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag


class TestIsHighAccuracyRisk:
    def test_no_risk_when_empty(self):
        assert ag.is_high_accuracy_risk("", None) is False
        assert ag.is_high_accuracy_risk(None or "", {}) is False

    @pytest.mark.parametrize(
        "args",
        [
            "--kv-cache-dtype fp8_e4m3",
            "--enforce-eager",
            "--compilation-config '{\"x\": 1}'",
            "--attention-backend aiter",
            "--decode-attention-backend triton",
        ],
    )
    def test_high_risk_cli_flags(self, args):
        assert ag.is_high_accuracy_risk(args, None) is True

    def test_high_risk_env_keys(self):
        assert ag.is_high_accuracy_risk("", {"VLLM_ROCM_USE_AITER": "1"}) is True
        assert ag.is_high_accuracy_risk("", {"SGLANG_USE_AITER": "1"}) is True

    def test_neutral_inputs_are_low_risk(self):
        assert ag.is_high_accuracy_risk("--max-num-seqs 64", {"FOO": "BAR"}) is False


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
        assert ag.accuracy_passed(0.0, 0.42) is True
        assert ag.accuracy_passed(-0.1, 0.0) is True

    def test_within_default_threshold(self):
        assert ag.accuracy_passed(0.80, 0.76) is True

    def test_outside_default_threshold(self):
        assert ag.accuracy_passed(0.80, 0.70) is False

    def test_custom_threshold(self):
        assert ag.accuracy_passed(0.80, 0.70, threshold=0.11) is True
        assert ag.accuracy_passed(0.80, 0.68, threshold=0.10) is False


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
        from hyperloom.orchestrator.kernel import request_handlers as krh
        from hyperloom.orchestrator.state import shared_state as ss

        monkeypatch.setattr(
            ss.SharedState,
            "load_or_init",
            classmethod(lambda cls, _dir: ss.SharedState(baseline_accuracy=baseline)),
        )
        return krh._grade_integrate_accuracy(bench, session_dir=tmp_path)

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
