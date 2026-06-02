"""Unit tests for ``action_executors._accuracy_gate``.

Covers the three public helpers (``is_high_accuracy_risk``,
``parse_eval_results``, ``accuracy_passed``) including the error / missing-
metric branches that integration tests don't trigger.
"""

from __future__ import annotations

import json

import pytest

from inference_optimizer.orchestrator.action_executors import _accuracy_gate as ag


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
        (eval_dir / "results.json").write_text(json.dumps({
            "results": {"task": {"unknown_metric": 0.5}},
        }))
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
        # ``baseline <= 0`` short-circuits to True.
        assert ag.accuracy_passed(0.0, 0.42) is True
        assert ag.accuracy_passed(-0.1, 0.0) is True

    def test_within_default_threshold(self):
        assert ag.accuracy_passed(0.80, 0.76) is True

    def test_outside_default_threshold(self):
        assert ag.accuracy_passed(0.80, 0.70) is False

    def test_custom_threshold(self):
        # 0.11 drop threshold lets 0.70 pass cleanly.
        assert ag.accuracy_passed(0.80, 0.70, threshold=0.11) is True
        assert ag.accuracy_passed(0.80, 0.68, threshold=0.10) is False
