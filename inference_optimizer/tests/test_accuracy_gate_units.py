# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``action_executors._accuracy_gate``."""

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

    # Tests for the extended parse_eval_results: single-task back-compat with
    # tasks_used, acc_norm recognition, multi-task averaging, group-aggregate-row
    # exclusion, and mixed-metric labeling (sparse tinyBenchmarks / metabench).
    def test_single_task_reports_tasks_used(self, tmp_path):
        eval_dir = tmp_path / "eval_one"
        eval_dir.mkdir()
        (eval_dir / "results.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.81}}})
        )
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.81)
        assert out["task"] == "gsm8k"
        assert out["tasks_used"] == ["gsm8k"]
        assert out["metric"] == "exact_match,strict-match"

    def test_recognizes_acc_norm(self, tmp_path):
        eval_dir = tmp_path / "eval_an"
        eval_dir.mkdir()
        (eval_dir / "results.json").write_text(
            json.dumps({"results": {"tinyMMLU": {"acc_norm,none": 0.64}}})
        )
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.64)
        assert out["metric"] == "acc_norm,none"

    def test_averages_multiple_leaf_tasks(self, tmp_path):
        # A sparse group (tinyBenchmarks-style) emits several leaf tasks; the
        # gate averages them into one comparable number.
        eval_dir = tmp_path / "eval_grp"
        eval_dir.mkdir()
        results = {
            "results": {
                "tinyArc": {"acc_norm,none": 0.50},
                "tinyHellaswag": {"acc_norm,none": 0.70},
            }
        }
        (eval_dir / "results.json").write_text(json.dumps(results))
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.60)
        assert set(out["tasks_used"]) == {"tinyArc", "tinyHellaswag"}
        assert out["metric"] == "acc_norm,none"

    def test_excludes_group_aggregate_rows(self, tmp_path):
        # lm-eval reports the group row alongside its leaf subtasks; averaging it
        # in would double-count, so group_subtasks keys are excluded.
        eval_dir = tmp_path / "eval_mb"
        eval_dir.mkdir()
        results = {
            "results": {
                "metabench": {"acc,none": 0.99},  # group aggregate — must be ignored
                "metabench_arc": {"acc,none": 0.40, "acc_norm,none": 0.48},
                "metabench_mmlu": {"acc,none": 0.60},
            },
            "group_subtasks": {"metabench": ["metabench_arc", "metabench_mmlu"]},
        }
        (eval_dir / "results.json").write_text(json.dumps(results))
        out = ag.parse_eval_results(tmp_path)
        # mean(0.40, 0.60) = 0.50; the 0.99 group row is excluded; acc,none
        # outranks acc_norm,none on metabench_arc.
        assert out["accuracy"] == pytest.approx(0.50)
        assert "metabench" not in out["tasks_used"]
        assert set(out["tasks_used"]) == {"metabench_arc", "metabench_mmlu"}

    def test_mixed_metrics_marks_metric_mixed(self, tmp_path):
        eval_dir = tmp_path / "eval_mix"
        eval_dir.mkdir()
        results = {
            "results": {
                "tinyGSM8k": {"exact_match,strict-match": 0.50},
                "tinyArc": {"acc_norm,none": 0.60},
            }
        }
        (eval_dir / "results.json").write_text(json.dumps(results))
        out = ag.parse_eval_results(tmp_path)
        assert out["accuracy"] == pytest.approx(0.55)
        assert out["metric"] == "mixed"


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
