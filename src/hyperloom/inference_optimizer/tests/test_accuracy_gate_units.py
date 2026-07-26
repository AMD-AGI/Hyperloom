# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
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


class TestEnablementReaders:
    def test_on_eval_fail_default_on(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL", raising=False)
        assert ag.enablement_on_eval_fail_enabled() is True

    def test_on_eval_fail_disabled(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL", "off")
        assert ag.enablement_on_eval_fail_enabled() is False

    def test_floor_default(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR", raising=False)
        assert ag.enablement_accuracy_floor() == 0.0

    def test_floor_valid(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR", "0.7")
        assert ag.enablement_accuracy_floor() == pytest.approx(0.7)

    @pytest.mark.parametrize("bad", ["1.5", "-0.1", "nonsense", "nan", "inf"])
    def test_floor_invalid_or_out_of_range_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR", bad)
        assert ag.enablement_accuracy_floor() == 0.0


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
        assert ag.accuracy_meets_floor(score, floor) is expected

    def test_classify(self):
        assert ag.classify_accuracy_failure(0.5, 0.2) is None
        assert ag.classify_accuracy_failure(None, 0.2) == ag.EVAL_KIND_ACCURACY_UNAVAILABLE
        assert ag.classify_accuracy_failure(float("nan"), 0.2) == ag.EVAL_KIND_ACCURACY_UNAVAILABLE
        assert ag.classify_accuracy_failure(0.1, 0.2) == ag.EVAL_KIND_ACCURACY_BELOW_FLOOR
        assert ag.classify_accuracy_failure(0.0, 0.0) == ag.EVAL_KIND_ACCURACY_BELOW_FLOOR


class TestEvalContractFingerprint:
    def test_stable_and_sensitive(self):
        a = ag.eval_contract_fingerprint(config_path=None, framework="sglang", model="m", task="gsm8k", metric="em")
        b = ag.eval_contract_fingerprint(config_path=None, framework="sglang", model="m", task="gsm8k", metric="em")
        c = ag.eval_contract_fingerprint(config_path=None, framework="sglang", model="m", task="mmlu", metric="em")
        assert a == b and a != c and len(a) == 16

    def test_hashes_config_bytes(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("a: 1\n")
        fp1 = ag.eval_contract_fingerprint(config_path=cfg, framework="f", model="m", task="t", metric="x")
        cfg.write_text("a: 2\n")
        fp2 = ag.eval_contract_fingerprint(config_path=cfg, framework="f", model="m", task="t", metric="x")
        assert fp1 != fp2
        # A missing config path contributes its string form and never raises.
        assert ag.eval_contract_fingerprint(
            config_path="/no/such/file.yaml", framework="f", model="m", task="t", metric="x"
        )
