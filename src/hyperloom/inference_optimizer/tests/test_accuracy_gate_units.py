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
        assert ag.enablement_accuracy_floor() == ag.DEFAULT_ENABLEMENT_ACCURACY_FLOOR

    def test_floor_default_rejects_a_collapsed_model(self):
        """The default must be strong enough to be the only correctness authority.

        A run once KEPT a candidate scoring gsm8k=0.00076 because the floor was
        0.0 and the gate degenerated to ``accuracy > 0``.
        """
        assert ag.DEFAULT_ENABLEMENT_ACCURACY_FLOOR > 0.0
        assert not ag.accuracy_meets_floor(0.00076, ag.DEFAULT_ENABLEMENT_ACCURACY_FLOOR)

    def test_floor_valid(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR", "0.7")
        assert ag.enablement_accuracy_floor() == pytest.approx(0.7)

    @pytest.mark.parametrize("bad", ["1.5", "-0.1", "nonsense", "nan", "inf"])
    def test_floor_invalid_or_out_of_range_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR", bad)
        assert ag.enablement_accuracy_floor() == ag.DEFAULT_ENABLEMENT_ACCURACY_FLOOR


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


def _write_minimal_bench_yaml(path, *, framework="sglang", model="/m", conc=64, isl=1024, osl=1024, run_eval="true"):
    """Write a minimal Magpie YAML for fingerprint tests."""
    import yaml as _yaml

    cfg = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "benchmark_script": f"{framework}_mi300x.sh",
            "precision": "fp8",
            "envs": {
                "CONC": conc,
                "ISL": isl,
                "OSL": osl,
                "TP": 8,
                "RUN_EVAL": run_eval,
            },
        }
    }
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")


class TestEvalContractFingerprint:
    def test_stable_across_calls(self, tmp_path):
        """Same config produces the same fingerprint on repeated calls."""
        cfg = tmp_path / "c.yaml"
        _write_minimal_bench_yaml(cfg)
        a = ag.eval_contract_fingerprint(config_path=cfg)
        b = ag.eval_contract_fingerprint(config_path=cfg)
        assert a == b and len(a) == 16

    def test_task_metric_do_not_affect_fingerprint(self, tmp_path):
        """Result-level task/metric do not change the fingerprint."""
        cfg = tmp_path / "c.yaml"
        _write_minimal_bench_yaml(cfg)
        fp_no_task = ag.eval_contract_fingerprint(config_path=cfg)
        fp_with_task = ag.eval_contract_fingerprint(config_path=cfg, task="gsm8k", metric="exact_match")
        fp_other_task = ag.eval_contract_fingerprint(config_path=cfg, task="mmlu", metric="acc")
        assert fp_no_task == fp_with_task == fp_other_task

    def test_crash_and_success_produce_same_fingerprint(self, tmp_path):
        """eval crash (no task/metric) and successful eval produce identical fingerprint."""
        cfg = tmp_path / "c.yaml"
        _write_minimal_bench_yaml(cfg)
        fp_crash = ag.eval_contract_fingerprint(config_path=cfg, task=None, metric=None)
        fp_success = ag.eval_contract_fingerprint(config_path=cfg, task="gsm8k", metric="exact_match,strict-match")
        assert fp_crash == fp_success

    def test_workload_change_changes_fingerprint(self, tmp_path):
        """A change to workload shape (ISL) changes the fingerprint."""
        import yaml as _yaml

        cfg1 = tmp_path / "c1.yaml"
        cfg2 = tmp_path / "c2.yaml"
        _write_minimal_bench_yaml(cfg1, isl=1024)
        _write_minimal_bench_yaml(cfg2, isl=2048)
        assert ag.eval_contract_fingerprint(config_path=cfg1) != ag.eval_contract_fingerprint(config_path=cfg2)

    def test_eval_control_change_changes_fingerprint(self, tmp_path):
        """Changing RUN_EVAL changes the fingerprint."""
        cfg_on = tmp_path / "on.yaml"
        cfg_off = tmp_path / "off.yaml"
        _write_minimal_bench_yaml(cfg_on, run_eval="true")
        _write_minimal_bench_yaml(cfg_off, run_eval="false")
        assert ag.eval_contract_fingerprint(config_path=cfg_on) != ag.eval_contract_fingerprint(config_path=cfg_off)

    def test_server_args_do_not_affect_fingerprint(self, tmp_path):
        """Server arg changes (allowed tuning) do not change the fingerprint."""
        import yaml as _yaml

        cfg = tmp_path / "c.yaml"
        _write_minimal_bench_yaml(cfg)
        fp_base = ag.eval_contract_fingerprint(config_path=cfg)
        # Add server args to YAML (different from contract fields).
        data = _yaml.safe_load(cfg.read_text())
        data["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = "--some-tuning-flag"
        cfg.write_text(_yaml.safe_dump(data))
        fp_with_args = ag.eval_contract_fingerprint(config_path=cfg)
        assert fp_base == fp_with_args

    def test_missing_config_returns_empty_string(self):
        """Missing config returns empty string (invalid contract sentinel)."""
        fp = ag.eval_contract_fingerprint(config_path="/no/such/file.yaml")
        assert fp == ""

    def test_none_config_returns_empty_string(self):
        """None config returns empty string."""
        fp = ag.eval_contract_fingerprint(config_path=None)
        assert fp == ""
