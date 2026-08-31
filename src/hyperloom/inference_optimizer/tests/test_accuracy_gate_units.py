# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``actions.executors._accuracy_gate``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _accuracy_gate as ag


class TestNoHighRiskPredicate:
    def test_predicate_and_its_flag_lists_are_gone(self):
        """The accuracy gate is unconditional; nothing may reintroduce a list.

        A by-name catalogue of "risky" flags under-reports for any framework
        that spells a knob differently or adds its own, which is how a whole
        session's measured accuracies were discarded.
        """
        for name in ("is_high_accuracy_risk", "_HIGH_RISK_CLI_PATTERNS", "_HIGH_RISK_ENV_KEYS"):
            assert not hasattr(ag, name), f"{name} should not be reintroduced"


class TestAccuracyGateApplies:
    """What replaced the flag catalogue: a reference exists, or it does not."""

    @pytest.mark.parametrize(
        ("scriptable", "baseline_accuracy", "expected"),
        [
            # Any measured reference gates, whatever flags the variant carries --
            # this is the case the by-name catalogue used to miss.
            pytest.param(False, 0.9704, True, id="serving_with_baseline"),
            pytest.param(False, 0.0, False, id="serving_without_baseline"),
            # A scriptable framework's own quality verdict is its only signal, so
            # it gates with no baseline at all.
            pytest.param(True, 0.0, True, id="scriptable_needs_no_baseline"),
            pytest.param(True, 0.9704, True, id="scriptable_with_baseline"),
        ],
    )
    def test_gates_on_whether_a_reference_exists(self, scriptable, baseline_accuracy, expected):
        assert (
            ag.accuracy_gate_applies(
                scriptable=scriptable,
                baseline_accuracy=baseline_accuracy,
                eval_disabled=False,
            )
            is expected
        )

    @pytest.mark.parametrize(
        ("scriptable", "baseline_accuracy"),
        [
            # Both of the ways the gate could otherwise still fire: a baseline
            # measured before the opt-out, and a scriptable framework, which
            # does not consult the baseline at all.
            pytest.param(False, 0.9704, id="stale_baseline_on_state"),
            pytest.param(True, 0.0, id="scriptable"),
            pytest.param(True, 0.9704, id="scriptable_with_baseline"),
        ],
    )
    def test_disabled_eval_skips_the_gate_not_just_the_injection(self, scriptable, baseline_accuracy):
        """Opting out of eval must opt out of grading too.

        Forcing RUN_EVAL and reading the score were two separate conditions, so
        a round could be graded on an eval it was never going to run and REVERT
        every variant as ``accuracy_unavailable``.
        """
        assert (
            ag.accuracy_gate_applies(
                scriptable=scriptable,
                baseline_accuracy=baseline_accuracy,
                eval_disabled=True,
            )
            is False
        )

    def test_a_negative_baseline_is_not_a_reference(self):
        assert ag.accuracy_gate_applies(scriptable=False, baseline_accuracy=-1.0, eval_disabled=False) is False

    @pytest.mark.parametrize(
        ("scriptable", "baseline_accuracy"),
        [
            pytest.param(False, 0.9704, id="baseline_from_an_earlier_phase"),
            pytest.param(True, 0.0, id="scriptable"),
            pytest.param(True, 0.9704, id="scriptable_with_baseline"),
        ],
    )
    def test_the_rounds_own_contract_can_also_opt_out(self, scriptable, baseline_accuracy):
        """A base YAML / reference_envs ``RUN_EVAL=false`` is a real opt-out.

        It says the environment cannot evaluate at all — no lm-eval in the
        benchmark venv, typically. Forcing ``RUN_EVAL=true`` per variant over it
        produces no score either, so grading anyway REVERTs every variant as
        ``accuracy_unavailable``: the exact failure the forced injection exists
        to prevent.
        """
        assert (
            ag.accuracy_gate_applies(
                scriptable=scriptable,
                baseline_accuracy=baseline_accuracy,
                eval_disabled=False,
                run_eval_disabled=True,
            )
            is False
        )

    def test_a_variant_supplied_opt_out_is_still_overridden(self):
        """The two ``RUN_EVAL=false`` spellings must not collapse into one.

        ``run_eval_disabled`` is read from the config materialized BEFORE variant
        envs fold in, so a proposal switching off the eval its own KEEP is gated
        on never reaches this predicate — it is overridden by the injection.
        """
        assert (
            ag.accuracy_gate_applies(
                scriptable=False,
                baseline_accuracy=0.9704,
                eval_disabled=False,
                run_eval_disabled=False,
            )
            is True
        )


class TestReconcileBaselineAccuracy:
    """``SharedState.baseline_accuracy`` is the only supported channel."""

    def test_the_measured_baseline_outranks_a_proposed_one(self, caplog):
        with caplog.at_level("WARNING"):
            out = ag.reconcile_baseline_accuracy(0.42, SimpleNamespace(baseline_accuracy=0.9704), where="explore")
        assert out == pytest.approx(0.9704)
        assert "Ignoring proposed accuracy_baseline" in caplog.text
        # The log has to name how a genuinely re-measured reference can win.
        assert "writeback" in caplog.text

    def test_a_proposed_value_is_used_when_nothing_was_measured(self):
        assert ag.reconcile_baseline_accuracy(0.42, SimpleNamespace(baseline_accuracy=0.0)) == pytest.approx(0.42)

    @pytest.mark.parametrize("corrupt", ["corrupt", "0.9x", object()])
    def test_a_non_numeric_state_value_degrades_instead_of_raising(self, corrupt, caplog):
        """Both sides of the comparison have to survive a bad value.

        The proposed side was guarded and the state side was not, so a resumed
        session whose ``state.json`` carries a non-numeric ``baseline_accuracy``
        raised ``ValueError`` straight out of the KEEP decision. The behaviour it
        replaced passed the raw attribute on and degraded to throughput-only.
        """
        with caplog.at_level("WARNING"):
            out = ag.reconcile_baseline_accuracy(0.42, SimpleNamespace(baseline_accuracy=corrupt))
        assert out == pytest.approx(0.42)
        assert "is not a number" in caplog.text

    def test_a_non_numeric_proposed_value_is_ignored(self):
        assert ag.reconcile_baseline_accuracy("nonsense", SimpleNamespace(baseline_accuracy=0.0)) == 0.0

    def test_a_missing_shared_state_falls_back_to_the_proposal(self):
        assert ag.reconcile_baseline_accuracy(0.42, None) == pytest.approx(0.42)


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
        return krh._grade_integrate_accuracy(bench, session_dir=tmp_path, workspace=tmp_path)

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
        """The double-run evaluates in the warmup round only, so the result dict
        carries no accuracy and the score must be recovered from the workspace."""
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
        """The default must be strong enough to be the only correctness authority.

        A run once KEPT a candidate scoring gsm8k=0.00076 because the floor was
        0.0 and the gate degenerated to ``accuracy > 0``.
        """
        assert ag.DEFAULT_ENABLEMENT_ACCURACY_FLOOR > 0.0
        assert not ag.accuracy_meets_floor(0.00076, ag.DEFAULT_ENABLEMENT_ACCURACY_FLOOR)

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

    @pytest.mark.parametrize("mode", ["eval", "all"])
    def test_no_eval_closes_the_eval_lane_but_not_launch(self, mode):
        state = SimpleNamespace(enablement_mode=mode, eval_disabled=True)
        assert ag.eval_enablement_allowed(state) is False
        assert ag.launch_enablement_allowed(state) is (mode == "all")


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
