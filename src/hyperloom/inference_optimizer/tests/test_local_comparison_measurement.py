# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read comparison metrics only from the accepted measurement and its recipe."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import yaml

from hyperloom.inference_optimizer.baseline_comparison import local_measurement as lm


@pytest.fixture
def accepted(tmp_path):
    workspace = tmp_path / "accepted"
    workspace.mkdir()
    recipe = workspace / "config.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "run_mode": "local",
                    "model": "/models/GLM-5.2",
                    "envs": {"TP": 2, "CONC": 4, "EXTRA_SGLANG_ARGS": "--page-size 16"},
                }
            }
        ),
        encoding="utf-8",
    )
    raw = workspace / "inferencex_result.json"
    raw.write_text(
        json.dumps(
            {
                "output_throughput": 100.0,
                "input_throughput": 900.0,
                "total_token_throughput": 1000.0,
                "e2e_norm_intvty_p90": 22.6,
                "submission_valid": True,
                "comparison_metrics": {
                    "status": "ok",
                    "reason": "",
                    "sample_count": 2,
                    "metric_basis": "inverse_linear_p90_e2el_per_output_token",
                    "unit": "tok/s/user",
                    "e2e_norm_intvty_p90": 1.0 / 0.91,
                    "source_sha256": "a" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    best = {
        "tput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 22.6,
        "measurement": {
            "tput": 100.0,
            "raw_result_path": str(raw),
            "benchmark_workspace": str(workspace),
            "launch_evidence": {
                "framework": "sglang",
                "model_path": "/models/GLM-5.2",
                "materialized_config_path": str(recipe),
                "recipe_digest": "sha256:" + hashlib.sha256(recipe.read_bytes()).hexdigest(),
                "requested_server_env": {"TP": "2", "CONC": "4"},
                "requested_server_args": "--page-size 16",
                "observed_server_identity": {"tp_size": 2, "dp_size": 1},
            },
        },
    }
    return best, raw, recipe


def _rewrite_recipe(best, recipe, *, envs=None, args=None):
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    if envs:
        data["benchmark"]["envs"].update(envs)
    if args is not None:
        data["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = args
        best["measurement"]["launch_evidence"]["requested_server_args"] = args
    recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    evidence = best["measurement"]["launch_evidence"]
    evidence["recipe_digest"] = "sha256:" + hashlib.sha256(recipe.read_bytes()).hexdigest()
    evidence["requested_server_env"].update({str(k): str(v) for k, v in (envs or {}).items()})


def test_local_view_uses_exact_p90_and_effective_recipe_tp(accepted):
    best, raw, _recipe = accepted
    before = deepcopy(best)
    result = lm.load_local_measurement(best)
    assert result["status"] == "ok"
    assert result["e2e_norm_intvty_p90"] == pytest.approx(1.0 / 0.91)
    assert result["total_tput_per_gpu"] == 500.0
    assert result["gpu_count"] == 2
    assert result["conc"] == 4
    assert result["raw_result_path"] == str(raw)
    assert best == before


def test_local_view_never_selects_newest_or_neighbor_result(accepted, tmp_path):
    best, raw, _recipe = accepted
    (raw.parent / "newest_result.json").write_text('{"output_throughput":9999}', encoding="utf-8")
    best["measurement"].pop("raw_result_path")
    result = lm.load_local_measurement(best)
    assert result["status"] == "unavailable"
    assert result["reason"] == "raw_result_missing"
    assert result["total_tput_per_gpu"] is None


@pytest.mark.parametrize(
    "field,value", [("output_throughput", 200), ("total_token_throughput", 2000), ("e2e_norm_intvty_p90", 40)]
)
def test_replaced_result_does_not_borrow_accepted_measurement_identity(accepted, field, value):
    best, raw, _recipe = accepted
    data = json.loads(raw.read_text(encoding="utf-8"))
    data[field] = value
    raw.write_text(json.dumps(data), encoding="utf-8")
    result = lm.load_local_measurement(best)
    assert result["status"] == "unavailable"
    assert result["reason"] == "measurement_mismatch"


def test_local_precision_comes_only_from_verified_measurement_recipe(accepted):
    best, _raw, recipe = accepted
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["benchmark"]["precision"] = "bf16"
    recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    evidence = best["measurement"]["launch_evidence"]
    evidence["recipe_digest"] = "sha256:" + hashlib.sha256(recipe.read_bytes()).hexdigest()
    assert lm.load_local_measurement(best)["precision"] == "bf16"
    recipe.write_text(recipe.read_text(encoding="utf-8").replace("bf16", "fp8"), encoding="utf-8")
    assert lm.load_local_measurement(best)["precision"] == ""


def test_changed_recipe_is_not_used_for_gpu_normalization(accepted):
    best, _raw, recipe = accepted
    recipe.write_text(recipe.read_text(encoding="utf-8") + "# replaced\n", encoding="utf-8")
    result = lm.load_local_measurement(best)
    assert result["e2e_norm_intvty_p90"] == pytest.approx(1.0 / 0.91)
    assert result["total_tput_per_gpu"] is None
    assert result["throughput_reason"] == "recipe_mismatch"


@pytest.mark.parametrize("env", [{"PP": 2}, {"DP": 2}, {"PCP": 2}, {"NNODES": 2}, {"PD_MODE": "disaggregated"}])
def test_unsupported_topology_never_normalizes_by_tp_alone(accepted, env):
    best, _raw, recipe = accepted
    _rewrite_recipe(best, recipe, envs=env)
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] is None
    assert result["throughput_reason"] == "unsupported_topology"


@pytest.mark.parametrize(
    "args",
    [
        "--pp-size 2",
        "--pipeline-parallel-size=2",
        "--data-parallel-size 2",
        "-dp 2",
        "--prefill-context-parallel-size 2",
        "-pcp 2",
        "--nnodes 2",
        "--tp-size 4",
    ],
)
def test_server_flags_cannot_override_recipe_topology_silently(accepted, args):
    best, _raw, recipe = accepted
    _rewrite_recipe(best, recipe, args=args)
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] is None


@pytest.mark.parametrize("framework,inactive_key", [("sglang", "EXTRA_VLLM_ARGS"), ("vllm", "EXTRA_SGLANG_ARGS")])
def test_inactive_framework_args_do_not_change_topology(accepted, framework, inactive_key):
    best, _raw, recipe = accepted
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["benchmark"]["framework"] = framework
    data["benchmark"]["envs"][inactive_key] = "--pipeline-parallel-size 2"
    recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    evidence = best["measurement"]["launch_evidence"]
    evidence["framework"] = framework
    evidence["recipe_digest"] = "sha256:" + hashlib.sha256(recipe.read_bytes()).hexdigest()
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] == 500.0
    assert result["throughput_reason"] == ""


def test_missing_effective_tp_is_not_replaced_with_one(accepted):
    best, _raw, recipe = accepted
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["benchmark"]["envs"].pop("TP")
    recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
    evidence = best["measurement"]["launch_evidence"]
    evidence["recipe_digest"] = "sha256:" + hashlib.sha256(recipe.read_bytes()).hexdigest()
    evidence["requested_server_env"].pop("TP")
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] is None
    assert result["throughput_reason"] == "gpu_count_missing"


def test_observed_tp_conflict_invalidates_per_gpu_value(accepted):
    best, _raw, _recipe = accepted
    best["measurement"]["launch_evidence"]["observed_server_identity"]["tp_size"] = 1
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] is None
    assert result["throughput_reason"] == "topology_mismatch"


@pytest.mark.parametrize("comparison", [None, {"status": "unavailable", "reason": "request_records_missing"}])
def test_missing_exact_p90_preserves_total_without_using_internal_p10(accepted, comparison):
    best, raw, _recipe = accepted
    data = json.loads(raw.read_text(encoding="utf-8"))
    data["comparison_metrics"] = comparison
    raw.write_text(json.dumps(data), encoding="utf-8")
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] == 500.0
    assert result["e2e_norm_intvty_p90"] is None
    assert result["interactivity_reason"]


def test_noncanonical_measurement_does_not_supply_external_gaps(accepted):
    best, raw, _recipe = accepted
    data = json.loads(raw.read_text(encoding="utf-8"))
    data["submission_valid"] = False
    raw.write_text(json.dumps(data), encoding="utf-8")
    result = lm.load_local_measurement(best)
    assert result["status"] == "unavailable"
    assert result["reason"] == "submission_not_valid"


@pytest.mark.parametrize("contents", ["not-json", "[]", "null"])
def test_bad_raw_result_is_explicitly_unavailable(accepted, contents):
    best, raw, _recipe = accepted
    raw.write_text(contents, encoding="utf-8")
    result = lm.load_local_measurement(best)
    assert result["status"] == "unavailable"
    assert result["reason"] in {"raw_result_unreadable", "raw_result_invalid"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "ms"),
        ("metric_basis", "mean_tpot"),
        ("e2e_norm_intvty_p90", float("nan")),
        ("e2e_norm_intvty_p90", True),
    ],
)
def test_incompatible_p90_contract_preserves_only_total(accepted, field, value):
    best, raw, _recipe = accepted
    data = json.loads(raw.read_text(encoding="utf-8"))
    data["comparison_metrics"][field] = value
    raw.write_text(json.dumps(data), encoding="utf-8")
    result = lm.load_local_measurement(best)
    assert result["e2e_norm_intvty_p90"] is None
    assert result["total_tput_per_gpu"] == 500.0


def test_caller_reused_server_without_topology_evidence_has_no_gpu_normalization(accepted):
    best, _raw, _recipe = accepted
    best["measurement"]["launch_evidence"]["warm_reuse"] = {"provenance": "caller_ready_server"}
    result = lm.load_local_measurement(best)
    assert result["total_tput_per_gpu"] is None
    assert result["throughput_reason"] == "topology_unverified"


def test_total_fallback_uses_same_raw_input_and_output(accepted):
    best, raw, _recipe = accepted
    data = json.loads(raw.read_text(encoding="utf-8"))
    data.pop("total_token_throughput")
    raw.write_text(json.dumps(data), encoding="utf-8")
    assert lm.load_local_measurement(best)["total_tput_per_gpu"] == 500.0
