# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for aiperf export -> InferenceX-schema mapping."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from hyperloom.inference_optimizer.agentx.mapping import map_aiperf, pct, stat


def _metric(avg, **pct):
    d = {"unit": "ms", "avg": avg}
    d.update(pct)
    return d


def _sample():
    return {
        "request_throughput": {"unit": "req/s", "avg": 3.0},
        "output_token_throughput": {"unit": "tok/s", "avg": 500.0},
        "input_token_throughput": {"unit": "tok/s", "avg": 1500.0},
        "total_token_throughput": {"unit": "tok/s", "avg": 2000.0},
        "request_count": {"unit": "req", "avg": 42},
        "input_sequence_length": {"unit": "tok", "avg": 100.0},
        "total_isl": {"unit": "tok", "avg": 4200.0},
        "total_output_tokens": {"unit": "tok", "avg": 2100.0},
        "benchmark_duration": {"unit": "s", "avg": 14.0},
        "time_to_first_token": _metric(120.0, p50=110.0, p99=200.0, std=15.0),
        "inter_token_latency": _metric(20.0, p50=18.0, p90=34.3, p99=40.0, std=5.0),
        # e2e_output_token_throughput is OSL/E2EL_s per request (larger = faster); the slow tail is its P10.
        "e2e_output_token_throughput": _metric(209.9, p10=22.6, p50=55.0, p90=447.2, p99=2028.5),
        # 1/ITL, deliberately far from the e2e figure so reading the wrong axis cannot pass.
        "output_token_throughput_per_user": _metric(686.1, p50=84.1, p90=1092.6),
        "request_latency": _metric(900.0, p50=850.0, p99=1500.0, std=120.0),
        "theoretical_prefix_cache_hit": {"unit": "%", "avg": 0.73},
    }


def test_stat_reads_sub_key_and_default():
    m = {"x": {"avg": 1.0, "p99": 9.0}}
    assert stat(m, "x") == 1.0
    assert stat(m, "x", "p99") == 9.0
    assert stat(m, "missing") == 0.0
    assert stat(m, "x", "p50") == 1.0  # falls back to avg when sub absent


def test_pct_does_not_fall_back_to_avg():
    m = {"x": {"avg": 1.0, "p99": 9.0}}
    assert pct(m, "x", "p99") == 9.0
    assert pct(m, "x", "p50") == 0.0  # absent -> default, not avg
    assert pct(m, "missing", "p90") == 0.0


def test_map_core_throughput_and_counts():
    r = map_aiperf(_sample())
    assert r["request_throughput"] == 3.0
    assert r["output_throughput"] == 500.0
    assert r["input_throughput"] == 1500.0
    assert r["total_token_throughput"] == 2000.0
    assert r["completed"] == 42
    assert r["total_input_tokens"] == 4200
    assert r["total_output_tokens"] == 2100
    assert r["duration"] == 14.0


def test_map_latency_fields():
    r = map_aiperf(_sample())
    assert r["mean_ttft_ms"] == 120.0
    assert r["median_ttft_ms"] == 110.0
    assert r["p99_ttft_ms"] == 200.0
    assert r["std_ttft_ms"] == 15.0
    assert r["mean_itl_ms"] == 20.0
    assert r["p99_itl_ms"] == 40.0
    # tpot mirrors inter_token_latency in the aiperf schema
    assert r["mean_tpot_ms"] == 20.0
    assert r["p90_tpot_ms"] == 34.3
    # e2e_norm_intvty_p90 is the slow-tail (P10 of the per-request rate = 22.6).
    assert r["e2e_norm_intvty_p90"] == pytest.approx(22.6)
    assert r["mean_e2el_ms"] == 900.0
    assert r["p99_e2el_ms"] == 1500.0


def test_map_prefix_cache_hit():
    r = map_aiperf(_sample())
    assert r["theoretical_prefix_cache_hit"] == 0.73


def test_map_total_tput_fallback_from_in_plus_out():
    s = _sample()
    del s["total_token_throughput"]
    r = map_aiperf(s)
    assert r["total_token_throughput"] == 2000.0  # 1500 in + 500 out


def test_e2e_norm_intvty_p90_reads_p10_slow_tail():
    """Must read P10 (slow tail) not P90: P10 of OSL/E2EL_s is 1/P90(E2EL/OSL), upstream's definition."""
    s = _sample()
    r = map_aiperf(s)
    assert r["e2e_norm_intvty_p90"] == pytest.approx(22.6)  # p10, not p90=447.2


def test_e2e_norm_intvty_p90_is_zero_when_export_has_no_p10():
    """An export where e2e_output_token_throughput carries no p10 must emit 0.0, not the mean."""
    s = _sample()
    s["e2e_output_token_throughput"] = {"unit": "tok/s", "avg": 209.9}
    r = map_aiperf(s)
    assert r["e2e_norm_intvty_p90"] == 0.0, f"expected 0.0 (no p10 present), got {r['e2e_norm_intvty_p90']!r}"


def test_map_accepts_metrics_wrapped():
    r = map_aiperf({"metrics": _sample()})
    assert r["output_throughput"] == 500.0


def test_map_missing_metric_defaults_zero():
    r = map_aiperf({"output_token_throughput": {"avg": 10.0}})
    assert r["output_throughput"] == 10.0
    assert r["mean_ttft_ms"] == 0.0
    assert r["completed"] == 0


def test_noncanonical_reasons_force_the_verdict_false():
    """The client sees deviations the scenario cannot."""
    export = {"output_token_throughput": {"avg": 10.0}, "metadata": {"submission_valid": True}}
    r = map_aiperf(export, noncanonical_reasons=["entries=50(canonical 393)"])
    assert r["submission_valid"] is False
    assert "entries=50(canonical 393)" in r["submission_invalid_reasons"]


def test_noncanonical_reasons_append_to_scenario_reasons():
    export = {
        "output_token_throughput": {"avg": 10.0},
        "metadata": {"submission_valid": False, "submission_invalid_reasons": ["unsafe_override"]},
    }
    r = map_aiperf(export, noncanonical_reasons=["duration=120s(canonical 3600s)"])
    assert r["submission_valid"] is False
    assert r["submission_invalid_reasons"] == ["unsafe_override", "duration=120s(canonical 3600s)"]


def test_empty_noncanonical_reasons_leave_the_verdict_alone():
    """A canonical run must not be demoted by an empty or blank list."""
    export = {"output_token_throughput": {"avg": 10.0}, "metadata": {"submission_valid": True}}
    for reasons in (None, [], ["", "  "]):
        r = map_aiperf(export, noncanonical_reasons=reasons)
        assert r["submission_valid"] is True, reasons
        assert r["submission_invalid_reasons"] == []


def test_vendored_asset_fallback_honours_noncanonical_reasons(monkeypatch):
    """The fallback runs on boxes where the package is not importable, i.e."""
    import importlib.util
    import sys

    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

    monkeypatch.setitem(sys.modules, "hyperloom.inference_optimizer.agentx.mapping", None)
    spec = importlib.util.spec_from_file_location("_asset_map_aiperf_nc", str(agentx_asset_dir() / "map_aiperf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    export = {"output_token_throughput": {"avg": 10.0}, "metadata": {"submission_valid": True}}
    assert mod.map_aiperf(export, noncanonical_reasons=["entries=50"]) == map_aiperf(
        export, noncanonical_reasons=["entries=50"]
    )
    assert mod.map_aiperf(export, noncanonical_reasons=["entries=50"])["submission_valid"] is False


def test_vendored_asset_fallback_matches_package(monkeypatch):
    """The deployed asset vendors a fallback map_aiperf for when the package is not importable; guard it against drifting from the package implementation."""
    import importlib.util
    import sys

    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

    asset = agentx_asset_dir() / "map_aiperf.py"
    # Force the asset's `from ...mapping import map_aiperf` to raise so the vendored fallback branch is the one
    # exercised.
    monkeypatch.setitem(sys.modules, "hyperloom.inference_optimizer.agentx.mapping", None)
    spec = importlib.util.spec_from_file_location("_asset_map_aiperf", str(asset))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.map_aiperf(_sample()) == map_aiperf(_sample())


def test_map_corpus_shape_projects_the_measured_distributions():
    """The shape the prompts render comes from the export, not from constants."""
    from hyperloom.inference_optimizer.agentx.mapping import map_corpus_shape

    shape = map_corpus_shape(map_aiperf(_sample()))
    assert shape["corpus_loader"] == ""  # _sample() carries no dataset metadata
    assert shape["isl"] == {"avg": 100}  # only avg present in the sample
    assert shape["completed_requests"] == 42
    assert shape["duration_s"] == pytest.approx(14.0)
    assert shape["prefix_cache_hit"] == pytest.approx(0.73)
    assert shape["source"] == "measured"


def test_map_corpus_shape_carries_the_loader_and_the_percentiles():
    s = _sample()
    s["metadata"] = {"dataset": {"loader": "semianalysis_cc_traces_weka_062126"}}
    s["input_sequence_length"] = {"avg": 113814.0, "p50": 94821.0, "p90": 163328.0, "p99": 506158.0}
    s["output_sequence_length"] = {"avg": 806.0, "p50": 333.0, "p90": 1874.0, "p99": 6386.0}
    s["request_error_rate"] = {"unit": "ratio", "avg": 0.007}

    from hyperloom.inference_optimizer.agentx.mapping import map_corpus_shape

    shape = map_corpus_shape(map_aiperf(s))
    assert shape["corpus_loader"] == "semianalysis_cc_traces_weka_062126"
    assert shape["isl"] == {"avg": 113814, "p50": 94821, "p90": 163328, "p99": 506158}
    assert shape["osl"] == {"avg": 806, "p50": 333, "p90": 1874, "p99": 6386}
    assert shape["request_error_rate"] == pytest.approx(0.007)


def test_an_export_with_no_sequence_metrics_yields_empty_distributions():
    """A synthetic result carries none of this; the record must not invent it."""
    from hyperloom.inference_optimizer.agentx.mapping import map_corpus_shape

    shape = map_corpus_shape(map_aiperf({"output_token_throughput": {"avg": 10.0}}))
    assert shape["isl"] == {} and shape["osl"] == {}
    assert shape["completed_requests"] == 0


def test_a_non_list_invalid_reason_is_coerced_to_one():
    """aiperf may stamp a bare string; the schema field is a list."""
    export = {
        "output_token_throughput": {"avg": 10.0},
        "metadata": {"submission_valid": False, "submission_invalid_reasons": "unsafe_override"},
    }
    assert map_aiperf(export)["submission_invalid_reasons"] == ["unsafe_override"]


@pytest.fixture(params=[False, True], ids=["installed-package", "standalone-fallback"])
def asset_mapper(monkeypatch, request):
    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

    if request.param:
        monkeypatch.setitem(sys.modules, "hyperloom.inference_optimizer.agentx.mapping", None)
    path = agentx_asset_dir() / "map_aiperf.py"
    spec = importlib.util.spec_from_file_location("_comparison_asset_mapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request_record(latency_ms, output_tokens, *, phase="profiling", **overrides):
    record = {
        "metadata": {"benchmark_phase": phase},
        "metrics": {
            "request_latency": {"value": latency_ms, "unit": "ms"},
            "time_to_first_token": {"value": 10.0, "unit": "ms"},
            "input_sequence_length": {"value": 128, "unit": "tokens"},
            "output_sequence_length": {"value": output_tokens, "unit": "tokens"},
        },
        "error": None,
    }
    record.update(overrides)
    return record


def _run_asset_mapper(mapper, tmp_path, records, *, summary=None):
    source = tmp_path / "aiperf_artifacts" / "profile_export_aiperf.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(_sample() if summary is None else summary), encoding="utf-8")
    if records is not None:
        (source.parent / "profile_export.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    destination = tmp_path / "inferencex_result.json"
    mapper.main(str(source), str(destination))
    return json.loads(destination.read_text(encoding="utf-8"))


def test_file_mapper_adds_exact_comparison_p90_without_changing_grading(asset_mapper, tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTX_NONCANONICAL_REASONS", raising=False)
    records = [_request_record(1000, 10), _request_record(20000, 20)]
    result = _run_asset_mapper(asset_mapper, tmp_path, records)
    comparison = result.pop("comparison_metrics")

    assert result == map_aiperf(_sample())
    assert result["e2e_norm_intvty_p90"] == 22.6
    assert comparison["status"] == "ok"
    assert comparison["e2e_norm_intvty_p90"] == pytest.approx(1.0 / 0.91)
    assert comparison["e2e_norm_intvty_p90"] != pytest.approx(1.9)
    assert comparison["metric_basis"] == "inverse_linear_p90_e2el_per_output_token"
    assert comparison["unit"] == "tok/s/user"
    assert comparison["sample_count"] == 2
    assert comparison["source"] == "profile_export.jsonl"
    records_path = tmp_path / "aiperf_artifacts/profile_export.jsonl"
    assert comparison["source_sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("ratios,expected", [([0.1], 10.0), ([1.0, 1.0], 1.0), ([3.0, 1.0, 2.0], 1.0 / 2.8)])
def test_comparison_p90_uses_linear_quantile_over_request_ratios(asset_mapper, tmp_path, ratios, expected):
    result = _run_asset_mapper(asset_mapper, tmp_path, [_request_record(ratio * 10000, 10) for ratio in ratios])
    assert result["comparison_metrics"]["e2e_norm_intvty_p90"] == pytest.approx(expected)


def test_comparison_samples_match_upstream_phase_and_turn_filter(asset_mapper, tmp_path):
    missing_ttft = _request_record(100000, 1)
    missing_ttft["metrics"].pop("time_to_first_token")
    zero_isl = _request_record(100000, 1)
    zero_isl["metrics"]["input_sequence_length"]["value"] = 0
    records = [
        _request_record(100000, 1, phase="warmup"),
        missing_ttft,
        zero_isl,
        _request_record(1000, 0),
        _request_record(1000, 10),
        _request_record(1000, 10, phase=None),
        _request_record(1000, 10, error={"message": "record still has all metrics"}),
    ]
    result = _run_asset_mapper(asset_mapper, tmp_path, records)
    assert result["comparison_metrics"]["sample_count"] == 3
    assert result["comparison_metrics"]["e2e_norm_intvty_p90"] == pytest.approx(10.0)


def test_comparison_supports_upstream_bare_numeric_metrics(asset_mapper, tmp_path):
    record = _request_record(1000, 10)
    record["metrics"] = {key: metric["value"] for key, metric in record["metrics"].items()}
    result = _run_asset_mapper(asset_mapper, tmp_path, [record])
    assert result["comparison_metrics"]["e2e_norm_intvty_p90"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "records,reason",
    [
        (None, "request_records_missing"),
        ([], "no_eligible_requests"),
        ([_request_record(1000, 10, phase="warmup")], "no_eligible_requests"),
    ],
)
def test_missing_comparison_evidence_does_not_fail_the_benchmark(asset_mapper, tmp_path, records, reason):
    result = _run_asset_mapper(asset_mapper, tmp_path, records)
    comparison = result.pop("comparison_metrics")
    assert result == map_aiperf(_sample())
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == reason
    assert comparison["e2e_norm_intvty_p90"] is None


@pytest.mark.parametrize("bad_line", ['{"metrics":', "[]", '{"metadata": "wrong"}', '{"metrics": []}'])
def test_malformed_request_records_do_not_publish_partial_p90(asset_mapper, tmp_path, bad_line):
    _run_asset_mapper(asset_mapper, tmp_path, [_request_record(1000, 10)])
    source = tmp_path / "aiperf_artifacts/profile_export_aiperf.json"
    records = source.with_name("profile_export.jsonl")
    with records.open("a", encoding="utf-8") as handle:
        handle.write(bad_line + "\n")
    destination = tmp_path / "second_result.json"
    asset_mapper.main(str(source), str(destination))
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert result["comparison_metrics"]["status"] == "unavailable"
    assert result["comparison_metrics"]["reason"] == "request_records_invalid"
    assert result["comparison_metrics"]["e2e_norm_intvty_p90"] is None
    assert result["output_throughput"] == 500.0


def test_unknown_metric_unit_does_not_silently_scale_p90(asset_mapper, tmp_path):
    record = _request_record(1000, 10)
    record["metrics"]["request_latency"]["unit"] = "seconds"
    result = _run_asset_mapper(asset_mapper, tmp_path, [record])
    assert result["comparison_metrics"]["reason"] == "request_records_invalid"
    assert result["comparison_metrics"]["e2e_norm_intvty_p90"] is None


def test_comparison_reads_only_the_matching_summary_directory(asset_mapper, tmp_path):
    other = tmp_path / "old-run"
    _run_asset_mapper(asset_mapper, other, [_request_record(1000, 10)])
    result = _run_asset_mapper(asset_mapper, tmp_path / "new-run", None)
    assert result["comparison_metrics"]["reason"] == "request_records_missing"


def test_deployed_mapper_computes_comparison_without_installed_hyperloom(tmp_path):
    from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets

    deployed = deploy_agentx_assets(tmp_path / "benchmarks")
    script = next(path for path in deployed if path.name == "map_aiperf.py")
    source = tmp_path / "profile_export_aiperf.json"
    source.write_text(json.dumps(_sample()), encoding="utf-8")
    source.with_name("profile_export.jsonl").write_text(json.dumps(_request_record(1000, 10)) + "\n", encoding="utf-8")
    output = tmp_path / "result.json"
    proc = subprocess.run([sys.executable, "-I", str(script), str(source), str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["comparison_metrics"]["e2e_norm_intvty_p90"] == pytest.approx(10.0)
    assert result["e2e_norm_intvty_p90"] == 22.6
