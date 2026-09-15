# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for aiperf export -> InferenceX-schema mapping."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

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
    """Scoring and comparison use the summary rate P10, not P90."""
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


@pytest.mark.parametrize("standalone", [False, True], ids=["installed-package", "standalone-fallback"])
@pytest.mark.parametrize(
    "records",
    [
        None,
        '{"metrics": {"request_latency": 1000, "time_to_first_token": 10, '
        '"input_sequence_length": 128, "output_sequence_length": 10}}\n',
        '{"metrics":\n',
    ],
    ids=["missing-records", "valid-records", "malformed-records"],
)
def test_file_mapper_uses_only_summary(monkeypatch, tmp_path, capsys, standalone, records):
    from hyperloom.inference_optimizer.agentx.deploy import agentx_asset_dir

    monkeypatch.delenv("AGENTX_NONCANONICAL_REASONS", raising=False)
    if standalone:
        monkeypatch.setitem(sys.modules, "hyperloom.inference_optimizer.agentx.mapping", None)
    spec = importlib.util.spec_from_file_location("_summary_asset_mapper", agentx_asset_dir() / "map_aiperf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "profile_export_aiperf.json"
    source.write_text(json.dumps(_sample()), encoding="utf-8")
    records_path = source.with_name("profile_export.jsonl")
    if records is not None:
        records_path.write_text(records, encoding="utf-8")
    output = tmp_path / "inferencex_result.json"
    with patch("builtins.open", wraps=open) as builtin_open, patch("io.open", wraps=io.open) as io_open:
        module.main(str(source), str(output))
    for call in builtin_open.call_args_list + io_open.call_args_list:
        assert Path(call.args[0]) != records_path

    result = json.loads(output.read_text(encoding="utf-8"))
    assert "comparison_metrics" not in result
    assert result["e2e_norm_intvty_p90"] == 22.6
    assert result == map_aiperf(_sample())
    assert json.loads(capsys.readouterr().out) == result


def test_deployed_mapper_maps_summary_without_installed_hyperloom(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets

    monkeypatch.delenv("AGENTX_NONCANONICAL_REASONS", raising=False)
    deployed = deploy_agentx_assets(tmp_path / "benchmarks")
    script = next(path for path in deployed if path.name == "map_aiperf.py")
    source = tmp_path / "profile_export_aiperf.json"
    source.write_text(json.dumps(_sample()), encoding="utf-8")
    output = tmp_path / "result.json"
    proc = subprocess.run([sys.executable, "-I", str(script), str(source), str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert "comparison_metrics" not in result
    assert result["e2e_norm_intvty_p90"] == 22.6
    assert result == map_aiperf(_sample())
    assert json.loads(proc.stdout) == result
