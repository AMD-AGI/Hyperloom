# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for MLPerf harness -> InferenceX-schema mapping."""

from __future__ import annotations

import json
import sys

from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets
from hyperloom.inference_optimizer.agentx.mapping import map_mlperf


def _summary(**over):
    data = {
        "complete": True,
        "tps": 1200.0,
        "e2e_avg_interactivity": 75.0,
        "duration_ns": 10_000_000_000,
        "n_samples_completed": 150,
        "n_samples_failed": 0,
        "output_sequence_lengths": {"total": 12000, "avg": 80},
        "input_sequence_lengths": {"total": 30000, "avg": 200},
        "ttft": {"avg": 12_000_000, "percentiles": {"50": 10_000_000, "99": 20_000_000}},
        "tpot": {"avg": 8_000_000, "percentiles": {"50": 7_000_000, "90": 9_000_000, "99": 11_000_000}},
        "latency": {"avg": 900_000_000, "percentiles": {"50": 800_000_000, "99": 1_200_000_000}},
    }
    data.update(over)
    return data


def test_map_mlperf_happy_path():
    mapped = map_mlperf(_summary(), accuracy={"score": 0.91})
    assert mapped["output_throughput"] == 1200.0
    assert mapped["e2e_norm_intvty_p90"] == 75.0
    assert mapped["completed"] == 150
    assert mapped["request_error_rate"] == 0.0
    assert mapped["submission_valid"] is True
    assert mapped["accuracy_score"] == 0.91
    assert mapped["mean_ttft_ms"] == 12.0
    assert mapped["p99_tpot_ms"] == 11.0
    assert mapped["corpus_loader"] == "agentic_combined_v6"


def test_map_mlperf_incomplete_fail_closed():
    mapped = map_mlperf(_summary(complete=False), accuracy={"score": 0.9})
    assert mapped["submission_valid"] is False
    assert "incomplete_run" in mapped["submission_invalid_reasons"]
    assert mapped["mlperf_complete"] is False


def test_map_mlperf_error_rate():
    mapped = map_mlperf(_summary(n_samples_completed=90, n_samples_failed=10))
    assert mapped["request_error_rate"] == 10.0


def test_deployed_map_mlperf_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "agentx_mapping", raising=False)
    deployed = deploy_agentx_assets(tmp_path / "benchmarks")
    mapper = next(path for path in deployed if path.name == "map_mlperf.py")
    src = tmp_path / "result_summary.json"
    acc = tmp_path / "accuracy_results.json"
    dst = tmp_path / "inferencex_result.json"
    src.write_text(json.dumps(_summary()), encoding="utf-8")
    acc.write_text(json.dumps({"score": 0.88}), encoding="utf-8")
    import runpy

    runpy.run_path(str(mapper), run_name="not_main")
    from importlib.machinery import SourceFileLoader

    mod = SourceFileLoader("map_mlperf_cli", str(mapper)).load_module()
    mod.main(str(src), str(dst), str(acc))
    out = json.loads(dst.read_text(encoding="utf-8"))
    assert out["accuracy_score"] == 0.88
    assert out["output_throughput"] == 1200.0
