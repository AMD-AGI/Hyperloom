# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ci/artifact_normalizer.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import artifact_normalizer as an  # noqa: E402


# ── small coercion helpers ──


def test_read_json_none():
    assert an._read_json(None, []) is None


def test_read_json_ok(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert an._read_json(p, []) == {"x": 1}


def test_read_json_bad(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text("{bad", encoding="utf-8")
    warnings: list[str] = []
    assert an._read_json(p, warnings) is None
    assert warnings and "failed to parse" in warnings[0]


def test_to_float_variants():
    assert an._to_float(None) is None
    assert an._to_float(5) == 5.0
    assert an._to_float("1,024.5") == 1024.5
    assert an._to_float("SKIPPED") is None
    assert an._to_float("") is None
    assert an._to_float("abc") is None


def test_to_int_variants():
    assert an._to_int("12.9") == 12
    assert an._to_int(None) is None


def test_first_of():
    assert an._first_of({"a": None, "b": 2}, "a", "b") == 2
    assert an._first_of({}, "a") is None


def test_first_nested():
    data = {"baseline": {"output_throughput_per_gpu": 3.0}}
    assert an._first_nested(data, "missing", "baseline.output_throughput_per_gpu") == 3.0
    assert an._first_nested({}, "a.b") is None


def test_relative(tmp_path: Path):
    root = tmp_path
    inside = tmp_path / "a" / "b.txt"
    assert an._relative(inside, root) == "a/b.txt"
    assert an._relative(None, root) is None
    # outside root -> absolute posix
    outside = Path("/totally/other/x.txt")
    assert an._relative(outside, root) == outside.as_posix()


# ── find_artifact ──


def test_find_artifact_by_tail(tmp_path: Path):
    f = tmp_path / "hyperloom" / "ci_metrics.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}", encoding="utf-8")
    found = an.find_artifact(tmp_path, "ci_metrics.json")
    assert found == f


def test_find_artifact_missing_root(tmp_path: Path):
    assert an.find_artifact(tmp_path / "nope", "x.json") is None


def test_find_artifact_no_match(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert an.find_artifact(tmp_path, "ci_metrics.json") is None


# ── parse_env_file ──


def test_parse_env_file(tmp_path: Path):
    p = tmp_path / "run_context.env"
    p.write_text("# comment\nFOO='bar'\nBAZ=\"qux\"\nNOEQ\n\n", encoding="utf-8")
    assert an.parse_env_file(p) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_env_file_missing(tmp_path: Path):
    assert an.parse_env_file(tmp_path / "nope.env") == {}
    assert an.parse_env_file(None) == {}


# ── parse_ci_metrics ──


def test_parse_ci_metrics_flat():
    data = {"baseline_throughput": "100", "optimized_throughput": "120", "tp": "8", "conc": "64", "model": "m"}
    m = an.parse_ci_metrics(data)
    assert m["baseline_throughput"] == 100.0
    assert m["optimized_throughput"] == 120.0
    assert m["gain_pct"] == 20.0  # computed
    assert m["tp"] == 8


def test_parse_ci_metrics_ratio_gain():
    data = {"best": {"speedup_vs_baseline": 1.10}}
    m = an.parse_ci_metrics(data)
    assert round(m["gain_pct"], 1) == 10.0


def test_parse_ci_metrics_none():
    m = an.parse_ci_metrics(None)
    assert m["baseline_throughput"] is None
    assert m["actions"] == []


def test_parse_ci_metrics_explicit_gain():
    m = an.parse_ci_metrics({"gain_pct": 7.5})
    assert m["gain_pct"] == 7.5


# ── parse_baseline_summary ──


def test_parse_baseline_summary():
    m = an.parse_baseline_summary({"baseline_tput_per_gpu": "50", "tp": "8"})
    assert m["baseline_tput_per_gpu"] == 50.0
    assert m["tp"] == 8


def test_parse_baseline_summary_none():
    assert an.parse_baseline_summary(None)["baseline_tput_per_gpu"] is None


# ── parse_sweep_results ──


def test_parse_sweep_results(tmp_path: Path):
    p = tmp_path / "sweep_results.csv"
    p.write_text(
        "CONC,ISL,OSL,NUM_PROMPTS,output_throughput_tok_s,mean_tpot_ms,mean_ttft_ms\n"
        "64,1024,1024,100,123.4,5.6,7.8\n"
        "SKIPPED,1024,1024,100,,,\n",
        encoding="utf-8",
    )
    pts = an.parse_sweep_results(p, [])
    assert len(pts) == 2
    assert pts[0]["status"] == "ok"
    assert pts[1]["status"] == "skipped"


def test_parse_sweep_results_missing(tmp_path: Path):
    assert an.parse_sweep_results(tmp_path / "no.csv", []) == []
    assert an.parse_sweep_results(None, []) == []


# ── parse_kernel_candidates / results ──


def test_parse_kernel_candidates():
    data = [{"rank": "1", "name": "k", "gpu_pct": "10.5", "count": "3", "time_ms": "1.2"}, "bad"]
    out = an.parse_kernel_candidates(data)
    assert len(out) == 1
    assert out[0]["rank"] == 1
    assert out[0]["gpu_pct"] == 10.5


def test_parse_kernel_candidates_non_list():
    assert an.parse_kernel_candidates({"x": 1}) == []


def test_parse_kernel_results():
    data = {"kernels": [{"name": "k", "micro_speedup": "1.5"}, 5], "summary": {"total": 1}}
    kernels, summary = an.parse_kernel_results(data)
    assert len(kernels) == 1
    assert kernels[0]["micro_speedup"] == 1.5
    assert summary == {"total": 1}


def test_parse_kernel_results_none():
    kernels, summary = an.parse_kernel_results(None)
    assert kernels == []
    assert summary == {}


# ── classify_artifact ──


def test_classify_artifact():
    assert an.classify_artifact("x/ci_metrics.json") == "ci_metrics"
    assert an.classify_artifact("session_breakdown_v2.json") == "session_breakdown"
    assert an.classify_artifact("results/baseline_summary.json") == "baseline_summary"
    assert an.classify_artifact("results/sweep_results.csv") == "sweep_results"
    assert an.classify_artifact("kernel_candidates.json") == "kernel_candidates"
    assert an.classify_artifact("kernel_opt/kernel_results.json") == "kernel_results"
    assert an.classify_artifact("optimization_report.md") == "optimization_report"
    assert an.classify_artifact("results/run_context.env") == "run_context"
    assert an.classify_artifact("server.log") == "log"
    assert an.classify_artifact("misc.bin") == "artifact"


# ── build_artifact_index ──


def test_build_artifact_index(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "ci_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "x.log").write_text("hello", encoding="utf-8")
    idx = an.build_artifact_index(tmp_path)
    kinds = {e["kind"] for e in idx}
    assert "ci_metrics" in kinds and "log" in kinds
    assert all("size_bytes" in e for e in idx)


def test_build_artifact_index_missing(tmp_path: Path):
    assert an.build_artifact_index(tmp_path / "nope") == []


# ── normalize_task_result (end-to-end) ──


def test_normalize_task_result_full(tmp_path: Path):
    task_dir = tmp_path / "task1"
    (task_dir / "hyperloom").mkdir(parents=True)
    (task_dir / "hyperloom" / "ci_metrics.json").write_text(
        json.dumps({"baseline_throughput": 100, "optimized_throughput": 130, "model": "org/m", "tp": 8}),
        encoding="utf-8",
    )
    (task_dir / "results").mkdir()
    (task_dir / "results" / "sweep_results.csv").write_text("CONC,ISL,OSL\n64,1024,1024\n", encoding="utf-8")
    record = {
        "task_id": "task1",
        "model": "org/m",
        "display_name": "m",
        "status": "submitted",
        "final_status": "Succeeded",
    }
    result = an.normalize_task_result(task_dir, record, {"source": "test"})
    assert result["schema_version"] == an.SCHEMA_VERSION
    assert result["task"]["task_id"] == "task1"
    assert result["metrics"]["gain_pct"] == 30.0
    assert result["sweep_points"]
    assert result["source_files"]["ci_metrics"] is not None


def test_normalize_task_result_missing_dir(tmp_path: Path):
    result = an.normalize_task_result(tmp_path / "nope", {"task_id": "t"}, None)
    assert result["task"]["task_id"] == "t"
    assert result["metrics"]["baseline_throughput"] is None
    assert result["run"] == {}


def test_normalize_task_result_session_breakdown_fallback(tmp_path: Path):
    task_dir = tmp_path / "t"
    task_dir.mkdir()
    # only a session_breakdown variant via rglob fallback
    (task_dir / "session_breakdown_2026.json").write_text(
        json.dumps({"baseline_throughput": 10, "optimized_throughput": 11}), encoding="utf-8"
    )
    result = an.normalize_task_result(task_dir, {"task_id": "t"}, None)
    assert result["source_files"]["session_breakdown"] is not None


# ── collect_normalized_results ──


def test_collect_normalized_results(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    manifests = tmp_path / "manifests"
    (artifacts / "task1").mkdir(parents=True)
    (artifacts / "task1" / "ci_metrics.json").write_text(
        json.dumps({"baseline_throughput": 1, "optimized_throughput": 2}), encoding="utf-8"
    )
    manifests.mkdir()
    (manifests / "submission_manifest.json").write_text(
        json.dumps({"submitted_at": "now", "records": [{"task_id": "task1", "model": "m"}]}), encoding="utf-8"
    )
    results = an.collect_normalized_results(artifacts, manifests, {"source": "ci"})
    assert len(results) == 1
    assert results[0]["task"]["task_id"] == "task1"


def test_collect_normalized_results_no_manifests(tmp_path: Path):
    (tmp_path / "m").mkdir()
    assert an.collect_normalized_results(tmp_path / "a", tmp_path / "m", None) == []


def test_collect_normalized_results_malformed_manifest(tmp_path: Path):
    manifests = tmp_path / "m"
    manifests.mkdir()
    (manifests / "submission_manifest.json").write_text("{bad", encoding="utf-8")
    assert an.collect_normalized_results(tmp_path / "a", manifests, None) == []


# ── write_single_result / main ──


def test_write_single_result(tmp_path: Path):
    an.write_single_result({"a": 1}, tmp_path / "out")
    assert (tmp_path / "out" / "normalized_result.json").exists()
    assert (tmp_path / "out" / "normalized_results.json").exists()
    assert (tmp_path / "out" / "normalized_results.ndjson").exists()


def test_main_success(tmp_path: Path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "ci_metrics.json").write_text(
        json.dumps({"baseline_throughput": 1, "optimized_throughput": 2}), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["a.py", "--task-dir", str(task_dir), "--out-dir", str(out_dir)])
    assert an.main() == 0
    assert (out_dir / "normalized_result.json").exists()


def test_main_missing_task_dir(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["a.py", "--task-dir", str(tmp_path / "nope")])
    assert an.main() == 2
    assert "not found" in capsys.readouterr().err
