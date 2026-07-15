# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/report_generator.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import report_generator as rg  # noqa: E402


# ── _fmt_pct / _avg / _first_of ──


def test_fmt_pct():
    assert rg._fmt_pct(None) == "N/A"
    assert rg._fmt_pct(0.0) == "--"
    assert rg._fmt_pct(7.94) == "+7.9%"
    assert rg._fmt_pct(-3.0) == "-3.0%"


def test_avg():
    assert rg._avg([1.0, 2.0, 3.0]) == 2.0
    assert rg._avg([]) is None


def test_first_of():
    assert rg._first_of({"a": None, "b": 5}, "a", "b") == 5
    assert rg._first_of({}, "x") is None


# ── _parse_metrics_from_report ──


def test_parse_metrics_from_report_gpu_row():
    content = "Gain: **12.5%**\n| tok/s/GPU | ~100 | ~120 |\n"
    out = rg._parse_metrics_from_report(content)
    assert out["gain_pct"] == 12.5
    assert out["baseline_throughput"] == 100.0
    assert out["optimized_throughput"] == 120.0


def test_parse_metrics_from_report_output_throughput_row():
    content = "| Output Throughput (tok/s) | 200 | 240 |"
    out = rg._parse_metrics_from_report(content)
    assert out["baseline_throughput"] == 200.0
    assert out["optimized_throughput"] == 240.0


def test_parse_metrics_from_report_empty():
    assert rg._parse_metrics_from_report("nothing here") == {}


# ── _extract_metrics_via_llm ──


def test_extract_metrics_via_llm_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert rg._extract_metrics_via_llm("report") == {}


def test_extract_metrics_via_llm_success(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "baseline_throughput": 10,
                                    "optimized_throughput": 12,
                                    "tok_per_gpu_baseline": 5,
                                    "tok_per_gpu_optimized": 6,
                                    "gain_pct": 20,
                                }
                            )
                        }
                    }
                ]
            }

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp())
    out = rg._extract_metrics_via_llm("report text")
    assert out["gain_pct"] == 20
    assert out["baseline_throughput"] == 10


def test_extract_metrics_via_llm_fenced(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            fenced = "```json\n" + json.dumps({"gain_pct": 3}) + "\n```"
            return {"choices": [{"message": {"content": fenced}}]}

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp())
    assert rg._extract_metrics_via_llm("r")["gain_pct"] == 3


def test_extract_metrics_via_llm_failure(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    import requests

    def boom(*a, **k):
        raise RuntimeError("net down")

    monkeypatch.setattr(requests, "post", boom)
    assert rg._extract_metrics_via_llm("r") == {}


# ── extract_optimization_data ──


def test_extract_optimization_data_ci_metrics(tmp_path: Path):
    (tmp_path / "ci_metrics.json").write_text(
        json.dumps({"tok_per_gpu_baseline": 100, "tok_per_gpu_optimized": 130, "actions_taken": ["a", "b"]}),
        encoding="utf-8",
    )
    data = rg.extract_optimization_data(str(tmp_path))
    assert data["baseline_throughput"] == 100
    assert data["optimized_throughput"] == 130
    assert data["gain_pct"] == 30.0
    assert data["actions"] == ["a", "b"]


def test_extract_optimization_data_report_regex(tmp_path: Path):
    (tmp_path / "optimization_report.md").write_text("Gain **10.0%**\n| tok/s/GPU | ~50 | ~55 |", encoding="utf-8")
    data = rg.extract_optimization_data(str(tmp_path))
    assert data["report_exists"] is True
    assert data["baseline_throughput"] == 50.0
    assert data["optimized_throughput"] == 55.0


def test_extract_optimization_data_nested_schema(tmp_path: Path):
    (tmp_path / "ci_metrics.json").write_text(
        json.dumps(
            {
                "baseline": {"tok_s_per_gpu": 100},
                "optimized": {"tok_s_per_gpu": 110},
                "improvement": {"output_throughput_pct": 10},
            }
        ),
        encoding="utf-8",
    )
    data = rg.extract_optimization_data(str(tmp_path))
    assert data["baseline_throughput"] == 100
    assert data["optimized_throughput"] == 110
    assert data["gain_pct"] == 10.0


def test_extract_optimization_data_empty_dir(tmp_path: Path):
    data = rg.extract_optimization_data(str(tmp_path))
    assert data["baseline_throughput"] is None
    assert data["report_exists"] is False


def test_extract_optimization_data_bad_ci_metrics(tmp_path: Path):
    (tmp_path / "ci_metrics.json").write_text("{bad", encoding="utf-8")
    data = rg.extract_optimization_data(str(tmp_path))
    assert data["baseline_throughput"] is None


# ── build_model_result ──


def test_build_model_result_basic(tmp_path: Path):
    (tmp_path / "ci_metrics.json").write_text(
        json.dumps({"tok_per_gpu_baseline": 100, "tok_per_gpu_optimized": 120}), encoding="utf-8"
    )
    result = rg.build_model_result("M", "k", "img:1", "fp8", "completed", "ts", str(tmp_path))
    assert result["model"] == "M"
    assert result["optimized_tok_per_gpu"] == 120


def test_build_model_result_with_ifx_reference(tmp_path: Path):
    (tmp_path / "ci_metrics.json").write_text(
        json.dumps({"tok_per_gpu_baseline": 80, "tok_per_gpu_optimized": 100}), encoding="utf-8"
    )
    ifx = {"metrics": {"output_tput_per_gpu": 90}, "decode_tp": 8}
    result = rg.build_model_result("M", "k", "img", "fp8", "completed", "ts", str(tmp_path), ifx_reference=ifx)
    assert result["inferenceX_tok_per_gpu"] == 90
    assert result["vs_inferenceX_pct"] is not None


def test_build_model_result_total_throughput_correction(tmp_path: Path):
    # optimized >3x ifx -> treated as total throughput, divided by TP.
    (tmp_path / "ci_metrics.json").write_text(
        json.dumps({"tok_per_gpu_baseline": 800, "tok_per_gpu_optimized": 1000}), encoding="utf-8"
    )
    ifx = {"metrics": {"output_tput_per_gpu": 100}, "decode_tp": 8}
    result = rg.build_model_result("M", "k", "img", "fp8", "completed", "ts", str(tmp_path), ifx_reference=ifx)
    assert result["optimized_tok_per_gpu"] == 125.0  # 1000 / 8


# ── report renderers ──


def _results():
    return [
        {
            "model": "A",
            "precision": "fp8",
            "image": "img:1",
            "status": "completed",
            "baseline_tok_per_gpu": 100,
            "optimized_tok_per_gpu": 120,
            "gain_pct": 20.0,
            "vs_inferenceX_pct": 5.0,
            "inferenceX_tok_per_gpu": 114,
            "report_exists": True,
            "actions": ["x"],
        },
        {"model": "B", "precision": "bf16", "image": "img:2", "status": "failed", "actions": []},
        {"model": "C", "precision": "fp8", "image": "img:3", "status": "timeout"},
    ]


def test_generate_markdown_report():
    md = rg.generate_markdown_report(_results(), "manual", "abcdef1234", "run42")
    assert "# Inference Optimization CI Report" in md
    assert "run42" in md
    assert "abcdef1" in md


def test_generate_json_summary():
    summary = rg.generate_json_summary(_results(), "cron", "deadbeef", "run1")
    assert summary["stats"]["total"] == 3
    assert summary["stats"]["completed"] == 1
    assert summary["stats"]["failed"] == 1
    assert summary["stats"]["timeout"] == 1
    assert summary["stats"]["avg_gain_pct"] == 20.0


def test_generate_github_summary():
    out = rg.generate_github_summary(_results(), "manual", "abcdef1234")
    assert "Hyperloom Inference Optimization Results" in out
    assert "## ✅ A (fp8)" in out
    assert "Optimization Gain" in out


def test_generate_github_summary_overall_table():
    results = [
        {
            "model": f"M{i}",
            "precision": "fp8",
            "image": "i",
            "status": "completed",
            "baseline_tok_per_gpu": 10,
            "optimized_tok_per_gpu": 12,
            "gain_pct": 20.0,
        }
        for i in range(2)
    ]
    out = rg.generate_github_summary(results, "t", "c")
    assert "Overall Summary" in out
