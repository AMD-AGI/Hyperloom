# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the bypass report writer (pure functions, no GPU/server)."""

from __future__ import annotations

import json

from hyperloom.orchestrator.actions.executors import bypass_report as br


def test_create_workspace_layout(tmp_path):
    ws = br.create_workspace(tmp_path, "sglang")
    assert ws.is_dir()
    assert ws.name.startswith("benchmark_sglang_")
    assert (ws / "torch_trace").is_dir()
    assert (ws / "system_profile").is_dir()


def test_build_report_reads_non_finite_client_values_as_zero():
    """A client that reports inf/NaN still yields a report, written as strict JSON."""
    report = br.build_report(
        br.ReportSubject("vllm", "/models/x"),
        br.RunOutcome(
            False,
            raw_result={"completed": float("inf"), "output_throughput": float("inf"), "mean_ttft_ms": float("nan")},
        ),
        workspace_dir="/ws",
        execution_time=1.0,
    )
    assert report["throughput"]["completed_requests"] == 0
    assert report["throughput"]["output_throughput"] == 0.0
    assert report["latency"]["ttft"]["mean_ms"] == 0.0
    json.dumps(report, allow_nan=False)


def test_build_report_without_raw_leaves_metrics_none():
    report = br.build_report(
        br.ReportSubject("vllm", "/models/x"), br.RunOutcome(False, ["boom"]), workspace_dir="/ws", execution_time=1.0
    )
    assert report["success"] is False
    assert report["throughput"] is None
    assert report["latency"] is None
    assert report["errors"] == ["boom"]
    assert report["profiling_enabled"] is False
    # model falls back to raw model_id only when model is empty.
    assert report["model"] == "/models/x"


def test_build_report_model_falls_back_to_raw_model_id():
    report = br.build_report(
        br.ReportSubject("vllm", ""),
        br.RunOutcome(True, raw_result={"model_id": "raw-model", "output_throughput": 1.0}),
        workspace_dir="/ws",
        execution_time=1.0,
    )
    assert report["model"] == "raw-model"


def test_build_report_carries_scriptable_extras():
    report = br.build_report(
        br.ReportSubject("xdit", "/models/flux"),
        br.RunOutcome(
            True,
            raw_result={
                "output_throughput": 1.5,
                "workload_kind": "scriptable",
                "throughput_unit": "img/s",
                "quality_gate": {"passed": True},
                "latency_s": 0.66,
            },
        ),
        workspace_dir="/ws",
        execution_time=2.0,
    )
    assert report["workload_kind"] == "scriptable"
    assert report["throughput_unit"] == "img/s"
    assert report["quality_gate"] == {"passed": True}
    assert report["latency_s"] == 0.66


def test_format_summary_text_minimal_report():
    # No throughput/latency dicts, no errors: only the header lines are emitted.
    report = {
        "success": False,
        "framework": "sglang",
        "model": "/m",
        "profiling_enabled": False,
        "execution_time": 3.0,
    }
    text = br.format_summary_text(report)
    assert "success: False" in text
    assert "framework: sglang" in text
    assert "output_throughput" not in text
    assert "errors:" not in text
    assert text.endswith("\n")


def test_format_summary_text_with_errors_and_latency():
    report = {
        "success": False,
        "framework": "vllm",
        "model": "/m",
        "profiling_enabled": True,
        "execution_time": 4.0,
        "throughput": {"output_throughput": 10.0},
        "latency": {"ttft": {"mean_ms": 5.0}, "tpot": {"mean_ms": 1.0}},
        "errors": ["run_eval failed with exit code 1"],
    }
    text = br.format_summary_text(report)
    assert "output_throughput: 10.0" in text
    assert "mean_ttft_ms: 5.0" in text
    assert "mean_tpot_ms: 1.0" in text
    assert "errors:" in text
    assert "  - run_eval failed with exit code 1" in text


def test_write_log_aliases_missing_logs_writes_nothing(tmp_path):
    # No per-phase logs present: no alias files are created (best-effort no-op).
    br.write_log_aliases(tmp_path)
    assert not (tmp_path / "benchmark_stdout.log").exists()
    assert not (tmp_path / "benchmark_stderr.log").exists()


def test_write_log_aliases_aggregates_phase_logs(tmp_path):
    (tmp_path / "client_stdout.log").write_text("client-out\n", encoding="utf-8")
    (tmp_path / "eval_stdout.log").write_text("eval-out\n", encoding="utf-8")
    (tmp_path / "scriptable_stderr.log").write_text("script-err\n", encoding="utf-8")
    br.write_log_aliases(tmp_path)
    stdout = (tmp_path / "benchmark_stdout.log").read_text(encoding="utf-8")
    assert "client-out" in stdout
    assert "eval-out" in stdout
    stderr = (tmp_path / "benchmark_stderr.log").read_text(encoding="utf-8")
    assert "script-err" in stderr


def test_write_report_emits_all_artifacts(tmp_path):
    report = br.build_report(
        br.ReportSubject("sglang", "/models/x", profiling_enabled=True),
        br.RunOutcome(True, raw_result={"output_throughput": 700.0, "completed": 40, "duration": 30.0}),
        workspace_dir=str(tmp_path),
        execution_time=30.0,
    )
    path = br.write_report(tmp_path, report)
    assert path == tmp_path / "benchmark_report.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["profiling_enabled"] is True
    assert (tmp_path / "summary.txt").read_text(encoding="utf-8").startswith("success: True")
