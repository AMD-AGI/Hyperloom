"""Unit tests for kernel_profiling collector (v1.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import build


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_kernel_profiling_from_profile_run(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kp"})
    _write_json(sd / "state.json", {
        "session_id": "kp",
        "framework": "sglang",
        "profile_attempts": [{
            "ts": "2026-05-15T11:00:00+00:00",
            "task_id": "p1",
            "status": "succeeded",
            "decision": "promoted",
            "extras": {"profile_args": "python -m sglang.launch_server --tp 8"},
        }],
    })
    pdir = sd / "runs/profile/p1/benchmark_001"
    _write_json(pdir / "benchmark_report.json", {
        "success": True,
        "kernel_summary": [
            {"kernel_id": "k001", "name": "rmsnorm", "gpu_pct": 12.0,
             "time_ms": 0.5, "bottleneck": "memory"},
        ],
    })
    trace_dir = pdir / "torch_trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "run.trace.json.gz").write_bytes(b"\x1f\x8b")

    runs = build(sd)["kernel_profiling"]
    assert len(runs) == 1
    run = runs[0]
    assert run["task_id"] == "p1"
    assert run["outputs"]["tool"] == "magpie_torch_profiler"
    assert run["outputs"]["top_kernels"][0]["kernel_id"] == "k001"
    assert run["artifacts"]["trace_paths"]
    assert "sglang.launch_server" in run["launch"]["framework_args"]


def test_kernel_profiling_tracelens_status_json(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "tl"})
    _write_json(sd / "state.json", {"session_id": "tl", "framework": "vllm"})
    kar = sd / "kernel-agent/runs/sess-tl"
    status_dir = kar / "status/tracelens_analysis"
    status_dir.mkdir(parents=True)
    _write_json(status_dir / "run-abc.json", {
        "status": "ok",
        "summary": "3 kernels profiled",
        "top_kernels": [{"kernel_id": "k99", "name": "gemm", "gpu_pct": 20.0}],
    })

    runs = build(sd)["kernel_profiling"]
    assert len(runs) == 1
    assert runs[0]["outputs"]["tool"] == "tracelens_analysis"
    assert runs[0]["outputs"]["analysis_summary"] == "3 kernels profiled"
    assert runs[0]["artifacts"]["tracelens_status_json"] is not None
