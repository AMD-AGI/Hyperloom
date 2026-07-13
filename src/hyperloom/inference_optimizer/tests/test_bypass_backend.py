# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Stage-2a tests for the bypass benchmark backend.

Covers backend selection, the Magpie-compatible report contract (parsed by the
same extract_benchmark_measurement Hyperloom uses), and RUN_EVAL results
parsing. No real GPU/server: the InferenceX subprocess is monkeypatched to
drop a fake inferencex_result.json into the workspace.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb
from hyperloom.orchestrator.actions.executors import bypass_report
from hyperloom.orchestrator.actions.executors import bypass_runner
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement,
)


def test_bypass_backend_selected(monkeypatch):
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "bypass")
    backend = bb.resolve_backend()
    assert backend.name == "bypass"
    cmd = backend.build_command(
        python_exe="PY",
        config_path=Path("/cfg.yaml"),
        output_dir=Path("/out"),
    )
    assert cmd == [
        "PY",
        "-m",
        "hyperloom.orchestrator.actions.executors.bypass_runner",
        "benchmark",
        "--benchmark-config",
        "/cfg.yaml",
        "--output-dir",
        "/out",
        "--run-mode",
        "local",
    ]


def test_bypass_report_is_measurement_compatible():
    raw = {
        "request_throughput": 2.0,
        "output_throughput": 1234.5,
        "total_token_throughput": 2000.0,
        "completed": 64,
        "duration": 60.0,
        "mean_ttft_ms": 100.0,
        "p99_ttft_ms": 200.0,
        "mean_tpot_ms": 5.0,
        "mean_e2el_ms": 2000.0,
        "p99_e2el_ms": 3000.0,
    }
    report = bypass_report.build_report(
        raw,
        framework="sglang",
        model="/models/x",
        success=True,
        workspace_dir="/ws/benchmark_sglang_x",
        execution_time=61.0,
    )
    m = extract_benchmark_measurement(report)
    assert m["valid_measurement"] is True
    assert m["output_throughput"] == 1234.5
    assert m["completed_requests"] == 64
    assert m["ttft_mean_ms"] == 100.0
    assert m["e2el_mean_ms"] == 2000.0


def test_bypass_run_writes_compatible_workspace(tmp_path, monkeypatch):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    # Provide the generic script the 3-tier resolver expects.
    (inferencex / "benchmarks" / "sglang_mi300x.sh").write_text(
        "#!/bin/bash\necho fake\n", encoding="utf-8"
    )

    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/models/x",
            "precision": "bf16",
            "runner_type": "mi300x",
            "run_mode": "local",
            "inferencex_path": str(inferencex),
            "timeout_seconds": 60,
            "envs": {"TP": 1, "CONC": 4, "ISL": 128, "OSL": 64, "RUN_EVAL": "false"},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def fake_run(cmd, env=None, capture_output=True, text=True, timeout=None):
        result_dir = Path(env["RESULT_DIR"])
        raw = {
            "output_throughput": 999.0,
            "request_throughput": 1.0,
            "total_token_throughput": 1500.0,
            "completed": 40,
            "duration": 30.0,
            "mean_ttft_ms": 50.0,
            "mean_e2el_ms": 900.0,
        }
        (result_dir / "inferencex_result.json").write_text(
            json.dumps(raw), encoding="utf-8"
        )

        class _P:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 0

    workspaces = list((tmp_path / "out").glob("benchmark_sglang_*"))
    assert len(workspaces) == 1
    ws = workspaces[0]
    report = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["framework"] == "sglang"

    m = extract_benchmark_measurement(report, workspace=ws)
    assert m["valid_measurement"] is True
    assert m["output_throughput"] == 999.0
    assert m["completed_requests"] == 40


def test_bypass_run_missing_inferencex_fails(tmp_path):
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/models/x",
            "run_mode": "local",
            "inferencex_path": str(tmp_path / "does-not-exist"),
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
    assert rc == 2
    workspaces = list((tmp_path / "out").glob("benchmark_sglang_*"))
    assert len(workspaces) == 1
    report = json.loads(
        (workspaces[0] / "benchmark_report.json").read_text(encoding="utf-8")
    )
    assert report["success"] is False
    assert report["errors"]


def test_bypass_cli_rejects_non_local(tmp_path):
    rc = bypass_runner.main(
        [
            "benchmark",
            "--benchmark-config",
            str(tmp_path / "c.yaml"),
            "--output-dir",
            str(tmp_path / "o"),
            "--run-mode",
            "docker",
        ]
    )
    assert rc == 2