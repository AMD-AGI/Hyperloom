# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive ``BaselineExecutor`` end to end over a mocked Magpie subprocess."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

QWEN3_8B = "/path/models/Qwen-Qwen3-8B"
RUN_WITH_SESSION_KILL = "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill"


def write_sglang_yaml(path: Path, model: str = QWEN3_8B) -> Path:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path


def fake_workspace(slot: Path, *, tput: float = 1500.0) -> Path:
    """A successful sglang benchmark workspace under ``slot`` reporting ``tput`` output tok/s."""
    ws = slot / "benchmark_sglang_20260513_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": QWEN3_8B,
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
            }
        )
    )
    return ws


def capture_launch(captured: dict):
    """A Magpie run that records the launched config and env, then succeeds."""

    def fake_run(cmd, *args, **kwargs):
        captured["cfg"] = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
        captured["env"] = dict(kwargs.get("env") or {})
        fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    return fake_run


def run_baseline(base: Path, session_dir: Path, ctx, fake_run, **executor_kwargs) -> dict:
    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=session_dir,
        **executor_kwargs,
    )
    with patch(RUN_WITH_SESSION_KILL, side_effect=fake_run):
        return asyncio.run(executor(ctx))
