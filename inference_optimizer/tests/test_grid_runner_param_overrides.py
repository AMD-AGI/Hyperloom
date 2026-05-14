"""Grid runner parameter override + RESULT_DIR plumbing tests.

The grid runner (used by backends / params / sweep) gets the same Magpie
leak-path recovery knobs the baseline executor gets, applied per-variant:

* ``benchmark_script`` rewrites ``benchmark.benchmark_script`` in every
  variant's materialized YAML — applied AFTER the gpu_type-driven pop
  so the operator override beats Magpie's runner_type auto-selection.
* ``result_dir`` (optional) overrides ``$RESULT_DIR``; when omitted,
  ``$RESULT_DIR`` defaults to the per-variant slot so the leak salvage
  path is unnecessary for scripts that honor the env var.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    _build_variant_yaml,
    _run_magpie,
    apply_runtime_benchmark_overrides,
    run_grid,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox so the
    grid runner's always-on artifact harvest does not scrape the
    host's real ``/workspace`` directory during this test module.
    """
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _write_baseline_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "dsr1_fp8_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float = 800.0) -> Path:
    workspace = slot / "benchmark_sglang_20260513_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": tput / 256,
            "output_throughput": tput,
            "total_token_throughput": tput * 2,
            "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {
            "ttft": {"mean_ms": 140.0, "p99_ms": 160.0},
            "e2el": {"mean_ms": 2500.0, "p99_ms": 2800.0},
        },
    }))
    return workspace


def test_apply_runtime_overrides_pins_benchmark_script_after_gpu_pop():
    bench = {
        "framework": "sglang",
        "benchmark_script": "dsr1_fp8_mi300x.sh",
        "envs": {},
    }
    apply_runtime_benchmark_overrides(
        bench, gpu_type="mi300x", benchmark_script="sglang_mi300x.sh",
    )
    assert bench["benchmark_script"] == "sglang_mi300x.sh"
    assert bench["runner_type"] == "mi300x"


def test_build_variant_yaml_propagates_benchmark_script(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    out = _build_variant_yaml(
        base,
        base_extra_args="",
        variant=GridVariant("vA", "--attention-backend aiter"),
        output_subdir=tmp_path / "vA",
        gpu_type="mi300x",
        benchmark_script="sglang_mi300x.sh",
    )
    cfg = yaml.safe_load(out.read_text())
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"
    assert cfg["benchmark"]["runner_type"] == "mi300x"


def test_run_magpie_default_result_dir_is_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )
    assert captured["env"]["RESULT_DIR"] == str(tmp_path / "slot")


def test_run_magpie_explicit_result_dir_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
            result_dir="/tmp/redirect_leak",
        )
    assert captured["env"]["RESULT_DIR"] == "/tmp/redirect_leak"


@pytest.mark.asyncio
async def test_run_grid_forwards_benchmark_script_per_variant(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    grid = [GridVariant("v0"), GridVariant("v1")]
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=5,
            gpu_type="mi300x",
            benchmark_script="sglang_mi300x.sh",
        )

    assert len(results) == 2
    for i in range(2):
        slot = output_root / f"variant_{i:02d}_v{i}"
        cfg = yaml.safe_load((slot / "config.yaml").read_text())
        assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"


@pytest.mark.asyncio
async def test_run_grid_forwards_result_dir_to_subprocess_env(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"
    captured_envs: list[dict] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured_envs.append(dict(kwargs.get("env") or {}))
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    grid = [GridVariant("v0"), GridVariant("v1")]
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=5,
            result_dir="/tmp/redirect",
        )

    assert len(captured_envs) == 2
    for env in captured_envs:
        assert env["RESULT_DIR"] == "/tmp/redirect"


@pytest.mark.asyncio
async def test_run_grid_default_result_dir_is_per_variant_slot(tmp_path):
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"
    captured_envs: list[tuple[str, str]] = []

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        captured_envs.append((str(slot), env["RESULT_DIR"]))
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    grid = [GridVariant("vA"), GridVariant("vB")]
    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=grid, output_root=output_root, variant_timeout_sec=5,
        )

    # Each variant's $RESULT_DIR defaults to its own slot — Magpie scripts
    # that respect the env var land their inferencex_result.json inside
    # the per-variant workspace instead of in /workspace/.
    for slot_path, result_dir in captured_envs:
        assert slot_path == result_dir
    assert len({rd for _, rd in captured_envs}) == 2
