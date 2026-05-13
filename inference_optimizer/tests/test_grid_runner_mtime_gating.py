"""Per-variant mtime gating for Magpie leak-path salvage in ``run_grid``.

Previously ``_grid_runner.py:run_grid`` called
``extract_benchmark_measurement(report, workspace=workspace)`` without
passing ``subprocess_started_unix``. That meant any stale
``/workspace/inferencex_result.json`` left over from a prior baseline /
variant silently masqueraded as the current variant's measurement —
quietly producing fake winners.

The fix captures ``variant_started_unix = time.time()`` immediately
before each ``_run_magpie`` call and forwards it to
``extract_benchmark_measurement(subprocess_started_unix=...)`` so the
salvage path rejects any leak file whose mtime predates the variant's
launch.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    run_grid,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox.

    The grid runner's always-on ``harvest_leaked_artifacts`` pass
    defaults to scanning ``/workspace`` for wrapper-side leak files
    (server.log / gpu_metrics.csv / profile_*.trace.json.gz /
    inferencex_result*.json). On the dev host that directory is real
    and contains old profile traces; without this fixture the harvest
    can mtime-gate-allow a stale artifact into the test workspace
    and corrupt the salvage assertions in
    ``test_run_grid_salvages_fresh_leak_per_variant``. The rescue
    path still honours ``INFERENCE_OPTIMIZER_RESCUE_PATHS`` (set
    per-test via ``monkeypatch.setenv``) so the salvage behaviour
    under test is unchanged.
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
            "benchmark_script": "sglang_mi300x.sh",
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


def _empty_workspace(slot: Path) -> Path:
    """Create a Magpie workspace with NO valid in-workspace result.

    The wrapper reports ``success=False`` and the workspace contains no
    ``inferencex_result.json`` — without salvage the variant would
    record ``valid_measurement=False``.
    """
    ws = slot / "benchmark_sglang_20260513_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(json.dumps({
        "success": False,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
    }))
    return ws


def _write_leak(path: Path, *, tput: float = 1761.6, completed: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "output_throughput": tput,
        "request_throughput": tput / 10,
        "completed_requests": completed,
        "duration_seconds": 120.0,
    }))


@pytest.mark.asyncio
async def test_run_grid_rejects_stale_leak_from_previous_run(
    tmp_path, monkeypatch,
):
    """A leak file written BEFORE the variant launched is never salvaged.

    Without per-variant mtime gating, this stale file would be adopted
    as the current variant's result — silently inflating throughput by
    >2x and producing a fake winner.
    """
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    leak_dir = tmp_path / "stale_leak"
    leak_path = leak_dir / "inferencex_result.json"
    _write_leak(leak_path, tput=9999.0)
    # Force the leak mtime far into the past so it's strictly older than
    # any subsequent ``variant_started_unix`` snapshot.
    stale_mtime = time.time() - 3600.0
    os.utime(leak_path, (stale_mtime, stale_mtime))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _empty_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=[GridVariant("vA")], output_root=output_root,
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    r = results[0]
    # Stale leak → no rescue → variant remains failed (no valid measurement).
    assert r.status == "failed"
    assert r.output_throughput is None
    # The leaked 9999.0 number must NOT bleed into the result.
    assert all(
        "rescued_from_leaked_path" not in (w or "")
        for w in r.nonfatal_warnings
    )


@pytest.mark.asyncio
async def test_run_grid_salvages_fresh_leak_per_variant(tmp_path, monkeypatch):
    """A leak file written DURING the variant subprocess is salvaged.

    The fake ``subprocess.run`` writes a fresh leak file under
    ``$INFERENCE_OPTIMIZER_RESCUE_PATHS``; that mtime is strictly newer
    than ``variant_started_unix`` so salvage adopts it.
    """
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_root = tmp_path / "out"

    leak_dir = tmp_path / "fresh_leak"
    leak_dir.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _empty_workspace(slot)
        # Simulate Magpie writing the leak DURING the run — mtime is
        # strictly after the variant's start snapshot.
        _write_leak(leak_dir / "inferencex_result.json", tput=1234.0)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "inference_optimizer.orchestrator.action_executors._grid_runner."
        "subprocess.run",
        side_effect=fake_run,
    ):
        results = await run_grid(
            base_yaml_path=base, base_extra_args="",
            grid=[GridVariant("vA")], output_root=output_root,
            variant_timeout_sec=5,
        )

    assert len(results) == 1
    r = results[0]
    assert r.status == "succeeded"
    assert r.output_throughput == pytest.approx(1234.0)
    assert any(
        (w or "").startswith("rescued_from_leaked_path:")
        for w in r.nonfatal_warnings
    )
