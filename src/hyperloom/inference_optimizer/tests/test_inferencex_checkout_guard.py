# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round must refuse to launch from a tree that is not an InferenceX checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    _run_magpie,
    run_grid,
    MagpieRound,
    GridLaunch,
    GridRecipe,
)
from hyperloom.orchestrator.actions.executors._subprocess_kill import INFERENCEX_CHECKOUT_INVALID_RETURNCODE
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

_GRID = "hyperloom.orchestrator.actions.executors._grid_runner"


def _config(path: Path, **benchmark: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/models/m",
                    "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256, "RUN_EVAL": "false"},
                    "timeout_seconds": 600,
                    **benchmark,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def framework_tree(tmp_path: Path) -> Path:
    """A framework source tree: it has benchmarks of its own but not InferenceX's library."""
    tree = tmp_path / "sglang"
    (tree / "benchmarks").mkdir(parents=True)
    return tree


@pytest.fixture
def inferencex_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "InferenceX"
    (tree / "benchmarks").mkdir(parents=True)
    (tree / "benchmarks" / "benchmark_lib.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return tree


def _run(tmp_path: Path, launched: list[object]) -> tuple[int, str, str]:
    with patch(
        f"{_GRID}.run_with_session_kill",
        side_effect=lambda cmd, *a, **k: launched.append(cmd) or subprocess.CompletedProcess(cmd, 0, "ok", ""),
    ):
        return _run_magpie(
            MagpieRound(
                magpie_python="/opt/venv/bin/python",
                config_path=_config(tmp_path / "config.yaml"),
                output_dir=tmp_path / "slot",
                timeout_sec=5,
                cwd=str(tmp_path),
            )
        )


def test_a_round_does_not_launch_from_a_tree_without_the_benchmark_library(tmp_path, monkeypatch, framework_tree):
    monkeypatch.setenv("INFERENCEX_PATH", str(framework_tree))
    launched: list[object] = []

    rc, _out, err = _run(tmp_path, launched)

    assert rc == INFERENCEX_CHECKOUT_INVALID_RETURNCODE
    assert launched == [], "Magpie must not start a server whose script cannot source its library"
    assert "benchmarks/benchmark_lib.sh" in err
    assert str(framework_tree) in err


def test_a_round_launches_from_an_inferencex_checkout(tmp_path, monkeypatch, inferencex_tree):
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex_tree))
    launched: list[object] = []

    rc, _out, _err = _run(tmp_path, launched)

    assert rc == 0
    assert len(launched) == 1


@pytest.mark.asyncio
async def test_run_grid_reports_the_invalid_checkout_as_the_variant_error(tmp_path, monkeypatch, framework_tree):
    monkeypatch.setenv("INFERENCEX_PATH", str(framework_tree))

    with patch(f"{_GRID}.run_with_session_kill") as launch:
        results = await run_grid(
            grid=[GridVariant("vA")],
            output_root=tmp_path / "out",
            recipe=GridRecipe(base_yaml_path=_config(tmp_path / "base.yaml")),
            launch=GridLaunch(warmup_before_measure=False),
        )

    launch.assert_not_called()
    assert [(r.status, r.returncode) for r in results] == [("failed", INFERENCEX_CHECKOUT_INVALID_RETURNCODE)]
    assert "benchmarks/benchmark_lib.sh" in results[0].error


def test_baseline_refuses_a_config_pinning_a_tree_without_the_benchmark_library(tmp_path, monkeypatch, framework_tree):
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    config = _config(tmp_path / "baseline.yaml", inferencex_path=str(framework_tree))

    out = BaselineExecutor()._after_materialize_config(config, tmp_path / "out")

    assert isinstance(out, dict)
    assert out["status"] == "failed"
    assert out["error_class"] == "inferencex_checkout_invalid"
    assert "benchmarks/benchmark_lib.sh" in out["error"]
