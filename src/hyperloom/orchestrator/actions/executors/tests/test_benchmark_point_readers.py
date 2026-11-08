# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.common.benchmark_config import BenchmarkPointError
from hyperloom.orchestrator.actions.executors import _grid_runner
from hyperloom.orchestrator.actions.executors._accuracy_gate import materialized_run_eval_disabled
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.actions.executors._launch_restatement import read_launch

_NO_BLOCK = {"other": {}}


def _point(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_a_config_without_a_benchmark_block_keeps_the_eval_on(tmp_path: Path) -> None:
    assert materialized_run_eval_disabled(_point(tmp_path, _NO_BLOCK)) is False


def test_the_eval_switch_reads_the_point(tmp_path: Path) -> None:
    path = _point(tmp_path, {"benchmark": {"framework": "vllm", "model": "/m", "envs": {"RUN_EVAL": "0", "TP": 4}}})

    assert materialized_run_eval_disabled(path) is True


def test_grid_readers_of_a_point(tmp_path: Path) -> None:
    path = _point(tmp_path, {"benchmark": {"framework": "xdit", "envs": {"TP": 4}}})

    assert _grid_runner.lease_cards(path, tmp_path) == 4
    assert _grid_runner.lease_cards(_point(tmp_path, _NO_BLOCK), tmp_path) == 1


def test_launch_evidence_of_a_config_that_is_not_a_point_declares_nothing(tmp_path: Path) -> None:
    path = _point(tmp_path, _NO_BLOCK)

    evidence = build_launch_evidence(RequestedLaunch(path, "vllm"), actual_server_log=None, slot=tmp_path)

    assert (evidence["framework"], evidence["model_path"], evidence["requested_server_env"]) == ("vllm", "", {})
    assert evidence["materialized_config_path"] == str(path)


def test_an_agentx_launch_is_restated_from_its_overrides(tmp_path: Path) -> None:
    overrides = {
        "env": {"A": 1},
        "executable": "/py",
        "append_args": ["--x", "2"],
        "remove_args": ["--b", "--a"],
        "replace_args": True,
        "unset_env": ["B"],
        "source_files": {"/dst.py": "/src.py"},
    }
    path = _point(
        tmp_path,
        {"benchmark": {"framework": "sglang", "envs": {}, "agentx": {"launch_overrides": overrides}}},
    )

    launch = read_launch(path, ["C"])

    assert dict(launch.settings) == {"x": ["2"]}
    assert dict(launch.envs) == {"A": "1"}
    assert dict(launch.controls) == {
        "unset_envs": ["C"],
        "remove_args": ["--a", "--b"],
        "replace_args": True,
        "unset_env": ["B"],
        "executable": "/py",
        "source_files": {"/dst.py": "/src.py"},
    }


def test_a_launch_without_a_benchmark_block_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkPointError):
        read_launch(_point(tmp_path, _NO_BLOCK))
