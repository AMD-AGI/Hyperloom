# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.common.benchmark_config import BenchmarkPointError
from hyperloom.orchestrator.actions.executors import _grid_runner
from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    _extract_eval_contract_fields,
    materialized_run_eval_disabled,
)
from hyperloom.orchestrator.actions.executors._geak_sweep import _replay_serving_env
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.actions.executors._launch_restatement import read_launch

_NO_BLOCK = {"other": {}}


def _point(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_a_config_without_a_benchmark_block_keeps_the_eval_on_and_states_no_contract(tmp_path: Path) -> None:
    path = _point(tmp_path, _NO_BLOCK)

    assert materialized_run_eval_disabled(path) is False
    assert _extract_eval_contract_fields(path) == {}


def test_the_eval_contract_reads_the_point(tmp_path: Path) -> None:
    path = _point(tmp_path, {"benchmark": {"framework": "vllm", "model": "/m", "envs": {"RUN_EVAL": "0", "TP": 4}}})

    assert materialized_run_eval_disabled(path) is True
    assert _extract_eval_contract_fields(path) == {
        "framework": "vllm",
        "model": "/m",
        "benchmark_script": "",
        "precision": "",
        "RUN_EVAL": "0",
        "TP": "4",
    }


def test_a_replay_recipe_that_is_not_a_point_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkPointError):
        _replay_serving_env({"launch_recipe": str(_point(tmp_path, _NO_BLOCK))}, {})


def test_a_replay_takes_model_and_framework_from_the_recipe(tmp_path: Path) -> None:
    recipe = _point(
        tmp_path,
        {"benchmark": {"framework": "vllm", "model": "/m", "envs": {"TP": 2, "HIP_VISIBLE_DEVICES": "0,1"}}},
    )

    env = _replay_serving_env({"launch_recipe": str(recipe)}, {})

    assert (env["MODEL"], env["BACKEND"], env["TP"]) == ("/m", "vllm", "2")


def test_grid_readers_of_a_point(tmp_path: Path) -> None:
    path = _point(tmp_path, {"benchmark": {"framework": "xdit", "envs": {"TP": 4}}})

    assert _grid_runner._num_gpus_for_config(path) == 4.0
    assert _grid_runner._benchmark_server_log(path, tmp_path) is None
    assert _grid_runner._num_gpus_for_config(_point(tmp_path, _NO_BLOCK)) == 1.0


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
