# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``run``'s paths: a relative path on the command line or in the config names a path below the caller's
directory, whatever directory the round's children start in."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from inference_suite import cli
from inference_suite.contracts import BaseRef, Lifecycle, LifecycleMode, ProfileConfig, RunLayout, WorkloadKind
from inference_suite.load.mlperf import harness_env
from inference_suite.tests.test_config_contract import _FIXED, _MLPERF, _config, _mlperf_config


def _relative(config):
    server = replace(
        config.server,
        base=replace(config.server.base, path="session/base_launch.json"),
        model_path="models/GLM-5.3",
        framework_python="venv/bin/python",
    )
    paths = replace(config.paths, tool_cache="tools")
    return replace(config, server=server, paths=paths, lifecycle=Lifecycle(LifecycleMode.BOOT, "pids"))


def _read(tmp_path: Path, config) -> object:
    (tmp_path / "models" / "GLM-5.3").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config.to_document()))
    return cli.read_config(Path("config.yaml"), Path("runs/r1"))


def _assert_anchored(read, tmp_path: Path) -> None:
    assert read.paths.run_dir == str(tmp_path / "runs" / "r1")
    assert read.paths.tool_cache == str(tmp_path / "tools")
    assert read.server.base == BaseRef(str(tmp_path / "session" / "base_launch.json"), read.server.base.digest)
    assert read.server.model_path == str(tmp_path / "models" / "GLM-5.3")
    assert read.server.framework_python == str(tmp_path / "venv" / "bin" / "python")
    assert read.lifecycle.pid_dir == str(tmp_path / "pids")


def test_a_fixed_round_reads_its_relative_paths_below_the_callers_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixed = _config(workload=WorkloadKind.FIXED, agentx=None, fixed=_FIXED, profile=ProfileConfig(False))

    _assert_anchored(_read(tmp_path, _relative(fixed)), tmp_path)


def test_an_mlperf_harness_writes_into_the_run_dir_below_the_callers_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlperf = replace(_MLPERF, dataset="data/corpus.jsonl", tokenizer="data/tokenizer", endpoints_dir="endpoints")

    read = _read(tmp_path, _relative(_mlperf_config(mlperf=mlperf)))

    _assert_anchored(read, tmp_path)
    env = harness_env(read, RunLayout(Path(read.paths.run_dir)))
    assert env["RESULTS_DIR"] == str(RunLayout(tmp_path / "runs" / "r1").mlperf_artifacts)
    assert (env["AGENTIC_DATASET_PATH"], env["MLPERF_TOKENIZER_DIR"]) == (
        str(tmp_path / "data" / "corpus.jsonl"),
        str(tmp_path / "data" / "tokenizer"),
    )
    assert read.mlperf is not None and read.mlperf.endpoints_dir == str(tmp_path / "endpoints")


@pytest.mark.parametrize("model_path", ["zai-org/GLM-5.3", "/models/GLM-5.3"])
def test_a_hub_id_or_an_absolute_model_path_is_kept(tmp_path, monkeypatch, model_path):
    monkeypatch.chdir(tmp_path)
    config = _config(server=replace(_config().server, model_path=model_path, framework_python="python3"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config.to_document()))

    read = cli.read_config(Path("config.yaml"), Path("r1"))

    assert (read.server.model_path, read.server.framework_python) == (model_path, "python3")
