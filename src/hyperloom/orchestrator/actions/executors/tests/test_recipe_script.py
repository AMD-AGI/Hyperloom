# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which launcher boots the server, and the env names it pins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml
from hyperloom.orchestrator.actions.executors._recipe_script import (
    launcher_overwritten_envs,
    resolve_launch_server_script,
)
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("", encoding="utf-8")
    (benchmarks / "vllm_mi355x.sh").write_text(
        'export HSA_NO_SCRATCH_RECLAIM=1\nexport PORT="${PORT:-8000}"\nvllm serve $MODEL $EXTRA_VLLM_ARGS\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(benchmarks.parent))
    monkeypatch.delenv("AGENTX_SERVER_SCRIPT", raising=False)
    return benchmarks


def _config(tmp_path: Path, **bench) -> Path:
    base = tmp_path / "base.yaml"
    body = {"framework": "vllm", "model": str(tmp_path / "model"), "envs": {}, **bench}
    base.write_text(yaml.safe_dump({"benchmark": body}), encoding="utf-8")
    return base


def test_the_agentx_client_resolves_to_the_builtin_beside_it_not_a_same_named_one_below(checkout):
    (checkout / "single_node").mkdir()
    (checkout / "single_node" / "vllm_mi355x.sh").write_text("# a different script\n", encoding="utf-8")
    bench = {"benchmark_script": "aiperf_client.sh", "framework": "vllm", "runner_type": "mi355x"}

    assert resolve_launch_server_script(bench) == str(checkout / "vllm_mi355x.sh")


def test_only_unguarded_exports_of_the_launcher_count_as_overwritten(checkout):
    # ``export PORT="${PORT:-8000}"`` defers to a caller-supplied value.
    assert launcher_overwritten_envs({"benchmark_script": "vllm_mi355x.sh"}) == {"HSA_NO_SCRATCH_RECLAIM"}


def test_a_variant_env_the_launcher_overwrites_is_dropped(checkout, tmp_path):
    base = _config(tmp_path, benchmark_script="vllm_mi355x.sh")
    variant = GridVariant(name="v", extra_envs={"HSA_NO_SCRATCH_RECLAIM": "0", "VLLM_SOMETHING_ELSE": "1"})

    out = _build_variant_yaml(base, "", variant, output_subdir=tmp_path / "slot")

    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]
    assert "HSA_NO_SCRATCH_RECLAIM" not in envs
    assert envs["VLLM_SOMETHING_ELSE"] == "1"


@pytest.mark.parametrize("agentx", [False, True], ids=["synthetic", "agentx"])
def test_the_launcher_gets_launcher_defaults(checkout, tmp_path, agentx):
    model = tmp_path / "model"
    model.mkdir()
    config = {"text_config": {"sparse_attention_config": {"sparse_block_size": 128}}}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")

    out = materialize_config_with_envs(
        _config(tmp_path), tmp_path / "out", model_path=str(model), gpu_type="mi355x", agentx_mode=agentx
    )

    assert "--block-size 128" in yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
