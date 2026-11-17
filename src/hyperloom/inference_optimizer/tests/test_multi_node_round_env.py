# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A multi-node restart forwards the session base's server env, the operator's ``--extra-env`` pins over it, and the
round's own envs over both, on the baseline and the grid path alike."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from hyperloom.common.env_safety import OPERATOR_EXTRA_ENV_VAR
from hyperloom.orchestrator.actions.executors import _grid_runner
from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as lifecycle
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.state import base_launch
from hyperloom.orchestrator.state.base_launch import ServerBase
from inference_suite.recipe.defaults import default_table

_TABLE_ROW = default_table().row("vllm", "gfx950")


def _base(monkeypatch, env):
    server_base = ServerBase(args=" ".join(_TABLE_ROW.args), env=MappingProxyType(dict(env)))
    monkeypatch.setattr(base_launch, "session_server_base", lambda: server_base)


def _pin(monkeypatch, **pins):
    """What ``--extra-env NAME=VALUE`` leaves in the environment: the serialized pins, each also under its own name."""
    monkeypatch.setenv(OPERATOR_EXTRA_ENV_VAR, json.dumps(pins))
    for name, value in pins.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def forwarded(monkeypatch):
    """The ``extra_env`` each multi-node restart hands ``restart_server_for_round``."""
    monkeypatch.delenv(OPERATOR_EXTRA_ENV_VAR, raising=False)
    calls: list[dict[str, str] | None] = []

    async def _restart(server, *, extra_env=None, unset_env=None, **_kwargs):
        calls.append(extra_env)

    monkeypatch.setattr(lifecycle, "restart_server_for_round", _restart)
    return calls


async def _restart_baseline(tmp_path: Path) -> None:
    round_ = SimpleNamespace(ctx_extra={}, output_dir=tmp_path)
    executor = BaselineExecutor.__new__(BaselineExecutor)
    assert await executor._restart_multi_node_server(round_, server_args="", resolved_model="/models/m") is None


async def _restart_grid(tmp_path: Path, variant: GridVariant | None = None) -> None:
    run = SimpleNamespace(recipe=SimpleNamespace(target=SimpleNamespace(model_path="/models/m")))
    at = SimpleNamespace(variant=variant or GridVariant(name="args-only"))
    await _grid_runner._restart_multi_node_server(run, at, SimpleNamespace(mn_effective_args=""))


@pytest.mark.parametrize("restart", [_restart_baseline, _restart_grid], ids=["baseline", "grid"])
async def test_an_operator_pin_wins_over_the_default_tables_env_on_the_server(
    monkeypatch, tmp_path, forwarded, restart
):
    _base(monkeypatch, _TABLE_ROW.env)
    _pin(monkeypatch, VLLM_ROCM_USE_AITER="0")

    await restart(tmp_path)

    assert _TABLE_ROW.env["VLLM_ROCM_USE_AITER"] == "1"
    assert forwarded == [{"VLLM_ROCM_USE_AITER": "0"}]


@pytest.mark.parametrize("restart", [_restart_baseline, _restart_grid], ids=["baseline", "grid"])
async def test_the_base_env_reaches_the_server_with_no_pin(monkeypatch, tmp_path, forwarded, restart):
    _base(monkeypatch, {"SGLANG_REF_KNOB": "1"})

    await restart(tmp_path)

    assert forwarded == [{"SGLANG_REF_KNOB": "1"}]


async def test_a_variant_env_wins_over_the_pin_and_the_base(monkeypatch, tmp_path, forwarded):
    _base(monkeypatch, {"VLLM_ROCM_USE_AITER": "1", "OTHER": "keep"})
    _pin(monkeypatch, VLLM_ROCM_USE_AITER="0")
    variant = GridVariant(name="v", extra_envs={"VLLM_ROCM_USE_AITER": "candidate"})

    await _restart_grid(tmp_path, variant)

    assert forwarded == [{"VLLM_ROCM_USE_AITER": "candidate", "OTHER": "keep"}]


async def test_a_variant_unset_drops_the_base_and_the_pin(monkeypatch, tmp_path, forwarded):
    _base(monkeypatch, {"SGLANG_REF_KNOB": "reference"})
    _pin(monkeypatch, VLLM_ROCM_USE_AITER="0")
    variant = GridVariant(name="v", unset_envs=["SGLANG_REF_KNOB", "VLLM_ROCM_USE_AITER"])

    await _restart_grid(tmp_path, variant)

    assert forwarded == [{}]


async def test_no_base_env_and_no_pin_forwards_only_the_variants_envs(monkeypatch, tmp_path, forwarded):
    _base(monkeypatch, {})

    await _restart_grid(tmp_path, GridVariant(name="v", extra_envs={"MORI_X": "1"}))

    assert forwarded == [{"MORI_X": "1"}]


@pytest.mark.parametrize("restart", [_restart_baseline, _restart_grid], ids=["baseline", "grid"])
async def test_a_credential_pin_never_reaches_the_server(monkeypatch, tmp_path, forwarded, restart):
    _base(monkeypatch, {})
    _pin(monkeypatch, OPENAI_API_KEY="placeholder", NCCL_IB_DISABLE="1")

    await restart(tmp_path)

    assert forwarded == [{"NCCL_IB_DISABLE": "1"}]
