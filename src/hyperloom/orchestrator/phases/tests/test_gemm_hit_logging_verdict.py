# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuned table is judged unread only when the server that ran it logged hits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors.tests.test_suite_config import (
    _base,
    _launch,
    _point,
    _settings,
)
from hyperloom.orchestrator.actions.executors._suite.config import suite_config
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState

_TUNER = "sglang_dense_bf16"
# Prefill lookups aiter pads to 32768, which the table's M=8192 row cannot serve.
_MISSES = "\n".join(
    f"[aiter] shape is M:{m}, N:32, K:6144, not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, "
    "will use default config!"
    for m in (16639, 26174)
)


def _write_launch(run_dir: Path, server_env: dict[str, str]) -> None:
    """The ``launch.json`` inference_suite leaves beside the round's ``server.log``."""
    base = _base("sglang")
    point = _point("sglang")
    envs = point["benchmark"]["envs"]
    envs.pop("AITER_LOG_TUNED_CONFIG")
    envs.update(server_env)
    launch = _launch(suite_config(point, base, _settings()), base)
    (run_dir / "launch.json").write_text(json.dumps(launch.to_document()), encoding="utf-8")


def _phase(tmp_path: Path, server_env: dict[str, str] | None) -> tuple[KernelPhase, dict[str, str]]:
    table = tmp_path / "merged_tuned_dense_bf16.csv"
    table.write_text(
        "cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName\n"
        "256,8192,32,6144,False,torch.bfloat16,torch.bfloat16,False,False,asm,1,0,10.0,k\n"
    )
    env = {"AITER_CONFIG_GEMM_BF16": str(table)}
    bench = tmp_path / "runs" / "integrate" / f"integrate-gemm_tune_{_TUNER}" / "benchmark_sglang_1"
    bench.mkdir(parents=True)
    (bench / "server.log").write_text(_MISSES + "\n", encoding="utf-8")
    if server_env is not None:
        _write_launch(bench, {**env, **server_env})
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(model_path=str(tmp_path / "absent"), tp=4, framework="sglang", baseline_tput=100.0)
    return coord.phase_kernel, env


@pytest.mark.parametrize("server_env", [{}, {"AITER_LOG_TUNED_CONFIG": "0"}, None], ids=["unset", "off", "no-record"])
def test_an_unread_record_from_a_server_that_logged_no_hits_is_inconclusive(tmp_path, server_env):
    phase, env = _phase(tmp_path, server_env=server_env)

    applied = phase._gemm_apply_verdict(_TUNER, env)
    coverage = phase._gemm_tuned_config_coverage(_TUNER, env)

    assert (applied["verdict"], applied["blocks_keep"]) == ("inconclusive_no_hit_logging", False)
    assert (coverage["artifact_applied"], coverage["conclusive"]) == (False, False)


@pytest.mark.parametrize("flag", ["1", "true"])
def test_a_server_that_logged_hits_and_read_none_is_a_zero_hit(tmp_path, flag):
    phase, env = _phase(tmp_path, server_env={"AITER_LOG_TUNED_CONFIG": flag})

    applied = phase._gemm_apply_verdict(_TUNER, env)
    coverage = phase._gemm_tuned_config_coverage(_TUNER, env)

    assert (applied["verdict"], applied["blocks_keep"]) == ("zero_hit", True)
    assert (coverage["artifact_applied"], coverage["conclusive"]) == (False, True)
