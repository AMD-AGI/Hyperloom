# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuned table whose rows no lookup the baseline logged can reach is not measured.

On GLM-5.3 an a8w8 and a bf16 table written at an M aiter never looks up each spent an AgentX round to read
``zero_hit`` afterwards; the baseline's own log already said so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState

# Prefill lookups of GLM-5.3's q_a/kv_a projection, as aiter logged them; it reaches 32768 for these through gl=1.
_A8W8_MISSES = "\n".join(
    f"[aiter] shape is M:{m}, N:2624, K:6144, not found tuned config in "
    "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!"
    for m in (16639, 26174, 32400)
)
_FMOE_KEY = (
    "'gfx950', 256, {token}, 6144, 2048, 65, 8, <ActivationType.Silu: 0>, 'torch.bfloat16', "
    "'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False"
)
_FMOE_DISPATCHES = "\n".join(
    "[aiter] [fused_moe] using 2stage default for (" + _FMOE_KEY.format(token=t) + ")" for t in (8, 32768)
)


def _phase(tmp_path: Path, monkeypatch, serving_log: str) -> KernelPhase:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(model_path=str(tmp_path / "absent"), tp=4, framework="sglang", baseline_tput=100.0)
    monkeypatch.setattr(KernelPhase, "_serving_log_text", lambda self: serving_log)
    monkeypatch.setattr(KernelPhase, "_merge_gemm_candidate_with_runtime", lambda self, env_var, env_value: env_value)
    return coord.phase_kernel


@pytest.fixture
def integrate_spy(monkeypatch):
    calls: list[dict] = []

    async def _fake(payload, *, session_dir):
        calls.append(payload)
        return {"status": "ok", "decision": "KEEP", "new_tput": 130.0, "gain_pct": 30.0}

    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers.integrate_handler", _fake)
    return calls


def _result(tuner: str, env_var: str, csv: Path) -> dict:
    return {"candidates": [{"tuner": tuner, "env": {env_var: str(csv)}, "best_micro_speedup": 1.5}]}


def _a8w8_table(tmp_path: Path, m: int) -> dict:
    csv = tmp_path / "candidate_a8w8_blockscale_bpreshuffle.csv"
    csv.write_text(
        "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        f"gfx950,256,{m},2624,6144,ck,7,0,540.2,a8w8_blockscale_bpreshuffle_x,1.0,1.0,0.0\n",
        encoding="utf-8",
    )
    return _result("a8w8_blockscale_bpreshuffle", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", csv)


def _fmoe_table(tmp_path: Path, token: int) -> dict:
    csv = tmp_path / "candidate_fmoe.csv"
    csv.write_text(
        "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,"
        "doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage\n"
        f"gfx950,256,{token},6144,2048,65,8,ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,"
        "torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,32,0,1.0,flydsl_moe1_x,0.0%,1.0,flydsl_moe2_x,0.0%,2.0,0\n",
        encoding="utf-8",
    )
    return _result("fmoe_ck", "AITER_CONFIG_FMOE", csv)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("serving_log", "table", "reachable_at", "unreachable_at"),
    [(_A8W8_MISSES, _a8w8_table, 32768, 8192), (_FMOE_DISPATCHES, _fmoe_table, 32768, 16639)],
    ids=["a8w8", "fused-moe"],
)
async def test_a_table_no_logged_lookup_reaches_is_not_measured(
    tmp_path, monkeypatch, integrate_spy, serving_log, table, reachable_at, unreachable_at
):
    result = table(tmp_path, unreachable_at)

    await _phase(tmp_path, monkeypatch, serving_log)._validate_gemm_tuning_e2e(result)

    assert [r["reason"] for r in result["e2e_results"]["reverted"]] == ["tuned_rows_unreachable"]
    assert integrate_spy == []

    result = table(tmp_path, reachable_at)

    await _phase(tmp_path, monkeypatch, serving_log)._validate_gemm_tuning_e2e(result)

    assert result["e2e_results"]["reverted"] == []
    assert len(integrate_spy) == 1


@pytest.mark.asyncio
async def test_a_log_without_lookups_for_the_table_cannot_tell_and_measures(tmp_path, monkeypatch, integrate_spy):
    result = _a8w8_table(tmp_path, 8192)

    await _phase(tmp_path, monkeypatch, "INFO server started\n")._validate_gemm_tuning_e2e(result)

    assert len(integrate_spy) == 1
