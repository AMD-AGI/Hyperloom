"""The forge fmoe_ck tuner must not be E2E-validated on a shape CK cannot serve.

aiter's CK fused-MoE rejects an ``intermediate_size_per_partition`` that is not
128-aligned, so validating the candidate anyway costs a full server cold start
that can only end in a dead server. The predicate has its own unit coverage in
test_moe_runner_backend_injection.py; what is checked here is that the phase
actually consults it and skips before spending the restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState

_SKIP_REASON = "aiter_ck_moe_shape_unsupported"


def _qwen3_moe_model(root: Path) -> str:
    """Write a Qwen3-30B-A3B-shaped config: 768 shards to 96 at TP 8."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "num_experts": 128,
                "moe_intermediate_size": 768,
            }
        ),
        encoding="utf-8",
    )
    return str(root)


def _phase(tmp_path: Path, model_path: str, tp: int) -> KernelPhase:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        model_path=model_path,
        tp=tp,
        framework="sglang",
        baseline_tput=100.0,
    )
    return KernelPhase(coord)


def _fmoe_ck_result(csv_path: str) -> dict:
    """A forge run whose fmoe_ck tuner produced a usable candidate."""
    return {
        "tuners_run": [
            {
                "tuner": "fmoe_ck",
                "status": "ok",
                "candidate": True,
                "env_var": "AITER_CONFIG_FMOE_CK",
                "env_value": csv_path,
                "best_micro_speedup": 1.4,
            }
        ]
    }


@pytest.fixture
def integrate_spy(monkeypatch):
    """Record every E2E validation the phase attempts."""
    calls: list[dict] = []

    async def _fake(payload, *, session_dir):
        calls.append(payload)
        return {"status": "ok", "decision": "KEEP", "new_tput": 130.0, "gain_pct": 30.0}

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.integrate_handler",
        _fake,
    )
    return calls


@pytest.fixture
def tuned_csv(tmp_path: Path, monkeypatch) -> str:
    """A candidate artefact that exists, so only the shape gate can reject it."""
    csv = tmp_path / "fmoe_ck.csv"
    csv.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
    monkeypatch.setattr(
        KernelPhase,
        "_merge_gemm_candidate_with_runtime",
        lambda self, env_var, env_value: env_value,
    )
    return str(csv)


@pytest.mark.asyncio
async def test_unsupported_shape_is_skipped_before_the_restart(
    tmp_path: Path, integrate_spy, tuned_csv
):
    """TP 8 on a 768 intermediate size: skipped, and no server was started."""
    model = _qwen3_moe_model(tmp_path / "Qwen3-30B-A3B")
    result = _fmoe_ck_result(tuned_csv)

    await _phase(tmp_path, model, tp=8)._validate_gemm_tuning_e2e(result)

    reverted = result["e2e_results"]["reverted"]
    assert [r["reason"] for r in reverted] == [_SKIP_REASON]
    assert result["e2e_results"]["kept"] == []
    # The whole point of the gate: no cold start was paid for.
    assert integrate_spy == []


@pytest.mark.asyncio
async def test_supported_shape_still_reaches_validation(
    tmp_path: Path, integrate_spy, tuned_csv
):
    """TP 2 leaves 384, which is 128-aligned, so the candidate must run."""
    model = _qwen3_moe_model(tmp_path / "Qwen3-30B-A3B")
    result = _fmoe_ck_result(tuned_csv)

    await _phase(tmp_path, model, tp=2)._validate_gemm_tuning_e2e(result)

    assert [c["task_id"] for c in integrate_spy] == ["gemm_tune_e2e_fmoe_ck"]
    assert [k["tuner"] for k in result["e2e_results"]["kept"]] == ["fmoe_ck"]


@pytest.mark.asyncio
async def test_gate_only_applies_to_fmoe_ck(tmp_path: Path, integrate_spy, tuned_csv):
    """A different tuner on the same unservable shape is unaffected."""
    model = _qwen3_moe_model(tmp_path / "Qwen3-30B-A3B")
    result = _fmoe_ck_result(tuned_csv)
    result["tuners_run"][0]["tuner"] = "fmoe_asm"

    await _phase(tmp_path, model, tp=8)._validate_gemm_tuning_e2e(result)

    assert [c["task_id"] for c in integrate_spy] == ["gemm_tune_e2e_fmoe_asm"]
    assert result["e2e_results"]["reverted"] == []


@pytest.mark.asyncio
async def test_undecidable_model_is_not_skipped(tmp_path: Path, integrate_spy, tuned_csv):
    """No readable config: leave the call to sglang rather than skip on a guess."""
    result = _fmoe_ck_result(tuned_csv)

    await _phase(tmp_path, str(tmp_path / "absent"), tp=8)._validate_gemm_tuning_e2e(result)

    assert [c["task_id"] for c in integrate_spy] == ["gemm_tune_e2e_fmoe_ck"]
    assert [r["reason"] for r in result["e2e_results"]["reverted"]] != [_SKIP_REASON]
