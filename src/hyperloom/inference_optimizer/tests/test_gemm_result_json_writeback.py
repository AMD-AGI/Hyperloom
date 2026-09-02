"""The GEMM workspace ``result.json`` must carry the E2E verdict, not the pre-E2E snapshot.

forge's CLI writes ``result.json`` when micro tuning ends, so without a
writeback the file stays at ``status=ok`` / ``requires_e2e_validation=true``
forever while ``state.json`` already records a REVERT. The fusion and
collective lanes treat ``result.json`` as the final verdict, so the two ledgers
disagreeing is how a rejected candidate reads as "undecided" on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState


def _moe_model(root: Path) -> str:
    """A Qwen3-30B-A3B-shaped config: 768 shards to 96 at TP 8, which CK cannot serve."""
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


def _phase(tmp_path: Path, model_path: str, tp: int = 8) -> KernelPhase:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        model_path=model_path,
        tp=tp,
        framework="sglang",
        baseline_tput=100.0,
    )
    return KernelPhase(coord)


def _workspace_with_pre_e2e_result(tmp_path: Path) -> Path:
    """The snapshot forge leaves behind the moment micro tuning ends."""
    ws = tmp_path / "runs" / "gemm_tuning" / "abc123"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "micro_decision": "candidate",
                "requires_e2e_validation": True,
            }
        ),
        encoding="utf-8",
    )
    return ws


def _fmoe_ck_result(csv_path: str, workspace: Path) -> dict:
    return {
        "workspace": str(workspace),
        "tuners_run": [
            {
                "tuner": "fmoe_ck",
                "status": "ok",
                "candidate": True,
                "env_var": "AITER_CONFIG_FMOE_CK",
                "env_value": csv_path,
                "best_micro_speedup": 1.4,
            }
        ],
    }


@pytest.fixture
def tuned_csv(tmp_path: Path, monkeypatch) -> str:
    csv = tmp_path / "fmoe_ck.csv"
    csv.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
    monkeypatch.setattr(
        KernelPhase,
        "_merge_gemm_candidate_with_runtime",
        lambda self, env_var, env_value: env_value,
    )
    return str(csv)


@pytest.mark.asyncio
async def test_e2e_verdict_is_written_back_to_result_json(tmp_path: Path, tuned_csv):
    """The unservable-shape REVERT must land on disk, not only in state."""
    ws = _workspace_with_pre_e2e_result(tmp_path)
    model = _moe_model(tmp_path / "Qwen3-30B-A3B")
    result = _fmoe_ck_result(tuned_csv, ws)

    await _phase(tmp_path, model)._validate_gemm_tuning_e2e(result)

    on_disk = json.loads((ws / "result.json").read_text(encoding="utf-8"))
    assert on_disk["decision"] == "REVERT"
    assert on_disk["e2e_validated"] is True
    assert on_disk["requires_e2e_validation"] is False
    # The stale pre-E2E snapshot must be gone, not merely appended to.
    assert on_disk["micro_decision"] != "candidate"


@pytest.mark.asyncio
async def test_writeback_leaves_no_temp_files(tmp_path: Path, tuned_csv):
    ws = _workspace_with_pre_e2e_result(tmp_path)
    model = _moe_model(tmp_path / "Qwen3-30B-A3B")

    await _phase(tmp_path, model)._validate_gemm_tuning_e2e(_fmoe_ck_result(tuned_csv, ws))

    assert sorted(p.name for p in ws.iterdir()) == ["result.json"]


def test_writeback_failure_preserves_the_existing_file(tmp_path: Path, monkeypatch, caplog):
    """A read-only / reaped workspace must not crash the phase or truncate the file."""
    ws = _workspace_with_pre_e2e_result(tmp_path)
    original = (ws / "result.json").read_text(encoding="utf-8")
    model = _moe_model(tmp_path / "Qwen3-30B-A3B")

    def _boom(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr("hyperloom.orchestrator.phases.kernel.atomic_write_json", _boom)

    _phase(tmp_path, model)._writeback_gemm_result_json({"workspace": str(ws), "decision": "REVERT"})

    assert (ws / "result.json").read_text(encoding="utf-8") == original
    assert sorted(p.name for p in ws.iterdir()) == ["result.json"]


def test_writeback_is_a_noop_without_a_workspace(tmp_path: Path):
    """A result that never reached a workspace has nothing to reconcile."""
    model = _moe_model(tmp_path / "Qwen3-30B-A3B")
    # Must not raise.
    _phase(tmp_path, model)._writeback_gemm_result_json({"decision": "REVERT"})
    _phase(tmp_path, model)._writeback_gemm_result_json({"workspace": str(tmp_path / "gone"), "decision": "R"})


@pytest.mark.asyncio
async def test_validation_exception_syncs_state_and_disk(tmp_path: Path, monkeypatch):
    """The exception arm rewrote only its local dict; state kept the bridge's KEEP.

    ``record_gemm_tuning`` stores a shallow copy, so mutating ``result``
    afterwards does not update the recorded entry. An arm that was never
    measured must read as REVERT in both ledgers.
    """
    ws = _workspace_with_pre_e2e_result(tmp_path)
    model = _moe_model(tmp_path / "Qwen3-30B-A3B")
    phase = _phase(tmp_path, model)

    async def _raise(_result):
        raise RuntimeError("server never came up")

    monkeypatch.setattr(phase, "_validate_gemm_tuning_e2e", _raise)

    result = {"workspace": str(ws), "status": "ok", "decision": "KEEP", "recommended_env": {"A": "1"}}
    await phase._handle_gemm_tuning_result(result)

    assert phase.shared_state.last_gemm_tuning["decision"] == "REVERT"
    assert phase.shared_state.last_gemm_tuning["micro_decision"] == "e2e_validation_exception"
    on_disk = json.loads((ws / "result.json").read_text(encoding="utf-8"))
    assert on_disk["decision"] == "REVERT"
    assert on_disk["requires_e2e_validation"] is False
