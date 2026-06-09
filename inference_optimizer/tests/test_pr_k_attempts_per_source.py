# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-K — per-source attempts ledger unlocks device retry post-promotion.

The new ``attempts_per_source`` ledger lets ``_is_live`` allow a fresh attempt
against a promoted device ``.cu`` even when the cumulative ``attempts`` counter
(all against the wrapper) exceeds max_attempts; legacy entries fall back to cumulative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.shared_state import SharedState


# fixtures
@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = tmp_path
    (sd / "manifest.json").write_text("{}", encoding="utf-8")
    return sd


@pytest.fixture
def candidates_factory(tmp_path: Path):
    """Write a kernel_candidates.json fixture and return its path."""

    def _make(hot_kernels: list[dict], task_groups: list[dict] | None = None) -> str:
        path = tmp_path / "kernel_candidates.json"
        path.write_text(
            json.dumps({
                "hot_kernels": hot_kernels,
                "task_groups": task_groups or [],
                "reusable_native_kernel_ids": [],
            }),
            encoding="utf-8",
        )
        return str(path)

    return _make


# record_kernel_opt — writes the per-source ledger.
def test_record_kernel_opt_writes_attempts_per_source(
    session_dir: Path,
) -> None:
    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "ok",
        "kernel_id": "k001",
        "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.02},
    })
    entry = state.kernel_opt_attempts["k001"]
    assert entry["attempts"] == 1
    assert entry["attempts_per_source"] == {
        "/sgl-workspace/aiter/aiter/ops/moe_op.py": 1,
    }


def test_record_kernel_opt_increments_per_source_independently(
    session_dir: Path,
) -> None:
    """Two distinct source paths produce separate per-source counters; cumulative ``attempts`` sums them."""
    state = SharedState.load_or_init(session_dir)
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    device = "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"

    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 0.99},
    })
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": device,
        "proposal": {"decision": "REVERT"},
        "verification": {"micro_speedup": 0.95},
    })
    entry = state.kernel_opt_attempts["k001"]
    assert entry["attempts"] == 2
    assert entry["attempts_per_source"] == {wrapper: 1, device: 1}


def test_record_kernel_opt_normalizes_empty_source_file(
    session_dir: Path,
) -> None:
    """A missing source_file uses the empty key ``""`` so the ledger stays a valid dict."""
    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "failed", "kernel_id": "k042",
        # source_file omitted on purpose.
        "proposal": {"decision": "REVERT"},
    })
    entry = state.kernel_opt_attempts["k042"]
    assert entry["attempts_per_source"] == {"": 1}


# End-to-end: promotion unlocks a fresh attempt against the device source.
def test_batch_candidates_unlocks_promoted_device_source_after_wrapper_attempt(
    session_dir: Path, candidates_factory,
) -> None:
    """PR-K core: a candidate whose source was promoted wrapper→device .cu IS dispatched despite a recorded wrapper attempt."""
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    device = "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"

    state = SharedState.load_or_init(session_dir)
    # Simulate a wrapper attempt that PARTIAL'd.
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.01},
    })
    state.save(session_dir)

    # Round 2 candidates: source_file now points at device .cu.
    cpath = candidates_factory([
        {
            "kernel_id": "k001", "gpu_pct": 25.0,
            "reusable_native_kernel": True,
            "source_file": device,
            "name": "aiter::ck_moe_stage1",
        },
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert [c["kernel_id"] for c in out] == ["k001"], (
        "promoted device source must unlock a fresh dispatch"
    )


def test_batch_candidates_skips_kernel_when_same_source_already_attempted(
    session_dir: Path, candidates_factory,
) -> None:
    """When round 2's source_file matches the round-1 attempted source, the kernel is skipped (no retry loop)."""
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"

    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k001",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.0},
    })
    state.save(session_dir)

    cpath = candidates_factory([
        {
            "kernel_id": "k001", "gpu_pct": 25.0,
            "reusable_native_kernel": True,
            "source_file": wrapper,  # same as the recorded attempt
        },
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert out == []


def test_batch_candidates_falls_back_to_cumulative_for_legacy_entry(
    session_dir: Path, candidates_factory,
) -> None:
    """A pre-PR-K entry (no ``attempts_per_source``) falls back to the cumulative counter on resume."""
    state = SharedState.load_or_init(session_dir)
    # Synthesize a v1 entry directly, bypassing record_kernel_opt.
    state.kernel_opt_attempts = {
        "k001": {
            "attempts": 1, "partial_count": 1,
            "last_decision": "PARTIAL",
            "last_source_file": "/p/moe_op.py",
        },
    }
    state.save(session_dir)

    cpath = candidates_factory([
        {
            "kernel_id": "k001", "gpu_pct": 25.0,
            "reusable_native_kernel": True,
            "source_file": "/p/different.cu",  # different source on round 2
        },
    ])
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    # Legacy entry: cumulative attempts >= 1 → skipped (conservative resume contract).
    assert out == []


def test_batch_candidates_task_group_promotion_unlocks_primary(
    session_dir: Path, candidates_factory,
) -> None:
    """Task_group flow: after promoting both members to device source, the group dispatches primary k002 again, not group_exhausted."""
    wrapper = "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    device = "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu"

    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k002",
        "source_file": wrapper,
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.0},
    })
    state.save(session_dir)

    cpath = candidates_factory(
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
             "source_file": device, "name": "aiter::ck_moe_stage1"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
             "source_file": device, "name": "aiter::ck_moe_stage2"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    out = krh._batch_kernel_candidates(
        {"candidates_path": cpath}, session_dir=session_dir,
    )
    assert len(out) == 1
    assert out[0]["kernel_id"] == "k002"
    assert out[0]["source_file"] == device
