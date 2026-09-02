# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR-K — per-source attempts ledger unlocks device retry post-promotion.

The ``attempts_per_source`` ledger lets ``_is_live`` allow a fresh attempt
against a promoted device ``.cu`` even when the cumulative ``attempts`` counter
exceeds max_attempts; legacy entries fall back to cumulative.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.state.shared_state import SharedState


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
            json.dumps(
                {
                    "hot_kernels": hot_kernels,
                    "task_groups": task_groups or [],
                    "reusable_native_kernel_ids": [],
                }
            ),
            encoding="utf-8",
        )
        return str(path)

    return _make


def test_record_kernel_opt_writes_attempts_per_source(
    session_dir: Path,
) -> None:
    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k001",
            "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
            "proposal": {"decision": "PARTIAL"},
            "verification": {"micro_speedup": 1.02},
        }
    )
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

    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k001",
            "source_file": wrapper,
            "proposal": {"decision": "PARTIAL"},
            "verification": {"micro_speedup": 0.99},
        }
    )
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k001",
            "source_file": device,
            "proposal": {"decision": "REVERT"},
            "verification": {"micro_speedup": 0.95},
        }
    )
    entry = state.kernel_opt_attempts["k001"]
    assert entry["attempts"] == 2
    assert entry["attempts_per_source"] == {wrapper: 1, device: 1}


def test_record_kernel_opt_normalizes_empty_source_file(
    session_dir: Path,
) -> None:
    """A missing source_file uses the empty key ``""`` so the ledger stays a valid dict."""
    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt(
        {
            "status": "failed",
            "kernel_id": "k042",
            "proposal": {"decision": "REVERT"},
        }
    )
    entry = state.kernel_opt_attempts["k042"]
    assert entry["attempts_per_source"] == {"": 1}
