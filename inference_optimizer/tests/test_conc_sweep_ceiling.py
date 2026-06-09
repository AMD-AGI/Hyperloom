# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests pinning ``orchestrator.conc_sweep._build_roofline_ceiling`` (MoE/dense ceiling + MBU + safe-degrade)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# Imported first to break a circular dependency between conc_sweep and
# action_executors.conc_sweep (else the module fails to collect in isolation).
from inference_optimizer.orchestrator.action_executors._grid_runner import (  # noqa: F401
    VariantResult,
)
from inference_optimizer.orchestrator.conc_sweep import _build_roofline_ceiling
from inference_optimizer.orchestrator.roofline_ceiling import ModelMeta
from inference_optimizer.orchestrator.shared_state import SharedState

_LOAD_META_PATH = "inference_optimizer.orchestrator.conc_sweep.load_model_meta"


def _make_state(
    *,
    model_path: str = "/fake/model",
    gpu_type: str = "mi355x",
    tp: int = 2,
    precision: str = "bf16",
    isl: int = 1024,
    osl: int = 1024,
) -> SharedState:
    s = SharedState()
    s.model_path = model_path
    s.gpu_type = gpu_type
    s.tp = tp
    s.precision = precision
    s.isl = isl
    s.osl = osl
    return s


def _qwen3_30b_a3b_meta() -> ModelMeta:
    """ModelMeta shaped like Qwen3-30B-A3B (real config values)."""
    # Geometry pinned to the Qwen3-MoE config.json to exercise real MoE branching.
    num_layers = 48
    num_kv_heads = 4
    head_dim = 128
    hidden_size = 2048
    moe_inter = 768
    num_experts = 128
    experts_per_tok = 8
    dtype_bytes = 2.0
    expert_bytes_per_layer = int(
        num_experts * 3 * hidden_size * moe_inter * dtype_bytes
    )
    expert_weight_bytes = num_layers * expert_bytes_per_layer
    # ~60GB; large enough that subtracting expert_weight_bytes stays positive.
    weight_bytes = 60 * 1024**3
    non_expert_bytes = weight_bytes - expert_weight_bytes
    active_weight_bytes = (
        non_expert_bytes
        + int((experts_per_tok / num_experts) * expert_weight_bytes)
    )
    return ModelMeta(
        weight_bytes=weight_bytes,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        weight_dtype_bytes=dtype_bytes,
        active_weight_bytes=active_weight_bytes,
        num_experts=num_experts,
        experts_per_tok=experts_per_tok,
        expert_weight_bytes=expert_weight_bytes,
    )


@pytest.fixture
def stub_meta() -> Any:
    """Patch ``load_model_meta`` so tests do not need real HF assets."""
    meta = _qwen3_30b_a3b_meta()
    with patch(_LOAD_META_PATH, return_value=meta):
        yield meta


# Happy path
def test_happy_path_moe_qwen3_30b_a3b_mi355x(stub_meta: ModelMeta) -> None:
    state = _make_state()
    concs = [1, 2, 4, 8, 16, 32, 64, 128]
    baseline_points = [
        {"conc": c, "output_throughput": float(c) * 100.0} for c in concs
    ]
    optimized_points = [
        {"conc": c, "output_throughput": float(c) * 120.0} for c in concs
    ]

    block = _build_roofline_ceiling(
        state,
        concs=concs,
        isl=1024,
        osl=1024,
        baseline_points=baseline_points,
        optimized_points=optimized_points,
    )
    assert block is not None
    assert block["schema_version"] == 1
    assert block["source"] == "roofline_ceiling.py"
    assert block["gpu_type"] == "mi355x"
    assert block["tp"] == 2
    assert block["precision"] == "bf16"
    assert block["isl"] == 1024 and block["osl"] == 1024
    assert block["model_meta"]["num_experts"] == stub_meta.num_experts
    assert block["model_meta"]["experts_per_tok"] == stub_meta.experts_per_tok

    rows = block["rows"]
    assert [r["conc"] for r in rows] == concs

    t_mems = [r["t_mem_tok_s"] for r in rows]
    t_cmps = [r["t_cmp_tok_s"] for r in rows]
    t_peaks = [r["t_peak_tok_s"] for r in rows]
    bound_kinds = [r["bound_kind"] for r in rows]

    # T_mem strictly increasing in batch.
    assert t_mems == sorted(t_mems)
    assert all(a < b for a, b in zip(t_mems, t_mems[1:]))
    # T_cmp B-independent → flat column.
    assert len(set(t_cmps)) == 1 and t_cmps[0] > 0
    # Decode roofline puts memory-bound at every B; T_peak follows T_mem.
    assert bound_kinds == ["memory"] * len(concs)
    assert t_peaks == t_mems
    # T_cmp must dominate (decode-on-MoE is far left of the ridge).
    assert t_cmps[0] > t_mems[-1]


def test_mbu_matches_measured_over_t_peak(stub_meta: ModelMeta) -> None:
    state = _make_state()
    concs = [4, 8]
    baseline_points = [
        {"conc": 4, "output_throughput": 800.0},
        {"conc": 8, "output_throughput": 1500.0},
    ]
    optimized_points = [
        {"conc": 4, "output_throughput": 950.0},
        {"conc": 8, "output_throughput": 1800.0},
    ]
    block = _build_roofline_ceiling(
        state,
        concs=concs,
        isl=1024,
        osl=1024,
        baseline_points=baseline_points,
        optimized_points=optimized_points,
    )
    assert block is not None
    for row, base, opt in zip(block["rows"], [800.0, 1500.0], [950.0, 1800.0]):
        peak = row["t_peak_tok_s"]
        assert row["mbu_baseline_pct"] == pytest.approx(
            round(base / peak * 100.0, 2), rel=1e-6,
        )
        assert row["mbu_optimized_pct"] == pytest.approx(
            round(opt / peak * 100.0, 2), rel=1e-6,
        )


def test_mbu_none_on_failed_or_missing_measurement(
    stub_meta: ModelMeta,
) -> None:
    state = _make_state()
    concs = [1, 2]
    # conc=1 has only baseline; conc=2 has zero throughput on baseline.
    baseline_points = [
        {"conc": 1, "output_throughput": 200.0},
        {"conc": 2, "output_throughput": 0.0},
    ]
    optimized_points = [
        {"conc": 2, "output_throughput": 400.0},
    ]
    block = _build_roofline_ceiling(
        state,
        concs=concs,
        isl=1024,
        osl=1024,
        baseline_points=baseline_points,
        optimized_points=optimized_points,
    )
    assert block is not None
    rows = {r["conc"]: r for r in block["rows"]}
    assert rows[1]["mbu_baseline_pct"] is not None
    assert rows[1]["mbu_optimized_pct"] is None  # missing optimized
    assert rows[2]["mbu_baseline_pct"] is None   # zero throughput
    assert rows[2]["mbu_optimized_pct"] is not None


def test_dense_fallback_no_moe_fields() -> None:
    """num_experts=0 → dense path; ceiling still computed."""
    dense_meta = ModelMeta(
        weight_bytes=16 * 1024**3,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
        weight_dtype_bytes=2.0,
        active_weight_bytes=0,
        num_experts=0,
        experts_per_tok=0,
        expert_weight_bytes=0,
    )
    state = _make_state()
    with patch(_LOAD_META_PATH, return_value=dense_meta):
        block = _build_roofline_ceiling(
            state,
            concs=[1, 8, 128],
            isl=1024,
            osl=1024,
            baseline_points=[],
            optimized_points=[],
        )
    assert block is not None
    rows = block["rows"]
    assert len(rows) == 3
    assert rows[0]["t_mem_tok_s"] < rows[-1]["t_mem_tok_s"]
    assert all(r["bound_kind"] == "memory" for r in rows)
    # No measurements → all MBU None.
    assert all(
        r["mbu_baseline_pct"] is None and r["mbu_optimized_pct"] is None
        for r in rows
    )


# Safe degrade
def test_returns_none_when_meta_unavailable() -> None:
    state = _make_state()
    with patch(_LOAD_META_PATH, return_value=None):
        block = _build_roofline_ceiling(
            state,
            concs=[1, 8],
            isl=1024,
            osl=1024,
            baseline_points=[],
            optimized_points=[],
        )
    assert block is None


def test_returns_none_when_gpu_type_missing(stub_meta: ModelMeta) -> None:
    state = _make_state(gpu_type="")
    block = _build_roofline_ceiling(
        state,
        concs=[1],
        isl=1024,
        osl=1024,
        baseline_points=[],
        optimized_points=[],
    )
    assert block is None


def test_returns_none_when_tp_zero(stub_meta: ModelMeta) -> None:
    state = _make_state(tp=0)
    block = _build_roofline_ceiling(
        state,
        concs=[1],
        isl=1024,
        osl=1024,
        baseline_points=[],
        optimized_points=[],
    )
    assert block is None
