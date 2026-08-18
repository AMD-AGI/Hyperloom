# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the roofline projection stored on a recipe.

The roofline records where a session finished against its ceiling. It rides the
``extras`` channel so it is written on every CLOSE, independent of the
``has_validated_win`` gate that guards ``best_config`` — a session that improved
nothing still measured a distance to the roofline, and that measurement is the
part worth keeping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.orchestrator.kernel.roofline_snapshot import (
    MAX_RECIPE_PERFMODEL_OPS,
    build_recipe_roofline,
)
from hyperloom.orchestrator.knowledge.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    recipe_canonical_id,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)

_MODEL = "qwen3-30b-a3b"
_HW = "mi300x"
_FW = "sglang"
_FWV = "0.4.5"
_PREC = "fp8"


def _snapshot(**overrides: Any) -> dict[str, Any]:
    """Build a roofline snapshot shaped like ``build_roofline_snapshot`` output."""
    snap: dict[str, Any] = {
        "snapshot_id": 1,
        "ts": "2026-08-17T04:05:19.522322+00:00",
        "throughput_unit": "tok/s",
        "theoretical_peak_tok_per_sec": 11573.63,
        "roofline_mem_ceiling_tok_per_sec": 11573.63,
        "roofline_cmp_ceiling_tok_per_sec": 172038.66,
        "roofline_bound_kind": "memory",
        "achieved_tok_per_sec": 240.5,
        "within_roofline_pct": 2.08,
        "gap_to_roofline_pct": 97.92,
        "compute_pct": 17.54,
        "idle_pct": 0.14,
        "comm_pct": 81.88,
        "top_bottleneck": "InferenceAttention",
        "top_kernel": {
            "name": "vllm::unified_attention_with_output",
            "gpu_pct": 2.13,
            "efficiency_pct": 1.11,
            "bound_type": "memory",
        },
        "roofline_provenance": {
            "formula": "perfmodel",
            "compute_peak_tflops": 1686.0,
            "compute_peak_convention": "achievable",
            "runtime_tp": 8,
        },
        "perfmodel_breakdown": {
            "bound_kind": "memory",
            "hbm_bw_gbps": 28480.0,
            "peak_achievable_tflops": 6744.0,
            "ops": [
                {
                    "name": "q_proj",
                    "flops": 9.66e10,
                    "bytes_moved": 6.07e9,
                    "ai": 15.9,
                    "time_s": 0.0002,
                    "bound": "memory",
                    "pct_time": 0.2,
                },
                {
                    "name": "k_proj",
                    "flops": 6.04e9,
                    "bytes_moved": 3.9e8,
                    "ai": 15.5,
                    "time_s": 0.00001,
                    "bound": "memory",
                    "pct_time": 0.0,
                },
            ],
        },
    }
    snap.update(overrides)
    return snap


def _make_coordinator(tmp_path: Path) -> Coordinator:
    """Build a Coordinator backed by a real on-disk local recipe store."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle),
        "critic": MockBackend(idle),
        "robustness": MockBackend(idle),
    }
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=kb,
        knowledge_plane=None,
    )
    ss = coord.shared_state
    ss.model_name = _MODEL
    ss.gpu_type = _HW
    ss.framework = _FW
    ss.framework_version = _FWV
    ss.precision = _PREC
    return coord


def _expected_cid() -> str:
    return recipe_canonical_id(
        model=_MODEL,
        hardware=_HW,
        framework_name=_FW,
        framework_version=_FWV,
        precision=_PREC,
    )


def test_no_snapshots_project_to_nothing() -> None:
    """A session that never ran a roofline yields no payload at all."""
    assert build_recipe_roofline(None) == {}
    assert build_recipe_roofline([]) == {}
    assert build_recipe_roofline(["not-a-dict"]) == {}  # type: ignore[list-item]


def test_projection_carries_both_ceilings_and_the_per_op_breakdown() -> None:
    """Ceilings, the trace mix, provenance, and per-op rows all survive."""
    out = build_recipe_roofline([_snapshot()])
    assert out["roofline_mem_ceiling_tok_per_sec"] == 11573.63
    assert out["roofline_cmp_ceiling_tok_per_sec"] == 172038.66
    assert out["roofline_bound_kind"] == "memory"
    assert out["within_roofline_pct"] == 2.08
    assert out["comm_pct"] == 81.88
    assert out["top_kernel"]["name"] == "vllm::unified_attention_with_output"
    assert out["roofline_provenance"]["compute_peak_tflops"] == 1686.0
    pm = out["perfmodel_breakdown"]
    assert pm["hbm_bw_gbps"] == 28480.0
    assert pm["peak_achievable_tflops"] == 6744.0
    assert [op["name"] for op in pm["ops"]] == ["q_proj", "k_proj"]
    assert pm["ops"][0]["ai"] == 15.9
    assert pm["ops"][0]["bytes_moved"] == 6.07e9


def test_projection_records_the_latest_snapshot() -> None:
    """``snapshots[-1]`` wins, matching ``SharedState.current_top_bottleneck``."""
    first = _snapshot(snapshot_id=1, within_roofline_pct=2.08)
    second = _snapshot(snapshot_id=2, within_roofline_pct=41.5)
    out = build_recipe_roofline([first, second])
    assert out["snapshot_id"] == 2
    assert out["within_roofline_pct"] == 41.5
    assert out["snapshot_count"] == 2


def test_a_snapshot_without_a_perfmodel_still_projects() -> None:
    """The PerfModel is best-effort; its absence must not drop the ceilings."""
    snap = _snapshot()
    snap.pop("perfmodel_breakdown")
    out = build_recipe_roofline([snap])
    assert "perfmodel_breakdown" not in out
    assert out["roofline_mem_ceiling_tok_per_sec"] == 11573.63


def test_the_per_op_array_is_capped() -> None:
    """A pathological op count is truncated, and the original size recorded."""
    ops = [
        {"name": f"op{i}", "flops": float(i), "bytes_moved": 1.0, "ai": 1.0, "bound": "memory"}
        for i in range(MAX_RECIPE_PERFMODEL_OPS + 10)
    ]
    snap = _snapshot(perfmodel_breakdown={"bound_kind": "memory", "ops": ops})
    pm = build_recipe_roofline([snap])["perfmodel_breakdown"]
    assert len(pm["ops"]) == MAX_RECIPE_PERFMODEL_OPS
    assert pm["ops_truncated_from"] == MAX_RECIPE_PERFMODEL_OPS + 10


def test_close_stores_the_roofline_without_a_validated_win(tmp_path: Path) -> None:
    """The roofline lands even when nothing beat the incumbent config.

    This is the whole point of routing it outside the champion gate: every
    recipe on this host has sessions but no ``best_config``, so a
    win-gated roofline would never be written.
    """
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.roofline_snapshots = [_snapshot()]
    # No optimization stack, no validated gain, no current_best: has_validated_win is False.
    ss.optimization_stack = []
    ss.cumulative_gain_validated = 0.0
    ss.current_best = {}
    coord.finalize_recipe_and_journal()
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "finalize wrote no recipe at all"
    assert not row.get("best_config"), "precondition broken: a champion was written"
    roofline = row.get("roofline")
    assert roofline, "roofline was gated behind the champion write"
    assert roofline["roofline_bound_kind"] == "memory"
    assert roofline["perfmodel_breakdown"]["ops"][0]["name"] == "q_proj"


def test_a_session_without_a_roofline_keeps_the_previous_one(tmp_path: Path) -> None:
    """Omitting the key preserves a prior roofline instead of erasing it."""
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        canonical_id=cid,
        model=_MODEL,
        hardware=_HW,
        framework_name=_FW,
        framework_version=_FWV,
        precision=_PREC,
        extras={"roofline": {"roofline_bound_kind": "compute", "within_roofline_pct": 88.0}},
    )
    coord.shared_state.roofline_snapshots = []
    coord.finalize_recipe_and_journal()
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["roofline"]["roofline_bound_kind"] == "compute"
    assert row["roofline"]["within_roofline_pct"] == 88.0
