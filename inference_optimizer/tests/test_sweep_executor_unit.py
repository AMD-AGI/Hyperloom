# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the sweep ActionRunner helpers and __call__ flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.action_executors import sweep as sw
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    VariantResult,
)


# ---- _coerce_int ----

@pytest.mark.parametrize("value,expected", [
    (None, 0), ("", 0), ("42", 42), (7, 7), ("bad", 0), ("  9 ", 9),
])
def test_coerce_int(value, expected):
    assert sw._coerce_int(value) == expected


# ---- _build_grid ----

def test_build_grid_basic():
    grid, skipped = sw._build_grid(
        conc_values=[4], isl_osl_configs=["1024:1024"],
        num_prompts_factor=5, base_extra_args="--x",
    )
    assert len(grid) == 1
    assert skipped == []
    v = grid[0]
    assert v.extra_envs["CONC"] == "4"
    assert v.extra_envs["NUM_PROMPTS"] == "20"


def test_build_grid_malformed_io_skipped():
    grid, skipped = sw._build_grid(
        conc_values=[4], isl_osl_configs=["nonsense"],
        num_prompts_factor=5, base_extra_args="",
    )
    assert grid == []
    assert skipped == []


def test_build_grid_max_model_len_skip():
    grid, skipped = sw._build_grid(
        conc_values=[4], isl_osl_configs=["8192:1024"],
        num_prompts_factor=5, base_extra_args="", max_model_len=4096,
    )
    assert grid == []
    assert skipped[0]["status"] == "skipped"
    assert "exceeds max_model_len" in skipped[0]["skip_reason"]


# ---- _result_dict ----

def test_result_dict_surfaces_dims():
    vr = VariantResult(
        name="v", extra_server_args="", extra_envs={"CONC": "16", "ISL": "1024", "OSL": "512"},
        status="succeeded", output_throughput=100.0, e2el_mean_ms=50.0,
    )
    d = sw._result_dict(vr)
    assert d["conc"] == 16 and d["isl"] == 1024 and d["osl"] == 512


# ---- _pareto_front ----

def test_pareto_front_dominance():
    entries = [
        {"status": "succeeded", "output_throughput": 100, "e2el_mean_ms": 10},
        {"status": "succeeded", "output_throughput": 90, "e2el_mean_ms": 20},  # dominated
        {"status": "succeeded", "output_throughput": 80, "e2el_mean_ms": 5},   # not dominated
        {"status": "failed", "output_throughput": 999, "e2el_mean_ms": 1},     # excluded
    ]
    front = sw._pareto_front(entries)
    tputs = sorted(e["output_throughput"] for e in front)
    assert tputs == [80, 100]


def test_pareto_front_ignores_non_numeric():
    entries = [{"status": "succeeded", "output_throughput": None, "e2el_mean_ms": 10}]
    assert sw._pareto_front(entries) == []


# ---- SweepExecutor.__call__ ----

def _ctx(tmp_path, params):
    return SimpleNamespace(
        task=SimpleNamespace(params=params, task_id="t1"),
        extra={"workspace": str(tmp_path)},
    )


async def test_call_missing_config(tmp_path):
    ex = sw.SweepExecutor(session_dir=tmp_path)
    ctx = _ctx(tmp_path, {"config_path": str(tmp_path / "nope.yaml")})
    out = await ex(ctx)
    assert out["status"] == "failed"
    assert out["error_class"] == "missing_config"


async def test_call_bad_param(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("k: v\n", encoding="utf-8")
    ex = sw.SweepExecutor(session_dir=tmp_path)
    ctx = _ctx(tmp_path, {"config_path": str(cfg), "benchmark_script": "bad name; rm -rf /"})
    out = await ex(ctx)
    assert out["status"] == "failed"
    assert out["error_class"] == "bad_param"


async def test_call_success(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("k: v\n", encoding="utf-8")

    monkeypatch.setattr(sw, "materialize_config_with_envs", lambda *a, **k: cfg)

    async def fake_run_grid(**kwargs):
        return [
            VariantResult(
                name="conc4_isl1024_osl1024", extra_server_args="",
                extra_envs={"CONC": "4", "ISL": "1024", "OSL": "1024"},
                status="succeeded", output_throughput=120.0, e2el_mean_ms=40.0,
            ),
        ]

    monkeypatch.setattr(sw, "run_grid", fake_run_grid)

    ex = sw.SweepExecutor(session_dir=tmp_path)
    ctx = _ctx(tmp_path, {
        "config_path": str(cfg),
        "conc_values": [4],
        "isl_osl_configs": ["1024:1024"],
    })
    out = await ex(ctx)
    assert out["status"] == "succeeded"
    assert out["grid_size"] == 1
    assert out["best_for_each_conc"]["4"]["output_throughput"] == 120.0
    assert len(out["pareto_front"]) == 1
