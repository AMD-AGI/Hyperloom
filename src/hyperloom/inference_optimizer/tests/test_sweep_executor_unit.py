# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the sweep ActionRunner helpers and __call__ flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.common.coerce import to_int
from hyperloom.orchestrator.actions.executors import sweep as sw
from hyperloom.orchestrator.actions.executors._grid_base import pareto_front
from hyperloom.orchestrator.actions.executors._grid_runner import (
    VariantResult,
)


# ---- int coercion ----


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0),
        ("", 0),
        ("42", 42),
        (7, 7),
        ("bad", 0),
        ("  9 ", 9),
    ],
)
def test_coerce_int(value, expected):
    assert to_int(value, default=0) == expected


# ---- _build_grid ----


def test_build_grid_basic():
    grid, skipped = sw._build_grid(
        conc_values=[4],
        isl_osl_configs=["1024:1024"],
        num_prompts_factor=5,
        base_extra_args="--x",
    )
    assert len(grid) == 1
    assert skipped == []
    v = grid[0]
    assert v.extra_envs["CONC"] == "4"
    assert v.extra_envs["NUM_PROMPTS"] == "20"


def test_build_grid_threads_removal_controls():
    grid, skipped = sw._build_grid(
        conc_values=[4],
        isl_osl_configs=["1024:1024"],
        num_prompts_factor=5,
        base_extra_args="--x",
        base_remove_args=["--bad-base"],
        base_unset_envs=["SGLANG_BAD_ENV"],
    )
    assert skipped == []
    assert grid[0].remove_args == ["--bad-base"]
    assert grid[0].unset_envs == ["SGLANG_BAD_ENV"]


def test_build_grid_threads_replace_mode():
    grid, _ = sw._build_grid(
        conc_values=[4],
        isl_osl_configs=["1024:1024"],
        num_prompts_factor=5,
        base_extra_args="--replacement",
        base_args_mode="replace",
    )
    assert grid[0].args_mode == "replace"


def test_build_grid_always_disables_eval():
    """Sweep variants always skip the (concurrency-invariant) accuracy eval."""
    grid, _ = sw._build_grid(
        conc_values=[4, 16],
        isl_osl_configs=["1024:1024"],
        num_prompts_factor=5,
        base_extra_args="",
    )
    assert grid
    assert all(v.extra_envs.get("RUN_EVAL") == "false" for v in grid)


def test_build_grid_malformed_io_skipped():
    grid, skipped = sw._build_grid(
        conc_values=[4],
        isl_osl_configs=["nonsense"],
        num_prompts_factor=5,
        base_extra_args="",
    )
    assert grid == []
    assert skipped == []


def test_build_grid_max_model_len_skip():
    grid, skipped = sw._build_grid(
        conc_values=[4],
        isl_osl_configs=["8192:1024"],
        num_prompts_factor=5,
        base_extra_args="",
        max_model_len=4096,
    )
    assert grid == []
    assert skipped[0]["status"] == "skipped"
    assert "exceeds max_model_len" in skipped[0]["skip_reason"]


# ---- _result_dict ----


def test_result_dict_surfaces_dims():
    vr = VariantResult(
        name="v",
        extra_server_args="",
        extra_envs={"CONC": "16", "ISL": "1024", "OSL": "512"},
        status="succeeded",
        output_throughput=100.0,
        e2el_mean_ms=50.0,
    )
    d = sw._result_dict(vr)
    assert d["conc"] == 16 and d["isl"] == 1024 and d["osl"] == 512


# ---- pareto_front (shared by the native and GEAK sweeps) ----


@pytest.mark.parametrize("latency_key", ["e2el_mean_ms", "ttft_mean_ms"])
def test_pareto_front_dominance(latency_key: str):
    entries = [
        {"status": "succeeded", "output_throughput": 100, latency_key: 10},
        {"status": "succeeded", "output_throughput": 90, latency_key: 20},  # dominated
        {"status": "succeeded", "output_throughput": 80, latency_key: 5},  # not dominated
        {"status": "failed", "output_throughput": 999, latency_key: 1},  # excluded (failed)
    ]
    front = pareto_front(entries, latency_key=latency_key)
    tputs = sorted(e["output_throughput"] for e in front)
    assert tputs == [80, 100]


def test_pareto_front_ignores_non_numeric():
    entries = [{"status": "succeeded", "output_throughput": None, "e2el_mean_ms": 10}]
    assert pareto_front(entries) == []


def test_pareto_front_max_total_tput_vs_max_intvty():
    """Composite Pareto: maximize total tput AND maximize p90 intvty."""
    entries = [
        {"status": "succeeded", "total_token_throughput": 1000, "intvty_p90": 500, "name": "a"},
        {"status": "succeeded", "total_token_throughput": 900, "intvty_p90": 800, "name": "b"},
        {"status": "succeeded", "total_token_throughput": 800, "intvty_p90": 400, "name": "c"},
        {"status": "succeeded", "total_token_throughput": 1100, "intvty_p90": 500, "name": "d"},
        {"status": "failed", "total_token_throughput": 9999, "intvty_p90": 9999, "name": "fail"},
    ]
    front = pareto_front(
        entries,
        x_key="total_token_throughput",
        y_key="intvty_p90",
        y_higher_is_better=True,
    )
    names = {e["name"] for e in front}
    assert names == {"b", "d"}


def test_best_entry_for_each_conc_composite_prefers_score(monkeypatch):
    """Flag on: a flat-output input lift beats a higher-output cell at the same conc."""
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    from hyperloom.orchestrator.actions.executors._grid_base import best_entry_for_each_conc

    baseline = {
        "output_throughput": 800.0,
        "input_throughput": 10000.0,
        "intvty_p90": 700.0,
    }
    high_out = {
        "status": "succeeded",
        "conc": 4,
        "name": "high_out",
        "output_throughput": 900.0,
        "input_throughput": 10000.0,
        "intvty_p90": 700.0,
    }
    input_lift = {
        "status": "succeeded",
        "conc": 4,
        "name": "input_lift",
        "output_throughput": 800.0,
        "input_throughput": 12000.0,
        "intvty_p90": 700.0,
    }
    best = best_entry_for_each_conc(
        [high_out, input_lift],
        framework="sglang",
        baseline_perf=baseline,
    )
    assert best["4"]["name"] == "input_lift"


def test_best_entry_for_each_conc_falls_back_without_triple(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    from hyperloom.orchestrator.actions.executors._grid_base import best_entry_for_each_conc

    entries = [
        {"status": "succeeded", "conc": 4, "name": "low", "output_throughput": 100.0},
        {"status": "succeeded", "conc": 4, "name": "high", "output_throughput": 200.0},
    ]
    best = best_entry_for_each_conc(entries, framework="sglang", baseline_perf=None)
    assert best["4"]["name"] == "high"


def test_select_sweep_pareto_composite_then_fallback(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    from hyperloom.orchestrator.actions.executors._grid_base import select_sweep_pareto

    composite_cells = [
        {
            "status": "succeeded",
            "total_token_throughput": 1000,
            "intvty_p90": 500,
            "output_throughput": 10,
            "e2el_mean_ms": 1,
            "name": "a",
        },
        {
            "status": "succeeded",
            "total_token_throughput": 900,
            "intvty_p90": 800,
            "output_throughput": 999,
            "e2el_mean_ms": 1,
            "name": "b",
        },
    ]
    front = select_sweep_pareto(composite_cells, framework="sglang")
    assert {e["name"] for e in front} == {"a", "b"}

    tput_only = [
        {"status": "succeeded", "output_throughput": 100, "e2el_mean_ms": 10, "name": "fast"},
        {"status": "succeeded", "output_throughput": 90, "e2el_mean_ms": 20, "name": "slow"},
    ]
    fallback = select_sweep_pareto(tput_only, framework="sglang")
    assert [e["name"] for e in fallback] == ["fast"]


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
                name="conc4_isl1024_osl1024",
                extra_server_args="",
                extra_envs={"CONC": "4", "ISL": "1024", "OSL": "1024"},
                status="succeeded",
                output_throughput=120.0,
                e2el_mean_ms=40.0,
            ),
        ]

    monkeypatch.setattr(sw, "run_grid", fake_run_grid)

    ex = sw.SweepExecutor(session_dir=tmp_path)
    ctx = _ctx(
        tmp_path,
        {
            "config_path": str(cfg),
            "conc_values": [4],
            "isl_osl_configs": ["1024:1024"],
        },
    )
    out = await ex(ctx)
    assert out["status"] == "succeeded"
    assert out["grid_size"] == 1
async def test_call_success_composite_ranks_on_score(tmp_path, monkeypatch):
    """Flag on: best-per-conc is *S*; Pareto is total tput vs p90 intvty."""
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "composite_v1")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("k: v\n", encoding="utf-8")
    monkeypatch.setattr(sw, "materialize_config_with_envs", lambda *a, **k: cfg)

    async def fake_run_grid(**kwargs):
        return [
            VariantResult(
                name="high_out",
                extra_server_args="",
                extra_envs={"CONC": "4", "ISL": "1024", "OSL": "1024"},
                status="succeeded",
                output_throughput=900.0,
                total_token_throughput=1000.0,
                input_throughput=10000.0,
                intvty_p90=500.0,
                e2el_mean_ms=40.0,
            ),
            VariantResult(
                name="input_lift",
                extra_server_args="",
                extra_envs={"CONC": "4", "ISL": "8192", "OSL": "1024"},
                status="succeeded",
                output_throughput=800.0,
                total_token_throughput=900.0,
                input_throughput=12000.0,
                intvty_p90=800.0,
                e2el_mean_ms=40.0,
            ),
        ]

    monkeypatch.setattr(sw, "run_grid", fake_run_grid)
    baseline_perf = {
        "output_throughput": 800.0,
        "input_throughput": 10000.0,
        "intvty_p90": 700.0,
    }
    ex = sw.SweepExecutor(session_dir=tmp_path)
    ctx = _ctx(
        tmp_path,
        {
            "config_path": str(cfg),
            "conc_values": [4],
            "isl_osl_configs": ["1024:1024", "8192:1024"],
        },
    )
    ctx.extra["shared_state"] = SimpleNamespace(
        framework="sglang",
        baseline_perf=baseline_perf,
        model_path="",
    )
    out = await ex(ctx)
    assert out["status"] == "succeeded"
    assert out["best_for_each_conc"]["4"]["name"] == "input_lift"
    assert {e["name"] for e in out["pareto_front"]} == {"high_out", "input_lift"}
