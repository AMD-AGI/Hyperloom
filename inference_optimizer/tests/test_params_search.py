"""Params round-search regression tests.

The intended flow is:
1. test single candidates against the same base
2. combine positive candidates and promote the best observed config
3. persist accepted/rejected/tested so resume continues with the next candidate
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.action_executors import params as params_mod
from inference_optimizer.orchestrator.action_executors._grid_runner import VariantResult
from inference_optimizer.orchestrator.action_executors.params import (
    GridVariant,
    ParamsExecutor,
)


def _ctx(tmp_path: Path, params: dict) -> SimpleNamespace:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("benchmark:\n  envs: {}\n", encoding="utf-8")
    params = {"config_path": str(cfg), "output_dir": str(tmp_path / "out"), **params}
    return SimpleNamespace(task=SimpleNamespace(task_id="task123456", params=params))


@pytest.mark.asyncio
async def test_params_search_single_then_combo(monkeypatch, tmp_path):
    """baseline=100, A=110, B=105, AB=112, C=90 → choose A+B."""
    grid = [
        GridVariant("A", "--A"),
        GridVariant("B", "--B"),
        GridVariant("C", "--C"),
    ]

    async def fake_run_grid(*, base_yaml_path, base_extra_args, grid, output_root,
                            variant_timeout_sec, **kwargs):
        out = []
        for v in grid:
            tput = {
                "A": 110.0,
                "B": 105.0,
                "C": 90.0,
                "combo_A_B": 112.0,
            }[v.name]
            out.append(VariantResult(
                name=v.name,
                extra_sglang_args=v.extra_sglang_args,
                extra_envs=v.extra_envs,
                status="succeeded",
                output_throughput=tput,
            ))
        return out

    monkeypatch.setattr(params_mod, "run_grid", fake_run_grid)
    ex = ParamsExecutor(default_grid=grid, default_max_candidates_per_round=3)
    result = await ex(_ctx(tmp_path, {"base_tput": 100.0}))

    assert result["output_throughput"] == pytest.approx(112.0)
    assert result["best_variant"]["name"] == "combo_A_B"
    assert result["best_variant"]["extra_sglang_args"] == "--A --B"

    search = result["params_search_update"]
    assert [v["name"] for v in search["accepted"]] == ["A", "B"]
    assert [v["name"] for v in search["rejected"]] == ["C"]
    assert set(search["tested"]) == {"A", "B", "C"}
    assert search["cursor"] == 3


@pytest.mark.asyncio
async def test_params_search_resume_skips_tested_and_uses_current_combo(
    monkeypatch, tmp_path,
):
    """If A+B is accepted and C rejected, resume tests D as A+B+D."""
    grid = [
        GridVariant("A", "--A"),
        GridVariant("B", "--B"),
        GridVariant("C", "--C"),
        GridVariant("D", "--D"),
    ]
    search = {
        "schema_version": 1,
        "accepted": [
            {"name": "A", "extra_sglang_args": "--A", "extra_envs": {}, "note": ""},
            {"name": "B", "extra_sglang_args": "--B", "extra_envs": {}, "note": ""},
        ],
        "rejected": [
            {"name": "C", "extra_sglang_args": "--C", "extra_envs": {}, "note": ""},
        ],
        "tested": {"A": {}, "B": {}, "C": {}},
        "cursor": 3,
    }
    seen = []

    async def fake_run_grid(*, base_yaml_path, base_extra_args, grid, output_root,
                            variant_timeout_sec, **kwargs):
        seen.append((base_extra_args, [v.name for v in grid]))
        return [
            VariantResult(
                name=v.name,
                extra_sglang_args=v.extra_sglang_args,
                extra_envs=v.extra_envs,
                status="succeeded",
                output_throughput=120.0,
            )
            for v in grid
        ]

    monkeypatch.setattr(params_mod, "run_grid", fake_run_grid)
    ex = ParamsExecutor(default_grid=grid, default_max_candidates_per_round=3)
    result = await ex(_ctx(tmp_path, {
        "base_tput": 112.0,
        "base_extra_args": "--A --B",
        "params_search": search,
    }))

    assert seen == [("--A --B", ["D"])]
    assert result["best_variant"]["name"] == "D"
    assert result["best_variant"]["extra_sglang_args"] == "--A --B --D"

    updated = result["params_search_update"]
    assert [v["name"] for v in updated["accepted"]] == ["A", "B", "D"]
    assert [v["name"] for v in updated["rejected"]] == ["C"]
    assert set(updated["tested"]) == {"A", "B", "C", "D"}


@pytest.mark.asyncio
async def test_params_search_seeds_current_best_for_legacy_resume(
    monkeypatch, tmp_path,
):
    """Old sessions may have current_best.variant_name but no params_search."""
    grid = [
        GridVariant("A", "--A"),
        GridVariant("B", "--B"),
    ]
    seen = []

    async def fake_run_grid(*, base_yaml_path, base_extra_args, grid, output_root,
                            variant_timeout_sec, **kwargs):
        seen.append((base_extra_args, [v.name for v in grid]))
        return [
            VariantResult(
                name=v.name,
                extra_sglang_args=v.extra_sglang_args,
                extra_envs=v.extra_envs,
                status="succeeded",
                output_throughput=115.0,
            )
            for v in grid
        ]

    monkeypatch.setattr(params_mod, "run_grid", fake_run_grid)
    ex = ParamsExecutor(default_grid=grid)
    result = await ex(_ctx(tmp_path, {
        "base_tput": 110.0,
        "base_extra_args": "--A",
        "base_variant_name": "A",
        "params_search": {},
    }))

    assert seen == [("--A", ["B"])]
    updated = result["params_search_update"]
    assert [v["name"] for v in updated["accepted"]] == ["A", "B"]
