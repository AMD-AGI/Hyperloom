"""BackendsExecutor ``backends_search`` ledger regression tests.

Verifies the four guarantees added in Phase 2:

1. The ``backends_search.tested`` ledger is keyed by **content
   fingerprint** (sha1 prefix), not by display name.
2. LLM-supplied ``params.grid`` is filtered through the same ledger as
   the default grid — so renaming an already-tested variant cannot
   bypass dedup.
3. ``backends_search_exhausted`` is True iff no fresh fingerprint
   survived dedup this round.
4. The Phase 1 dedup path tolerates legacy ledgers that lack
   fingerprints (entries keyed by name with stored args/envs).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.action_executors import backends as backends_mod
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
    variant_fingerprint,
)
from inference_optimizer.orchestrator.action_executors.backends import (
    BackendsExecutor,
)


def _ctx(tmp_path: Path, params: dict) -> SimpleNamespace:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "benchmark:\n  envs: {}\n  framework: sglang\n", encoding="utf-8"
    )
    params = {
        "config_path": str(cfg),
        "output_dir": str(tmp_path / "out"),
        # AST discovery would scan a (possibly absent) sglang source tree,
        # producing noise unrelated to the dedup logic under test.
        "disable_discovery": True,
        **params,
    }
    return SimpleNamespace(
        task=SimpleNamespace(task_id="backends-task", params=params),
    )


def _fake_runner(tput_by_name: dict[str, float]):
    async def fake_run_grid(*, grid, **kwargs):
        return [
            VariantResult(
                name=v.name,
                extra_sglang_args=v.extra_sglang_args,
                extra_envs=v.extra_envs,
                status="succeeded",
                output_throughput=tput_by_name.get(v.name, 100.0),
            )
            for v in grid
        ]
    return fake_run_grid


@pytest.mark.asyncio
async def test_backends_search_tested_keyed_by_fingerprint(monkeypatch, tmp_path):
    grid = [
        GridVariant("A", "--A"),
        GridVariant("B", "--B"),
    ]
    monkeypatch.setattr(backends_mod, "run_grid", _fake_runner({"A": 110.0, "B": 90.0}))
    ex = BackendsExecutor(default_grid=grid, default_max_candidates_per_round=0,
                          default_max_synergy_combos=0)
    result = await ex(_ctx(tmp_path, {"base_tput": 100.0}))

    update = result["backends_search_update"]
    fp_a = variant_fingerprint("--A", {})
    fp_b = variant_fingerprint("--B", {})
    assert set(update["tested"].keys()) == {fp_a, fp_b}
    assert update["tested"][fp_a]["name"] == "A"
    assert update["tested"][fp_b]["name"] == "B"
    assert update["name_index"]["A"] == fp_a
    assert update["name_index"]["B"] == fp_b
    # B did not beat base_tput * 1.01 → rejected, A did → not in rejected.
    rejected_fps = {r["fingerprint"] for r in update["rejected"]}
    assert fp_b in rejected_fps
    assert fp_a not in rejected_fps


@pytest.mark.asyncio
async def test_backends_search_filters_default_grid_by_fingerprint(monkeypatch, tmp_path):
    """Resume with A tested → default grid drops A even though name matches."""
    grid = [GridVariant("A", "--A"), GridVariant("B", "--B")]
    monkeypatch.setattr(backends_mod, "run_grid", _fake_runner({"A": 110.0, "B": 90.0}))
    fp_a = variant_fingerprint("--A", {})
    prior_search = {
        "schema_version": 1,
        "accepted": [],
        "rejected": [],
        "tested": {fp_a: {"name": "A", "extra_sglang_args": "--A", "extra_envs": {}}},
        "name_index": {"A": fp_a},
        "cursor": 1,
    }
    ex = BackendsExecutor(default_grid=grid, default_max_candidates_per_round=0,
                          default_max_synergy_combos=0)
    result = await ex(_ctx(tmp_path, {
        "base_tput": 100.0,
        "backends_search": prior_search,
    }))

    tested_names_this_round = [r["name"] for r in result["all_results"]]
    assert tested_names_this_round == ["B"]


@pytest.mark.asyncio
async def test_backends_search_filters_llm_grid_by_fingerprint(monkeypatch, tmp_path):
    """LLM rename of an already-tested variant must NOT bypass dedup."""
    monkeypatch.setattr(backends_mod, "run_grid", _fake_runner({"B": 130.0}))
    fp_a = variant_fingerprint("--A", {})
    prior_search = {
        "schema_version": 1,
        "accepted": [],
        "rejected": [],
        "tested": {fp_a: {"name": "A", "extra_sglang_args": "--A", "extra_envs": {}}},
        "name_index": {"A": fp_a},
        "cursor": 1,
    }
    # LLM resubmits the same content under a freshly invented name plus a
    # genuinely new variant B.
    llm_grid = [
        {"name": "A_renamed_by_llm", "extra_sglang_args": "--A", "extra_envs": {}},
        {"name": "B",                "extra_sglang_args": "--B", "extra_envs": {}},
    ]
    ex = BackendsExecutor(default_max_candidates_per_round=0,
                          default_max_synergy_combos=0)
    result = await ex(_ctx(tmp_path, {
        "base_tput": 100.0,
        "grid": llm_grid,
        "backends_search": prior_search,
    }))

    tested_names_this_round = [r["name"] for r in result["all_results"]]
    assert "A_renamed_by_llm" not in tested_names_this_round
    assert tested_names_this_round == ["B"]


@pytest.mark.asyncio
async def test_backends_search_legacy_name_index_blocks_renamed_resubmit(
    monkeypatch, tmp_path,
):
    """Legacy ledger entry without a fingerprint still blocks a rename."""
    monkeypatch.setattr(backends_mod, "run_grid", _fake_runner({"B": 130.0}))
    # Simulate a pre-fingerprint state.json where ``tested`` is keyed by
    # name and the rows have no ``fingerprint`` field, but ``extra_sglang_args``
    # is preserved so dedup can re-derive the fingerprint at runtime.
    prior_search = {
        "schema_version": 1,
        "accepted": [],
        "rejected": [],
        "tested": {"A": {"name": "A", "extra_sglang_args": "--A", "extra_envs": {}}},
        "name_index": {},
        "cursor": 1,
    }
    llm_grid = [
        {"name": "A_v2", "extra_sglang_args": "--A", "extra_envs": {}},
        {"name": "B",    "extra_sglang_args": "--B", "extra_envs": {}},
    ]
    ex = BackendsExecutor(default_max_candidates_per_round=0,
                          default_max_synergy_combos=0)
    result = await ex(_ctx(tmp_path, {
        "base_tput": 100.0,
        "grid": llm_grid,
        "backends_search": prior_search,
    }))
    tested_names_this_round = [r["name"] for r in result["all_results"]]
    assert tested_names_this_round == ["B"]


@pytest.mark.asyncio
async def test_backends_search_exhausted_when_all_filtered(monkeypatch, tmp_path):
    """If every LLM-supplied variant is already in the ledger,
    ``backends_search_exhausted`` is True and no variants run."""
    called = {"count": 0}

    async def fake_run_grid(*, grid, **kwargs):
        called["count"] += len(grid)
        return []

    monkeypatch.setattr(backends_mod, "run_grid", fake_run_grid)
    fp_a = variant_fingerprint("--A", {})
    fp_b = variant_fingerprint("--B", {})
    prior_search = {
        "schema_version": 1,
        "accepted": [],
        "rejected": [],
        "tested": {
            fp_a: {"name": "A", "extra_sglang_args": "--A", "extra_envs": {}},
            fp_b: {"name": "B", "extra_sglang_args": "--B", "extra_envs": {}},
        },
        "name_index": {"A": fp_a, "B": fp_b},
        "cursor": 2,
    }
    llm_grid = [
        {"name": "A2", "extra_sglang_args": "--A", "extra_envs": {}},
        {"name": "B2", "extra_sglang_args": "--B", "extra_envs": {}},
    ]
    ex = BackendsExecutor(default_max_candidates_per_round=0,
                          default_max_synergy_combos=0)
    result = await ex(_ctx(tmp_path, {
        "base_tput": 100.0,
        "grid": llm_grid,
        "backends_search": prior_search,
    }))
    assert result["backends_search_exhausted"] is True
    assert called["count"] == 0
