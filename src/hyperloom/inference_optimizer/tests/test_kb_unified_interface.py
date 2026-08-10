"""Tests for the local Recipe KB warm-start interface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hyperloom.orchestrator.knowledge.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    recipe_canonical_id,
)
from hyperloom.orchestrator.knowledge.recipe_kb.dispatcher import _prefer_score
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import (
    _build_warm_prefer,
    _build_warm_start_context,
    _recipe_is_actionable,
)


def test_prefer_score_counts_local_matches() -> None:
    row = {"framework_version": "0.5.11", "tp": 8, "conc": 32}
    assert _prefer_score(row, {"tp": 8}) == 1
    assert (
        _prefer_score(
            row,
            {"tp": 8, "framework_version": "0.5.11"},
        )
        == 2
    )
    assert _prefer_score(row, {"tp": 4, "ep": 2}) == 0


def test_local_search_prefer_reorders_without_dropping(
    tmp_path: Path,
) -> None:
    store = LocalRecipeStore(tmp_path / "kb")
    kb = RecipeKB(local=store)
    rows = [
        ("mi300x", 4),
        ("mi355x", 8),
    ]
    for hardware, tp in rows:
        cid = recipe_canonical_id(
            model="qwen3-32b",
            hardware=hardware,
            framework_name="sglang",
            framework_version="0.5.11",
            precision="fp8",
        )
        kb.put_recipe(
            canonical_id=cid,
            model="qwen3-32b",
            hardware=hardware,
            extras={"tp": tp},
        )
    found = kb.search(
        label_match={"model": "qwen3-32b"},
        prefer={"tp": 8},
    )
    assert len(found) == 2
    assert found[0]["tp"] == 8


def test_recipe_actionability_and_warm_context() -> None:
    assert _recipe_is_actionable(
        {"best_config": {"extra_server_args": "--x 1"}}
    )
    assert not _recipe_is_actionable(
        {"canonical_id": "inference:m:h:f:mt:a:v:p", "best_config": {}}
    )
    context = _build_warm_start_context(
        status="hit",
        tier="exact",
        confidence=1.0,
        canonical_id="inference:m:h:f:mt:a:v:p",
        source="recipe-kb",
        recipe={
            "canonical_id": "inference:m:h:f:mt:a:v:p",
            "best_config": {
                "extra_server_args": "--x 1",
                "extra_envs": {"A": "1"},
            },
            "best_throughput": 100.0,
        },
    )
    assert context["status"] == "hit"
    assert context["recommended_replay"]["extra_server_args"] == "--x 1"
    assert context["recommended_replay"]["extra_envs"] == {"A": "1"}


def test_build_warm_prefer_skips_empty_values() -> None:
    prefer = _build_warm_prefer(
        SimpleNamespace(
            tp=8,
            ep=0,
            pp=0,
            conc=0,
            isl=0,
            osl=0,
            max_model_len=0,
            baseline_workload_extra={"quant_scheme": ""},
        ),
        "0.5.11",
    )
    assert prefer == {"tp": 8, "framework_version": "0.5.11"}
