# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the local-only RecipeKB compatibility facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.recipe_snapshot_constants import RecipeIdentity
from hyperloom.orchestrator.knowledge.recipe_kb import (
    RecipeUpsert,
    LocalRecipeStore,
    Recipe,
    RecipeKB,
    recipe_canonical_id,
)


def _cid() -> str:
    return recipe_canonical_id(
        RecipeIdentity(
            model="m",
            hardware="mi300x",
            framework_name="sglang",
            framework_version="0.4.5",
            precision="fp8",
        )
    )


def test_removed_prs_tested_field_does_not_round_trip() -> None:
    recipe = Recipe.from_dict(
        {
            "canonical_id": _cid(),
            "prs_tested": [{"url": "https://example.test/pr/1"}],
        }
    )

    assert "prs_tested" not in recipe.extras
    assert "prs_tested" not in recipe.to_dict()


@pytest.fixture
def kb(tmp_path: Path) -> RecipeKB:
    return RecipeKB(local=LocalRecipeStore(tmp_path / "kb"))


def test_local_crud_and_authority_read(kb: RecipeKB) -> None:
    cid = _cid()
    result = kb.put_recipe(RecipeUpsert(canonical_id=cid, model="m", hardware="mi300x", best_throughput=10.0))
    assert result["created"] is True
    assert kb.get_recipe(canonical_id=cid)["best_throughput"] == 10.0
    assert kb.get_authoritative_recipe(canonical_id=cid)["canonical_id"] == cid
    assert kb.search(label_match={"model": "m"})[0]["canonical_id"] == cid


def test_audit_is_local_and_best_effort(kb: RecipeKB) -> None:
    events: list[dict] = []
    kb.audit_hook = events.append
    kb.put_recipe(
        RecipeUpsert(
            canonical_id=_cid(),
            lessons=[{"statement": "x", "measured_impact": "1%"}],
            provenance={
                "generator": "coordinator",
                "details": {"phase": "close_finalize"},
            },
        )
    )
    kb.get_recipe(canonical_id=_cid())
    assert events[0]["resolution"] == "local_write"
    assert events[0]["delta"]["lessons"] == 1
    assert events[1]["resolution"] == "local"
    assert all(event["remote"] == "none" for event in events)

    kb.audit_hook = lambda _event: (_ for _ in ()).throw(OSError("down"))
    assert kb.get_recipe(canonical_id=_cid()) is not None


def test_mode_and_backend_are_fixed_local_labels(kb: RecipeKB) -> None:
    assert (kb.mode, kb.backend_name) == ("local", "local-json")

    events: list[dict] = []
    kb.audit_hook = events.append
    kb.search(label_match={"model": "m"})
    assert events[0]["backend"] == events[0]["provenance"]["backend"] == "local-json"
    assert events[0]["request"] == {"canonical_id": None, "label_match": {"model": "m"}}
