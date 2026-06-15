# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the dual remote recipe-KB composite client."""
from __future__ import annotations

import argparse
from typing import Any


CID = "inference:test-model:mi300x:sglang:0.5.11:fp8"


class _FakeSource:
    returns_arbor_shape = True
    enabled = True

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.seen_prefer: dict[str, Any] | None = None

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.seen_prefer = kwargs.get("prefer")
        return list(self.rows)

    def health(self) -> bool:
        return True


def test_composite_search_accepts_prefer_and_merges_sources() -> None:
    from inference_optimizer.recipe_kb.composite_remote import CompositeRemoteRecipeClient

    gbrain = _FakeSource([{
        "canonical_id": CID,
        "model": "test-model",
        "hardware": "mi300x",
        "framework": "sglang",
        "framework_version": "0.5.11",
        "precision": "fp8",
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
        "best_config": {"extra_envs": {"SGLANG_USE_AITER": "1"}},
        "best_throughput": 0.0,
    }])
    cortex = _FakeSource([{
        "canonical_id": CID,
        "model": "test-model",
        "hardware": "mi300x",
        "framework": "sglang",
        "framework_version": "0.5.11",
        "precision": "fp8",
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
        "best_config": {},
        "best_throughput": 9868.3,
        "lessons": [{"statement": "cortex lesson"}],
    }])

    client = CompositeRemoteRecipeClient(
        [gbrain, cortex],
        names=["gbrain", "cortex"],
    )
    rows = client.search(
        label_match={"model": "test-model"},
        limit=5,
        prefer={"isl": 1024, "osl": 1024},
    )

    assert gbrain.seen_prefer == {"isl": 1024, "osl": 1024}
    assert cortex.seen_prefer == {"isl": 1024, "osl": 1024}
    assert len(rows) == 1
    row = rows[0]
    assert set(row["_sources"]) == {"gbrain", "cortex"}
    assert row["best_config"] == {"extra_envs": {"SGLANG_USE_AITER": "1"}}
    assert row["best_throughput"] == 9868.3
    assert row["_field_sources"]["best_config"] == "gbrain"
    assert row["_field_sources"]["best_throughput"] == "cortex"
    assert row["_field_sources"]["lessons"] == ["cortex"]


def _gbrain_actionable_row() -> dict[str, Any]:
    return {
        "canonical_id": CID, "model": "test-model", "hardware": "mi300x",
        "framework": "sglang", "framework_version": "0.5.11", "precision": "fp8",
        "authority": "EXPERIENTIAL", "confidence": 0.85,
        "best_config": {"extra_envs": {"SGLANG_USE_AITER": "1"}},
        "best_throughput": 0.0,
    }


def _cortex_tput_row() -> dict[str, Any]:
    return {
        "canonical_id": CID, "model": "test-model", "hardware": "mi300x",
        "framework": "sglang", "framework_version": "0.5.11", "precision": "fp8",
        "authority": "EXPERIENTIAL", "confidence": 0.85,
        "best_config": {}, "best_throughput": 9868.3,
        "lessons": [{"statement": "cortex lesson"}],
    }


def test_composite_stamps_source_candidates() -> None:
    from inference_optimizer.recipe_kb.composite_remote import CompositeRemoteRecipeClient

    gbrain = _FakeSource([_gbrain_actionable_row()])
    cortex = _FakeSource([_cortex_tput_row()])
    client = CompositeRemoteRecipeClient([gbrain, cortex], names=["gbrain", "cortex"])

    rows = client.search(label_match={"model": "test-model"}, limit=5)

    assert len(rows) == 1
    assert rows[0]["_source_candidates"] == {"gbrain": 1, "cortex": 1}


def test_dispatcher_audit_surfaces_per_path_provenance(tmp_path) -> None:
    from inference_optimizer.recipe_kb import RecipeKB
    from inference_optimizer.recipe_kb.composite_remote import CompositeRemoteRecipeClient
    from inference_optimizer.recipe_kb.local_store import LocalRecipeStore

    gbrain = _FakeSource([_gbrain_actionable_row()])
    cortex = _FakeSource([_cortex_tput_row()])
    composite = CompositeRemoteRecipeClient([gbrain, cortex], names=["gbrain", "cortex"])
    events: list[dict[str, Any]] = []
    kb = RecipeKB(
        local=LocalRecipeStore(root=tmp_path),
        remote=composite,
        audit_hook=events.append,
    )

    row = kb.get_recipe(canonical_id=CID)

    assert row is not None
    assert events, "expected a recipe-snapshot audit event"
    ev = events[-1]
    assert ev["remote"] == "composite"
    res = ev["result"]
    assert set(res["sources"]) == {"gbrain", "cortex"}
    assert res["best_config_source"] == "gbrain"
    assert res["field_sources"]["best_throughput"] == "cortex"
    assert res["source_candidates"] == {"gbrain": 1, "cortex": 1}


def test_cli_kb_both_mode_builds_composite(tmp_path, monkeypatch) -> None:
    from inference_optimizer.cli_kb import _build_recipe_kb_dispatcher
    from inference_optimizer.recipe_kb.composite_remote import CompositeRemoteRecipeClient

    monkeypatch.setenv("RECIPE_KB_REMOTE", "both")
    monkeypatch.setenv("GBRAIN_BASE_URL", "http://gbrain.invalid")
    monkeypatch.setenv("GBRAIN_TOKEN", "token")
    monkeypatch.setenv("CORTEX_KB_URL", "http://cortex.invalid")

    kb = _build_recipe_kb_dispatcher(argparse.Namespace(
        degraded_kb=False,
        cortex_kb_url=None,
        local_kb_root=str(tmp_path),
    ))

    assert isinstance(kb.remote, CompositeRemoteRecipeClient)
    assert kb.remote._names == ["gbrain", "cortex"]
    assert len(kb.remote._sources) == 2
    assert kb.remote.returns_arbor_shape is True
