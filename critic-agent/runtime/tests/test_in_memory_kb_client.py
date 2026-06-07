# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for :class:`runtime.in_memory_kb_client.InMemoryKBClient`.

The contract (§7.3) lists 4 promises this mock must keep; one test per
promise plus a couple of edge cases.
"""

from __future__ import annotations

import pytest

from runtime.errors import KBValidationError
from runtime.in_memory_kb_client import InMemoryKBClient


def _scope():
    return {
        "org": "hyperloom",
        "framework": "sglang",
        "model": "deepseek-r1",
        "model_family": "deepseek",
        "workload": "decode",
        "precision": "fp8",
    }


def test_upsert_creates_then_returns_existing():
    kb = InMemoryKBClient()
    out1 = kb.upsert({
        "scope": _scope(),
        "kind": "pitfall",
        "slug": "mla-fp8-torch-compile-incompat",
        "importance": 0.5,
        "summary": "first",
        "metadata": {"topic": "t"},
    })
    assert out1["created"] is True
    out2 = kb.upsert({
        "scope": _scope(),
        "kind": "pitfall",
        "slug": "mla-fp8-torch-compile-incompat",
        "importance": 0.4,
        "summary": "second",
        "metadata": {"topic": "t2"},
    })
    assert out2["created"] is False
    # Importance protected (G-2): incoming < existing keeps existing.
    assert out2["row"]["importance"] == 0.5
    assert "importance_protected" in out2["warnings"]
    # Summary partial-merged (G-1).
    assert out2["row"]["summary"] == "second"
    # Metadata deep-merged.
    assert out2["row"]["metadata"]["topic"] == "t2"


def test_upsert_missing_field_raises():
    kb = InMemoryKBClient()
    with pytest.raises(KBValidationError):
        kb.upsert({"kind": "pitfall", "slug": "x", "importance": 0.5})


def test_contradicts_edge_auto_mirrors():
    kb = InMemoryKBClient()
    a = kb.upsert({
        "scope": _scope(), "kind": "pitfall", "slug": "a-aaaaaa", "importance": 0.5,
    })["row"]["id"]
    b = kb.upsert({
        "scope": _scope(), "kind": "pitfall", "slug": "b-bbbbbb", "importance": 0.5,
    })["row"]["id"]
    out = kb.add_edges([{"kind": "contradicts", "from_id": a, "to_id": b}])
    assert {"from_id": a, "to_id": b, "kind": "contradicts"} in out["added"]
    assert any(m["from_id"] == b and m["to_id"] == a for m in out["mirrored_to"])
    rows = kb.all_rows()
    by_id = {r["id"]: r for r in rows}
    assert b in by_id[a]["edges"]["contradicts"]
    assert a in by_id[b]["edges"]["contradicts"]


def test_contradicts_with_missing_target_records_skip():
    kb = InMemoryKBClient()
    a = kb.upsert({
        "scope": _scope(), "kind": "pitfall", "slug": "a-aaaaaa", "importance": 0.5,
    })["row"]["id"]
    out = kb.add_edges([{"kind": "contradicts", "from_id": a, "to_id": "ghost"}])
    assert out["mirror_skipped"]
    assert out["mirror_skipped"][0]["reason"] == "dst_missing"


def test_list_returns_only_matching_scope_after_normalisation():
    kb = InMemoryKBClient()
    kb.upsert({
        "scope": _scope(), "kind": "pitfall", "slug": "a-pitfall1",
        "importance": 0.5,
    })
    other = dict(_scope())
    other["framework"] = "vllm"
    kb.upsert({
        "scope": other, "kind": "pitfall", "slug": "b-pitfall2",
        "importance": 0.5,
    })
    out = kb.list(scope_filter={"framework": "  SGLang "})
    assert all(r["scope"]["framework"] == "sglang" for r in out["entries"])
    assert any(r["slug"] == "a-pitfall1" for r in out["entries"])


def test_metadata_filter_supports_nested_paths():
    kb = InMemoryKBClient()
    kb.upsert({
        "scope": _scope(), "kind": "pitfall", "slug": "deep-pitfall",
        "importance": 0.5,
        "metadata": {"evidence": {"packet_evidence": ["benchmark.after.gain_pct"]}},
    })
    out = kb.list(
        scope_filter=_scope(),
        metadata_filter={"evidence": {"packet_evidence": ["benchmark.after.gain_pct"]}},
    )
    assert len(out["entries"]) == 1


def test_metadata_filter_array_contains_subset():
    kb = InMemoryKBClient()
    kb.upsert({
        "scope": _scope(), "kind": "technique", "slug": "tag-test",
        "importance": 0.5,
        "metadata": {"tags": ["dispatch", "kernel", "active_path"]},
    })
    hit = kb.list(
        scope_filter=_scope(),
        metadata_filter={"tags": ["dispatch"]},
    )
    miss = kb.list(
        scope_filter=_scope(),
        metadata_filter={"tags": ["nonexistent"]},
    )
    assert len(hit["entries"]) == 1
    assert miss["entries"] == []


def test_simulate_failure_drains_one_failure_per_endpoint():
    kb = InMemoryKBClient()
    kb.simulate_failure(endpoint="upsert", times=2, error={"code": 503})
    with pytest.raises(KBValidationError):
        kb.upsert({"scope": _scope(), "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5})
    with pytest.raises(KBValidationError):
        kb.upsert({"scope": _scope(), "kind": "pitfall", "slug": "abcdef-2", "importance": 0.5})
    out = kb.upsert({"scope": _scope(), "kind": "pitfall", "slug": "abcdef-3", "importance": 0.5})
    assert out["created"] is True


def test_batch_insert_with_upsert_on_conflict():
    kb = InMemoryKBClient()
    items = [
        {"scope": _scope(), "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5},
        {"scope": _scope(), "kind": "pitfall", "slug": "abcdef-2", "importance": 0.5},
    ]
    out = kb.batch_insert(items)
    assert out["count"] == 2


def test_list_default_excludes_deleted_rows():
    kb = InMemoryKBClient()
    out = kb.upsert({"scope": _scope(), "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5})
    rid = out["row"]["id"]
    kb._rows[rid].deleted = True
    assert kb.list(scope_filter=_scope())["entries"] == []
    assert kb.list(scope_filter=_scope(), include_deleted=True)["entries"]


def test_list_scope_value_normalisation_handles_uppercase_in_filter_too():
    kb = InMemoryKBClient()
    kb.upsert({"scope": _scope(), "kind": "pitfall", "slug": "abcdef-1", "importance": 0.5})
    out = kb.list(scope_filter={"model": "DEEPSEEK-R1"})
    assert out["entries"]
