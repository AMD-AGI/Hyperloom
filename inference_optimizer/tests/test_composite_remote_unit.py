# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ``recipe_kb.composite_remote``: precedence/richness helpers,
list-union + group merge with field provenance, and the fan-out client
(search, get_recipe, list_* delegation, health/close, best-effort skips)."""
from __future__ import annotations

import pytest

from inference_optimizer.recipe_kb import composite_remote as cr
from inference_optimizer.recipe_kb.composite_remote import (
    CompositeRemoteRecipeClient,
)
from inference_optimizer.recipe_kb.remote_client import RemoteRecipeClientError


# -- pure helpers ----------------------------------------------------------
def test_richness_counts_populated_fields() -> None:
    assert cr._richness({}) == 0
    assert cr._richness({
        "best_config": {"a": 1}, "what_worked": ["x"], "lessons": ["l"],
        "stack_fingerprint": {"rocm": "6"}, "prs_tested": ["pr"],
    }) == 5


def test_precedence_key_handles_bad_numbers() -> None:
    key = cr._precedence_key({
        "authority": "EXPERIENTIAL", "confidence": "nan-ish",
        "best_throughput": "bad",
    })
    assert key[0] == 3  # EXPERIENTIAL rank
    assert key[1] == 0.0 and key[3] == 0.0  # coerced from bad strings


def test_is_empty_cases() -> None:
    assert cr._is_empty(None) is True
    assert cr._is_empty("") is True
    assert cr._is_empty([]) is True
    assert cr._is_empty(0) is True
    assert cr._is_empty(0.0) is True
    assert cr._is_empty("x") is False
    assert cr._is_empty(5) is False


def test_dedup_preserve_order() -> None:
    assert cr._dedup_preserve(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_union_lists_unhashable_items() -> None:
    rows = [
        {"_source": "gbrain", "lessons": [{"s": "use aiter"}]},
        {"_source": "cortex", "lessons": [{"s": "use aiter"}, {"s": "tune gemm"}]},
    ]
    merged, contributors = cr._union_lists(rows, "lessons")
    assert merged == [{"s": "use aiter"}, {"s": "tune gemm"}]
    assert contributors == ["gbrain", "cortex"]


def test_merge_group_backfill_and_provenance() -> None:
    rows = [
        {  # higher precedence base, missing best_config
            "_source": "gbrain", "canonical_id": "c1", "authority": "EXPERIENTIAL",
            "confidence": 0.9, "what_worked": ["aiter"], "best_config": {},
        },
        {  # lower precedence donor supplies best_config
            "_source": "cortex", "canonical_id": "c1", "authority": "VALIDATED",
            "confidence": 0.5, "best_config": {"extra_server_args": "--tp 1"},
            "what_worked": ["mla"],
        },
    ]
    merged = cr._merge_group(rows)
    assert merged["best_config"] == {"extra_server_args": "--tp 1"}  # back-filled
    assert set(merged["what_worked"]) == {"aiter", "mla"}  # unioned
    assert "gbrain" in merged["_sources"] and "cortex" in merged["_sources"]
    assert "_source" not in merged  # base _source dropped
    assert merged["_field_sources"]["best_config"] == "cortex"


# -- client ----------------------------------------------------------------
class _FakeSource:
    def __init__(self, name, rows=None, *, enabled=True, healthy=True):
        self._name = name
        self._rows = rows or []
        self.enabled = enabled
        self._healthy = healthy
        self.closed = False

    def search(self, **kwargs):
        return list(self._rows)

    def list_attempts(self, *, canonical_id, limit=100):
        return [{"attempt": canonical_id}]

    def list_session_attempts(self, *, session_id, limit=500):
        return [{"session": session_id}]

    def session_summary(self, *, session_id):
        return {"session_id": session_id}

    def health(self):
        return self._healthy

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _identity_arbor(monkeypatch):
    # Keep rows verbatim so assertions are deterministic (still executes line).
    monkeypatch.setattr(cr, "_v2_to_arbor", lambda row: row)


def test_init_drops_none_and_default_names() -> None:
    c = CompositeRemoteRecipeClient([_FakeSource("a"), None])
    assert len(c._sources) == 1
    assert c._names == ["a"]


def test_enabled_and_active() -> None:
    c = CompositeRemoteRecipeClient(
        [_FakeSource("a", enabled=True), _FakeSource("b", enabled=False)],
        names=["a", "b"],
    )
    assert c.enabled is True
    assert [n for n, _ in c._active()] == ["a"]


def test_search_merges_by_cid() -> None:
    rows_a = [{"canonical_id": "c1", "authority": "EXPERIENTIAL", "confidence": 0.9}]
    rows_b = [{"canonical_id": "c1", "authority": "VALIDATED", "confidence": 0.5}]
    c = CompositeRemoteRecipeClient(
        [_FakeSource("a", rows_a), _FakeSource("b", rows_b)], names=["a", "b"],
    )
    out = c.search(limit=5)
    assert len(out) == 1  # merged into one cid
    assert out[0]["canonical_id"] == "c1"


def test_fan_out_search_skips_failing_source() -> None:
    class _BoomRemote(_FakeSource):
        def search(self, **kwargs):
            raise RemoteRecipeClientError("down")

    class _BoomGeneric(_FakeSource):
        def search(self, **kwargs):
            raise RuntimeError("kaboom")

    c = CompositeRemoteRecipeClient(
        [_BoomRemote("a"), _BoomGeneric("b"),
         _FakeSource("c", [{"canonical_id": "c1"}])],
        names=["a", "b", "c"],
    )
    out = c.search(limit=5)
    assert len(out) == 1  # only the healthy source contributed


def test_fan_out_search_skips_empty_arbor(monkeypatch) -> None:
    monkeypatch.setattr(cr, "_v2_to_arbor", lambda row: {} if row.get("drop") else row)
    c = CompositeRemoteRecipeClient(
        [_FakeSource("a", [{"drop": True}, {"canonical_id": "c1"}])], names=["a"],
    )
    out = c.search(limit=5)
    assert len(out) == 1


def test_get_recipe_invalid_cid() -> None:
    c = CompositeRemoteRecipeClient([_FakeSource("a")], names=["a"])
    assert c.get_recipe(canonical_id="not-a-valid-cid") is None


def test_get_recipe_exact_and_fallback(monkeypatch) -> None:
    monkeypatch.setattr(cr, "_labels_from_canonical_id", lambda cid: {"model": "m"})
    rows = [{"canonical_id": "c1", "authority": "EXPERIENTIAL"}]
    c = CompositeRemoteRecipeClient([_FakeSource("a", rows)], names=["a"])
    assert c.get_recipe(canonical_id="c1")["canonical_id"] == "c1"
    # No exact match -> returns top merged row as fallback
    assert c.get_recipe(canonical_id="c2")["canonical_id"] == "c1"


def test_list_recent() -> None:
    rows = [{"canonical_id": "c1"}]
    c = CompositeRemoteRecipeClient([_FakeSource("a", rows)], names=["a"])
    assert len(c.list_recent(limit=10)) == 1


def test_list_delegation_to_first_active() -> None:
    c = CompositeRemoteRecipeClient(
        [_FakeSource("a", enabled=False), _FakeSource("b")], names=["a", "b"],
    )
    assert c.list_attempts(canonical_id="c1") == [{"attempt": "c1"}]
    assert c.list_session_attempts(session_id="s1") == [{"session": "s1"}]
    assert c.session_summary(session_id="s1") == {"session_id": "s1"}


def test_list_delegation_no_active() -> None:
    c = CompositeRemoteRecipeClient([_FakeSource("a", enabled=False)], names=["a"])
    assert c.list_attempts(canonical_id="c1") == []
    assert c.list_session_attempts(session_id="s1") == []
    assert c.session_summary(session_id="s1") is None


def test_health_and_close() -> None:
    healthy = _FakeSource("a", healthy=True)
    c = CompositeRemoteRecipeClient([healthy], names=["a"])
    assert c.health() is True
    c.close()
    assert healthy.closed is True


def test_health_false_when_unhealthy_and_raising() -> None:
    class _RaiseHealth(_FakeSource):
        def health(self):
            raise RuntimeError("boom")

    c = CompositeRemoteRecipeClient(
        [_RaiseHealth("a"), _FakeSource("b", healthy=False)], names=["a", "b"],
    )
    assert c.health() is False


def test_close_swallows_errors() -> None:
    class _RaiseClose(_FakeSource):
        def close(self):
            raise RuntimeError("nope")

    c = CompositeRemoteRecipeClient([_RaiseClose("a")], names=["a"])
    c.close()  # error swallowed
