# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for in_memory_kb_client + request_models validators."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.errors import KBValidationError, RequestValidationError
from hyperloom.agents.critic.runtime.in_memory_kb_client import (
    InMemoryKBClient,
    _deep_merge,
    _matches_metadata,
    _normalise_value,
)
from hyperloom.agents.critic.runtime.request_models import (
    _optional_list,
    _optional_str,
    _require_str,
    parse_request,
)


# --------------------------------------------------------------------------- #
# in_memory_kb_client pure helpers                                            #
# --------------------------------------------------------------------------- #
def test_normalise_value_none() -> None:
    assert _normalise_value(None) == ""  # line 36
    assert _normalise_value("  HeLLo ") == "hello"


def test_matches_metadata_branches() -> None:
    # Nested expected but value not a dict (line 88).
    assert _matches_metadata({"k": "x"}, {"k": {"a": 1}}) is False
    # Nested mismatch (line 90).
    assert _matches_metadata({"k": {"a": 2}}, {"k": {"a": 1}}) is False
    # List filter but haystack not a list (line 100).
    assert _matches_metadata({"k": "x"}, {"k": [1]}) is False
    # Scalar mismatch (lines 101-102).
    assert _matches_metadata({"k": "x"}, {"k": "y"}) is False
    # Array-contains success.
    assert _matches_metadata({"k": [1, 2, 3]}, {"k": [1, 2]}) is True


def test_deep_merge_nested() -> None:
    out = _deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    assert out == {"a": {"b": 1, "c": 2}}  # nested recursion (line 504)


# --------------------------------------------------------------------------- #
# in_memory_kb_client list / upsert / batch / edges                          #
# --------------------------------------------------------------------------- #
def test_list_validation_and_filters() -> None:
    kb = InMemoryKBClient(time_fn=lambda: 1.0)
    with pytest.raises(KBValidationError):
        kb.list(scope_filter="nope")  # type: ignore[arg-type]  # line 258

    kb.upsert({"scope": {"model": "M"}, "kind": "technique", "slug": "slug-aaaa", "importance": 0.5})
    # kind filter mismatch -> filtered out (line 264).
    res = kb.list(scope_filter={"model": "m"}, kind="pitfall")
    assert res["count"] == 0
    res2 = kb.list(scope_filter={"model": "m"}, kind="technique")
    assert res2["count"] == 1


def test_upsert_normalization_warning_and_merge() -> None:
    kb = InMemoryKBClient(time_fn=lambda: 1.0)
    res = kb.upsert(
        {"scope": {"model": "big-model"}, "kind": "technique", "slug": "slug-aaaa", "importance": 0.5}
    )
    assert res["created"] is True

    # Second upsert merges edges + raises importance (349-352, 357).
    res2 = kb.upsert(
        {
            "scope": {"model": "big-model"},
            "kind": "technique",
            "slug": "slug-aaaa",
            "importance": 0.8,
            "edges": {"relates_to": ["kb_x"]},
            "summary": "updated",
        }
    )
    assert res2["created"] is False
    assert res2["row"]["importance"] == 0.8
    assert "kb_x" in res2["row"]["edges"]["relates_to"]

    # Lower importance -> protected.
    res3 = kb.upsert(
        {"scope": {"model": "big-model"}, "kind": "technique", "slug": "slug-aaaa", "importance": 0.1}
    )
    assert "importance_protected" in res3["warnings"]


def test_batch_insert_conflict_modes() -> None:
    kb = InMemoryKBClient(time_fn=lambda: 1.0)
    with pytest.raises(KBValidationError):
        kb.batch_insert([], on_conflict="bogus")  # line 388

    item = {"scope": {"model": "m"}, "kind": "technique", "slug": "slug-aaaa", "importance": 0.5}
    kb.batch_insert([item], on_conflict="error")  # inserts (394-401)
    with pytest.raises(KBValidationError):
        kb.batch_insert([item], on_conflict="error")  # conflict now raises


def test_add_edges_branches() -> None:
    kb = InMemoryKBClient(time_fn=lambda: 1.0)
    with pytest.raises(KBValidationError):
        kb.add_edges([{"kind": "relates_to"}])  # missing from/to -> line 435

    # Source row missing -> skipped (439-440).
    out = kb.add_edges([{"kind": "relates_to", "from_id": "missing", "to_id": "x"}])
    assert out["mirror_skipped"][0]["reason"] == "src_missing"


# --------------------------------------------------------------------------- #
# request_models validators                                                   #
# --------------------------------------------------------------------------- #
def test_require_and_optional_validators() -> None:
    with pytest.raises(RequestValidationError):
        _require_str({"k": 5}, "k", where="t")  # non-string -> line 159
    with pytest.raises(RequestValidationError):
        _optional_str({"k": 5}, "k", where="t")  # present non-string -> line 181
    with pytest.raises(RequestValidationError):
        _optional_list({"k": 5}, "k", where="t")  # present non-list -> line 230
    assert _optional_str({}, "k", where="t") is None
    assert _optional_list({}, "k", where="t") == []


def test_parse_request_errors() -> None:
    with pytest.raises(RequestValidationError):
        parse_request("not a dict")  # type: ignore[arg-type]  # line 253

    # proposals[i] not an object -> line 281.
    with pytest.raises(RequestValidationError):
        parse_request(
            {
                "kind": "coordinator_inbox",
                "session_id": "s1",
                "raw_prompt": "do something",
                "proposals": ["not-an-object"],
            }
        )
