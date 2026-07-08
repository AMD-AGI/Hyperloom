# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :mod:`runtime.session_memory` edge cases.

Covers input-validation guards, empty-log reads, malformed-cache handling,
corrupt-JSON detection, and the MergeResult serialiser that the higher-level
DecisionReviewer tests do not exercise directly.
"""

from __future__ import annotations

import json

import pytest

from runtime.errors import SessionMemoryError
from runtime.session_memory import MergeResult, SessionMemory


@pytest.fixture()
def sm(tmp_path):
    return SessionMemory(root=tmp_path / "sm")


def test_merge_result_to_dict_is_a_copy():
    mr = MergeResult(
        merged={"model": "m"},
        explicit_keys=["model"],
        from_memory_keys=["framework"],
        missing_keys=["workload"],
    )
    d = mr.to_dict()
    assert d == {
        "merged": {"model": "m"},
        "explicit_keys": ["model"],
        "from_memory_keys": ["framework"],
        "missing_keys": ["workload"],
    }
    d["merged"]["model"] = "changed"
    assert mr.merged["model"] == "m"  # original untouched


def test_session_dir_rejects_invalid_ids(sm):
    with pytest.raises(SessionMemoryError):
        sm.session_dir("")
    with pytest.raises(SessionMemoryError):
        sm.session_dir(None)  # type: ignore[arg-type]
    with pytest.raises(SessionMemoryError):
        sm.session_dir("a/b")
    with pytest.raises(SessionMemoryError):
        sm.session_dir("..")


def test_save_context_rejects_non_dict(sm):
    with pytest.raises(SessionMemoryError):
        sm.save_context("s1", ["not", "dict"])  # type: ignore[arg-type]


def test_merge_context_rejects_non_dict(sm):
    with pytest.raises(SessionMemoryError):
        sm.merge_context("s1", ["nope"])  # type: ignore[arg-type]


def test_merge_context_fills_from_memory(sm):
    sm.save_context("s1", {"model": "m1", "framework": "sglang"})
    result = sm.merge_context("s1", {"workload": "decode"}, persist=False)
    assert result.merged["model"] == "m1"
    assert "framework" in result.from_memory_keys
    assert "workload" in result.explicit_keys


def test_list_decisions_and_events_empty(sm):
    assert sm.list_decisions("s_empty") == []
    assert sm.list_events("s_empty") == []


def test_append_decision_and_event_roundtrip(sm):
    sm.append_decision("s1", {"verdict": "approve"})
    sm.append_event("s1", {"kind": "note", "text": "hi"})
    decisions = sm.list_decisions("s1")
    events = sm.list_events("s1")
    assert decisions and decisions[0]["decision_review"]["verdict"] == "approve"
    assert events and events[0]["kind"] == "note"


def test_append_decision_rejects_non_dict(sm):
    with pytest.raises(SessionMemoryError):
        sm.append_decision("s1", "nope")  # type: ignore[arg-type]


def test_append_event_rejects_non_dict(sm):
    with pytest.raises(SessionMemoryError):
        sm.append_event("s1", "nope")  # type: ignore[arg-type]


def test_get_cached_priors_absent_and_malformed(sm):
    assert sm.get_cached_priors("s1", "k") is None  # no cache file yet
    sm.put_cached_priors("s1", "good", [{"id": "x"}])
    assert sm.get_cached_priors("s1", "good") == [{"id": "x"}]

    # Write a malformed entry directly and confirm it is rejected.
    path = sm._priors_cache_path("s1")
    cache = json.loads(path.read_text("utf-8"))
    cache["bad_entry"] = "not-a-dict"
    cache["bad_shape"] = {"ts": "not-a-number", "priors": "not-a-list"}
    path.write_text(json.dumps(cache), encoding="utf-8")
    assert sm.get_cached_priors("s1", "bad_entry") is None
    assert sm.get_cached_priors("s1", "bad_shape") is None


def test_get_cached_priors_expired(sm):
    sm.put_cached_priors("s1", "k", [{"id": "x"}])
    # Far-future now exceeds the TTL -> expired -> None.
    assert sm.get_cached_priors("s1", "k", now=1e18) is None


def test_put_cached_priors_rejects_non_list(sm):
    with pytest.raises(SessionMemoryError):
        sm.put_cached_priors("s1", "k", {"nope": 1})  # type: ignore[arg-type]


def test_reviewed_helpers_roundtrip(sm):
    assert sm.is_msg_already_reviewed("s1", "m1") is False
    assert sm.reviewed_verdict_for("s1", "m1") is None
    sm.mark_reviewed("s1", "m1", "approve", decision_id="d1")
    assert sm.is_msg_already_reviewed("s1", "m1") is True
    assert sm.reviewed_verdict_for("s1", "m1") == "approve"
    assert sm.filter_unreviewed("s1", ["m1", "m2"]) == ["m2"]


def test_mark_reviewed_requires_msg_and_verdict(sm):
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "", "approve")
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "m1", "")


def test_reviewed_verdict_for_non_dict_entry(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps({"m1": "just-a-string"}), encoding="utf-8")
    assert sm.reviewed_verdict_for("s1", "m1") is None


def test_reviewed_helpers_tolerate_non_dict_top_level(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert sm.is_msg_already_reviewed("s1", "m1") is False
    assert sm.reviewed_verdict_for("s1", "m1") is None
    assert sm.filter_unreviewed("s1", ["m1"]) == ["m1"]


def test_mark_reviewed_resets_non_dict_data(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps(["corrupt"]), encoding="utf-8")
    sm.mark_reviewed("s1", "m1", "reject")
    assert sm.reviewed_verdict_for("s1", "m1") == "reject"


def test_read_json_corrupt_raises(sm):
    sm._ensure_session_dir("s1")
    path = sm._context_path("s1")
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.load_context("s1")


def test_read_jsonl_skips_blank_and_flags_corrupt(sm):
    sm._ensure_session_dir("s1")
    path = sm._events_path("s1")
    path.write_text('{"kind": "a"}\n\n   \n', encoding="utf-8")
    assert [e["kind"] for e in sm.list_events("s1")] == ["a"]

    path.write_text('{"kind": "a"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.list_events("s1")
