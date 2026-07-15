# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :mod:`runtime.session_memory` edge cases.

Covers input-validation guards, empty-log reads, malformed-cache handling,
corrupt-JSON detection, and the MergeResult serialiser.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.agents.critic.runtime.errors import SessionMemoryError
from hyperloom.agents.critic.runtime.session_memory import MergeResult, SessionMemory


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
    assert mr.merged["model"] == "m"


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


def test_append_decision_and_event_write_jsonl(sm):
    sm.append_decision("s1", {"verdict": "approve"})
    sm.append_event("s1", {"kind": "note", "text": "hi"})
    decision = json.loads(sm._decisions_path("s1").read_text("utf-8").splitlines()[0])
    event = json.loads(sm._events_path("s1").read_text("utf-8").splitlines()[0])
    assert decision["decision_review"]["verdict"] == "approve"
    assert event["kind"] == "note"


def test_append_decision_rejects_non_dict(sm):
    with pytest.raises(SessionMemoryError):
        sm.append_decision("s1", "nope")  # type: ignore[arg-type]


def test_append_event_rejects_non_dict(sm):
    with pytest.raises(SessionMemoryError):
        sm.append_event("s1", "nope")  # type: ignore[arg-type]


def test_get_cached_priors_absent_and_malformed(sm):
    assert sm.get_cached_priors("s1", "k") is None
    sm.put_cached_priors("s1", "good", [{"id": "x"}])
    assert sm.get_cached_priors("s1", "good") == [{"id": "x"}]

    path = sm._priors_cache_path("s1")
    cache = json.loads(path.read_text("utf-8"))
    cache["bad_entry"] = "not-a-dict"
    cache["bad_shape"] = {"ts": "not-a-number", "priors": "not-a-list"}
    path.write_text(json.dumps(cache), encoding="utf-8")
    assert sm.get_cached_priors("s1", "bad_entry") is None
    assert sm.get_cached_priors("s1", "bad_shape") is None


def test_get_cached_priors_expired(sm):
    sm.put_cached_priors("s1", "k", [{"id": "x"}])
    assert sm.get_cached_priors("s1", "k", now=1e18) is None


def test_put_cached_priors_rejects_non_list(sm):
    with pytest.raises(SessionMemoryError):
        sm.put_cached_priors("s1", "k", {"nope": 1})  # type: ignore[arg-type]


def test_reviewed_helpers_roundtrip(sm):
    sm.mark_reviewed("s1", "m1", "approve", decision_id="d1")
    assert sm.filter_unreviewed("s1", ["m1", "m2"]) == ["m2"]


def test_mark_reviewed_requires_msg_and_verdict(sm):
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "", "approve")
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "m1", "")


def test_reviewed_helpers_tolerate_non_dict_top_level(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert sm.filter_unreviewed("s1", ["m1"]) == ["m1"]


def test_mark_reviewed_resets_non_dict_data(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps(["corrupt"]), encoding="utf-8")
    sm.mark_reviewed("s1", "m1", "reject")
    assert json.loads(path.read_text("utf-8"))["m1"]["verdict"] == "reject"


def test_read_json_corrupt_raises(sm):
    sm._ensure_session_dir("s1")
    path = sm._context_path("s1")
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.load_context("s1")


def test_event_jsonl_records_are_parseable(sm):
    sm._ensure_session_dir("s1")
    path = sm._events_path("s1")
    path.write_text('{"kind": "a"}\n\n   \n', encoding="utf-8")
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    assert [e["kind"] for e in rows] == ["a"]

    path.write_text('{"kind": "a"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
