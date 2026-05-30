"""Tests for :mod:`runtime.session_memory`.

We test against a temp directory so we don't touch the real
``CRITIC_SESSION_MEMORY_DIR``.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from hyperloom.agents.critic.runtime.errors import SessionMemoryError
from hyperloom.agents.critic.runtime.session_memory import (
    DEFAULT_PRIOR_CACHE_TTL_SECONDS,
    MergeResult,
    SessionMemory,
)


def test_session_dir_rejects_path_traversal(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    with pytest.raises(SessionMemoryError):
        sm.session_dir("../escape")
    with pytest.raises(SessionMemoryError):
        sm.session_dir("a/b")


def test_load_context_when_absent_returns_empty(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    assert sm.load_context("sess_1") == {}


def test_save_and_load_context_round_trip(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    ctx = {"model": "Qwen3-14B", "framework": "sglang"}
    sm.save_context("sess_1", ctx)
    assert sm.load_context("sess_1") == ctx


def test_merge_context_explicit_wins_and_persists(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.save_context("sess_1", {"model": "Qwen3-14B", "framework": "vllm"})
    res = sm.merge_context("sess_1", {"framework": "sglang"})
    assert isinstance(res, MergeResult)
    assert res.merged["framework"] == "sglang"
    assert res.merged["model"] == "Qwen3-14B"
    assert "framework" in res.explicit_keys
    assert "model" in res.from_memory_keys
    # Persisted
    assert sm.load_context("sess_1")["framework"] == "sglang"


def test_merge_context_unknown_treated_as_missing(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.save_context("sess_1", {"model": "Qwen3-14B"})
    res = sm.merge_context("sess_1", {"model": "unknown", "precision": ""})
    assert res.merged["model"] == "Qwen3-14B"
    assert "model" in res.from_memory_keys
    assert "precision" in res.missing_keys


def test_merge_context_lists_missing_critical_keys(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    res = sm.merge_context("sess_1", {})
    assert "model" in res.missing_keys
    assert "framework" in res.missing_keys


def test_append_and_list_decisions(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.append_decision("sess_1", {"verdict": "approve"})
    sm.append_decision("sess_1", {"verdict": "reject"})
    out = sm.list_decisions("sess_1")
    assert len(out) == 2
    assert out[0]["decision_review"]["verdict"] == "approve"
    assert out[1]["decision_review"]["verdict"] == "reject"


def test_append_decision_rejects_non_dict(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    with pytest.raises(SessionMemoryError):
        sm.append_decision("sess_1", "not a dict")  # type: ignore[arg-type]


def test_events_jsonl_roundtrip(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.append_event("sess_1", {"kind": "kb_cache_miss"})
    sm.append_event("sess_1", {"kind": "kb_write_ok", "id": "kb_xxx"})
    events = sm.list_events("sess_1")
    assert [e["kind"] for e in events] == ["kb_cache_miss", "kb_write_ok"]


def test_priors_cache_hit_and_miss(tmp_session_root, monkeypatch):
    sm = SessionMemory(root=tmp_session_root)
    sm.put_cached_priors("sess_1", "scope-x|topic-y", [{"id": "kb_a"}])
    assert sm.get_cached_priors("sess_1", "scope-x|topic-y") == [{"id": "kb_a"}]
    assert sm.get_cached_priors("sess_1", "scope-x|topic-z") is None


def test_priors_cache_expires_after_ttl(tmp_session_root):
    os_environ_backup = dict(os.environ)
    try:
        os.environ["CRITIC_PRIOR_CACHE_TTL_SECONDS"] = "0.001"
        sm = SessionMemory(root=tmp_session_root)
        sm.put_cached_priors("sess_1", "k", [{"id": "kb_a"}])
        time.sleep(0.01)
        assert sm.get_cached_priors("sess_1", "k") is None
    finally:
        os.environ.clear()
        os.environ.update(os_environ_backup)


def test_default_ttl_constant_exposed():
    assert DEFAULT_PRIOR_CACHE_TTL_SECONDS == 3600


def test_mark_reviewed_dedup(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.mark_reviewed("sess_1", "msg_a", "approve", decision_id="dec_1")
    assert sm.is_msg_already_reviewed("sess_1", "msg_a") is True
    assert sm.is_msg_already_reviewed("sess_1", "msg_b") is False
    assert sm.reviewed_verdict_for("sess_1", "msg_a") == "approve"


def test_filter_unreviewed_returns_only_new(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.mark_reviewed("sess_1", "m1", "approve")
    out = sm.filter_unreviewed("sess_1", ["m1", "m2", "m3"])
    assert out == ["m2", "m3"]


def test_corrupt_context_file_raises(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm._ensure_session_dir("sess_1")
    sm._context_path("sess_1").write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.load_context("sess_1")


def test_atomic_write_does_not_leave_tmp_files(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.save_context("sess_1", {"a": 1})
    sd = sm.session_dir("sess_1")
    assert (sd / "context.json").exists()
    assert not list(sd.glob("*.tmp"))


def test_jsonl_records_are_well_formed_lines(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.append_event("sess_1", {"kind": "x"})
    sm.append_event("sess_1", {"kind": "y"})
    raw = (sm.session_dir("sess_1") / "events.jsonl").read_text("utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        obj = json.loads(ln)
        assert "ts" in obj and "kind" in obj
