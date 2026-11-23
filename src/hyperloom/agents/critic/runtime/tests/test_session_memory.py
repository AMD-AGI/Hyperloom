# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for :mod:`runtime.session_memory`, run against a temp directory."""

from __future__ import annotations

import json

import pytest

from hyperloom.agents.critic.runtime.errors import SessionMemoryError
from hyperloom.agents.critic.runtime.session_memory import (
    SessionMemory,
)


def test_session_dir_rejects_path_traversal(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    with pytest.raises(SessionMemoryError):
        sm.session_dir("../escape")
    with pytest.raises(SessionMemoryError):
        sm.session_dir("a/b")


def test_mark_reviewed_dedup(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.mark_reviewed("sess_1", "msg_a", "approve")
    reviewed = json.loads((sm.session_dir("sess_1") / "reviewed_msg_ids.json").read_text("utf-8"))
    assert reviewed["msg_a"]["verdict"] == "approve"


def test_filter_unreviewed_returns_only_new(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.mark_reviewed("sess_1", "m1", "approve")
    out = sm.filter_unreviewed("sess_1", ["m1", "m2", "m3"])
    assert out == ["m2", "m3"]


def test_corrupt_reviewed_file_raises(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm._ensure_session_dir("sess_1")
    sm._reviewed_path("sess_1").write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.filter_unreviewed("sess_1", ["m1"])


def test_atomic_write_does_not_leave_tmp_files(tmp_session_root):
    sm = SessionMemory(root=tmp_session_root)
    sm.mark_reviewed("sess_1", "m1", "approve")
    sd = sm.session_dir("sess_1")
    assert (sd / "reviewed_msg_ids.json").exists()
    assert not list(sd.glob("*.tmp"))
