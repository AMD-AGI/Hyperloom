# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`runtime.session_memory` edge cases."""

from __future__ import annotations

import json

import pytest

from hyperloom.agents.critic.runtime.errors import SessionMemoryError
from hyperloom.agents.critic.runtime.session_memory import SessionMemory


@pytest.fixture()
def sm(tmp_path):
    return SessionMemory(root=tmp_path / "sm")


def test_session_dir_rejects_invalid_ids(sm):
    with pytest.raises(SessionMemoryError):
        sm.session_dir("")
    with pytest.raises(SessionMemoryError):
        sm.session_dir(None)  # type: ignore[arg-type]
    with pytest.raises(SessionMemoryError):
        sm.session_dir("a/b")
    with pytest.raises(SessionMemoryError):
        sm.session_dir("..")


def test_reviewed_helpers_roundtrip(sm):
    sm.mark_reviewed("s1", "m1", "approve")
    assert sm.filter_unreviewed("s1", ["m1", "m2"]) == ["m2"]


def test_mark_reviewed_requires_msg_and_verdict(sm):
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "", "approve")
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "m1", "")


def test_reviewed_helpers_reject_non_dict_top_level(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.filter_unreviewed("s1", ["m1"])


def test_mark_reviewed_rejects_non_dict_data(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text(json.dumps(["corrupt"]), encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "m1", "reject")


def test_read_json_corrupt_raises(sm):
    sm._ensure_session_dir("s1")
    path = sm._reviewed_path("s1")
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionMemoryError):
        sm.mark_reviewed("s1", "m1", "approve")
