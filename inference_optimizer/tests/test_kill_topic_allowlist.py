# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression guard for Issue 1: the Robustness KILL_TASK handler broadcasts a
bus message on topic ``"kill"``, which must be in ``TOPIC_ALLOWLIST`` or
``append_and_seq`` raises ``unknown topic`` and crashes the kill path.

Also guards against future recurrences by asserting every literal topic emitted
by the Coordinator (``Message.new(..., "<topic>", ...)``) is allow-listed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.message_bus import (
    Message,
    MessageBus,
    TOPIC_ALLOWLIST,
)
from inference_optimizer.storage import SqliteConnection


_COORDINATOR_SRC = (
    Path(__file__).resolve().parents[1]
    / "orchestrator" / "coordinator.py"
)


@pytest.fixture
def db(tmp_path) -> SqliteConnection:
    sc = SqliteConnection(tmp_path / "kill_topic.db")
    yield sc
    sc.close()


def test_kill_topic_is_allowlisted():
    assert "kill" in TOPIC_ALLOWLIST


@pytest.mark.asyncio
async def test_kill_message_appends_without_raising(db):
    bus = MessageBus(db)
    msg = Message.new(
        "robustness", "*", "kill",
        {"task_id": "t-123", "reason": "stale specialist reap"},
    )
    seq = await bus.append_and_seq(msg)
    assert isinstance(seq, int) and seq > 0
    persisted = await bus.tail(topic="kill", n=10)
    assert any(m.payload.get("task_id") == "t-123" for m in persisted)


def test_all_coordinator_emitted_topics_are_allowlisted():
    """Every literal topic in ``Message.new(src, dst, "<topic>", ...)`` inside
    coordinator.py must be in TOPIC_ALLOWLIST (prevents future Issue-1 dupes).
    """
    tree = ast.parse(_COORDINATOR_SRC.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match ``Message.new(...)`` calls.
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "new"
            and isinstance(func.value, ast.Name)
            and func.value.id == "Message"
        ):
            continue
        # 3rd positional arg is the topic.
        if len(node.args) < 3:
            continue
        topic_arg = node.args[2]
        if isinstance(topic_arg, ast.Constant) and isinstance(
            topic_arg.value, str
        ):
            if topic_arg.value not in TOPIC_ALLOWLIST:
                offenders.append(topic_arg.value)
    assert not offenders, (
        f"coordinator emits un-allowlisted topic(s): {sorted(set(offenders))}"
    )
