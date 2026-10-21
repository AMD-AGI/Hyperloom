# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delivering an admitted ``SEND_MESSAGE``: one bus message per recipient, and a specialist's copy in its inbox."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.protocol.intent import Intent
from hyperloom.inference_optimizer.session.session_paths import runs_dir

from ..bus.message_bus import TOPIC_ALLOWLIST, Message, MessageBus
from ..policy.gate import SPECIALIST_FROM_AGENT_PREFIX

log = logging.getLogger(__name__)

__all__ = ["deliver_specialist_inbox", "publish_send_message", "recipients"]


def recipients(to: Any) -> list[str]:
    """The ``to_agent`` of each message ``to`` asks for: ``*`` when absent, else each name once, in order."""
    if isinstance(to, list):
        return list(dict.fromkeys(str(name) for name in to)) or ["*"]
    return [str(to or "*")]


async def publish_send_message(bus: MessageBus, session_dir: Path, source: str, intent: Intent) -> None:
    """Publish a free-form message onto the bus, once per recipient, and into each specialist recipient's inbox."""
    topic = intent.payload.get("topic", "observation")
    if topic not in TOPIC_ALLOWLIST:
        # Soft-degrade unknown topic.
        topic = "observation"
    body = {k: v for k, v in intent.payload.items() if k != "to"}
    for to_agent in recipients(intent.payload.get("to")):
        await bus.append_and_seq(Message.new(source, to_agent, topic, dict(body)))
        if to_agent.startswith(SPECIALIST_FROM_AGENT_PREFIX):
            deliver_specialist_inbox(session_dir, source, to_agent, intent.payload)


def deliver_specialist_inbox(session_dir: Path, source: str, to_agent: str, payload: dict[str, Any]) -> None:
    """Append a message to a running specialist's workspace inbox."""
    task_id = to_agent[len(SPECIALIST_FROM_AGENT_PREFIX) :].strip()
    if not task_id:
        return
    try:
        workspace = runs_dir(session_dir, "specialist", task_id)
        workspace.mkdir(parents=True, exist_ok=True)
        # The prompt advertises the worktree when one exists; match it.
        worktree = workspace / "worktree"
        inbox = (worktree if worktree.is_dir() else workspace) / "inbox.json"
        existing: list[Any] = []
        if inbox.exists():
            loaded = json.loads(inbox.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        existing.append(
            {
                "from": source,
                "ts": now_iso(),
                "body": {k: v for k, v in payload.items() if k not in ("to", "topic")},
            }
        )
        # Keep the last 32 so the file stays prompt-sized.
        tmp = inbox.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing[-32:], indent=2), encoding="utf-8")
        tmp.replace(inbox)
    except Exception:
        log.exception("failed to deliver inbox message to %s", to_agent)
