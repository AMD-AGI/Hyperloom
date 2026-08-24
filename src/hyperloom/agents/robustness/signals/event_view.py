# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified, sorted, deduplicated view over coordinator events and inbox items.

Detectors call :func:`build_event_view` once per tick instead of merging the
two sources themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import InboxItem


@dataclass(frozen=True)
class EventRow:
    """One bus event, from the coordinator DB or the rendered inbox.

    Attributes:
        seq: ``events.seq`` primary key, or ``None`` when the source supplied
            none.
        agent: Name of the agent that emitted the event.
        topic: Message topic string.
        payload: Decoded payload dict.
        ts: Timestamp, or ``None`` when the source carried none.
    """

    seq: int | None
    agent: str
    topic: str
    payload: dict[str, Any]
    ts: Any


def build_event_view(
    inbox: list[InboxItem],
    coordinator_events: list[dict[str, Any]],
) -> list[EventRow]:
    """Merge both event sources into one chronological, deduplicated list.

    An inbox item is dropped when a coordinator row already carries its seq:
    both render the same ``events`` row, but the coordinator payload is the
    stored JSON while the inbox payload is re-parsed from prompt text.

    Rows with a seq sort ascending ahead of rows without one, which keep their
    input order so a caller supplying no seq still gets coordinator events
    before inbox items.

    Args:
        inbox: Parsed inbox items from the reactor context.
        coordinator_events: Coordinator event dicts in ascending seq order.

    Returns:
        The merged rows, oldest first.
    """
    rows: list[EventRow] = []
    seen: set[int] = set()

    for ev in coordinator_events:
        seq = ev.get("id")
        if isinstance(seq, int):
            seen.add(seq)
        else:
            seq = None
        payload = ev.get("payload")
        rows.append(
            EventRow(
                seq=seq,
                agent=ev.get("agent") or "",
                topic=ev.get("topic") or "",
                # _maybe_decode_json returns the raw column when it is not JSON.
                payload=payload if isinstance(payload, dict) else {},
                ts=ev.get("ts") or ev.get("timestamp"),
            )
        )

    for item in inbox:
        if item.seq in seen:
            continue
        rows.append(
            EventRow(
                seq=item.seq,
                agent=item.from_agent,
                topic=item.topic,
                payload=item.payload,
                ts=item.payload.get("ts"),
            )
        )

    rows.sort(key=lambda r: (0, r.seq) if r.seq is not None else (1, 0))
    return rows


def family_of(payload: dict[str, Any]) -> str:
    """Resolve an event payload's action family, falling back to ``kind``.

    Args:
        payload: An event payload dict.

    Returns:
        The family, or ``""`` when neither key carries one.
    """
    for key in ("family", "kind"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


__all__ = [
    "EventRow",
    "build_event_view",
    "family_of",
]
