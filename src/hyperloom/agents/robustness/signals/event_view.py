# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified, sorted, deduplicated view over coordinator events and inbox items.

Consumers should call :func:`build_event_view` once per tick and pass the
resulting list to each detector.  The view is always in chronological
(ascending seq) order with cross-source duplicates removed; coord rows win
over inbox rows for the same seq because the coord payload is the raw
JSON from the DB, while the inbox payload is reconstructed from the
rendered prompt text and may be lossy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import InboxItem


@dataclass(frozen=True)
class EventRow:
    """One event from either the coordinator DB or the rendered inbox.

    Attributes:
        seq: Monotonic sequence number from the ``events`` table, or
            ``None`` when the row comes from a test fixture that does not
            carry one.
        agent: Name of the agent that emitted the event.
        topic: Message topic string.
        payload: Decoded payload dict.
        ts: Timestamp, or ``None`` when not available.
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
    """Build a chronological, deduplicated event view for one reactor tick.

    Merges *coordinator_events* (already in ascending seq order after the
    producer's reverse) and *inbox* items, deduplicates by seq (coord rows
    win), then sorts so rows with a known seq come first (ascending) and rows
    without a seq follow in their original input order (coord before inbox).

    Args:
        inbox: Parsed inbox items from the reactor context.
        coordinator_events: Coordinator event dicts, ascending seq order.

    Returns:
        A stable-sorted list of :class:`EventRow` ready for detector
        consumption.
    """
    seen_seq: set[int] = set()
    rows: list[EventRow] = []

    for ev in coordinator_events:
        seq = ev.get("id")
        if isinstance(seq, int):
            seen_seq.add(seq)
        payload = ev.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        rows.append(
            EventRow(
                seq=seq if isinstance(seq, int) else None,
                agent=str(ev.get("agent") or ""),
                topic=str(ev.get("topic") or ""),
                payload=payload,
                ts=ev.get("ts") or ev.get("timestamp"),
            )
        )

    for item in inbox:
        seq = item.seq if isinstance(item.seq, int) else None
        if seq is not None and seq in seen_seq:
            continue
        payload = item.payload if isinstance(item.payload, dict) else {}
        rows.append(
            EventRow(
                seq=seq,
                agent=str(item.from_agent or ""),
                topic=str(item.topic or ""),
                payload=payload,
                ts=payload.get("ts"),
            )
        )

    rows.sort(key=lambda r: (0, r.seq) if r.seq is not None else (1, 0))
    return rows


def family_of(payload: dict[str, Any]) -> str:
    """Infer the action family from an event payload.

    Checks ``family`` first, falls back to ``kind``.

    Args:
        payload: An event payload dict.

    Returns:
        The family string, or ``""`` when neither key is set.
    """
    family = payload.get("family")
    if isinstance(family, str) and family.strip():
        return family.strip()
    kind = payload.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    return ""


__all__ = [
    "EventRow",
    "build_event_view",
    "family_of",
]
