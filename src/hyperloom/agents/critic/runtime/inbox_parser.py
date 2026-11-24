# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse the proposals out of the Coordinator-style ``compose_prompt`` text."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .request_models import Proposal


# ``=== Section title ===`` marker; capture group 1 is the trimmed title.
_SECTION_RE = re.compile(r"^\s*===\s*(.+?)\s*===\s*$")

# Inbox row layout (matches ConversationCollaborator.compose_prompt).
_INBOX_ROW_RE = re.compile(
    r"^\s*seq=(?P<seq>\d+)\s+"
    r"msg_id=(?P<msg_id>[A-Za-z0-9_\-]+)\s+"
    r"from=(?P<from_agent>[A-Za-z0-9_\-]+)\s+"
    r"topic=(?P<topic>[A-Za-z0-9_\-]+)\s+"
    r"payload=(?P<payload>.*)$"
)

# Section titles we recognise.
_INBOX_PREFIX = "Inbox for "


# ---------------------------------------------------------------------------
@dataclass
class _InboxRow:
    """One parsed line from the inbox tail."""

    seq: int
    msg_id: str
    from_agent: str
    topic: str
    payload: dict[str, Any] | None


# Payload parser
def _try_parse_payload(raw: str) -> dict[str, Any] | None:
    """Return a dict if ``raw`` parses as either ``ast.literal_eval`` or JSON."""
    raw = raw.strip()
    if not raw:
        return None
    # ``str(dict)`` form (Python repr, single quotes) — most common.
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError):  # tolerate malformed payloads (return None)
        value = None
    if isinstance(value, dict):
        return value
    # JSON fallback (allows callers that emit JSON directly).
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    return None


# Section iterator
def _iter_sections(text: str) -> Iterable[tuple[str | None, list[str]]]:
    """Yield ``(title, lines)`` tuples in source order."""
    current_title: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            if current_title is not None or current_lines:
                yield current_title, current_lines
            current_title = m.group(1)
            current_lines = []
            continue
        current_lines.append(line)
    if current_title is not None or current_lines:
        yield current_title, current_lines


# ---------------------------------------------------------------------------
def parse_inbox_prompt(text: str) -> list[Proposal]:
    """Return the proposals in a Coordinator-style prompt's inbox sections."""
    proposals: list[Proposal] = []
    for title, lines in _iter_sections(text):
        if title is None:
            # Preamble before any section header — treat as ignorable.
            continue
        if title.startswith(_INBOX_PREFIX):
            for raw_line in lines:
                if not raw_line.strip() or raw_line.strip().startswith("("):
                    # Empty line or "(no new messages)" placeholder.
                    continue
                row = _parse_inbox_row(raw_line)
                if row is not None and row.topic == "proposal" and isinstance(row.payload, dict):
                    proposals.append(
                        Proposal.from_payload(
                            msg_id=row.msg_id, from_agent=row.from_agent, payload=dict(row.payload), seq=row.seq
                        )
                    )
    return proposals


def _parse_inbox_row(raw: str) -> _InboxRow | None:
    """Parse a single inbox line into an :class:`_InboxRow`."""
    m = _INBOX_ROW_RE.match(raw)
    if not m:
        return None
    return _InboxRow(
        seq=int(m.group("seq")),
        msg_id=m.group("msg_id"),
        from_agent=m.group("from_agent"),
        topic=m.group("topic"),
        payload=_try_parse_payload(m.group("payload")),
    )


__all__ = [
    "parse_inbox_prompt",
]
