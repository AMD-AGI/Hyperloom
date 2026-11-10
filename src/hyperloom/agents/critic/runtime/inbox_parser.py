# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse the Coordinator-style ``compose_prompt`` text into structured fields."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
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
_SHARED_STATE_TITLE = "Shared session state"
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


@dataclass
class ParsedPrompt:
    """The shared session state and the proposals a Coordinator-style prompt carries."""

    shared_state: dict[str, str] = field(default_factory=dict)
    proposals: list[Proposal] = field(default_factory=list)


# Shared-state token parser
_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=")


def _parse_shared_state(text: str) -> dict[str, str]:
    """Best-effort ``key=value`` token parser."""
    out: dict[str, str] = {}
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return out
    matches = list(_TOKEN_RE.finditer(cleaned))
    for i, m in enumerate(matches):
        key = m.group("key")
        value_start = m.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        value = cleaned[value_start:value_end].strip()
        # Trim trailing punctuation that is clearly a separator (comma, semicolon).
        while value.endswith((",", ";")):
            value = value[:-1].rstrip()
        out[key] = value
    return out


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
def parse_inbox_prompt(text: str) -> ParsedPrompt:
    """Parse a Coordinator-style prompt into :class:`ParsedPrompt`."""
    parsed = ParsedPrompt()
    for title, lines in _iter_sections(text):
        if title is None:
            # Preamble before any section header — treat as ignorable.
            continue
        if title == _SHARED_STATE_TITLE:
            parsed.shared_state = _parse_shared_state("\n".join(lines))
            continue
        if title.startswith(_INBOX_PREFIX):
            for raw_line in lines:
                if not raw_line.strip() or raw_line.strip().startswith("("):
                    # Empty line or "(no new messages)" placeholder.
                    continue
                row = _parse_inbox_row(raw_line)
                if row is not None and row.topic == "proposal" and isinstance(row.payload, dict):
                    parsed.proposals.append(
                        Proposal.from_payload(
                            msg_id=row.msg_id, from_agent=row.from_agent, payload=dict(row.payload), seq=row.seq
                        )
                    )
    return parsed


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
    "ParsedPrompt",
    "parse_inbox_prompt",
]
