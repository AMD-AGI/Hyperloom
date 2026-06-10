# Copyright Advanced Micro Devices, Inc. All rights reserved.

r"""Parse the Coordinator-style ``_compose_prompt`` text into structured fields.

The Coordinator () hands every reactor a prompt with
the following layout:

```
=== Shared session state ===
<key=value tokens, possibly across several lines>

=== Knowledge base hints ===
<free text, only for the orchestration role>

=== Inbox for <agent> (newest last) ===
  seq=<int> msg_id=<hex> from=<agent> topic=<topic> payload=<python-repr-dict>
  ...
```

The Critic mock backend also uses the exact regex
``r"^\s*seq=(\d+)\s+msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=proposal\s+payload=(.*)$"``
to spot proposals; we keep behaviour identical so a real Critic agent
remains drop-in compatible.

Intentionally tolerant: unknown sections go to ``extras``; shared-state is
best-effort ``key=value`` tokens; payloads try :func:`ast.literal_eval`
(the ``str(dict)`` form) then :func:`json.loads`. Unparseable rows are kept
for audit but excluded from ``proposals`` so the Critic never verdicts a
proposal it could not parse.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import InboxParseError
from .request_models import Proposal


# ``=== Section title ===`` marker. Title may include parentheses, spaces and
# words. Capture group 1 is the trimmed title.
_SECTION_RE = re.compile(r"^\s*===\s*(.+?)\s*===\s*$")

# Inbox row layout (matches Coordinator._compose_prompt). Hex32 is the legacy
# canonical msg_id, but we accept any non-empty hex run.
_INBOX_ROW_RE = re.compile(
    r"^\s*seq=(?P<seq>\d+)\s+"
    r"msg_id=(?P<msg_id>[A-Za-z0-9_\-]+)\s+"
    r"from=(?P<from_agent>[A-Za-z0-9_\-]+)\s+"
    r"topic=(?P<topic>[A-Za-z0-9_\-]+)\s+"
    r"payload=(?P<payload>.*)$"
)

# Section titles we recognise.
_SHARED_STATE_TITLE = "Shared session state"
_KB_HINTS_TITLE = "Knowledge base hints"
_INBOX_PREFIX = "Inbox for "


# ---------------------------------------------------------------------------
@dataclass
class InboxRow:
    """One parsed line from the inbox tail."""

    seq: int
    msg_id: str
    from_agent: str
    topic: str
    payload: dict[str, Any] | None
    raw_payload: str

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a plain JSON-serialisable dict.

        Returns:
            dict[str, Any]: All row fields keyed by name.
        """
        return {
            "seq": self.seq,
            "msg_id": self.msg_id,
            "from_agent": self.from_agent,
            "topic": self.topic,
            "payload": self.payload,
            "raw_payload": self.raw_payload,
        }


@dataclass
class ParsedPrompt:
    """Structured view of a Coordinator-style prompt.

    Attributes:
        agent_name: Inferred from the inbox section title (e.g. ``critic``).
        shared_state: ``key -> value`` tokens parsed from the shared state
            section. Values are kept as strings; the caller may coerce.
        inbox: All inbox rows, including non-proposal topics.
        proposals: Subset of ``inbox`` whose ``topic == "proposal"`` and
            whose payload parsed cleanly. Already converted to
            :class:`Proposal` for downstream convenience.
        kb_hints_text: Raw KB hints section (orchestration only); ``""`` if
            absent.
        extras: Any other ``=== Title ===`` section, mapped title -> raw text.
    """

    agent_name: str | None = None
    shared_state: dict[str, str] = field(default_factory=dict)
    inbox: list[InboxRow] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    kb_hints_text: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the parsed prompt as a plain JSON-serialisable dict.

        Returns:
            dict[str, Any]: All fields, with nested rows/proposals converted
            to dicts via their own ``to_dict``.
        """
        return {
            "agent_name": self.agent_name,
            "shared_state": dict(self.shared_state),
            "inbox": [r.to_dict() for r in self.inbox],
            "proposals": [p.to_dict() for p in self.proposals],
            "kb_hints_text": self.kb_hints_text,
            "extras": dict(self.extras),
        }


# ---------------------------------------------------------------------------
# Shared-state token parser
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=")


def _parse_shared_state(text: str) -> dict[str, str]:
    """Best-effort ``key=value`` token parser.

    The Coordinator's ``shared_state.to_prompt_summary()`` emits whitespace-
    separated tokens. We tolerate values that contain spaces or symbols by
    consuming until the next ``key=`` token start. Quoted values and
    multi-token values both round-trip back to a single string.

    Args:
        text (str): The raw shared-state section text (possibly multi-line).

    Returns:
        dict[str, str]: Parsed ``key -> value`` token pairs.
    """
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


# ---------------------------------------------------------------------------
# Payload parser
# ---------------------------------------------------------------------------
def _try_parse_payload(raw: str) -> dict[str, Any] | None:
    """Return a dict if ``raw`` parses as either ``ast.literal_eval`` or JSON.

    Prefers the Python ``repr`` form (single quotes) and falls back to JSON.

    Args:
        raw (str): The raw payload text from an inbox row.

    Returns:
        dict[str, Any] | None: The parsed dict, or ``None`` when ``raw`` is
        empty or does not parse to a dict.
    """
    raw = raw.strip()
    if not raw:
        return None
    # ``str(dict)`` form (Python repr, single quotes) — most common.
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
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


# ---------------------------------------------------------------------------
# Section iterator
# ---------------------------------------------------------------------------
def _iter_sections(text: str) -> Iterable[tuple[str | None, list[str]]]:
    """Yield ``(title, lines)`` tuples in source order.

    Lines before the first ``=== ===`` marker are emitted with title=``None``
    so the caller can either ignore them or surface them as preamble.

    Args:
        text (str): The full prompt text to split into sections.

    Yields:
        tuple[str | None, list[str]]: A section title (``None`` for preamble)
        paired with its raw lines.
    """
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


def _agent_from_inbox_title(title: str) -> str | None:
    """Extract ``critic`` from titles like ``Inbox for critic (newest last)``.

    Args:
        title (str): A section title to inspect.

    Returns:
        str | None: The agent name, or ``None`` when ``title`` is not an
        inbox section or yields no name.
    """
    if not title.startswith(_INBOX_PREFIX):
        return None
    rest = title[len(_INBOX_PREFIX) :].strip()
    # Strip a trailing parenthesised qualifier such as ``(newest last)``.
    paren = rest.find("(")
    if paren >= 0:
        rest = rest[:paren].strip()
    return rest or None


# ---------------------------------------------------------------------------
def parse_inbox_prompt(text: str) -> ParsedPrompt:
    """Parse a Coordinator-style prompt into :class:`ParsedPrompt`.

    Args:
        text (str): The raw Coordinator prompt text.

    Returns:
        ParsedPrompt: The structured view with shared state, inbox rows,
        proposals, KB hints, and any unknown sections preserved in
        ``extras``.

    Raises:
        InboxParseError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise InboxParseError(f"prompt must be str, got {type(text).__name__}")
    parsed = ParsedPrompt()
    for title, lines in _iter_sections(text):
        if title is None:
            # Preamble before any section header — treat as ignorable.
            continue
        if title == _SHARED_STATE_TITLE:
            parsed.shared_state = _parse_shared_state("\n".join(lines))
            continue
        if title == _KB_HINTS_TITLE:
            parsed.kb_hints_text = "\n".join(lines).strip()
            continue
        if title.startswith(_INBOX_PREFIX):
            parsed.agent_name = _agent_from_inbox_title(title)
            for raw_line in lines:
                if not raw_line.strip() or raw_line.strip().startswith("("):
                    # Empty line or "(no new messages)" placeholder.
                    continue
                row = _parse_inbox_row(raw_line)
                if row is None:
                    # Malformed line is a coordinator bug, not fatal — surface
                    # via ``extras['malformed_inbox']`` for audit.
                    bucket = parsed.extras.setdefault("malformed_inbox", "")
                    parsed.extras["malformed_inbox"] = (
                        bucket + ("\n" if bucket else "") + raw_line.strip()
                    )
                    continue
                parsed.inbox.append(row)
                if row.topic == "proposal" and isinstance(row.payload, dict):
                    parsed.proposals.append(_proposal_from_row(row))
            continue
        # Unknown section — preserve verbatim for audit.
        parsed.extras[title] = "\n".join(lines).strip()
    return parsed


def _parse_inbox_row(raw: str) -> InboxRow | None:
    """Parse a single inbox line into an :class:`InboxRow`.

    Args:
        raw (str): One raw inbox line.

    Returns:
        InboxRow | None: The parsed row, or ``None`` when ``raw`` does not
        match the expected inbox-row layout.
    """
    m = _INBOX_ROW_RE.match(raw)
    if not m:
        return None
    payload_text = m.group("payload").strip()
    payload = _try_parse_payload(payload_text)
    return InboxRow(
        seq=int(m.group("seq")),
        msg_id=m.group("msg_id"),
        from_agent=m.group("from_agent"),
        topic=m.group("topic"),
        payload=payload,
        raw_payload=payload_text,
    )


def _proposal_from_row(row: InboxRow) -> Proposal:
    """Build a :class:`Proposal` from a parsed proposal inbox row.

    Pulls ``action_name`` and ``predicted_gain_pct`` out of the payload as
    typed convenience fields.

    Args:
        row (InboxRow): A row whose ``topic == "proposal"``.

    Returns:
        Proposal: The structured proposal.
    """
    payload = dict(row.payload or {})
    action_name = payload.get("action_name") if isinstance(payload.get("action_name"), str) else None
    gain_raw = payload.get("predicted_gain_pct")
    if isinstance(gain_raw, (int, float)):
        predicted_gain_pct: float | None = float(gain_raw)
    else:
        predicted_gain_pct = None
    return Proposal(
        msg_id=row.msg_id,
        from_agent=row.from_agent,
        payload=payload,
        seq=row.seq,
        action_name=action_name,
        predicted_gain_pct=predicted_gain_pct,
    )


__all__ = [
    "InboxRow",
    "ParsedPrompt",
    "parse_inbox_prompt",
]
