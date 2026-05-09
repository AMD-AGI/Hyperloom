"""Parse Coordinator-rendered prompts and inbox.jsonl into ReactorContext.

The Coordinator's ``_compose_prompt`` (DESIGN v0.6 §8.3) emits a
deterministic two-section text:

    === Shared session state ===
    session_id=...
    model=<name>  class=<klass>
    baseline_tput=...  baseline_acc=...
    ...
    crash_count=...
    current_action=...
    ...
    === Inbox for <agent> [(newest last)] ===
      seq=<int> msg_id=<hex> from=<agent> topic=<topic> payload={'k': 'v', ...}
      ...

Or, when no new messages exist:

    === Inbox for <agent> ===
    (no new messages)

The robustness reactor only consumes a handful of the SharedState
fields plus the inbox tail, so this module deliberately ignores most of
the prompt.  Parse failures are logged once and surface as an empty
:class:`ReactorContext` rather than raising — the reactor's heartbeat
fallback keeps the loop alive even if upstream renames a section.
"""

from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger(__name__)


# Regex ordering: anchored to the row prefix the Coordinator emits with two
# leading spaces; ``.+?`` for topic guards against payloads whose dict repr
# contains a literal ``topic=`` substring.
_INBOX_LINE_RE = re.compile(
    r"^\s+seq=(?P<seq>\d+)\s+msg_id=(?P<msg_id>\S+)\s+from=(?P<from_agent>\S+)\s+"
    r"topic=(?P<topic>\S+)\s+payload=(?P<payload>.+)$"
)

_SHARED_HEADER = "=== Shared session state ==="
_INBOX_HEADER_PREFIX = "=== Inbox for "
_KB_HEADER_PREFIX = "=== Knowledge base hints"

# SharedState lines we care about for M1.
_SCALAR_KEYS = {
    "session_id",
    "baseline_tput",
    "cumulative_gain",
    "crash_count",
    "current_action",
}


@dataclass
class InboxItem:
    seq: int
    msg_id: str
    from_agent: str
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    raw_payload: str = ""


@dataclass
class SharedStateSnapshot:
    session_id: str = ""
    model_name: str = ""
    model_class: str = ""
    baseline_tput: float = 0.0
    cumulative_gain: float = 0.0
    crash_count: int = 0
    current_action: str = ""


@dataclass
class ReactorContext:
    """Per-tick input for :class:`Reactor`.

    Built by :func:`from_coordinator_prompt` (SINGLE_PROC) or
    :func:`from_inbox_jsonl` (MULTI_CLI; M3).  The reactor does not need
    to know which transport produced the context.
    """

    tick_index: int = 0
    shared_state: SharedStateSnapshot = field(default_factory=SharedStateSnapshot)
    inbox: list[InboxItem] = field(default_factory=list)
    now_unix: float = field(default_factory=time.time)
    parse_warnings: list[str] = field(default_factory=list)


def from_coordinator_prompt(
    prompt: str,
    *,
    tick_index: int = 0,
    now_unix: float | None = None,
) -> ReactorContext:
    """Parse the text produced by ``Coordinator._compose_prompt``.

    Returns an empty / partially populated context on parse drift; the
    reactor's heartbeat fallback ensures liveness, and a single WARN is
    logged so deployments notice schema drift.
    """
    if now_unix is None:
        now_unix = time.time()
    if not prompt:
        return ReactorContext(
            tick_index=tick_index,
            now_unix=now_unix,
            parse_warnings=["empty prompt"],
        )

    sections = _split_sections(prompt)
    snapshot = _parse_shared_state(sections.get("shared_state", ""))
    inbox, warnings = _parse_inbox(sections.get("inbox", ""))
    if not sections:
        warnings.append("no recognised sections in prompt")
    return ReactorContext(
        tick_index=tick_index,
        shared_state=snapshot,
        inbox=inbox,
        now_unix=now_unix,
        parse_warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_sections(prompt: str) -> dict[str, str]:
    """Walk the prompt line-by-line and group lines by section.

    Returns a dict with keys ``shared_state`` / ``inbox``; KB hints and
    other sections are dropped because the robustness reactor does not
    consume them.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped == _SHARED_HEADER:
            current = "shared_state"
            sections.setdefault(current, [])
            continue
        if stripped.startswith(_INBOX_HEADER_PREFIX) and stripped.endswith("==="):
            current = "inbox"
            sections.setdefault(current, [])
            continue
        if stripped.startswith(_KB_HEADER_PREFIX):
            current = "kb"
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# ---------------------------------------------------------------------------
# Shared state parsing
# ---------------------------------------------------------------------------

def _parse_shared_state(body: str) -> SharedStateSnapshot:
    snapshot = SharedStateSnapshot()
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("model="):
            snapshot.model_name, snapshot.model_class = _parse_model_line(line)
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if key not in _SCALAR_KEYS:
            continue
        head = _split_double_space(value)
        if key == "session_id":
            snapshot.session_id = "" if head == "(unset)" else head
        elif key == "baseline_tput":
            snapshot.baseline_tput = _coerce_float(head)
        elif key == "cumulative_gain":
            snapshot.cumulative_gain = _coerce_float(head.rstrip("%"))
        elif key == "crash_count":
            snapshot.crash_count = _coerce_int(head)
        elif key == "current_action":
            snapshot.current_action = "" if head == "(idle)" else head
    return snapshot


def _parse_model_line(line: str) -> tuple[str, str]:
    """Decode ``model=<name>  class=<klass>`` (double-space separator)."""
    body = line[len("model="):]
    name, _, rest = body.partition("  class=")
    name = name.strip()
    klass = rest.strip()
    if name == "(unset)":
        name = ""
    if klass == "(unset)":
        klass = ""
    return name, klass


def _split_double_space(value: str) -> str:
    """Trim a SharedState scalar value at the next ``key=`` neighbour.

    ``to_prompt_summary`` joins two scalars with two spaces in a few
    lines (``baseline_tput=...  baseline_acc=...``); the second key/value
    is not interesting to the robustness reactor, so we cut at the
    double-space boundary.
    """
    return value.split("  ", 1)[0].strip()


def _coerce_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _coerce_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Inbox parsing
# ---------------------------------------------------------------------------

def _parse_inbox(body: str) -> tuple[list[InboxItem], list[str]]:
    items: list[InboxItem] = []
    warnings: list[str] = []
    for raw in body.splitlines():
        if not raw.strip():
            continue
        if raw.lstrip().startswith("(no new messages)"):
            return [], warnings
        match = _INBOX_LINE_RE.match(raw)
        if not match:
            warnings.append(f"unparsable inbox line: {raw!r}")
            log.warning("prompt_inputs: skipping unparsable inbox line: %r", raw)
            continue
        try:
            seq = int(match.group("seq"))
        except ValueError:
            warnings.append(f"non-integer seq in {raw!r}")
            continue
        payload_text = match.group("payload")
        payload, payload_warn = _decode_payload(payload_text)
        if payload_warn:
            warnings.append(payload_warn)
        items.append(
            InboxItem(
                seq=seq,
                msg_id=match.group("msg_id"),
                from_agent=match.group("from_agent"),
                topic=match.group("topic"),
                payload=payload,
                raw_payload=payload_text,
            )
        )
    return items, warnings


def _decode_payload(text: str) -> tuple[dict[str, Any], str | None]:
    text = text.rstrip()
    if not text:
        return {}, None
    try:
        decoded = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {"raw": text}, f"payload not a python literal: {text!r}"
    if isinstance(decoded, dict):
        return decoded, None
    return {"raw": text, "decoded_type": type(decoded).__name__}, None
