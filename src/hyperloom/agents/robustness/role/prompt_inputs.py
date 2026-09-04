# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse Coordinator-rendered prompts into a ReactorContext."""

from __future__ import annotations

import ast
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.coerce import to_float, to_int


log = logging.getLogger(__name__)


# Anchored to the two-space row prefix; ``\S+`` topic guards against payloads whose dict repr contains a literal
# ``topic=``.
_INBOX_LINE_RE = re.compile(
    r"^\s+seq=(?P<seq>\d+)\s+(?:msg_id=(?P<msg_id>\S+)\s+)?from=(?P<from_agent>\S+)\s+"
    r"topic=(?P<topic>\S+)\s*(?P<tail>.*)$"
)

# One ``key=<python literal>`` pair of a tail.
_INBOX_FIELD_RE = re.compile(r"(?P<key>\w+)=(?P<value>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)")

_SHARED_HEADER = "=== Shared session state ==="
_INBOX_HEADER_PREFIX = "=== Inbox for "
_KB_HEADER_PREFIX = "=== Knowledge base hints"
_TIME_BUDGET_HEADER = "=== Time budget ==="
_PHASE_HEADER = "=== Phase ==="
_PHASE_BUDGET_HEADER = "=== Phase budget telemetry ==="
_CONVERSATION_PROGRESS_HEADER = "=== Conversation progress ==="

_PHASE_BUDGET_LINE_RE = re.compile(
    r"^\s+(?P<phase>[A-Z_]+):\s+"
    r"elapsed=(?P<elapsed>\d+)s\s+"
    r"(?:cap=(?P<cap>\d+)s|cap=unlimited)\s+"
    r"used=(?P<used>-?\d+(?:\.\d+)?)%\s*$"
)

_CONVERSATION_PROGRESS_LINE_RE = re.compile(
    r"^\s*ticks_without_progress=(?P<ticks>\d+)\s+"
    r"threshold=(?P<threshold>\d+)\s+"
    r"severity=(?P<severity>\S+)\s+"
    r"last_progress_tick=(?P<last>\d+)\s*$"
)

# SharedState lines we care about.
_SCALAR_KEYS = {
    "baseline_tput",
    "cumulative_gain_validated",
    "crash_count",
    "current_action",
    "tick",
    "macro_cycle",
    "stop_reason",
    "optimization_stack",
    # In-flight kernel-opt visibility lets ``_no_levers_symptom`` short-circuit.
    "kernel_opt_attempts_count",
    "has_keep_pending_integrate",
    # Aggregated into ``SharedStateSnapshot.explore_started``; ``(none)`` is the never-yet sentinel.
    "last_explore",
    "agent_last_active",
}

# Subset of ``_SCALAR_KEYS`` whose presence with a non-``(none)`` value flips
# :attr:`SharedStateSnapshot.explore_started` to True.
_EXPLORE_FAMILY_KEYS = frozenset({"last_explore"})

# Coordinator Time-budget body line; ``budget=0min`` is the "no wall-clock budget" sentinel.
_TIME_BUDGET_LINE_RE = re.compile(
    r"^\s*elapsed=(?P<elapsed>-?\d+(?:\.\d+)?)min\s+"
    r"remaining=(?P<remaining>-?\d+(?:\.\d+)?)min\s+"
    r"budget=(?P<budget>-?\d+(?:\.\d+)?)min\s+"
    r"closing_phase=(?P<closing>True|False)\s*$"
)


@dataclass
class PhaseBudgetRow:
    """One parsed row from the ``=== Phase budget telemetry ===`` block."""

    phase: str
    elapsed_sec: int
    cap_sec: int
    used_pct: float


@dataclass
class ConversationProgress:
    """Parsed ``=== Conversation progress ===`` block."""

    ticks_without_progress: int
    threshold: int
    severity: str
    last_progress_tick: int


@dataclass
class InboxItem:
    """One parsed inbox row from the Coordinator's rendered prompt."""

    seq: int
    msg_id: str
    from_agent: str
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedStateSnapshot:
    """Subset of the Coordinator SharedState the robustness reactor reads."""

    model_name: str = ""
    model_class: str = ""
    baseline_tput: float = 0.0
    cumulative_gain_validated: float = 0.0
    crash_count: int = 0
    current_action: str = ""
    tick: int = 0
    macro_cycle: int = 0
    stop_reason: str = ""
    optimization_stack_size: int = 0
    explore_started: bool = False
    elapsed_minutes: float = 0.0
    remaining_minutes: float = 0.0
    budget_minutes: float = 0.0
    closing_phase: bool = False
    kernel_opt_attempts_count: int = 0
    has_keep_pending_integrate: bool = False
    agent_last_active_unix: dict[str, float] = field(default_factory=dict)


@dataclass
class ReactorContext:
    """Per-tick input for :class:`Reactor`."""

    tick_index: int = 0
    shared_state: SharedStateSnapshot = field(default_factory=SharedStateSnapshot)
    inbox: list[InboxItem] = field(default_factory=list)
    now_unix: float = field(default_factory=time.time)
    parse_warnings: list[str] = field(default_factory=list)
    phase: str = ""
    phase_budget: list[PhaseBudgetRow] = field(default_factory=list)
    conversation_progress: ConversationProgress | None = None


def from_coordinator_prompt(
    prompt: str,
    *,
    tick_index: int = 0,
    now_unix: float | None = None,
) -> ReactorContext:
    """Parse the text produced by ``Coordinator._compose_prompt``."""
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
    _parse_time_budget_into(snapshot, sections.get("time_budget", ""))
    inbox, warnings = _parse_inbox(sections.get("inbox", ""))
    if not sections:
        warnings.append("no recognised sections in prompt")
    phase = _parse_phase(sections.get("phase", ""))
    phase_budget = _parse_phase_budget(sections.get("phase_budget", ""))
    conversation_progress = _parse_conversation_progress(sections.get("conversation_progress", ""))
    return ReactorContext(
        tick_index=tick_index,
        shared_state=snapshot,
        inbox=inbox,
        now_unix=now_unix,
        parse_warnings=warnings,
        phase=phase,
        phase_budget=phase_budget,
        conversation_progress=conversation_progress,
    )


# Section splitting


def _split_sections(prompt: str) -> dict[str, str]:
    """Walk the prompt line-by-line and group lines by section."""
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
        if stripped == _TIME_BUDGET_HEADER:
            current = "time_budget"
            sections.setdefault(current, [])
            continue
        if stripped.startswith(_KB_HEADER_PREFIX):
            current = "kb"
            sections.setdefault(current, [])
            continue
        if stripped == _PHASE_HEADER:
            current = "phase"
            sections.setdefault(current, [])
            continue
        if stripped == _PHASE_BUDGET_HEADER:
            current = "phase_budget"
            sections.setdefault(current, [])
            continue
        if stripped == _CONVERSATION_PROGRESS_HEADER:
            current = "conversation_progress"
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# Shared state parsing


def _parse_shared_state(body: str) -> SharedStateSnapshot:
    """Decode the ``=== Shared session state ===`` body into a snapshot."""
    import time as _time

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
        if key == "agent_last_active":
            snapshot.agent_last_active_unix = _parse_agent_last_active(head, now_unix=_time.time())
            continue
        spec = _SCALAR_FIELD_TABLE.get(key)
        if spec is not None:
            attr, coerce = spec
            setattr(snapshot, attr, coerce(head))
        elif key in _EXPLORE_FAMILY_KEYS:
            # Any non-``(none)`` value flips ``explore_started`` True; never cleared once set.
            if head and head != "(none)":
                snapshot.explore_started = True
    return snapshot


def _parse_agent_last_active(text: str, *, now_unix: float) -> dict[str, float]:
    """Parse the rendered ``agent_last_active`` value back into Unix timestamps.

    Rendered as ``orchestration=2s ago, critic=45s ago`` (or ``(none)``).

    Args:
        text (str): The rendered agent_last_active value.
        now_unix (float): Current time used to reconstruct timestamps.

    Returns:
        dict[str, float]: Mapping of agent name to its estimated Unix timestamp.
    """
    if not text or text == "(none)":
        return {}
    result: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        agent, sep, rest = part.partition("=")
        if not sep:
            continue
        agent = agent.strip()
        rest = rest.strip()
        # rest looks like "2s ago"
        age_str = rest.rstrip(" ago").strip().rstrip("s")
        try:
            age_s = float(age_str)
            result[agent] = now_unix - age_s
        except (ValueError, TypeError):
            continue
    return result


def _count_optimization_stack(head: str) -> int:
    """Decode the size of the rendered ``optimization_stack`` value."""
    if not head or head == "(none)":
        return 0
    try:
        value = ast.literal_eval(head)
    except (SyntaxError, ValueError):
        # Fallback: comma-joined string.
        return len([part for part in head.split(",") if part.strip()])
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        return 0 if value == "(none)" else 1
    return 0


def _coerce_cumulative_gain_validated(head: str) -> float:
    """Decode a ``cumulative_gain_validated`` head into a float percentage."""
    head_clean = head.rstrip("%")
    for sep in (" ", "%"):
        head_clean = head_clean.split(sep, 1)[0]
    return to_float(head_clean, default=0.0)


#: ``rendered key -> (SharedStateSnapshot attr, head-string coercion)`` table
#: driving :func:`_parse_shared_state`. Replaces the per-key ``if/elif`` ladder
#: with a single ``setattr`` loop; ``optimization_stack`` is the one key whose
#: attr name differs from its rendered key. Explore-family keys are handled
#: separately because they set a shared flag idempotently rather than a 1:1 attr.
_SCALAR_FIELD_TABLE: dict[str, tuple[str, Callable[[str], Any]]] = {
    "baseline_tput": ("baseline_tput", lambda head: to_float(head, default=0.0)),
    "cumulative_gain_validated": ("cumulative_gain_validated", _coerce_cumulative_gain_validated),
    "crash_count": ("crash_count", lambda head: to_int(head, default=0)),
    "current_action": ("current_action", lambda head: "" if head == "(idle)" else head),
    "tick": ("tick", lambda head: to_int(head, default=0)),
    "macro_cycle": ("macro_cycle", lambda head: to_int(head, default=0)),
    "stop_reason": ("stop_reason", lambda head: "" if head == "(none)" else head),
    "optimization_stack": ("optimization_stack_size", _count_optimization_stack),
    "kernel_opt_attempts_count": ("kernel_opt_attempts_count", lambda head: to_int(head, default=0)),
    "has_keep_pending_integrate": ("has_keep_pending_integrate", lambda head: head.lower() == "true"),
}


def _parse_phase(body: str) -> str:
    """Extract the phase name from the ``phase : <NAME>`` line."""
    for raw in body.splitlines():
        key, sep, value = raw.strip().partition(":")
        if sep and key.strip() == "phase":
            return value.strip().upper()
    return ""


def _parse_phase_budget(body: str) -> list[PhaseBudgetRow]:
    """Parse the phase budget block into typed rows."""
    rows: list[PhaseBudgetRow] = []
    for raw in body.splitlines():
        match = _PHASE_BUDGET_LINE_RE.match(raw)
        if not match:
            continue
        cap_str = match.group("cap")
        cap_sec = int(cap_str) if cap_str is not None else -1
        rows.append(
            PhaseBudgetRow(
                phase=match.group("phase"),
                elapsed_sec=int(match.group("elapsed")),
                cap_sec=cap_sec,
                used_pct=to_float(match.group("used"), default=0.0),
            )
        )
    return rows


def _parse_conversation_progress(body: str) -> ConversationProgress | None:
    """Parse the conversation progress block."""
    for raw in body.splitlines():
        match = _CONVERSATION_PROGRESS_LINE_RE.match(raw)
        if match:
            return ConversationProgress(
                ticks_without_progress=int(match.group("ticks")),
                threshold=int(match.group("threshold")),
                severity=match.group("severity").lower(),
                last_progress_tick=int(match.group("last")),
            )
    return None


def _parse_time_budget_into(snapshot: SharedStateSnapshot, body: str) -> None:
    """Decode the ``=== Time budget ===`` section in place onto ``snapshot``."""
    if not body:
        return
    for raw in body.splitlines():
        match = _TIME_BUDGET_LINE_RE.match(raw)
        if not match:
            continue
        snapshot.elapsed_minutes = to_float(match.group("elapsed"), default=0.0)
        snapshot.remaining_minutes = to_float(match.group("remaining"), default=0.0)
        snapshot.budget_minutes = to_float(match.group("budget"), default=0.0)
        snapshot.closing_phase = match.group("closing") == "True"
        return


def _parse_model_line(line: str) -> tuple[str, str]:
    """Decode ``model=<name> class=<klass>`` (double-space separator)."""
    body = line[len("model=") :]
    name, _, rest = body.partition("  class=")
    name = name.strip()
    klass = rest.strip()
    if name == "(unset)":
        name = ""
    if klass == "(unset)":
        klass = ""
    return name, klass


def _split_double_space(value: str) -> str:
    """Trim a SharedState scalar value at the next ``key=`` neighbour."""
    return value.split("  ", 1)[0].strip()


# Inbox parsing


def _parse_inbox(body: str) -> tuple[list[InboxItem], list[str]]:
    """Parse the inbox section body into items plus parse warnings."""
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
        payload, tail_warnings = _decode_tail(match.group("tail"))
        warnings.extend(tail_warnings)
        items.append(
            InboxItem(
                seq=seq,
                msg_id=match.group("msg_id") or "",
                from_agent=match.group("from_agent"),
                topic=match.group("topic"),
                payload=payload,
            )
        )
    return items, warnings


def _decode_tail(tail: str) -> tuple[dict[str, Any], list[str]]:
    """Decode the per-topic field tail of an inbox line into a payload dict."""
    head, sep, payload_text = tail.strip().partition("payload=")
    fields: dict[str, Any] = {}
    warnings: list[str] = []
    for match in _INBOX_FIELD_RE.finditer(head):
        key, raw = match.group("key"), match.group("value")
        try:
            fields[key] = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            fields[key] = raw
            warnings.append(f"field {key} not a python literal: {raw!r}")
    if not sep:
        return fields, warnings
    payload, payload_warn = _decode_payload(payload_text)
    if payload_warn:
        warnings.append(payload_warn)
    return {**fields, **payload}, warnings


def _decode_payload(text: str) -> tuple[dict[str, Any], str | None]:
    """Decode a rendered payload literal into a dict."""
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
