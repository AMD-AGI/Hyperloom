# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse Coordinator-rendered prompts into a ReactorContext.

The Coordinator's ``_compose_prompt`` emits blocks separated by ``=== X ===``
headers. For the robustness role the expected blocks are:

    === Phase ===
    phase     : FRAMEWORK_AGENT
    ...

    === Shared session state ===
    tick=...
    ...

    === Time budget ===
    elapsed=...min  remaining=...min  budget=...min  closing_phase=False

    === Phase budget telemetry ===
      PRELUDE: elapsed=123s cap=456s used=27%
      FRAMEWORK_AGENT: elapsed=789s cap=unlimited used=0%

    === Conversation progress ===
    ticks_without_progress=3 threshold=12 severity=ok last_progress_tick=45

    === Inbox for <agent> [(newest last)] ===
    seq=<int> msg_id=<hex> from=<agent> topic=<topic> payload={'k': 'v', ...}

Parse failures surface as an empty :class:`ReactorContext` rather than
raising, keeping the reactor's heartbeat fallback alive.
"""

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


# Anchored to the two-space row prefix; ``\S+`` topic guards against payloads
# whose dict repr contains a literal ``topic=``. ``msg_id`` is absent on
# messages carrying none, and the fields after ``topic`` are per-topic
# (``_format_inbox_event``), so the remainder is captured as a tail.
_INBOX_LINE_RE = re.compile(
    r"^\s+seq=(?P<seq>\d+)\s+(?:msg_id=(?P<msg_id>\S+)\s+)?from=(?P<from_agent>\S+)\s+"
    r"topic=(?P<topic>\S+)\s*(?P<tail>.*)$"
)

# One ``key=<python literal>`` pair of a tail. Quoted values are matched whole so
# a ``k=v`` inside a rendered ``error=``/``notes=`` string cannot split it.
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
    "last_sweep",
}

# Subset of ``_SCALAR_KEYS`` whose presence with a non-``(none)`` value
# flips :attr:`SharedStateSnapshot.explore_started` to True.
_EXPLORE_FAMILY_KEYS = frozenset(
    {
        "last_explore",
        "last_sweep",
    }
)

# Coordinator Time-budget body line; ``budget=0min`` is the "no wall-clock budget" sentinel.
_TIME_BUDGET_LINE_RE = re.compile(
    r"^\s*elapsed=(?P<elapsed>-?\d+(?:\.\d+)?)min\s+"
    r"remaining=(?P<remaining>-?\d+(?:\.\d+)?)min\s+"
    r"budget=(?P<budget>-?\d+(?:\.\d+)?)min\s+"
    r"closing_phase=(?P<closing>True|False)\s*$"
)


@dataclass
class PhaseBudgetRow:
    """One parsed row from the ``=== Phase budget telemetry ===`` block.

    Attributes:
        phase (str): Phase name in upper-case (e.g. ``"FRAMEWORK_AGENT"``).
        elapsed_sec (int): Seconds elapsed in this phase.
        cap_sec (int): Budget cap in seconds; ``-1`` when unlimited.
        used_pct (float): Fraction of budget consumed (0–100). Meaningful
            only when ``cap_sec >= 0``.
    """

    phase: str
    elapsed_sec: int
    cap_sec: int
    used_pct: float


@dataclass
class ConversationProgress:
    """Parsed ``=== Conversation progress ===`` block.

    Attributes:
        ticks_without_progress (int): Ticks since the last measurable
            advancement (new KEEP / stack growth / validated-gain / phase).
        threshold (int): Tick count above which severity is ``"high"``.
        severity (str): ``"ok"`` or ``"high"`` as emitted by the Coordinator.
        last_progress_tick (int): Session-wide tick index of the most recent
            progress event.
    """

    ticks_without_progress: int
    threshold: int
    severity: str
    last_progress_tick: int


@dataclass
class InboxItem:
    """One parsed inbox row from the Coordinator's rendered prompt.

    Attributes:
        seq (int): Monotonic per-agent sequence number of the message.
        msg_id (str): Hex message id assigned by the Coordinator.
        from_agent (str): Name of the agent that sent the message.
        topic (str): Message topic string.
        payload (dict[str, Any]): Decoded payload dict; empty when the
            payload was absent or could not be decoded.
    """

    seq: int
    msg_id: str
    from_agent: str
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedStateSnapshot:
    """Subset of the Coordinator SharedState the robustness reactor reads.

    Only the fields the reactor consumes are parsed; every other
    rendered line is ignored. All fields default to a neutral value so a
    parse miss degrades to "no signal" rather than raising.

    Attributes:
        model_name (str): Target model name, or ``""`` when unset.
        model_class (str): Target model class, or ``""`` when unset.
        baseline_tput (float): Baseline throughput reported by the
            Coordinator.
        cumulative_gain_validated (float): Cumulative validated gain
            percentage.
        crash_count (int): Number of crashes recorded this session.
        current_action (str): Currently dispatched action name, or ``""``
            when idle.
        tick (int): Coordinator's session-wide monotonic tick counter.
        stop_reason (str): Stop reason, or ``""`` on a live run.
        optimization_stack_size (int): Number of validated entries on the
            optimization stack.
        explore_started (bool): True once any explore family has produced
            at least one record.
        elapsed_minutes (float): Minutes elapsed against the wall-clock
            budget.
        remaining_minutes (float): Minutes remaining against the budget.
        budget_minutes (float): Total wall-clock budget in minutes; ``0.0``
            means no budget configured.
        closing_phase (bool): True when the Coordinator signals the
            closing phase.
        kernel_opt_attempts_count (int): Count of unique kernel task identities with at
            least one recorded kernel_opt attempt.
        has_keep_pending_integrate (bool): True when a multi-KEEP integrate
            queue still has work pending.
    """

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


@dataclass
class ReactorContext:
    """Per-tick input for :class:`Reactor`.

    Built by :func:`from_coordinator_prompt` from the rendered Coordinator
    prompt, which is the only transport.

    Attributes:
        tick_index (int): In-process tick counter.
        shared_state (SharedStateSnapshot): Parsed SharedState fields.
        inbox (list[InboxItem]): Parsed inbox messages for this tick.
        now_unix (float): Wall-clock timestamp for this tick.
        parse_warnings (list[str]): Non-fatal parse issues; logged once.
        phase (str): Current pipeline phase from ``=== Phase ===``; ``""``
            when the block is absent.
        phase_budget (list[PhaseBudgetRow]): Per-phase budget rows from
            ``=== Phase budget telemetry ===``; empty when absent.
        conversation_progress (ConversationProgress | None): Parsed progress
            signal from ``=== Conversation progress ===``; ``None`` when absent.
    """

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
    """Parse the text produced by ``Coordinator._compose_prompt``.

    Returns an empty / partially populated context on parse drift; the
    reactor's heartbeat fallback ensures liveness, and a single WARN is
    logged so deployments notice schema drift.

    Args:
        prompt (str): The rendered Coordinator prompt text. An empty
            string yields an empty context with a parse warning.
        tick_index (int): In-process tick index to stamp on the context.
        now_unix (float | None): Override for the context's wall-clock
            timestamp; ``None`` uses the current time.

    Returns:
        ReactorContext: The parsed context, possibly partial when the
        prompt drifts from the expected schema.
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


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------


def _split_sections(prompt: str) -> dict[str, str]:
    """Walk the prompt line-by-line and group lines by section.

    Returns a dict keyed by section name. Recognised keys:
    ``shared_state``, ``time_budget``, ``inbox``, ``kb``,
    ``phase``, ``phase_budget``, ``conversation_progress``.

    Args:
        prompt (str): The full rendered Coordinator prompt text.

    Returns:
        dict[str, str]: Mapping of recognised section name to the joined
        body lines for that section.
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


# ---------------------------------------------------------------------------
# Shared state parsing
# ---------------------------------------------------------------------------


def _parse_shared_state(body: str) -> SharedStateSnapshot:
    """Decode the ``=== Shared session state ===`` body into a snapshot.

    Only keys in :data:`_SCALAR_KEYS` (plus the ``model=`` line) are
    consumed; unknown or unparsable lines are skipped. Parsing stays
    idempotent so repeated explore-family lines never clear the
    ``explore_started`` flag once set.

    Args:
        body (str): The joined body lines of the shared-state section.

    Returns:
        SharedStateSnapshot: Populated snapshot with defaults for any
        field whose line was absent.
    """
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
        spec = _SCALAR_FIELD_TABLE.get(key)
        if spec is not None:
            attr, coerce = spec
            setattr(snapshot, attr, coerce(head))
        elif key in _EXPLORE_FAMILY_KEYS:
            # Any non-``(none)`` value flips ``explore_started`` True; never cleared once set.
            if head and head != "(none)":
                snapshot.explore_started = True
    return snapshot


def _count_optimization_stack(head: str) -> int:
    """Decode the size of the rendered ``optimization_stack`` value.

    ``SharedState._format_optimization_stack`` emits ``"(none)"`` (empty) or a
    Python list repr (e.g. ``['baseline:v1', 'integrate:v2']``).

    Args:
        head: The rendered ``optimization_stack`` head value.

    Returns:
        The number of stack entries (``0`` when empty/unparseable).
    """
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
    """Decode a ``cumulative_gain_validated`` head into a float percentage.

    Rendered as ``20.5%`` or ``20.5% (stack_len_at_validation=2, ts=...)``;
    take the leading number only.

    Args:
        head: The rendered ``cumulative_gain_validated`` head value.

    Returns:
        The leading percentage as a float (``0.0`` when unparseable).
    """
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
    """Extract the phase name from the ``phase : <NAME>`` line.

    Args:
        body (str): The phase block body text.

    Returns:
        str: Upper-case phase name, or ``""`` when absent.
    """
    for raw in body.splitlines():
        key, sep, value = raw.strip().partition(":")
        if sep and key.strip() == "phase":
            return value.strip().upper()
    return ""


def _parse_phase_budget(body: str) -> list[PhaseBudgetRow]:
    """Parse the phase budget block into typed rows.

    ``cap=unlimited`` maps to ``cap_sec=-1``; unmatched lines (including the
    ``(no phase history yet)`` sentinel) are skipped.

    Args:
        body (str): The phase budget block body text.

    Returns:
        list[PhaseBudgetRow]: One entry per recognised phase line.
    """
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
    """Parse the conversation progress block.

    Args:
        body (str): The conversation progress block body text.

    Returns:
        ConversationProgress | None: Parsed progress, or ``None`` when the
        block is absent or the body line does not match.
    """
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
    """Decode the ``=== Time budget ===`` section in place onto ``snapshot``.

    The Coordinator emits one body line below the header; an absent section
    leaves defaults so ``evaluate_budget_signals`` / ``deadline_imminent``
    short-circuit.

    Args:
        snapshot: Snapshot mutated in place with parsed budget fields.
        body: The time-budget section body text.
    """
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
    """Decode ``model=<name>  class=<klass>`` (double-space separator).

    Args:
        line (str): The rendered ``model=`` line.

    Returns:
        tuple[str, str]: The ``(model_name, model_class)`` pair, with the
        ``(unset)`` sentinel mapped to ``""``.
    """
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
    """Trim a SharedState scalar value at the next ``key=`` neighbour.

    ``to_prompt_summary`` joins two scalars with two spaces
    (``baseline_tput=...  baseline_acc=...``); cut at that boundary.

    Args:
        value: The raw scalar text possibly containing a trailing neighbour.

    Returns:
        The value trimmed at the first double-space boundary.
    """
    return value.split("  ", 1)[0].strip()


# ---------------------------------------------------------------------------
# Inbox parsing
# ---------------------------------------------------------------------------


def _parse_inbox(body: str) -> tuple[list[InboxItem], list[str]]:
    """Parse the inbox section body into items plus parse warnings.

    Lines that do not match :data:`_INBOX_LINE_RE` are skipped with a
    warning rather than raising. The ``(no new messages)`` sentinel
    short-circuits to an empty item list.

    Args:
        body (str): The joined body lines of the inbox section.

    Returns:
        tuple[list[InboxItem], list[str]]: The parsed inbox items and a
        list of human-readable parse warnings.
    """
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
    """Decode the per-topic field tail of an inbox line into a payload dict.

    ``payload=`` carries the whole dict and wins; the summary fields rendered
    before it are folded in underneath, so a topic that emits no ``payload=``
    (``delegated_result``) still yields ``kind`` / ``state`` / ``error``.

    Args:
        tail (str): Everything after ``topic=<topic>`` on the line.

    Returns:
        tuple[dict[str, Any], list[str]]: The payload dict and any decode
        warnings.
    """
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
    """Decode a rendered payload literal into a dict.

    The payload is expected to be a Python literal dict. Non-dict literals
    and unparsable text are wrapped so the caller never loses the raw
    value.

    Args:
        text (str): The raw payload text from an inbox line.

    Returns:
        tuple[dict[str, Any], str | None]: The decoded payload dict and an
        optional warning string when decoding fell back to a raw wrapper.
    """
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
