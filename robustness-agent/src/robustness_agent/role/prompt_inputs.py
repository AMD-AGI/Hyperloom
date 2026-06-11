# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Parse Coordinator-rendered prompts and inbox.jsonl into ReactorContext.

The Coordinator's ``_compose_prompt`` emits a
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

Or, when no new messages exist::

    === Inbox for <agent> ===
    (no new messages)

Consumes only a few SharedState fields plus the inbox tail; parse failures
are logged once and surface as an empty :class:`ReactorContext` rather than
raising, so the reactor's heartbeat fallback keeps the loop alive.
"""

from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger(__name__)


# Anchored to the two-space row prefix the Coordinator emits; ``\S+`` topic
# guards against payloads whose dict repr contains a literal ``topic=``.
_INBOX_LINE_RE = re.compile(
    r"^\s+seq=(?P<seq>\d+)\s+msg_id=(?P<msg_id>\S+)\s+from=(?P<from_agent>\S+)\s+"
    r"topic=(?P<topic>\S+)\s+payload=(?P<payload>.+)$"
)

_SHARED_HEADER = "=== Shared session state ==="
_INBOX_HEADER_PREFIX = "=== Inbox for "
_KB_HEADER_PREFIX = "=== Knowledge base hints"
_TIME_BUDGET_HEADER = "=== Time budget ==="

# SharedState lines we care about.
_SCALAR_KEYS = {
    "session_id",
    "baseline_tput",
    "cumulative_gain",
    "cumulative_gain_validated",
    "crash_count",
    "current_action",
    "tick",
    "stop_reason",
    "optimization_stack",
    # In-flight kernel-opt visibility lets ``_no_levers_symptom`` short-circuit when in-flight work explains stack_size=0.
    "kernel_opt_attempts_count",
    "has_keep_pending_integrate",
    # Aggregated into ``SharedStateSnapshot.explore_started``; ``(none)`` is the never-yet sentinel.
    "last_explore",
    "last_sweep",
}

# Subset of ``_SCALAR_KEYS`` whose presence with a non-``(none)`` value
# flips :attr:`SharedStateSnapshot.explore_started` to True.
_EXPLORE_FAMILY_KEYS = frozenset({
    "last_explore",
    "last_sweep",
})

# Coordinator Time-budget body line; ``budget=0min`` is the "no wall-clock budget" sentinel.
_TIME_BUDGET_LINE_RE = re.compile(
    r"^\s*elapsed=(?P<elapsed>-?\d+(?:\.\d+)?)min\s+"
    r"remaining=(?P<remaining>-?\d+(?:\.\d+)?)min\s+"
    r"budget=(?P<budget>-?\d+(?:\.\d+)?)min\s+"
    r"closing_phase=(?P<closing>True|False)\s*$"
)


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
        raw_payload (str): The raw, undecoded payload text as rendered.
    """

    seq: int
    msg_id: str
    from_agent: str
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    raw_payload: str = ""


@dataclass
class SharedStateSnapshot:
    """Subset of the Coordinator SharedState the robustness reactor reads.

    Only the fields the M1 reactor consumes are parsed; every other
    rendered line is ignored. All fields default to a neutral value so a
    parse miss degrades to "no signal" rather than raising.

    Attributes:
        session_id (str): Current session id, or ``""`` when unset.
        model_name (str): Target model name, or ``""`` when unset.
        model_class (str): Target model class, or ``""`` when unset.
        baseline_tput (float): Baseline throughput reported by the
            Coordinator.
        cumulative_gain (float): Cumulative (unvalidated) gain percentage.
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
        kernel_opt_attempts_count (int): Count of unique kernel ids with at
            least one recorded kernel_opt attempt.
        has_keep_pending_integrate (bool): True when a multi-KEEP integrate
            queue still has work pending.
    """

    session_id: str = ""
    model_name: str = ""
    model_class: str = ""
    baseline_tput: float = 0.0
    cumulative_gain: float = 0.0
    cumulative_gain_validated: float = 0.0
    crash_count: int = 0
    current_action: str = ""
    # ``tick`` is the Coordinator's monotonic per-pass counter; non-empty ``stop_reason`` means winding down so stagnation signals skip.
    tick: int = 0
    stop_reason: str = ""
    # Validated-entry count from ``optimization_stack=``; 0 + many ticks is the ``no_levers_found`` signature.
    optimization_stack_size: int = 0
    # True once any explore family (explore / sweep) emitted a non-``(none)`` record; defers ``no_levers_found`` past cold-start.
    explore_started: bool = False
    # Populated from the ``=== Time budget ===`` section; absent section leaves defaults so deadline signals short-circuit safely.
    elapsed_minutes: float = 0.0
    remaining_minutes: float = 0.0
    budget_minutes: float = 0.0
    closing_phase: bool = False
    # Non-zero ``kernel_opt_attempts_count`` or a pending integrate means do NOT claim ``no_levers_found``.
    kernel_opt_attempts_count: int = 0
    has_keep_pending_integrate: bool = False


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
        if key == "session_id":
            snapshot.session_id = "" if head == "(unset)" else head
        elif key == "baseline_tput":
            snapshot.baseline_tput = _coerce_float(head)
        elif key == "cumulative_gain":
            snapshot.cumulative_gain = _coerce_float(head.rstrip("%"))
        elif key == "cumulative_gain_validated":
            # Rendered as ``20.5%`` or ``20.5% (stack_len_at_validation=2, ts=...)``; take the leading number.
            head_clean = head.rstrip("%")
            for sep in (" ", "%"):
                head_clean = head_clean.split(sep, 1)[0]
            snapshot.cumulative_gain_validated = _coerce_float(head_clean)
        elif key == "crash_count":
            snapshot.crash_count = _coerce_int(head)
        elif key == "current_action":
            snapshot.current_action = "" if head == "(idle)" else head
        elif key == "tick":
            snapshot.tick = _coerce_int(head)
        elif key == "stop_reason":
            snapshot.stop_reason = "" if head == "(none)" else head
        elif key == "optimization_stack":
            snapshot.optimization_stack_size = _count_optimization_stack(head)
        elif key == "kernel_opt_attempts_count":
            snapshot.kernel_opt_attempts_count = _coerce_int(head)
        elif key == "has_keep_pending_integrate":
            snapshot.has_keep_pending_integrate = head.lower() == "true"
        elif key in _EXPLORE_FAMILY_KEYS:
            # Any non-``(none)`` value flips ``explore_started`` True; idempotent so a later ``(none)`` must not clear it.
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
        # Fallback: comma-joined string, defensive against format drift.
        return len([part for part in head.split(",") if part.strip()])
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        return 0 if value == "(none)" else 1
    return 0


def _parse_time_budget_into(snapshot: SharedStateSnapshot, body: str) -> None:
    """Decode the ``=== Time budget ===`` section in place onto ``snapshot``.

    The Coordinator emits one body line below the header; an absent section
    leaves defaults so BudgetMonitor / deadline_imminent signals
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
        snapshot.elapsed_minutes = _coerce_float(match.group("elapsed"))
        snapshot.remaining_minutes = _coerce_float(match.group("remaining"))
        snapshot.budget_minutes = _coerce_float(match.group("budget"))
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

    ``to_prompt_summary`` joins two scalars with two spaces
    (``baseline_tput=...  baseline_acc=...``); cut at that boundary.

    Args:
        value: The raw scalar text possibly containing a trailing neighbour.

    Returns:
        The value trimmed at the first double-space boundary.
    """
    return value.split("  ", 1)[0].strip()


def _coerce_float(value: str) -> float:
    """Parse a string into a float, defaulting to ``0.0`` on failure.

    Args:
        value (str): The string to parse.

    Returns:
        float: The parsed float, or ``0.0`` when ``value`` is not numeric.
    """
    try:
        return float(value)
    except ValueError:
        return 0.0


def _coerce_int(value: str) -> int:
    """Parse a string into an int, defaulting to ``0`` on failure.

    Args:
        value (str): The string to parse.

    Returns:
        int: The parsed integer, or ``0`` when ``value`` is not an int.
    """
    try:
        return int(value)
    except ValueError:
        return 0


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
