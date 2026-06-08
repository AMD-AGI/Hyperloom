# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Parse Coordinator-rendered prompts and inbox.jsonl into ReactorContext.

The Coordinator's ``_compose_prompt`` emits a deterministic two-section
text::

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
    # PR-B Fix 2: in-flight kernel-opt visibility. ``no_levers_found``
    # used to fire while a long batch was still running (stack_size=0 +
    # cumulative_gain=0 + elapsed>threshold), even though the LLM had
    # already dispatched kernel_opt and a KEEP was waiting to integrate.
    # Reading these two fields lets ``_no_levers_symptom`` short-circuit
    # when in-flight work is the explanation, not "no lever found".
    "kernel_opt_attempts_count",
    "has_keep_pending_integrate",
    # ``last_*`` lines are aggregated by ``_parse_shared_state`` into
    # ``SharedStateSnapshot.explore_started`` so ``no_levers_found``
    # can defer until at least one explore family has been attempted.
    # The canonical writers are ``last_explore`` / ``last_sweep``. The
    # values surface as either ``(none)`` (Coordinator sentinel for
    # "never") or a status= record.
    "last_explore",
    "last_sweep",
}

# Subset of ``_SCALAR_KEYS`` whose presence with a non-``(none)`` value
# flips :attr:`SharedStateSnapshot.explore_started` to True.
_EXPLORE_FAMILY_KEYS = frozenset({
    "last_explore",
    "last_sweep",
})

# Pattern for the Coordinator's Time-budget body line, e.g.:
#   ``elapsed=12.3min  remaining=347.7min  budget=360min  closing_phase=False``
# ``budget=0min`` is the "no wall-clock budget" sentinel and surfaces as
# :attr:`SharedStateSnapshot.budget_minutes = 0.0`.
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
    # Tick & stop_reason — used by progress-stagnation signals
    # (``gain_plateau`` / ``no_levers_found``). ``tick`` is the
    # Coordinator's monotonic per-pass counter (one increment per
    # reactor tick across the 4 agents), not the per-agent backend
    # turn counter. ``stop_reason`` is empty on a live run; we skip
    # the progress signals when it is non-empty (the session is
    # already winding down).
    tick: int = 0
    stop_reason: str = ""
    # ``optimization_stack_size`` is the number of validated entries
    # the Coordinator has accepted onto the stack. 0 + many ticks
    # elapsed is the signature of ``no_levers_found`` (remain_issue.md
    # #8). We parse the size from the rendered ``optimization_stack=``
    # line by counting commas; an exact list is too noisy for a signal.
    optimization_stack_size: int = 0
    # ``explore_started`` is True once any explore family (explore /
    # sweep) has produced at least one
    # ``last_*`` record (i.e. its rendered Coordinator line is no
    # longer ``(none)``). ``no_levers_found`` defers until this flag
    # flips so the cold-start window (sglang launch + baseline +
    # profile + turnaround on multi-node large-model) does not get
    # mistaken for an empty exploration.
    explore_started: bool = False
    # Time-budget fields populated from the Coordinator's
    # ``=== Time budget ===`` section. When the section is absent (legacy
    # prompt or no wall-clock deadline configured) the three fields stay
    # at ``0.0`` and ``closing_phase`` stays ``False`` so signals that
    # consume them can short-circuit safely.
    elapsed_minutes: float = 0.0
    remaining_minutes: float = 0.0
    budget_minutes: float = 0.0
    closing_phase: bool = False
    # PR-B Fix 2: kernel-opt in-flight visibility (see _SCALAR_KEYS).
    # ``kernel_opt_attempts_count`` is the number of unique kernel_ids
    # that have at least one recorded attempt (KEEP / REVERT / PARTIAL /
    # failure). Non-zero means "the LLM has run kernel_opt at least once
    # this session", which is a strong reason NOT to claim
    # ``no_levers_found``. ``has_keep_pending_integrate`` is True when
    # the multi-KEEP integrate queue still has work queued; firing
    # ``no_levers`` then is also wrong because the next integrate is
    # imminent.
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
            # ``to_prompt_summary`` renders this as ``20.5% (stack_len_at_validation=...,...)``.
            # ``_split_double_space`` already trimmed the trailing parens because the
            # parens are joined by single spaces, not double; strip the
            # ``%`` and any trailing parenthetical inline.
            head_clean = head.rstrip("%")
            # ``20.5%`` is the common shape; ``20.5% (stack_len_at_validation=2, ts=2026-...)``
            # is the shape when validation has fired. Take the leading number.
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
            # Any non-``(none)`` value (e.g. ``status=succeeded ...``)
            # flips ``explore_started`` to True. The parse stays
            # idempotent across the keys: once any of them sets the
            # flag, later ``(none)`` lines must not clear it.
            if head and head != "(none)":
                snapshot.explore_started = True
    return snapshot


def _count_optimization_stack(head: str) -> int:
    """Decode the size of the rendered ``optimization_stack`` value.

    ``SharedState._format_optimization_stack`` emits one of:

    * ``"(none)"`` when the stack is empty
    * a Python list repr (e.g. ``['baseline:v1', 'integrate:v2']``) when
      ``f"{parts}"`` formats a non-empty list inside the f-string

    Both shapes survive ``_split_double_space`` because there are no
    double spaces in either, so we only need to handle them here.

    Args:
        head (str): The trimmed rendered ``optimization_stack`` value.

    Returns:
        int: The number of entries on the stack; ``0`` for the empty
        ``"(none)"`` sentinel.
    """
    if not head or head == "(none)":
        return 0
    # Try Python literal first — covers the list-repr shape exactly.
    try:
        value = ast.literal_eval(head)
    except (SyntaxError, ValueError):
        # Fallback: comma-joined string (defensive against future
        # format drift).
        return len([part for part in head.split(",") if part.strip()])
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        return 0 if value == "(none)" else 1
    return 0


def _parse_time_budget_into(snapshot: SharedStateSnapshot, body: str) -> None:
    """Decode the ``=== Time budget ===`` section in place onto ``snapshot``.

    The Coordinator emits exactly one body line below the header (see
    ``Coordinator._compose_prompt``). When the section is absent — older
    prompts, agents that don't opt in, or runs without a wall-clock
    budget — ``body`` is empty and ``snapshot`` keeps its defaults so
    BudgetMonitor / deadline_imminent signals short-circuit cleanly.

    Args:
        snapshot (SharedStateSnapshot): Snapshot mutated in place with the
            parsed time-budget fields.
        body (str): The joined body lines of the time-budget section; an
            empty string leaves ``snapshot`` unchanged.
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

    ``to_prompt_summary`` joins two scalars with two spaces in a few
    lines (``baseline_tput=...  baseline_acc=...``); the second key/value
    is not interesting to the robustness reactor, so we cut at the
    double-space boundary.

    Args:
        value (str): The raw scalar value, possibly containing a trailing
            double-space-separated neighbour.

    Returns:
        str: The leading scalar with the neighbour trimmed off.
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
