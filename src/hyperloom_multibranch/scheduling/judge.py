# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge: an LLM that, inside the scheduler's hard caps, decides the branches' K, what to open and the stops.

The scheduler hands the judge a :class:`JudgeView` (the roster, each branch's frontier progress, the remaining budget,
the integration branch's recorded value, how similar archived sessions went, whether a branch opened now would
still pay off, which running branches made much the same change, the events since its last call, how far each
branch lies behind the integration head, which branches lead it while their commits fail to merge, whether a profile
of the integration head runs, the GPU hours (tests' and leases' alike) and LLM tokens the session has used, and what
is known of each running branch's specialist and which explore tests each branch has queued). The judge answers with
one JSON
object, which :func:`parse_decision` reads and :func:`cap_decision` cuts to the caps and to the branches that are
still live. A failed call or an unreadable reply gives ``None``, and the scheduler keeps the status quo. The judge
never calls a peer; the scheduler applies its decision. The judge may neither stop nor lower the K of a held branch
(:func:`held_branches`): a specialist branch until its grace of explore tests measured (:func:`session_grace`) and a
repair branch for its protected period, unless its specialist gave up or its judging failed, and a branch with a
candidate waiting for its confirm, unless its judging failed. In a session that plays a scheduling playbook, the judge
reads :data:`~hyperloom_multibranch.scheduling.judge_prompt.PLAYBOOK_JUDGE_PROMPT`, where the stall evidence is the one
stop rule for a stall, and the playbook's safety envelope refuses the stop, though not a lower K, of a branch with a
near-bar or a stranded lead and of one drawn for probation (:func:`envelope_holds`). The scheduler finds the held branches again as it applies the decision, since a
candidate can start waiting while the judge answers, and :func:`hold_decision` refuses their orders and names them in
the decision.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.contracts.measurement import Purpose, Validity
from hyperloom_multibranch.contracts.lifecycle import SpecialistStatus
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.intent import Intent
from hyperloom_multibranch.contracts.blackboard.records import BranchProgress, PendingCandidate, TestRecord
from hyperloom_multibranch.contracts.toolbox import BuildStatus
from hyperloom_multibranch.blackboard.rsi.frontier import (
    NEAR_BAR_HOLD,
    PROBATION_HOLD,
    SETTLED_TESTS,
    STRANDED_HOLD,
    envelope_hold,
)
from hyperloom_multibranch.blackboard.rsi.playbook import PlaybookPlay
from hyperloom_multibranch.blackboard.rsi.stranded import StrandedLead
from hyperloom_multibranch.scheduling.branches import REPAIR_PROTECTED_SECONDS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.events import SessionEvent
from hyperloom_multibranch.scheduling.forks import PendingBranch
from hyperloom_multibranch.scheduling.judge_prompt import JUDGE_SYSTEM_PROMPT, PLAYBOOK_JUDGE_PROMPT, REPLY_FORMAT
from hyperloom_multibranch.contracts.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.roster import (
    BranchEntry,
    BranchKind,
    BranchOrigin,
    RosterCaps,
    specialist_open,
)
from hyperloom_multibranch.scheduling.similarity import SimilarBranch
from hyperloom_multibranch.scheduling.widening import WideningSignals

_KEY_OPEN = "open"
_KEY_OPEN_K = "open_k"
_KEY_FOCUS = "focus"
_KEY_BRANCHES = "branches"
_KEY_STOP = "stop"
_KEY_STOP_SESSION = "stop_session"
_KEY_BRANCH = "branch"
#: What the runtime knows of a specialist but the judge's prompt does not name: why an agent's run ended of itself is
#: the scheduler's stop reason, since the scheduler stops that branch.
_NOT_JUDGED = frozenset({"finished"})
_KEY_REASON = "reason"
#: Decimals the GPU hours in the judge's view are rounded to.
_GPU_HOURS_DIGITS = 3
#: Decimals the dollars in the judge's view are rounded to.
_USD_DIGITS = 2


class HoldReason(str, Enum):
    """Why the judge may neither stop a branch nor lower its K: a specialist branch in its grace, a repair branch in
    its protected period, a branch with a candidate waiting for its confirm, or a branch no judge steers, whose K the
    judge may not raise either. In a session that plays a playbook its safety envelope holds three more
    (:func:`envelope_holds`), named as :func:`~hyperloom_multibranch.blackboard.rsi.frontier.envelope_hold` names them: a
    probation draw, a near-bar lead and a stranded lead, whose stop the judge may not order but whose K it may lower
    (:data:`ENVELOPE_HOLDS`)."""

    GRACE = "grace"
    REPAIR = "repair"
    PENDING_CONFIRM = "pending_confirm"
    UNSTEERED = "unsteered"
    NEAR_BAR = NEAR_BAR_HOLD
    STRANDED = STRANDED_HOLD
    PROBATION = PROBATION_HOLD


#: The holds of the safety envelope: a stop of such a branch is refused, a lower K is not.
ENVELOPE_HOLDS = frozenset({HoldReason.NEAR_BAR, HoldReason.STRANDED, HoldReason.PROBATION})


@dataclass(frozen=True)
class JudgeView:
    """What the judge sees; ``opening`` is true only for the call that sizes the opening batch, which opens at least
    one branch unless a specialist branch is open already.

    ``free_slots`` is the roster's own count of branches that may still be opened; ``pending`` are the forks and
    repairs the fork desk queued for a slot, oldest first; ``history`` is the blackboard's
    digest of similar archived sessions, empty when the archive holds none. ``widening`` says whether a branch
    opened now would still pay off. ``similar_to`` maps a branch to the running branches whose own change is much the
    same as its own; a branch similar to none is not in it. ``events`` are the events detected since the last call,
    and ``behind_integration_pct`` maps a running specialist branch to how far its head lies behind the integration
    head, in percent of the root's value. ``stranded_leads`` are the branches whose head leads the integration head by
    more than the bar while commits of theirs failed to merge, the largest lead first. ``profile_running`` is whether a profile of the integration head runs.
    ``gpu_hours_used`` are the GPU hours of the session's tests that have a result plus the card time any lease holds
    outside them, an open one up to the reading: the root's baseline and a PRELUDE child's leased cards are among them,
    a test withdrawn before its result and a PRELUDE child that held no lease are not. ``explore_gpu_hours`` is where
    EXPLORE ends under ``--max-gpu-hours``, ``None`` without a limit, ``prelude_gpu_hours_upper_bound`` the PRELUDE
    child's wall time times the workload's TP where it held no lease, and 0 where it did. ``specialist_llm_usd`` is what the specialists' recorded LLM calls
    cost, ``scheduler_llm_usd`` what the judge's, the patch resolver's and the fix critic's recorded calls cost, and
    ``max_llm_usd`` where their sum ends EXPLORE under ``--max-llm-usd``, ``None`` without a limit. ``specialists``
    maps a running specialist branch to what the specialist runtime knows of its specialist; a branch whose
    specialist's state is not known is not in it. ``held`` maps each branch the judge may neither stop nor lower the
    K of to why, as :func:`held_branches` finds them when the view is built; it is what the prompt shows.
    ``queued`` maps a branch to its explore tests not measured yet, as :func:`queued_intents` finds them, and
    ``building`` to the native builds not ready yet that its tests in flight wait for; a branch with none is not in
    either. ``open_areas`` says where a new branch could open and ``playbook`` is the scheduling playbook the session
    plays, as the RSI renders them; both are empty for a session that reads no archive.
    """

    now: float
    explore_remaining_seconds: float
    session_remaining_seconds: float
    caps: RosterCaps
    roster: tuple[BranchEntry, ...]
    progress: tuple[BranchProgress, ...]
    integration_value: Mapping[str, float]
    roofline_ceiling: Mapping[str, float]
    metric: str
    opening: bool
    free_slots: int
    pending: tuple[PendingBranch, ...]
    history: str
    widening: WideningSignals
    similar_to: Mapping[str, tuple[SimilarBranch, ...]]
    events: tuple[SessionEvent, ...]
    behind_integration_pct: Mapping[str, float]
    stranded_leads: tuple[StrandedLead, ...]
    profile_running: bool
    gpu_hours_used: float
    explore_gpu_hours: float | None
    prelude_gpu_hours_upper_bound: float
    specialist_llm_usd: float
    max_llm_usd: float | None
    scheduler_llm_usd: float
    specialists: Mapping[str, SpecialistStatus]
    held: Mapping[str, HoldReason]
    queued: Mapping[str, tuple[QueuedIntent, ...]]
    building: Mapping[str, tuple[BuildStatus, ...]]
    open_areas: str = ""
    playbook: str = ""


@dataclass(frozen=True)
class QueuedIntent:
    """An explore test requested at ``requested_at`` and neither measured nor withdrawn yet; ``intent`` is the target and
    mechanism it tests, ``""`` when the blackboard holds no intent for it."""

    request_id: str
    intent: str
    requested_at: float


def queued_intents(records: Iterable[TestRecord], intents: Mapping[str, Intent]) -> dict[str, tuple[QueuedIntent, ...]]:
    """Each branch's explore tests in flight, in the order they were requested, read off the blackboard's ``records``
    and its ``intents`` by fingerprint; a branch with none is left out."""

    queued: dict[str, list[QueuedIntent]] = {}
    for record in records:
        if record.request.purpose is not Purpose.EXPLORE or not record.in_flight:
            continue
        intent = intents.get(record.request.intent_fingerprint or "")
        text = "" if intent is None else f"{intent.target}: {intent.mechanism}"
        queued.setdefault(record.request.branch, []).append(QueuedIntent(record.request_id, text, record.requested_at))
    return {branch: tuple(sorted(items, key=lambda item: item.requested_at)) for branch, items in queued.items()}


def _measured_with_verdict(record: TestRecord) -> bool:
    """An explore test that succeeded and was judged: what a branch's grace counts. A failed test, whether the candidate
    or the measurement failed, a result still waiting for its verdict and a withdrawn test show nothing of the
    branch's direction."""

    return (
        record.request.purpose is Purpose.EXPLORE
        and record.judged
        and record.result is not None
        and record.result.validity is Validity.SUCCEEDED
    )


#: The ``policy_versions`` key of the grace the holds read (:func:`session_grace`): two sessions' stops compare only
#: within one grace.
GRACE_KEY = "grace"


def session_grace(play: PlaybookPlay | None) -> int:
    """How many judged explore tests a specialist branch's grace lasts: the stop rule's of the scheduling playbook the
    session plays now, the champion's once a challenger fell back, where the new-branch floor ends too, and
    :data:`SETTLED_TESTS` in a session that plays none."""

    return play.stop_rule().grace_tests if play is not None else SETTLED_TESTS


@dataclass(frozen=True)
class HoldRules:
    """When the holds are read: at ``now`` on the session clock, a specialist branch's grace lasting ``grace_tests``
    of its judged explore tests (:func:`session_grace`)."""

    now: float
    grace_tests: int


def held_branches(
    roster: Iterable[BranchEntry],
    progress: Iterable[BranchProgress],
    records: Iterable[TestRecord],
    specialists: Mapping[str, SpecialistStatus],
    rules: HoldRules,
) -> dict[str, HoldReason]:
    """The live branches the judge may neither stop nor lower the K of, with why, as ``rules`` read them.

    A specialist branch is in its grace while fewer than the rules' grace of its explore tests succeeded and were
    judged, and its specialist neither gave up nor was stopped for a failed judging. A repair branch is held for :data:`REPAIR_PROTECTED_SECONDS` from its open, however its tests
    read: the fix it carries is worth the gain its blocked candidate measured, which takes longer to show than the
    few tests a new branch is scored on. :data:`REPAIR_BRANCH_SECONDS` still ends it, as do a give-up and a failed
    judging. A branch whose progress row holds a pending candidate is held until its confirm is judged, unless
    its judging failed: the confirm's verdict does not need the specialist. A branch whose origin the judge does not
    steer is held for that, or for its pending confirm, which the scheduler's own stops must still wait for."""

    measured = Counter(record.request.branch for record in records if _measured_with_verdict(record))
    waiting = {row.branch for row in progress if row.pending}
    held: dict[str, HoldReason] = {}
    for entry in roster:
        state = specialists.get(entry.branch)
        if entry.stop is not None or (state is not None and state.judge_failed):
            continue
        gave_up = state is not None and state.gave_up
        if not entry.origin.steered_by_judge:
            held[entry.branch] = HoldReason.PENDING_CONFIRM if entry.branch in waiting else HoldReason.UNSTEERED
        elif entry.kind is BranchKind.SPECIALIST and measured[entry.branch] < rules.grace_tests and not gave_up:
            held[entry.branch] = HoldReason.GRACE
        elif (
            entry.origin is BranchOrigin.REPAIR
            and not gave_up
            and rules.now - entry.opened_at < REPAIR_PROTECTED_SECONDS
        ):
            held[entry.branch] = HoldReason.REPAIR
        elif entry.branch in waiting:
            held[entry.branch] = HoldReason.PENDING_CONFIRM
    return held


def envelope_holds(
    scored: Iterable[BranchProgress], specialists: Mapping[str, SpecialistStatus]
) -> dict[str, HoldReason]:
    """The running branches the safety envelope holds, as the blackboard's :func:`envelope_hold` reads the stall evidence
    of the frontier ``scored`` rows: drawn for probation, or holding a near-bar or a stranded lead. A branch whose
    specialist gave up, or was stopped for a failed judging, has nobody to follow its lead and is held by none of
    them."""

    held: dict[str, HoldReason] = {}
    for row in scored:
        state = specialists.get(row.branch)
        if row.frontier is None or (state is not None and (state.gave_up or state.judge_failed)):
            continue
        hold = envelope_hold(row.frontier.components)
        if hold:
            held[row.branch] = HoldReason(hold)
    return held


@dataclass(frozen=True)
class KOrder:
    branch: str
    k: int


@dataclass(frozen=True)
class StopOrder:
    branch: str
    reason: str


@dataclass(frozen=True)
class HeldOrder:
    """A stop or a K the judge gave a held or unsteered branch, refused for ``reason``."""

    branch: str
    reason: HoldReason


@dataclass(frozen=True)
class JudgeDecision:
    """A non-empty ``session_stop`` is the reason the judge asks for a soft stop of the session.

    ``open_k`` is the K of the branches opened now; a K order cannot name them before they exist. ``focus`` is the op
    the judge names for each branch opened now, in the order they open; ``""`` and a branch past its end leave the
    branch to the default focus. ``held`` names the branches whose stop or lower K was refused, once each.
    """

    open_count: int = 0
    open_k: int = 1
    focus: tuple[str, ...] = ()
    k: tuple[KOrder, ...] = ()
    stop: tuple[StopOrder, ...] = ()
    session_stop: str = ""
    held: tuple[HeldOrder, ...] = ()


class _MalformedReply(ValueError):
    pass


def _hours(hours: float | None) -> float | None:
    return None if hours is None else round(hours, _GPU_HOURS_DIGITS)


def _headroom_pct(view: JudgeView) -> float | None:
    value = view.integration_value.get(view.metric)
    ceiling = view.roofline_ceiling.get(view.metric)
    if value is None or not ceiling:
        return None
    return (ceiling - value) / ceiling * 100.0


def _progress_row(progress: BranchProgress | None) -> dict[str, object]:
    if progress is None:
        return {}
    row: dict[str, object] = {
        "running": progress.running,
        "head_commit": progress.head_commit,
        "tests": progress.tests,
        "commits": progress.commits,
        "judged_since_commit": progress.judged_since_commit,
        "gpu_seconds": progress.gpu_seconds,
        "held_card_seconds": progress.held_card_seconds,
        "head_value": dict(progress.head_value),
        "pending_candidates": [_pending_candidate_row(item) for item in progress.pending],
    }
    if progress.frontier is not None:
        row["expected_gain_pct"] = progress.frontier.expected_gain_pct
        row["components"] = dict(progress.frontier.components)
        if progress.frontier.basis is not None:
            row["basis"] = progress.frontier.basis.value
    return row


def _pending_candidate_row(candidate: PendingCandidate) -> dict[str, object]:
    return {"request_id": candidate.request_id, "gain_over_head_pct": round(candidate.gain_over_head_pct, 2)}


def _branch_row(entry: BranchEntry, progress: BranchProgress | None, view: JudgeView) -> dict[str, object]:
    row: dict[str, object] = {
        _KEY_BRANCH: entry.branch,
        "kind": entry.kind.value,
        "status": entry.status.value,
        "origin": entry.origin.value,
        "from_branch": entry.from_branch,
        "start_commit": entry.start_commit,
        "k": entry.k,
        "similar_to": [asdict(item) for item in view.similar_to.get(entry.branch, ())],
        "queued_intents": [_queued_row(item, view.now) for item in view.queued.get(entry.branch, ())],
        "waiting_builds": [_build_row(build) for build in view.building.get(entry.branch, ())],
    }
    if entry.branch == INTEGRATION_BRANCH:
        row["profile_running"] = view.profile_running
    behind = view.behind_integration_pct.get(entry.branch)
    if behind is not None:
        row["behind_integration_pct"] = round(behind, 2)
    if entry.stop is not None:
        row["stopped"] = entry.stop.summary
    hold = view.held.get(entry.branch)
    if hold is not None:
        row["held"] = hold.value
    if not entry.origin.steered_by_judge:
        row["steered"] = False
    specialist = view.specialists.get(entry.branch)
    if specialist is not None:
        row["specialist"] = {name: value for name, value in asdict(specialist).items() if name not in _NOT_JUDGED}
    row.update(_progress_row(progress))
    return row


def _build_row(build: BuildStatus) -> dict[str, object]:
    return {"component": build.component, "eta_seconds": round(build.eta_seconds)}


def _queued_row(queued: QueuedIntent, now: float) -> dict[str, object]:
    return {
        "request_id": queued.request_id,
        "intent": queued.intent,
        "waiting_seconds": round(now - queued.requested_at),
    }


def _pending_row(pending: PendingBranch, now: float) -> dict[str, object]:
    return {
        "kind": pending.origin.value,
        _KEY_BRANCH: pending.branch,
        "from_branch": pending.from_branch,
        "waiting_seconds": round(now - pending.queued_at),
    }


def _cold_start(widening: WideningSignals) -> int | str:
    seconds = widening.cold_start_seconds
    return "unknown" if seconds is None else round(seconds)


def system_prompt(view: JudgeView) -> str:
    """The system prompt for ``view``: :data:`PLAYBOOK_JUDGE_PROMPT` when its session plays a playbook, which the view
    then shows, and :data:`JUDGE_SYSTEM_PROMPT` otherwise."""

    return PLAYBOOK_JUDGE_PROMPT if view.playbook else JUDGE_SYSTEM_PROMPT


def render_prompt(view: JudgeView) -> str:
    """The user prompt for ``view``; the system prompt is :func:`system_prompt`'s."""

    progress = {row.branch: row for row in view.progress}
    state = {
        "metric": view.metric,
        "opening": view.opening,
        "caps": asdict(view.caps),
        "free_branch_slots": view.free_slots,
        "explore_remaining_seconds": round(view.explore_remaining_seconds),
        "min_useful_seconds": round(view.widening.min_useful_seconds),
        "cold_start": _cold_start(view.widening),
        "toolbox_headroom": view.widening.toolbox_headroom,
        "pending": [_pending_row(pending, view.now) for pending in view.pending],
        "session_remaining_seconds": round(view.session_remaining_seconds),
        "integration": {
            "recorded_value": dict(view.integration_value),
            "roofline_ceiling": dict(view.roofline_ceiling),
            "headroom_pct": _headroom_pct(view),
        },
        "spend": {
            "gpu_hours_used": round(view.gpu_hours_used, _GPU_HOURS_DIGITS),
            "explore_gpu_hours": _hours(view.explore_gpu_hours),
            "prelude_gpu_hours_upper_bound": round(view.prelude_gpu_hours_upper_bound, _GPU_HOURS_DIGITS),
            "specialist_llm_usd": round(view.specialist_llm_usd, _USD_DIGITS),
            "max_llm_usd": view.max_llm_usd,
            "scheduler_llm_usd": round(view.scheduler_llm_usd, _USD_DIGITS),
        },
        "events": [{"event": event.kind.value, "reason": event.reason, **event.detail} for event in view.events],
        "stranded_leads": [lead.describe() for lead in view.stranded_leads],
        _KEY_BRANCHES: [_branch_row(entry, progress.get(entry.branch), view) for entry in view.roster],
    }
    history = f"How similar past sessions went:\n{view.history}\n\n" if view.history else ""
    sections = "".join(f"{text}\n\n" for text in (view.open_areas, view.playbook) if text)
    return f"Session state:\n{json.dumps(state, indent=2, sort_keys=True)}\n\n{history}{sections}{REPLY_FORMAT}"


def _int_field(item: Mapping[str, object], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _MalformedReply(f"{key} is not an integer: {value!r}")
    return value


def _str_field(item: Mapping[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _MalformedReply(f"{key} is not a non-empty string: {value!r}")
    return value.strip()


def _objects(payload: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    items = payload.get(key) or []
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise _MalformedReply(f"{key} is not a list of objects")
    return items


def _focus(payload: Mapping[str, object]) -> tuple[str, ...]:
    items = payload.get(_KEY_FOCUS) or []
    if not isinstance(items, list) or not all(item is None or isinstance(item, str) for item in items):
        raise _MalformedReply(f"{_KEY_FOCUS} is not a list of ops")
    return tuple((item or "").strip() for item in items)


def _session_stop(payload: Mapping[str, object]) -> str:
    request = payload.get(_KEY_STOP_SESSION)
    if request is None:
        return ""
    if not isinstance(request, dict):
        raise _MalformedReply(f"{_KEY_STOP_SESSION} is neither null nor an object")
    return _str_field(request, _KEY_REASON)


def _stop_only(reply: str) -> JudgeDecision | None:
    payload = extract_last_json_with_key(reply, _KEY_STOP_SESSION)
    if payload is None or payload[_KEY_STOP_SESSION] is None:
        return None
    try:
        return JudgeDecision(session_stop=_session_stop(payload))
    except _MalformedReply:
        return None


def parse_decision(reply: str) -> JudgeDecision | None:
    """The decision in ``reply``'s last JSON object carrying ``"open"``; ``None`` when it is malformed.

    A reply with no such object decides only when its last object carrying ``"stop_session"`` asks to stop the
    session: that decision opens nothing and gives no order. Any other reply holds no decision.
    """

    payload = extract_last_json_with_key(reply, _KEY_OPEN)
    if payload is None:
        return _stop_only(reply)
    try:
        return JudgeDecision(
            open_count=_int_field(payload, _KEY_OPEN),
            open_k=_int_field(payload, _KEY_OPEN_K) if _KEY_OPEN_K in payload else 1,
            focus=_focus(payload),
            k=tuple(
                KOrder(_str_field(item, _KEY_BRANCH), _int_field(item, "k"))
                for item in _objects(payload, _KEY_BRANCHES)
            ),
            stop=tuple(
                StopOrder(_str_field(item, _KEY_BRANCH), _str_field(item, _KEY_REASON))
                for item in _objects(payload, _KEY_STOP)
            ),
            session_stop=_session_stop(payload),
        )
    except _MalformedReply:
        return None


def cap_decision(decision: JudgeDecision, view: JudgeView) -> JudgeDecision:
    """``decision`` cut to the caps and to the live branches; applying it twice changes nothing more.

    Orders naming an unknown or stopped branch are dropped, the integration branch cannot be stopped, a branch being
    stopped gets no K order, and a later order for the same branch replaces an earlier one. The opening call opens
    at least one branch unless a specialist branch is open already; no call opens more than ``max_branches``. The
    free slots are not a cut here: they are read before the stops free theirs, and the branches that find no slot
    are cut as they open. The held branches are not a cut here either: :func:`hold_decision` cuts them as the
    decision is applied.
    """

    live = {entry.branch: entry for entry in view.roster if entry.stop is None}
    asked = {
        order.branch: order
        for order in decision.stop
        if order.branch in live and live[order.branch].kind is BranchKind.SPECIALIST
    }
    ks = {
        order.branch: KOrder(order.branch, view.caps.clamp_k(order.k))
        for order in decision.k
        if order.branch in live and order.branch not in asked
    }
    floor = 1 if view.opening and not specialist_open(view.roster) else 0
    opened = min(max(decision.open_count, floor), view.caps.max_branches)
    return JudgeDecision(
        open_count=opened,
        open_k=view.caps.clamp_k(decision.open_k),
        focus=decision.focus[:opened],
        k=tuple(ks.values()),
        stop=tuple(asked.values()),
        session_stop=decision.session_stop,
        held=decision.held,
    )


def hold_decision(
    decision: JudgeDecision, held: Mapping[str, HoldReason], roster: Iterable[BranchEntry]
) -> JudgeDecision:
    """``decision`` without the stops and the lower K orders of the ``held`` branches, the K of a branch the safety
    envelope holds excepted (:data:`ENVELOPE_HOLDS`), and without any K order for a branch the judge does not steer,
    held or not, each named in ``held`` once; ``roster`` gives the K a branch runs at now and its origin. Applying it
    twice changes nothing more."""

    entries = tuple(roster)
    running_k = {entry.branch: entry.k for entry in entries}
    unsteered = {entry.branch for entry in entries if not entry.origin.steered_by_judge}
    narrowed = {branch for branch, reason in held.items() if reason not in ENVELOPE_HOLDS}
    refused_stops = [order for order in decision.stop if order.branch in held]
    refused_ks = [
        order
        for order in decision.k
        if order.branch in unsteered or (order.branch in narrowed and order.k < running_k[order.branch])
    ]
    named = {item.branch: item for item in decision.held} | {
        order.branch: HeldOrder(order.branch, held.get(order.branch, HoldReason.UNSTEERED))
        for order in (*refused_stops, *refused_ks)
    }
    return replace(
        decision,
        stop=tuple(order for order in decision.stop if order not in refused_stops),
        k=tuple(order for order in decision.k if order not in refused_ks),
        held=tuple(named.values()),
    )


@dataclass(frozen=True)
class Judge:
    """Asks ``completion`` for a decision and appends every exchange to the ``transcript`` JSONL file."""

    completion: Completion
    transcript: Path
    clock: Clock

    async def decide(self, view: JudgeView) -> JudgeDecision | None:
        """The capped decision for ``view``; ``None`` when the call failed or the reply held none."""

        prompt = render_prompt(view)
        reply = ""
        decision = None
        try:
            reply = (await self.completion(system_prompt(view), prompt)).text
        except CompletionFailed as exc:
            error = str(exc)
        else:
            decision = parse_decision(reply)
            error = "" if decision is not None else "the reply holds no well-formed decision object"
        append_jsonl(
            self.transcript,
            {"at": self.clock.now(), "prompt": prompt, "reply": reply, "error": error},
            make_parents=True,
        )
        return cap_decision(decision, view) if decision is not None else None


__all__ = [
    "ENVELOPE_HOLDS",
    "GRACE_KEY",
    "HeldOrder",
    "HoldReason",
    "HoldRules",
    "Judge",
    "JudgeDecision",
    "JudgeView",
    "KOrder",
    "QueuedIntent",
    "StopOrder",
    "cap_decision",
    "envelope_holds",
    "parse_decision",
    "held_branches",
    "queued_intents",
    "hold_decision",
    "render_prompt",
    "session_grace",
    "system_prompt",
]
