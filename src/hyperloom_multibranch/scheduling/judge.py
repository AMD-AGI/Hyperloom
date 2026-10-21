# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge: an LLM that, inside the scheduler's hard caps, decides the branches' K, what to open and the stops.

The scheduler hands the judge a :class:`JudgeView` (the roster, each branch's frontier progress, the remaining budget,
the integration branch's recorded value, how similar archived sessions went, whether a branch opened now would
still pay off, which running branches made much the same change, the events since its last call, how far each
branch lies behind the integration head, whether a profile of the integration head runs, the GPU hours and LLM
tokens the session has used, and what is known of each running branch's specialist). The judge answers with one JSON
object, which :func:`parse_decision` reads and :func:`cap_decision` cuts to the caps and to the branches that are
still live. A failed call or an unreadable reply gives ``None``, and the scheduler keeps the status quo. The judge
never calls a peer; the scheduler applies its decision. The judge may neither stop nor lower the K of a held branch
(:func:`held_branches`): a specialist branch until :data:`SETTLED_TESTS` of its explore tests ended, unless its
specialist gave up or its judging failed, and a branch with a candidate waiting for its confirm, unless its judging
failed. The scheduler finds the held branches again as it applies the decision, since a candidate can start waiting
while the judge answers, and :func:`hold_decision` refuses their orders and names them in the decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom_multibranch.contracts.lifecycle import SpecialistStatus
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, PendingCandidate, TestRecord
from hyperloom_multibranch.notebook import STALL_TESTS
from hyperloom_multibranch.notebook.rsi.calibration import PolicyChoice
from hyperloom_multibranch.notebook.rsi.frontier import SETTLED_TESTS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.events import MB_BIG_GAIN_BARS, MB_BOTTLENECK_SHIFT_PP, SessionEvent
from hyperloom_multibranch.scheduling.forks import PendingBranch
from hyperloom_multibranch.scheduling.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind, RosterCaps, specialist_open
from hyperloom_multibranch.scheduling.similarity import SimilarBranch
from hyperloom_multibranch.scheduling.widening import WideningSignals
from hyperloom_multibranch.specialist.prompts import VERSION_HEX

_KEY_OPEN = "open"
_KEY_OPEN_K = "open_k"
_KEY_FOCUS = "focus"
_KEY_BRANCHES = "branches"
_KEY_STOP = "stop"
_KEY_STOP_SESSION = "stop_session"
_KEY_BRANCH = "branch"
_KEY_REASON = "reason"
#: Decimals the GPU hours in the judge's view are rounded to.
_GPU_HOURS_DIGITS = 3
#: Decimals the dollars in the judge's view are rounded to.
_USD_DIGITS = 2

JUDGE_SYSTEM_PROMPT = f"""You are the scheduler's judge in a multi-branch LLM inference optimization session.
Specialist branches explore optimizations in parallel; the integration branch merges their kept commits.
Each branch has K, the number of tests it may run at once. Within the hard caps you are given, decide:
how many branches to open, each branch's K (the integration branch included), which specialist branches to stop and
why, and whether the whole session should stop early.
You may open branches at any call. opening is true only for the call that sizes the opening batch, which opens from
the root commit, at least one branch unless user branches opened already. A branch opened at a later call starts from
the integration head, briefed on its focus: the op it should speed up. Name the focus of each, or leave it out and
the branch is briefed on the largest op of the integration head's latest profile, or on none when that head was never
profiled. The opening batch is given no focus. free_branch_slots counts the branches that may still be opened before
your stops take effect: a branch you stop frees its slot for a branch you open in the same answer, and branches
beyond the slots left are not opened. pending lists the forks and repairs the rule gate asked for that wait for a
slot, oldest first, with how long each has waited: a slot your stops free goes to them first and only the slots left
open your branches. You do not admit them; stop a branch that is worth less than a pending one to let it start.
Spend the budget where the expected gain is highest: a branch whose frontier expected_gain_pct is low and that is not
observing a slow-paying direction is a candidate to stop or to lower K; keep K where the gain is high.
A branch's basis says what its expected gain rests on: its own trend, a prior from similar past sessions, the floor
of a direction still being observed, the floor of a branch too new to have a trend, or the roofline headroom that
capped it; trust a measured trend over a floor. frontier_policy names the scoring's variation and the evidence it was
chosen on: default, or the one that ranked the branches of similar past sessions better. frontier_policy's scorer is
the formula behind every expected_gain_pct now: frontier-v1, or another scorer the calibration or the user chose,
which falls back to frontier-v1 for the rest of the session once it has failed too often.
A branch's judged_since_commit counts its tests judged since its last commit (since it opened, before any). When it
reaches {STALL_TESTS}, its specialist is asked to read its brief again and reconsider its direction: a branch whose
count runs well past that is stalling, a candidate to stop or to lower K unless its frontier still expects a gain.
A branch's similar_to lists the other running branches whose own change since their start commit is much the same as
its own, with score the share of the two changes they have in common. identical is true when both heads hold the same
recipe: two branches marked identical spend twice the budget on one result, so stop the one of the pair with the
lower expected_gain_pct. A similar pair that is not identical is yours to weigh.
events lists what happened since your last call that changes what the running branches are worth; you are asked at
once when one happens. integration_moved: a merge moved the integration head by gain_pct, at least {MB_BIG_GAIN_BARS}
times the bar threshold_pct it was judged against, or merged a foundation fix (foundation_fix true), which merges
whatever its gain because the other branches build on what it fixes. A branch's behind_integration_pct is how far its
head's recorded value lies below the integration head's, in percent of the root's value: a branch far behind gains
little even once merged, so it is a candidate to stop, and a branch you open now starts from the new integration
head.
bottleneck_moved: the profile of the integration head taken after it moved found its largest op changed, or that op's
share fell by more than {MB_BOTTLENECK_SHIFT_PP} percentage points: branches still speeding up the previous largest op
have less to gain, and a branch you open now is briefed on the new largest op unless you name its focus.
The integration branch's profile_running is true while that profile runs: its PROFILE test holds one of the
integration branch's K slots, so merge tests wait for it while that K is 1; size the integration branch's K with it.
environment_changed: the toolbox quarantined a measurement node after an infrastructure failure (gpu_ids is empty for
the whole node), or a node's measurement environment moved between two of its results (changed names each of
image_digest, gpu_model, power_cap_w and partition_mode that moved, before and after, and request_id the first result
measured after the move). Tests still run, on a single node even on the quarantined one, but results measured before
and after it may compare less well; weigh whether opening more branches is still worth it.
Before opening a branch, weigh whether it would still pay off. min_useful_seconds is the least EXPLORE time a branch
opened now needs to show a trend: the frontier's trend window, or its specialist's start-up plus the sandbox cold
starts its first tests wait through, when that is longer. cold_start is the recent sandbox cold start in seconds, or
"unknown" before the first one, and min_useful_seconds then leaves the cold starts out and reads low. When
explore_remaining_seconds is below min_useful_seconds, a branch opened now is unlikely to show anything.
toolbox_headroom is how many more tests the toolbox would start at once: the max_parallel_tests cap less the tests
running and queued. A branch opened now makes progress only while it is above 0, and runs at its full K only while it
is at least that K.
spend is what the session has used so far. gpu_hours_used is the GPU time of every test of this session that has a
result, a failed one and the baseline measured on the root commit included; a test taken back before its result is
not in it, and neither are the measurements the PRELUDE run that built the root commit made on its own.
explore_gpu_hours is the gpu_hours_used at which EXPLORE ends to leave FINAL its share, or null when GPU hours are not
limited: as gpu_hours_used nears it, a branch opened now has few tests left to run, and the session stops exploring
there whatever you decide. prelude_gpu_hours_upper_bound is at most the GPU time the PRELUDE run used on its own: its
wall time times the cards one test uses, which nothing limits and which gpu_hours_used leaves out.
specialist_llm_usd is what the specialists' LLM launches cost, each priced per call from its log when it ended, a
launch cut short, lost or stopped included; a launch still running is not in it yet.
scheduler_llm_usd is what your own calls, the patch resolver's and the rule gate's fix critic's cost.
max_llm_usd is the specialist_llm_usd plus scheduler_llm_usd at which the session stops exploring whatever you decide,
or null when it is not limited: as that sum nears it, a branch opened now has few launches left.
A specialist branch's specialist, shown only when the specialist runtime reports it, says whether its specialist is
running, how many times it crashed (crashes) and how many times it was restarted after its heartbeat was lost (lost),
whether the runtime gave up restarting it after repeated crashes, after several sessions in a row lost their
heartbeat or after several sessions in a row found nothing to test (gave_up), and whether it was stopped because
judging its tests failed (judge_failed). A branch whose specialist gave up or whose judging failed makes no more
progress but holds its slot until you stop it: stop it, and open a branch in its place if one would still pay off.
Each lost heartbeat costs minutes: weigh a specialist's lost count against its progress, and stop it when it keeps
losing its heartbeat without gaining.
A branch of origin user follows a direction the user wrote; such branches open from the root commit ahead of the
opening batch, and the opening batch fills the slots they leave. Every specialist branch, of any origin, has a grace:
while its grace is true, until {SETTLED_TESTS} of its explore tests ended with a result, a failed one included, a
stop or a lower K you give it is not applied, unless its specialist gave up or its judging failed.
A branch whose steered is false runs a third-party agent the user named, by the agent's own logic, one test at a time:
it has no grace, and no stop or K you give it is applied; weigh the slot it holds as given.
A branch's pending_candidates are its measured candidates that beat its head by at least the bar while their confirm
runs, each with its gain_over_head_pct. A branch with a pending candidate cannot be stopped or narrowed until its
confirm is judged: a stop or a lower K you give it is not applied, unless its judging failed.
Stop the session only when no branch is expected to gain anything worth the remaining time.
Answer with exactly ONE JSON object and nothing after it."""

_REPLY_FORMAT = """Reply with one JSON object of this shape:
{"open": <int, branches to open now; 0 opens none>,
 "open_k": <int, the K of every branch opened now>,
 "focus": ["<op the first branch opened now should speed up>", ...] (optional; null for a branch you leave to the
  default),
 "branches": [{"branch": "<branch id>", "k": <int>}],
 "stop": [{"branch": "<branch id>", "reason": "<why>"}],
 "stop_session": null or {"reason": "<why the session should stop now>"}}
List only the branches whose K should change. Branches not named keep their K and keep running."""

#: The judge prompt's version in the session's manifest fingerprint, derived from its text as the specialist prompts'.
JUDGE_PROMPT_VERSION = hashlib.sha256(f"{JUDGE_SYSTEM_PROMPT}\n{_REPLY_FORMAT}".encode()).hexdigest()[:VERSION_HEX]


class HoldReason(str, Enum):
    """Why the judge may neither stop a branch nor lower its K: a specialist branch in its grace, a branch with a
    candidate waiting for its confirm, or a branch no judge steers, whose K the judge may not raise either."""

    GRACE = "grace"
    PENDING_CONFIRM = "pending_confirm"
    UNSTEERED = "unsteered"


@dataclass(frozen=True)
class JudgeView:
    """What the judge sees; ``opening`` is true only for the call that sizes the opening batch, which opens at least
    one branch unless a specialist branch is open already.

    ``free_slots`` is the roster's own count of branches that may still be opened; ``pending`` are the forks and
    repairs the fork desk queued for a slot, oldest first; ``history`` is the notebook's
    digest of similar archived sessions, empty when the archive holds none. ``widening`` says whether a branch
    opened now would still pay off. ``similar_to`` maps a branch to the running branches whose own change is much the
    same as its own; a branch similar to none is not in it. ``events`` are the events detected since the last call,
    and ``behind_integration_pct`` maps a running specialist branch to how far its head lies behind the integration
    head, in percent of the root's value. ``profile_running`` is whether a profile of the integration head runs.
    ``gpu_hours_used`` are the GPU hours of the session's tests that have a result: the root's baseline is among them,
    a test withdrawn before its result and the PRELUDE child's own measurements are not. ``explore_gpu_hours`` is where
    EXPLORE ends under ``--max-gpu-hours``, ``None`` without a limit, ``prelude_gpu_hours_upper_bound`` the PRELUDE
    child's wall time times the workload's TP. ``specialist_llm_usd`` is what the specialists' recorded LLM calls
    cost, ``scheduler_llm_usd`` what the judge's, the patch resolver's and the fix critic's recorded calls cost, and
    ``max_llm_usd`` where their sum ends EXPLORE under ``--max-llm-usd``, ``None`` without a limit. ``specialists``
    maps a running specialist branch to what the specialist runtime knows of its specialist; a branch whose
    specialist's state is not known is not in it. ``held`` maps each branch the judge may neither stop nor lower the
    K of to why, as :func:`held_branches` finds them when the view is built; it is what the prompt shows.
    ``frontier_policy`` is the policy the frontier scores rest on, with its label and evidence, and
    ``frontier_scorer`` the label of the scorer that scores now, ``frontier-v1`` once the chosen one fell back.
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
    profile_running: bool
    gpu_hours_used: float
    explore_gpu_hours: float | None
    prelude_gpu_hours_upper_bound: float
    specialist_llm_usd: float
    max_llm_usd: float | None
    scheduler_llm_usd: float
    specialists: Mapping[str, SpecialistStatus]
    held: Mapping[str, HoldReason]
    frontier_policy: PolicyChoice
    frontier_scorer: str


def held_branches(
    roster: Iterable[BranchEntry],
    progress: Iterable[BranchProgress],
    records: Iterable[TestRecord],
    specialists: Mapping[str, SpecialistStatus],
) -> dict[str, HoldReason]:
    """The live branches the judge may neither stop nor lower the K of, with why.

    A specialist branch is in its grace while fewer than :data:`SETTLED_TESTS` of its explore tests have a result,
    whatever its validity, and its specialist neither gave up nor was stopped for a failed judging; a withdrawn test
    has no result. A branch whose progress row holds a pending candidate is held until its confirm is judged, unless
    its judging failed: the confirm's verdict does not need the specialist. A branch whose origin the judge does not
    steer is held for that, or for its pending confirm, which the scheduler's own stops must still wait for."""

    ended = Counter(
        record.request.branch
        for record in records
        if record.request.kind is TestKind.TEST
        and record.request.purpose is Purpose.EXPLORE
        and record.result is not None
    )
    waiting = {row.branch for row in progress if row.pending}
    held: dict[str, HoldReason] = {}
    for entry in roster:
        state = specialists.get(entry.branch)
        if entry.stop is not None or (state is not None and state.judge_failed):
            continue
        if not entry.origin.steered_by_judge:
            held[entry.branch] = HoldReason.PENDING_CONFIRM if entry.branch in waiting else HoldReason.UNSTEERED
        elif (
            entry.kind is BranchKind.SPECIALIST
            and ended[entry.branch] < SETTLED_TESTS
            and not (state is not None and state.gave_up)
        ):
            held[entry.branch] = HoldReason.GRACE
        elif entry.branch in waiting:
            held[entry.branch] = HoldReason.PENDING_CONFIRM
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
    """A stop or a lower K the judge gave a held branch, refused for ``reason``."""

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
    }
    if entry.branch == INTEGRATION_BRANCH:
        row["profile_running"] = view.profile_running
    behind = view.behind_integration_pct.get(entry.branch)
    if behind is not None:
        row["behind_integration_pct"] = round(behind, 2)
    if entry.stop is not None:
        row["stopped"] = entry.stop.summary
    if entry.kind is BranchKind.SPECIALIST:
        row["grace"] = view.held.get(entry.branch) is HoldReason.GRACE
    if not entry.origin.steered_by_judge:
        row["steered"] = False
    specialist = view.specialists.get(entry.branch)
    if specialist is not None:
        row["specialist"] = asdict(specialist)
    row.update(_progress_row(progress))
    return row


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


def render_prompt(view: JudgeView) -> str:
    """The user prompt for ``view``; the system prompt is :data:`JUDGE_SYSTEM_PROMPT`."""

    progress = {row.branch: row for row in view.progress}
    state = {
        "metric": view.metric,
        "frontier_policy": {
            "label": view.frontier_policy.label,
            "evidence": view.frontier_policy.evidence,
            "scorer": view.frontier_scorer,
        },
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
        _KEY_BRANCHES: [_branch_row(entry, progress.get(entry.branch), view) for entry in view.roster],
    }
    history = f"How similar past sessions went:\n{view.history}\n\n" if view.history else ""
    return f"Session state:\n{json.dumps(state, indent=2, sort_keys=True)}\n\n{history}{_REPLY_FORMAT}"


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
    """``decision`` without the stops and the lower K orders of the ``held`` branches, each named in ``held`` once;
    ``roster`` gives the K a branch runs at now. Applying it twice changes nothing more."""

    running_k = {entry.branch: entry.k for entry in roster}
    refused_stops = [order for order in decision.stop if order.branch in held]
    refused_ks = [
        order
        for order in decision.k
        if order.branch in held and (order.k < running_k[order.branch] or held[order.branch] is HoldReason.UNSTEERED)
    ]
    named = {item.branch: item for item in decision.held} | {
        order.branch: HeldOrder(order.branch, held[order.branch]) for order in (*refused_stops, *refused_ks)
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
            reply = (await self.completion(JUDGE_SYSTEM_PROMPT, prompt)).text
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
    "JUDGE_PROMPT_VERSION",
    "JUDGE_SYSTEM_PROMPT",
    "HeldOrder",
    "HoldReason",
    "Judge",
    "JudgeDecision",
    "JudgeView",
    "KOrder",
    "StopOrder",
    "cap_decision",
    "parse_decision",
    "held_branches",
    "hold_decision",
    "render_prompt",
]
