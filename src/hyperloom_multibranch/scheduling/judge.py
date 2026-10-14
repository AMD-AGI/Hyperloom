# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge: an LLM that, inside the scheduler's hard caps, decides the branches' K, the opening count and the stops.

The scheduler hands the judge a :class:`JudgeView` (the roster, each branch's frontier progress, the remaining budget
and the integration branch's recorded value). The judge answers with one JSON object, which :func:`parse_decision`
reads and :func:`cap_decision` cuts to the caps and to the branches that are still live. A failed call or an
unreadable reply gives ``None``, and the scheduler keeps the status quo. The judge never calls a peer; the scheduler
applies its decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom_multibranch.contracts.notebook.records import BranchProgress
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind, RosterCaps

JUDGE_PROMPT_VERSION = "mb1-judge-1"

_KEY_OPEN = "open"
_KEY_OPEN_K = "open_k"
_KEY_BRANCHES = "branches"
_KEY_STOP = "stop"
_KEY_STOP_SESSION = "stop_session"
_KEY_BRANCH = "branch"
_KEY_REASON = "reason"

JUDGE_SYSTEM_PROMPT = """You are the scheduler's judge in a multi-branch LLM inference optimization session.
Specialist branches explore optimizations in parallel; the integration branch merges their kept commits.
Each branch has K, the number of tests it may run at once. Within the hard caps you are given, decide:
how many branches to open (only when the view says opening is true), each branch's K (the integration branch
included), which specialist branches to stop and why, and whether the whole session should stop early.
Spend the budget where the expected gain is highest: a branch whose frontier expected_gain_pct is low and that is not
observing a slow-paying direction is a candidate to stop or to lower K; keep K where the gain is high.
Stop the session only when no branch is expected to gain anything worth the remaining time.
Answer with exactly ONE JSON object and nothing after it."""

_REPLY_FORMAT = """Reply with one JSON object of this shape:
{"open": <int, branches to open now; 0 unless opening is true>,
 "open_k": <int, the K of every branch opened now; ignored unless opening is true>,
 "branches": [{"branch": "<branch id>", "k": <int>}],
 "stop": [{"branch": "<branch id>", "reason": "<why>"}],
 "stop_session": null or {"reason": "<why the session should stop now>"}}
List only the branches whose K should change. Branches not named keep their K and keep running."""


@dataclass(frozen=True)
class JudgeView:
    """What the judge sees; ``opening`` is true only for the call that sizes the opening batch.

    ``free_slots`` is the roster's own count of branches that may still be opened.
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


@dataclass(frozen=True)
class KOrder:
    branch: str
    k: int


@dataclass(frozen=True)
class StopOrder:
    branch: str
    reason: str


@dataclass(frozen=True)
class JudgeDecision:
    """A non-empty ``session_stop`` is the reason the judge asks for a soft stop of the session.

    ``open_k`` is the K of the branches opened now; a K order cannot name them before they exist.
    """

    open_count: int = 0
    open_k: int = 1
    k: tuple[KOrder, ...] = ()
    stop: tuple[StopOrder, ...] = ()
    session_stop: str = ""


class _MalformedReply(ValueError):
    pass


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
        "gpu_seconds": progress.gpu_seconds,
        "head_value": dict(progress.head_value),
    }
    if progress.frontier is not None:
        row["expected_gain_pct"] = progress.frontier.expected_gain_pct
        row["components"] = dict(progress.frontier.components)
    return row


def _branch_row(entry: BranchEntry, progress: BranchProgress | None) -> dict[str, object]:
    row: dict[str, object] = {
        _KEY_BRANCH: entry.branch,
        "kind": entry.kind.value,
        "status": entry.status.value,
        "origin": entry.origin.value,
        "from_branch": entry.from_branch,
        "start_commit": entry.start_commit,
        "k": entry.k,
    }
    if entry.stop is not None:
        row["stopped"] = entry.stop.summary
    row.update(_progress_row(progress))
    return row


def render_prompt(view: JudgeView) -> str:
    """The user prompt for ``view``; the system prompt is :data:`JUDGE_SYSTEM_PROMPT`."""

    progress = {row.branch: row for row in view.progress}
    state = {
        "metric": view.metric,
        "opening": view.opening,
        "caps": asdict(view.caps),
        "free_branch_slots": view.free_slots,
        "explore_remaining_seconds": round(view.explore_remaining_seconds),
        "session_remaining_seconds": round(view.session_remaining_seconds),
        "integration": {
            "recorded_value": dict(view.integration_value),
            "roofline_ceiling": dict(view.roofline_ceiling),
            "headroom_pct": _headroom_pct(view),
        },
        _KEY_BRANCHES: [_branch_row(entry, progress.get(entry.branch)) for entry in view.roster],
    }
    return f"Session state:\n{json.dumps(state, indent=2, sort_keys=True)}\n\n{_REPLY_FORMAT}"


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


def _session_stop(payload: Mapping[str, object]) -> str:
    request = payload.get(_KEY_STOP_SESSION)
    if request is None:
        return ""
    if not isinstance(request, dict):
        raise _MalformedReply(f"{_KEY_STOP_SESSION} is neither null nor an object")
    return _str_field(request, _KEY_REASON)


def parse_decision(reply: str) -> JudgeDecision | None:
    """The decision in ``reply``'s last JSON object carrying ``"open"``; ``None`` when there is none or it is malformed."""

    payload = extract_last_json_with_key(reply, _KEY_OPEN)
    if payload is None:
        return None
    try:
        return JudgeDecision(
            open_count=_int_field(payload, _KEY_OPEN),
            open_k=_int_field(payload, _KEY_OPEN_K) if _KEY_OPEN_K in payload else 1,
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
    stopped gets no K order, and a later order for the same branch replaces an earlier one.
    """

    live = {entry.branch: entry for entry in view.roster if entry.stop is None}
    stops = {
        order.branch: order
        for order in decision.stop
        if order.branch in live and live[order.branch].kind is BranchKind.SPECIALIST
    }
    ks = {
        order.branch: KOrder(order.branch, view.caps.clamp_k(order.k))
        for order in decision.k
        if order.branch in live and order.branch not in stops
    }
    opened = min(max(decision.open_count, 1), view.free_slots) if view.opening else 0
    return JudgeDecision(
        open_count=opened,
        open_k=view.caps.clamp_k(decision.open_k),
        k=tuple(ks.values()),
        stop=tuple(stops.values()),
        session_stop=decision.session_stop,
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
            reply = await self.completion(JUDGE_SYSTEM_PROMPT, prompt)
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
    "Judge",
    "JudgeDecision",
    "JudgeView",
    "KOrder",
    "StopOrder",
    "cap_decision",
    "parse_decision",
    "render_prompt",
]
