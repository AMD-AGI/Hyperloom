# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base-guided planning: one knowledge base round at the start of a campaign, then KernelForge as it is.

Off unless the campaign runs with ``--wiki-guided``. When on, the campaign's first planning round asks a read-only
builder to turn the knowledge base and the measured evidence into one batch (``wiki_playbook``): every direction both
support now, grouped so that one group edits one piece of code, the groups in the order to implement them. The round
plans one part per group, one Implementer session implements every part on the same kernel, and the result is measured
once. When that batch is kept with items the session did not reach, one more iteration plans and implements just those
(``open_groups``). Then the round is over (``PHASE_FREE``): the campaign plans exactly as without the flag, the
knowledge base is withdrawn from every role, and the planners are handed only what the round settled
(``round_summary``).

A plan claims playbook items by citing them as ``[playbook:P1,P3]`` and declines one as ``[playbook-skip:P4]``. A
batch session ends by citing the items that are in its final kernel, the ones it tried and dropped, and the ones it did
not get to (``[playbook-untried:P5]``, which stay open), and the ledger is credited from that report when there is one,
else from the plan; either way a claim counts only once a measurement decided it. A session the stop hook sends back
reports again when it submits again, and only its latest report says what is in the kernel that was measured.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.loop.recovery import atomic_write_json
from kernelforge.loop.wiki_playbook import Playbook
from kernelforge.orchestrator.wiki_batch import (
    OPEN_STATUSES,
    PHASE_FREE,
    PHASE_GUIDED,
    STATUS_APPLIED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_REFUTED,
    STATUS_SKIPPED,
    BatchSession,
)
from kernelforge.tools.durable_io import atomic_write_text

log = logging.getLogger(__name__)

PHASES = frozenset({PHASE_GUIDED, PHASE_FREE})
ITEM_STATUSES = frozenset({STATUS_PENDING, STATUS_APPLIED, STATUS_REFUTED, STATUS_FAILED, STATUS_SKIPPED})

# Why the knowledge base round ended.
REASON_ROUND_DONE = "ROUND_DONE"
REASON_NO_USEFUL_DIRECTION = "NO_USEFUL_DIRECTION"
REASON_PLAYBOOK_UNAVAILABLE = "PLAYBOOK_UNAVAILABLE"
REASON_NO_BATCH_PLAN = "NO_BATCH_PLAN"

# Builder calls the round makes before giving up on an outage of the builder's backend.
MAX_PLAYBOOK_ATTEMPTS = 2
# Times the round runs its batch again after a session whose outcome says nothing about it.
UNSETTLED_RETRIES = 1
# Decision labels (``runner._decision_label`` and the loop's own labels) by what they settle about a cited item.
DECISION_KEEP = "KEEP"
APPLIED_DECISIONS = frozenset({DECISION_KEEP})
REFUTED_DECISIONS = frozenset({"REVERT_PERF"})
FAILED_DECISIONS = frozenset(
    {
        "BUILD_FAILED",
        "CRASH",
        "NO_CHANGES",
        "REVERT_INTEGRITY",
        "REVERT_VALIDATION",
        "REVERT_VALIDATION_ERROR",
        "REVERT_VALIDATION_TIMEOUT",
    }
)
_SETTLING_DECISIONS = APPLIED_DECISIONS | REFUTED_DECISIONS | FAILED_DECISIONS

_CITATION = re.compile(r"\[playbook:([^\]]*)\]", re.IGNORECASE)
_SKIP_CITATION = re.compile(r"\[playbook-skip:([^\]]*)\]", re.IGNORECASE)
_UNTRIED_CITATION = re.compile(r"\[playbook-untried:([^\]]*)\]", re.IGNORECASE)
_CITED_ID = re.compile(r"\bP\d{1,3}\b", re.IGNORECASE)
# Inside one citation, an entry is a run of item ids at the start of the block or after a `;`, `,` or line break,
# followed by its reason; an id that only appears inside another entry's reason is not cited. In a skip or untried
# citation, after a comma the run opens an entry only when its reason starts right after it ("P6 slower, P7:
# spilled"): in "slower than the split path, P21 would be needed first" P21 is part of P6's reason. A claim lists
# what is in the kernel, each id followed by what it is ("P18 bf16x3 tile, P19 per-shape dispatch"), so there any id
# after a comma opens an entry.
_ENTRY_START = re.compile(r"(?:^|[;,\n])\s*(?=P\d)", re.IGNORECASE)
_ID_RUN = re.compile(r"P\d{1,3}\b(?:\s*(?:,|/|&|\band\b)\s*P\d{1,3}\b)*", re.IGNORECASE)
_REASON_OPENER = re.compile(r"[ \t]*(?:$|[\n:;(=\-\u2013\u2014])")
# The line that closes each submission a session makes; the text after the one before the last is its latest report.
_SUBMIT_LINE = re.compile(r"^[ \t]*SUBMIT_CANDIDATE[ \t]*$", re.IGNORECASE | re.MULTILINE)
# A skip whose reason says the item was never tried or not finished leaves it open rather than settling it. "Not
# implemented" is not among them: a session also writes it of an item whose precondition proved false.
_UNTRIED_REASON = re.compile(
    r"not (?:attempted|tried|started|reached|finished)|untried|"
    r"(?:lack|out|short) of (?:session )?time|ran out of time|deadline|time budget|no time",
    re.IGNORECASE,
)
_NOTE_CHARS = 300
_NO_REASON = "no reason given"

GUIDED_INSTRUCTIONS = (
    "Knowledge base round. `playbook` lists the directions the knowledge base and the measured evidence support for "
    "this kernel, each item with its status; `batch` is this round's groups, in the order to implement them. Plan from "
    "it:\n"
    "1. The whole batch is one iteration. One Implementer session implements every group in `batch`, in order, on "
    "the same kernel, and the result is measured once when the session ends. Each group is one part of that session, "
    "implemented whole, and each part is built from its group's analysis (role `playbook_group`).\n"
    "2. Plan each part so that the session can keep or drop it on its own: it must stay correct, and say how it "
    "adapts, when an earlier part was dropped.\n"
    "3. On its own line, cite every item a plan implements as [playbook:P1,P3], and every item the evidence shows "
    "does not apply as [playbook-skip:P4 followed by that evidence].\n"
    "4. Adapt each item to this kernel's evidence (profile, cases, code). Read an item's knowledge base pages with the "
    "knowledge tools before planning it; an item whose `support` is `profiling` has no page, and `applies_when` names "
    "the measurement behind it.\n"
    "5. An APPLIED or SKIPPED item is settled; plan only the open ones.\n"
    "Add work outside the playbook only when the evidence shows a cited item cannot pay off without it."
)

BATCH_SESSION_INSTRUCTIONS = (
    "This plan is one knowledge base batch: {parts} part(s), each one playbook group, implemented in this one session "
    "on the same kernel and measured once, when the session ends. Implement the parts in the order given. After each "
    "part, run the correctness check and time the cases: keep the part when it is correct and not slower than the "
    "kernel was before it, and otherwise revert that part's edits and go on to the next part. Leave a part out only "
    "when its precondition proves false or it measured worse; being hard or long is not a reason. Submit only after "
    "the last part, or at the wrap-up time the session deadline gives. End the session with these lines of its own, "
    "each entry an item id followed by why, entries separated by `;`: [playbook:...] naming every item whose change "
    "is in the final kernel; [playbook-skip:...] naming every item you implemented and dropped or reverted, or whose "
    "precondition proved false, with that evidence; and [playbook-untried:...] naming every item you did not get to "
    "by the wrap-up time, which stays open."
)

BATCH_UNFINISHED = (
    "The batch is not finished, and {minutes} minutes remain before the wrap-up time. The kernel as it stands is "
    "correct and faster, so keep it and continue from it with part {position} of {parts} (playbook group {group}): "
    "{items} is neither in the kernel nor dropped with evidence. Implement it, check and time it, keep or revert it, "
    "then go on with the parts after it. Leave an item out only as [playbook-skip:<id> <evidence>] when its "
    "precondition proves false or it measured worse; being hard or long is not a reason. End with the [playbook:...], "
    "[playbook-skip:...] and [playbook-untried:...] lines once every part is done or the wrap-up time is reached."
)

IMPLEMENTER_NOTE = (
    "## Wiki guidance\n"
    "Knowledge base round. The plan implements knowledge base playbook items it cites as [playbook:...]. Implement "
    "every cited item; when an item's detail matters, read its knowledge base pages with the knowledge tools (an item "
    "supported by profiling alone has none, and the plan states its evidence). The plan is implemented whole in this "
    "session and measured once at its end: follow its instructions for keeping or dropping each part and for the "
    "lines that end the session."
)


@dataclass
class ItemRecord:
    """What the measured iterations have settled about one item."""

    status: str = STATUS_PENDING
    iterations: list[int] = field(default_factory=list)
    last_decision: str = ""
    # What the latest session that skipped or did not get to the item said about it.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "iterations": list(self.iterations),
            "last_decision": self.last_decision,
        }
        if self.note:
            payload["note"] = self.note
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ItemRecord:
        status = str(payload.get("status") or STATUS_PENDING)
        if status not in ITEM_STATUSES:
            raise ValueError(f"unknown playbook item status {status!r}")
        return cls(
            status=status,
            iterations=[int(value) for value in payload.get("iterations") or []],
            last_decision=str(payload.get("last_decision") or ""),
            note=str(payload.get("note") or ""),
        )


@dataclass
class GuidanceState:
    """The durable state of one campaign's knowledge base round."""

    phase: str = PHASE_GUIDED
    reason_codes: list[str] = field(default_factory=list)
    playbook: Playbook | None = None
    ledger: dict[str, ItemRecord] = field(default_factory=dict)
    # The groups the round's current iteration implements, in implementation order.
    batch: list[str] = field(default_factory=list)
    # Each measured batch iteration: its iteration, decision and, for one that was not kept, why.
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    # Iterations whose outcome the ledger already holds, so a recovery that replays one cannot count it twice.
    recorded_iterations: list[int] = field(default_factory=list)
    # The plan the latest iteration is measuring, kept here so a KEEP finished by resume recovery is still credited.
    inflight_iteration: int = 0
    inflight_plan: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reason_codes": list(self.reason_codes),
            "playbook": self.playbook.to_dict() if self.playbook is not None else None,
            "ledger": {item_id: record.to_dict() for item_id, record in self.ledger.items()},
            "batch": list(self.batch),
            "outcomes": [dict(outcome) for outcome in self.outcomes],
            "recorded_iterations": list(self.recorded_iterations),
            "inflight_iteration": self.inflight_iteration,
            "inflight_plan": self.inflight_plan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GuidanceState:
        phase = str(payload.get("phase") or PHASE_GUIDED)
        if phase not in PHASES:
            raise ValueError(f"unknown guidance phase {phase!r}")
        raw_playbook = payload.get("playbook")
        playbook = Playbook.from_dict(raw_playbook) if isinstance(raw_playbook, Mapping) else None
        ledger = {
            str(item_id): ItemRecord.from_dict(record) for item_id, record in (payload.get("ledger") or {}).items()
        }
        if playbook is not None:
            # Every item has a record, whatever an older state file carried.
            ledger = {item_id: ledger.get(item_id, ItemRecord()) for item_id in playbook.item_ids()}
        return cls(
            phase=phase,
            reason_codes=[str(code) for code in payload.get("reason_codes") or []],
            playbook=playbook,
            ledger=ledger,
            batch=[str(group_id) for group_id in payload.get("batch") or []],
            outcomes=[dict(outcome) for outcome in payload.get("outcomes") or [] if isinstance(outcome, Mapping)],
            recorded_iterations=[int(value) for value in payload.get("recorded_iterations") or []],
            inflight_iteration=int(payload.get("inflight_iteration") or 0),
            inflight_plan=str(payload.get("inflight_plan") or ""),
        )

    def add_build(self, build: Playbook) -> None:
        """Make the builder's answer the round's playbook, every item open."""
        self.playbook = build
        self.ledger = {item_id: ItemRecord() for item_id in build.item_ids()}

    def open_groups(self) -> list[str]:
        """The playbook's groups that still hold an open item, in implementation order."""
        if self.playbook is None:
            return []
        open_items = {item_id for item_id, record in self.ledger.items() if record.status in OPEN_STATUSES}
        return [
            group.group_id
            for group in self.playbook.groups
            if any(item.group == group.group_id and item.item_id in open_items for item in self.playbook.items)
        ]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.ledger.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class Citations:
    """What a plan or a session report cites: the items it claims, skips and did not try, and why for the last two."""

    claimed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    untried: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)

    @property
    def any(self) -> bool:
        return bool(self.claimed or self.skipped or self.untried)


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _entries(block: str, *, strict_commas: bool) -> list[tuple[list[str], str]]:
    """Each entry of one citation block: the item ids it opens with, and the reason that follows them."""
    opened: list[tuple[re.Match[str], re.Match[str]]] = []
    cursor = 0
    for start in _ENTRY_START.finditer(block):
        if start.end() < cursor:
            continue
        run = _ID_RUN.match(block, start.end())
        if run is None:
            continue
        if strict_commas and block[start.start()] == "," and not _REASON_OPENER.match(block, run.end()):
            continue
        opened.append((start, run))
        cursor = run.end()
    entries: list[tuple[list[str], str]] = []
    for index, (_, run) in enumerate(opened):
        following = opened[index + 1][0].start() if index + 1 < len(opened) else len(block)
        reason = _text(block[run.end() : following]).strip(" -\u2013\u2014:;,.")
        entries.append(([match.upper() for match in _CITED_ID.findall(run.group(0))], reason[:_NOTE_CHARS]))
    return entries


def latest_report(text: str) -> str:
    """The part of a session's text that holds its latest report, or ``""`` when it wrote none: the stop hook can send
    a session back after it submitted, and an earlier submission's lines describe a kernel that was not measured."""
    text = text or ""
    bounds = [0, *(match.end() for match in _SUBMIT_LINE.finditer(text))]
    segments = [text[start:end] for start, end in zip(bounds, [*bounds[1:], len(text)])]
    for segment in reversed(segments):
        if any(pattern.search(segment) for pattern in (_CITATION, _SKIP_CITATION, _UNTRIED_CITATION)):
            return segment
    return ""


def _cited(pattern: re.Pattern[str], text: str, known: set[str]) -> tuple[list[str], dict[str, str]]:
    found: list[str] = []
    reasons: dict[str, str] = {}
    cited = (
        (item_id, reason)
        for block in pattern.findall(text)
        for item_ids, reason in _entries(block, strict_commas=pattern is not _CITATION)
        for item_id in item_ids
    )
    for item_id, reason in cited:
        if item_id not in known or item_id in found:
            continue
        found.append(item_id)
        if reason:
            reasons[item_id] = reason
    return found, reasons


def read_citations(text: str, known: Iterable[str]) -> Citations:
    """The items ``text`` claims, skips and did not try, among ``known``; a claim wins over any other mention, since a
    report names an item it kept only in part both as kept and as dropped. ``text`` is one plan, or one session's
    latest report (``latest_report``)."""
    known_ids = set(known)
    text = text or ""
    claimed, _ = _cited(_CITATION, text, known_ids)
    skipped, skip_reasons = _cited(_SKIP_CITATION, text, known_ids)
    untried, untried_reasons = _cited(_UNTRIED_CITATION, text, known_ids)
    for item_id in list(skipped):
        if _UNTRIED_REASON.search(skip_reasons.get(item_id, "")):
            skipped.remove(item_id)
            if item_id not in untried:
                untried.append(item_id)
                untried_reasons[item_id] = skip_reasons[item_id]
    skipped = [item_id for item_id in skipped if item_id not in claimed]
    untried = [item_id for item_id in untried if item_id not in claimed and item_id not in skipped]
    reasons = {item_id: skip_reasons[item_id] for item_id in skipped if item_id in skip_reasons}
    reasons.update({item_id: untried_reasons[item_id] for item_id in untried if item_id in untried_reasons})
    return Citations(claimed=tuple(claimed), skipped=tuple(skipped), untried=tuple(untried), reasons=reasons)


def cited_items(plan_text: str, known: Iterable[str]) -> tuple[list[str], list[str]]:
    """The items a plan claims and the ones it declines (skipped or not tried), among ``known``."""
    cites = read_citations(plan_text, known)
    return list(cites.claimed), [*cites.skipped, *cites.untried]


def _note(record: ItemRecord, *, iteration: int, label: str, note: str) -> None:
    record.iterations.append(iteration)
    record.last_decision = label
    record.note = note


def settles_nothing(decision: str) -> bool:
    """Whether an iteration's decision says nothing about the items it was given: infrastructure, a contended
    measurement, or a session that measured nothing."""
    return str(decision or "").strip().upper() not in _SETTLING_DECISIONS


def _claimed_status(label: str) -> str:
    if label in APPLIED_DECISIONS:
        return STATUS_APPLIED
    return STATUS_REFUTED if label in REFUTED_DECISIONS else STATUS_FAILED


def record_outcome(state: GuidanceState, *, iteration: int, plan_text: str, decision: str) -> list[str]:
    """Credit one measured iteration to the items its plan cited; the item ids that changed."""
    if state.playbook is None or iteration in state.recorded_iterations:
        return []
    if settles_nothing(decision):
        return []
    label = str(decision).strip().upper()
    state.recorded_iterations.append(iteration)
    cites = read_citations(plan_text, state.ledger)
    changed: list[str] = []
    for item_id in (*cites.skipped, *cites.untried):
        record = state.ledger[item_id]
        if record.status not in OPEN_STATUSES:
            continue
        reason = cites.reasons.get(item_id) or _NO_REASON
        if item_id in cites.skipped:
            record.status = STATUS_SKIPPED
            _note(record, iteration=iteration, label=label, note=f"skipped in iteration {iteration}: {reason}")
        else:
            # Not tried is no evidence about the item: it stays open.
            _note(record, iteration=iteration, label=label, note=f"not tried in iteration {iteration}: {reason}")
        changed.append(item_id)
    for item_id in cites.claimed:
        record = state.ledger[item_id]
        record.iterations.append(iteration)
        record.last_decision = label
        # A later plan re-citing an item already in the incumbent and losing does not take it out again.
        if record.status != STATUS_APPLIED:
            record.status = _claimed_status(label)
        changed.append(item_id)
    return changed


def batch_plan(parts: Sequence[tuple[str, str]]) -> str:
    """One session's plan for a batch: the instructions, then each (label, plan) part in implementation order."""
    sections = [
        f"# Knowledge base batch: {len(parts)} part(s) in one session",
        BATCH_SESSION_INSTRUCTIONS.format(parts=len(parts)),
    ]
    for index, (label, plan) in enumerate(parts, start=1):
        heading = f"## Part {index} of {len(parts)}" + (f": {label}" if label else "")
        sections.append(f"{heading}\n\n{plan.strip()}")
    return "\n\n".join(sections) + "\n"


def batch_unfinished(batch: BatchSession, message: str | None, *, now: float) -> str:
    """Why a batch session stopping with ``message`` as its last word must go on, or ``""`` when it may stop: past its
    wrap-up time, with every item in the kernel or dropped, or when its last word could not be read (``None``)."""
    if message is None or now >= batch.wrap_up_at:
        return ""
    cites = read_citations(latest_report(message), [item for _, items in batch.parts for item in items])
    settled = {*cites.claimed, *cites.skipped}
    for position, (group_id, items) in enumerate(batch.parts, start=1):
        left = [item for item in items if item not in settled]
        if left:
            return BATCH_UNFINISHED.format(
                minutes=max(1, round((batch.wrap_up_at - now) / 60)),
                position=position,
                parts=len(batch.parts),
                group=group_id,
                items=", ".join(left),
            )
    return ""


def _listed(state: GuidanceState, item_ids: Sequence[str]) -> str:
    titles = {item.item_id: item.title for item in state.playbook.items}
    return "; ".join(
        f"{item_id} {titles[item_id]}" + (f" ({state.ledger[item_id].note})" if state.ledger[item_id].note else "")
        for item_id in item_ids
    )


def round_summary(state: GuidanceState) -> str:
    """What the knowledge base round settled, for the planners that come after it; ``""`` when it measured nothing."""
    settled = [outcome for outcome in state.outcomes if not settles_nothing(outcome.get("decision", ""))]
    if state.playbook is None or not settled:
        return ""
    by_status: dict[str, list[str]] = {}
    for item_id, record in state.ledger.items():
        by_status.setdefault(record.status, []).append(item_id)
    measured = ", ".join(
        f"iteration {outcome['iteration']} {outcome['decision']}"
        + (f" ({outcome['detail']})" if outcome.get("detail") else "")
        for outcome in settled
    )
    lines = [f"The campaign's knowledge base round measured {measured}."]
    for status, heading in (
        (STATUS_APPLIED, "In the incumbent"),
        (STATUS_SKIPPED, "Tried and dropped, or found not to apply"),
        (STATUS_REFUTED, "Measured slower with the batch"),
        (STATUS_FAILED, "In a batch that failed"),
        (STATUS_PENDING, "Not reached"),
    ):
        if by_status.get(status):
            lines.append(f"{heading}: {_listed(state, by_status[status])}.")
    return "\n".join(lines)


def render_for_planning(state: GuidanceState) -> dict[str, Any] | None:
    """What every planning role sees as ``context.wiki_guidance``; ``None`` when there is nothing to show."""
    if state.playbook is None:
        return None
    if state.phase != PHASE_GUIDED:
        summary = round_summary(state)
        return {"phase": state.phase, "knowledge_base_round": summary} if summary else None
    items = [
        {
            **item.to_dict(),
            "status": state.ledger[item.item_id].status,
            **({"note": state.ledger[item.item_id].note} if state.ledger[item.item_id].note else {}),
        }
        for item in state.playbook.items
    ]
    groups = [
        {**group.to_dict(), "items": [item.item_id for item in state.playbook.items if item.group == group.group_id]}
        for group in state.playbook.groups
    ]
    return {
        "phase": state.phase,
        "instructions": GUIDED_INSTRUCTIONS,
        "playbook": {"summary": state.playbook.summary, "groups": groups, "items": items},
        "batch": list(state.batch),
    }


def render_for_implementer(state: GuidanceState) -> str:
    """The Implementer's note during the knowledge base round; empty after it."""
    return IMPLEMENTER_NOTE if state.phase == PHASE_GUIDED and state.playbook is not None else ""


class WikiGuidanceStore:
    """The campaign's guidance state and playbook, under ``forge_experiments/wiki_guidance``."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir).resolve() / "forge_experiments" / "wiki_guidance"
        self.state_path = self.root / "state.json"
        self.playbook_path = self.root / "playbook.json"
        self.response_path = self.root / "playbook_response.md"

    def load(self) -> GuidanceState:
        """The persisted state, or a fresh GUIDED one when none is readable."""
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return GuidanceState()
        except (OSError, ValueError) as error:
            log.warning("wiki guidance state at %s is unreadable (%s); starting GUIDED afresh", self.state_path, error)
            return GuidanceState()
        try:
            return GuidanceState.from_dict(payload)
        except (TypeError, ValueError) as error:
            log.warning("wiki guidance state at %s is invalid (%s); starting GUIDED afresh", self.state_path, error)
            return GuidanceState()

    def save(self, state: GuidanceState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, state.to_dict())
        if state.playbook is not None:
            atomic_write_json(self.playbook_path, state.playbook.to_dict())

    def save_response(self, text: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.response_path, text or "")
