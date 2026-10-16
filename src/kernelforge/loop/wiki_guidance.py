# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base-guided planning: the playbook, its ledger, and the phase the campaign plans in.

Off unless the campaign runs with ``--wiki-guided``. When on, the first planning round first asks a read-only builder
to turn the knowledge base into a playbook: techniques for this kernel, grouped so that one group is one
implementation step, each item citing the knowledge base pages behind it. The playbook says what the knowledge base
supports; what to do now is the measured evidence's call, so every GUIDED round first has a ranker choose the round's
batch: every open group whose premise the round's profile supports, in implementation order. The round plans one
step per batch group, and the loop runs the steps one after another on the canonical tree, each an Implementer
session followed by its own measurement, with no new analysis or planning until the last step has been measured.
While the campaign is GUIDED every planning role sees the playbook, the batch and the ledger, and plans from them. The
campaign leaves GUIDED for FREE once the playbook is spent, the search policy reports a stall, or -- after at least
``MIN_GUIDED_ROUNDS`` GUIDED rounds -- the guided share of the budget left when GUIDED began is used; from then on the
planners see what the playbook settled and are told to look beyond it.

A plan claims playbook items by citing them as ``[playbook:P1,P3]`` and declines one as ``[playbook-skip:P4]``. The
ledger is credited from the plan the iteration actually measured, so a claim counts only once a measurement decided
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kernelforge.orchestrator.structured_output import extract_json_object

log = logging.getLogger(__name__)

PHASE_GUIDED = "GUIDED"
PHASE_FREE = "FREE"
PHASES = frozenset({PHASE_GUIDED, PHASE_FREE})

# Why a campaign left GUIDED.
REASON_PLAYBOOK_UNAVAILABLE = "PLAYBOOK_UNAVAILABLE"
REASON_PLAYBOOK_EXHAUSTED = "PLAYBOOK_EXHAUSTED"
REASON_GUIDED_BUDGET_SPENT = "GUIDED_BUDGET_SPENT"
REASON_GUIDED_STALL = "GUIDED_STALL"

# The search policy's own stall signals (``search_policy``); any of them ends GUIDED, because each says the direction
# the campaign is spending sessions on has stopped paying.
STALL_REASONS = frozenset({"NO_IMPROVEMENT_STALL", "DIMINISHING_RETURNS", "REPEATED_NO_CHANGES"})

DEFAULT_GUIDED_BUDGET_FRACTION = 0.35
# The guided budget share cannot end GUIDED before this many GUIDED planning rounds: the second is the first that
# plans from a profile taken after the first round's steps.
MIN_GUIDED_ROUNDS = 2
# The most steps one batch runs before the profile is refreshed.
MAX_BATCH_STEPS = 6

STATUS_PENDING = "PENDING"
STATUS_APPLIED = "APPLIED"
STATUS_REFUTED = "REFUTED"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
DONE_STATUSES = frozenset({STATUS_APPLIED, STATUS_REFUTED, STATUS_SKIPPED})
ITEM_STATUSES = DONE_STATUSES | {STATUS_PENDING, STATUS_FAILED}

# A FAILED item may be retried; the second failure settles it as REFUTED.
MAX_FAILED_ATTEMPTS = 2
# An outage on the builder's backend is retried on the next planning round, up to this many attempts in all.
MAX_PLAYBOOK_ATTEMPTS = 2
MAX_ITEMS = 12

# Decision labels (``runner._decision_label`` and the loop's own labels) by what they settle about a cited item.
APPLIED_DECISIONS = frozenset({"KEEP"})
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

_ITEM_ID = re.compile(r"^P\d{1,2}$")
_GROUP_ID = re.compile(r"^G\d{1,2}$")
_CITATION = re.compile(r"\[playbook:([^\]]*)\]", re.IGNORECASE)
_SKIP_CITATION = re.compile(r"\[playbook-skip:([^\]]*)\]", re.IGNORECASE)
_CITED_ID = re.compile(r"\bP\d{1,2}\b", re.IGNORECASE)

GUIDED_INSTRUCTIONS = (
    "GUIDED phase. `playbook` lists techniques the knowledge base supports for this kernel, in groups whose items "
    "edit the same code and belong in one step. `batch`, when present, is the groups this round implements, in "
    "order, chosen from this round's measured evidence: a group's `rank_evidence` says what put it there, and "
    "`batch_excluded` says why an open group was left out. Plan from it:\n"
    "1. Each group in `batch` is one step: one Implementer session and then one measurement, run one after another "
    "on the same kernel in the batch's order. A step starts from the kernel as the earlier steps left it -- with "
    "their changes when their measurement kept them, without them when it did not -- and the profile is not "
    "refreshed until the batch has ended. Implement all of a group's applicable items together in its step; a group "
    "is one step, not a sequence of rounds, and one step never spans two groups.\n"
    "2. Plan each step to stand on its own measurement: it must not depend on a later step, and it must stay correct "
    "and say how it adapts when an earlier step was not kept.\n"
    "3. On its own line, cite every item a plan implements as [playbook:P1,P3], and every item the evidence shows "
    "does not apply as [playbook-skip:P4] together with that evidence. Work a plan does not cite is not credited "
    "to the playbook.\n"
    "4. Adapt each item to this kernel's evidence (profile, cases, code). Its knowledge base pages are available "
    "through the knowledge tools; read them before planning the item.\n"
    "5. Do not re-plan REFUTED or SKIPPED items. A FAILED item may be retried once, with the failure fixed.\n"
    "6. This round's analyses are one per batch group (role `playbook_group`, the group named in each assignment), "
    "each covering its group whole across compute, memory and algorithm. Build each step from its group's analysis; "
    "an analyst of another role, when there is one, analyzes how this round's playbook items apply to this kernel "
    "rather than surveying it broadly.\n"
    "Add work outside the playbook only when the evidence shows a cited item cannot pay off without it."
)

FREE_INSTRUCTIONS = (
    "FREE phase: the playbook phase has ended ({reasons}). APPLIED items are already in the incumbent; REFUTED "
    "items were measured and did not help in the form tried; SKIPPED items did not apply. Optimize beyond the "
    "playbook with your own knowledge and the evidence. The knowledge base stays available, but it is no longer "
    "the source of the plan. Re-try a REFUTED item only when new evidence changes its premise, and say what changed."
)


@dataclass(frozen=True)
class PlaybookItem:
    """One technique the knowledge base supports for this kernel."""

    item_id: str
    title: str
    group: str
    wiki_pages: tuple[str, ...]
    applies_when: str = ""
    cases: str = ""
    expected_effect: str = ""
    verify: str = ""
    conflicts_with: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "title": self.title,
            "group": self.group,
            "wiki_pages": list(self.wiki_pages),
            "applies_when": self.applies_when,
            "cases": self.cases,
            "expected_effect": self.expected_effect,
            "verify": self.verify,
            "conflicts_with": list(self.conflicts_with),
        }


@dataclass(frozen=True)
class PlaybookGroup:
    """Items that edit the same code and are implemented as one step."""

    group_id: str
    title: str
    why_together: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.group_id, "title": self.title, "why_together": self.why_together}


@dataclass(frozen=True)
class Playbook:
    """The builder's answer, groups in priority order."""

    items: tuple[PlaybookItem, ...]
    groups: tuple[PlaybookGroup, ...]
    summary: str = ""

    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "groups": [group.to_dict() for group in self.groups],
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Playbook:
        playbook, notes = _playbook_from_payload(payload)
        if playbook is None:
            raise ValueError("; ".join(notes) or "playbook has no usable item")
        return playbook


@dataclass
class ItemRecord:
    """What the measured iterations have settled about one item."""

    status: str = STATUS_PENDING
    failures: int = 0
    iterations: list[int] = field(default_factory=list)
    last_decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failures": self.failures,
            "iterations": list(self.iterations),
            "last_decision": self.last_decision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ItemRecord:
        status = str(payload.get("status") or STATUS_PENDING)
        if status not in ITEM_STATUSES:
            raise ValueError(f"unknown playbook item status {status!r}")
        return cls(
            status=status,
            failures=int(payload.get("failures") or 0),
            iterations=[int(value) for value in payload.get("iterations") or []],
            last_decision=str(payload.get("last_decision") or ""),
        )


@dataclass
class GuidanceState:
    """The durable state of one campaign's guided planning."""

    phase: str = PHASE_GUIDED
    reason_codes: list[str] = field(default_factory=list)
    playbook: Playbook | None = None
    playbook_attempts: int = 0
    # Set once the builder's answer is settled, usable or not; an outage leaves it unset so the next round retries.
    playbook_settled: bool = False
    playbook_error: str = ""
    ledger: dict[str, ItemRecord] = field(default_factory=dict)
    phase_changed_at_iteration: int = 0
    # Iterations whose outcome the ledger already holds, so a recovery that replays one cannot count it twice.
    recorded_iterations: list[int] = field(default_factory=list)
    # The plan the latest iteration is measuring, kept here so a KEEP finished by resume recovery is still credited.
    inflight_iteration: int = 0
    inflight_plan: str = ""
    # The latest ranking of the open groups (``parse_ranking``): their ids, first to plan first, and what in the
    # evidence put each there. Groups it does not name keep the builder's order behind the ones it does.
    group_order: list[str] = field(default_factory=list)
    rank_evidence: dict[str, str] = field(default_factory=dict)
    ranked_at_iteration: int = 0
    # The latest GUIDED round's batch: the groups it implements, in step order, and why each open group it left out
    # was left out.
    batch: list[str] = field(default_factory=list)
    batch_excluded: dict[str, str] = field(default_factory=dict)
    # Seconds of the campaign budget already used when the first GUIDED round began planning; the guided share is
    # taken of what was left then, so setup before the first round is not charged to GUIDED.
    guided_started_used_sec: float | None = None
    guided_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reason_codes": list(self.reason_codes),
            "playbook": self.playbook.to_dict() if self.playbook is not None else None,
            "playbook_attempts": self.playbook_attempts,
            "playbook_settled": self.playbook_settled,
            "playbook_error": self.playbook_error,
            "ledger": {item_id: record.to_dict() for item_id, record in self.ledger.items()},
            "phase_changed_at_iteration": self.phase_changed_at_iteration,
            "recorded_iterations": list(self.recorded_iterations),
            "inflight_iteration": self.inflight_iteration,
            "inflight_plan": self.inflight_plan,
            "group_order": list(self.group_order),
            "rank_evidence": dict(self.rank_evidence),
            "ranked_at_iteration": self.ranked_at_iteration,
            "batch": list(self.batch),
            "batch_excluded": dict(self.batch_excluded),
            "guided_started_used_sec": self.guided_started_used_sec,
            "guided_rounds": self.guided_rounds,
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
            playbook_attempts=int(payload.get("playbook_attempts") or 0),
            playbook_settled=bool(payload.get("playbook_settled")),
            playbook_error=str(payload.get("playbook_error") or ""),
            ledger=ledger,
            phase_changed_at_iteration=int(payload.get("phase_changed_at_iteration") or 0),
            recorded_iterations=[int(value) for value in payload.get("recorded_iterations") or []],
            inflight_iteration=int(payload.get("inflight_iteration") or 0),
            inflight_plan=str(payload.get("inflight_plan") or ""),
            group_order=[str(group_id) for group_id in payload.get("group_order") or []],
            rank_evidence={str(key): str(value) for key, value in (payload.get("rank_evidence") or {}).items()},
            ranked_at_iteration=int(payload.get("ranked_at_iteration") or 0),
            batch=[str(group_id) for group_id in payload.get("batch") or []],
            batch_excluded={str(key): str(value) for key, value in (payload.get("batch_excluded") or {}).items()},
            guided_started_used_sec=(
                None if payload.get("guided_started_used_sec") is None else float(payload["guided_started_used_sec"])
            ),
            guided_rounds=int(payload.get("guided_rounds") or 0),
        )

    def adopt(self, playbook: Playbook) -> None:
        """Start the ledger for a freshly built playbook."""
        self.playbook = playbook
        self.playbook_settled = True
        self.playbook_error = ""
        self.ledger = {item_id: ItemRecord() for item_id in playbook.item_ids()}
        self.group_order = []
        self.rank_evidence = {}
        self.ranked_at_iteration = 0
        self.batch = []
        self.batch_excluded = {}

    def open_group_ids(self) -> list[str]:
        """The groups that still have a PENDING or FAILED item, in the builder's order."""
        if self.playbook is None:
            return []
        open_items = {
            item_id for item_id, record in self.ledger.items() if record.status in {STATUS_PENDING, STATUS_FAILED}
        }
        return [
            group.group_id
            for group in self.playbook.groups
            if any(item.group == group.group_id and item.item_id in open_items for item in self.playbook.items)
        ]

    def ordered_groups(self) -> list[PlaybookGroup]:
        """The playbook's groups, the latest ranking's first and the rest in the builder's order."""
        if self.playbook is None:
            return []
        by_id = {group.group_id: group for group in self.playbook.groups}
        ranked = [by_id[group_id] for group_id in self.group_order if group_id in by_id]
        return ranked + [group for group in self.playbook.groups if group.group_id not in self.group_order]

    def open_batch(self) -> list[str]:
        """Every open group in the current order, at most one batch's worth: the batch when no ranking chose one."""
        open_ids = set(self.open_group_ids())
        return [group.group_id for group in self.ordered_groups() if group.group_id in open_ids][:MAX_BATCH_STEPS]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.ledger.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        return dict(sorted(counts.items()))


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return ()
    return tuple(dict.fromkeys(text for text in (_text(entry) for entry in value) if text))


def _playbook_from_payload(payload: Mapping[str, Any]) -> tuple[Playbook | None, list[str]]:
    notes: list[str] = []
    groups: list[PlaybookGroup] = []
    for entry in payload.get("groups") or []:
        if not isinstance(entry, Mapping):
            notes.append("dropped a group that was not an object")
            continue
        group_id = _text(entry.get("id")).upper()
        title = _text(entry.get("title"))
        if not _GROUP_ID.match(group_id) or not title:
            notes.append(f"dropped group {group_id or '?'}: it needs an id like G1 and a title")
            continue
        if any(group.group_id == group_id for group in groups):
            notes.append(f"dropped a second group {group_id}")
            continue
        groups.append(PlaybookGroup(group_id=group_id, title=title, why_together=_text(entry.get("why_together"))))
    known_groups = {group.group_id for group in groups}

    items: list[PlaybookItem] = []
    for entry in payload.get("items") or []:
        if not isinstance(entry, Mapping):
            notes.append("dropped an item that was not an object")
            continue
        item_id = _text(entry.get("id")).upper()
        title = _text(entry.get("title"))
        group = _text(entry.get("group")).upper()
        pages = _strings(entry.get("wiki_pages"))
        if not _ITEM_ID.match(item_id) or not title:
            notes.append(f"dropped item {item_id or '?'}: it needs an id like P1 and a title")
            continue
        if any(item.item_id == item_id for item in items):
            notes.append(f"dropped a second item {item_id}")
            continue
        if group not in known_groups:
            notes.append(f"dropped item {item_id}: its group {group or '?'} is not one of the playbook's groups")
            continue
        if not pages:
            # The playbook is the knowledge base's contribution; an item no page supports is the builder's own idea.
            notes.append(f"dropped item {item_id}: it cites no knowledge base page")
            continue
        items.append(
            PlaybookItem(
                item_id=item_id,
                title=title,
                group=group,
                wiki_pages=pages,
                applies_when=_text(entry.get("applies_when")),
                cases=_text(entry.get("cases")),
                expected_effect=_text(entry.get("expected_effect")),
                verify=_text(entry.get("verify")),
                conflicts_with=tuple(item.upper() for item in _strings(entry.get("conflicts_with"))),
            )
        )
        if len(items) == MAX_ITEMS:
            if len(payload.get("items") or []) > MAX_ITEMS:
                notes.append(f"kept the first {MAX_ITEMS} items")
            break
    known_items = {item.item_id for item in items}
    items = [
        replace(item, conflicts_with=tuple(other for other in item.conflicts_with if other in known_items))
        for item in items
    ]
    used_groups = {item.group for item in items}
    groups = [group for group in groups if group.group_id in used_groups]
    if not items:
        notes.append("the playbook has no usable item")
        return None, notes
    return Playbook(items=tuple(items), groups=tuple(groups), summary=_text(payload.get("summary"))), notes


def parse_playbook(text: str) -> tuple[Playbook | None, list[str]]:
    """Read the builder's answer; the playbook, or ``None`` and why."""
    try:
        payload = extract_json_object(text or "", "wiki playbook")
    except ValueError as error:
        return None, [str(error)]
    return _playbook_from_payload(payload)


@dataclass(frozen=True)
class Ranking:
    """The ranker's batch: open groups in step order, what put each there, and why the others were left out."""

    batch: tuple[str, ...]
    evidence: Mapping[str, str]
    excluded: Mapping[str, str]
    notes: tuple[str, ...] = ()


def parse_ranking(text: str, open_groups: Sequence[str]) -> Ranking:
    """Read the ranker's answer against the round's open groups.

    Raises ``ValueError`` when the answer puts no open group in the batch.
    """
    payload = extract_json_object(text or "", "wiki playbook ranking")
    known = set(open_groups)
    batch: list[str] = []
    evidence: dict[str, str] = {}
    excluded: dict[str, str] = {}
    notes: list[str] = []
    for entry in payload.get("batch") or []:
        if not isinstance(entry, Mapping):
            notes.append("dropped a batch entry that was not an object")
            continue
        group_id = _text(entry.get("group")).upper()
        if group_id not in known:
            notes.append(f"dropped {group_id or '?'}: it is not an open group")
            continue
        if group_id in batch:
            notes.append(f"dropped a second {group_id}")
            continue
        if len(batch) == MAX_BATCH_STEPS:
            notes.append(f"dropped {group_id}: a batch runs at most {MAX_BATCH_STEPS} steps")
            continue
        batch.append(group_id)
        parts = (
            ("cases", _text(entry.get("cases"))),
            ("expected gain", _text(entry.get("expected_gain"))),
            ("evidence", _text(entry.get("evidence"))),
        )
        reason = "; ".join(f"{label}: {value}" for label, value in parts if value)
        if reason:
            evidence[group_id] = reason
    for entry in payload.get("excluded") or []:
        if not isinstance(entry, Mapping):
            continue
        group_id = _text(entry.get("group")).upper()
        if group_id in known and group_id not in batch:
            excluded[group_id] = _text(entry.get("reason")) or "no reason given"
    if not batch:
        raise ValueError("the ranking puts no open group in the batch")
    return Ranking(batch=tuple(batch), evidence=evidence, excluded=excluded, notes=tuple(notes))


def _ids(blocks: Iterable[str], known: set[str]) -> list[str]:
    found: list[str] = []
    for block in blocks:
        for match in _CITED_ID.findall(block):
            item_id = match.upper()
            if item_id in known and item_id not in found:
                found.append(item_id)
    return found


def cited_items(plan_text: str, known: Iterable[str]) -> tuple[list[str], list[str]]:
    """The items a plan claims and the ones it declines, among ``known``."""
    known_ids = set(known)
    text = plan_text or ""
    skipped = _ids(_SKIP_CITATION.findall(text), known_ids)
    claimed = [item_id for item_id in _ids(_CITATION.findall(text), known_ids) if item_id not in skipped]
    return claimed, skipped


def record_outcome(state: GuidanceState, *, iteration: int, plan_text: str, decision: str) -> list[str]:
    """Credit one measured iteration to the items its plan cited; the item ids that changed."""
    if state.playbook is None or iteration in state.recorded_iterations:
        return []
    label = str(decision or "").strip().upper()
    if label not in APPLIED_DECISIONS | REFUTED_DECISIONS | FAILED_DECISIONS:
        # Infrastructure, or a session that measured nothing: no evidence about any item.
        return []
    state.recorded_iterations.append(iteration)
    claimed, skipped = cited_items(plan_text, state.ledger)
    changed: list[str] = []
    for item_id in skipped:
        record = state.ledger[item_id]
        if record.status in {STATUS_PENDING, STATUS_FAILED}:
            record.status = STATUS_SKIPPED
            record.iterations.append(iteration)
            record.last_decision = label
            changed.append(item_id)
    for item_id in claimed:
        record = state.ledger[item_id]
        before = record.status
        record.iterations.append(iteration)
        record.last_decision = label
        if label in APPLIED_DECISIONS:
            record.status = STATUS_APPLIED
        elif label in REFUTED_DECISIONS:
            # A later plan re-citing an item already in the incumbent and losing does not take it out again.
            if record.status != STATUS_APPLIED:
                record.status = STATUS_REFUTED
        else:
            record.failures += 1
            if record.status in {STATUS_PENDING, STATUS_FAILED}:
                record.status = STATUS_FAILED if record.failures < MAX_FAILED_ATTEMPTS else STATUS_REFUTED
        if record.status != before or label in FAILED_DECISIONS:
            changed.append(item_id)
    return changed


def exhausted(state: GuidanceState) -> bool:
    """Whether every playbook item is settled."""
    return state.playbook is not None and all(record.status in DONE_STATUSES for record in state.ledger.values())


def decide_phase(
    state: GuidanceState,
    *,
    search_reasons: Sequence[str],
    budget_used_fraction: float,
    budget_fraction: float,
) -> tuple[str, tuple[str, ...]] | None:
    """The phase to enter before this iteration plans, or ``None`` to stay.

    ``budget_used_fraction`` is GUIDED's own share: of the budget left when the first GUIDED round began, the part
    used since.
    """
    if state.phase != PHASE_GUIDED:
        return None
    if state.playbook_settled and state.playbook is None:
        return PHASE_FREE, (REASON_PLAYBOOK_UNAVAILABLE,)
    if exhausted(state):
        return PHASE_FREE, (REASON_PLAYBOOK_EXHAUSTED,)
    if state.guided_rounds >= MIN_GUIDED_ROUNDS and budget_used_fraction >= budget_fraction:
        return PHASE_FREE, (REASON_GUIDED_BUDGET_SPENT,)
    stall = tuple(sorted(set(search_reasons) & STALL_REASONS))
    if stall and state.playbook is not None:
        return PHASE_FREE, (REASON_GUIDED_STALL, *stall)
    return None


def render_for_planning(state: GuidanceState) -> dict[str, Any] | None:
    """What every planning role sees as ``context.wiki_guidance``; ``None`` when there is no playbook to show."""
    if state.playbook is None:
        return None
    instructions = (
        GUIDED_INSTRUCTIONS
        if state.phase == PHASE_GUIDED
        else FREE_INSTRUCTIONS.format(reasons=", ".join(state.reason_codes) or "unrecorded")
    )
    items = []
    for item in state.playbook.items:
        record = state.ledger.get(item.item_id, ItemRecord())
        items.append(
            {
                **item.to_dict(),
                "status": record.status,
                "failures": record.failures,
                "iterations": list(record.iterations),
            }
        )
    groups = []
    for group in state.ordered_groups():
        entry = {
            **group.to_dict(),
            "items": [item.item_id for item in state.playbook.items if item.group == group.group_id],
        }
        if group.group_id in state.rank_evidence:
            entry["rank_evidence"] = state.rank_evidence[group.group_id]
        groups.append(entry)
    rendered = {
        "phase": state.phase,
        "reason_codes": list(state.reason_codes),
        "instructions": instructions,
        "playbook": {"summary": state.playbook.summary, "groups": groups, "items": items},
    }
    if state.ranked_at_iteration:
        rendered["ranked_at_iteration"] = state.ranked_at_iteration
    if state.phase == PHASE_GUIDED and state.batch:
        rendered["batch"] = list(state.batch)
        if state.batch_excluded:
            rendered["batch_excluded"] = dict(state.batch_excluded)
    return rendered


def render_for_implementer(state: GuidanceState) -> str:
    """The Implementer's short note on the phase; empty when there is no playbook."""
    if state.playbook is None:
        return ""
    if state.phase == PHASE_GUIDED:
        return (
            "## Wiki guidance\n"
            "Phase: GUIDED. The plan implements knowledge base playbook items it cites as [playbook:...]. Implement "
            "every cited item; when an item's detail matters, read its knowledge base pages with the knowledge tools. "
            "A plan headed as a batch step is one of several run one after another on this kernel: the tree already "
            "carries the earlier steps that were kept, and this step is measured on its own when it ends."
        )
    return (
        "## Wiki guidance\n"
        f"Phase: FREE ({', '.join(state.reason_codes) or 'unrecorded'}). The playbook is spent; the plan may go "
        "beyond it."
    )


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
        from kernelforge.loop.recovery import atomic_write_json

        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, state.to_dict())
        if state.playbook is not None:
            atomic_write_json(self.playbook_path, state.playbook.to_dict())

    def save_response(self, text: str) -> None:
        from kernelforge.loop.recovery import atomic_write_text

        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.response_path, text or "")
