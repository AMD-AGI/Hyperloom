# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base-guided planning: each GUIDED round's batch, the ledger, and the phase the campaign plans in.

Off unless the campaign runs with ``--wiki-guided``. When on, every GUIDED planning round first asks a read-only
builder to turn the knowledge base and that round's measured evidence into the round's batch: every direction the
evidence supports now, grouped so that one group edits one piece of code, the groups in the order to implement them.
Each item cites the knowledge base pages behind it, or is marked as supported by the profiling evidence alone when the
evidence ranks it among the largest levers left and no page covers it. The whole batch is one iteration: the round
plans one part per group, one Implementer session implements every part on the same kernel, and the result is
measured once; the next round then plans from the profile taken after it. A GUIDED round runs no plan critic: its
parts are not competing lanes, so none is narrowed away. Each round's batch is added to one campaign playbook under
fresh ids, so the builder sees what earlier batches settled, and why, and does not propose it again.

The campaign leaves GUIDED for FREE when the builder finds nothing more the knowledge base and the evidence support
(``NO_USEFUL_DIRECTION``), when a whole batch is measured without a gain (``BATCH_NO_GAIN``) or with less than
``MIN_BATCH_GAIN`` (``BATCH_LOW_GAIN``), or when no playbook can be had (``PLAYBOOK_UNAVAILABLE``); from then on the
planners see what the playbook settled and plan freely.

A plan claims playbook items by citing them as ``[playbook:P1,P3]`` and declines one as ``[playbook-skip:P4]``. A
batch session ends by citing the items that are in its final kernel, the ones it tried and dropped, and the ones it did
not get to (``[playbook-untried:P5]``, which stay open), and the ledger is credited from that report when there is one,
else from the plan; either way a claim counts only once a measurement decided it. A session the stop hook sends back
reports again when it submits again, and only its latest report says what is in the kernel that was measured.
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
REASON_NO_USEFUL_DIRECTION = "NO_USEFUL_DIRECTION"
REASON_BATCH_NO_GAIN = "BATCH_NO_GAIN"
REASON_BATCH_LOW_GAIN = "BATCH_LOW_GAIN"

# A KEEP batch that raises the incumbent's mean case speedup by less than this fraction ends GUIDED.
MIN_BATCH_GAIN = 0.02

# What an item rests on: knowledge base pages, or the round's profiling evidence alone.
SUPPORT_WIKI = "wiki"
SUPPORT_PROFILING = "profiling"
SUPPORTS = frozenset({SUPPORT_WIKI, SUPPORT_PROFILING})

STATUS_PENDING = "PENDING"
STATUS_APPLIED = "APPLIED"
STATUS_REFUTED = "REFUTED"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
DONE_STATUSES = frozenset({STATUS_APPLIED, STATUS_REFUTED, STATUS_SKIPPED})
ITEM_STATUSES = DONE_STATUSES | {STATUS_PENDING, STATUS_FAILED}

# A FAILED item may be retried; the second failure settles it as REFUTED.
MAX_FAILED_ATTEMPTS = 2
# Consecutive outages on the builder's backend before the campaign plans without it.
MAX_PLAYBOOK_ATTEMPTS = 2
# What one build may add: items, and the groups one batch session implements.
MAX_ITEMS = 12
MAX_BATCH_GROUPS = 6

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

# The builder numbers its own answer from 1; the campaign playbook renumbers each build after the ones before it.
_ITEM_ID = re.compile(r"^P\d{1,3}$")
_GROUP_ID = re.compile(r"^G\d{1,3}$")
_ID_NUMBER = re.compile(r"\d+$")
_CITATION = re.compile(r"\[playbook:([^\]]*)\]", re.IGNORECASE)
_SKIP_CITATION = re.compile(r"\[playbook-skip:([^\]]*)\]", re.IGNORECASE)
_UNTRIED_CITATION = re.compile(r"\[playbook-untried:([^\]]*)\]", re.IGNORECASE)
_CITED_ID = re.compile(r"\bP\d{1,3}\b", re.IGNORECASE)
# Inside one citation, an entry is a run of item ids at the start of the block or after a `;`, `,` or line break,
# followed by its reason; an id that only appears inside another entry's reason is not cited. In a skip or untried
# block, after a comma the run opens an entry only when its reason starts right after it ("P6 slower, P7: spilled"):
# in "slower than the split path, P21 would be needed first" P21 is part of P6's reason.
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

GUIDED_INSTRUCTIONS = (
    "GUIDED phase. `playbook` lists the techniques the knowledge base supports for this kernel, built round by round "
    "from the knowledge base and the profile, each item with the status its measurement settled. `batch` is this "
    "round's groups, in the order to implement them: every direction this round's evidence supports. Plan from it:\n"
    "1. The whole batch is one iteration. One Implementer session implements every group in `batch`, in order, on "
    "the same kernel, and the result is measured once when the session ends; the profile is refreshed after that "
    "measurement and the next round builds the next batch from it. Each group is one part of that session, "
    "implemented whole; a group is never spread over two parts.\n"
    "2. Plan each part so that the session can keep or drop it on its own: it must stay correct, and say how it "
    "adapts, when an earlier part was dropped.\n"
    "3. On its own line, cite every item a plan implements as [playbook:P1,P3], and every item the evidence shows "
    "does not apply as [playbook-skip:P4 followed by that evidence]. Work a plan does not cite is not credited to "
    "the playbook.\n"
    "4. Adapt each item to this kernel's evidence (profile, cases, code). Its knowledge base pages are available "
    "through the knowledge tools; read them before planning the item. An item whose `support` is `profiling` has "
    "no page: the round's evidence ranks it among the largest levers left, `applies_when` says which measurement, "
    "and it is planned from that evidence.\n"
    "5. Do not re-plan REFUTED or SKIPPED items; an item's `note`, when it has one, is what the session reported "
    "about it. A FAILED item may be retried once, with the failure fixed.\n"
    "6. This round's analyses are one per batch group (role `playbook_group`, the group named in each assignment), "
    "each covering its group whole across compute, memory and algorithm. Build each part from its group's analysis; "
    "an analyst of another role, when there is one, analyzes how this round's playbook items apply to this kernel "
    "rather than surveying it broadly.\n"
    "Add work outside the playbook only when the evidence shows a cited item cannot pay off without it."
)

FREE_INSTRUCTIONS = (
    "FREE phase: the playbook phase has ended ({reasons}). APPLIED items are already in the incumbent; REFUTED "
    "items were measured and did not help in the form tried; SKIPPED items did not apply or were dropped. Optimize "
    "beyond the playbook with your own knowledge and the evidence. The knowledge base stays available, but it is no "
    "longer the source of the plan. Re-try a REFUTED item only when new evidence changes its premise, and say what "
    "changed."
)

BATCH_SESSION_INSTRUCTIONS = (
    "This plan is one GUIDED batch: {parts} part(s), each one playbook group, implemented in this one session on "
    "the same kernel and measured once, when the session ends. Implement the parts in the order given. After each "
    "part, run the correctness check and time the cases: keep the part when it is correct and not slower than the "
    "kernel was before it, and otherwise revert that part's edits and go on to the next part. Submit only after the "
    "last part. End the session with these lines of its own, each entry an item id followed by why, entries "
    "separated by `;`: [playbook:...] naming every item whose change is in the final kernel; [playbook-skip:...] "
    "naming every item you implemented and dropped or reverted, or whose precondition proved false; and "
    "[playbook-untried:...] naming every item you did not get to or did not finish, which stays open for a later "
    "batch."
)


@dataclass(frozen=True)
class PlaybookItem:
    """One direction for this kernel: a technique the knowledge base supports, or one only the evidence ranks."""

    item_id: str
    title: str
    group: str
    wiki_pages: tuple[str, ...]
    applies_when: str = ""
    cases: str = ""
    expected_effect: str = ""
    verify: str = ""
    conflicts_with: tuple[str, ...] = ()
    support: str = SUPPORT_WIKI

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "title": self.title,
            "group": self.group,
            "support": self.support,
            "wiki_pages": list(self.wiki_pages),
            "applies_when": self.applies_when,
            "cases": self.cases,
            "expected_effect": self.expected_effect,
            "verify": self.verify,
            "conflicts_with": list(self.conflicts_with),
        }


@dataclass(frozen=True)
class PlaybookGroup:
    """Items that edit the same code and are implemented as one part of a batch."""

    group_id: str
    title: str
    why_together: str = ""
    # The iteration whose GUIDED round built this group; 0 for a builder answer not yet added to a playbook.
    built_at_iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.group_id, "title": self.title, "why_together": self.why_together}
        if self.built_at_iteration:
            payload["built_at_iteration"] = self.built_at_iteration
        return payload


@dataclass(frozen=True)
class Playbook:
    """Groups and items, in the order they were built; within one build, groups in implementation order."""

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
        playbook, notes = _playbook_from_payload(payload, max_items=None, max_groups=None)
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
    # What the latest session that skipped or did not get to the item said about it.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "failures": self.failures,
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
            failures=int(payload.get("failures") or 0),
            iterations=[int(value) for value in payload.get("iterations") or []],
            last_decision=str(payload.get("last_decision") or ""),
            note=str(payload.get("note") or ""),
        )


@dataclass
class GuidanceState:
    """The durable state of one campaign's guided planning."""

    phase: str = PHASE_GUIDED
    reason_codes: list[str] = field(default_factory=list)
    playbook: Playbook | None = None
    ledger: dict[str, ItemRecord] = field(default_factory=dict)
    # Builds added to the playbook, and outages on the builder's backend since the last answer.
    builds: int = 0
    build_failures: int = 0
    playbook_error: str = ""
    # The latest GUIDED round's batch: the groups it built, in implementation order.
    batch: list[str] = field(default_factory=list)
    phase_changed_at_iteration: int = 0
    # Iterations whose outcome the ledger already holds, so a recovery that replays one cannot count it twice.
    recorded_iterations: list[int] = field(default_factory=list)
    # The plan the latest iteration is measuring, kept here so a KEEP finished by resume recovery is still credited.
    inflight_iteration: int = 0
    inflight_plan: str = ""
    # The incumbent's mean case speedup when that iteration began, which its batch's gain is measured against.
    inflight_best_before: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reason_codes": list(self.reason_codes),
            "playbook": self.playbook.to_dict() if self.playbook is not None else None,
            "ledger": {item_id: record.to_dict() for item_id, record in self.ledger.items()},
            "builds": self.builds,
            "build_failures": self.build_failures,
            "playbook_error": self.playbook_error,
            "batch": list(self.batch),
            "phase_changed_at_iteration": self.phase_changed_at_iteration,
            "recorded_iterations": list(self.recorded_iterations),
            "inflight_iteration": self.inflight_iteration,
            "inflight_plan": self.inflight_plan,
            "inflight_best_before": self.inflight_best_before,
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
            builds=int(payload.get("builds") or 0),
            build_failures=int(payload.get("build_failures") or 0),
            playbook_error=str(payload.get("playbook_error") or ""),
            batch=[str(group_id) for group_id in payload.get("batch") or []],
            phase_changed_at_iteration=int(payload.get("phase_changed_at_iteration") or 0),
            recorded_iterations=[int(value) for value in payload.get("recorded_iterations") or []],
            inflight_iteration=int(payload.get("inflight_iteration") or 0),
            inflight_plan=str(payload.get("inflight_plan") or ""),
            inflight_best_before=float(payload.get("inflight_best_before") or 0.0),
        )

    def add_build(self, build: Playbook, *, iteration: int) -> list[str]:
        """Add one builder answer under fresh ids and make its groups the batch; the batch's group ids."""
        existing = self.playbook
        item_base = max((int(item_id[1:]) for item_id in (existing.item_ids() if existing else ())), default=0)
        group_base = max((int(group.group_id[1:]) for group in (existing.groups if existing else ())), default=0)
        item_ids = {item.item_id: f"P{item_base + index}" for index, item in enumerate(build.items, start=1)}
        group_ids = {group.group_id: f"G{group_base + index}" for index, group in enumerate(build.groups, start=1)}
        items = tuple(
            replace(
                item,
                item_id=item_ids[item.item_id],
                group=group_ids[item.group],
                conflicts_with=tuple(item_ids[other] for other in item.conflicts_with if other in item_ids),
            )
            for item in build.items
        )
        groups = tuple(
            replace(group, group_id=group_ids[group.group_id], built_at_iteration=iteration) for group in build.groups
        )
        self.playbook = Playbook(
            items=(existing.items if existing else ()) + items,
            groups=(existing.groups if existing else ()) + groups,
            summary=build.summary,
        )
        self.ledger.update({item.item_id: ItemRecord() for item in items})
        self.batch = [group.group_id for group in groups]
        self.builds += 1
        self.build_failures = 0
        self.playbook_error = ""
        return list(self.batch)

    def ordered_groups(self) -> list[PlaybookGroup]:
        """The playbook's groups, the batch's first in its order and the earlier builds' after them."""
        if self.playbook is None:
            return []
        by_id = {group.group_id: group for group in self.playbook.groups}
        batch = [by_id[group_id] for group_id in self.batch if group_id in by_id]
        return batch + [group for group in self.playbook.groups if group.group_id not in self.batch]

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


def _playbook_from_payload(
    payload: Mapping[str, Any],
    *,
    max_items: int | None = MAX_ITEMS,
    max_groups: int | None = MAX_BATCH_GROUPS,
) -> tuple[Playbook | None, list[str]]:
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
        if max_groups is not None and len(groups) == max_groups:
            notes.append(f"dropped group {group_id}: a batch implements at most {max_groups} groups")
            continue
        groups.append(
            PlaybookGroup(
                group_id=group_id,
                title=title,
                why_together=_text(entry.get("why_together")),
                built_at_iteration=int(entry.get("built_at_iteration") or 0),
            )
        )
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
        support = _text(entry.get("support")).lower() or SUPPORT_WIKI
        applies_when = _text(entry.get("applies_when"))
        if not _ITEM_ID.match(item_id) or not title:
            notes.append(f"dropped item {item_id or '?'}: it needs an id like P1 and a title")
            continue
        if any(item.item_id == item_id for item in items):
            notes.append(f"dropped a second item {item_id}")
            continue
        if group not in known_groups:
            notes.append(f"dropped item {item_id}: its group {group or '?'} is not one of the playbook's groups")
            continue
        if support not in SUPPORTS:
            notes.append(f"dropped item {item_id}: its support {support!r} is neither wiki nor profiling")
            continue
        if pages:
            # A page that supports it makes it a knowledge base item, whatever the answer called it.
            support = SUPPORT_WIKI
        elif support != SUPPORT_PROFILING:
            notes.append(f"dropped item {item_id}: it cites no knowledge base page and is not marked profiling")
            continue
        elif not applies_when:
            notes.append(f"dropped item {item_id}: a profiling item must name the evidence behind it in applies_when")
            continue
        items.append(
            PlaybookItem(
                item_id=item_id,
                title=title,
                group=group,
                wiki_pages=pages,
                applies_when=applies_when,
                cases=_text(entry.get("cases")),
                expected_effect=_text(entry.get("expected_effect")),
                verify=_text(entry.get("verify")),
                conflicts_with=tuple(item.upper() for item in _strings(entry.get("conflicts_with"))),
                support=support,
            )
        )
        if max_items is not None and len(items) == max_items:
            if len(payload.get("items") or []) > max_items:
                notes.append(f"kept the first {max_items} items")
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


def _renumbered(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``payload`` with every titled group and item whose id is missing or not shaped like G1 / P1 given the next free
    one, and the references to it rewritten: an answer that numbered its items N1, N2 is still a usable build."""
    result = dict(payload)
    notes: list[str] = []
    maps: dict[str, dict[str, str]] = {}
    for key, pattern, prefix in (("groups", _GROUP_ID, "G"), ("items", _ITEM_ID, "P")):
        entries = list(payload.get(key) or [])
        ids = [_text(entry.get("id")).upper() if isinstance(entry, Mapping) else "" for entry in entries]
        number = max((int(_ID_NUMBER.search(value).group()) for value in ids if pattern.match(value)), default=0)
        mapping: dict[str, str] = {}
        fixed: list[Any] = []
        for entry, old in zip(entries, ids):
            if isinstance(entry, Mapping) and _text(entry.get("title")) and not pattern.match(old):
                number += 1
                new = f"{prefix}{number}"
                if old:
                    mapping.setdefault(old, new)
                notes.append(f"renumbered {key[:-1]} {old or '(no id)'} as {new}")
                entry = {**entry, "id": new}
            fixed.append(entry)
        result[key] = fixed
        maps[key] = mapping
    if maps["groups"] or maps["items"]:
        result["items"] = [
            {
                **entry,
                "group": maps["groups"].get(_text(entry.get("group")).upper(), entry.get("group")),
                "conflicts_with": [
                    maps["items"].get(other.upper(), other) for other in _strings(entry.get("conflicts_with"))
                ],
            }
            if isinstance(entry, Mapping)
            else entry
            for entry in result["items"]
        ]
    return result, notes


def parse_playbook(text: str) -> tuple[Playbook | None, list[str]]:
    """Read one builder answer; the round's build, or ``None`` and why."""
    try:
        payload = extract_json_object(text or "", "wiki playbook")
    except ValueError as error:
        return None, [str(error)]
    payload, renumbered = _renumbered(payload)
    build, notes = _playbook_from_payload(payload)
    return build, [*renumbered, *notes]


def declined(text: str) -> bool:
    """Whether a builder answer says, in its own shape, that nothing more is worth building."""
    try:
        payload = extract_json_object(text or "", "wiki playbook")
    except ValueError:
        return False
    return not (payload.get("items") or payload.get("groups"))


@dataclass(frozen=True)
class Citations:
    """What a plan or a session report cites: the items it claims, skips and did not try, and why for the last two."""

    claimed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    untried: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)


def _entries(block: str, *, strict: bool) -> list[tuple[list[str], str]]:
    """Each entry of one citation block: the item ids it opens with, and the reason that follows them. ``strict``
    applies the comma rule above; a claim block lists what it kept, often with a description after each id ("P18
    bf16x3 tile, P19 per-shape dispatch"), so there an id after a comma always opens an entry."""
    opened: list[tuple[re.Match[str], re.Match[str]]] = []
    cursor = 0
    for start in _ENTRY_START.finditer(block):
        if start.end() < cursor:
            continue
        run = _ID_RUN.match(block, start.end())
        if run is None:
            continue
        if strict and block[start.start()] == "," and not _REASON_OPENER.match(block, run.end()):
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


def _cited(pattern: re.Pattern[str], text: str, known: set[str], *, strict: bool) -> tuple[list[str], dict[str, str]]:
    found: list[str] = []
    reasons: dict[str, str] = {}
    for block in pattern.findall(text):
        for item_ids, reason in _entries(block, strict=strict):
            for item_id in item_ids:
                if item_id in known and item_id not in found:
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
    # A skip read into a reason settles an item for good, so skips and untried items take the strict comma rule; a
    # claim missed costs the item its credit, so claims do not.
    claimed, _ = _cited(_CITATION, text, known_ids, strict=False)
    skipped, skip_reasons = _cited(_SKIP_CITATION, text, known_ids, strict=True)
    untried, untried_reasons = _cited(_UNTRIED_CITATION, text, known_ids, strict=True)
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


def record_outcome(state: GuidanceState, *, iteration: int, plan_text: str, decision: str) -> list[str]:
    """Credit one measured iteration to the items its plan cited; the item ids that changed."""
    if state.playbook is None or iteration in state.recorded_iterations:
        return []
    label = str(decision or "").strip().upper()
    if label not in APPLIED_DECISIONS | REFUTED_DECISIONS | FAILED_DECISIONS:
        # Infrastructure, or a session that measured nothing: no evidence about any item.
        return []
    state.recorded_iterations.append(iteration)
    cites = read_citations(plan_text, state.ledger)
    changed: list[str] = []
    for item_id in cites.skipped:
        record = state.ledger[item_id]
        if record.status in {STATUS_PENDING, STATUS_FAILED}:
            record.status = STATUS_SKIPPED
            record.iterations.append(iteration)
            record.last_decision = label
            record.note = f"skipped in iteration {iteration}: {cites.reasons.get(item_id) or 'no reason given'}"
            changed.append(item_id)
    for item_id in cites.untried:
        # Not tried is no evidence about the item: it stays open, and a later batch may propose it again.
        record = state.ledger[item_id]
        if record.status in {STATUS_PENDING, STATUS_FAILED}:
            record.iterations.append(iteration)
            record.last_decision = label
            record.note = f"not tried in iteration {iteration}: {cites.reasons.get(item_id) or 'no reason given'}"
            changed.append(item_id)
    for item_id in cites.claimed:
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


def batch_gain(best_before: float, mean_case_speedup: float | None) -> float | None:
    """A batch's gain as a fraction of the incumbent it began from; ``None`` when either side is unknown."""
    if not best_before or best_before <= 0 or not mean_case_speedup or mean_case_speedup <= 0:
        return None
    return mean_case_speedup / best_before - 1.0


def batch_exit(decision: str, gain: float | None = None) -> str | None:
    """The reason a measured batch ends GUIDED, or ``None``: a KEEP worth ``MIN_BATCH_GAIN``, or nothing measured."""
    label = str(decision or "").strip().upper()
    if label in REFUTED_DECISIONS | FAILED_DECISIONS:
        return REASON_BATCH_NO_GAIN
    if label in APPLIED_DECISIONS and gain is not None and gain < MIN_BATCH_GAIN:
        return REASON_BATCH_LOW_GAIN
    return None


def batch_plan(parts: Sequence[tuple[str, str]]) -> str:
    """One session's plan for a batch: the instructions, then each (label, plan) part in implementation order."""
    sections = [
        f"# GUIDED batch: {len(parts)} part(s) in one session",
        BATCH_SESSION_INSTRUCTIONS.format(parts=len(parts)),
    ]
    for index, (label, plan) in enumerate(parts, start=1):
        heading = f"## Part {index} of {len(parts)}" + (f": {label}" if label else "")
        sections.append(f"{heading}\n\n{plan.strip()}")
    return "\n\n".join(sections) + "\n"


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
                **({"note": record.note} if record.note else {}),
            }
        )
    groups = [
        {
            **group.to_dict(),
            "items": [item.item_id for item in state.playbook.items if item.group == group.group_id],
        }
        for group in state.ordered_groups()
    ]
    rendered = {
        "phase": state.phase,
        "reason_codes": list(state.reason_codes),
        "instructions": instructions,
        "playbook": {"summary": state.playbook.summary, "groups": groups, "items": items},
    }
    if state.phase == PHASE_GUIDED and state.batch:
        rendered["batch"] = list(state.batch)
    return rendered


def render_for_implementer(state: GuidanceState) -> str:
    """The Implementer's short note on the phase; empty when there is no playbook."""
    if state.playbook is None:
        return ""
    if state.phase == PHASE_GUIDED:
        return (
            "## Wiki guidance\n"
            "Phase: GUIDED. The plan implements knowledge base playbook items it cites as [playbook:...]. Implement "
            "every cited item; when an item's detail matters, read its knowledge base pages with the knowledge tools "
            "(an item supported by profiling alone has none, and the plan states its evidence). "
            "A plan headed as a GUIDED batch is implemented whole in this session and measured once at its end: "
            "follow its instructions for keeping or dropping each part and for the lines that end the session."
        )
    return (
        "## Wiki guidance\n"
        f"Phase: FREE ({', '.join(state.reason_codes) or 'unrecorded'}). The knowledge base no longer supplies the "
        "plan; the plan may go beyond the playbook."
    )


class WikiGuidanceStore:
    """The campaign's guidance state and playbook, under ``forge_experiments/wiki_guidance``."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.root = Path(workspace_dir).resolve() / "forge_experiments" / "wiki_guidance"
        self.state_path = self.root / "state.json"
        self.playbook_path = self.root / "playbook.json"

    def response_path(self, build: int) -> Path:
        """Where the builder's raw answer for one build (counted from 1) is kept."""
        return self.root / f"playbook_response_{build:03d}.md"

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

    def save_response(self, text: str, *, build: int) -> None:
        from kernelforge.loop.recovery import atomic_write_text

        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.response_path(build), text or "")
