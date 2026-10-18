# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist pulls from the notebook: its branch's inbox, record search, and profile records.

The inbox holds pointers only. A cursor is the last sequence number seen of every writer, encoded opaquely, so a poll
returns each entry once however the writers' clocks interleave.

Given :class:`BriefTriggers`, the inbox also holds ``brief`` items, pointing at the branch's head and asking its
specialist to read its brief again before the next claim. They are derived from the entries as the other items are,
so a poll returns each once: when a judgement brings the branch's tests judged since its last commit to
``stall_tests``, once a stall, and when another branch's pitfall learning names a direction the branch holds a claim
on.

Search: ``,`` separates alternatives and ``&`` joins terms that must all appear, so ``fp8&kv,attention`` matches an
entry mentioning both ``fp8`` and ``kv``, or one mentioning ``attention``. Matching is case-insensitive substring search
over an entry's kind, branch, actor, summary and stored detail.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook.peers import TestKind
from hyperloom_multibranch.contracts.notebook.records import (
    Claim,
    InboxItem,
    InboxPage,
    Learning,
    LineageEvent,
    LineageView,
    SearchHit,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import InboxKind, LearningKind
from hyperloom_multibranch.notebook.learning_search import names
from hyperloom_multibranch.notebook.local.segments import Entry, EntryKind
from hyperloom_multibranch.notebook.local.state import JudgementWrite, NotebookState
from hyperloom_multibranch.notebook.local.views import judged_since_commit, lineage_view

#: How many tests judged since the last commit make a stall, unless the run entry names another number.
STALL_TESTS = 4
_REREAD = "read notebook_brief from {head} and reconsider the direction before the next claim"


@dataclass(frozen=True)
class BriefTriggers:
    """When a running specialist's inbox gets a ``brief`` item: its branch's ``stall_tests`` tests judged since its
    last commit, and, with ``pitfalls``, another branch's pitfall learning in a direction it holds a claim on."""

    stall_tests: int = STALL_TESTS
    pitfalls: bool = True

    def __post_init__(self) -> None:
        if self.stall_tests < 1:
            raise ValueError(f"a stall takes at least one judged test, not {self.stall_tests}")


@dataclass(frozen=True)
class _Briefing:
    triggers: BriefTriggers
    held: tuple[Claim, ...]
    records: tuple[TestRecord, ...]
    view: LineageView

    def head(self, branch: str) -> str:
        node = next((item for item in self.view.branches if item.branch == branch), None)
        return node.head_commit if node is not None else ""


_INBOX_KINDS = {
    EntryKind.RESULT: InboxKind.RESULT,
    EntryKind.JUDGEMENT: InboxKind.JUDGEMENT,
    EntryKind.TOOL_RESULT: InboxKind.TOOL_RESULT,
    EntryKind.LINEAGE: InboxKind.LINEAGE,
}


def _read_cursor(cursor: str) -> dict[str, int]:
    if not cursor:
        return {}
    seen = json.loads(cursor)
    if not isinstance(seen, dict) or not all(isinstance(seq, int) for seq in seen.values()):
        raise ValueError(f"not an inbox cursor: {cursor!r}")
    return seen


def poll(
    state: NotebookState,
    branch: str,
    cursor: str,
    triggers: BriefTriggers | None = None,
    held: Sequence[Claim] = (),
) -> InboxPage:
    """The items for ``branch`` written since ``cursor``, and the cursor past everything seen now.

    With ``triggers``, ``held`` are the claims the branch holds now, which a pitfall learning is matched against.
    """

    seen = _read_cursor(cursor)
    briefing = (
        _Briefing(triggers, tuple(held), tuple(state.records()), lineage_view(state)) if triggers is not None else None
    )
    items = tuple(
        item
        for entry, value in state.decoded
        if entry.seq > seen.get(entry.writer, 0)
        for item in (
            _inbox_item(entry, value, branch),
            _brief_item(entry, value, branch, briefing) if briefing is not None else None,
        )
        if item is not None
    )
    latest = dict(seen)
    for entry in state.entries:
        latest[entry.writer] = max(latest.get(entry.writer, 0), entry.seq)
    return InboxPage(items, json.dumps(dict(sorted(latest.items())), separators=(",", ":")))


def _inbox_item(entry: Entry, value: object, branch: str) -> InboxItem | None:
    kind = _INBOX_KINDS.get(entry.kind)
    split_off = isinstance(value, LineageEvent) and value.from_branch == branch
    if kind is None or (entry.branch != branch and not split_off):
        return None
    pointer = getattr(value, "request_id", None) or getattr(value, "job_id", None) or entry.entry_id
    return InboxItem(kind, branch, str(pointer), entry.summary, entry.at)


def _brief_item(entry: Entry, value: object, branch: str, briefing: _Briefing) -> InboxItem | None:
    head = briefing.head(branch)
    reread = _REREAD.format(head=head)
    if isinstance(value, JudgementWrite) and entry.branch == branch and value.commit_id is None:
        stalled = judged_since_commit(briefing.records, briefing.view, branch, until=entry.at)
        if stalled == briefing.triggers.stall_tests:
            summary = f"stall: {stalled} tests judged since this branch's last commit, none committed; {reread}"
            return InboxItem(InboxKind.BRIEF, branch, head, summary, entry.at)
    if (
        briefing.triggers.pitfalls
        and isinstance(value, Learning)
        and value.kind is LearningKind.PITFALL
        and value.branch != branch
        and any(names(value, claim.intent) for claim in briefing.held)
    ):
        summary = f"pitfall: {value.branch} learnt, in a direction this branch holds: {value.content}; {reread}"
        return InboxItem(InboxKind.BRIEF, branch, head, summary, entry.at)
    return None


def _alternatives(query: str) -> list[list[str]]:
    groups = [[term.strip().lower() for term in group.split("&") if term.strip()] for group in query.split(",")]
    alternatives = [terms for terms in groups if terms]
    if not alternatives:
        raise ValueError("a query needs at least one term")
    return alternatives


def _searchable(entry: Entry) -> str:
    detail = json.dumps(entry.detail, sort_keys=True, ensure_ascii=False)
    return " ".join((entry.kind.value, entry.branch, entry.actor, entry.summary, detail)).lower()


def search(state: NotebookState, query: str, limit: int) -> tuple[SearchHit, ...]:
    """The newest ``limit`` entries matching ``query``."""

    alternatives = _alternatives(query)
    hits = []
    for entry in reversed(state.entries):
        text = _searchable(entry)
        if any(all(term in text for term in terms) for terms in alternatives):
            hits.append(SearchHit(entry.entry_id, entry.kind.value, entry.branch, entry.summary))
        if len(hits) >= limit:
            break
    return tuple(hits)


def profiles(state: NotebookState, commit_id: str) -> tuple[TestRecord, ...]:
    """The profile tests of ``commit_id`` whose result points at an artifact, oldest first."""

    return tuple(
        record
        for record in state.records()
        if record.request.kind is TestKind.PROFILE
        and record.request.parent_commit == commit_id
        and record.result is not None
        and record.result.profile_location
    )


__all__ = ["STALL_TESTS", "BriefTriggers", "poll", "profiles", "search"]
