# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist pulls from the notebook: its branch's inbox, record search, and profile records.

The inbox holds pointers only. A cursor is the last sequence number seen of every writer, encoded opaquely, so a poll
returns each entry once however the writers' clocks interleave.

Search: ``,`` separates alternatives and ``&`` joins terms that must all appear, so ``fp8&kv,attention`` matches an
entry mentioning both ``fp8`` and ``kv``, or one mentioning ``attention``. Matching is case-insensitive substring search
over an entry's kind, branch, actor, summary and stored detail.
"""

from __future__ import annotations

import json

from hyperloom_multibranch.contracts.notebook.peers import TestKind
from hyperloom_multibranch.contracts.notebook.records import (
    InboxItem,
    InboxPage,
    LineageEvent,
    SearchHit,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import InboxKind
from hyperloom_multibranch.notebook.local.segments import Entry, EntryKind
from hyperloom_multibranch.notebook.local.state import NotebookState

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


def poll(state: NotebookState, branch: str, cursor: str) -> InboxPage:
    """The items for ``branch`` written since ``cursor``, and the cursor past everything seen now."""

    seen = _read_cursor(cursor)
    items = tuple(
        item
        for entry, value in state.decoded
        if entry.seq > seen.get(entry.writer, 0)
        for item in (_inbox_item(entry, value, branch),)
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


__all__ = ["poll", "profiles", "search"]
