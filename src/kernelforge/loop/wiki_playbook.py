# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``--wiki-guided`` campaign's playbook: the directions its knowledge base round's builder answered with.

Each item is a technique a knowledge base page supports for this kernel, or a direction only the round's profiling
evidence ranks among the largest levers left. Items that edit the same code form a group, and one group is one part
of the batch session.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from kernelforge.orchestrator.structured_output import extract_json_object

# What an item rests on: knowledge base pages, or the round's profiling evidence alone.
SUPPORT_WIKI = "wiki"
SUPPORT_PROFILING = "profiling"
SUPPORTS = frozenset({SUPPORT_WIKI, SUPPORT_PROFILING})

# What one build may add: items, and the groups one batch session implements.
MAX_ITEMS = 12
MAX_BATCH_GROUPS = 6

_ITEM_ID = re.compile(r"^P\d{1,3}$")
_GROUP_ID = re.compile(r"^G\d{1,3}$")
_ID_NUMBER = re.compile(r"\d+$")
_LABEL = "wiki playbook"


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

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.group_id, "title": self.title, "why_together": self.why_together}


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


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return ()
    return tuple(dict.fromkeys(text for text in (_text(entry) for entry in value) if text))


def _group_from_entry(entry: object, groups: Sequence[PlaybookGroup], notes: list[str]) -> PlaybookGroup | None:
    if not isinstance(entry, Mapping):
        notes.append("dropped a group that was not an object")
        return None
    group_id = _text(entry.get("id")).upper()
    title = _text(entry.get("title"))
    if not _GROUP_ID.match(group_id) or not title:
        notes.append(f"dropped group {group_id or '?'}: it needs an id like G1 and a title")
        return None
    if any(group.group_id == group_id for group in groups):
        notes.append(f"dropped a second group {group_id}")
        return None
    return PlaybookGroup(group_id=group_id, title=title, why_together=_text(entry.get("why_together")))


def _groups_from_payload(payload: Mapping[str, Any], max_groups: int | None, notes: list[str]) -> list[PlaybookGroup]:
    groups: list[PlaybookGroup] = []
    for entry in payload.get("groups") or []:
        group = _group_from_entry(entry, groups, notes)
        if group is None:
            continue
        if max_groups is not None and len(groups) == max_groups:
            notes.append(f"dropped group {group.group_id}: a batch implements at most {max_groups} groups")
            continue
        groups.append(group)
    return groups


def _item_support(item_id: str, entry: Mapping[str, Any], pages: tuple[str, ...], notes: list[str]) -> str | None:
    """What the item rests on, or None when it rests on nothing it may."""
    support = _text(entry.get("support")).lower() or SUPPORT_WIKI
    if support not in SUPPORTS:
        notes.append(f"dropped item {item_id}: its support {support!r} is neither wiki nor profiling")
        return None
    if pages:
        # A page that supports it makes it a knowledge base item, whatever the answer called it.
        return SUPPORT_WIKI
    if support != SUPPORT_PROFILING:
        notes.append(f"dropped item {item_id}: it cites no knowledge base page and is not marked profiling")
        return None
    if not _text(entry.get("applies_when")):
        notes.append(f"dropped item {item_id}: a profiling item must name the evidence behind it in applies_when")
        return None
    return SUPPORT_PROFILING


def _item_from_entry(
    entry: object, items: Sequence[PlaybookItem], known_groups: set[str], notes: list[str]
) -> PlaybookItem | None:
    if not isinstance(entry, Mapping):
        notes.append("dropped an item that was not an object")
        return None
    item_id = _text(entry.get("id")).upper()
    title = _text(entry.get("title"))
    group = _text(entry.get("group")).upper()
    if not _ITEM_ID.match(item_id) or not title:
        notes.append(f"dropped item {item_id or '?'}: it needs an id like P1 and a title")
        return None
    if any(item.item_id == item_id for item in items):
        notes.append(f"dropped a second item {item_id}")
        return None
    if group not in known_groups:
        notes.append(f"dropped item {item_id}: its group {group or '?'} is not one of the playbook's groups")
        return None
    pages = _strings(entry.get("wiki_pages"))
    support = _item_support(item_id, entry, pages, notes)
    if support is None:
        return None
    return PlaybookItem(
        item_id=item_id,
        title=title,
        group=group,
        wiki_pages=pages,
        applies_when=_text(entry.get("applies_when")),
        cases=_text(entry.get("cases")),
        expected_effect=_text(entry.get("expected_effect")),
        verify=_text(entry.get("verify")),
        conflicts_with=tuple(item.upper() for item in _strings(entry.get("conflicts_with"))),
        support=support,
    )


def _items_from_payload(
    payload: Mapping[str, Any], known_groups: set[str], max_items: int | None, notes: list[str]
) -> list[PlaybookItem]:
    entries = list(payload.get("items") or [])
    items: list[PlaybookItem] = []
    for entry in entries:
        item = _item_from_entry(entry, items, known_groups, notes)
        if item is None:
            continue
        items.append(item)
        if max_items is not None and len(items) == max_items:
            if len(entries) > max_items:
                notes.append(f"kept the first {max_items} items")
            break
    known_items = {item.item_id for item in items}
    return [
        replace(item, conflicts_with=tuple(other for other in item.conflicts_with if other in known_items))
        for item in items
    ]


def _playbook_from_payload(
    payload: Mapping[str, Any],
    *,
    max_items: int | None = MAX_ITEMS,
    max_groups: int | None = MAX_BATCH_GROUPS,
) -> tuple[Playbook | None, list[str]]:
    notes: list[str] = []
    groups = _groups_from_payload(payload, max_groups, notes)
    items = _items_from_payload(payload, {group.group_id for group in groups}, max_items, notes)
    if not items:
        notes.append("the playbook has no usable item")
        return None, notes
    used_groups = {item.group for item in items}
    playbook = Playbook(
        items=tuple(items),
        groups=tuple(group for group in groups if group.group_id in used_groups),
        summary=_text(payload.get("summary")),
    )
    return playbook, notes


def _renumber(entries: list[Any], pattern: re.Pattern[str], prefix: str, notes: list[str]) -> dict[str, str]:
    """Give every titled entry whose id is missing or misshaped the next free id, in place; old id to new id."""
    ids = [_text(entry.get("id")).upper() if isinstance(entry, Mapping) else "" for entry in entries]
    number = max((int(_ID_NUMBER.search(value).group()) for value in ids if pattern.match(value)), default=0)
    kind = "group" if prefix == "G" else "item"
    mapping: dict[str, str] = {}
    for index, (entry, old) in enumerate(zip(entries, ids)):
        if not (isinstance(entry, Mapping) and _text(entry.get("title"))) or pattern.match(old):
            continue
        number += 1
        new = f"{prefix}{number}"
        if old:
            mapping.setdefault(old, new)
        notes.append(f"renumbered {kind} {old or '(no id)'} as {new}")
        entries[index] = {**entry, "id": new}
    return mapping


def _remapped(entry: Any, groups: Mapping[str, str], items: Mapping[str, str]) -> Any:
    """One item with its group and conflicts pointing at their renumbered ids."""
    if not isinstance(entry, Mapping):
        return entry
    return {
        **entry,
        "group": groups.get(_text(entry.get("group")).upper(), entry.get("group")),
        "conflicts_with": [items.get(other.upper(), other) for other in _strings(entry.get("conflicts_with"))],
    }


def _renumbered(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``payload`` with every titled group and item whose id is missing or not shaped like G1 / P1 given the next free
    one, and the references to it rewritten: an answer that numbered its items N1, N2 is still a usable build."""
    notes: list[str] = []
    groups = list(payload.get("groups") or [])
    items = list(payload.get("items") or [])
    group_ids = _renumber(groups, _GROUP_ID, "G", notes)
    item_ids = _renumber(items, _ITEM_ID, "P", notes)
    if group_ids or item_ids:
        items = [_remapped(entry, group_ids, item_ids) for entry in items]
    return {**payload, "groups": groups, "items": items}, notes


def parse_playbook(text: str) -> tuple[Playbook | None, list[str]]:
    """Read one builder answer; the round's build, or ``None`` and why."""
    try:
        payload = extract_json_object(text or "", _LABEL)
    except ValueError as error:
        return None, [str(error)]
    payload, renumbered = _renumbered(payload)
    build, notes = _playbook_from_payload(payload)
    return build, [*renumbered, *notes]


def declined(text: str) -> bool:
    """Whether a builder answer says, in its own shape, that nothing more is worth building."""
    try:
        payload = extract_json_object(text or "", _LABEL)
    except ValueError:
        return False
    return not (payload.get("items") or payload.get("groups"))
