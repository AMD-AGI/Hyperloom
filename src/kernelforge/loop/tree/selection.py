# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the selector's answer into the slots and requests a round may legally carry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from kernelforge.loop.tree.model import DirectionStatus, SearchTree, Slot, SlotType


@dataclass(frozen=True)
class StatusChange:
    """The selector closing or reopening one direction."""

    direction_id: str
    status: DirectionStatus
    reason: str


@dataclass(frozen=True)
class Selection:
    """What one round will expand, and everything the selector asked for that it will not."""

    slots: tuple[Slot, ...]
    remeasure: tuple[str, ...] = ()
    refresh_analysis: tuple[str, ...] = ()
    status_changes: tuple[StatusChange, ...] = ()
    dropped: tuple[str, ...] = ()
    fallback: str = ""


@dataclass
class _Reading:
    """Accumulates one answer's legal parts and the reasons the rest were dropped."""

    tree: SearchTree
    statuses: dict[str, DirectionStatus]
    slots: list[Slot] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    keys: set[tuple[str, str, str]] = field(default_factory=set)


def read_selection(tree: SearchTree, payload: dict | None, *, lanes: int, failure: str = "") -> Selection:
    """The legal part of the selector's answer, or the fallback slot when nothing in it is legal."""
    data = payload if isinstance(payload, dict) else {}
    changes, dropped = _status_changes(tree, data.get("direction_status"))
    reading = _Reading(tree, {item.direction_id: item.status for item in tree.directions}, dropped=dropped)
    reading.statuses.update({change.direction_id: change.status for change in changes})
    raw_slots = data.get("slots") if isinstance(data.get("slots"), list) else []
    for raw in raw_slots:
        if len(reading.slots) >= lanes:
            reading.dropped.append(f"slot beyond the round's {lanes} lane(s): {raw!r}"[:300])
            continue
        _read_slot(reading, raw)
    remeasure, dropped_remeasure = _nodes(tree, data.get("remeasure"), "remeasure")
    refresh, dropped_refresh = _nodes(tree, data.get("refresh_analysis"), "refresh_analysis")
    reading.dropped.extend([*dropped_remeasure, *dropped_refresh])
    fallback = ""
    if not reading.slots:
        fallback = failure or "the selector named no legal slot"
        reading.slots.append(fallback_slot(tree, reading.statuses))
    return Selection(
        slots=tuple(reading.slots),
        remeasure=remeasure,
        refresh_analysis=refresh,
        status_changes=changes,
        dropped=tuple(reading.dropped),
        fallback=fallback,
    )


def fallback_slot(tree: SearchTree, statuses: dict[str, DirectionStatus]) -> Slot:
    """Continue the published best's own direction, or open one at it when it has none still open."""
    best = tree.require_node(tree.published_best)
    reason = "fallback: continue from the published best"
    if best.direction and statuses.get(best.direction) is DirectionStatus.OPEN:
        return Slot(1, SlotType.DEEPEN, best.commit, reason, direction=best.direction)
    return Slot(1, SlotType.BRANCH, best.commit, reason)


def _read_slot(reading: _Reading, raw: object) -> None:
    if not isinstance(raw, dict):
        reading.dropped.append(f"slot is not an object: {raw!r}"[:300])
        return
    try:
        slot_type = SlotType(str(raw.get("type") or "").strip().upper())
    except ValueError:
        reading.dropped.append(f"slot names no slot type: {raw!r}"[:300])
        return
    start = reading.tree.resolve(str(raw.get("start") or ""))
    if not start:
        reading.dropped.append(f"{slot_type.value} slot names no node of the tree as its start: {raw.get('start')!r}")
        return
    slot = Slot(
        index=len(reading.slots) + 1,
        slot_type=slot_type,
        start=start,
        reason=str(raw.get("reason") or "").strip(),
        hint=str(raw.get("hint") or "").strip() if slot_type is SlotType.BRANCH else "",
    )
    refusal, slot = _completed(reading, slot, raw)
    if refusal:
        reading.dropped.append(refusal)
        return
    reading.slots.append(slot)


def _completed(reading: _Reading, slot: Slot, raw: dict) -> tuple[str, Slot]:
    """The slot with its type's own fields, or why it cannot run."""
    if slot.slot_type is SlotType.DEEPEN:
        return _deepen(reading, slot, str(raw.get("direction") or "").strip())
    if slot.slot_type is SlotType.PORT:
        return _port(reading, slot, raw)
    return "", slot


def _deepen(reading: _Reading, slot: Slot, direction_id: str) -> tuple[str, Slot]:
    direction = reading.tree.direction(direction_id)
    if direction is None:
        return f"DEEPEN slot names an unknown direction {direction_id!r}", slot
    if reading.statuses.get(direction_id) is not DirectionStatus.OPEN:
        return f"DEEPEN slot names closed direction {direction_id}", slot
    node = reading.tree.require_node(slot.start)
    if node.direction != direction_id and slot.start != direction.origin:
        return f"DEEPEN slot starts at {slot.start[:12]}, which is not on direction {direction_id}", slot
    return _unique(reading, replace(slot, direction=direction_id), direction_id)


def _port(reading: _Reading, slot: Slot, raw: dict) -> tuple[str, Slot]:
    if reading.tree.require_node(slot.start).is_root:
        return "PORT slot onto the root has no direction to carry gains into; BRANCH there instead", slot
    source = reading.tree.resolve(str(raw.get("source") or ""))
    if not source or source == slot.start:
        return f"PORT slot from {slot.start[:12]} names no other node as its source", slot
    gains = str(raw.get("gains") or "").strip()
    if not gains:
        return f"PORT slot onto {slot.start[:12]} names no gains to carry", slot
    return _unique(reading, replace(slot, source=source, gains=gains), source)


def _unique(reading: _Reading, slot: Slot, second: str) -> tuple[str, Slot]:
    key = (slot.slot_type.value, slot.start, second)
    if key in reading.keys:
        return f"{slot.slot_type.value} slot repeats one already in the round: {slot.start[:12]} / {second[:12]}", slot
    reading.keys.add(key)
    return "", slot


def _status_changes(tree: SearchTree, raw: object) -> tuple[tuple[StatusChange, ...], list[str]]:
    changes: list[StatusChange] = []
    dropped: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        direction_id = str(item.get("direction") or "").strip() if isinstance(item, dict) else ""
        try:
            status = DirectionStatus(str(item.get("status") or "").strip().upper())
        except (AttributeError, ValueError):
            dropped.append(f"direction status names no status: {item!r}"[:300])
            continue
        reason = str(item.get("reason") or "").strip()
        if tree.direction(direction_id) is None or not reason:
            dropped.append(f"direction status needs a known direction and a reason: {item!r}"[:300])
            continue
        changes.append(StatusChange(direction_id, status, reason))
    return tuple(changes), dropped


def _nodes(tree: SearchTree, raw: object, label: str) -> tuple[tuple[str, ...], list[str]]:
    found: list[str] = []
    dropped: list[str] = []
    for token in raw if isinstance(raw, list) else []:
        commit = tree.resolve(str(token))
        if commit and commit not in found:
            found.append(commit)
        elif not commit:
            dropped.append(f"{label} names no node of the tree: {token!r}")
    return tuple(found), dropped


__all__ = ["Selection", "StatusChange", "fallback_slot", "read_selection"]
