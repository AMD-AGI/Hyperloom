# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the selector's answer into the slots and requests a round may legally carry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from kernelforge.agent_backends.structured_output import AgentAnswer
from kernelforge.workflows.forge_loop.roles.slot_selector import DirectionStatusAnswer, SlotAnswer, SlotSelectionAnswer
from kernelforge.workflows.forge_loop.tree.model import DirectionStatus, SearchTree, Slot, SlotType
from kernelforge.workflows.forge_loop.tree.profiles import profile_at

# How much of a dropped request its record quotes.
_QUOTED_CHARS = 300
_NO_ANSWER = SlotSelectionAnswer(rationale="", slots=[], remeasure=[], refresh_analysis=[], direction_status=[])


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


def read_selection(
    tree: SearchTree,
    answer: SlotSelectionAnswer | None,
    *,
    lanes: int,
    failure: str = "",
    profile_cycle: Sequence[str] = (),
) -> Selection:
    """The legal part of the selector's answer, or the fallback slot when nothing in it is legal.

    Each slot runs on the profile ``profile_cycle`` gives the position the selector named it at, so a slot dropped
    before it does not move it onto another profile.
    """
    given = answer if answer is not None else _NO_ANSWER
    changes, dropped = _status_changes(tree, given.direction_status)
    reading = _Reading(tree, {item.direction_id: item.status for item in tree.directions}, dropped=dropped)
    reading.statuses.update({change.direction_id: change.status for change in changes})
    for position, raw in enumerate(given.slots):
        if len(reading.slots) >= lanes:
            reading.dropped.append(_quoted(f"slot beyond the round's {lanes} lane(s)", raw))
            continue
        _read_slot(reading, raw, profile_at(profile_cycle, position))
    remeasure, dropped_remeasure = _nodes(tree, given.remeasure, "remeasure")
    refresh, dropped_refresh = _nodes(tree, given.refresh_analysis, "refresh_analysis")
    reading.dropped.extend([*dropped_remeasure, *dropped_refresh])
    fallback = ""
    if not reading.slots:
        fallback = failure or "the selector named no legal slot"
        reading.slots.append(
            replace(fallback_slot(tree, reading.statuses), agent_profile_id=profile_at(profile_cycle, 0))
        )
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


def _quoted(message: str, item: AgentAnswer) -> str:
    return f"{message}: {item.model_dump()!r}"[:_QUOTED_CHARS]


def _read_slot(reading: _Reading, raw: SlotAnswer, profile_id: str) -> None:
    slot_type = SlotType(raw.slot_type)
    start = reading.tree.resolve(raw.start)
    if not start:
        reading.dropped.append(f"{slot_type.value} slot names no node of the tree as its start: {raw.start!r}")
        return
    slot = Slot(
        index=len(reading.slots) + 1,
        slot_type=slot_type,
        start=start,
        reason=raw.reason.strip(),
        hint=raw.hint.strip() if slot_type is SlotType.BRANCH else "",
        agent_profile_id=profile_id,
    )
    refusal, slot = _completed(reading, slot, raw)
    if refusal:
        reading.dropped.append(refusal)
        return
    reading.slots.append(slot)


def _completed(reading: _Reading, slot: Slot, raw: SlotAnswer) -> tuple[str, Slot]:
    """The slot with its type's own fields, or why it cannot run."""
    if slot.slot_type is SlotType.DEEPEN:
        return _deepen(reading, slot, raw.direction.strip())
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


def _port(reading: _Reading, slot: Slot, raw: SlotAnswer) -> tuple[str, Slot]:
    if reading.tree.require_node(slot.start).is_root:
        return "PORT slot onto the root has no direction to carry gains into; BRANCH there instead", slot
    source = reading.tree.resolve(raw.source)
    if not source or source == slot.start:
        return f"PORT slot from {slot.start[:12]} names no other node as its source", slot
    gains = raw.gains.strip()
    if not gains:
        return f"PORT slot onto {slot.start[:12]} names no gains to carry", slot
    return _unique(reading, replace(slot, source=source, gains=gains), source)


def _unique(reading: _Reading, slot: Slot, second: str) -> tuple[str, Slot]:
    key = (slot.slot_type.value, slot.start, second)
    if key in reading.keys:
        return f"{slot.slot_type.value} slot repeats one already in the round: {slot.start[:12]} / {second[:12]}", slot
    reading.keys.add(key)
    return "", slot


def _status_changes(
    tree: SearchTree, requested: Sequence[DirectionStatusAnswer]
) -> tuple[tuple[StatusChange, ...], list[str]]:
    changes: list[StatusChange] = []
    dropped: list[str] = []
    for item in requested:
        direction_id, reason = item.direction.strip(), item.reason.strip()
        if tree.direction(direction_id) is None or not reason:
            dropped.append(_quoted("direction status needs a known direction and a reason", item))
            continue
        changes.append(StatusChange(direction_id, DirectionStatus(item.status), reason))
    return tuple(changes), dropped


def _nodes(tree: SearchTree, tokens: Sequence[str], label: str) -> tuple[tuple[str, ...], list[str]]:
    found: list[str] = []
    dropped: list[str] = []
    for token in tokens:
        commit = tree.resolve(token)
        if commit and commit not in found:
            found.append(commit)
        elif not commit:
            dropped.append(f"{label} names no node of the tree: {token!r}")
    return tuple(found), dropped


__all__ = ["Selection", "StatusChange", "fallback_slot", "read_selection"]
