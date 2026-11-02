# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round partition: the instructions a lane round is divided under, and how the answer becomes lane grounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kernelforge.workflows.forge_loop.roles.contracts import LaneGround, OrchestrationContext, SpecialistOutcome
from kernelforge.workflows.forge_loop.roles.plan_prompts import (
    PARTITION_ASSIGNED_BLOCK,
    PARTITION_CHALLENGER_BLOCK,
    PARTITION_SYSTEM_PROMPT,
)
from kernelforge.workflows.forge_loop.roles.structured_output import extract_json_object

_CHALLENGER_FALLBACK = LaneGround(
    lane_id=1,
    ground=(
        "the alternative route named in `last_plan_critic`, in the smallest form that would settle whether it beats "
        "the current implementation; not that implementation's own code"
    ),
    reason=(
        "fallback partition: the previous round's critic returned REPLACE, and the round it judged collapses to the "
        "one lane that validates the route it named"
    ),
)

_NOT_A_LANE_NUMBER = "cross-cutting move named a lane_id that is not a lane number"


def is_challenged(context: OrchestrationContext) -> bool:
    """Whether the previous round's critic asked this round to validate a different route."""
    return context.last_critic_verdict == "REPLACE"


def partition_system_prompt(context: OrchestrationContext) -> str:
    """The partition turn's instructions, with the blocks this round's assignments and challenge add."""
    return (
        PARTITION_SYSTEM_PROMPT
        + (PARTITION_ASSIGNED_BLOCK if context.lane_assignments else "")
        + (PARTITION_CHALLENGER_BLOCK if is_challenged(context) else "")
    )


def fallback_grounds(context: OrchestrationContext, analyses: Sequence[SpecialistOutcome]) -> list[LaneGround]:
    """The lanes a round runs when its partition gave it none.

    Assigned lanes each plan their own assignment over ground no one divided; any other round collapses to one lane,
    the challenger's when the previous critic asked for one.
    """
    if context.lane_assignments:
        return [
            LaneGround(
                lane_id=lane_id,
                ground="whatever code this lane's assignment needs",
                reason=(
                    "fallback partition: the round's own split was unavailable, so each assigned lane is planned on "
                    "its assignment alone and the lanes may edit the same code"
                ),
            )
            for lane_id in range(1, len(context.lane_assignments) + 1)
        ]
    if is_challenged(context):
        return [_CHALLENGER_FALLBACK]
    roles = ", ".join(sorted({outcome.role_id for outcome in analyses}))
    return [
        LaneGround(
            lane_id=1,
            ground=f"whatever the {roles} analysis recommends, planned as a single lane",
            reason=(
                "fallback partition: the round's own split was unavailable, so the round runs as one lane rather than "
                "over ground the lanes would share"
            ),
        )
    ]


@dataclass(frozen=True)
class PartitionAnswer:
    """The lane grounds one partition answer carried, its largest cross-cutting move, and what was wrong with it."""

    grounds: tuple[LaneGround, ...]
    cross_cutting_move: Mapping[str, object]
    notes: tuple[str, ...]

    @classmethod
    def unavailable(cls, note: str) -> PartitionAnswer:
        return cls((), {"status": "unavailable"}, (note,))


def read_partition_answer(response: str, *, lanes: int, assignments: Sequence[str] = ()) -> PartitionAnswer:
    """Read lane grounds and the round's cross-cutting move out of one answer.

    With ``assignments``, lane k is the k-th entry and every assigned lane comes back: one the answer left out or
    could not ground comes back unplannable, with why.
    """
    try:
        payload = extract_json_object(response, "round partition")
    except ValueError as error:
        return PartitionAnswer.unavailable(str(error))
    raw = payload.get("lanes")
    if not isinstance(raw, list):
        return PartitionAnswer.unavailable("partition response carried no lanes list")
    reader = _LaneReader(assigned=bool(assignments))
    for position, entry in enumerate(raw, start=1):
        reader.read(position, entry)
    # Before the move is read, so a move can never be recorded as owned by a lane the round is over its ceiling to run.
    grounds = reader.grounds[:lanes]
    if assignments:
        grounds += [
            _unplannable_lane(lane_id, "the partition returned no lane for it")
            for lane_id in range(len(grounds) + 1, lanes + 1)
        ]
    move, move_notes = _read_cross_cutting_move(payload, grounds)
    notes = (*reader.notes, *_ground_notes(grounds, reader.unreadable_joint), *move_notes)
    return PartitionAnswer(tuple(grounds), move, notes)


def _unplannable_lane(lane_id: int, reason: str) -> LaneGround:
    return LaneGround(lane_id=lane_id, ground="", unplannable=reason)


@dataclass
class _LaneReader:
    """Reads the partition's lanes entry by entry; an assigned round keeps a lane, planned or not, for every entry."""

    assigned: bool
    grounds: list[LaneGround] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unreadable_joint: dict[int, object] = field(default_factory=dict)

    def read(self, position: int, entry: object) -> None:
        lane_id = position if self.assigned else len(self.grounds) + 1
        if not isinstance(entry, dict):
            self._drop(
                lane_id, "dropped a lane that was not an object", "the partition answered it with no lane object"
            )
            return
        joint, joint_readable = _as_bool(entry.get("joint"))
        try:
            ground = LaneGround(
                lane_id=lane_id,
                ground=str(entry.get("ground") or ""),
                reason=str(entry.get("reason") or ""),
                joint=joint,
                fallback=str(entry.get("fallback") or ""),
                unplannable=str(entry.get("unplannable") or "").strip() if self.assigned else "",
            )
        except ValueError as error:
            self._drop(lane_id, f"dropped a lane: {error}", f"the partition gave it no usable ground: {error}")
            return
        self.grounds.append(ground)
        if not joint_readable:
            self.unreadable_joint[lane_id] = entry.get("joint")

    def _drop(self, lane_id: int, note: str, reason: str) -> None:
        self.notes.append(note)
        if self.assigned:
            self.grounds.append(_unplannable_lane(lane_id, reason))


def _as_bool(value: object) -> tuple[bool, bool]:
    """Read a JSON field that was asked for as a boolean and may not be one."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True, True
        return False, text in {"false", "no", "0", ""}
    if value is None or isinstance(value, bool):
        return bool(value), True
    return False, False


def _ground_notes(grounds: Sequence[LaneGround], unreadable_joint: Mapping[int, object]) -> list[str]:
    notes: list[str] = []
    for ground in grounds:
        if ground.lane_id in unreadable_joint:
            notes.append(
                f"lane {ground.lane_id} answered joint with {unreadable_joint[ground.lane_id]!r}, which is not a "
                "boolean, so it is read as not joint and keeps the narrow ground"
            )
        if ground.joint and not ground.fallback:
            notes.append(
                f"lane {ground.lane_id} claims joint ground and named no fallback, so an abandoned joint step leaves it "
                "nothing to measure"
            )
    return notes


def _read_cross_cutting_move(
    payload: Mapping[str, object], grounds: Sequence[LaneGround]
) -> tuple[dict[str, object], list[str]]:
    """Read the largest move that fits no one region, and who owns it."""
    raw = payload.get("cross_cutting_move")
    if raw is not None and not isinstance(raw, (dict, str)):
        unreadable = f"cross_cutting_move came back as a {type(raw).__name__}, which no move can be read out of"
        return {"status": "unreadable", "field": unreadable}, []
    entry = raw if isinstance(raw, dict) else {"move": raw}
    move = str(entry.get("move") or "").strip()
    notes: list[str] = []
    if isinstance(raw, str) and move:
        notes.append("cross-cutting move came back as a string, which names a move and no lane to own it")
    if not move:
        return {"status": "missing"}, [*notes, "partition named no largest cross-cutting move"]
    lane_id = _lane_number(entry.get("lane_id"), notes)
    if 1 <= lane_id <= len(grounds):
        return {
            "status": "assigned",
            "move": move,
            "lane_id": lane_id,
            "lane_ground": grounds[lane_id - 1].ground,
        }, notes
    if lane_id:
        notes.append(
            f"cross-cutting move names lane {lane_id}, which this partition of {len(grounds)} lane(s) does not have"
        )
    reason = str(entry.get("unassigned_reason") or "").strip()
    if not reason:
        notes.append("cross-cutting move was left unassigned and no reason was given")
    return {"status": "unassigned", "move": move, "lane_id": 0, "unassigned_reason": reason}, notes


def _lane_number(value: object, notes: list[str]) -> int:
    """The lane ``value`` names, or 0 for none; a value that is not a lane number is noted as such."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0
    if isinstance(value, float) and not value.is_integer():
        notes.append(_NOT_A_LANE_NUMBER)
        return 0
    try:
        return int(value)
    except ValueError:
        notes.append(_NOT_A_LANE_NUMBER)
        return 0
