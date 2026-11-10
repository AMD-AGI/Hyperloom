# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round partition: the request a lane round is divided by, the answer it is held to, and how that answer becomes
lane grounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernelforge.agent_backends.structured_output import AgentAnswer
from kernelforge.workflows.forge_loop.roles.contracts import (
    LaneGround,
    OrchestrationContext,
    SpecialistOutcome,
    SpecialistRound,
)
from kernelforge.workflows.forge_loop.roles.plan_prompts import (
    ASSIGNED_PARTITION_SYSTEM_PROMPT,
    PARTITION_CHALLENGER_BLOCK,
    PARTITION_SYSTEM_PROMPT,
)
from kernelforge.workflows.forge_loop.roles.plan_requests import (
    CrossCuttingMoveAnswer,
    PartitionAnswer,
    assigned_partition_answer_model,
)

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


def is_challenged(context: OrchestrationContext) -> bool:
    """Whether the previous round's critic asked this round to validate a different route."""
    return context.last_critic_verdict == "REPLACE"


def partition_system_prompt(context: OrchestrationContext) -> str:
    """The partition turn's instructions: an assigned round's own, else the division by code, plus any challenge."""
    base = ASSIGNED_PARTITION_SYSTEM_PROMPT if context.lane_assignments else PARTITION_SYSTEM_PROMPT
    return base + (PARTITION_CHALLENGER_BLOCK if is_challenged(context) else "")


def partition_payload(context: OrchestrationContext, specialists: SpecialistRound, lanes: int) -> dict:
    """The partition request: the whole round's evidence, every analysis in view, and the lanes it divides into."""
    return {
        "task": _partition_task(context, lanes),
        "context": context.to_prompt_dict(),
        "dispatch_plan": specialists.dispatch_plan.to_dict(),
        "specialist_coverage": dict(specialists.coverage),
        "specialist_analyses": [
            {"assignment_id": outcome.assignment_id, "role_id": outcome.role_id, "analysis": outcome.content}
            for outcome in specialists.analyses
        ],
    }


def partition_answer_model(context: OrchestrationContext) -> type[AgentAnswer]:
    """The answer a round's partition is held to; an assigned round names its focus cases among the scored ones."""
    if context.lane_assignments:
        return assigned_partition_answer_model(tuple(case.case_id for case in context.cases))
    return PartitionAnswer


def _partition_task(context: OrchestrationContext, lanes: int) -> str:
    if context.lane_assignments:
        task = (
            f"Give each of the {lanes} assigned lanes, in order, an attempt that serves its assignment and differs "
            "materially from the other lanes: another approach, other code, or other cases. "
        )
    else:
        task = f"Divide this round into at most {lanes} lanes that would not edit the same code. "
    task += (
        "Name each lane's ground in files, functions and mechanisms. Name the largest cross-cutting move you found "
        "and either give it to a lane or say why no lane has it."
    )
    if is_challenged(context):
        task += " One lane validates the alternative the previous round's critic asked for."
    return task


def fallback_grounds(context: OrchestrationContext, analyses: Sequence[SpecialistOutcome]) -> list[LaneGround]:
    """The lanes a round runs when its partition gave it none.

    Assigned lanes each plan their own assignment; any other round collapses to one lane, the challenger's when the
    previous critic asked for one.
    """
    if context.lane_assignments:
        return [
            LaneGround(
                lane_id=lane_id,
                ground="whatever code this lane's assignment needs",
                reason=(
                    "fallback partition: the round's own split was unavailable, so each assigned lane is planned on "
                    "its assignment alone"
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
class PartitionReading:
    """The lane grounds one partition answer carried, its largest cross-cutting move, and what reading it found."""

    grounds: tuple[LaneGround, ...]
    cross_cutting_move: Mapping[str, object]
    notes: tuple[str, ...]

    @classmethod
    def unavailable(cls, note: str) -> PartitionReading:
        return cls((), {"status": "unavailable"}, (note,))


def read_partition(answer: PartitionAnswer, *, lanes: int) -> PartitionReading:
    """The lane grounds and the round's cross-cutting move one partition answer names, at most ``lanes`` of them."""
    grounds: list[LaneGround] = []
    notes: list[str] = []
    for entry in answer.lanes:
        try:
            ground = LaneGround(
                lane_id=len(grounds) + 1,
                ground=entry.ground,
                reason=entry.reason,
                joint=entry.joint,
                fallback=entry.fallback,
            )
        except ValueError as error:
            notes.append(f"dropped a lane: {error}")
            continue
        grounds.append(ground)
    # Before the move is read, so a move can never be recorded as owned by a lane the round is over its ceiling to run.
    grounds = grounds[:lanes]
    notes.extend(
        f"lane {ground.lane_id} claims joint ground and named no fallback, so an abandoned joint step leaves it "
        "nothing to measure"
        for ground in grounds
        if ground.joint and not ground.fallback
    )
    move, move_notes = _placed_move(answer.cross_cutting_move, grounds)
    return PartitionReading(tuple(grounds), move, (*notes, *move_notes))


def read_assigned_partition(answer: AgentAnswer, *, lanes: int) -> PartitionReading:
    """Lane k of an assigned round's partition for assignment k, planned or not, and the round's cross-cutting move.

    ``answer`` is an instance of the model ``partition_answer_model`` built for the round. A lane the answer could
    not ground, or left out, comes back unplannable with why.
    """
    grounds: list[LaneGround] = []
    notes: list[str] = []
    for lane_id, entry in enumerate(answer.lanes[:lanes], start=1):
        try:
            ground = LaneGround(
                lane_id=lane_id,
                ground=entry.ground,
                reason=entry.reason,
                fallback=entry.fallback,
                unplannable=entry.unplannable.strip(),
                approach=entry.approach.strip(),
                focus_cases=tuple(dict.fromkeys(entry.focus_cases)),
            )
        except ValueError as error:
            notes.append(f"dropped a lane: {error}")
            ground = _unplannable_lane(lane_id, f"the partition gave it no usable ground: {error}")
        grounds.append(ground)
    grounds += [
        _unplannable_lane(lane_id, "the partition returned no lane for it")
        for lane_id in range(len(grounds) + 1, lanes + 1)
    ]
    move, move_notes = _placed_move(answer.cross_cutting_move, grounds)
    return PartitionReading(tuple(grounds), move, (*notes, *move_notes))


def _unplannable_lane(lane_id: int, reason: str) -> LaneGround:
    return LaneGround(lane_id=lane_id, ground="", unplannable=reason)


def _placed_move(move: CrossCuttingMoveAnswer, grounds: Sequence[LaneGround]) -> tuple[dict[str, object], list[str]]:
    """Record the round's largest cross-cutting move as owned by the lane it names, or as unassigned with why."""
    named = move.move.strip()
    if not named:
        return {"status": "missing"}, ["partition named no largest cross-cutting move"]
    if 1 <= move.lane_id <= len(grounds):
        owner = grounds[move.lane_id - 1]
        return {"status": "assigned", "move": named, "lane_id": move.lane_id, "lane_ground": owner.ground}, []
    notes: list[str] = []
    if move.lane_id:
        notes.append(
            f"cross-cutting move names lane {move.lane_id}, which this partition of {len(grounds)} lane(s) does not "
            "have"
        )
    reason = move.unassigned_reason.strip()
    if not reason:
        notes.append("cross-cutting move was left unassigned and no reason was given")
    return {"status": "unassigned", "move": named, "lane_id": 0, "unassigned_reason": reason}, notes
