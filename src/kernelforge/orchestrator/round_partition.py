# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round partition: what a lane round asks before it plans its lanes, and how the answer becomes lane grounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kernelforge.orchestrator.contracts import LaneGround, OrchestrationContext, SpecialistOutcome, SpecialistRound
from kernelforge.orchestrator.structured_output import extract_json_object

_PARTITION_SYSTEM_PROMPT = """\
You are the read-only round planner for an autonomous GPU-kernel optimization
search. Several Implementers will work this round concurrently, each in its own
workspace copy, each producing one candidate that is measured on its own.

Divide the round into lanes so that no two lanes would edit the same code. That
is the only thing this step decides, and it is decided in the terms an edit
lands in: files, functions, and mechanisms. Do not divide by specialist role.
The roles are three readings of one kernel, so a lane that owns "the memory
analysis" owns nothing an editor can stay inside, and two such lanes routinely
rewrite the same lines.

Every lane's ground must be worth one Implementer session on its own: a
direction the supplied evidence supports, that one session can finish, and that
earns a measurement distinguishable from the others. Ground that only repeats
another lane's change in different words is not a second lane.

A change is one ground however many places it lands. When the strongest move
the evidence supports rewrites the same shape everywhere it appears -- a
subexpression several functions each recompute, a layout they all read, a
launch they all repeat -- that move is one lane's ground, not one lane per
site. Splitting it does not divide the work, it removes it: each site is
already the best it can be alone, so every lane reports there was nothing to
find and the round spends its sessions establishing that, while the change that
was there goes unattempted.

The launch configuration is not ground of its own. A tile geometry, an unroll
factor, a staging depth -- anything that changes what one iteration of the loop
weighs -- invalidates the `num_warps`, `waves_per_eu`, `num_stages` and grid
shape that were tuned for the old weight, so the lane that changes the body
owns the configuration that serves it. A lane holding only the launch knobs
cannot choose them; it is a pass-through for whatever the body lane leaves
behind. A body lane forbidden to re-tune them is measured at a configuration
built for code it deleted, and closes an axis that was never tested. Measured
on this class of kernel: a wider tile read 2.80 ms at the narrow tile's
configuration and 1.28 ms at its own. Give both to one lane and divide the rest
around it.

One launch site belongs to exactly one lane, because two lanes editing one
dispatch is the overlap this division exists to prevent. When two bodies you
would divide are served by the same launch, name that launch in exactly one
lane's ground -- every other lane sees it as ground it does not own -- or keep
both bodies in one lane.

Name the largest move the evidence supports that no single region contains --
a launch two kernels could share, a buffer that need not round-trip, a dispatch
that could be deleted -- and say which lane owns it. If no lane does, say so
and say why. A move worth more than any lane you did divide is this round's
most important finding whether or not it fits the division, and a round that
leaves it out of its own answer leaves the next round to rediscover it. That it
spans two kernels is not a reason to leave it unowned; that is what makes it
one ground.

Mark a lane `joint` when it claims a body and the configuration that serves it,
or a move that spans what no other lane would own, and give that lane a
`fallback`: the smaller, lower-risk change inside the same ground its
Implementer lands if the main step does not converge. A joint lane returns a
gain that cannot be split between its parts, so it is worth its width only when
the session cannot end with nothing measured.

Return fewer lanes than requested when the evidence supports fewer. The
requested count is a ceiling on how many directions a round may pursue, never a
number of pieces to cut one direction into. A round of two real directions
beats a round of three where one was invented to fill a slot, because the
invented one still costs a full session.

Use only supplied evidence. Do not edit files, run shell commands, alter
measurement inputs, or present an inference as a profiler fact. Return one JSON
object containing the lane grounds. Do not use Markdown fences or add text
outside the JSON object.
"""

_PARTITION_ASSIGNED_BLOCK = """\

This round's lanes are already decided and listed in the payload under
`lane_assignments`: lane k serves assignment k. Return exactly one lane per
assignment, in that order, and do not merge, drop or add lanes; the
instruction above to return fewer lanes does not apply to this round. Give
each lane the ground that serves its assignment, still divided so that no two
lanes would edit the same code. When the supplied evidence cannot serve an
assignment at all, return its lane with an empty `ground` and say why in
`unplannable`: that reason goes back to whoever made the assignment.
"""

_PARTITION_CHALLENGER_BLOCK = """\

The previous round's critic returned REPLACE: it judged that the route this
search is on is strategically dominated and that a materially different one
should be validated instead. Its review is in the payload under
`last_plan_critic`.

Give exactly one lane to that challenge. Its ground is not a region carved out
of the current implementation -- it is the alternative the review names, stated
concretely enough for one session to build and measure the smallest version of
it that would settle whether the route is worth taking. Say in that lane's
reason what result would decide it.

The remaining lanes are divided as usual, over ground the challenger does not
touch. A challenger is allowed to lose: it is measured under the unchanged
correctness and KEEP gates like any other lane, and one lane is what the round
is willing to spend to find out.
"""

_OUTPUT_SCHEMA = {
    "lanes": [
        {
            "ground": "The files, functions and mechanisms this lane owns and may edit",
            "reason": "Why this is one session's worth of work",
            "joint": (
                "true when this lane owns a change together with the launch configuration that serves it, or a move "
                "no single region contains"
            ),
            "fallback": "For a joint lane: the smaller change inside the same ground to land if the main step is abandoned",
            "unplannable": (
                "Only when the round's lanes are assigned: why the evidence cannot serve this lane's assignment, with "
                "its ground left empty"
            ),
        }
    ],
    "cross_cutting_move": {
        "move": "The largest move the evidence supports that no single region contains",
        "lane_id": "The 1-based lane that owns it, or 0 if no lane does",
        "unassigned_reason": "Why no lane owns it, when no lane does",
    },
}

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
    return (
        _PARTITION_SYSTEM_PROMPT
        + (_PARTITION_ASSIGNED_BLOCK if context.lane_assignments else "")
        + (_PARTITION_CHALLENGER_BLOCK if is_challenged(context) else "")
    )


def partition_payload(context: OrchestrationContext, specialists: SpecialistRound, lanes: int) -> dict:
    """What the partition turn reads: the whole round's evidence, every analysis in view."""
    return {
        "task": _partition_task(context, lanes),
        "context": context.to_prompt_dict(),
        "dispatch_plan": specialists.dispatch_plan.to_dict(),
        "specialist_coverage": dict(specialists.coverage),
        "specialist_analyses": [
            {"assignment_id": outcome.assignment_id, "role_id": outcome.role_id, "analysis": outcome.content}
            for outcome in specialists.analyses
        ],
        "output_schema": _OUTPUT_SCHEMA,
    }


def _partition_task(context: OrchestrationContext, lanes: int) -> str:
    if context.lane_assignments:
        task = (
            f"Give each of the {lanes} assigned lanes, in order, the ground that serves its assignment, divided so "
            "that no two lanes would edit the same code. "
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


__all__ = [
    "PartitionAnswer",
    "fallback_grounds",
    "is_challenged",
    "partition_payload",
    "partition_system_prompt",
    "read_partition_answer",
]
