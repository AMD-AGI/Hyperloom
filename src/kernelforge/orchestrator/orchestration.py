"""Read-only orchestration sessions for specialist dispatch and synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.agent_backends import (
    AgentBackend,
    AgentProviderError,
    AgentRunResult,
    AgentRunSpec,
    AgentToolPolicy,
    create_registered_backend,
    watchdog_timeout_sec,
)
from kernelforge.config import Config
from kernelforge.orchestrator.agent_response import (
    AgentResponseIncompleteError,
    AgentResponseInfrastructureError,
    validated_agent_text,
)
from kernelforge.orchestrator.contracts import (
    DispatchIntent,
    DispatchPlan,
    EvidenceRef,
    LaneGround,
    OrchestrationContext,
    OrchestrationRunResult,
    SynthesizedPlan,
    SpecialistAssignment,
    SpecialistDefinition,
    SpecialistOutcome,
)
from kernelforge.orchestrator.specialists import (
    SpecialistAgent,
    SpecialistPool,
    SpecialistProbeConfig,
)
from kernelforge.orchestrator.plan_critic import (
    PLAN_CRITIC_MAX_TURNS,
    PLAN_CRITIC_TIMEOUT_SEC,
    PlanCriticAgent,
)
from kernelforge.orchestrator.structured_output import (
    build_repair_prompt,
    extract_json_object,
)


log = logging.getLogger(__name__)

PLAN_REVISION_MAX_TURNS = 100
PLAN_REVISION_TIMEOUT_SEC = 600
# The round partition divides ground the analyses already name, so it is given
# no tools and a bound short enough that a slow answer costs the round its
# partition rather than its planning window. Overrunning falls back to dealing
# the analyses out, which is what the round did before this step existed.
ROUND_PARTITION_MAX_TURNS = 8
# Dividing ground the analyses already name is a reading task, not the deepest
# reasoning the round does; the plans behind it keep the maximum.
ROUND_PARTITION_EFFORT = "high"
ROUND_PARTITION_TIMEOUT_SEC = 900


@contextmanager
def _phase_timer(
    target: MutableMapping[str, float],
    name: str,
) -> Iterator[None]:
    """Record one planning phase's wall-clock against ``name``.

    Repeat entries accumulate, so a phase that runs in more than one place --
    synthesis, which has a single-lane path and a fan-out path -- is one number
    rather than the last one to finish. Recorded in ``finally``: a phase that
    raised still cost the round its wall-clock, and a round that died is exactly
    when the question of where the window went gets asked.
    """
    started_at = time.monotonic()
    try:
        yield
    finally:
        target[name] = target.get(name, 0.0) + (time.monotonic() - started_at)


def _as_bool(value: object) -> tuple[bool, bool]:
    """Read a JSON field that was asked for as a boolean and may not be one.

    ``bool("false")`` is ``True``, so a model that quoted its answer would mark
    every lane joint -- and joint is the flag that widens ground. Returns the
    flag and whether the answer was readable as one.

    A JSON string is read as the boolean it spells, so ``"true"`` and ``"0"``
    are both answers; a string that spells no boolean, and any other non-bool
    value, is not read at all and reads as not joint. That is narrower than
    ``bool``, and deliberately: ``joint: 1`` is a shape models emit constantly,
    and putting it through ``bool`` stored the opposite of the note the caller
    then wrote about the same lane. ``joint: 0`` is unread for the same reason
    -- it lands on the narrow ground by luck rather than by an answer -- and it
    gets the same note, which is the only thing that distinguishes it from a
    lane the partition deliberately left narrow.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True, True
        return False, text in {"false", "no", "0", ""}
    if value is None or isinstance(value, bool):
        return bool(value), True
    return False, False


@dataclass(frozen=True)
class _RevisedPlan:
    """One critic-directed revision plus how its session was obtained."""

    text: str
    mode: str
    duration_sec: float


_DISPATCH_SYSTEM_PROMPT = """\
You are the read-only orchestration planner for an autonomous GPU-kernel search.

Use only the supplied evidence. Do not edit files, run shell commands, alter
measurement inputs, or present an inference as a profiler fact. Return one JSON
object containing the specialist assignments. Do not use Markdown fences or add
text outside the JSON object.
"""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are the read-only lead planner for an autonomous GPU-kernel optimization
search. Multiple specialists have independently analyzed the current kernel.

Synthesize their work into one coherent, implementer-ready optimization plan. Judge
the recommendations rather than copying them: compare expected value, evidence,
feasibility, correctness risk, implementation cost, dependencies, and conflicts.
Combine compatible ideas, reject weak or contradictory ideas, and choose a clear
implementation sequence. The result must be an integrated plan, not a catalog of
specialist suggestions.

Use only supplied evidence. Do not edit files, run shell commands, alter
measurement inputs, or present an inference as a profiler fact. Return ordinary
Markdown with no fixed schema.
"""

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

_LANE_SYNTHESIS_SYSTEM_PROMPT = """\
You are the read-only lead planner for one lane of an autonomous GPU-kernel
optimization search. Several lanes are implemented concurrently this round from
the same analysis, each by its own Implementer in its own workspace.

Plan only the ground your lane owns. Every lane's patch is measured on its own
and adopted on its own, so work that overlaps another lane's ground spends two
Implementer sessions on one change and returns one answer for the price of two.

Your ground and every other lane's are stated in the payload, in the terms an
edit lands in. They are the boundary, not a hint: when your strongest idea needs
code another lane owns, plan the strongest idea that fits inside your own ground
and say plainly what you had to leave out.

When the payload marks your lane `joint`, your ground deliberately spans more
than one region -- a change and the launch configuration it invalidates, or a
move no single region contains -- and its result cannot be attributed to either
part alone. Plan it time-boxed: name what is measured first, name the point at
which the joint step is abandoned, and sequence the lane's `fallback` behind
that point, so the session lands a measured candidate either way.

You are given the whole round's evidence, not a slice of it. Every analysis may
have something to say about your ground; read all of them and use whatever bears
on the code you own.

Within your own ground, judge the recommendations rather than copying them:
compare expected value, evidence, feasibility, correctness risk, cost and
dependencies, and choose a clear implementation sequence.

Use only supplied evidence. Do not edit files, run shell commands, alter
measurement inputs, or present an inference as a profiler fact. Return ordinary
Markdown with no fixed schema.
"""

_REVISION_SYSTEM_PROMPT = """\
You are the read-only lead planner revising one GPU-kernel optimization plan
after an independent critic review.

Produce one final, implementer-ready Markdown plan. Address every substantive
critic concern using the supplied evidence. For REVISE, correct evidence,
scope, sequencing, feasibility, and risk gaps while preserving worthwhile
parts. For REPLACE, discard the dominated implementation route and formulate a
concrete validation plan for the critic's alternative.

Do not dispatch specialists again, edit files, run shell commands, alter
measurement inputs, or present inference as profiler fact. Return only the
final ordinary Markdown plan.
"""


class OrchestrationInfrastructureError(RuntimeError):
    """Report an orchestration backend outage that produced no model answer."""


class OrchestrationOutputError(ValueError):
    """Report a non-infrastructure response that cannot serve as a plan."""


class OrchestrationAgent:
    """Run dispatch and synthesis turns through an injected read-only backend."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        timeout_sec: int,
        max_turns: int,
        min_assignments: int = 1,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if min_assignments <= 0:
            raise ValueError("min_assignments must be greater than zero")
        self.backend = backend
        self.timeout_sec = timeout_sec
        self.max_turns = max_turns
        self.min_assignments = min_assignments
        self.structured_output_diagnostics: dict[str, dict] = {}
        # What each planning phase cost this round, in the units the round is
        # budgeted in. Kept beside the structured-output diagnostics because it
        # is the same kind of thing -- what the round did, readable afterwards
        # without a log -- and filled by the phases themselves.
        self.phase_durations_sec: dict[str, float] = {}

    async def plan_dispatch(
        self,
        context: OrchestrationContext,
        definitions: Mapping[str, SpecialistDefinition],
        *,
        usage=None,
    ) -> DispatchPlan:
        """Normalize probabilistic role/case intent into canonical assignments."""
        if not definitions:
            raise ValueError("specialist definitions must not be empty")
        ruling_guidance = (
            "The latest Supervisor Ruling is the current planning authority and "
            "overrides subjective recommendations or conclusions in historical "
            "lesson records; objective measurements remain authoritative. "
            if context.supervisor_guidance
            else ""
        )
        mode_guidance = (
            "A latest Supervisor Ruling is present. Use that ruling to decide "
            "whether this planning cycle should persist or diversify; the "
            f"recorded search mode ({context.search_mode}) is background state."
            if context.supervisor_guidance
            else (
                "Seek materially different mechanisms across scored cases."
                if context.search_mode == "DIVERSIFY"
                else "Target the strongest immediate canonical gain."
            )
        )
        payload = {
            "task": (
                "Select the specialist roles needed for the supplied cases. "
                "Assignments may overlap cases when independent expertise is "
                "useful. Return role/case intent only; the framework binds "
                "assignment IDs and exact evidence paths. " + ruling_guidance + mode_guidance
            ),
            "context": context.to_prompt_dict(),
            "available_specialists": [definitions[role_id].to_dict() for role_id in sorted(definitions)],
            "output_schema": {
                "assignments": [
                    {
                        "role_id": "registered-role-id",
                        "target_case_ids": ["case-id"],
                        "reason": "Why this specialist is required",
                    }
                ],
            },
        }
        raw_responses: list[str] = []
        notes: list[str] = []
        intents: tuple[DispatchIntent, ...] = ()
        first = await self._run(
            context,
            system_prompt=_DISPATCH_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2, sort_keys=True),
            usage=usage,
            allow_incomplete=True,
        )
        raw_responses.append(first)
        intents, parse_notes = self._parse_dispatch_intents(first)
        notes.extend(parse_notes)
        if not intents:
            repaired = await self._run(
                context,
                system_prompt=_DISPATCH_SYSTEM_PROMPT,
                user_prompt=build_repair_prompt(
                    label="dispatch intent",
                    original_response=first,
                    validation_error="no usable role/case intent",
                    output_schema=payload["output_schema"],
                ),
                usage=usage,
                allow_incomplete=True,
            )
            raw_responses.append(repaired)
            intents, parse_notes = self._parse_dispatch_intents(repaired)
            notes.extend(parse_notes)
        plan = self._bind_dispatch(
            context=context,
            definitions=definitions,
            intents=intents,
            notes=notes,
        )
        self.structured_output_diagnostics["dispatch"] = {
            "raw_responses": raw_responses,
            "normalization_notes": list(plan.normalization_notes),
        }
        return plan

    @staticmethod
    def _parse_dispatch_intents(
        text: str,
    ) -> tuple[tuple[DispatchIntent, ...], tuple[str, ...]]:
        notes: list[str] = []
        try:
            payload = extract_json_object(text, "dispatch intent")
        except ValueError as error:
            return (), (f"invalid dispatch JSON: {error}",)
        raw_assignments = payload.get("assignments")
        if not isinstance(raw_assignments, list):
            return (), ("dispatch assignments were not a list",)
        intents = []
        for index, raw in enumerate(raw_assignments):
            if not isinstance(raw, dict):
                notes.append(f"dropped non-object assignment {index}")
                continue
            role_id = str(raw.get("role_id") or "").strip().lower()
            raw_cases = raw.get("target_case_ids")
            if isinstance(raw_cases, str):
                raw_cases = [raw_cases]
            case_ids = tuple(
                dict.fromkeys(str(case_id).strip() for case_id in (raw_cases or []) if str(case_id).strip())
            )
            if not role_id:
                notes.append(f"dropped assignment {index} without role_id")
                continue
            intents.append(
                DispatchIntent(
                    role_id=role_id,
                    target_case_ids=case_ids,
                    reason=str(raw.get("reason") or "").strip(),
                )
            )
        return tuple(intents), tuple(notes)

    def _bind_dispatch(
        self,
        *,
        context: OrchestrationContext,
        definitions: Mapping[str, SpecialistDefinition],
        intents: Sequence[DispatchIntent],
        notes: list[str],
    ) -> DispatchPlan:
        case_ids = tuple(case.case_id for case in context.cases)
        allowed_cases = set(case_ids)
        merged: dict[str, dict[str, object]] = {}
        for intent in intents:
            if intent.role_id not in definitions:
                notes.append(f"dropped unknown role {intent.role_id!r}")
                continue
            selected_cases = [case_id for case_id in intent.target_case_ids if case_id in allowed_cases]
            if not selected_cases:
                selected_cases = list(case_ids)
                notes.append(f"bound role {intent.role_id!r} to all scored cases")
            if intent.role_id in merged:
                notes.append(f"merged duplicate role {intent.role_id!r}")
                current = merged[intent.role_id]["case_ids"]
                selected_cases = list(dict.fromkeys([*current, *selected_cases]))
            merged[intent.role_id] = {
                "case_ids": selected_cases,
                "reason": (intent.reason or f"Framework-assigned {intent.role_id} analysis"),
            }

        required_roles = (
            len(definitions) if context.search_mode == "DIVERSIFY" else min(self.min_assignments, len(definitions))
        )
        for role_id in sorted(definitions):
            if len(merged) >= required_roles:
                break
            if role_id not in merged:
                merged[role_id] = {
                    "case_ids": list(case_ids),
                    "reason": "Framework-completed dispatch coverage",
                }
                notes.append(f"added default role {role_id!r}")

        covered = {case_id for item in merged.values() for case_id in item["case_ids"]}
        missing = [case_id for case_id in case_ids if case_id not in covered]
        if missing and merged:
            first_role = next(iter(merged))
            merged[first_role]["case_ids"] = [
                *merged[first_role]["case_ids"],
                *missing,
            ]
            notes.append(f"added missing cases to {first_role!r}: {', '.join(missing)}")

        global_kinds = {
            "analysis_artifact_catalog",
            "analysis_bundle",
            "analysis_cumulative_diff",
            "analysis_summary",
            "analysis_source_map",
            "analysis_workflow",
            "latest_lesson",
            "supervisor_guidance",
        }
        global_refs = [reference for reference in context.evidence_refs if reference.kind in global_kinds]
        source_ref = EvidenceRef(
            kind="source_map",
            path=context.source_map_path,
            summary="Current source map or anchor source.",
        )
        case_by_id = {case.case_id: case for case in context.cases}
        assignments = []
        for index, (role_id, item) in enumerate(merged.items(), start=1):
            refs = {reference.path: reference for reference in global_refs}
            refs[source_ref.path] = source_ref
            target_cases = tuple(item["case_ids"])
            for case_id in target_cases:
                case = case_by_id[case_id]
                measurement = EvidenceRef(
                    kind="measurement",
                    path=f"case:{case_id}",
                    summary=(f"Canonical timing and Analysis flags for {case_id}."),
                )
                refs[measurement.path] = measurement
                if case.profile_summary_path:
                    profiled = any(
                        flag
                        in {
                            "analysis_profiled",
                            "analysis_profile_incremental",
                            "analysis_checkpoint_profile_interpretation",
                            "analysis_checkpoint_normalized_only",
                            "analysis_checkpoint_raw_profile_only",
                        }
                        for flag in case.flags
                    )
                    reference = EvidenceRef(
                        kind=("profile" if profiled else "analysis_interpretation"),
                        path=case.profile_summary_path,
                        summary=("Measured profile" if profiled else "Analysis interpretation")
                        + f" for {case_id}. Flags: "
                        + f"{', '.join(case.flags) or '(none)'}.",
                    )
                    refs[reference.path] = reference
            assignments.append(
                SpecialistAssignment(
                    assignment_id=f"{role_id}-{index}",
                    role_id=role_id,
                    target_case_ids=target_cases,
                    evidence_refs=tuple(refs.values()),
                    reason=str(item["reason"]),
                )
            )
        return DispatchPlan(
            analysis_commit=context.analysis_commit,
            assignments=tuple(assignments),
            normalization_notes=tuple(notes),
        )

    async def synthesize_optimization_plan(
        self,
        context: OrchestrationContext,
        specialist_outcomes: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        *,
        usage=None,
    ) -> str:
        """Fuse all successful specialist analyses into one Markdown plan."""
        result = await self._synthesize_optimization_plan_result(
            context,
            specialist_outcomes,
            dispatch_plan,
            coverage,
            usage=usage,
        )
        return result.text

    async def _synthesize_optimization_plan_result(
        self,
        context: OrchestrationContext,
        specialist_outcomes: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        *,
        usage=None,
    ) -> SynthesizedPlan:
        """Synthesize one plan while preserving its resumable session."""
        analyses = [
            {
                "assignment_id": outcome.assignment_id,
                "role_id": outcome.role_id,
                "analysis": outcome.content,
            }
            for outcome in specialist_outcomes
            if outcome.content is not None
        ]
        if not analyses:
            raise ValueError("no specialist analysis is available for synthesis")
        failures = [outcome.to_dict() for outcome in specialist_outcomes if outcome.failure is not None]
        search_guidance = (
            "Use the latest Supervisor Ruling to choose whether immediate gain "
            "or mechanism diversity is appropriate for this planning cycle."
            if context.supervisor_guidance
            else (
                "Prioritize mechanisms that can produce the strongest immediate canonical gain."
                if context.search_mode == "EXPLOIT"
                else (
                    "Prioritize meaningful mechanism diversity and cross-case "
                    "headroom while keeping the resulting plan feasible."
                )
            )
        )
        ruling_guidance = (
            "Honor the latest Supervisor Ruling over subjective conclusions in "
            "historical lesson records, while preserving objective validation "
            "and measurement facts. "
            if context.supervisor_guidance
            else ""
        )
        payload = {
            "task": (
                "Produce the optimization plan that the Implementer should execute. "
                + ruling_guidance
                + f"{search_guidance} Reconcile all specialist analyses into one "
                "decision: select and sequence the most valuable compatible "
                "work, explain critical trade-offs, and omit ideas that do not "
                "justify their cost or risk. Do not merely summarize each "
                "specialist in turn."
            ),
            "context": context.to_prompt_dict(),
            "dispatch_plan": dispatch_plan.to_dict(),
            "specialist_coverage": dict(coverage),
            "specialist_analyses": analyses,
            "specialist_failures": failures,
        }
        result = await self._run_result(
            context,
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, indent=2, sort_keys=True),
            usage=usage,
            role="orchestration synthesis",
        )
        return SynthesizedPlan(
            text=self._validated_text(
                result,
                role="orchestration synthesis",
            ),
            session_id=str(result.session_id or "").strip(),
        )

    async def synthesize_lane_plans(
        self,
        context: OrchestrationContext,
        specialist_outcomes: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        *,
        lanes: int,
        usage=None,
    ) -> list[str]:
        """Return lane plan text while keeping session details internal."""
        results = await self.synthesize_lane_plan_results(
            context,
            specialist_outcomes,
            dispatch_plan,
            coverage,
            lanes=lanes,
            usage=usage,
        )
        return [result.text for result in results]

    async def synthesize_lane_plan_results(
        self,
        context: OrchestrationContext,
        specialist_outcomes: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        *,
        lanes: int,
        usage=None,
    ) -> list[SynthesizedPlan]:
        """Partition the round into plans that do not compete for one another's ground.

        Fusing every analysis into a single plan spends the round on one bet and
        yields one measurement, which cannot say which of the fused ideas earned
        the result. Splitting the round buys one measured score per direction,
        and only disjoint changes can be stacked afterwards, so the partition is
        what the whole round structure rests on.

        The partition is over the code, decided by a step that has read every
        analysis, and each lane is then given the whole round's evidence to plan
        its own share from. Dealing the analyses out instead -- one specialist
        report per lane -- divides nothing: the roles are three readings of one
        kernel, so two lanes holding different reports still reach for the same
        lines, and a lane holding a report about ground it does not own can only
        discard it.

        What the partition may hand one lane is wider than a region where the
        code is: a change and the launch configuration it invalidates are one
        ground, and so is a move spanning what no single lane would own. The
        lanes stay disjoint under that width, because a launch site two bodies
        share is given to one of them by name. That lane's gain cannot be
        attributed to either part, which is what its fallback pays for.

        A partition that cannot be bought collapses the round to a single lane,
        because a round that divides no code is not worth its N sessions. A
        pending REPLACE keeps that one lane as its challenger.
        """
        analyses = [outcome for outcome in specialist_outcomes if outcome.content is not None]
        width = max(1, min(int(lanes), len(analyses)))
        if width <= 1:
            with _phase_timer(self.phase_durations_sec, "synthesis"):
                return [
                    await self._synthesize_optimization_plan_result(
                        context,
                        specialist_outcomes,
                        dispatch_plan,
                        coverage,
                        usage=usage,
                    )
                ]

        grounds = await self._partition_round(
            context,
            analyses,
            dispatch_plan,
            coverage,
            lanes=width,
            usage=usage,
        )
        challenged = context.last_critic_verdict == "REPLACE"
        if len(grounds) <= 1 and not (challenged and grounds):
            # One real direction is a single-lane round, planned the way one has
            # always been planned rather than as a fan-out of width one. A
            # challenger ground is the exception: the ordinary synthesis would
            # refine the route the critic dominated, so its one lane runs over
            # the challenge instead.
            with _phase_timer(self.phase_durations_sec, "synthesis"):
                return [
                    await self._synthesize_optimization_plan_result(
                        context,
                        specialist_outcomes,
                        dispatch_plan,
                        coverage,
                        usage=usage,
                    )
                ]
        width = len(grounds)

        async def _lane(index: int) -> SynthesizedPlan | None:
            payload = {
                "task": (
                    "Produce the optimization plan this lane's Implementer "
                    "should execute on the ground this lane owns. Judge the "
                    "analyses rather than copying them, and choose a clear "
                    "implementation sequence."
                ),
                "context": context.to_prompt_dict(),
                "dispatch_plan": dispatch_plan.to_dict(),
                "specialist_coverage": dict(coverage),
                "lane": {
                    "ground": grounds[index].ground,
                    "joint": grounds[index].joint,
                    "fallback": grounds[index].fallback,
                    "ground_owned_by_other_lanes": [
                        other.ground for position, other in enumerate(grounds) if position != index
                    ],
                },
                "specialist_analyses": [
                    {
                        "assignment_id": outcome.assignment_id,
                        "role_id": outcome.role_id,
                        "analysis": outcome.content,
                    }
                    for outcome in analyses
                ],
            }
            result = await self._run_result(
                context,
                system_prompt=_LANE_SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, indent=2, sort_keys=True),
                usage=usage,
                role=f"orchestration lane {index + 1} synthesis",
            )
            text = self._validated_text(
                result,
                role=f"orchestration lane {index + 1} synthesis",
                allow_empty=True,
            )
            if not text:
                return None
            return SynthesizedPlan(
                text=text,
                session_id=str(result.session_id or "").strip(),
                ground=grounds[index].ground,
                joint=grounds[index].joint,
                fallback=grounds[index].fallback,
            )

        # Each lane is an independent call, so one that fails is one lane lost
        # and not the round. Letting it propagate would discard the siblings
        # that already answered -- and, because the loop reads a raised
        # synthesis as a planning outage, would multiply the chance of tripping
        # the orchestration circuit breaker by the number of lanes asked for.
        with _phase_timer(self.phase_durations_sec, "synthesis"):
            answers = await asyncio.gather(
                *(_lane(index) for index in range(width)),
                return_exceptions=True,
            )
        plans: list[SynthesizedPlan] = []
        for index, answer in enumerate(answers):
            if isinstance(answer, BaseException):
                log.warning(
                    "lane %d of %d lost its plan: %s: %s",
                    index + 1,
                    width,
                    type(answer).__name__,
                    answer,
                )
                continue
            if answer is None:
                # A lane that returned nothing is a lane the round paid for and
                # cannot use. Said out loud, because silently narrowing the
                # round makes an empty answer indistinguishable from having
                # asked for fewer lanes.
                log.warning("lane %d of %d returned an empty plan", index + 1, width)
                continue
            plans.append(answer)
        if not plans:
            raise OrchestrationOutputError("orchestration synthesis returned no lane plan")
        return plans

    @staticmethod
    def _collapsed_grounds(
        analyses: Sequence[SpecialistOutcome],
        *,
        challenged: bool = False,
    ) -> list[LaneGround]:
        """Collapse a round that could not be divided by code to a single lane.

        A fan-out round is worth its N sessions only when each lane edits code
        no other lane edits, so each candidate earns a score that can be
        attributed to it and stacked on the others. When the partition times
        out or comes back unparseable there is no such division: dealing the
        analyses out by role divides the evidence without dividing the code --
        observed in production before the partition existed, one lane's edited
        files a subset of its sibling's -- so a wide round spends N Implementer
        sessions and may get one answer for them. A round that cannot be divided
        runs as a single lane instead.

        A pending REPLACE still lands on its own challenger lane. The verdict
        says the current route is dominated, so a single ordinary lane would
        refine that very route -- the one outcome the verdict exists to stop.
        The fallback cannot name the alternative, but the review that named it
        is in the lane's payload, so the ground points there rather than
        restating it.
        """
        if challenged:
            return [
                LaneGround(
                    lane_id=1,
                    ground=(
                        "the alternative route named in `last_plan_critic`, in "
                        "the smallest form that would settle whether it beats "
                        "the current implementation; not that implementation's "
                        "own code"
                    ),
                    reason=(
                        "fallback partition: the previous round's critic "
                        "returned REPLACE, and the round it judged collapses to "
                        "the one lane that validates the route it named"
                    ),
                )
            ]
        return [
            LaneGround(
                lane_id=1,
                ground=(
                    "whatever the "
                    + ", ".join(sorted({outcome.role_id for outcome in analyses}))
                    + " analysis recommends, planned as a single lane"
                ),
                reason=(
                    "fallback partition: the round's own split was unavailable, "
                    "so the round runs as one lane rather than over ground the "
                    "lanes would share"
                ),
            )
        ]

    def _parse_lane_grounds(self, response: str, *, lanes: int) -> tuple[list[LaneGround], dict, list[str]]:
        """Read lane grounds and the round's cross-cutting move out of one answer.

        Reports what it dropped, and reports the move separately from the
        lanes: a move nobody owns is the one part of a partition that cannot be
        read off the lanes, because what makes it unowned is that it is not
        there.
        """
        notes: list[str] = []
        try:
            payload = extract_json_object(response, "round partition")
        except ValueError as error:
            return [], {"status": "unavailable"}, [str(error)]
        raw = payload.get("lanes")
        if not isinstance(raw, list):
            return (
                [],
                {"status": "unavailable"},
                ["partition response carried no lanes list"],
            )
        grounds: list[LaneGround] = []
        unreadable_joint: dict[int, object] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                notes.append("dropped a lane that was not an object")
                continue
            joint, joint_readable = _as_bool(entry.get("joint"))
            try:
                ground = LaneGround(
                    lane_id=len(grounds) + 1,
                    ground=str(entry.get("ground") or ""),
                    reason=str(entry.get("reason") or ""),
                    joint=joint,
                    fallback=str(entry.get("fallback") or ""),
                )
            except ValueError as error:
                notes.append(f"dropped a lane: {error}")
                continue
            grounds.append(ground)
            if not joint_readable:
                unreadable_joint[ground.lane_id] = entry.get("joint")
        # Before the move is read, so a move can never be recorded as owned by
        # a lane the round is over its ceiling to run.
        grounds = grounds[:lanes]
        for ground in grounds:
            if ground.lane_id in unreadable_joint:
                notes.append(
                    f"lane {ground.lane_id} answered joint with "
                    f"{unreadable_joint[ground.lane_id]!r}, which is not a "
                    "boolean, so it is read as not joint and keeps the narrow "
                    "ground"
                )
            if ground.joint and not ground.fallback:
                notes.append(
                    f"lane {ground.lane_id} claims joint ground and named no "
                    "fallback, so an abandoned joint step leaves it nothing to "
                    "measure"
                )
        move, move_notes = self._parse_cross_cutting_move(payload, grounds)
        return grounds, move, notes + move_notes

    @staticmethod
    def _parse_cross_cutting_move(
        payload: Mapping[str, object],
        grounds: Sequence[LaneGround],
    ) -> tuple[dict, list[str]]:
        """Read the largest move that fits no one region, and who owns it.

        Four outcomes, and they are kept apart because an operator acts on
        each differently: the move is owned by a lane this round will run, the
        move exists and no lane took it, the partition never named one, or the
        field came back in a shape no move can be read out of. Only the first
        needs nothing further; the rest are the shape that cost four kernels a
        mechanism -- named in an analysis, filed under nobody's ground, and
        absent from every artifact the next round reads.
        """
        notes: list[str] = []
        raw = payload.get("cross_cutting_move")
        reason = ""
        lane_id: object = 0
        if isinstance(raw, dict):
            move = str(raw.get("move") or "").strip()
            reason = str(raw.get("unassigned_reason") or "").strip()
            lane_id = raw.get("lane_id")
        elif isinstance(raw, str):
            move = raw.strip()
            if move:
                notes.append("cross-cutting move came back as a string, which names a move and no lane to own it")
        elif raw is None:
            move = ""
        else:
            return (
                {
                    "status": "unreadable",
                    "field": (
                        f"cross_cutting_move came back as a {type(raw).__name__}, which no move can be read out of"
                    ),
                },
                notes,
            )
        if not move:
            return {"status": "missing"}, notes + ["partition named no largest cross-cutting move"]
        if isinstance(lane_id, bool) or not isinstance(lane_id, (int, float, str)):
            lane_id = 0
        if isinstance(lane_id, float) and not lane_id.is_integer():
            notes.append("cross-cutting move named a lane_id that is not a lane number")
            lane_id = 0
        try:
            lane_id = int(lane_id)
        except ValueError:
            notes.append("cross-cutting move named a lane_id that is not a lane number")
            lane_id = 0
        if 1 <= lane_id <= len(grounds):
            return (
                {
                    "status": "assigned",
                    "move": move,
                    "lane_id": lane_id,
                    "lane_ground": grounds[lane_id - 1].ground,
                },
                notes,
            )
        if lane_id:
            notes.append(
                f"cross-cutting move names lane {lane_id}, which this partition of {len(grounds)} lane(s) does not have"
            )
        if not reason:
            notes.append("cross-cutting move was left unassigned and no reason was given")
        return (
            {
                "status": "unassigned",
                "move": move,
                "lane_id": 0,
                "unassigned_reason": reason,
            },
            notes,
        )

    async def _partition_round(
        self,
        context: OrchestrationContext,
        analyses: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        *,
        lanes: int,
        usage=None,
    ) -> list[LaneGround]:
        """Decide, once and with every analysis in view, what each lane owns.

        One call rather than one per lane, because the only question here is
        where the boundaries fall, and that cannot be answered from a slice. The
        expensive part of the round -- the plans, and the sessions that execute
        them -- stays parallel behind it.

        Given no workspace tools and a short bound. The analyses in the payload
        already name the files and functions they are about, so a partition that
        goes reading source is re-deriving the analysis rather than dividing it,
        and this call sits on the critical path where every lane waits for it.
        Measured before that bound existed: one partition over three analyses
        was still exploring after eighteen minutes.

        Lanes stay disjoint, with one exception: a change and the launch
        configuration it invalidates are one lane's ground, because the
        alternative is not two attributable measurements but one measurement of
        a body at a configuration tuned for the body it replaced. The exception
        widens one lane, never two -- a launch site shared by two bodies is
        owned by exactly one of them and named there -- so disjointness holds
        under it. Such a lane is marked joint and carries a fallback, so the
        width it buys cannot cost the round its candidate.

        The largest move that fits no one region is recorded whether or not a
        lane took it. A partition can only divide the code it is dividing, so a
        move spanning what no lane owns has nowhere to land -- and unrecorded,
        it is indistinguishable from a round that never found one.

        Any failure falls back to a round collapsed to one lane. This step
        exists to make a round's lanes disjoint, not to be another way for a
        round to die.
        """
        challenged = context.last_critic_verdict == "REPLACE"
        system_prompt = _PARTITION_SYSTEM_PROMPT + (_PARTITION_CHALLENGER_BLOCK if challenged else "")
        payload = {
            "task": (
                f"Divide this round into at most {lanes} lanes that would not "
                "edit the same code. Name each lane's ground in files, "
                "functions and mechanisms. Name the largest cross-cutting move "
                "you found and either give it to a lane or say why no lane has "
                "it."
                + (" One lane validates the alternative the previous round's critic asked for." if challenged else "")
            ),
            "context": context.to_prompt_dict(),
            "dispatch_plan": dispatch_plan.to_dict(),
            "specialist_coverage": dict(coverage),
            "specialist_analyses": [
                {
                    "assignment_id": outcome.assignment_id,
                    "role_id": outcome.role_id,
                    "analysis": outcome.content,
                }
                for outcome in analyses
            ],
            "output_schema": {
                "lanes": [
                    {
                        "ground": ("The files, functions and mechanisms this lane owns and may edit"),
                        "reason": "Why this is one session's worth of work",
                        "joint": (
                            "true when this lane owns a change together with "
                            "the launch configuration that serves it, or a "
                            "move no single region contains"
                        ),
                        "fallback": (
                            "For a joint lane: the smaller change inside the "
                            "same ground to land if the main step is abandoned"
                        ),
                    }
                ],
                "cross_cutting_move": {
                    "move": ("The largest move the evidence supports that no single region contains"),
                    "lane_id": ("The 1-based lane that owns it, or 0 if no lane does"),
                    "unassigned_reason": ("Why no lane owns it, when no lane does"),
                },
            },
        }
        with _phase_timer(self.phase_durations_sec, "partition"):
            try:
                response = await self._run(
                    context,
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(payload, indent=2, sort_keys=True),
                    usage=usage,
                    allow_incomplete=True,
                    max_turns=ROUND_PARTITION_MAX_TURNS,
                    timeout_sec=min(self.timeout_sec, ROUND_PARTITION_TIMEOUT_SEC),
                    tools=False,
                    reasoning_effort=ROUND_PARTITION_EFFORT,
                )
                grounds, move, notes = self._parse_lane_grounds(response, lanes=lanes)
            except (
                OrchestrationInfrastructureError,
                OrchestrationOutputError,
            ) as error:
                grounds, move, notes = (
                    [],
                    {"status": "unavailable"},
                    [f"{type(error).__name__}: {error}"],
                )
        status = "planned"
        if not grounds:
            grounds = self._collapsed_grounds(analyses, challenged=challenged)
            status = "fallback"
            log.warning(
                "round partition unavailable; collapsing the round to a single "
                "%slane instead of %d over shared ground: %s",
                "challenger " if challenged else "",
                lanes,
                "; ".join(notes) or "no lane ground was usable",
            )
        elif move["status"] != "assigned":
            log.warning(
                "round partition gave no lane its largest cross-cutting move: %s",
                move.get("move") or move.get("field") or "the partition named none",
            )
        unbacked = [ground.lane_id for ground in grounds if ground.joint and not ground.fallback]
        if unbacked:
            log.warning(
                "joint lane(s) %s carry no fallback; a joint step that is "
                "abandoned leaves them with nothing to measure",
                ", ".join(str(lane_id) for lane_id in unbacked),
            )
        self.structured_output_diagnostics["partition"] = {
            "status": status,
            "requested": int(lanes),
            "planned": len(grounds),
            "collapsed": status == "fallback",
            "challenger_requested": challenged,
            "joint": [ground.lane_id for ground in grounds if ground.joint],
            "cross_cutting_move": move,
            "notes": notes,
            "grounds": [ground.to_dict() for ground in grounds],
        }
        return grounds

    async def revise_optimization_plan(
        self,
        context: OrchestrationContext,
        *,
        synthesis_session_id: str,
        draft_plan: str,
        critic_review: str,
        critic_verdict: str,
        specialist_outcomes: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        usage=None,
    ) -> _RevisedPlan:
        """Revise one draft exactly once without rerunning specialists."""
        revision_task = (
            "Revise the draft into the final optimization plan. Address the critic's substantive concerns exactly once."
        )
        # A resume re-enters the lane's own synthesis session, which already
        # holds the dispatch, every specialist analysis and the whole synthesis
        # conversation. Re-sending that bundle in the feedback ran the revision
        # out of context: one 12-hour run compacted the revision 17 times across
        # 7 rounds, once per lane, dropping the recap it went on to answer over.
        # The revision instructions are the resume's system prompt, so they are
        # not copied back into the payload either. The resumed session carries
        # only what the critic added and the draft it is revising.
        resumed_payload = {
            "task": revision_task,
            "draft_plan": draft_plan,
            "critic_verdict": critic_verdict,
            "critic_review": critic_review,
        }
        # The fresh-session fallback holds no prior context, so it genuinely
        # needs the whole planning bundle to revise from.
        fresh_payload = {
            **resumed_payload,
            "revision_instructions": _REVISION_SYSTEM_PROMPT.strip(),
            "context": context.to_prompt_dict(),
            "dispatch_plan": dispatch_plan.to_dict(),
            "specialist_coverage": dict(coverage),
            "specialist_outcomes": [outcome.to_dict() for outcome in specialist_outcomes],
        }
        session_id = str(synthesis_session_id or "").strip()
        timeout_sec = min(self.timeout_sec, PLAN_REVISION_TIMEOUT_SEC)
        started_at = time.monotonic()
        if session_id and self._backend_supports_resume():
            result = await self._resume_result(
                context,
                session_id=session_id,
                feedback=json.dumps(resumed_payload, indent=2, sort_keys=True),
                system_prompt=_REVISION_SYSTEM_PROMPT,
                usage=usage,
                max_turns=PLAN_REVISION_MAX_TURNS,
                timeout_sec=timeout_sec,
                role="orchestration revision",
            )
            mode = "resumed"
        else:
            if session_id:
                log.warning(
                    "orchestration backend cannot resume synthesis session %s; starting a fresh revision session",
                    session_id,
                )
            else:
                log.warning("orchestration synthesis returned no session ID; starting a fresh revision session")
            result = await self._run_result(
                context,
                system_prompt=_REVISION_SYSTEM_PROMPT,
                user_prompt=json.dumps(fresh_payload, indent=2, sort_keys=True),
                usage=usage,
                max_turns=PLAN_REVISION_MAX_TURNS,
                timeout_sec=timeout_sec,
                role="orchestration revision",
            )
            mode = "fresh"
        plan = self._validated_text(result, role="orchestration revision")
        return _RevisedPlan(
            text=plan,
            mode=mode,
            duration_sec=time.monotonic() - started_at,
        )

    async def _run(
        self,
        context: OrchestrationContext,
        *,
        system_prompt: str,
        user_prompt: str,
        usage,
        max_turns: int | None = None,
        allow_incomplete: bool = False,
        timeout_sec: int | None = None,
        tools: bool = True,
        reasoning_effort: str = "max",
    ) -> str:
        result = await self._run_result(
            context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            usage=usage,
            max_turns=max_turns,
            timeout_sec=timeout_sec,
            tools=tools,
            reasoning_effort=reasoning_effort,
            role="orchestration",
        )
        text = str(result.text or "")
        if allow_incomplete and (
            str(result.end_reason or "").strip() in {"turn_cap", "timeout"}
            or "[session ended with sdk error:" in text.lower()
        ):
            log.warning("orchestration dispatch returned an incomplete response; continuing through JSON repair")
        return self._validated_text(
            result,
            role="orchestration",
            allow_empty=True,
            allow_incomplete=allow_incomplete,
        )

    async def _run_result(
        self,
        context: OrchestrationContext,
        *,
        system_prompt: str,
        user_prompt: str,
        usage,
        role: str,
        max_turns: int | None = None,
        timeout_sec: int | None = None,
        tools: bool = True,
        reasoning_effort: str = "max",
    ) -> AgentRunResult:
        effective_timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        try:
            return await asyncio.wait_for(
                self.backend.run(
                    self._run_spec(
                        context,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_turns=max_turns,
                        timeout_sec=effective_timeout,
                        tools=tools,
                        reasoning_effort=reasoning_effort,
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(effective_timeout),
            )
        except asyncio.TimeoutError as error:
            raise OrchestrationInfrastructureError(f"{role} backend exceeded {effective_timeout}s timeout") from error
        except AgentProviderError as error:
            raise OrchestrationInfrastructureError(f"{type(error).__name__}: {error}") from error

    async def _resume_result(
        self,
        context: OrchestrationContext,
        *,
        session_id: str,
        feedback: str,
        system_prompt: str,
        usage,
        role: str,
        max_turns: int,
        timeout_sec: int,
    ) -> AgentRunResult:
        resume = getattr(self.backend, "resume")
        spec = self._run_spec(
            context,
            system_prompt=system_prompt,
            user_prompt="",
            max_turns=max_turns,
            timeout_sec=timeout_sec,
            read_only_resume=True,
        )
        try:
            return await asyncio.wait_for(
                resume(
                    spec,
                    session_id,
                    feedback,
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(timeout_sec),
            )
        except asyncio.TimeoutError as error:
            raise OrchestrationInfrastructureError(f"{role} backend exceeded {timeout_sec}s timeout") from error
        except AgentProviderError as error:
            raise OrchestrationInfrastructureError(f"{type(error).__name__}: {error}") from error

    def _run_spec(
        self,
        context: OrchestrationContext,
        *,
        system_prompt: str,
        user_prompt: str,
        max_turns: int | None,
        timeout_sec: int,
        read_only_resume: bool = False,
        tools: bool = True,
        reasoning_effort: str = "max",
    ) -> AgentRunSpec:
        """One read-only orchestration turn.

        ``tools`` off is for a step whose whole input is already in its payload.
        A step that may read the workspace will, and reading is unbounded work
        on a critical path -- worth it where the answer needs evidence the
        payload does not carry, and only there.
        """
        return AgentRunSpec(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cwd=context.workspace,
            writable=False,
            timeout_sec=timeout_sec,
            reasoning_effort=reasoning_effort,
            read_only_resume=read_only_resume,
            allow_dirty_targets=read_only_resume,
            allow_untracked=read_only_resume,
            tool_policy=AgentToolPolicy(
                read=tools,
                search=tools,
                write=False,
                shell=False,
                max_turns=(self.max_turns if max_turns is None else max_turns),
            ),
            protected_globs=["*"],
        )

    def _backend_supports_resume(self) -> bool:
        capabilities = getattr(self.backend, "capabilities", None)
        return bool(getattr(capabilities, "resumable", False) and callable(getattr(self.backend, "resume", None)))

    @staticmethod
    def _validated_text(
        result: AgentRunResult,
        *,
        role: str,
        allow_empty: bool = False,
        allow_incomplete: bool = False,
    ) -> str:
        try:
            return validated_agent_text(
                result,
                role=role,
                allow_empty=allow_empty,
                allow_incomplete=allow_incomplete,
            )
        except AgentResponseInfrastructureError as error:
            raise OrchestrationInfrastructureError(str(error)) from error
        except AgentResponseIncompleteError as error:
            raise OrchestrationOutputError(str(error)) from error


class OrchestrationService:
    """Coordinate dispatch, parallel specialists, and plan synthesis."""

    def __init__(
        self,
        *,
        agent: OrchestrationAgent,
        specialist_pool: SpecialistPool,
        definitions: Mapping[str, SpecialistDefinition],
        plan_critic: PlanCriticAgent | None = None,
    ) -> None:
        if not definitions:
            raise ValueError("definitions must not be empty")
        if set(definitions) != {definition.role_id for definition in definitions.values()}:
            raise ValueError("definition mapping keys must match specialist role_id values")
        self._agent = agent
        self._specialist_pool = specialist_pool
        self._definitions = dict(definitions)
        self._plan_critic = plan_critic

    async def run(
        self,
        context: OrchestrationContext,
        *,
        usage=None,
        lanes: int = 1,
    ) -> OrchestrationRunResult:
        """Produce a plan unless an explicit infrastructure outage prevents one.

        ``lanes`` above 1 partitions the round instead of fusing it, so each
        lane's Implementer works ground no other lane owns and every candidate
        earns its own measurement.
        """
        self._agent.structured_output_diagnostics = {}
        self._agent.phase_durations_sec = {}
        phase_durations = self._agent.phase_durations_sec
        run_started_at = time.monotonic()
        with _phase_timer(phase_durations, "dispatch"):
            dispatch_plan = await self._agent.plan_dispatch(
                context,
                self._definitions,
                usage=usage,
            )
        diagnostics = dict(self._agent.structured_output_diagnostics)

        with _phase_timer(phase_durations, "specialists"):
            specialist_run = await self._specialist_pool.run(
                dispatch_plan.assignments,
                context,
                usage=usage,
            )
        specialist_outcomes = specialist_run.outcomes
        if specialist_run.contended:
            # A probe that outlived its specialist is on the same GPU the
            # caller's canonical measurement is about to use. Reported in the
            # diagnostics because that is the channel that reaches the loop
            # in-process and in this same iteration, which is the iteration
            # whose measurement it has to stop; the loop is where it becomes a
            # recorded hazard, so ownership of that log stays in one place.
            diagnostics["probe_device_hazard"] = {
                "describe": specialist_run.reaped.describe(),
                "pids": list(specialist_run.reaped.blockers),
            }
        if (
            specialist_outcomes
            and not any(outcome.succeeded for outcome in specialist_outcomes)
            and any(
                outcome.failure is not None and outcome.failure.kind in {"backend_failure", "timeout"}
                for outcome in specialist_outcomes
            )
        ):
            raise OrchestrationInfrastructureError("specialist infrastructure failed before any analysis was produced")
        assignment_by_id = {assignment.assignment_id: assignment for assignment in dispatch_plan.assignments}
        successful_assignments = [
            assignment_by_id[outcome.assignment_id]
            for outcome in specialist_outcomes
            if outcome.succeeded and outcome.assignment_id in assignment_by_id
        ]
        covered_cases = sorted(
            {case_id for assignment in successful_assignments for case_id in assignment.target_case_ids}
        )
        coverage = {
            "successful_roles": sorted({assignment.role_id for assignment in successful_assignments}),
            "covered_cases": covered_cases,
            "missing_cases": sorted(context.case_ids - set(covered_cases)),
            "failed_roles": sorted(outcome.role_id for outcome in specialist_outcomes if not outcome.succeeded),
        }
        diagnostics["coverage"] = coverage

        synthesized_plans: list[SynthesizedPlan] = []
        if any(outcome.succeeded for outcome in specialist_outcomes):
            try:
                synthesized_plans = await self._agent.synthesize_lane_plan_results(
                    context,
                    specialist_outcomes,
                    dispatch_plan,
                    coverage,
                    lanes=lanes,
                    usage=usage,
                )
            except OrchestrationOutputError as error:
                diagnostics["synthesis"] = {
                    "status": "unavailable",
                    "message": f"{type(error).__name__}: {error}",
                }
        # Whatever synthesis recorded on the way through, which the snapshot
        # above was taken too early to hold. The round partition is the one
        # that matters: how the round was divided, whether the division was
        # bought or fallen back to, and whether a challenger was asked for are
        # answerable after the fact only from here, and a round is audited
        # after the fact or not at all.
        diagnostics.update(self._agent.structured_output_diagnostics)
        optimization_plan_executable = bool(synthesized_plans)
        if synthesized_plans:
            plans = [plan.text for plan in synthesized_plans]
        else:
            plans = [
                self._render_framework_plan(
                    context=context,
                    dispatch_plan=dispatch_plan,
                    specialist_outcomes=specialist_outcomes,
                    coverage=coverage,
                )
            ]
        planned_lanes = len(plans)
        # Every round a synthesis produced is reviewed, at any width. A wide
        # round is the one that most needs it: it commits several Implementer
        # sessions at once, and the question of whether the division earns them
        # exists only there.
        critic_eligible = bool(self._plan_critic is not None and optimization_plan_executable)
        draft_plan = ""
        critic_outcome = None
        plan_revised = False
        if critic_eligible:
            critic_outcome = await self._plan_critic.review(
                context=context,
                drafts=synthesized_plans,
                dispatch_plan=dispatch_plan,
                specialist_outcomes=specialist_outcomes,
                coverage=coverage,
                usage=usage,
            )
            diagnostics["plan_critic"] = critic_outcome.to_dict()
            # Narrowing before revision, so the round does not spend a revision
            # turn on a lane it has already decided not to run.
            synthesized_plans, narrowing_diagnostics = self._narrow_round(
                synthesized_plans,
                critic_outcome=critic_outcome,
                challenged=context.last_critic_verdict == "REPLACE",
            )
            plans = [plan.text for plan in synthesized_plans]
            diagnostics["lane_narrowing"] = narrowing_diagnostics
            # The draft the loop records is one of the plans that will run, so
            # it is read after narrowing and before revision.
            draft_plan = plans[0]
            if critic_outcome.requires_revision:
                revised, revision_diagnostics = await self._revise_round(
                    context,
                    synthesized_plans,
                    critic_outcome=critic_outcome,
                    specialist_outcomes=specialist_outcomes,
                    dispatch_plan=dispatch_plan,
                    coverage=coverage,
                    usage=usage,
                )
                plans = [plan.text for plan in revised]
                # A fallback also rewrites the text, so what was revised is read
                # from what the revision did, not from the text having changed.
                plan_revised = revision_diagnostics["status"] in {
                    "revised",
                    "partially_revised",
                }
                if revision_diagnostics["status"] == "framework_fallback":
                    optimization_plan_executable = False
                diagnostics["plan_revision"] = revision_diagnostics
        elif self._plan_critic is not None:
            diagnostics["plan_critic"] = {
                "status": "skipped_synthesis_unavailable",
            }
        diagnostics["lanes"] = {
            "requested": int(lanes),
            "planned": planned_lanes,
            # What the round actually hands to Implementer sessions, which is
            # the number the round is billed for. It differs from ``planned``
            # only when the review narrowed the round.
            "published": len(plans),
        }
        diagnostics["phase_durations_sec"] = self._phase_durations(
            phase_durations,
            critic_outcome=critic_outcome,
            revision_diagnostics=diagnostics.get("plan_revision"),
            run_started_at=run_started_at,
        )
        return OrchestrationRunResult(
            dispatch_plan=dispatch_plan,
            specialist_outcomes=specialist_outcomes,
            optimization_plan_executable=optimization_plan_executable,
            structured_output_diagnostics=diagnostics,
            optimization_plans=tuple(plans),
            optimization_plan_draft=draft_plan,
            plan_critic=critic_outcome,
            plan_revised=plan_revised,
        )

    @staticmethod
    def _narrow_round(
        drafts: Sequence[SynthesizedPlan],
        *,
        critic_outcome,
        challenged: bool,
    ) -> tuple[list[SynthesizedPlan], dict]:
        """Apply the review's per-lane width ruling to this round's drafts.

        Three decisions can move a round's width, and they are ordered here so
        they cannot contradict each other:

        1. The partition decides how wide the round is *planned*, and its
           collapse fallback is the floor it falls back to.
        2. This narrowing decides how many of those planned lanes are
           *published*. It runs last and reads the same drafts the review read,
           so on width it wins: a lane it drops does not reach an Implementer.
        3. A pending REPLACE outranks both. When the previous round's verdict
           challenged this one, exactly one drafted lane is validating the
           alternative that verdict named, and nothing downstream records which
           one -- so a drop here could silently spend the challenge. The whole
           narrowing is refused, with its reasons kept.

        A joint lane is not a fourth decision. The partition widened it because
        a body and the configuration it invalidates cannot be measured apart,
        and the review is told so -- ``joint`` and ``fallback`` reach it with
        the draft -- so a drop naming that lane is a ruling made in full view of
        the width, and it is carried out like any other. Refusing it would not
        recover the width, which the partition already spent; it would spend an
        additional Implementer session on a lane the review judged not worth
        one, and, with nothing bounding it, a partition that marked every lane
        joint would switch narrowing off. Dropping a joint lane breaks no
        invariant either: the other lanes were divided around it and stay
        disjoint without it. What is left is a cost, so the round records it --
        widened ground published nowhere and therefore never measured.

        Under all of them, one lane is the floor: a round that publishes nothing
        has spent its planning window for no measurement at all, so a ruling
        that would empty the round is refused whole rather than applied down to
        an arbitrary survivor the review never ranked.

        ``status`` records what happened to the ruling and ``block`` records
        where the ruling came from, because no count of drops distinguishes
        them: a round that kept every lane may have been asked to, or may have
        been handed a width block nobody could read. ``not_requested`` is
        reserved for the first -- a review that answered and named no lane --
        and anything the round could not carry out reports ``not_applied`` with
        the note saying why.

        Every note here says what was seen -- what the review asked for, what
        the round is carrying -- and stops there, because ``status`` and
        ``dropped`` are what say how it ended. The joint-lane cost is the one
        note written afterwards: it reports not how the ruling ended but what
        carrying it out spent, and no other field would carry it. This is also the only place that
        knows how it ended, so it is the only place entitled to log a narrowing
        as not applied.
        """
        requested = list(critic_outcome.lane_drops)
        notes = list(critic_outcome.narrowing_notes)

        joint_lanes = [index + 1 for index, draft in enumerate(drafts) if draft.joint]

        def _diagnostics(status: str, applied: Sequence) -> dict:
            dropped_joint = sorted({drop.lane_id for drop in applied} & set(joint_lanes))
            return {
                "status": status,
                "block": critic_outcome.narrowing_status,
                "planned": len(drafts),
                "kept": len(drafts) - len(applied),
                "dropped": [drop.to_dict() for drop in applied],
                # The widened lanes this round carries, and the ones it dropped.
                # Empty lists are the ordinary round: a reader can tell "no
                # joint lane" from "a joint lane the round published" without
                # leaving this block, and ``dropped_joint`` is the width the
                # partition bought and the round then never measured.
                "joint": joint_lanes,
                "dropped_joint": dropped_joint,
                "notes": notes,
            }

        def _kept_whole(
            status: str,
            note: str = "",
        ) -> tuple[list[SynthesizedPlan], dict]:
            """Publish every planned lane, at the severity that outcome earns.

            ``not_requested`` is the one outcome that lost nothing: the review
            was read and named no lane, which is the answer that means "run
            every lane". Every other one runs a lane the review asked about and
            the round could not act on, which is what an operator has to see.
            A round that was never held to a block -- one lane -- has nothing
            to report either way, so it says nothing.
            """
            if note:
                notes.append(note)
            if status != "not_requested":
                log.warning(
                    "plan critic narrowing was not applied (%s); the round keeps the %d lane(s) it planned",
                    status,
                    len(drafts),
                )
            elif critic_outcome.narrowing_status != "not_asked":
                log.info(
                    "plan critic asked for no narrowing; the round publishes the %d lane(s) it planned",
                    len(drafts),
                )
            return list(drafts), _diagnostics(status, [])

        if not requested:
            # Nothing to apply. Which of the two reasons for that -- the review
            # named no lane, or nothing it named could be used -- is the whole
            # point of the notes, so the status follows them.
            return _kept_whole("not_requested" if not notes else "not_applied")
        if challenged:
            return _kept_whole(
                "refused_challenger",
                "the round carries a challenger lane for the previous round's "
                "REPLACE, and which lane that is was never written down, so no "
                "drop can be told apart from spending the challenge",
            )
        if len(drafts) <= 1:
            return _kept_whole(
                "refused_single_lane",
                "a round publishes at least one lane, and this round planned exactly one",
            )
        applied = []
        for drop in requested:
            if 1 <= drop.lane_id <= len(drafts):
                applied.append(drop)
                continue
            notes.append(f"lane drop names lane {drop.lane_id}, which this round of {len(drafts)} lanes does not have")
        if not applied:
            return _kept_whole("not_applied")
        if len(applied) >= len(drafts):
            return _kept_whole(
                "refused_empty_round",
                "the review dropped every lane, which would leave the round "
                "nothing to measure, and it ranked no lane above another",
            )
        dropped_ids = {drop.lane_id for drop in applied}
        dropped_joint = sorted(dropped_ids & set(joint_lanes))
        if dropped_joint:
            # The drop stands -- the review was shown the width and ruled
            # anyway -- but the width is spent either way, and a round that
            # bought wider ground and then measured none of it has to say so
            # where the drops themselves are read.
            listed = ", ".join(str(lane_id) for lane_id in dropped_joint)
            notes.append(
                f"the round dropped joint lane(s) {listed}, so the wider "
                "ground the partition bought for them is spent and nothing "
                "measures it"
            )
            log.warning(
                "round dropped joint lane(s) %s; the wider ground the "
                "partition bought for them is spent and this round measures "
                "none of it",
                listed,
            )
        kept = [draft for index, draft in enumerate(drafts) if index + 1 not in dropped_ids]
        log.info(
            "round narrowed from %d lanes to %d by the plan critic: %s",
            len(drafts),
            len(kept),
            "; ".join(f"lane {drop.lane_id}: {drop.reason}" for drop in applied),
        )
        return kept, _diagnostics("narrowed", applied)

    @staticmethod
    def _phase_durations(
        measured: Mapping[str, float],
        *,
        critic_outcome,
        revision_diagnostics: Mapping[str, object] | None,
        run_started_at: float,
    ) -> dict[str, float]:
        """Report what each planning phase of this round cost, in order.

        Every number here was already being measured; only the reporting is
        new. Ten production campaigns spent a median 21.6 minutes per round on
        planning, of which about a third could only be arrived at by
        subtracting the phases that did persist their timings from the round's
        total -- which is to say the second most expensive phase of the
        planning window was the one nobody could see.

        ``total`` is this call's own wall-clock, not the sum of the parts, so
        what the named phases do not account for stays visible as the
        difference. Publishing the plans happens in the loop that called this
        and is not measured here.
        """
        durations: dict[str, float] = {}
        for name in ("dispatch", "specialists", "partition", "synthesis"):
            if name in measured:
                durations[name] = round(measured[name], 3)
        if critic_outcome is not None:
            durations["plan_critic"] = round(critic_outcome.duration_sec, 3)
        if revision_diagnostics is not None:
            durations["plan_revision"] = round(float(revision_diagnostics.get("duration_sec") or 0.0), 3)
        durations["total"] = round(time.monotonic() - run_started_at, 3)
        return durations

    async def _revise_round(
        self,
        context: OrchestrationContext,
        drafts: Sequence[SynthesizedPlan],
        *,
        critic_outcome,
        specialist_outcomes: Sequence[SpecialistOutcome],
        dispatch_plan: DispatchPlan,
        coverage: Mapping[str, object],
        usage=None,
    ) -> tuple[list[SynthesizedPlan], dict]:
        """Apply one round-level verdict to every lane it covers.

        Each lane resumes its own synthesis session, so a revision costs a short
        follow-up turn on context that already exists rather than a fresh plan,
        and the lanes revise concurrently for the same reason they were planned
        concurrently.

        A single-lane round that cannot be revised publishes the non-executable
        fallback, which is what a plan the critic distrusted and nobody could
        correct is worth. A wide round does not: the verdict was about the round,
        not about that lane being dangerous, and its siblings were revised. That
        lane keeps its draft and the diagnostics name it.
        """
        started_at = time.monotonic()

        async def _revise(draft: SynthesizedPlan) -> tuple[SynthesizedPlan, str]:
            revision = await self._agent.revise_optimization_plan(
                context,
                synthesis_session_id=draft.session_id,
                draft_plan=draft.text,
                critic_review=critic_outcome.review,
                critic_verdict=critic_outcome.verdict,
                specialist_outcomes=specialist_outcomes,
                dispatch_plan=dispatch_plan,
                coverage=coverage,
                usage=usage,
            )
            return replace(draft, text=revision.text), revision.mode

        answers = await asyncio.gather(
            *(_revise(draft) for draft in drafts),
            return_exceptions=True,
        )
        revised: list[SynthesizedPlan] = []
        unrevised: list[int] = []
        failures: list[str] = []
        modes: set[str] = set()
        for index, answer in enumerate(answers):
            if isinstance(answer, BaseException):
                log.warning(
                    "lane %d of %d could not be revised: %s: %s",
                    index + 1,
                    len(drafts),
                    type(answer).__name__,
                    answer,
                )
                unrevised.append(index + 1)
                failures.append(f"{type(answer).__name__}: {answer}")
                revised.append(drafts[index])
                continue
            plan, mode = answer
            revised.append(plan)
            modes.add(mode)
        duration_sec = time.monotonic() - started_at
        # One mode when every revised lane agreed, which is always so for a
        # single-lane round and usually so for a wide one.
        revision_mode = modes.pop() if len(modes) == 1 else "mixed"
        if not unrevised:
            log.info(
                "orchestration revision completed lanes=%d mode=%s duration=%.3fs",
                len(revised),
                revision_mode,
                duration_sec,
            )
            return revised, {
                "status": "revised",
                "critic_verdict": critic_outcome.verdict,
                "lanes": len(revised),
                "revision_mode": revision_mode,
                "duration_sec": duration_sec,
            }
        if len(drafts) == 1:
            log.warning("orchestration revision failed; publishing a non-executable framework fallback")
            return [
                replace(
                    drafts[0],
                    text=self._render_critic_revision_fallback(
                        draft_plan=drafts[0].text,
                        critic_review=critic_outcome.review,
                        critic_verdict=critic_outcome.verdict,
                    ),
                )
            ], {
                "status": "framework_fallback",
                "critic_verdict": critic_outcome.verdict,
                "revision_mode": "framework_fallback",
                "duration_sec": duration_sec,
                "message": "; ".join(failures),
            }
        return revised, {
            "status": "partially_revised",
            "critic_verdict": critic_outcome.verdict,
            "lanes": len(revised),
            "unrevised_lanes": unrevised,
            "revision_mode": revision_mode,
            "duration_sec": duration_sec,
            "message": "; ".join(failures),
        }

    @staticmethod
    def _render_critic_revision_fallback(
        *,
        draft_plan: str,
        critic_review: str,
        critic_verdict: str,
    ) -> str:
        """Preserve critic corrections when the model revision is unavailable."""
        return "\n".join(
            (
                "# Optimization plan",
                "",
                (
                    "The Orchestration revision was unavailable. Treat the "
                    "Critic review below as mandatory planning guidance before "
                    "editing; use the draft only as historical context."
                ),
                "",
                f"## Critic verdict: {critic_verdict}",
                "",
                critic_review.strip(),
                "",
                "## Original draft",
                "",
                draft_plan.strip(),
            )
        ).strip()

    @staticmethod
    def _render_framework_plan(
        *,
        context: OrchestrationContext,
        dispatch_plan: DispatchPlan,
        specialist_outcomes: Sequence[SpecialistOutcome],
        coverage: dict,
    ) -> str:
        """Render the canonical Implementer handoff without inventing advice."""
        lines = [
            "# Optimization plan",
            "",
            (
                "The planning agents did not produce a synthesized recommendation. "
                + "The Implementer must inspect the canonical Analysis evidence and source, "
                + "then formulate and execute its own evidence-grounded optimization."
            ),
            "",
            "## Planning status",
            f"- Search mode: {context.search_mode}",
            "- Successful specialist roles: " + (", ".join(coverage["successful_roles"]) or "(none)"),
            "- Failed specialist roles: " + (", ".join(coverage["failed_roles"]) or "(none)"),
            "- Covered cases: " + (", ".join(coverage["covered_cases"]) or "(none)"),
            "- Missing cases: " + (", ".join(coverage["missing_cases"]) or "(none)"),
            "",
            "## Evidence to inspect",
        ]
        evidence_paths = [context.source_map_path]
        evidence_paths.extend(
            reference.path
            for reference in context.evidence_refs
            if reference.kind
            in {
                "analysis_artifact_catalog",
                "analysis_bundle",
                "analysis_summary",
                "analysis_workflow",
            }
        )
        for path in dict.fromkeys(evidence_paths):
            lines.append(f"- {path}")
        successful = [outcome for outcome in specialist_outcomes if outcome.content is not None]
        if successful:
            lines.extend(("", "## Available specialist analyses"))
            for outcome in successful:
                lines.extend(
                    (
                        "",
                        f"### {outcome.role_id}",
                        outcome.content or "",
                    )
                )
        if dispatch_plan.normalization_notes:
            lines.extend(("", "## Dispatch normalization"))
            lines.extend(f"- {note}" for note in dispatch_plan.normalization_notes)
        return "\n".join(lines).strip()


def default_specialist_definitions() -> dict[str, SpecialistDefinition]:
    """Return the specialist registry."""
    definitions = (
        SpecialistDefinition(
            role_id="compute",
            description="Compute throughput and scheduling specialist",
            instructions=(
                "Analyze instruction throughput, dependency chains, "
                "vectorization, occupancy, register pressure, and "
                "backend-specific compute pipelines."
            ),
            capabilities=("compute", "scheduling", "registers"),
        ),
        SpecialistDefinition(
            role_id="memory",
            description="Memory hierarchy and data-layout specialist",
            instructions=(
                "Analyze memory layout, coalescing, cache behavior, data "
                "movement, reuse, bandwidth pressure, and synchronization "
                "around memory access."
            ),
            capabilities=("memory", "cache", "layout"),
        ),
        SpecialistDefinition(
            role_id="algorithm",
            description="Algorithm and implementation-structure specialist",
            instructions=(
                "Analyze algorithmic alternatives, dataflow restructuring, "
                "dispatch strategy, fusion opportunities, and multi-step "
                "structural changes."
            ),
            capabilities=("algorithm", "dataflow", "dispatch"),
        ),
    )
    return {definition.role_id: definition for definition in definitions}


def _overlaps(one: Path, other: Path) -> bool:
    """Whether either path is the other or contains it.

    Both directions: a scratch root under the workspace would break the
    read-only guarantee, and a workspace under the scratch root would be
    removed with the round's tree.
    """
    return one == other or one in other.parents or other in one.parents


def _specialist_probe_config(config: Config) -> SpecialistProbeConfig | None:
    """Resolve where and how much the round's specialists may measure.

    The scratch root is the campaign's experiments directory by default, or
    whatever ``specialist_probe_scratch_root`` names. In the default CLI path
    ``experiments_dir`` *is* ``<workspace>/forge_experiments`` -- inside the
    canonical tree, which is the one place the probe refuses to run -- and the
    fallback there is a sibling of the workspace, said out loud in the log
    rather than silently disabling the feature on every default campaign.

    Returns None when the probe is turned off, or when no placement outside the
    canonical tree can be found.
    """
    if not config.specialist_probe:
        return None
    workspace_raw = str(config.workspace or "").strip()
    if not workspace_raw:
        log.warning(
            "specialist probe disabled: this configuration declares no workspace, "
            "so there is no canonical tree to place a scratch root outside of"
        )
        return None
    workspace = Path(workspace_raw).expanduser().resolve()

    def _bounded(root: Path) -> SpecialistProbeConfig:
        return SpecialistProbeConfig(
            scratch_root=str(root),
            max_probes=int(config.specialist_probe_max),
            budget_sec=float(config.specialist_probe_budget_sec),
        )

    configured = str(config.specialist_probe_scratch_root or "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if _overlaps(workspace, candidate):
            log.warning(
                "specialist probe disabled: the configured scratch root %s overlaps the canonical tree %s",
                candidate,
                workspace,
            )
            return None
        return _bounded(candidate)

    experiments_dir = getattr(config, "experiments_dir", None)
    if experiments_dir is not None:
        candidate = Path(experiments_dir).expanduser().resolve() / "specialist_probe"
        if not _overlaps(workspace, candidate):
            return _bounded(candidate)
    fallback = workspace.parent / f"{workspace.name}.probe_scratch"
    # At warning level, like ``_no_probe`` and ``_probe_round``: ``forge_loop``
    # never calls ``logging.basicConfig``, so an info line about the campaign's
    # default layout is written nowhere at all -- and this docstring's promise
    # that the placement is "said out loud in the log" would be false.
    log.warning(
        "specialist probe scratch root placed at %s: the campaign experiments "
        "directory lies inside the canonical tree %s",
        fallback,
        workspace,
    )
    return _bounded(fallback)


def make_orchestration_service(
    *,
    config: Config,
    usage=None,
    definitions: Mapping[str, SpecialistDefinition] | None = None,
    enable_plan_critic: bool = False,
) -> OrchestrationService:
    """Build the default forge-loop planning chain through registered backends."""
    resolved_definitions = dict(default_specialist_definitions() if definitions is None else definitions)
    if not resolved_definitions:
        raise ValueError("definitions must not be empty")
    runtime = config.agent_runtime()
    orchestration_backend = create_registered_backend(
        runtime,
        probe_cwd=config.workspace,
        usage=usage,
    )
    effective_runtime = orchestration_backend.runtime
    critic_backend = (
        create_registered_backend(
            effective_runtime,
            preflight=False,
            usage=usage,
        )
        if enable_plan_critic
        else None
    )
    specialist_probe = _specialist_probe_config(config)
    specialist_agents = {
        role_id: SpecialistAgent(
            definition=definition,
            backend=create_registered_backend(
                effective_runtime,
                preflight=False,
                usage=usage,
            ),
            timeout_sec=effective_runtime.timeout_sec,
            max_turns=config.max_turns,
            probe=specialist_probe,
        )
        for role_id, definition in resolved_definitions.items()
    }
    return OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=effective_runtime.timeout_sec,
            max_turns=config.max_turns,
            min_assignments=min(2, len(resolved_definitions)),
        ),
        specialist_pool=SpecialistPool(
            specialist_agents,
            max_parallel=len(specialist_agents),
        ),
        definitions=resolved_definitions,
        plan_critic=(
            PlanCriticAgent(
                backend=critic_backend,
                timeout_sec=min(
                    effective_runtime.timeout_sec,
                    PLAN_CRITIC_TIMEOUT_SEC,
                ),
                # Per plan; a round of several is several times the reading.
                ceiling_sec=effective_runtime.timeout_sec,
                max_turns=min(
                    PLAN_CRITIC_MAX_TURNS,
                    max(1, config.max_turns),
                ),
            )
            if critic_backend is not None
            else None
        ),
    )
