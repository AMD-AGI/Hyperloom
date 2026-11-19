# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The read-only orchestration agent that runs the dispatch, partition, synthesis and revision turns."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

from kernelforge.agent_backends import (
    AgentBackend,
    AgentProviderError,
    AgentRunResult,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.structured_output import AgentAnswer, AnswerT, StructuredOutputError
from kernelforge.config.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.workflows.forge_loop.roles.agent_response import (
    AgentResponseIncompleteError,
    AgentResponseInfrastructureError,
    validated_agent_answer,
    validated_agent_text,
)
from kernelforge.workflows.forge_loop.roles.contracts import (
    CaseEvidence,
    DispatchPlan,
    EvidenceRef,
    LaneGround,
    OrchestrationContext,
    PlanCriticOutcome,
    SynthesizedPlan,
    SpecialistAssignment,
    SpecialistDefinition,
    SpecialistOutcome,
    SpecialistRound,
)
from kernelforge.workflows.forge_loop.roles.lane_prompts import lane_prompts, run_lanes_after_the_first_starts
from kernelforge.workflows.forge_loop.roles.plan_prompts import (
    LANE_SYNTHESIS_SYSTEM_PROMPT,
    PARTITION_CHALLENGER_BLOCK,
    PARTITION_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from kernelforge.workflows.forge_loop.roles.plan_requests import (
    PartitionAnswer,
    dispatch_answer_model,
    dispatch_payload,
    revision_payloads,
    synthesis_payload,
)


log = logging.getLogger(__name__)

PLAN_REVISION_MAX_TURNS = 100
PLAN_REVISION_TIMEOUT_SEC = 600
# The round partition divides ground the analyses already name, so it is given no tools and a bound short enough that
# a slow answer costs the round its partition rather than its planning window.
ROUND_PARTITION_MAX_TURNS = 8
ROUND_PARTITION_TIMEOUT_SEC = 900
# Dividing ground the analyses already name is a reading task, not the deepest reasoning the round does; the plans
# behind it keep the maximum. A ceiling rather than a value, so an operator running the campaign below it keeps their
# own setting -- see ``AgentRunSpec.resolved``.
ROUND_PARTITION_EFFORT_CEILING = "high"


@contextmanager
def _phase_timer(
    target: MutableMapping[str, float],
    name: str,
) -> Iterator[None]:
    """Record one planning phase's wall-clock against ``name``."""
    started_at = time.monotonic()
    try:
        yield
    finally:
        target[name] = target.get(name, 0.0) + (time.monotonic() - started_at)


_GLOBAL_EVIDENCE_KINDS = {
    "analysis_artifact_catalog",
    "analysis_bundle",
    "analysis_cumulative_diff",
    "analysis_summary",
    "analysis_source_map",
    "analysis_workflow",
    "latest_lesson",
    "supervisor_guidance",
}
_PROFILED_CASE_FLAGS = {
    "analysis_profiled",
    "analysis_profile_incremental",
    "analysis_checkpoint_profile_interpretation",
    "analysis_checkpoint_normalized_only",
    "analysis_checkpoint_raw_profile_only",
}


def _case_evidence_refs(case: CaseEvidence) -> dict[str, EvidenceRef]:
    """The evidence one scored case hands every specialist assigned to it, keyed by path."""
    measurement = EvidenceRef(
        kind="measurement",
        path=f"case:{case.case_id}",
        summary=(f"Canonical timing and Analysis flags for {case.case_id}."),
    )
    refs = {measurement.path: measurement}
    if case.profile_summary_path:
        if _PROFILED_CASE_FLAGS.isdisjoint(case.flags):
            kind, label = "analysis_interpretation", "Analysis interpretation"
        else:
            kind, label = "profile", "Measured profile"
        refs[case.profile_summary_path] = EvidenceRef(
            kind=kind,
            path=case.profile_summary_path,
            summary=label + f" for {case.case_id}. Flags: " + f"{', '.join(case.flags) or '(none)'}.",
        )
    return refs


class _PartitionReading(NamedTuple):
    """The lanes one partition answer divides the round into, the move it gave a lane, and what reading it found."""

    grounds: list[LaneGround]
    move: dict
    notes: list[str]


@dataclass(frozen=True)
class _RevisedPlan:
    """One critic-directed revision plus how its session was obtained."""

    text: str
    mode: str
    duration_sec: float


@dataclass(frozen=True)
class _Turn:
    """One read-only orchestration turn: its prompts, its role, and the turn, time and tool limits it runs under."""

    system_prompt: str
    user_prompt: str
    role: str = "orchestration"
    max_turns: int | None = None
    timeout_sec: int | None = None
    tools: bool = True
    max_reasoning_effort: str = ""
    shared_prompt_prefix: str = ""
    on_response_started: Callable[[], None] | None = None
    output_schema: type[AgentAnswer] | None = None


_DISPATCH_SYSTEM_PROMPT = """\
You are the read-only orchestration planner for an autonomous GPU-kernel search.

Use only the supplied evidence. Do not edit files, run shell commands, alter
measurement inputs, or present an inference as a profiler fact. Answer with the
specialist assignments the round needs.
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
        # What each planning phase cost this round, in the units the round is budgeted in.
        self.phase_durations_sec: dict[str, float] = {}

    async def plan_dispatch(
        self,
        context: OrchestrationContext,
        definitions: Mapping[str, SpecialistDefinition],
        *,
        usage=None,
    ) -> DispatchPlan:
        """Bind the specialist roles and cases the planner asked for into canonical assignments."""
        if not definitions:
            raise ValueError("specialist definitions must not be empty")
        model = dispatch_answer_model(tuple(sorted(definitions)), tuple(case.case_id for case in context.cases))
        turn = _Turn(
            _DISPATCH_SYSTEM_PROMPT,
            json.dumps(dispatch_payload(context, definitions), indent=2, sort_keys=True),
            output_schema=model,
        )
        result = await self._run_result(context, turn, usage=usage)
        notes: list[str] = []
        try:
            requested = self._validated_answer(result, model, role=turn.role).assignments
        except OrchestrationOutputError as error:
            requested = []
            notes.append(f"dispatch answer unusable: {error}")
        plan = self._bind_dispatch(
            context=context,
            definitions=definitions,
            requested=requested,
            notes=notes,
        )
        self.structured_output_diagnostics["dispatch"] = {
            "answer": result.structured_output if result.structured_output is not None else result.text,
            "normalization_notes": list(plan.normalization_notes),
        }
        return plan

    def _bind_dispatch(
        self,
        *,
        context: OrchestrationContext,
        definitions: Mapping[str, SpecialistDefinition],
        requested: Sequence,
        notes: list[str],
    ) -> DispatchPlan:
        case_evidence = {case.case_id: _case_evidence_refs(case) for case in context.cases}
        case_ids = tuple(case_evidence)
        merged: dict[str, dict[str, object]] = {}
        for assignment in requested:
            selected_cases = list(dict.fromkeys(assignment.target_case_ids))
            if not selected_cases:
                selected_cases = list(case_ids)
                notes.append(f"bound role {assignment.role_id!r} to all scored cases")
            if assignment.role_id in merged:
                notes.append(f"merged duplicate role {assignment.role_id!r}")
                current = merged[assignment.role_id]["case_ids"]
                selected_cases = list(dict.fromkeys([*current, *selected_cases]))
            merged[assignment.role_id] = {
                "case_ids": selected_cases,
                "reason": (assignment.reason.strip() or f"Framework-assigned {assignment.role_id} analysis"),
            }

        required_roles = len(definitions) if context.search_mode == "DIVERSIFY" else self.min_assignments
        for role_id in sorted(definitions.keys() - merged.keys())[: max(required_roles - len(merged), 0)]:
            merged[role_id] = {
                "case_ids": list(case_ids),
                "reason": "Framework-completed dispatch coverage",
            }
            notes.append(f"added default role {role_id!r}")

        # merged is never empty: plan_dispatch rejects empty definitions and at least one role is required.
        covered = {case_id for item in merged.values() for case_id in item["case_ids"]}
        missing = [case_id for case_id in case_ids if case_id not in covered]
        if missing:
            first_role = next(iter(merged))
            merged[first_role]["case_ids"] = [
                *merged[first_role]["case_ids"],
                *missing,
            ]
            notes.append(f"added missing cases to {first_role!r}: {', '.join(missing)}")

        return DispatchPlan(
            analysis_commit=context.analysis_commit,
            assignments=self._bound_assignments(context, merged, case_evidence),
            normalization_notes=tuple(notes),
        )

    @staticmethod
    def _bound_assignments(
        context: OrchestrationContext,
        merged: Mapping[str, Mapping[str, object]],
        case_evidence: Mapping[str, Mapping[str, EvidenceRef]],
    ) -> tuple[SpecialistAssignment, ...]:
        """One assignment per merged role, carrying the global, source-map and per-case evidence it reads."""
        global_refs = {
            reference.path: reference for reference in context.evidence_refs if reference.kind in _GLOBAL_EVIDENCE_KINDS
        }
        source_ref = EvidenceRef(
            kind="source_map",
            path=context.source_map_path,
            summary="Current source map or anchor source.",
        )
        assignments = []
        for index, (role_id, item) in enumerate(merged.items(), start=1):
            refs = {**global_refs, source_ref.path: source_ref}
            target_cases = tuple(item["case_ids"])
            for case_id in target_cases:
                refs.update(case_evidence[case_id])
            assignments.append(
                SpecialistAssignment(
                    assignment_id=f"{role_id}-{index}",
                    role_id=role_id,
                    target_case_ids=target_cases,
                    evidence_refs=tuple(refs.values()),
                    reason=str(item["reason"]),
                )
            )
        return tuple(assignments)

    async def _synthesize_optimization_plan_result(
        self,
        context: OrchestrationContext,
        specialists: SpecialistRound,
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
            for outcome in specialists.analyses
        ]
        if not analyses:
            raise ValueError("no specialist analysis is available for synthesis")
        payload = synthesis_payload(context, specialists, analyses)
        turn = _Turn(
            SYNTHESIS_SYSTEM_PROMPT,
            json.dumps(payload, indent=2, sort_keys=True),
            role="orchestration synthesis",
        )
        result = await self._run_result(context, turn, usage=usage)
        return SynthesizedPlan(
            text=self._validated_text(
                result,
                role="orchestration synthesis",
            ),
            session_id=str(result.session_id or "").strip(),
            system_prompt=turn.system_prompt,
        )

    async def synthesize_lane_plan_results(
        self,
        context: OrchestrationContext,
        specialists: SpecialistRound,
        *,
        lanes: int,
        usage=None,
    ) -> list[SynthesizedPlan]:
        """Partition the round into plans that do not compete for one another's ground."""
        analyses = specialists.analyses
        width = max(1, min(int(lanes), len(analyses)))
        if width <= 1:
            with _phase_timer(self.phase_durations_sec, "synthesis"):
                return [await self._synthesize_optimization_plan_result(context, specialists, usage=usage)]

        grounds = await self._partition_round(context, specialists, lanes=width, usage=usage)
        challenged = context.last_critic_verdict == "REPLACE"
        if len(grounds) <= 1 and not (challenged and grounds):
            # One real direction is a single-lane round, planned the way one has always been planned rather than as a
            # fan-out of width one.
            with _phase_timer(self.phase_durations_sec, "synthesis"):
                return [await self._synthesize_optimization_plan_result(context, specialists, usage=usage)]
        width = len(grounds)
        prompts = lane_prompts(context, specialists, grounds)

        async def _lane(index: int, on_response_started: Callable[[], None] | None = None) -> SynthesizedPlan | None:
            prompt, shared = prompts[index]
            turn = _Turn(
                LANE_SYNTHESIS_SYSTEM_PROMPT,
                prompt,
                role=f"orchestration lane {index + 1} synthesis",
                shared_prompt_prefix=shared,
                on_response_started=on_response_started,
            )
            result = await self._run_result(context, turn, usage=usage)
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
                system_prompt=turn.system_prompt,
                ground=grounds[index].ground,
                joint=grounds[index].joint,
                fallback=grounds[index].fallback,
            )

        # Each lane is an independent call, so one that fails is one lane lost and not the round.
        with _phase_timer(self.phase_durations_sec, "synthesis"):
            capabilities = getattr(self.backend, "capabilities", None)
            answers = await run_lanes_after_the_first_starts(
                _lane, width, shared_prompt_cache=getattr(capabilities, "shared_prompt_cache", False)
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
                # A lane that returned nothing is a lane the round paid for and cannot use.
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
        """Collapse a round that could not be divided by code to a single lane."""
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

    @staticmethod
    def _lane_grounds(answer: PartitionAnswer, *, lanes: int) -> _PartitionReading:
        """The lane grounds and the round's cross-cutting move one partition answer names."""
        notes: list[str] = []
        grounds: list[LaneGround] = []
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
        # Before the move is read, so a move can never be recorded as owned by a lane the round is over its ceiling to
        # run.
        grounds = grounds[:lanes]
        notes.extend(
            f"lane {ground.lane_id} claims joint ground and named no fallback, so an abandoned joint step leaves it "
            "nothing to measure"
            for ground in grounds
            if ground.joint and not ground.fallback
        )
        move = answer.cross_cutting_move
        if not move.move.strip():
            return _PartitionReading(
                grounds, {"status": "missing"}, [*notes, "partition named no largest cross-cutting move"]
            )
        placed, placement_notes = OrchestrationAgent._placed_move(
            move.move.strip(), move.lane_id, move.unassigned_reason.strip(), grounds
        )
        return _PartitionReading(grounds, placed, notes + placement_notes)

    @staticmethod
    def _placed_move(move: str, lane_id: int, reason: str, grounds: Sequence[LaneGround]) -> tuple[dict, list[str]]:
        """Record a named move as owned by the lane it names, or as unassigned with why."""
        notes: list[str] = []
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
        specialists: SpecialistRound,
        *,
        lanes: int,
        usage=None,
    ) -> list[LaneGround]:
        """Decide, once and with every analysis in view, what each lane owns."""
        challenged = context.last_critic_verdict == "REPLACE"
        turn = self._partition_turn(context, specialists, lanes=lanes, challenged=challenged)
        with _phase_timer(self.phase_durations_sec, "partition"):
            try:
                result = await self._run_result(context, turn, usage=usage)
                answer = self._validated_answer(result, PartitionAnswer, role=turn.role)
                grounds, move, notes = self._lane_grounds(answer, lanes=lanes)
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
            grounds = self._collapsed_grounds(specialists.analyses, challenged=challenged)
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
                move.get("move") or "the partition named none",
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

    def _partition_turn(
        self, context: OrchestrationContext, specialists: SpecialistRound, *, lanes: int, challenged: bool
    ) -> _Turn:
        """The tool-less partition turn asking for at most ``lanes`` lanes over every analysis of the round."""
        system_prompt = PARTITION_SYSTEM_PROMPT + (PARTITION_CHALLENGER_BLOCK if challenged else "")
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
            "dispatch_plan": specialists.dispatch_plan.to_dict(),
            "specialist_coverage": dict(specialists.coverage),
            "specialist_analyses": [
                {
                    "assignment_id": outcome.assignment_id,
                    "role_id": outcome.role_id,
                    "analysis": outcome.content,
                }
                for outcome in specialists.analyses
            ],
        }
        return _Turn(
            system_prompt,
            json.dumps(payload, indent=2, sort_keys=True),
            max_turns=ROUND_PARTITION_MAX_TURNS,
            timeout_sec=min(self.timeout_sec, ROUND_PARTITION_TIMEOUT_SEC),
            tools=False,
            max_reasoning_effort=ROUND_PARTITION_EFFORT_CEILING,
            output_schema=PartitionAnswer,
        )

    async def revise_optimization_plan(
        self,
        context: OrchestrationContext,
        specialists: SpecialistRound,
        *,
        draft: SynthesizedPlan,
        critic: PlanCriticOutcome,
        usage=None,
    ) -> _RevisedPlan:
        """Revise one draft exactly once without rerunning specialists."""
        resumed_payload, fresh_payload = revision_payloads(context, specialists, draft, critic)
        session_id = str(draft.session_id or "").strip()
        timeout_sec = min(self.timeout_sec, PLAN_REVISION_TIMEOUT_SEC)
        started_at = time.monotonic()
        if session_id and self._backend_supports_resume():
            result = await self._resume_result(
                context,
                _Turn(
                    draft.system_prompt,
                    "",
                    role="orchestration revision",
                    max_turns=PLAN_REVISION_MAX_TURNS,
                    timeout_sec=timeout_sec,
                ),
                session_id=session_id,
                feedback=json.dumps(resumed_payload, indent=2, sort_keys=True),
                usage=usage,
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
                _Turn(
                    REVISION_SYSTEM_PROMPT,
                    json.dumps(fresh_payload, indent=2, sort_keys=True),
                    role="orchestration revision",
                    max_turns=PLAN_REVISION_MAX_TURNS,
                    timeout_sec=timeout_sec,
                ),
                usage=usage,
            )
            mode = "fresh"
        plan = self._validated_text(result, role="orchestration revision")
        return _RevisedPlan(
            text=plan,
            mode=mode,
            duration_sec=time.monotonic() - started_at,
        )

    async def _run_result(self, context: OrchestrationContext, turn: _Turn, *, usage) -> AgentRunResult:
        effective_timeout = self.timeout_sec if turn.timeout_sec is None else turn.timeout_sec
        try:
            return await asyncio.wait_for(
                self.backend.run(
                    self._run_spec(context, turn, timeout_sec=effective_timeout),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(effective_timeout),
            )
        except asyncio.TimeoutError as error:
            raise OrchestrationInfrastructureError(
                f"{turn.role} backend exceeded {effective_timeout}s timeout"
            ) from error
        except AgentProviderError as error:
            raise OrchestrationInfrastructureError(f"{type(error).__name__}: {error}") from error

    async def _resume_result(
        self,
        context: OrchestrationContext,
        turn: _Turn,
        *,
        session_id: str,
        feedback: str,
        usage,
    ) -> AgentRunResult:
        timeout_sec = turn.timeout_sec
        role = turn.role
        resume = getattr(self.backend, "resume")
        spec = self._run_spec(context, turn, timeout_sec=timeout_sec, read_only_resume=True)
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
        turn: _Turn,
        *,
        timeout_sec: int,
        read_only_resume: bool = False,
    ) -> AgentRunSpec:
        """One read-only orchestration turn."""
        tools = turn.tools
        return AgentRunSpec(
            system_prompt=turn.system_prompt,
            user_prompt=turn.user_prompt,
            cwd=context.workspace,
            writable=False,
            timeout_sec=timeout_sec,
            role=turn.role,
            max_reasoning_effort=turn.max_reasoning_effort,
            read_only_resume=read_only_resume,
            allow_dirty_targets=read_only_resume,
            allow_untracked=read_only_resume,
            tool_policy=AgentToolPolicy(
                read=tools,
                search=tools,
                write=False,
                shell=False,
                max_turns=(self.max_turns if turn.max_turns is None else turn.max_turns),
            ),
            protected_path_rules=FORGE_PROTECTED_PATHS,
            protected_globs=["*"],
            shared_prompt_prefix=turn.shared_prompt_prefix,
            on_response_started=turn.on_response_started,
            output_schema=turn.output_schema,
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
    ) -> str:
        try:
            return validated_agent_text(
                result,
                role=role,
                allow_empty=allow_empty,
            )
        except AgentResponseInfrastructureError as error:
            raise OrchestrationInfrastructureError(str(error)) from error
        except AgentResponseIncompleteError as error:
            raise OrchestrationOutputError(str(error)) from error

    @staticmethod
    def _validated_answer(result: AgentRunResult, model: type[AnswerT], *, role: str) -> AnswerT:
        try:
            return validated_agent_answer(result, model, role=role)
        except AgentResponseInfrastructureError as error:
            raise OrchestrationInfrastructureError(str(error)) from error
        except (AgentResponseIncompleteError, StructuredOutputError) as error:
            raise OrchestrationOutputError(str(error)) from error
