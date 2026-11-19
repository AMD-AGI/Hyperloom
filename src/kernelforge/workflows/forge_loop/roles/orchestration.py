"""Read-only orchestration sessions for specialist dispatch and synthesis."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.agent_backends import create_registered_backend
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.config import Config
from kernelforge.workflows.forge_loop.roles.contracts import (
    DispatchPlan,
    OrchestrationContext,
    OrchestrationRunResult,
    PlanCriticOutcome,
    SynthesizedPlan,
    SpecialistDefinition,
    SpecialistOutcome,
    SpecialistRound,
)
from kernelforge.workflows.forge_loop.roles.critic_fallback import render_critic_fallback
from kernelforge.workflows.forge_loop.roles.orchestration_agent import (
    OrchestrationAgent,
    OrchestrationInfrastructureError,
    OrchestrationOutputError,
    _phase_timer,
)
from kernelforge.workflows.forge_loop.roles.specialists import (
    SpecialistAgent,
    SpecialistPool,
    SpecialistProbeConfig,
)
from kernelforge.workflows.forge_loop.roles.plan_critic import (
    PLAN_CRITIC_MAX_TURNS,
    PLAN_CRITIC_TIMEOUT_SEC,
    PlanCriticAgent,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RoundReview:
    """What the plan critic made of a synthesized round: the plans it publishes and how they were reached."""

    plans: list[str]
    draft_plan: str
    critic_outcome: PlanCriticOutcome
    plan_revised: bool
    executable: bool
    diagnostics: dict


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
        self.has_plan_critic = plan_critic is not None

    async def run(
        self,
        context: OrchestrationContext,
        *,
        usage=None,
        lanes: int = 1,
    ) -> OrchestrationRunResult:
        """Produce a plan unless an explicit infrastructure outage prevents one."""
        self._agent.structured_output_diagnostics = {}
        self._agent.phase_durations_sec = {}
        phase_durations = self._agent.phase_durations_sec
        run_started_at = time.monotonic()
        specialists, diagnostics = await self._specialist_round(context, phase_durations, usage=usage)
        dispatch_plan, specialist_outcomes = specialists.dispatch_plan, specialists.outcomes
        any_succeeded = any(outcome.succeeded for outcome in specialist_outcomes)

        synthesized_plans: list[SynthesizedPlan] = []
        if any_succeeded:
            try:
                synthesized_plans = await self._agent.synthesize_lane_plan_results(
                    context, specialists, lanes=lanes, usage=usage
                )
            except OrchestrationOutputError as error:
                diagnostics["synthesis"] = {
                    "status": "unavailable",
                    "message": f"{type(error).__name__}: {error}",
                }
        # Whatever synthesis recorded on the way through, which the snapshot above was taken too early to hold.
        diagnostics.update(self._agent.structured_output_diagnostics)
        optimization_plan_executable = bool(synthesized_plans)
        if synthesized_plans:
            plans = [plan.text for plan in synthesized_plans]
        else:
            plans = [self._render_framework_plan(context, specialists)]
        planned_lanes = len(plans)
        draft_plan = ""
        critic_outcome = None
        plan_revised = False
        # A round the loop sends for review is reviewed once a synthesis produced it, at any width.
        if self._plan_critic is not None and context.review_plan and optimization_plan_executable:
            review = await self._review_round(context, synthesized_plans, specialists, usage=usage)
            diagnostics.update(review.diagnostics)
            plans, draft_plan, critic_outcome = review.plans, review.draft_plan, review.critic_outcome
            plan_revised, optimization_plan_executable = review.plan_revised, review.executable
        elif self._plan_critic is not None:
            status = "skipped_synthesis_unavailable" if context.review_plan else "skipped_not_requested"
            diagnostics["plan_critic"] = {"status": status}
        diagnostics["lanes"] = {
            "requested": int(lanes),
            "planned": planned_lanes,
            # What the round actually hands to Implementer sessions, which is the number the round is billed for.
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

    async def _specialist_round(
        self, context: OrchestrationContext, phase_durations: MutableMapping[str, float], *, usage
    ) -> tuple[SpecialistRound, dict]:
        """Dispatch the round and run its specialists, with the diagnostics they recorded."""
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
            # A probe that outlived its specialist is on the same GPU the caller's canonical measurement is about to
            # use.
            diagnostics["probe_device_hazard"] = {
                "describe": specialist_run.reaped.describe(),
                "pids": list(specialist_run.reaped.blockers),
            }
        any_succeeded = any(outcome.succeeded for outcome in specialist_outcomes)
        if not any_succeeded and any(
            outcome.failure is not None and outcome.failure.kind in {"backend_failure", "timeout"}
            for outcome in specialist_outcomes
        ):
            raise OrchestrationInfrastructureError("specialist infrastructure failed before any analysis was produced")
        coverage = self._round_coverage(context.case_ids, dispatch_plan, specialist_outcomes)
        diagnostics["coverage"] = coverage
        return SpecialistRound(dispatch_plan, specialist_outcomes, coverage), diagnostics

    async def _review_round(
        self,
        context: OrchestrationContext,
        synthesized_plans: list[SynthesizedPlan],
        specialists: SpecialistRound,
        *,
        usage,
    ) -> _RoundReview:
        """Have the plan critic review every synthesized lane, then narrow and revise the round as it rules."""
        diagnostics: dict = {}
        plan_revised = False
        executable = True
        critic_outcome = await self._plan_critic.review(
            context=context,
            drafts=synthesized_plans,
            specialists=specialists,
            usage=usage,
        )
        diagnostics["plan_critic"] = critic_outcome.to_dict()
        # Narrowing before revision, so the round does not spend a revision turn on a lane it has already decided
        # not to run.
        synthesized_plans, narrowing_diagnostics = self._narrow_round(
            synthesized_plans,
            critic_outcome=critic_outcome,
            challenged=context.last_critic_verdict == "REPLACE",
        )
        plans = [plan.text for plan in synthesized_plans]
        diagnostics["lane_narrowing"] = narrowing_diagnostics
        # The draft the loop records is one of the plans that will run, so it is read after narrowing and before
        # revision.
        draft_plan = plans[0]
        if critic_outcome.requires_revision:
            revised, revision_diagnostics = await self._revise_round(
                context,
                synthesized_plans,
                specialists,
                critic_outcome=critic_outcome,
                usage=usage,
            )
            plans = [plan.text for plan in revised]
            # A fallback also rewrites the text, so what was revised is read from what the revision did, not from
            # the text having changed.
            plan_revised = revision_diagnostics["status"] in {
                "revised",
                "partially_revised",
            }
            if revision_diagnostics["status"] == "framework_fallback":
                executable = False
            diagnostics["plan_revision"] = revision_diagnostics
        return _RoundReview(
            plans=plans,
            draft_plan=draft_plan,
            critic_outcome=critic_outcome,
            plan_revised=plan_revised,
            executable=executable,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _round_coverage(
        case_ids: frozenset[str], dispatch_plan: DispatchPlan, outcomes: Sequence[SpecialistOutcome]
    ) -> dict[str, list[str]]:
        """The roles that succeeded or failed this round and the scored cases their analyses cover."""
        assignment_by_id = {assignment.assignment_id: assignment for assignment in dispatch_plan.assignments}
        successful_assignments = [assignment_by_id[outcome.assignment_id] for outcome in outcomes if outcome.succeeded]
        covered_cases = sorted(
            {case_id for assignment in successful_assignments for case_id in assignment.target_case_ids}
        )
        return {
            "successful_roles": sorted({assignment.role_id for assignment in successful_assignments}),
            "covered_cases": covered_cases,
            "missing_cases": sorted(case_ids - set(covered_cases)),
            "failed_roles": sorted(outcome.role_id for outcome in outcomes if not outcome.succeeded),
        }

    @staticmethod
    def _narrow_round(
        drafts: Sequence[SynthesizedPlan],
        *,
        critic_outcome,
        challenged: bool,
    ) -> tuple[list[SynthesizedPlan], dict]:
        """Apply the review's per-lane width ruling to this round's drafts."""
        requested = list(critic_outcome.lane_drops)
        notes: list[str] = []

        joint_lanes = [index + 1 for index, draft in enumerate(drafts) if draft.joint]

        def _diagnostics(status: str, applied: Sequence) -> dict:
            dropped_joint = sorted({drop.lane_id for drop in applied} & set(joint_lanes))
            return {
                "status": status,
                "planned": len(drafts),
                "kept": len(drafts) - len(applied),
                "dropped": [drop.to_dict() for drop in applied],
                # The widened lanes this round carries, and the ones it dropped.
                "joint": joint_lanes,
                "dropped_joint": dropped_joint,
                "notes": notes,
            }

        def _kept_whole(
            status: str,
            note: str = "",
        ) -> tuple[list[SynthesizedPlan], dict]:
            """Publish every planned lane, at the severity that outcome earns."""
            if note:
                notes.append(note)
            if status != "not_requested":
                log.warning(
                    "plan critic narrowing was not applied (%s); the round keeps the %d lane(s) it planned",
                    status,
                    len(drafts),
                )
            elif len(drafts) > 1 and not critic_outcome.fail_open:
                log.info(
                    "plan critic asked for no narrowing; the round publishes the %d lane(s) it planned",
                    len(drafts),
                )
            return list(drafts), _diagnostics(status, [])

        if not requested:
            return _kept_whole("not_requested")
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
            # The drop stands -- the review was shown the width and ruled anyway -- but the width is spent either way,
            # and a round that bought wider ground and then measured none of it has to say so where the drops
            # themselves are read.
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
        """Report what each planning phase of this round cost, in order."""
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
        specialists: SpecialistRound,
        *,
        critic_outcome,
        usage=None,
    ) -> tuple[list[SynthesizedPlan], dict]:
        """Apply one round-level verdict to every lane it covers."""
        started_at = time.monotonic()

        async def _revise(draft: SynthesizedPlan) -> tuple[SynthesizedPlan, str]:
            revision = await self._agent.revise_optimization_plan(
                context, specialists, draft=draft, critic=critic_outcome, usage=usage
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
        # One mode when every revised lane agreed, which is always so for a single-lane round and usually so for a
        # wide one.
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
            return self._framework_fallback_round(drafts[0], critic_outcome, duration_sec, failures)
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
    def _framework_fallback_round(
        draft: SynthesizedPlan, critic_outcome, duration_sec: float, failures: Sequence[str]
    ) -> tuple[list[SynthesizedPlan], dict]:
        """Replace a single lane whose revision failed with the non-executable critic fallback."""
        log.warning("orchestration revision failed; publishing a non-executable framework fallback")
        return [
            replace(
                draft,
                text=render_critic_fallback(
                    draft_plan=draft.text,
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

    @staticmethod
    def _render_framework_plan(context: OrchestrationContext, specialists: SpecialistRound) -> str:
        """Render the canonical Implementer handoff without inventing advice."""
        coverage = specialists.coverage
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
        successful = specialists.analyses
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
        if specialists.dispatch_plan.normalization_notes:
            lines.extend(("", "## Dispatch normalization"))
            lines.extend(f"- {note}" for note in specialists.dispatch_plan.normalization_notes)
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
    """Whether either path is the other or contains it."""
    return one == other or one in other.parents or other in one.parents


def _specialist_probe_config(config: Config) -> SpecialistProbeConfig | None:
    """Resolve where and how much the round's specialists may measure."""
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
    # At warning level, like ``_no_probe`` and ``_probe_round``: ``forge_loop`` never calls ``logging.basicConfig``,
    # so an info line about the campaign's default layout is written nowhere at all -- and this docstring's promise
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
    runtime = agent_runtime(config)
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
        plan_critic=_plan_critic_agent(critic_backend, effective_runtime.timeout_sec, config.max_turns),
    )


def _plan_critic_agent(critic_backend, timeout_sec: int, max_turns: int) -> PlanCriticAgent | None:
    """The round's plan critic on its own backend, or None when the critic is disabled."""
    return (
        PlanCriticAgent(
            backend=critic_backend,
            timeout_sec=min(
                timeout_sec,
                PLAN_CRITIC_TIMEOUT_SEC,
            ),
            # Per plan; a round of several is several times the reading.
            ceiling_sec=timeout_sec,
            max_turns=min(
                PLAN_CRITIC_MAX_TURNS,
                max(1, max_turns),
            ),
        )
        if critic_backend is not None
        else None
    )
