"""Read-only review of one synthesized forge-loop optimization plan."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, field_validator

from kernelforge.agent_backends import (
    AgentBackend,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.structured_output import AgentAnswer, StructuredOutputError
from kernelforge.workflows.forge_loop.roles.agent_response import (
    AgentResponseIncompleteError,
    AgentResponseInfrastructureError,
    validated_agent_answer,
)
from kernelforge.workflows.forge_loop.roles.contracts import (
    LaneDrop,
    OrchestrationContext,
    PlanCriticOutcome,
    SpecialistRound,
    SynthesizedPlan,
)


PLAN_CRITIC_MAX_TURNS = 100
PLAN_CRITIC_TIMEOUT_SEC = 600
_ERROR_DETAIL_MAX_CHARS = 2000

log = logging.getLogger(__name__)


def _bounded_error_detail(error: Exception) -> str:
    """Return one bounded log-safe line for persisted critic failures."""
    message = " ".join(str(error).split())
    detail = f"{type(error).__name__}: {message}".rstrip()
    if len(detail) <= _ERROR_DETAIL_MAX_CHARS:
        return detail
    return detail[: _ERROR_DETAIL_MAX_CHARS - 3].rstrip() + "..."


class CriticLaneDrop(AgentAnswer):
    """One lane the review judges not worth an Implementer session of its own."""

    lane_id: int = Field(description="The integer lane_id from draft_lane_plans.")
    reason: str = Field(description="One sentence saying what you found about this lane.")

    @field_validator("lane_id")
    @classmethod
    def _names_a_lane(cls, value: int) -> int:
        if value < 1:
            raise ValueError("lane_id must name a lane, counting from 1")
        return value

    @field_validator("reason")
    @classmethod
    def _states_a_reason(cls, value: str) -> str:
        line = " ".join(value.split())
        if not line:
            raise ValueError("a dropped lane needs the reason it is dropped")
        return line


class PlanCriticAnswer(AgentAnswer):
    """What one review of a round's plans comes to."""

    verdict: Literal["ACCEPT", "REVISE", "REPLACE"] = Field(
        description=(
            "ACCEPT: the draft is worth executing. REVISE: the same broad route needs evidence, scope, sequencing, or "
            "risk corrections. REPLACE: the draft continues a strategically dominated implementation route and should "
            "instead validate a concrete alternative."
        )
    )
    review: str = Field(
        description=(
            "The review in free-form Markdown, citing the source, profiling, benchmark, candidate-history, or "
            "knowledge paths behind its important claims."
        )
    )
    lane_narrowing: list[CriticLaneDrop] = Field(
        description=(
            "Every lane of the round not worth an Implementer session of its own. Every lane not named is run, so an "
            "empty list runs the whole round; a review of one plan leaves it empty."
        )
    )

    @field_validator("review")
    @classmethod
    def _says_something(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the review is empty")
        return value

    def lane_drops(self) -> tuple[LaneDrop, ...]:
        """The lanes the review drops, each once, in the order it named them."""
        drops: dict[int, LaneDrop] = {}
        for entry in self.lane_narrowing:
            drops.setdefault(entry.lane_id, LaneDrop(lane_id=entry.lane_id, reason=entry.reason))
        return tuple(drops.values())


_PLAN_CRITIC_SYSTEM_PROMPT = """\
You are the read-only critic for one GPU-kernel optimization plan. Review the
draft independently. Do not edit files, run shell commands, benchmark, profile,
or rewrite the plan yourself. You may read and search the supplied workspace
evidence paths when needed.

Use this checklist to guide judgment, but do not mechanically repeat every item:
- Question whether the current kernel, algorithm, programming model, and
  execution units should continue to exist.
- Verify that the bottleneck claim is supported by profiling and a plausible
  performance model.
- Compare the current route's performance ceiling with materially different
  alternatives.
- Search supplied source, dependency, and knowledge paths for existing GEMM,
  MFMA, fused-kernel, library, or alternate-backend implementations before
  recommending more work on the current implementation.
- Check for omitted structural options in fusion, algorithms, dataflow, layout,
  and hardware instructions.
- Require a clear causal link between every proposed change and the measured
  bottleneck.
- Detect whether repeated local gains have kept the search at one optimization
  level for too long.
- When useful, request one isolated challenger for a high-potential alternative.
  Exploration may regress temporarily, but the final candidate must still beat
  the canonical best under the unchanged correctness and KEEP gates.
- Check instruction, register, memory, occupancy, compiler, and implementation
  feasibility.
- Require explicit success, failure, stop, and route-switch conditions.
- Check correctness, boundary inputs, numerical accuracy, and representative
  workload coverage.
- Compare prior attempts and reject repetition without new evidence.
- Judge expected gain, implementation time, and opportunity cost for one
  iteration.

Cite concrete source, profiling, benchmark, candidate-history, or knowledge
paths for important claims. State uncertainty when evidence is missing. Focus
on issues that would change this iteration's plan rather than generic advice.

Write the review itself in `review` and route the draft with `verdict`.
"""

_PLAN_CRITIC_ROUND_BLOCK = """\

This round was divided into several lanes, each planned on its own ground and
implemented concurrently by its own Implementer. Review the division as well as
the plans, and answer for the round as a whole:

- Is any lane's ground not worth an Implementer session of its own? A lane the
  round cannot use still costs a full session.
- Do two lanes amount to the same change described differently? Their scores
  would then be one answer bought twice, and neither could be stacked on the
  other.
- Would any lane have to edit code another lane owns to carry out its plan?
- Taken together, is the round still working at one optimization level that has
  stopped paying, when the evidence supports a materially different route?

One verdict covers the round and applies to every lane that runs. REVISE and
REPLACE are for what the round should do differently, not for a wording
preference in one plan.

How wide the round runs is a separate answer, given per lane in
`lane_narrowing`: name every lane you judge not worth an Implementer session of
its own -- ground the evidence does not support, or another lane's change in
different words -- with the reason you found. The reason is recorded with the
round.

A lane whose `joint` is true was given wider ground than a region on purpose:
it holds a change and the launch configuration that change invalidates, whose
parts cannot be measured apart, and its `fallback` is the smaller change inside
that same ground its Implementer lands if the joint step does not converge.
That width is already spent by the time you read this. Naming such a lane drops
it, exactly as for any other lane, and the round then measures nothing on the
ground it widened -- so weigh the `fallback` as what the lane still returns, and
name the lane only if even that is not worth its session.

At least one lane always runs, so drop only what the round is better off
without.
"""


def build_plan_critic_prompts(
    *,
    context: OrchestrationContext,
    drafts: Sequence[SynthesizedPlan],
    specialists: SpecialistRound,
) -> tuple[str, str]:
    """Build one bounded critic request from persisted planning evidence."""
    if not drafts:
        raise ValueError("plan critic needs at least one draft to review")
    payload = {
        "task": (
            "Audit the draft plan before implementation. Decide whether to "
            "accept it, revise it, or replace its implementation route."
        ),
        "context": context.to_prompt_dict(),
        "dispatch_plan": specialists.dispatch_plan.to_dict(),
        "specialist_coverage": dict(specialists.coverage),
        "specialist_outcomes": [outcome.to_dict() for outcome in specialists.outcomes],
    }
    if len(drafts) == 1:
        payload["draft_plan"] = drafts[0].text
        return (
            _PLAN_CRITIC_SYSTEM_PROMPT,
            json.dumps(payload, indent=2, sort_keys=True),
        )
    payload["task"] = (
        "Audit this round's lane plans and the division that produced them, "
        "before any of them is implemented. Decide whether to accept the "
        "round, revise it, or replace its implementation route."
    )
    payload["draft_lane_plans"] = [
        {
            "lane_id": index + 1,
            "ground": draft.ground,
            "joint": draft.joint,
            "fallback": draft.fallback,
            "draft_plan": draft.text,
        }
        for index, draft in enumerate(drafts)
    ]
    return (
        _PLAN_CRITIC_SYSTEM_PROMPT + _PLAN_CRITIC_ROUND_BLOCK,
        json.dumps(payload, indent=2, sort_keys=True),
    )


class PlanCriticAgent:
    """Run one fail-open, read-only plan review in an independent session."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        timeout_sec: int,
        max_turns: int = PLAN_CRITIC_MAX_TURNS,
        ceiling_sec: int | None = None,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self.backend = backend
        # Budget for one plan.
        self.timeout_sec = timeout_sec
        self.ceiling_sec = max(timeout_sec, int(ceiling_sec or timeout_sec))
        self.max_turns = max_turns

    def _budget_for(self, drafts: int) -> int:
        """The wall-clock a review of this many plans is allowed."""
        return min(self.timeout_sec * max(1, drafts), self.ceiling_sec)

    def _review_spec(
        self, context: OrchestrationContext, system_prompt: str, user_prompt: str, budget_sec: int
    ) -> AgentRunSpec:
        """The read-only session that reviews one round's plans."""
        return AgentRunSpec(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            role="plan critic",
            cwd=context.workspace,
            writable=False,
            timeout_sec=budget_sec,
            tool_policy=AgentToolPolicy(
                read=True,
                search=True,
                write=False,
                shell=False,
                max_turns=self.max_turns,
            ),
            protected_globs=["*"],
            output_schema=PlanCriticAnswer,
        )

    async def review(
        self,
        *,
        context: OrchestrationContext,
        drafts: Sequence[SynthesizedPlan],
        specialists: SpecialistRound,
        usage=None,
    ) -> PlanCriticOutcome:
        """Review one round; every backend/output failure accepts fail-open."""
        system_prompt, user_prompt = build_plan_critic_prompts(context=context, drafts=drafts, specialists=specialists)
        budget_sec = self._budget_for(len(drafts))
        started_at = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.backend.run(
                    self._review_spec(context, system_prompt, user_prompt, budget_sec),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(budget_sec),
            )
        except Exception as error:  # noqa: BLE001 - isolation boundary: one round's plan-critic review
            return self._fail_open(error, started_at=started_at)

        try:
            answer = validated_agent_answer(result, PlanCriticAnswer, role="plan critic")
        except (
            AgentResponseInfrastructureError,
            AgentResponseIncompleteError,
            StructuredOutputError,
        ) as error:
            return self._fail_open(error, started_at=started_at)

        duration_sec = time.monotonic() - started_at
        lane_drops = answer.lane_drops()
        log.info(
            "plan critic completed verdict=%s lane_drops=%d duration=%.3fs",
            answer.verdict,
            len(lane_drops),
            duration_sec,
        )
        return PlanCriticOutcome(
            verdict=answer.verdict,
            review=answer.review,
            duration_sec=duration_sec,
            lane_drops=lane_drops,
        )

    @staticmethod
    def _fail_open(
        error: Exception,
        *,
        started_at: float,
    ) -> PlanCriticOutcome:
        detail = _bounded_error_detail(error)
        duration_sec = time.monotonic() - started_at
        log.warning(
            "plan critic failed open to the draft after %.3fs: %s",
            duration_sec,
            detail,
            exc_info=error,
        )
        return PlanCriticOutcome(
            verdict="NOT_REVIEWED",
            error=detail,
            duration_sec=duration_sec,
        )
