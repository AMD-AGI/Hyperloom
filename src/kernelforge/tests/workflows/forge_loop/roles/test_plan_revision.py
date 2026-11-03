# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Revising a plan after its review: one revision, the session it resumes, and the summary line every plan opens with."""

from __future__ import annotations

import json

import pytest

from kernelforge.agent_backends import AgentCapabilities, AgentRunResult
from kernelforge.workflows.forge_loop.roles.contracts import (
    DispatchPlan,
    PlanCriticOutcome,
    SpecialistRound,
    SynthesizedPlan,
)
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent, OrchestrationService
from kernelforge.workflows.forge_loop.roles.specialists import SpecialistAgent, SpecialistPool
from kernelforge.workflows.forge_loop.roles.plan_critic import PlanCriticAgent
from kernelforge.workflows.forge_loop.roles.summary_line import PLAN_SUMMARY_RULE
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    GROUND_A,
    GROUND_B,
    _PerCallBackend,
    _QueuedBackend,
    _StaticBackend,
    _assignment,
    _compute_and_memory_round,
    _context,
    _coverage,
    _definitions,
    _dispatch_payload,
    _outcome,
    _partition,
)


class _ResumableQueuedBackend(_QueuedBackend):
    capabilities = AgentCapabilities(resumable=True)

    def __init__(
        self,
        results: list[str | AgentRunResult],
    ) -> None:
        super().__init__(results)
        self.resumes = []

    async def resume(
        self,
        spec,
        session_id,
        feedback,
        usage=None,
    ) -> AgentRunResult:
        self.resumes.append((spec, session_id, feedback))
        result = self.results.pop(0)
        return result if isinstance(result, AgentRunResult) else AgentRunResult(text=result, session_id=session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["REVISE", "REPLACE"])
async def test_plan_critic_triggers_exactly_one_revision(verdict) -> None:
    draft = "# Optimization plan\nContinue VALU tuning."
    revised = "# Optimization plan\nBenchmark the existing GEMM path."
    orchestration_backend = _QueuedBackend([_dispatch_payload(), draft, revised])
    specialist_backend = _StaticBackend("Algorithm analysis")
    critic_backend = _StaticBackend(f"VERDICT: {verdict}\n\nCompare the existing GEMM path.")
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1800,
            max_turns=20,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=specialist_backend,
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
        plan_critic=PlanCriticAgent(
            backend=critic_backend,
            timeout_sec=1,
        ),
    )

    result = await service.run(_context())

    assert result.optimization_plan_draft == draft
    assert result.optimization_plans[0] == revised
    assert result.plan_revised is True
    assert result.plan_critic is not None
    assert result.plan_critic.verdict == verdict
    assert len(orchestration_backend.specs) == 3
    assert orchestration_backend.specs[2].tool_policy.max_turns == 100
    assert orchestration_backend.specs[2].timeout_sec == 600
    assert result.structured_output_diagnostics["plan_revision"]["revision_mode"] == "fresh"
    assert specialist_backend.calls == 1
    assert critic_backend.calls == 1
    revision_payload = json.loads(orchestration_backend.specs[2].user_prompt)
    assert revision_payload["draft_plan"] == draft
    assert revision_payload["critic_verdict"] == verdict
    assert "existing GEMM" in revision_payload["critic_review"]


@pytest.mark.asyncio
async def test_plan_revision_resumes_the_synthesis_session() -> None:
    draft = "# Optimization plan\nContinue VALU tuning."
    revised = "# Optimization plan\nBenchmark the existing GEMM path."
    orchestration_backend = _ResumableQueuedBackend(
        [
            _dispatch_payload(),
            AgentRunResult(
                text=draft,
                session_id="synthesis-session",
            ),
            revised,
        ]
    )
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1800,
            max_turns=20,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
        plan_critic=PlanCriticAgent(
            backend=_StaticBackend("VERDICT: REVISE\n\nCompare the existing GEMM path."),
            timeout_sec=1,
        ),
    )

    result = await service.run(_context())

    assert result.optimization_plans[0] == revised
    assert len(orchestration_backend.specs) == 2
    assert len(orchestration_backend.resumes) == 1
    spec, session_id, feedback = orchestration_backend.resumes[0]
    assert session_id == "synthesis-session"
    assert spec.read_only_resume is True
    assert spec.allow_dirty_targets is True
    assert spec.allow_untracked is True
    assert spec.tool_policy.max_turns == 100
    assert spec.timeout_sec == 600
    revision_payload = json.loads(feedback)
    assert revision_payload["draft_plan"] == draft
    # The resumed session already holds the planning bundle, and the revision instructions are the system prompt;
    # neither is copied back into the payload.
    assert "For REPLACE, discard the dominated implementation route" in (spec.system_prompt)
    assert "revision_instructions" not in revision_payload
    revision_diagnostics = result.structured_output_diagnostics["plan_revision"]
    assert revision_diagnostics["status"] == "revised"
    assert revision_diagnostics["critic_verdict"] == "REVISE"
    assert revision_diagnostics["revision_mode"] == "resumed"
    assert revision_diagnostics["duration_sec"] >= 0


@pytest.mark.asyncio
async def test_a_resumed_revision_carries_only_what_the_session_lacks() -> None:
    """The synthesis session already holds the whole planning bundle."""
    backend = _ResumableQueuedBackend(["# Revised plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    revised = await agent.revise_optimization_plan(
        _context(),
        specialists,
        draft=SynthesizedPlan(text="# Draft plan", session_id="synthesis-session"),
        critic=PlanCriticOutcome(verdict="REVISE", review="Compare the existing GEMM path."),
    )

    assert revised.mode == "resumed"
    spec, session_id, feedback = backend.resumes[0]
    assert session_id == "synthesis-session"
    payload = json.loads(feedback)
    assert payload["draft_plan"] == "# Draft plan"
    assert payload["critic_verdict"] == "REVISE"
    assert "existing GEMM" in payload["critic_review"]
    # The revision instructions are the system prompt, not a second copy inside the payload; the rest of the bundle is
    # already in the resumed session.
    assert "For REPLACE, discard the dominated implementation route" in (spec.system_prompt)
    for absent in (
        "revision_instructions",
        "context",
        "dispatch_plan",
        "specialist_outcomes",
        "specialist_coverage",
    ):
        assert absent not in payload


@pytest.mark.asyncio
async def test_a_fresh_revision_carries_the_whole_bundle() -> None:
    """A fresh session holds no prior context, so it needs the full bundle."""
    backend = _QueuedBackend(["# Revised plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    revised = await agent.revise_optimization_plan(
        _context(),
        specialists,
        draft=SynthesizedPlan(text="# Draft plan", session_id=""),
        critic=PlanCriticOutcome(verdict="REVISE", review="Compare the existing GEMM path."),
    )

    assert revised.mode == "fresh"
    payload = json.loads(backend.specs[0].user_prompt)
    for present in (
        "revision_instructions",
        "context",
        "dispatch_plan",
        "specialist_outcomes",
        "specialist_coverage",
        "draft_plan",
        "critic_verdict",
        "critic_review",
    ):
        assert present in payload


@pytest.mark.asyncio
async def test_a_single_plan_is_asked_to_open_with_its_summary_line() -> None:
    """The export reads a plan's problem and goal from its first line, so synthesis must write one."""
    backend = _PerCallBackend(["Summary: The scale stream is re-read. Stage it once.\n# Plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plan_results(
        _context(),
        SpecialistRound(dispatch, (await _outcome("memory", "The scale stream is re-read."),), _coverage(["memory"])),
        lanes=2,
    )

    assert PLAN_SUMMARY_RULE in backend.specs[0].system_prompt


@pytest.mark.asyncio
async def test_every_lane_plan_opens_with_a_summary_line_and_the_partition_does_not() -> None:
    """Only a step that writes a Markdown plan is asked for the line; the partition answers in JSON."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A\nRetime the MFMA.", "# Lane B\nStage."])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)

    await agent.synthesize_lane_plan_results(_context(), await _compute_and_memory_round(), lanes=2)

    partition, *lanes = backend.specs
    assert PLAN_SUMMARY_RULE not in partition.system_prompt
    assert len(lanes) == 2
    assert all(PLAN_SUMMARY_RULE in lane.system_prompt for lane in lanes)


@pytest.mark.asyncio
async def test_dispatch_is_not_asked_for_a_summary_line() -> None:
    backend = _StaticBackend(_dispatch_payload())
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2, min_assignments=1)

    await agent.plan_dispatch(_context(), _definitions())

    assert PLAN_SUMMARY_RULE not in backend.specs[0].system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("resumable", [True, False])
async def test_a_revised_plan_writes_its_own_summary_line(resumable) -> None:
    """A REPLACE changes the plan's goal, so the revision restates the line rather than keeping the draft's."""
    backend = (_ResumableQueuedBackend if resumable else _QueuedBackend)(["# Revised plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.revise_optimization_plan(
        _context(),
        SpecialistRound(dispatch, (await _outcome("memory", "The scale stream is re-read."),), _coverage(["memory"])),
        draft=SynthesizedPlan(
            text="Summary: The draft's goal.\n# Draft plan",
            session_id="synthesis-session" if resumable else "",
        ),
        critic=PlanCriticOutcome(verdict="REPLACE", review="Compare the existing GEMM path."),
    )

    spec = backend.resumes[0][0] if resumable else backend.specs[0]
    assert PLAN_SUMMARY_RULE in spec.system_prompt
    assert "for the plan as revised, not for the draft" in spec.system_prompt
    if not resumable:
        assert PLAN_SUMMARY_RULE.strip() in json.loads(spec.user_prompt)["revision_instructions"]
