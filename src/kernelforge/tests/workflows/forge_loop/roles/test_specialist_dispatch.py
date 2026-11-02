# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dispatching specialists: the prompt and deadline each one is given, how their failures are typed and isolated, and how the framework completes a dispatch."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from kernelforge.workflows.forge_loop.roles import orchestration as orchestration_module
from kernelforge.agent_backends import AgentProviderUnavailableError, AgentRunResult
from kernelforge.workflows.forge_loop.roles.contracts import EvidenceRef
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent, OrchestrationInfrastructureError
from kernelforge.workflows.forge_loop.roles import specialists as specialists_module
from kernelforge.workflows.forge_loop.roles.specialists import (
    SpecialistAgent,
    SpecialistPool,
    SpecialistProbeConfig,
    build_specialist_prompts,
)
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    _QueuedBackend,
    _StaticBackend,
    _assignment,
    _context,
    _definitions,
    _dispatch_payload,
)


@pytest.mark.asyncio
async def test_dispatch_framework_binds_real_evidence_paths() -> None:
    raw = json.loads(_dispatch_payload())
    raw["assignments"][0]["evidence_refs"][0]["path"] = "analysis/abc123/cases/case-a/analysis.md"
    backend = _StaticBackend(json.dumps(raw))
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )

    plan = await agent.plan_dispatch(_context(), _definitions())

    assert plan.assignments[0].role_id == "memory"
    evidence_paths = {reference.path for reference in plan.assignments[0].evidence_refs}
    assert "analysis/abc123/cases/case-a/analysis.md" not in evidence_paths
    assert "case:case-a" in evidence_paths
    assert "analysis/abc123/profiles/case-a/summary.json" in evidence_paths


@pytest.mark.asyncio
async def test_dispatch_turn_cap_still_runs_json_repair(caplog) -> None:
    backend = _QueuedBackend(
        [
            AgentRunResult(
                text='{"assignments": [',
                end_reason="turn_cap",
            ),
            _dispatch_payload(),
        ]
    )
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        plan = await agent.plan_dispatch(_context(), _definitions())

    assert len(backend.specs) == 2
    assert plan.assignments
    assert plan.assignments[0].role_id == "memory"
    assert "continuing through JSON repair" in caplog.text


def test_the_specialist_is_told_its_own_deadline_without_a_probe() -> None:
    """The session clock otherwise reaches the model only in a probe result."""
    system_prompt, _user_prompt = build_specialist_prompts(
        definition=_definitions()["memory"],
        assignment=_assignment(),
        context=_context(),
        session_timeout_sec=900,
    )

    flat = system_prompt.replace("\n", " ")
    assert "900s" in flat
    assert "15 min" in flat
    # The consequence, not just the number: a killed session yields nothing.
    assert "killed" in flat
    assert "no analysis at all" in flat
    # The reserve is stated as the real deadline, so writing is not optional.
    assert "120s are for writing" in flat


def test_the_deadline_is_stated_before_the_probe_section_that_relies_on_it() -> None:
    """The probe text says "your own session"; that clock must come first."""
    setup = specialists_module._ProbeSetup(
        enabled=True,
        config=SpecialistProbeConfig(scratch_root="/tmp/probe-scratch"),
        workspace="/tmp/workspace",
        scratch_dir=Path("/tmp/probe-scratch/memory-0"),
        ledger_path=Path("/tmp/probe-scratch/memory-0/ledger.jsonl"),
        session_deadline=time.time() + 1800,
    )
    system_prompt, _user_prompt = build_specialist_prompts(
        definition=_definitions()["memory"],
        assignment=_assignment(),
        context=_context(),
        session_timeout_sec=1800,
        probe_setup=setup,
    )

    assert system_prompt.index("Time limit") < system_prompt.index("Bounded measurement")


def test_specialist_prompt_is_free_form_read_only_and_scoped() -> None:
    system_prompt, user_prompt = build_specialist_prompts(
        definition=_definitions()["memory"],
        assignment=_assignment(),
        context=_context(),
        session_timeout_sec=1800,
    )

    assert "ordinary Markdown" in system_prompt.replace("\n", " ")
    assert "no fixed schema" in system_prompt
    assert '"case_id": "case-a"' in user_prompt
    assert '"case_id": "case-b"' not in user_prompt


def test_specialist_prompt_receives_stale_analysis_paths_and_commits() -> None:
    root = "/workspace/forge_experiments/analysis/abc123/generation-001"
    diff = "/workspace/forge_experiments/analysis/deltas/abc123_to_def456.patch"
    context = replace(
        _context(),
        analysis_commit="def456",
        canonical_commit="def456",
        evidence_commit="abc123",
        evidence_stale=True,
        evidence_status="profiled",
        evidence_mean_case_speedup=1.1,
        current_mean_case_speedup=1.12,
        cumulative_diff_path=diff,
        evidence_refs=(
            EvidenceRef(
                kind="analysis_bundle",
                path=root,
                summary="Published Analysis bundle.",
            ),
            EvidenceRef(
                kind="analysis_artifact_catalog",
                path=f"{root}/artifact_catalog.json",
                summary="Analysis artifact catalog.",
            ),
            EvidenceRef(
                kind="analysis_cumulative_diff",
                path=diff,
                summary="Cumulative source diff.",
            ),
        ),
    )

    _system_prompt, user_prompt = build_specialist_prompts(
        definition=_definitions()["memory"],
        assignment=_assignment(),
        context=context,
        session_timeout_sec=1800,
    )
    payload = json.loads(user_prompt.split("\n\n", 1)[1])
    scoped = payload["context"]
    paths = {reference["path"] for reference in scoped["evidence_refs"]}

    assert scoped["canonical_commit"] == "def456"
    assert scoped["analysis_evidence"]["commit"] == "abc123"
    assert scoped["analysis_evidence"]["stale"] is True
    assert scoped["analysis_evidence"]["cumulative_diff_path"] == diff
    assert root in paths
    assert f"{root}/artifact_catalog.json" in paths
    assert diff in paths


@pytest.mark.asyncio
async def test_specialist_accepts_free_form_markdown() -> None:
    backend = _StaticBackend("# Memory analysis\nUse vector loads.")
    specialist = SpecialistAgent(
        definition=_definitions()["memory"],
        backend=backend,
        timeout_sec=1,
        max_turns=2,
    )

    outcome = await specialist.run(_assignment(), _context())

    assert outcome.succeeded
    assert outcome.content == "# Memory analysis\nUse vector loads."
    assert backend.calls == 1
    assert backend.specs[0].writable is False
    assert backend.specs[0].tool_policy.shell is False


@pytest.mark.asyncio
async def test_specialist_converts_api_outage_to_typed_backend_failure() -> None:
    backend = _StaticBackend(
        "SDK error text must not become analysis",
        end_reason="api_error",
        stderr_tail="gateway unavailable",
    )
    specialist = SpecialistAgent(
        definition=_definitions()["memory"],
        backend=backend,
        timeout_sec=1,
        max_turns=2,
    )

    outcome = await specialist.run(_assignment(), _context())

    assert outcome.succeeded is False
    assert outcome.content is None
    assert outcome.failure is not None
    assert outcome.failure.kind == "backend_failure"
    assert outcome.failure.message == "gateway unavailable"


@pytest.mark.asyncio
async def test_specialist_converts_provider_exception_to_backend_failure() -> None:
    specialist = SpecialistAgent(
        definition=_definitions()["memory"],
        backend=_StaticBackend(error=AgentProviderUnavailableError("provider unavailable")),
        timeout_sec=1,
        max_turns=2,
    )

    outcome = await specialist.run(_assignment(), _context())

    assert outcome.failure is not None
    assert outcome.failure.kind == "backend_failure"
    assert "provider unavailable" in outcome.failure.message


@pytest.mark.asyncio
async def test_specialist_failures_are_isolated() -> None:
    pool = SpecialistPool(
        {
            "memory": SpecialistAgent(
                definition=_definitions()["memory"],
                backend=_StaticBackend("Memory analysis"),
                timeout_sec=1,
                max_turns=2,
            ),
            "compute": SpecialistAgent(
                definition=_definitions()["compute"],
                backend=_StaticBackend(error=RuntimeError("provider failed")),
                timeout_sec=1,
                max_turns=2,
            ),
        },
        max_parallel=2,
    )

    outcomes = (
        await pool.run(
            (
                _assignment(),
                _assignment(
                    assignment_id="compute-1",
                    role_id="compute",
                    case_id="case-b",
                ),
            ),
            _context(),
        )
    ).outcomes

    assert [item.succeeded for item in outcomes] == [False, True]
    assert outcomes[0].failure is not None
    assert outcomes[0].failure.kind == "backend_error"
    assert outcomes[1].content == "Memory analysis"


@pytest.mark.asyncio
async def test_specialist_pool_respects_parallel_limit() -> None:
    class ConcurrentBackend(_StaticBackend):
        def __init__(self) -> None:
            super().__init__("analysis")
            self.active = 0
            self.max_active = 0

        async def run(self, spec, usage=None) -> AgentRunResult:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return AgentRunResult(text=self.result)

    backend = ConcurrentBackend()
    agents = {
        role_id: SpecialistAgent(
            definition=definition,
            backend=backend,
            timeout_sec=1,
            max_turns=2,
        )
        for role_id, definition in _definitions().items()
    }
    pool = SpecialistPool(agents, max_parallel=1)

    await pool.run(
        (
            _assignment(),
            _assignment(
                assignment_id="compute-1",
                role_id="compute",
                case_id="case-b",
            ),
        ),
        _context(),
    )

    assert backend.max_active == 1


@pytest.mark.asyncio
async def test_dispatch_repairs_invalid_structured_response() -> None:
    backend = _QueuedBackend(["not-json", _dispatch_payload()])
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )

    plan = await agent.plan_dispatch(_context(), _definitions())

    assert plan.assignments[0].role_id == "memory"
    assert len(backend.specs) == 2
    assert agent.structured_output_diagnostics["dispatch"]["normalization_notes"]


@pytest.mark.asyncio
async def test_dispatch_api_outage_propagates_typed_infrastructure_failure() -> None:
    backend = _StaticBackend(
        '{"assignments": [{"role_id": "memory"}]}',
        end_reason="api_error",
        stderr_tail="service unavailable",
    )
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )

    with pytest.raises(
        OrchestrationInfrastructureError,
        match="service unavailable",
    ):
        await agent.plan_dispatch(_context(), _definitions())

    assert backend.calls == 1


@pytest.mark.asyncio
async def test_dispatch_provider_failure_is_typed_infrastructure() -> None:
    agent = OrchestrationAgent(
        backend=_StaticBackend(error=AgentProviderUnavailableError("provider unavailable")),
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )

    with pytest.raises(
        OrchestrationInfrastructureError,
        match="provider unavailable",
    ):
        await agent.plan_dispatch(_context(), _definitions())


@pytest.mark.asyncio
async def test_diversify_dispatch_is_completed_by_framework() -> None:
    backend = _StaticBackend(_dispatch_payload())
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )
    context = replace(
        _context(),
        search_mode="DIVERSIFY",
    )

    plan = await agent.plan_dispatch(context, _definitions())

    assert len(plan.assignments) == len(_definitions())
    assert {case_id for assignment in plan.assignments for case_id in assignment.target_case_ids} == context.case_ids
    assert any("added default role" in note for note in plan.normalization_notes)
