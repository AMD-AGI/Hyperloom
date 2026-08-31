"""Tests for forge-loop orchestration and specialist synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.llm.process_reaping import ReapReport

from kernelforge.orchestrator import orchestration as orchestration_module
from kernelforge.agent_backends import (
    AgentCapabilities,
    AgentProviderUnavailableError,
    AgentRunResult,
)
from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    DispatchPlan,
    EvidenceRef,
    LaneDrop,
    OrchestrationContext,
    PlanCriticOutcome,
    SpecialistAssignment,
    SpecialistDefinition,
    SynthesizedPlan,
)
from kernelforge.orchestrator.orchestration import (
    OrchestrationAgent,
    OrchestrationInfrastructureError,
    OrchestrationService,
    make_orchestration_service,
)
from kernelforge.orchestrator import specialists as specialists_module
from kernelforge.orchestrator.specialists import (
    SpecialistAgent,
    SpecialistPool,
    SpecialistProbeConfig,
    build_specialist_prompts,
)
from kernelforge.orchestrator.plan_critic import PlanCriticAgent


def _context() -> OrchestrationContext:
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace="/workspace",
        gpu_target="gfx942",
        objective="equal-weight mean case speedup",
        program_context="Optimize the operator.",
        source_map_path="analysis/abc123/source_map.json",
        knowledge_index="Curated local knowledge index.",
        cases=(
            CaseEvidence(
                case_id="case-a",
                latency_ms=1.0,
                bottleneck="memory",
                profile_summary_path=("analysis/abc123/profiles/case-a/summary.json"),
            ),
            CaseEvidence(
                case_id="case-b",
                latency_ms=4.0,
                bottleneck="compute",
                profile_summary_path=("analysis/abc123/profiles/case-b/summary.json"),
            ),
        ),
        evidence_refs=(
            EvidenceRef(
                kind="history",
                path="handoffs/iter_001.json",
                summary="Previous iteration outcome",
            ),
        ),
    )


def _definitions() -> dict[str, SpecialistDefinition]:
    return {
        "compute": SpecialistDefinition(
            role_id="compute",
            description="Compute optimization specialist",
            instructions="Analyze instruction throughput and scheduling.",
            capabilities=("compute", "scheduling"),
        ),
        "memory": SpecialistDefinition(
            role_id="memory",
            description="Memory optimization specialist",
            instructions="Analyze memory layout and cache behavior.",
            capabilities=("memory", "cache"),
        ),
    }


def test_the_planning_context_states_the_editable_source_set() -> None:
    """The planner is told what it may edit, in campaign order.

    A campaign that never states its own editable set leaves the planner to
    infer the edit surface from the one path in ``program_context``. That
    inference has closed real directions in writing: a module constant in a
    sibling file was rejected as "an environment variable, not one of the
    editable files", and a tuned CSV that was the FIRST entry of the editable
    list was never considered a file at all. Order carries meaning (entry 0 is
    the primary kernel path) and non-``.py`` entries are first-class.
    """
    campaign_source_files = (
        "/workspace/aiter/ops/triton/op.py",
        "/workspace/aiter/fused_moe.py",
        "/workspace/aiter/configs/tuned_shapes.csv",
        "/workspace/aiter/configs/dispatch.json",
    )
    context = replace(_context(), editable_sources=campaign_source_files)

    payload = context.to_prompt_dict()

    assert payload["editable_sources"] == list(campaign_source_files)
    assert payload["editable_sources"][0].endswith("op.py")
    assert any(not path.endswith(".py") for path in payload["editable_sources"])


def test_the_editable_source_set_defaults_to_empty_and_rejects_junk() -> None:
    assert _context().to_prompt_dict()["editable_sources"] == []
    with pytest.raises(ValueError, match="editable_sources"):
        replace(_context(), editable_sources=("/workspace/a.py", "  "))
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            _context(),
            editable_sources=("/workspace/a.py", "/workspace/a.py"),
        )


def test_explicit_empty_specialist_registry_is_rejected() -> None:
    with pytest.raises(ValueError, match="definitions must not be empty"):
        make_orchestration_service(
            config=None,
            definitions={},
        )


def test_factory_uses_same_runtime_for_independent_critic_backend(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(timeout_sec=1800)
    backends = []

    class Backend:
        def __init__(self):
            self.runtime = runtime
            self.name = "test"

    def create_backend(selected_runtime, **_kwargs):
        assert selected_runtime is runtime
        backend = Backend()
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        orchestration_module,
        "create_registered_backend",
        create_backend,
    )
    service = make_orchestration_service(
        config=SimpleNamespace(
            agent_runtime=lambda: runtime,
            workspace="/workspace",
            max_turns=500,
            # The probe fields the factory now reads as attributes rather than
            # through a ``getattr`` default that restated them a fourth time.
            specialist_probe=True,
            specialist_probe_max=6,
            specialist_probe_budget_sec=600.0,
            specialist_probe_scratch_root="",
        ),
        enable_plan_critic=True,
    )

    assert len(backends) == 2 + len(service._definitions)
    assert service._agent.backend is backends[0]
    assert service._plan_critic is not None
    assert service._plan_critic.backend is backends[1]
    assert service._plan_critic.backend is not service._agent.backend
    assert service._plan_critic.backend.runtime is (service._agent.backend.runtime)
    assert service._plan_critic.max_turns == 100
    assert service._plan_critic.timeout_sec == 600


def _assignment(
    *,
    assignment_id: str = "memory-1",
    role_id: str = "memory",
    case_id: str = "case-a",
) -> SpecialistAssignment:
    return SpecialistAssignment(
        assignment_id=assignment_id,
        role_id=role_id,
        target_case_ids=(case_id,),
        evidence_refs=(
            EvidenceRef(
                kind="profile",
                path=f"analysis/abc123/profiles/{case_id}/summary.json",
                summary=f"Profile for {case_id}",
            ),
        ),
        reason="The case is bottlenecked in this specialist domain.",
    )


def _dispatch_payload(*, both_roles: bool = False) -> str:
    assignments = [
        {
            "assignment_id": "memory-1",
            "role_id": "memory",
            "target_case_ids": ["case-a"],
            "evidence_refs": [
                {
                    "kind": "measurement",
                    "path": "case:case-a",
                    "summary": "Memory case",
                }
            ],
            "reason": "Inspect memory behavior.",
        }
    ]
    if both_roles:
        assignments.append(
            {
                "assignment_id": "compute-1",
                "role_id": "compute",
                "target_case_ids": ["case-b"],
                "evidence_refs": [
                    {
                        "kind": "measurement",
                        "path": "case:case-b",
                        "summary": "Compute case",
                    }
                ],
                "reason": "Inspect compute behavior.",
            }
        )
    return json.dumps({"assignments": assignments})


class _StaticBackend:
    def __init__(
        self,
        result: str = "",
        error: Exception | None = None,
        end_reason: str = "agent_stopped",
        stderr_tail: str = "",
    ) -> None:
        self.result = result
        self.error = error
        self.end_reason = end_reason
        self.stderr_tail = stderr_tail
        self.calls = 0
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
        return AgentRunResult(
            text=self.result,
            end_reason=self.end_reason,
            stderr_tail=self.stderr_tail,
        )


class _QueuedBackend:
    def __init__(
        self,
        results: list[str | AgentRunResult],
    ) -> None:
        self.results = list(results)
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        result = self.results.pop(0)
        return result if isinstance(result, AgentRunResult) else AgentRunResult(text=result)


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


@pytest.mark.asyncio
async def test_synthesis_fuses_analyses_without_schema() -> None:
    plan_markdown = "# Optimization plan\nVectorize loads first, then retune occupancy."
    backend = _StaticBackend(plan_markdown)
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
    )
    specialist_backend = _StaticBackend("Use vector loads.")
    outcome = await SpecialistAgent(
        definition=_definitions()["memory"],
        backend=specialist_backend,
        timeout_sec=1,
        max_turns=2,
    ).run(_assignment(), _context())
    dispatch = DispatchPlan(
        analysis_commit="abc123",
        assignments=(_assignment(),),
    )

    plan = await agent.synthesize_optimization_plan(
        _context(),
        (outcome,),
        dispatch,
        {
            "successful_roles": ["memory"],
            "covered_cases": ["case-a"],
            "missing_cases": ["case-b"],
            "failed_roles": [],
        },
    )

    assert plan == plan_markdown
    prompt = backend.specs[0].system_prompt
    assert "expected value" in prompt
    assert "feasibility" in prompt
    assert "not a catalog" in prompt


class _PerCallBackend:
    """A backend whose reply differs per call, so lanes are distinguishable."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.specs: list = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        index = min(len(self.specs) - 1, len(self.replies) - 1)
        return AgentRunResult(
            text=self.replies[index],
            end_reason="agent_stopped",
            stderr_tail="",
        )


async def _outcome(role: str, analysis: str):
    return await SpecialistAgent(
        definition=_definitions()[role],
        backend=_StaticBackend(analysis),
        timeout_sec=1,
        max_turns=2,
    ).run(_assignment(assignment_id=f"{role}-1", role_id=role), _context())


def _coverage(roles: list[str]) -> dict:
    return {
        "successful_roles": roles,
        "covered_cases": ["case-a"],
        "missing_cases": [],
        "failed_roles": [],
    }


def _partition(*grounds: str) -> str:
    """The round partition a lane round now buys before it synthesizes."""
    return json.dumps({"lanes": [{"ground": ground, "reason": "one session's worth"} for ground in grounds]})


_GROUND_A = "chunk_intra.py: the MFMA issue schedule"
_GROUND_B = "chunk.py: the scale stream's staging"


@pytest.mark.asyncio
async def test_each_lane_is_given_its_ground_and_every_other_lane_s() -> None:
    """Overlapping lanes spend two Implementer sessions on one change.

    The boundary has to reach the lane as the code it may edit. Told only which
    specialist role a sibling holds, a lane can do no better than guess where
    that role's edits will land -- and the roles are three readings of one
    kernel, so the guess is regularly wrong.
    """
    backend = _PerCallBackend(
        [
            _partition(_GROUND_A, _GROUND_B),
            "# Lane A\nRetime the MFMA.",
            "# Lane B\nStage via LDS.",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    plans = await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert plans == ["# Lane A\nRetime the MFMA.", "# Lane B\nStage via LDS."]
    owned = [json.loads(spec.user_prompt)["lane"] for spec in backend.specs[1:]]
    assert owned[0]["ground"] == _GROUND_A
    assert owned[0]["ground_owned_by_other_lanes"] == [_GROUND_B]
    assert owned[1]["ground"] == _GROUND_B
    assert owned[1]["ground_owned_by_other_lanes"] == [_GROUND_A]


@pytest.mark.asyncio
async def test_every_lane_receives_the_whole_round_s_evidence() -> None:
    """A lane's ground is a slice of the code, not a slice of the reading.

    Each role reads the same kernel, so the memory analysis has something to say
    about the compute lane's ground. Handing a lane only its "own" report leaves
    it planning from a fraction of what was bought, and holding a report about
    ground it may not touch.
    """
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    for spec in backend.specs[1:]:
        analyses = json.loads(spec.user_prompt)["specialist_analyses"]
        assert [item["role_id"] for item in analyses] == ["compute", "memory"]
        assert "scale stream" in json.dumps(analyses)


@pytest.mark.asyncio
async def test_the_partition_reads_every_analysis_before_it_divides() -> None:
    """Where the boundaries fall cannot be answered from a slice."""
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    partition = json.loads(backend.specs[0].user_prompt)
    assert [item["role_id"] for item in partition["specialist_analyses"]] == ["compute", "memory"]
    assert agent.structured_output_diagnostics["partition"]["status"] == "planned"


@pytest.mark.asyncio
async def test_a_replace_verdict_spends_one_lane_on_the_challenge() -> None:
    """A critic rules on a plan that exists, so REPLACE lands on the next round.

    The verdict says the route itself is dominated. Dividing that round into
    more regions of the same implementation would spend every lane refining the
    thing the critic just said to stop refining.
    """
    backend = _PerCallBackend([_partition("validate the CK GEMM path", _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))
    context = replace(
        _context(),
        last_critic_verdict="REPLACE",
        last_critic_review="A CK GEMM already exists for this shape.",
    )

    await agent.synthesize_lane_plans(
        context,
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    partition = backend.specs[0]
    assert "Give exactly one lane to that challenge" in partition.system_prompt
    assert "A CK GEMM already exists" in partition.user_prompt
    assert agent.structured_output_diagnostics["partition"]["challenger_requested"]


@pytest.mark.asyncio
async def test_a_round_nobody_challenged_is_divided_as_usual() -> None:
    """Guards the challenger block from reaching every ordinary round."""
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert "Give exactly one lane" not in backend.specs[0].system_prompt
    assert agent.structured_output_diagnostics["partition"]["challenger_requested"] is False


@pytest.mark.asyncio
async def test_an_accepted_round_raises_no_challenger() -> None:
    """Only REPLACE says the route is wrong; REVISE corrected it in place."""
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))
    context = replace(
        _context(),
        last_critic_verdict="REVISE",
        last_critic_review="Add a stop condition.",
    )

    await agent.synthesize_lane_plans(
        context,
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert "Give exactly one lane" not in backend.specs[0].system_prompt


@pytest.mark.asyncio
async def test_the_partition_is_told_a_cross_cutting_move_is_one_lane() -> None:
    """The width follows the directions, not the other way around.

    Asked to divide into at most N, a planner will reach for N pieces, and the
    move that rewrites one shape everywhere it appears is the one that pieces
    worst: each site is already the best it can be alone, so a lane per site
    finds nothing and the change that was there is never attempted by anyone.
    """
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1800, max_turns=500)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=3,
    )

    partition = backend.specs[0].system_prompt
    assert "A change is one ground however many places it lands" in partition
    assert "not one lane per\nsite" in partition
    assert "never a\nnumber of pieces to cut one direction into" in partition


_JOINT_GROUND = "chunk_intra.py: the BLOCK_H tile and the num_warps/waves_per_eu/num_stages the launch passes with it"
_CROSS_CUTTING = "fuse the post kernel into the GEMM, deleting one dispatch"


def _partition_lanes(lanes: list[dict], move: object | None = None) -> str:
    """A partition answer that can carry joint lanes and its widest move."""
    payload: dict = {"lanes": lanes}
    if move is not None:
        payload["cross_cutting_move"] = move
    return json.dumps(payload)


def _lane(ground: str, **extra) -> dict:
    return {"ground": ground, "reason": "one session's worth", **extra}


@pytest.mark.asyncio
async def test_the_partition_keeps_a_launch_config_with_the_body_it_serves() -> None:
    """A body lane measured at the old body's config closes an untested axis.

    Split across lanes, the config lane is a pass-through -- it cannot choose
    values for a body it may not read -- and the body lane is scored at numbers
    tuned for code it deleted. Measured on this kernel class: the same wider
    tile read 2.80 ms at the narrow tile's config and 1.28 ms at its own, so
    the split does not buy two attributable answers, it buys one wrong one.
    """
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    partition = backend.specs[0].system_prompt
    assert "The launch configuration is not ground of its own" in partition
    assert "owns the configuration that serves it" in partition
    assert "2.80 ms" in partition and "1.28 ms" in partition


@pytest.mark.asyncio
async def test_a_joint_lane_reaches_its_implementer_time_boxed() -> None:
    """Wider ground is bought with a fallback, not granted for free.

    A joint lane's gain cannot be attributed to the body or to the config, and
    a joint lane that does not converge returns nothing at all. The fallback is
    what keeps the second failure mode off the table, so it has to reach the
    Implementer's planner along with the ground -- and be visible afterwards to
    anyone asking which of the round's scores were decomposable.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [
                    _lane(
                        _JOINT_GROUND,
                        joint=True,
                        fallback="re-tune num_warps at the current tile alone",
                    ),
                    _lane(_GROUND_B),
                ]
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    joint_lane = json.loads(backend.specs[1].user_prompt)["lane"]
    assert joint_lane["joint"] is True
    assert joint_lane["fallback"] == "re-tune num_warps at the current tile alone"
    assert json.loads(backend.specs[2].user_prompt)["lane"]["joint"] is False
    assert "abandoned" in backend.specs[1].system_prompt
    assert "sequence the lane's `fallback`" in backend.specs[1].system_prompt
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == [1]
    assert diagnostics["grounds"][0]["fallback"]


@pytest.mark.asyncio
async def test_a_joint_lane_with_no_fallback_is_said_out_loud(caplog) -> None:
    """The lane still runs; what it is risking stops being invisible.

    Dropping it would spend the change this exception exists to allow, so the
    round keeps it. But a joint lane with nothing behind it is the one lane
    that can end a session with no measurement at all, and a reviewer reading
    only the ground cannot tell it from a lane that has a fallback.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(_JOINT_GROUND, joint=True), _lane(_GROUND_B)]),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        plans = await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    assert plans == ["# Lane A", "# Lane B"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == [1]
    assert any("named no fallback" in note for note in diagnostics["notes"])
    assert "joint lane(s) 1 carry no fallback" in caplog.text


@pytest.mark.asyncio
async def test_the_round_records_which_lane_owns_its_widest_move() -> None:
    """The move that fits no one region is what four kernels lost.

    Named in an analysis, filed under nobody's ground, absent from every
    artifact the next round reads. Assigned, it is recorded against the lane
    that took it and against that lane's ground, so the round's own output
    says both that the move was attempted and by whom.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(_GROUND_A), _lane(_GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": 2},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert "Name the largest move" in backend.specs[0].system_prompt
    assert "largest cross-cutting move" in json.loads(backend.specs[0].user_prompt)["task"]
    move = agent.structured_output_diagnostics["partition"]["cross_cutting_move"]
    assert move == {
        "status": "assigned",
        "move": _CROSS_CUTTING,
        "lane_id": 2,
        "lane_ground": _GROUND_B,
    }


@pytest.mark.asyncio
async def test_an_unowned_widest_move_is_in_the_round_s_output(caplog) -> None:
    """A move no lane took is a finding, not an absence.

    The round is still divided and still runs; what changes is that the move
    and the reason it was passed over are in the round's own output, where an
    operator reading the artifact finds them. Nothing feeds them to the next
    round. A partition that gave no reason is reported separately from one
    that did, because only the second was a decision.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(_GROUND_A), _lane(_GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": 0},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        plans = await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    assert plans == ["# Lane A", "# Lane B"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"] == {
        "status": "unassigned",
        "move": _CROSS_CUTTING,
        "lane_id": 0,
        "unassigned_reason": "",
    }
    assert any("no reason was given" in note for note in diagnostics["notes"])
    assert _CROSS_CUTTING in caplog.text


@pytest.mark.asyncio
async def test_a_widest_move_naming_a_lane_the_round_lacks_is_unowned(
    caplog,
) -> None:
    """Owned by a lane that does not exist is unowned, and is reported as such.

    The ceiling truncates the lanes, so a move pointed past it would otherwise
    read as assigned in the diagnostics while no session on earth was going to
    attempt it -- the exact failure this record exists to make impossible.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(_GROUND_A), _lane(_GROUND_B), _lane("scale.py: the epilogue")],
                {"move": _CROSS_CUTTING, "lane_id": 3, "unassigned_reason": ""},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["planned"] == 2
    assert diagnostics["cross_cutting_move"]["status"] == "unassigned"
    assert any("which this partition of 2 lane(s) does not have" in note for note in diagnostics["notes"])
    assert "gave no lane its largest cross-cutting move" in caplog.text


@pytest.mark.asyncio
async def test_a_partition_that_named_no_widest_move_reports_that(caplog) -> None:
    """Named none and left one unowned are different answers.

    Both leave the round with no lane on the move, but only the second says
    the partition looked. Collapsing them would let a partition that skipped
    the question read exactly like one that answered it.
    """
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"] == {"status": "missing"}
    assert any("named no largest cross-cutting move" in note for note in diagnostics["notes"])
    assert "the partition named none" in caplog.text


@pytest.mark.asyncio
async def test_a_move_named_as_a_plain_string_is_still_a_named_move(
    caplog,
) -> None:
    """The schema shows an object; a model answering it with prose is ordinary.

    Read as "no move", that answer is filed as `missing` -- the partition
    skipped the question -- when what happened is the `mhc-fused` shape
    exactly: a move named out loud and owned by nobody. The two must not read
    alike, so the string is taken as the move and the absent lane is said.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(_GROUND_A), _lane(_GROUND_B)], _CROSS_CUTTING),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"] == {
        "status": "unassigned",
        "move": _CROSS_CUTTING,
        "lane_id": 0,
        "unassigned_reason": "",
    }
    assert any("came back as a string" in note for note in diagnostics["notes"])
    assert _CROSS_CUTTING in caplog.text


@pytest.mark.asyncio
async def test_a_move_field_of_the_wrong_shape_is_not_reported_as_no_move(
    caplog,
) -> None:
    """A shape the parser cannot read is not evidence the partition looked.

    `missing` is a statement about the partition -- it named none. A list where
    an object was asked for says nothing about what the partition found, so it
    gets its own status and the field is quoted, rather than being folded into
    the one answer an operator would stop reading at.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(_GROUND_A), _lane(_GROUND_B)],
                [{"move": _CROSS_CUTTING}],
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    move = agent.structured_output_diagnostics["partition"]["cross_cutting_move"]
    assert move["status"] == "unreadable"
    assert "came back as a list" in move["field"]
    assert "came back as a list" in caplog.text
    assert "the partition named none" not in caplog.text


@pytest.mark.asyncio
async def test_an_owned_move_records_the_ground_that_owns_it() -> None:
    """The lane_id is the partitioner grading its own answer.

    Nothing downstream can tell a move handed to the lane that can actually
    make it from one pointed at whichever lane came to mind: both read
    `assigned` and neither warns. Recording the owning lane's ground next to
    the move makes the claim falsifiable from the artifact alone.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(_GROUND_A), _lane(_GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": 1},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    move = agent.structured_output_diagnostics["partition"]["cross_cutting_move"]
    assert move["lane_id"] == 1
    assert move["lane_ground"] == _GROUND_A


@pytest.mark.asyncio
async def test_a_move_naming_a_lane_that_is_not_a_number_is_unowned() -> None:
    """A lane_id that is not a lane number is not a lane, and says so."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(_GROUND_A), _lane(_GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": "lane 2"},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"]["status"] == "unassigned"
    assert any("not a lane number" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_an_unreadable_joint_answer_is_recorded_not_just_narrowed() -> None:
    """Falling back to narrow ground is right; doing it without a word is not.

    "partial" is neither true nor false, and the safe reading is not joint --
    but then a lane whose ground spans a body and the config that serves it
    runs with no fallback required and nothing in the artifact saying the
    width was refused rather than never asked for.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(_JOINT_GROUND, joint="partial"), _lane(_GROUND_B)]),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == []
    assert any("'partial', which is not a boolean" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_a_lane_that_declined_joint_ground_does_not_get_it() -> None:
    """A quoted "false" is a no, and the flag it sets is the one that widens.

    Read with ``bool``, every string is true, so a partition that quoted its
    booleans would hand joint ground -- and the attribution cost that comes
    with it -- to lanes that never asked for any.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [
                    _lane(_GROUND_A, joint="false"),
                    _lane(_JOINT_GROUND, joint="true", fallback="re-tune only"),
                ]
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert agent.structured_output_diagnostics["partition"]["joint"] == [2]


def test_an_answer_that_is_not_a_boolean_is_read_as_the_note_promises() -> None:
    """The flag stored and the note written have to describe one reading.

    `joint: 1` is a shape a model emits constantly, and read through ``bool``
    it widened the lane while the note in the same artifact said the answer was
    unreadable and the lane kept its narrow ground. Whichever of the two an
    operator believed, the other was a lie about the round that ran.
    """
    assert orchestration_module._as_bool(True) == (True, True)
    assert orchestration_module._as_bool("true") == (True, True)
    assert orchestration_module._as_bool(None) == (False, True)
    assert orchestration_module._as_bool("false") == (False, True)
    # Not booleans, whichever way they would have gone through ``bool``.
    assert orchestration_module._as_bool(1) == (False, False)
    assert orchestration_module._as_bool(0) == (False, False)
    assert orchestration_module._as_bool(2.5) == (False, False)
    assert orchestration_module._as_bool({"a": 1}) == (False, False)
    assert orchestration_module._as_bool([]) == (False, False)


@pytest.mark.asyncio
async def test_a_lane_that_answered_joint_with_a_number_keeps_narrow_ground() -> None:
    """The lane list and the note are read by the same operator.

    A truthy non-boolean filed the lane under the round's joint lanes and under
    the note saying it was read as not joint. The lane also reached its own
    Implementer marked joint, so the artifact contradicted the payload as well
    as itself.
    """
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(_JOINT_GROUND, joint=1), _lane(_GROUND_B)]),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == []
    assert diagnostics["grounds"][0]["joint"] is False
    assert json.loads(backend.specs[1].user_prompt)["lane"]["joint"] is False
    assert any("1, which is not a boolean" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_a_launch_site_two_bodies_share_is_owned_by_one_lane() -> None:
    """The exception that widens a lane cannot be handed to every lane.

    "the lane that changes the body owns the configuration that serves it" is
    stated to the whole partition, so two body lanes dispatched from one launch
    site each own that site by it, and the disjointness the round asserts --
    and stacks its candidates on -- is gone. Stacking then drops the second
    patch, which is one Implementer session bought and thrown away.
    """
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    partition = backend.specs[0].system_prompt
    assert "One launch site belongs to exactly one lane" in partition
    assert "name that launch in exactly one\nlane's ground" in partition
    # The boundary is derived, never written twice: a lane's ground says only
    # what it owns, so the prompt must not ask for a "stay off" clause that
    # would reach the owning lane as ground it does not own.
    assert "every other lane sees it as ground it does not own" in partition
    assert "keep the other lane off it" not in partition


@pytest.mark.asyncio
async def test_a_partition_that_could_not_be_bought_claims_no_move() -> None:
    """A collapsed round found nothing, so it must not report having looked.

    The fallback ground is assembled from the specialist role names; it has no
    reading of the kernel behind it and cannot name a move or rule one out.
    """
    backend = _PerCallBackend(["not json at all", "# One plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["collapsed"] is True
    assert diagnostics["cross_cutting_move"] == {"status": "unavailable"}
    assert diagnostics["joint"] == []


@pytest.mark.asyncio
async def test_the_partition_divides_what_it_was_given_without_reading_more() -> None:
    """The analyses already name the files and functions they are about.

    A partition allowed to read source re-derives the analysis instead of
    dividing it, and it does so on the critical path where every lane of the
    round is waiting. Measured before this bound existed: one partition over
    three analyses was still exploring after eighteen minutes.
    """
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1800, max_turns=500)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    partition, lane = backend.specs[0], backend.specs[1]
    assert partition.tool_policy.read is False
    assert partition.tool_policy.search is False
    assert partition.tool_policy.max_turns == orchestration_module.ROUND_PARTITION_MAX_TURNS
    assert partition.timeout_sec == orchestration_module.ROUND_PARTITION_TIMEOUT_SEC
    assert partition.reasoning_effort == orchestration_module.ROUND_PARTITION_EFFORT
    assert lane.reasoning_effort == "max"
    # A lane plan does need the source; only the division does not.
    assert lane.tool_policy.read is True
    assert lane.timeout_sec == 1800


@pytest.mark.asyncio
async def test_a_partition_that_cannot_be_bought_still_leaves_a_round() -> None:
    """A round that cannot be divided by code runs as one lane, not a wide one.

    The value of a fan-out round is that each lane edits code no other lane
    edits, so each candidate earns a score that can be attributed to it. The
    old fallback dealt the analyses out by role, which divides the evidence
    without dividing the code -- observed in production, one lane's edited files
    a subset of its sibling's -- so the round spent N sessions and could get one
    answer. A partition that cannot be bought collapses to a single lane.
    """
    backend = _PerCallBackend(["not json at all", "# One plan", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    plans = await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert plans == ["# One plan"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["status"] == "fallback"
    assert diagnostics["collapsed"] is True
    assert diagnostics["planned"] == 1
    assert len(diagnostics["grounds"]) == 1
    assert "compute" in diagnostics["grounds"][0]["ground"]
    assert "memory" in diagnostics["grounds"][0]["ground"]


@pytest.mark.asyncio
async def test_a_challenge_survives_a_partition_that_could_not_be_bought() -> None:
    """A REPLACE is not asked for again, and the round it judged is this one.

    A single ordinary lane would answer "this route is dominated" by refining
    that very route, through the one path that never reports having done so. A
    challenged round that falls back collapses to the one challenger lane.
    """
    backend = _PerCallBackend(["not json at all", "# Challenger plan", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))
    context = replace(
        _context(),
        last_critic_verdict="REPLACE",
        last_critic_review="A CK GEMM already exists for this shape.",
    )

    plans = await agent.synthesize_lane_plans(
        context,
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert plans == ["# Challenger plan"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["status"] == "fallback"
    assert diagnostics["collapsed"] is True
    assert diagnostics["planned"] == 1
    assert diagnostics["challenger_requested"] is True
    grounds = diagnostics["grounds"]
    assert [ground["lane_id"] for ground in grounds] == [1]
    assert "last_plan_critic" in grounds[0]["ground"]
    # The single lane's own payload carries the challenger ground, not the
    # ordinary synthesis prompt.
    lane_spec = backend.specs[1]
    assert "last_plan_critic" in json.loads(lane_spec.user_prompt)["lane"]["ground"]


@pytest.mark.asyncio
async def test_one_real_direction_is_planned_as_a_single_lane_round() -> None:
    """A slot filled to reach the requested width still costs a whole session."""
    backend = _PerCallBackend([_partition(_GROUND_A), "# One plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    plans = await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert plans == ["# One plan"]
    assert "lane" not in json.loads(backend.specs[1].user_prompt)


class _OneLaneFailsBackend:
    """A backend that partitions, then loses the lane owning one named ground."""

    def __init__(self, *, failing_ground: str, answer: str) -> None:
        self.failing_ground = failing_ground
        self.answer = answer
        self.specs: list = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        lane = json.loads(spec.user_prompt).get("lane")
        if lane is None:
            return AgentRunResult(
                text=_partition(_GROUND_A, _GROUND_B),
                end_reason="agent_stopped",
                stderr_tail="",
            )
        if self.failing_ground in lane["ground"]:
            raise AgentProviderUnavailableError("the lane's provider went away")
        return AgentRunResult(
            text=self.answer,
            end_reason="agent_stopped",
            stderr_tail="",
        )


@pytest.mark.asyncio
async def test_one_lost_lane_does_not_cost_the_round_its_healthy_plans() -> None:
    """Each lane is its own call, so one failure is one lane, not the round.

    Letting it propagate would throw away siblings that already answered and
    were already paid for -- and the loop reads a raised synthesis as a planning
    outage, so asking for N lanes would multiply the chance of tripping the
    orchestration circuit breaker by N.
    """
    backend = _OneLaneFailsBackend(failing_ground=_GROUND_A, answer="# Lane B\nStage via LDS.")
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    plans = await agent.synthesize_lane_plans(
        _context(),
        outcomes,
        dispatch,
        _coverage(["compute", "memory"]),
        lanes=2,
    )

    assert plans == ["# Lane B\nStage via LDS."]


@pytest.mark.asyncio
async def test_a_round_that_loses_every_lane_is_still_a_planning_outage() -> None:
    """Tolerating one loss must not turn a total outage into a silent success."""
    backend = _OneLaneFailsBackend(failing_ground=_GROUND_B, answer="")
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with pytest.raises(Exception) as outage:
        await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    assert "lane plan" in str(outage.value)


@pytest.mark.asyncio
async def test_a_lane_that_answered_with_nothing_is_reported(caplog) -> None:
    """A round narrowed by an empty answer must not look like a narrower round."""
    backend = _PerCallBackend([_partition(_GROUND_A, _GROUND_B), "   ", "# Lane B\nStage via LDS."])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    with caplog.at_level("WARNING"):
        plans = await agent.synthesize_lane_plans(
            _context(),
            outcomes,
            dispatch,
            _coverage(["compute", "memory"]),
            lanes=2,
        )

    assert plans == ["# Lane B\nStage via LDS."]
    assert "empty plan" in caplog.text


@pytest.mark.asyncio
async def test_one_usable_analysis_is_one_lane() -> None:
    """A single lane must behave exactly like the fused single-plan path."""
    backend = _PerCallBackend(["# The only plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcome = await _outcome("memory", "The scale stream is re-read.")
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    plans = await agent.synthesize_lane_plans(
        _context(),
        (outcome,),
        dispatch,
        _coverage(["memory"]),
        lanes=2,
    )

    assert plans == ["# The only plan"]
    assert "not a catalog" in backend.specs[0].system_prompt


@pytest.mark.asyncio
async def test_synthesis_api_outage_is_not_returned_as_a_plan() -> None:
    backend = _StaticBackend(
        "# SDK failure\nThis is not a plan.",
        end_reason="sdk_stream_error",
        stderr_tail="stream disconnected",
    )
    agent = OrchestrationAgent(
        backend=backend,
        timeout_sec=1,
        max_turns=2,
    )
    outcome = await SpecialistAgent(
        definition=_definitions()["memory"],
        backend=_StaticBackend("Use vector loads."),
        timeout_sec=1,
        max_turns=2,
    ).run(_assignment(), _context())

    with pytest.raises(
        OrchestrationInfrastructureError,
        match="stream disconnected",
    ):
        await agent.synthesize_optimization_plan(
            _context(),
            (outcome,),
            DispatchPlan(
                analysis_commit="abc123",
                assignments=(_assignment(),),
            ),
            {
                "successful_roles": ["memory"],
                "covered_cases": ["case-a"],
                "missing_cases": ["case-b"],
                "failed_roles": [],
            },
        )


@pytest.mark.asyncio
async def test_orchestration_service_dispatches_and_synthesizes() -> None:
    plan_markdown = "# Optimization plan\nImplement vector loads."
    orchestration_backend = _QueuedBackend([_dispatch_payload(), plan_markdown])
    agent = OrchestrationAgent(
        backend=orchestration_backend,
        timeout_sec=1,
        max_turns=2,
        min_assignments=1,
    )
    specialist = SpecialistAgent(
        definition=_definitions()["memory"],
        backend=_StaticBackend("# Analysis\nVector loads are feasible."),
        timeout_sec=1,
        max_turns=2,
    )
    service = OrchestrationService(
        agent=agent,
        specialist_pool=SpecialistPool(
            {"memory": specialist},
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )

    result = await service.run(_context())

    assert result.optimization_plan == plan_markdown
    assert result.specialist_outcomes[0].content == ("# Analysis\nVector loads are feasible.")
    assert len(orchestration_backend.specs) == 2


@pytest.mark.asyncio
async def test_a_leaked_probe_is_reported_to_the_loop_that_must_refuse(tmp_path, monkeypatch) -> None:
    """The round's teardown finding has to leave the analysis phase.

    Reaping the probe tree answers nothing on its own: what a probe left on the
    device holds the same GPU this round's canonical measurement is about to
    use, and only the loop can decide not to take it. So the finding rides out
    with the planning diagnostics -- in this process and this iteration, which
    is the one whose measurement it has to stop.
    """

    async def _leaked(directory, *, description):
        return ReapReport(
            directory=str(directory),
            unkillable=(4321,),
            holding_device=(4321,),
        )

    monkeypatch.setattr(specialists_module, "reap_processes_under", _leaked)
    orchestration_backend = _QueuedBackend([_dispatch_payload(), "# Optimization plan\nUse vector loads."])
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                    probe=SpecialistProbeConfig(scratch_root=str(tmp_path / "probe")),
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )

    result = await service.run(_context())

    finding = result.structured_output_diagnostics["probe_device_hazard"]
    assert finding["pids"] == [4321]
    assert "4321" in finding["describe"]
    # The round still planned: the contention is the device's, not the plan's.
    assert result.optimization_plan


@pytest.mark.asyncio
async def test_a_clean_probe_round_reports_no_hazard(tmp_path, monkeypatch) -> None:
    """An ordinary round must not be made to look contended."""

    async def _clean(directory, *, description):
        return ReapReport(directory=str(directory))

    monkeypatch.setattr(specialists_module, "reap_processes_under", _clean)
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=_QueuedBackend([_dispatch_payload(), "# Optimization plan\nUse vector loads."]),
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                    probe=SpecialistProbeConfig(scratch_root=str(tmp_path / "probe")),
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )

    result = await service.run(_context())

    assert "probe_device_hazard" not in result.structured_output_diagnostics


@pytest.mark.asyncio
async def test_single_lane_without_critic_does_not_publish_a_draft() -> None:
    orchestration_backend = _QueuedBackend([_dispatch_payload(), "# Optimization plan\nUse vector loads."])
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
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
    )

    result = await service.run(_context())

    assert result.optimization_plan.startswith("# Optimization plan")
    assert result.optimization_plan_draft == ""
    assert result.plan_critic is None


@pytest.mark.asyncio
async def test_plan_critic_accepts_draft_without_revision() -> None:
    plan_markdown = "# Optimization plan\nImplement vector loads."
    orchestration_backend = _QueuedBackend([_dispatch_payload(), plan_markdown])
    specialist_backend = _StaticBackend("# Analysis\nVector loads are feasible.")
    critic_backend = _StaticBackend("VERDICT: ACCEPT\n\nThe plan is worth executing.")
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
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

    assert result.optimization_plan == plan_markdown
    assert result.optimization_plan_draft == plan_markdown
    assert result.plan_revised is False
    assert result.plan_critic is not None
    assert result.plan_critic.verdict == "ACCEPT"
    assert len(orchestration_backend.specs) == 2
    assert specialist_backend.calls == 1
    assert critic_backend.calls == 1


def _width_block(*drops: dict) -> str:
    """The trailing JSON block the round contract asks every review to end with."""
    return json.dumps({"lane_narrowing": list(drops)})


@pytest.mark.asyncio
async def test_plan_critic_reviews_a_multi_lane_round_once() -> None:
    definitions = _definitions()
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(_GROUND_A, _GROUND_B),
            "# Lane plan\nMemory route.",
            "# Lane plan\nCompute route.",
            "# Lane plan\nMemory route, revised.",
            "# Lane plan\nCompute route, revised.",
        ]
    )
    critic_backend = _StaticBackend("VERDICT: REVISE\n\nLane 2 repeats lane 1's change.\n" + _width_block())
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=2,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=definitions["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                ),
                "compute": SpecialistAgent(
                    definition=definitions["compute"],
                    backend=_StaticBackend("Compute analysis"),
                    timeout_sec=1,
                    max_turns=2,
                ),
            },
            max_parallel=2,
        ),
        definitions=definitions,
        plan_critic=PlanCriticAgent(
            backend=critic_backend,
            timeout_sec=1,
        ),
    )

    result = await service.run(_context(), lanes=2)

    assert len(result.optimization_plans) == 2
    # One review for the round, not one per lane: the width is what the round
    # most needs reviewed, and it is a question no single lane can be asked.
    assert critic_backend.calls == 1
    assert result.plan_critic is not None
    assert result.plan_critic.verdict == "REVISE"
    payload = json.loads(critic_backend.specs[0].user_prompt)
    assert [lane["lane_id"] for lane in payload["draft_lane_plans"]] == [1, 2]
    assert payload["draft_lane_plans"][0]["ground"] == _GROUND_A
    assert payload["draft_lane_plans"][1]["ground"] == _GROUND_B
    assert "draft_plan" not in payload
    assert "not worth an Implementer session of its own" in (critic_backend.specs[0].system_prompt)
    # The verdict was about the round, so it reached every lane in it.
    assert result.optimization_plans == (
        "# Lane plan\nMemory route, revised.",
        "# Lane plan\nCompute route, revised.",
    )
    assert result.plan_revised is True
    assert result.structured_output_diagnostics["plan_revision"]["status"] == ("revised")


def _two_lane_critic_service(orchestration_backend, critic_backend):
    """A two-role service whose round is reviewed by one critic."""
    definitions = _definitions()
    return OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=2,
        ),
        specialist_pool=SpecialistPool(
            {
                role: SpecialistAgent(
                    definition=definitions[role],
                    backend=_StaticBackend(f"{role} analysis"),
                    timeout_sec=1,
                    max_turns=2,
                )
                for role in ("memory", "compute")
            },
            max_parallel=2,
        ),
        definitions=definitions,
        plan_critic=PlanCriticAgent(backend=critic_backend, timeout_sec=1),
    )


@pytest.mark.asyncio
async def test_the_round_reports_how_it_was_divided() -> None:
    """A round is audited after the fact or not at all.

    The service snapshots the agent's diagnostics once dispatch has answered,
    which is before the partition has run, so how the round was divided --
    bought or fallen back to, and whether a challenger was asked for -- was
    recorded on the agent and then left out of everything published.
    """
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(_GROUND_A, _GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
        ]
    )
    service = _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend("VERDICT: ACCEPT\n\nBoth lanes are worth a session.\n" + _width_block()),
    )

    result = await service.run(_context(), lanes=2)

    partition = result.structured_output_diagnostics["partition"]
    assert partition["status"] == "planned"
    assert partition["challenger_requested"] is False
    assert [ground["ground"] for ground in partition["grounds"]] == [
        _GROUND_A,
        _GROUND_B,
    ]
    # The earlier snapshot's own entries survive the merge.
    assert "coverage" in result.structured_output_diagnostics
    assert result.structured_output_diagnostics["lanes"]["planned"] == 2


def _two_lane_round(critic_review: str, *, revisions: list[str] | None = None):
    """A two-lane round whose review says ``critic_review`` about its width."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(_GROUND_A, _GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
            *(revisions or []),
        ]
    )
    return orchestration_backend, _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend(critic_review),
    )


@pytest.mark.asyncio
async def test_a_lane_the_review_will_not_pay_for_is_not_published() -> None:
    """The round's width is the one thing the verdict could not say.

    Six production reviews found a named lane not worth its Implementer
    session; every one of those rounds ran that lane anyway, because
    ACCEPT/REVISE/REPLACE cover the round and cannot single a lane out.
    """
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {
                "lane_id": 2,
                "reason": "it is lane 1's epilogue rewrite in different words",
            }
        )
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft",)
    assert result.optimization_plan_draft == "# Lane 1 draft"
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["block"] == "answered"
    assert narrowing["planned"] == 2
    assert narrowing["kept"] == 1
    assert narrowing["dropped"] == [
        {
            "lane_id": 2,
            "reason": "it is lane 1's epilogue rewrite in different words",
        }
    ]
    assert result.structured_output_diagnostics["lanes"] == {
        "requested": 2,
        "planned": 2,
        "published": 1,
    }


@pytest.mark.asyncio
async def test_a_dropped_lane_is_not_revised_first() -> None:
    """A revision turn spent on a lane that will not run is the whole cost."""
    backend, service = _two_lane_round(
        "VERDICT: REVISE\n\n" + _width_block({"lane_id": 1, "reason": "no profile supports its ground"}),
        revisions=["# Lane 2 revised"],
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 2 revised",)
    # Dispatch, partition, two lane plans, and one revision -- not two.
    assert len(backend.specs) == 5
    revision = result.structured_output_diagnostics["plan_revision"]
    assert revision["status"] == "revised"
    assert revision["lanes"] == 1


@pytest.mark.asyncio
async def test_a_round_narrowed_to_nothing_keeps_every_lane() -> None:
    """A round that publishes nothing spent its planning window for no score.

    The review ranked no lane above another, so there is no principled single
    survivor to keep: the narrowing is refused whole and says so.
    """
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {"lane_id": 1, "reason": "unsupported"},
            {"lane_id": 2, "reason": "also unsupported"},
        )
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "refused_empty_round"
    assert narrowing["kept"] == 2
    assert narrowing["dropped"] == []
    assert any("nothing to measure" in note for note in narrowing["notes"])
    assert result.structured_output_diagnostics["lanes"]["published"] == 2


@pytest.mark.asyncio
async def test_a_round_collapsed_to_one_lane_cannot_be_narrowed() -> None:
    """Two decisions about width meet here, and the floor wins.

    The partition could not be bought, so the round already collapsed to the
    single lane that is the floor. Narrowing runs after it and would otherwise
    win on width; below one lane it does not run at all.
    """
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            "not json at all",
            "# One plan",
        ]
    )
    service = _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend("VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 1, "reason": "not worth a session"})),
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# One plan",)
    assert result.structured_output_diagnostics["partition"]["collapsed"] is True
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "refused_single_lane"
    assert narrowing["kept"] == 1
    assert any("at least one lane" in note for note in narrowing["notes"])


@pytest.mark.asyncio
async def test_a_challenged_round_refuses_narrowing_it_cannot_aim() -> None:
    """One of these lanes is the challenger, and nothing records which.

    The previous round's REPLACE bought exactly one lane to validate the route
    it named. A drop applied here could spend that challenge without anyone
    being able to tell afterwards, so the challenge outranks the narrowing.
    """
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 2, "reason": "it duplicates lane 1"})
    )
    context = replace(
        _context(),
        last_critic_verdict="REPLACE",
        last_critic_review="A CK GEMM already exists for this shape.",
    )

    result = await service.run(context, lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    assert result.structured_output_diagnostics["partition"]["challenger_requested"] is True
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "refused_challenger"
    assert narrowing["kept"] == 2
    assert any("challenger lane" in note for note in narrowing["notes"])


def _joint_round(joint_lanes: list[int], count: int, critic_review: str):
    """A round of ``count`` lanes, the ones named in ``joint_lanes`` widened."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition_lanes(
                [
                    _lane(
                        _JOINT_GROUND,
                        joint=True,
                        fallback="re-tune num_warps at the current tile alone",
                    )
                    if lane_id in joint_lanes
                    else _lane(f"{_GROUND_B} ({lane_id})")
                    for lane_id in range(1, count + 1)
                ]
            ),
            *[f"# Lane {lane_id} draft" for lane_id in range(1, count + 1)],
        ]
    )
    return orchestration_backend, _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend(critic_review),
    )


def _joint_lane_round(critic_review: str):
    """A two-lane round whose first lane the partition widened to joint ground."""
    return _joint_round([1], 2, critic_review)


@pytest.mark.asyncio
async def test_the_review_is_told_which_lane_the_partition_widened() -> None:
    """A ruling on width made blind to the width is not the ruling asked for.

    The narrowing step obeys the review, so the review is where the fact that a
    lane was deliberately widened -- and the smaller change it falls back to --
    has to arrive. Given only ground and draft, the step that decides whether a
    lane is worth its session could not tell a joint lane from any other, and
    the round's one linkage between the two decisions did not exist.
    """
    _backend, service = _joint_lane_round("VERDICT: ACCEPT\n\n" + _width_block())

    await service.run(_context(), lanes=2)

    critic_spec = service._plan_critic.backend.specs[0]
    lanes = json.loads(critic_spec.user_prompt)["draft_lane_plans"]
    assert lanes[0]["joint"] is True
    assert lanes[0]["fallback"] == ("re-tune num_warps at the current tile alone")
    assert lanes[1]["joint"] is False
    assert lanes[1]["fallback"] == ""
    assert "`joint` is true was given wider ground" in critic_spec.system_prompt
    assert "exactly as for any other lane" in critic_spec.system_prompt


@pytest.mark.asyncio
async def test_a_drop_aimed_at_the_joint_lane_is_carried_out_and_recorded() -> None:
    """The width is sunk; refusing the drop spends a session instead of saving one.

    The partition has already granted the wider ground by the time the review
    rules, so keeping the lane recovers nothing -- it buys an Implementer
    session for a lane the review, which was shown the width, judged not worth
    one. What the round owes is the record: it widened ground and measured none
    of it.
    """
    _backend, service = _joint_lane_round(
        "VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 1, "reason": "the tile rewrite is too large a bet"})
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 2 draft",)
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["kept"] == 1
    assert narrowing["dropped"] == [{"lane_id": 1, "reason": "the tile rewrite is too large a bet"}]
    assert narrowing["joint"] == [1]
    assert narrowing["dropped_joint"] == [1]
    assert any("wider ground the partition bought" in note for note in narrowing["notes"])
    assert result.structured_output_diagnostics["lanes"]["published"] == 1


@pytest.mark.asyncio
async def test_a_drop_aimed_past_the_joint_lane_leaves_the_widened_lane() -> None:
    """A round that carries a joint lane narrows around it like any other.

    The joint lane is published because nobody asked to drop it, and the record
    separates that from a round with no joint lane at all.
    """
    _backend, service = _joint_lane_round(
        "VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 2, "reason": "it is lane 1's staging in other words"})
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft",)
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["kept"] == 1
    assert narrowing["joint"] == [1]
    assert narrowing["dropped_joint"] == []
    assert result.structured_output_diagnostics["lanes"]["published"] == 1


def _drafts(*joint: bool) -> list:
    """One drafted lane per flag, the true ones widened to joint ground."""
    return [
        SynthesizedPlan(
            text=f"# Lane {index + 1} draft",
            ground=_JOINT_GROUND if is_joint else f"{_GROUND_B} ({index + 1})",
            joint=is_joint,
            fallback=("re-tune num_warps at the current tile alone" if is_joint else ""),
        )
        for index, is_joint in enumerate(joint)
    ]


def _ruling(*drops: tuple[int, str]) -> PlanCriticOutcome:
    """A review that answered its width block and named these lanes."""
    return PlanCriticOutcome(
        verdict="ACCEPT",
        review="VERDICT: ACCEPT",
        lane_drops=tuple(LaneDrop(lane_id=lane_id, reason=reason) for lane_id, reason in drops),
        narrowing_status="answered",
    )


def test_a_joint_lane_does_not_make_an_emptying_ruling_survivable() -> None:
    """The floor is one lane, and it must not be raised into a ranking.

    A refusal aimed at the joint lane alone took an emptying ruling apart: the
    drops on the other lanes applied, the emptiness check never fired because
    the refused drop was missing from the count, and the round published exactly
    the lane the review had named -- an arbitrary survivor chosen by the
    partition's width rather than by any ranking the review gave, and reported
    as an ordinary narrowing.
    """
    drafts = _drafts(True, False, False)

    kept, diagnostics = OrchestrationService._narrow_round(
        drafts,
        critic_outcome=_ruling(
            (1, "the tile rewrite is too large a bet"),
            (2, "the evidence does not support it"),
            (3, "it is lane 2 in other words"),
        ),
        challenged=False,
    )

    assert diagnostics["status"] == "refused_empty_round"
    assert diagnostics["kept"] == 3
    assert diagnostics["joint"] == [1]
    assert diagnostics["dropped_joint"] == []
    assert kept == drafts


def test_a_round_of_only_joint_lanes_still_narrows() -> None:
    """Marking every lane joint must not switch narrowing off.

    Nothing bounds how many lanes a partition may call joint, and the partition
    is told that the lane changing a body owns the configuration serving it --
    so every body lane that re-tunes its launch qualifies. A round that refused
    every drop it carried a joint flag for would hand the partition a veto over
    the review, by an answer the partition is invited to give.
    """
    drafts = _drafts(True, True, True)

    kept, diagnostics = OrchestrationService._narrow_round(
        drafts,
        critic_outcome=_ruling(
            (2, "the evidence does not support it"),
            (3, "it is lane 1 in other words"),
        ),
        challenged=False,
    )

    assert [draft.text for draft in kept] == ["# Lane 1 draft"]
    assert diagnostics["status"] == "narrowed"
    assert diagnostics["kept"] == 1
    assert diagnostics["joint"] == [1, 2, 3]
    assert diagnostics["dropped_joint"] == [2, 3]
    assert any("wider ground the partition bought" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_narrowing_the_round_cannot_read_names_what_it_kept() -> None:
    """ "The review wanted every lane" and "nobody could read it" differ."""
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {"lane_id": "two", "reason": "it duplicates lane 1"},
            {"lane_id": 5, "reason": "there is no lane 5"},
        )
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "not_applied"
    assert narrowing["block"] == "answered"
    assert narrowing["kept"] == 2
    assert any("lane drop names no lane" in note for note in narrowing["notes"])
    assert any("does not have" in note for note in narrowing["notes"])


@pytest.mark.asyncio
async def test_a_review_that_asks_for_nothing_leaves_the_round_alone() -> None:
    """The empty block is an answer; only it means "run every lane"."""
    _backend, service = _two_lane_round("VERDICT: ACCEPT\n\nBoth lanes earn it.\n" + _width_block())

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    assert result.structured_output_diagnostics["lane_narrowing"] == {
        "status": "not_requested",
        "block": "answered",
        "planned": 2,
        "kept": 2,
        "dropped": [],
        "joint": [],
        "dropped_joint": [],
        "notes": [],
    }


@pytest.mark.asyncio
async def test_a_round_whose_review_never_answered_on_width_says_so() -> None:
    """The case the DROP LANE regex passed over in silence.

    The review states a drop as a sentence and ends with no block. The regex
    matched neither the directive nor the "looks like a directive" pattern, so
    the round narrowed nothing and recorded nothing -- indistinguishable from a
    review that wanted both lanes. The static backend answers the repair pass
    with the same prose, so this is the worst case: the block is never read.
    The round still runs both lanes, and it says why.
    """
    _backend, service = _two_lane_round(
        "VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
        revisions=["# Lane 1 revised", "# Lane 2 revised"],
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 revised", "# Lane 2 revised")
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "not_applied"
    assert narrowing["block"] == "absent"
    assert narrowing["kept"] == 2
    assert narrowing["dropped"] == []
    assert narrowing["notes"] != []
    assert any("no lane_narrowing block" in note for note in narrowing["notes"])
    assert any("repair pass" in note for note in narrowing["notes"])
    assert result.structured_output_diagnostics["plan_critic"]["narrowing_status"] == "absent"


@pytest.mark.asyncio
async def test_a_round_narrows_on_a_ruling_its_review_was_asked_twice_for() -> None:
    """One repair call buys back an Implementer session the round would run.

    Same prose-only review as above, but the repair pass answers with the block
    the review owed. The drop is applied, and the round records that the ruling
    was not read the first time.
    """
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(_GROUND_A, _GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
        ]
    )
    critic_backend = _QueuedBackend(
        [
            "VERDICT: ACCEPT\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
            _width_block(
                {
                    "lane_id": 2,
                    "reason": "it re-derives lane 1's autotune lever",
                }
            ),
        ]
    )
    service = _two_lane_critic_service(orchestration_backend, critic_backend)

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft",)
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["block"] == "repaired"
    assert narrowing["dropped"] == [{"lane_id": 2, "reason": "it re-derives lane 1's autotune lever"}]
    assert result.structured_output_diagnostics["lanes"]["published"] == 1
    assert len(critic_backend.specs) == 2


@pytest.mark.asyncio
async def test_a_narrowed_round_never_records_that_it_kept_every_lane(
    caplog,
) -> None:
    """The record of a narrowed round has to agree with itself.

    Same recovered path as above: block absent, one repair pass restated it,
    lane 2 dropped. The round persisted a note saying it kept every lane it
    planned beside the `dropped` entry saying it did not, and warned twice that
    the narrowing was not applied one line above the line saying it had
    narrowed. Whoever audits the round afterwards is told both things, and an
    operator who sees that warning contradicted stops reading it.
    """
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(_GROUND_A, _GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
        ]
    )
    critic_backend = _QueuedBackend(
        [
            "VERDICT: ACCEPT\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
            _width_block(
                {
                    "lane_id": 2,
                    "reason": "it re-derives lane 1's autotune lever",
                }
            ),
        ]
    )
    service = _two_lane_critic_service(orchestration_backend, critic_backend)

    with caplog.at_level(logging.INFO):
        result = await service.run(_context(), lanes=2)

    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["kept"] == 1
    assert narrowing["dropped"] == [{"lane_id": 2, "reason": "it re-derives lane 1's autotune lever"}]
    # Each note says what reading the block found and leaves the outcome to
    # `status` and `dropped`, which is the only way the three can agree.
    assert narrowing["notes"] == [
        "the review ended with no lane_narrowing block",
        "the review did not end with a readable width block; one repair pass restated it",
    ]
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert not [line for line in warnings if "not applied" in line]
    assert not [line for line in warnings if "keeps every lane" in line]
    assert "round narrowed from 2 lanes to 1" in caplog.text


@pytest.mark.asyncio
async def test_a_narrowing_the_round_refused_is_still_reported_as_one(
    caplog,
) -> None:
    """The warning has to survive where it is true.

    The review named a lane, the round is running it anyway, and this is the
    outcome the operator is being warned about. It is logged here, where what
    the round did with the ruling is known, rather than once per note while the
    reading was still going on.
    """
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {"lane_id": 1, "reason": "unsupported"},
            {"lane_id": 2, "reason": "also unsupported"},
        )
    )

    with caplog.at_level(logging.INFO):
        result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert [line for line in warnings if "narrowing was not applied (refused_empty_round)" in line]


@pytest.mark.asyncio
async def test_a_review_that_asked_for_nothing_is_not_reported_as_a_failure(
    caplog,
) -> None:
    """The empty block is an answer, and answering costs the round nothing."""
    _backend, service = _two_lane_round("VERDICT: ACCEPT\n\nBoth lanes earn it.\n" + _width_block())

    with caplog.at_level(logging.INFO):
        result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING and "narrowing" in record.getMessage()
    ]
    assert "plan critic asked for no narrowing" in caplog.text


@pytest.mark.asyncio
async def test_the_round_records_what_each_planning_phase_cost() -> None:
    """A quarter of an eleven-hour budget went on planning, a third unseen.

    Every phase but one persisted or logged its own duration; the rest could
    only be arrived at by subtracting those from the round's total, which made
    the second most expensive phase of the planning window the only one nobody
    could look up.
    """
    _backend, service = _two_lane_round(
        "VERDICT: REVISE\n\nBoth lanes need a stop condition.",
        revisions=["# Lane 1 revised", "# Lane 2 revised"],
    )

    result = await service.run(_context(), lanes=2)

    durations = result.structured_output_diagnostics["phase_durations_sec"]
    assert list(durations) == [
        "dispatch",
        "specialists",
        "partition",
        "synthesis",
        "plan_critic",
        "plan_revision",
        "total",
    ]
    assert all(value >= 0 for value in durations.values())
    phases = [name for name in durations if name != "total"]
    # The named phases account for the round without exceeding it; what they do
    # not cover stays visible as the difference rather than being distributed.
    assert sum(durations[name] for name in phases) <= durations["total"] + 0.01
    assert durations["plan_critic"] == pytest.approx(result.plan_critic.duration_sec, abs=0.001)
    assert durations["plan_revision"] == pytest.approx(
        result.structured_output_diagnostics["plan_revision"]["duration_sec"],
        abs=0.001,
    )


@pytest.mark.asyncio
async def test_a_round_that_never_partitioned_reports_no_partition_cost() -> None:
    """An absent phase is absent, not zero: zero would read as instant."""
    orchestration_backend = _QueuedBackend([_dispatch_payload(), "# Optimization plan\nUse vector loads."])
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
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
    )

    result = await service.run(_context())

    durations = result.structured_output_diagnostics["phase_durations_sec"]
    assert list(durations) == ["dispatch", "specialists", "synthesis", "total"]


@pytest.mark.asyncio
async def test_a_lane_that_cannot_be_revised_keeps_its_draft() -> None:
    """The verdict was about the round, not about that lane being dangerous.

    Its siblings were revised, so replacing the whole round with the
    non-executable fallback would throw away work over one lost follow-up turn.
    A single-lane round still falls back: there is nothing else left in it.
    """
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(_GROUND_A, _GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
            AgentRunResult(text="", end_reason="turn_cap"),
            "# Lane 2 revised",
        ]
    )
    service = _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend("VERDICT: REVISE\n\nBoth lanes need a stop condition."),
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 revised")
    assert result.optimization_plan_executable is True
    assert result.plan_revised is True
    diagnostics = result.structured_output_diagnostics["plan_revision"]
    assert diagnostics["status"] == "partially_revised"
    assert diagnostics["unrevised_lanes"] == [1]
    assert "turn_cap" in diagnostics["message"]


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
    assert result.optimization_plan == revised
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

    assert result.optimization_plan == revised
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
    # The resumed session already holds the planning bundle, and the revision
    # instructions are the system prompt; neither is copied back into the payload.
    assert "For REPLACE, discard the dominated implementation route" in (spec.system_prompt)
    assert "revision_instructions" not in revision_payload
    revision_diagnostics = result.structured_output_diagnostics["plan_revision"]
    assert revision_diagnostics["status"] == "revised"
    assert revision_diagnostics["critic_verdict"] == "REVISE"
    assert revision_diagnostics["revision_mode"] == "resumed"
    assert revision_diagnostics["duration_sec"] >= 0


@pytest.mark.asyncio
async def test_a_resumed_revision_carries_only_what_the_session_lacks() -> None:
    """The synthesis session already holds the whole planning bundle.

    A resume re-enters the lane's own synthesis session, which already contains
    the dispatch, every specialist analysis and the synthesis conversation.
    Appending a near-complete duplicate of that bundle in the feedback ran the
    revision out of context: one 12-hour run compacted the revision 17 times
    across 7 rounds, once per lane, dropping the recap it went on to answer over.
    The feedback carries only the critic's addition and the draft it judged.
    """
    backend = _ResumableQueuedBackend(["# Revised plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    revised = await agent.revise_optimization_plan(
        _context(),
        synthesis_session_id="synthesis-session",
        draft_plan="# Draft plan",
        critic_review="Compare the existing GEMM path.",
        critic_verdict="REVISE",
        specialist_outcomes=outcomes,
        dispatch_plan=dispatch,
        coverage=_coverage(["compute", "memory"]),
    )

    assert revised.mode == "resumed"
    spec, session_id, feedback = backend.resumes[0]
    assert session_id == "synthesis-session"
    payload = json.loads(feedback)
    assert payload["draft_plan"] == "# Draft plan"
    assert payload["critic_verdict"] == "REVISE"
    assert "existing GEMM" in payload["critic_review"]
    # The revision instructions are the system prompt, not a second copy inside
    # the payload; the rest of the bundle is already in the resumed session.
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
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    revised = await agent.revise_optimization_plan(
        _context(),
        synthesis_session_id="",
        draft_plan="# Draft plan",
        critic_review="Compare the existing GEMM path.",
        critic_verdict="REVISE",
        specialist_outcomes=outcomes,
        dispatch_plan=dispatch,
        coverage=_coverage(["compute", "memory"]),
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
async def test_plan_revision_turn_cap_uses_non_executable_fallback(
    caplog,
) -> None:
    draft = "# Optimization plan\nUse vector loads."
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(),
            draft,
            AgentRunResult(
                text="# Partial revision\nRead more files next.",
                end_reason="turn_cap",
            ),
        ]
    )
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
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
            backend=_StaticBackend("VERDICT: REVISE\n\nAdd a canonical comparison."),
            timeout_sec=1,
        ),
    )

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        result = await service.run(_context())

    assert result.optimization_plan_executable is False
    assert result.plan_revised is False
    assert "# Partial revision" not in result.optimization_plan
    diagnostics = result.structured_output_diagnostics["plan_revision"]
    assert diagnostics["status"] == "framework_fallback"
    assert diagnostics["duration_sec"] >= 0
    assert "turn_cap" in diagnostics["message"]
    assert "publishing a non-executable framework fallback" in caplog.text


@pytest.mark.asyncio
async def test_plan_critic_error_uses_draft_without_revision() -> None:
    draft = "# Optimization plan\nUse vector loads."
    orchestration_backend = _QueuedBackend([_dispatch_payload(), draft])
    critic_backend = _StaticBackend(
        "provider error",
        end_reason="api_error",
        stderr_tail="gateway unavailable",
    )
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
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
            backend=critic_backend,
            timeout_sec=1,
        ),
    )

    result = await service.run(_context())

    assert result.optimization_plan == draft
    assert result.plan_revised is False
    assert result.plan_critic is not None
    assert result.plan_critic.fail_open is True
    assert len(orchestration_backend.specs) == 2
    assert critic_backend.calls == 1


@pytest.mark.asyncio
async def test_plan_revision_failure_preserves_critic_in_fallback() -> None:
    draft = "# Optimization plan\nUse vector loads."
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(),
            draft,
            AgentRunResult(
                text="provider error",
                end_reason="api_error",
                stderr_tail="revision gateway unavailable",
            ),
        ]
    )
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
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
            backend=_StaticBackend("VERDICT: REVISE\n\nAdd a canonical comparison."),
            timeout_sec=1,
        ),
    )

    result = await service.run(_context())

    assert result.optimization_plan.startswith("# Optimization plan")
    assert "Critic review below as mandatory" in result.optimization_plan
    assert "Add a canonical comparison" in result.optimization_plan
    assert draft in result.optimization_plan
    assert result.optimization_plan_executable is False
    assert result.plan_revised is False
    assert result.structured_output_diagnostics["plan_revision"]["status"] == "framework_fallback"
    assert len(orchestration_backend.specs) == 3


@pytest.mark.asyncio
async def test_synthesis_failure_skips_critic_and_stays_non_executable() -> None:
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(),
            AgentRunResult(
                text="# Partial synthesis\nMore evidence follows.",
                end_reason="turn_cap",
            ),
        ]
    )
    critic_backend = _StaticBackend("VERDICT: REVISE\n\nThis must not run.")
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
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
            backend=critic_backend,
            timeout_sec=1,
        ),
    )

    result = await service.run(_context())

    assert result.optimization_plan_executable is False
    assert result.plan_critic is None
    assert result.optimization_plan_draft == ""
    assert critic_backend.calls == 0
    assert result.structured_output_diagnostics["plan_critic"] == {
        "status": "skipped_synthesis_unavailable",
    }
    assert result.structured_output_diagnostics["synthesis"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_non_api_format_failure_still_produces_implementer_plan() -> None:
    orchestration_backend = _StaticBackend("not-json")
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend(error=RuntimeError("provider failed")),
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )

    result = await service.run(_context())

    assert result.optimization_plan.startswith("# Optimization plan")
    assert "Successful specialist roles: (none)" in (result.optimization_plan)
    assert "analysis/abc123/source_map.json" in (result.optimization_plan)
    assert result.optimization_plan_executable is False
    # One plan, and that is what keeps a round nobody could synthesize out of
    # lane recovery: the loop only recovers a published set of two or more, so
    # a recovered set is always one a synthesis produced. Recovery reports
    # executability by leaving it unset, which rests on exactly this.
    assert len(result.optimization_plans) == 1
    assert orchestration_backend.calls == 2
    assert any("invalid dispatch JSON" in note for note in result.dispatch_plan.normalization_notes)


@pytest.mark.asyncio
async def test_unexpected_dispatch_exception_is_not_silently_normalized() -> None:
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=_StaticBackend(error=RuntimeError("programming failure")),
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend("unused"),
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )

    with pytest.raises(RuntimeError, match="programming failure"):
        await service.run(_context())


@pytest.mark.asyncio
async def test_service_does_not_render_fallback_for_specialist_api_outage() -> None:
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=_StaticBackend(_dispatch_payload()),
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend(
                        "SDK error text",
                        end_reason="api_error",
                        stderr_tail="gateway unavailable",
                    ),
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )

    with pytest.raises(
        OrchestrationInfrastructureError,
        match="before any analysis",
    ):
        await service.run(_context())


@pytest.mark.asyncio
async def test_diversify_partial_coverage_still_produces_plan() -> None:
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            "# Optimization plan\nVectorize the memory-bound case.",
        ]
    )
    definitions = _definitions()
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=2,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=definitions["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                ),
                "compute": SpecialistAgent(
                    definition=definitions["compute"],
                    backend=_StaticBackend(error=RuntimeError("provider failed")),
                    timeout_sec=1,
                    max_turns=2,
                ),
            },
            max_parallel=2,
        ),
        definitions=definitions,
    )
    context = replace(
        _context(),
        search_mode="DIVERSIFY",
    )

    result = await service.run(context)

    assert result.optimization_plan.startswith("# Optimization plan")
    coverage = result.structured_output_diagnostics["coverage"]
    assert coverage["successful_roles"] == ["memory"]
    assert coverage["failed_roles"] == ["compute"]
    assert coverage["missing_cases"] == ["case-b"]
    assert result.optimization_plan_executable is True
    assert len(orchestration_backend.specs) == 2
