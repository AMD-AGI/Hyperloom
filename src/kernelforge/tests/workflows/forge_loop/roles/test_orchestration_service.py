# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration service end to end: dispatch then synthesis, outages it must not turn into plans, and the probe hazard it reports."""

from __future__ import annotations

import pytest

from kernelforge.utils.process_reaping import ReapReport
from kernelforge.workflows.forge_loop.roles.contracts import DispatchPlan, SpecialistRound
from kernelforge.workflows.forge_loop.roles.orchestration import (
    OrchestrationAgent,
    OrchestrationInfrastructureError,
    OrchestrationService,
)
from kernelforge.workflows.forge_loop.roles import specialists as specialists_module
from kernelforge.workflows.forge_loop.roles.specialists import SpecialistAgent, SpecialistPool, SpecialistProbeConfig
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    _QueuedBackend,
    _StaticBackend,
    _assignment,
    _context,
    _definitions,
    _dispatch_payload,
    _single_lane_service,
)


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

    result = await agent._synthesize_optimization_plan_result(
        _context(),
        SpecialistRound(
            dispatch,
            (outcome,),
            {
                "successful_roles": ["memory"],
                "covered_cases": ["case-a"],
                "missing_cases": ["case-b"],
                "failed_roles": [],
            },
        ),
    )

    assert result.text == plan_markdown
    prompt = backend.specs[0].system_prompt
    assert "expected value" in prompt
    assert "feasibility" in prompt
    assert "not a catalog" in prompt


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
        await agent._synthesize_optimization_plan_result(
            _context(),
            SpecialistRound(
                DispatchPlan(
                    analysis_commit="abc123",
                    assignments=(_assignment(),),
                ),
                (outcome,),
                {
                    "successful_roles": ["memory"],
                    "covered_cases": ["case-a"],
                    "missing_cases": ["case-b"],
                    "failed_roles": [],
                },
            ),
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

    assert result.optimization_plans[0] == plan_markdown
    assert result.specialist_outcomes[0].content == ("# Analysis\nVector loads are feasible.")
    assert len(orchestration_backend.specs) == 2


@pytest.mark.asyncio
async def test_a_leaked_probe_is_reported_to_the_loop_that_must_refuse(tmp_path, monkeypatch) -> None:
    """The round's teardown finding has to leave the analysis phase."""

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
    assert result.optimization_plans[0]


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
    result = await _single_lane_service().run(_context())

    assert result.optimization_plans[0].startswith("# Optimization plan")
    assert result.optimization_plan_draft == ""
    assert result.plan_critic is None
