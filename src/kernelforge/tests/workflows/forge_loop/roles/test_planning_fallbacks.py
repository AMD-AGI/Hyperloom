# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When a planning step fails: the fallback plan, what it preserves, and the failures it must not normalize."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kernelforge.workflows.forge_loop.roles import orchestration as orchestration_module
from kernelforge.agent_backends import AgentRunResult
from kernelforge.workflows.forge_loop.roles.orchestration import (
    OrchestrationAgent,
    OrchestrationInfrastructureError,
    OrchestrationService,
)
from kernelforge.workflows.forge_loop.roles.specialists import SpecialistAgent, SpecialistPool
from kernelforge.workflows.forge_loop.roles.plan_critic import PlanCriticAgent
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    _QueuedBackend,
    _StaticBackend,
    _context,
    _definitions,
    _dispatch_payload,
)


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
    assert "# Partial revision" not in result.optimization_plans[0]
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

    assert result.optimization_plans[0] == draft
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

    assert result.optimization_plans[0].startswith("# Optimization plan")
    assert "Critic review below as mandatory" in result.optimization_plans[0]
    assert "Add a canonical comparison" in result.optimization_plans[0]
    assert draft in result.optimization_plans[0]
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

    assert result.optimization_plans[0].startswith("# Optimization plan")
    assert "Successful specialist roles: (none)" in (result.optimization_plans[0])
    assert "analysis/abc123/source_map.json" in (result.optimization_plans[0])
    assert result.optimization_plan_executable is False
    # One plan, and that is what keeps a round nobody could synthesize out of lane recovery: the loop only recovers a
    # published set of two or more, so a recovered set is always one a synthesis produced.
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

    assert result.optimization_plans[0].startswith("# Optimization plan")
    coverage = result.structured_output_diagnostics["coverage"]
    assert coverage["successful_roles"] == ["memory"]
    assert coverage["failed_roles"] == ["compute"]
    assert coverage["missing_cases"] == ["case-b"]
    assert result.optimization_plan_executable is True
    assert len(orchestration_backend.specs) == 2
