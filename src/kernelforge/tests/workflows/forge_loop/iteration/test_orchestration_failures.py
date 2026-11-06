# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A planning chain that fails: the Implementer it skips, the circuit it opens, and the probe a resume still allows."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from kernelforge.agent_backends import AgentRunResult
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import (
    ORCHESTRATION_CIRCUIT_CLOSED,
    ORCHESTRATION_CIRCUIT_OPEN,
    LoopStateStore,
)
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.workflows.forge_loop.roles.contracts import SpecialistDefinition
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent, OrchestrationService
from kernelforge.workflows.forge_loop.roles.specialists import SpecialistAgent, SpecialistPool
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _AMPLE_BUDGET_SEC,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


class _OrchestrationTestBackend:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def run(self, _spec, usage=None):
        self.calls += 1
        return self.results.pop(0)


def _runner_orchestration_service(
    orchestration_backend,
    *,
    specialist_result: AgentRunResult | None = None,
):
    definition = SpecialistDefinition(
        role_id="memory",
        description="Memory specialist",
        instructions="Analyze memory access behavior.",
    )
    specialist_backend = _OrchestrationTestBackend([specialist_result or AgentRunResult(text="Use vector loads.")])
    return OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=definition,
                    backend=specialist_backend,
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": definition},
    )


def _orchestration_api_failure() -> AgentRunResult:
    return AgentRunResult(
        text="SDK error text must not become a dispatch plan",
        end_reason="api_error",
        stderr_tail="gateway unavailable",
    )


def test_orchestration_failure_skips_implementer_without_fallback(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    implementer_called = False

    async def orchestration(**_kwargs):
        return None, "synthesis failed"

    async def implementer(*_args, **_kwargs):
        nonlocal implementer_called
        implementer_called = True
        return "must not run"

    monkeypatch.setattr(loop, "_run_orchestration", orchestration)
    results = asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=implementer, orchestration_service=object(), supervisor_fn=_unused_supervisor
            )
        )
    )

    assert implementer_called is False
    assert results[0].validation_summary.startswith("ORCHESTRATION ERROR")
    state = LoopStateStore(str(workspace)).load()
    assert state.stall.no_improvement_iters == 0


def test_consecutive_orchestration_errors_open_circuit(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        session_count=5,
    )
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    backend = _OrchestrationTestBackend([_orchestration_api_failure() for _ in range(3)])
    service = _runner_orchestration_service(backend)

    async def implementer(*_args, **_kwargs):
        raise AssertionError("Implementer must not run without a plan")

    results = asyncio.run(
        loop.run(
            runner_module.LoopAgents(
                agent_fn=implementer, orchestration_service=service, supervisor_fn=_unused_supervisor
            )
        )
    )

    state = LoopStateStore(str(workspace)).load()
    assert len(results) == 3
    assert backend.calls == 3
    assert state.orchestration_error_streak == 3
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN
    assert state.termination_reason == "orchestration_failed"


def test_orchestration_failed_resume_allows_one_half_open_probe(
    tmp_path,
    monkeypatch,
):
    first, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        session_count=5,
    )
    runtime_config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    first.config = runtime_config
    first_backend = _OrchestrationTestBackend([_orchestration_api_failure() for _ in range(3)])
    asyncio.run(
        first.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                orchestration_service=_runner_orchestration_service(first_backend),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    failed_probe = IterationLoop(
        first.ic,
        first.tracker,
        config=runtime_config,
        resume=True,
    )
    monkeypatch.setattr(
        failed_probe,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(failed_probe.results) < 5 else 0.0,
    )
    failed_backend = _OrchestrationTestBackend([_orchestration_api_failure()])
    failed_results = asyncio.run(
        failed_probe.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                orchestration_service=_runner_orchestration_service(failed_backend),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    reopened = LoopStateStore(str(workspace)).load()
    assert len(failed_results) == 1
    assert failed_backend.calls == 1
    assert reopened.orchestration_error_streak == 4
    assert reopened.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN
    assert reopened.termination_reason == "orchestration_failed"

    successful_probe = IterationLoop(
        first.ic,
        first.tracker,
        config=runtime_config,
        resume=True,
    )
    monkeypatch.setattr(
        successful_probe,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(successful_probe.results) < 1 else 0.0,
    )
    successful_backend = _OrchestrationTestBackend(
        [
            AgentRunResult(
                text=json.dumps(
                    {
                        "assignments": [
                            {
                                "role_id": "memory",
                                "target_case_ids": ["case"],
                                "reason": "Inspect memory access.",
                            }
                        ]
                    }
                )
            ),
            AgentRunResult(text="# Optimization plan\nVectorize global loads."),
        ]
    )
    successful_results = asyncio.run(
        successful_probe.run(
            runner_module.LoopAgents(
                agent_fn=_no_change_agent,
                orchestration_service=_runner_orchestration_service(successful_backend),
                supervisor_fn=_unused_supervisor,
            )
        )
    )

    closed = LoopStateStore(str(workspace)).load()
    assert len(successful_results) == 1
    assert successful_backend.calls == 2
    assert closed.orchestration_error_streak == 0
    assert closed.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_CLOSED
    assert closed.termination_reason == "budget_exhausted"
