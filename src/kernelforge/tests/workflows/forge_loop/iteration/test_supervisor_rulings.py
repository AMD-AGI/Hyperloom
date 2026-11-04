# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The supervisor's rulings: when one expires, what an intervention persists first, and what a free-form ruling still requires."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import PHASE_STALLED, BestRecord, LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.workflows.forge_loop.iteration import locked_run
from kernelforge.workflows.forge_loop.roles.supervisor import latest_supervisor_ruling_path
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _make_loop


def test_keep_expires_active_supervisor_ruling(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.best_wall_ms = 0.8
    loop.best_mean_case_speedup = 1.25
    loop._supervisor_ruling = "Continue the previous stall direction."
    ruling_path = latest_supervisor_ruling_path(str(workspace))
    ruling_path.parent.mkdir(parents=True, exist_ok=True)
    ruling_path.write_text(loop._supervisor_ruling)
    result = IterationResult(
        iteration=1,
        duration_sec=0.1,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=0.8,
        mean_case_speedup=1.25,
        kept=True,
        commit_hash="deadbeef",
    )

    loop._record_iteration_outcome(result, plan="new canonical best")

    assert loop._supervisor_ruling == ""
    assert not ruling_path.exists()


def test_intervention_reset_is_persisted_before_agent_runs(tmp_path, monkeypatch):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        resume=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    store = LoopStateStore(str(workspace))
    state = RunState(
        baseline_case_times={"case": 1.0},
        best_case_times={"case": 0.8},
        best=BestRecord(
            iteration=2,
            wall_ms=0.8,
            mean_case_speedup=1.25,
            commit_hash=head,
        ),
        phase=PHASE_STALLED,
    )
    state.stall.no_improvement_iters = 1
    state.stall.last_supervisor_iter = -10_000
    store.save(state)

    async def supervisor(**_kwargs):
        return (
            "# Supervisor Ruling\n\n"
            "The prior hard-floor conclusion was unsupported. "
            "Try a different kernel decomposition."
        )

    async def cancelled_agent(*_args, **_kwargs):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=cancelled_agent, supervisor_fn=supervisor)))

    persisted = LoopStateStore(str(workspace)).load()
    assert loop.run_state.stall.no_improvement_iters == 0
    assert loop.run_state.stall.last_supervisor_iter == 1
    assert loop.run_state.phase != PHASE_STALLED
    assert persisted.stall.no_improvement_iters == 0
    assert persisted.stall.last_supervisor_iter == 1


def test_empty_supervisor_expires_prior_ruling_without_resetting_stall(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        session_count=3,
    )
    loop.ic.supervise_cooldown = 3
    histories = []
    supervisor_calls = 0
    agent_calls = 0
    stale_ruling = "Continue the stale memory-only direction."

    async def empty_supervisor(**_kwargs):
        nonlocal supervisor_calls
        supervisor_calls += 1
        return ""

    async def editing_agent(kernel_path, history, session_sink):
        nonlocal agent_calls
        agent_calls += 1
        histories.append(history)
        if agent_calls == 1:
            loop._supervisor_ruling = stale_ruling
            ruling_path = latest_supervisor_ruling_path(str(workspace))
            ruling_path.parent.mkdir(parents=True, exist_ok=True)
            ruling_path.write_text(stale_ruling)
        session_sink["plan"] = "repeat candidate"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "candidate"

    async def reverted_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.1,
            mean_case_speedup=0.9,
            kept=False,
            bench_detail={"case_times": {"case": 1.1}},
        )

    monkeypatch.setattr(loop, "run_one_iteration", reverted_iteration)
    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=editing_agent, supervisor_fn=empty_supervisor)))

    assert loop.monitor.intervention_count == 0
    assert loop.run_state.stall.no_improvement_iters == 3
    assert supervisor_calls == 1
    assert loop.run_state.stall.last_supervisor_attempt_iter == 2
    assert loop.run_state.stall.last_supervisor_iter == 0
    assert "Mode: DIVERSIFY" in histories[1]
    assert stale_ruling not in histories[1]
    assert loop._supervisor_ruling == ""
    assert not locked_run.latest_supervisor_ruling_path(str(workspace)).exists()


def test_free_form_supervisor_ruling_still_creates_fresh_plan_each_iteration(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        session_count=3,
    )
    loop.ic.supervise_cooldown = 2
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    orchestration_calls = []
    histories = []
    supervisor_evidence = []
    ruling = (
        "\n# Supervisor Ruling\n\n"
        "The memory path still has measured headroom. The earlier lesson's "
        "hard-floor conclusion is not supported by the recorded attempts.\n"
    )

    async def orchestration(*, iteration, **_kwargs):
        orchestration_calls.append(iteration)
        plan_path = workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text(f"# Optimization plan {iteration}\n")
        loop._latest_optimization_plan_path = str(plan_path)
        return plan_path, ""

    async def supervisor(**kwargs):
        supervisor_evidence.append(kwargs["evidence_context"])
        return ruling

    async def editing_agent(kernel_path, history, session_sink):
        histories.append(history)
        session_sink["plan"] = "continue memory path"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "implemented another memory milestone"

    async def reverted_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.1,
            mean_case_speedup=0.9,
            kept=False,
            bench_detail={"case_times": {"case": 1.1}},
        )

    monkeypatch.setattr(loop, "_run_orchestration", orchestration)
    monkeypatch.setattr(loop, "run_one_iteration", reverted_iteration)
    asyncio.run(
        loop.run(
            runner_module.LoopAgents(agent_fn=editing_agent, orchestration_service=object(), supervisor_fn=supervisor)
        )
    )

    assert orchestration_calls == [1, 2, 3]
    assert len(histories) == 3
    assert "iter_001/optimization_plan.md" in histories[0]
    assert "iter_002/optimization_plan.md" in histories[1]
    assert "iter_003/optimization_plan.md" in histories[2]
    assert "Mode: DIVERSIFY" in histories[1]
    assert "Latest Supervisor Ruling" in histories[1]
    assert "hard-floor conclusion is not supported" in histories[1]
    assert "hard-floor conclusion is not supported" in histories[2]
    assert len(supervisor_evidence) == 1
    evidence = json.loads(supervisor_evidence[0])
    assert evidence["latest_optimization_plan"].endswith("iter_001/optimization_plan.md")
    assert "orchestration_context" in evidence
    assert evidence["orchestration_context"]["search_policy"]["mode"] == "EXPLOIT"
    assert evidence["artifact_paths"]["latest_lesson"].endswith("lessons/iter_001.md")
    assert "latest_handoff" not in evidence["artifact_paths"]

    supervisor_root = workspace / "forge_experiments" / "supervisor"
    assert (supervisor_root / "latest.md").read_text() == ruling
    interaction = (supervisor_root / "intervention_iter_002.md").read_text()
    assert "source: injected callback" in interaction
    assert ruling in interaction
