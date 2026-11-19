# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The controller entry's allocation reaches the real CLI and operator scheduler."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from functools import partial
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from hyperloom.orchestrator.kernel import controller_submit
from hyperloom.orchestrator.phases import kernel_controller_attempt as controller_attempt
from hyperloom.orchestrator.phases.kernel import KernelPhase
from kernelforge import cli
from kernelforge.knowledge.kb_store.identity.kernel_recipe import KernelRecipeIdentity, kernel_recipe_canonical_id
from kernelforge.workflows.rewrite_controller import controller, scheduler
from kernelforge.workflows.rewrite_controller.contracts import (
    SERVING_CONTEXT_FILENAME,
    TRACE_EVIDENCE_FILENAME,
    WORKLOAD_FILENAME,
)
from kernelforge.workflows.rewrite_controller.dispatcher import SingleTaskResult
from kernelforge.workflows.rewrite_controller.opportunity_agent import (
    ANALYSIS_STATUS_COMPLETED,
    OpportunityAnalysisResult,
)
from kernelforge.workflows.rewrite_controller.task import load_task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allocation_sec", "expected"),
    [(36000, [19000, 37000]), (None, [6400, 11800])],
    ids=["shared-allocation", "default-operator-cap"],
)
async def test_controller_entry_applies_the_allocation_policy(tmp_path, monkeypatch, allocation_sec, expected):
    clock = [1000.0]
    deadlines = []
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    for filename in (WORKLOAD_FILENAME, SERVING_CONTEXT_FILENAME, TRACE_EVIDENCE_FILENAME):
        (handoff / filename).write_text("Test workload\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    def prepare_tasks(*, layout, **_kwargs):
        for priority, name in enumerate(("up", "down")):
            identity = KernelRecipeIdentity.from_mapping(
                dict(
                    producer="forge-loop",
                    kernel_name=name,
                    framework="vllm",
                    framework_version="0.27.1",
                    backend="triton",
                    gpu="mi355x",
                )
            )
            task_dir = layout.task_dir(kernel_recipe_canonical_id(identity))
            task_dir.mkdir(parents=True)
            (task_dir / "driver.py").write_text("pass\n", encoding="utf-8")
            payload = dict(
                identity=asdict(identity),
                base_commit="a" * 40,
                repo_root=str(repo),
                kernel_path="kernel.py",
                operator_name=name,
                driver_path="driver.py",
                priority=priority,
            )
            (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")
        return OpportunityAnalysisResult(status=ANALYSIS_STATUS_COMPLETED, published_task_count=2)

    def run_operator(task_dir, *, deadline_unix, **_kwargs):
        task = load_task(task_dir, record_state=False).task
        deadlines.append(deadline_unix)
        clock[0] = deadline_unix
        return SingleTaskResult(
            task=task, worktree=None, forge_outcome=None, patch_path=None, status="succeeded", reason=""
        )

    def run_cli(command, **_kwargs):
        result = CliRunner().invoke(cli.main, command[3:])
        assert result.exit_code == 0, result.output
        return subprocess.CompletedProcess(command, result.exit_code, stdout=result.output, stderr="")

    monkeypatch.setattr(controller, "run_opportunity_analysis", prepare_tasks)
    monkeypatch.setattr(controller, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(
        controller, "dispatch_prepared_tasks", partial(scheduler.dispatch_prepared_tasks, clock=lambda: clock[0])
    )
    monkeypatch.setattr(scheduler, "dispatch_single_task", run_operator)
    monkeypatch.setattr(controller_submit, "run_with_session_kill", run_cli)
    monkeypatch.setattr(controller_attempt, "monotonic", lambda: clock[0])
    phase = SimpleNamespace(
        shared_state=SimpleNamespace(
            forge_controller_allocation_sec=allocation_sec, kernel_rewrite_controller_result={}, macro_cycle=0
        ),
        _kernel_rewrite_controller_timeouts=lambda: (36000, 36030),
    )

    result = await KernelPhase._controller_attempt_result(phase, handoff, tmp_path / "output")

    assert deadlines == expected
    assert result["task_count"] == 2
