# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A loop over a campaign workspace that survives across processes, shared by the cross-process and coverage tests."""

from __future__ import annotations

from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _AMPLE_BUDGET_SEC
from kernelforge.workflows.forge_loop.iteration.runner import IterationConfig, IterationLoop, IterationResult


def _make_loop(workspace, kernel, driver, tracker, *, session_count, resume=False):
    config = IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        baseline_wall_ms=1.0,
        baseline_case_times={"case": 1.0},
        max_time_hours=1.0,
        git_branch="campaign-test",
        workspace_dir=str(workspace),
    )
    loop = IterationLoop(
        config,
        tracker,
        config=object(),
        resume=resume,
    )
    loop._time_remaining = lambda: _AMPLE_BUDGET_SEC if len(loop.results) < session_count else 0.0
    return loop


async def _successful_iteration(self, iteration, plan=""):
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=1.0 - iteration * 0.05,
        mean_case_speedup=1.0 / (1.0 - iteration * 0.05),
        snr_db=40.0,
        kept=True,
    )
