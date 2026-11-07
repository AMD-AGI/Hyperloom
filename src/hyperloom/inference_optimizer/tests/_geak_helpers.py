# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared GEAK test helpers: the measured round, its replay env spec, the delegation launch gate, resume-pending fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.phases.kernel import KernelPhase
from inference_suite.contracts import RunLayout
from inference_suite.contracts.launch import Launch
from inference_suite.tests.test_compose import _launch


def measured_in_suite_round(state, run_dir: Path, launch: Launch | None = None) -> Launch:
    """Record ``current_best`` as measured in ``run_dir``, whose ``launch.json`` records ``launch``."""
    launch = launch or _launch()
    replay_env_spec(run_dir, launch)
    state.current_best_measurement = {**(state.current_best_measurement or {}), "benchmark_workspace": str(run_dir)}
    return launch


def replay_env_spec(run_dir: Path, launch: Launch | None = None) -> dict[str, str]:
    """An env spec naming the ``launch.json`` a GEAK replay takes its serving identity from."""
    launch = launch or _launch()
    RunLayout(run_dir).root.mkdir(parents=True, exist_ok=True)
    RunLayout(run_dir).launch.write_text(json.dumps(launch.to_document()), encoding="utf-8")
    return {"launch_json": str(RunLayout(run_dir).launch)}


def stop_geak_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave GEAK no run budget, so the phase returns once the handoff is written."""
    monkeypatch.setattr(KernelPhase, "_geak_timeouts", lambda _self: (0, 0, True))


def forbid_geak_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the phase gets past its recovery short-circuit to the runner launch."""

    def _launch_reached(_self: KernelPhase) -> tuple[int, int, bool]:
        raise AssertionError("GEAK runner launch reached")

    monkeypatch.setattr(KernelPhase, "_geak_timeouts", _launch_reached)


def resume_pending(state) -> bool:
    """Compute resume_pending_revalidation the same way session_facts.py does."""
    stack_len = len(state.optimization_stack)
    validated_len = int(state.cumulative_gain_validated_stack_len or 0)
    working_gen = int(getattr(state, "working_recipe_generation", 0) or 0)
    validated_gen = int(state.validated_recipe_generation or 0)
    return stack_len > validated_len or working_gen != validated_gen


def set_resume_pending(state, value: bool) -> None:
    """Drive the underlying fields that session_facts reads to produce ``value``."""
    if value:
        state.working_recipe_generation = int(state.validated_recipe_generation or 0) + 1
    else:
        state.validated_recipe_generation = int(getattr(state, "working_recipe_generation", 0) or 0)
        state.cumulative_gain_validated_stack_len = len(state.optimization_stack)
