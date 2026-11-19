# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for Forge controller allocation, resume constraints, and elapsed-time accounting."""

from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import pytest
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.cli.bootstrap import constraint_budget_resume_conflict
from hyperloom.inference_optimizer.session.manifest import build_manifest
from hyperloom.orchestrator.kernel import controller_submit
from hyperloom.orchestrator.phases import kernel_controller_attempt as controller_attempt
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState


def test_explicit_allocation_is_a_recorded_optimize_option(tmp_path):
    args = _build_parser().parse_args(
        ["optimize", "--model", str(tmp_path), "--forge-controller-allocation-minutes", "600"]
    )
    assert args.forge_controller_allocation_minutes == 600
    assert build_manifest(tmp_path, args=args)["forge_controller_allocation_minutes"] == 600


@pytest.mark.parametrize(
    ("stored_sec", "requested_minutes", "conflicts"),
    [(36000, 600, False), (7200, 600, True), (36000, None, False)],
)
def test_resume_requires_an_explicit_allocation_to_match(tmp_path, stored_sec, requested_minutes, conflicts):
    argv = ["optimize", "--model", str(tmp_path)]
    if requested_minutes is not None:
        argv.extend(["--forge-controller-allocation-minutes", str(requested_minutes)])
    args = _build_parser().parse_args(argv)
    state = SharedState(framework="vllm", forge_controller_allocation_sec=stored_sec)

    reason = constraint_budget_resume_conflict(args, state)

    assert ("--forge-controller-allocation-minutes" in reason) is conflicts


def test_matching_allocation_keeps_the_persisted_agentx_accuracy_round(tmp_path):
    args = _build_parser().parse_args(
        ["optimize", "--model", str(tmp_path), "--forge-controller-allocation-minutes", "600", "--no-agentx-accuracy"]
    )
    state = SharedState(framework="vllm", forge_controller_allocation_sec=36000, agentx_accuracy=True)

    assert "accuracy round" in constraint_budget_resume_conflict(args, state)


@pytest.mark.parametrize(("remaining", "expected"), [(36000, (36000, 36030)), (7200, (7200, 7230)), (0, (0, 30))])
def test_controller_timeout_respects_the_remaining_allocation(remaining, expected):
    state = SimpleNamespace(
        forge_controller_allocation_sec=36000,
        kernel_rewrite_controller_result={"controller_budget_remaining_sec": remaining},
        closing_reserve_sec=lambda: 600,
    )
    phase = SimpleNamespace(shared_state=state)
    with (
        patch.object(controller_attempt._phase_state, "session_remaining_seconds", return_value=72000),
        patch.object(controller_attempt._phase_state, "phase_budget_remaining_seconds", return_value=45000),
        patch.object(controller_attempt, "measurement_round_sec", return_value=3600),
    ):
        assert KernelPhase._kernel_rewrite_controller_timeouts(phase) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_controller_charges_elapsed_time_even_when_invocation_fails(failure):
    state = SimpleNamespace(
        forge_controller_allocation_sec=36000,
        kernel_rewrite_controller_result={"controller_budget_remaining_sec": 7200},
        macro_cycle=1,
    )
    phase = SimpleNamespace(shared_state=state, _kernel_rewrite_controller_timeouts=lambda: (7200, 7230))
    result = {"status": "completed", "controller_budget_remaining_sec": 999999, "budget_minutes": 9999}
    with (
        patch.object(
            controller_submit,
            "run_controller_subprocess",
            return_value=result,
            side_effect=ValueError("invalid result") if failure else None,
        ),
        patch.object(controller_attempt, "monotonic", side_effect=[1000, 1600]),
    ):
        actual = await KernelPhase._controller_attempt_result(phase, Path("handoff"), Path("output"))
    assert actual["controller_budget_remaining_sec"] == 6600
    assert actual["budget_minutes"] == 120
    assert actual["hard_timeout_sec"] == 7230
    assert actual["status"] == ("failed" if failure else "completed")


@pytest.mark.asyncio
async def test_zero_budget_attempt_retains_zero_and_never_starts_another_controller():
    state = SimpleNamespace(
        forge_controller_allocation_sec=36000,
        kernel_rewrite_controller_result={"controller_budget_remaining_sec": 0},
        macro_cycle=2,
    )
    phase = SimpleNamespace(shared_state=state, _kernel_rewrite_controller_timeouts=lambda: (0, 30))
    with patch.object(controller_submit, "run_controller_subprocess") as run:
        actual = await KernelPhase._controller_attempt_result(phase, Path("handoff"), Path("output"))
    assert actual["status"] == "no_result"
    assert actual["controller_budget_remaining_sec"] == 0
    run.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_attempt_keeps_its_spend_when_the_session_resumes(tmp_path):
    state = SharedState(forge_controller_allocation_sec=7200)
    state.save(tmp_path)
    phase = SimpleNamespace(
        shared_state=state,
        session_dir=tmp_path,
        _kernel_rewrite_controller_timeouts=lambda: (7200, 7230),
    )
    phase._controller_attempt_result = MethodType(KernelPhase._controller_attempt_result, phase)
    with (
        patch.object(controller_submit, "run_controller_subprocess", return_value={"status": "cancelled"}),
        patch.object(controller_attempt, "monotonic", side_effect=[1000, 1600]),
    ):
        await KernelPhase._run_kernel_rewrite_controller(phase, Path("handoff"), Path("output"), {})

    resumed = SharedState.load_or_init(tmp_path)
    assert resumed.kernel_rewrite_controller_result["status"] == "cancelled"
    assert resumed.kernel_rewrite_controller_result["controller_budget_remaining_sec"] == 6600
    with (
        patch.object(controller_attempt._phase_state, "session_remaining_seconds", return_value=72000),
        patch.object(controller_attempt._phase_state, "phase_budget_remaining_seconds", return_value=45000),
        patch.object(controller_attempt, "measurement_round_sec", return_value=3600),
    ):
        assert KernelPhase._kernel_rewrite_controller_timeouts(SimpleNamespace(shared_state=resumed)) == (6600, 6630)
