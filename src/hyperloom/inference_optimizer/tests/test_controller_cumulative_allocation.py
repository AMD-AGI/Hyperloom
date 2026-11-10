# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fix(kernelforge): retain controller allocation across cycles and failures."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.cli.bootstrap import budget_resume_conflict
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
    state = SharedState(forge_controller_allocation_sec=600 * 60)
    state.save(tmp_path)
    assert SharedState.load_or_init(tmp_path).forge_controller_allocation_sec == 600 * 60


def test_omitted_allocation_is_absent_from_the_session_budget(tmp_path):
    args = _build_parser().parse_args(["optimize", "--model", str(tmp_path)])
    assert args.forge_controller_allocation_minutes is None
    assert build_manifest(tmp_path, args=args)["forge_controller_allocation_minutes"] is None
    assert SharedState().forge_controller_allocation_sec is None


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

    reason = budget_resume_conflict(args, state)

    assert ("--forge-controller-allocation-minutes" in reason) is conflicts


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
async def test_controller_charges_elapsed_time_even_when_invocation_fails(monkeypatch, failure):
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
        patch.object(controller_attempt.time, "monotonic", side_effect=[1000, 1600]),
    ):
        actual = await KernelPhase._controller_attempt_result(phase, Path("handoff"), Path("output"))
    assert actual["controller_budget_remaining_sec"] == 6600
    assert actual["budget_minutes"] == 120
    assert actual["hard_timeout_sec"] == 7230
    assert actual["status"] == ("failed" if failure else "completed")


@pytest.mark.asyncio
async def test_zero_budget_attempt_retains_zero_and_never_starts_another_controller(monkeypatch):
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


def test_unconfigured_allocation_uses_each_cycle_phase_budget_even_with_an_old_balance(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_FORGE_BUDGET_SEC", "36000")
    state = SimpleNamespace(
        forge_controller_allocation_sec=None,
        kernel_rewrite_controller_result={"controller_budget_remaining_sec": 0},
        closing_reserve_sec=lambda: 600,
        macro_cycle=0,
    )
    phase = SimpleNamespace(shared_state=state)
    with (
        patch.object(controller_attempt._phase_state, "session_remaining_seconds", return_value=72000),
        patch.object(controller_attempt._phase_state, "phase_budget_remaining_seconds", return_value=45000),
        patch.object(controller_attempt, "measurement_round_sec", return_value=3600),
    ):
        first = KernelPhase._kernel_rewrite_controller_timeouts(phase)
        state.macro_cycle = 1
        second = KernelPhase._kernel_rewrite_controller_timeouts(phase)
    assert first == second == (41370, 41400)


@pytest.mark.asyncio
async def test_unconfigured_allocation_does_not_persist_a_controller_balance():
    state = SimpleNamespace(
        forge_controller_allocation_sec=None,
        kernel_rewrite_controller_result={"controller_budget_remaining_sec": 0},
        macro_cycle=1,
    )
    phase = SimpleNamespace(shared_state=state, _kernel_rewrite_controller_timeouts=lambda: (5400, 5430))
    with patch.object(controller_submit, "run_controller_subprocess", return_value={"status": "completed"}) as run:
        result = await KernelPhase._controller_attempt_result(phase, Path("handoff"), Path("output"))
    assert result["budget_minutes"] == 90
    assert "controller_budget_remaining_sec" not in result
    assert run.call_args.kwargs["explicit_allocation"] is False
