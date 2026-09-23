# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a terminal says about the run, read the same way everywhere."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.breakdown import stop_reasons as sr
from hyperloom.inference_optimizer.breakdown.collectors.v6 import _outcome_status
from hyperloom.orchestrator.bringup import ARGV_INVALID, ENV_FAULT
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("target_reached", "completed"),
        ("sweep_done", "completed"),
        ("sweep_failed", "failed"),
        ("signal", "aborted"),
        ("", "aborted"),
        ("baseline_failed", "failed"),
        ("enablement_attempts_exhausted", "failed"),
        # A verdict about the model reached before the loop started is still a
        # verdict, so it stays on the failure side.
        ("unsupported_model_arch", "failed"),
    ],
)
def test_the_outcome_vocabulary_is_stable(stop_reason, expected):
    """The categories a reader already relies on keep their answers."""
    assert sr.outcome_status(stop_reason, 1.0) == expected


def test_a_success_shaped_reason_needs_a_baseline_measurement_to_read_completed():
    """time_exhausted with no measurement ever produced is not a completed run."""
    assert sr.outcome_status("target_reached", 0.0) == "failed"
    assert sr.outcome_status("time_exhausted", 0.0) == "failed"


def test_a_host_fault_is_infrastructure_not_a_verdict_about_the_model():
    """An environment terminal ends the run without judging what it was optimizing."""
    assert ENV_FAULT in sr.INFRASTRUCTURE_STOP_REASONS
    assert sr.outcome_status(ENV_FAULT, 1.0) == "aborted"
    assert sr.outcome_status(ENV_FAULT, 1.0) != "failed"


def test_a_refused_argv_is_the_harness_faulting_not_the_model():
    """An argument the installed parser never had is a harness fault, so it reads like one."""
    assert ARGV_INVALID in sr.INFRASTRUCTURE_STOP_REASONS
    assert sr.outcome_status(ARGV_INVALID, 1.0) == "aborted"


@pytest.mark.parametrize("reason", sorted(sr.INFRASTRUCTURE_STOP_REASONS))
def test_an_infrastructure_terminal_survives_being_written_to_the_state(reason):
    """The state's closed vocabulary admits every one of these terminals."""
    state = SharedState()

    written = state.set_stop_reason(reason)

    assert written == reason
    assert state.stop_reason == reason
    assert sr.outcome_status(state.stop_reason, 1.0) == "aborted"


def test_the_new_category_is_consulted_by_the_function_that_derives_the_outcome():
    """A category no derivation reads is a category that changes nothing."""
    for reason in sr.INFRASTRUCTURE_STOP_REASONS:
        assert sr.outcome_status(reason, 1.0) == "aborted"
        assert _outcome_status(reason, 1.0) == "aborted"


def test_the_projection_derives_the_outcome_from_the_shared_mapping():
    """One mapping, so a section cannot answer differently by keeping its own copy."""
    reasons = (
        *sr.SUCCESS_STOP_REASONS,
        *sr.ABORTED_STOP_REASONS,
        *sr.INFRASTRUCTURE_STOP_REASONS,
        *sr.MODEL_GATE_STOP_REASONS,
        "baseline_failed",
        "server_argv_invalid",
    )
    for reason in reasons:
        assert _outcome_status(reason, 1.0) == sr.outcome_status(reason, 1.0)


def test_every_classified_terminal_is_one_the_state_machine_can_actually_write():
    """A classified reason outside the vocabulary is a rule for a dead terminal."""
    from hyperloom.orchestrator.phases.machine_state import STOP_REASON_VOCAB

    classified = (
        sr.SUCCESS_STOP_REASONS | sr.ABORTED_STOP_REASONS | sr.INFRASTRUCTURE_STOP_REASONS | sr.MODEL_GATE_STOP_REASONS
    )
    assert not (classified - STOP_REASON_VOCAB)
