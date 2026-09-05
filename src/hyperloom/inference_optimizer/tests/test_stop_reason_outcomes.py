# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a terminal says about the run, read the same way everywhere.

Two consumers derive an outcome from a ``stop_reason``: the live run snapshot
the recorder writes while the session runs, and the session projection the
exporter writes at the end. They must agree, and a terminal that names a fault
in the host must not be reported as a verdict about the model.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.breakdown import stop_reasons as sr
from hyperloom.inference_optimizer.breakdown.collectors.v6 import _outcome_status
from hyperloom.inference_optimizer.breakdown.recorder.instrument import derive_outcome_status
from hyperloom.orchestrator.bringup import ARGV_INVALID, ENV_FAULT
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("target_reached", "completed"),
        ("sweep_done", "completed"),
        ("signal", "aborted"),
        ("user_stop_requested", "aborted"),
        ("", "aborted"),
        ("baseline_failed", "failed"),
        ("enablement_stalled", "failed"),
        # A verdict about the model reached before the loop started is still a
        # verdict, so it stays on the failure side.
        ("unsupported_model_arch", "failed"),
    ],
)
def test_the_outcome_vocabulary_is_stable(stop_reason, expected):
    """The categories a reader already relies on keep their answers."""
    assert sr.outcome_status(stop_reason) == expected


def test_a_host_fault_is_infrastructure_not_a_verdict_about_the_model():
    """An environment terminal ends the run without judging what it was optimizing."""
    assert ENV_FAULT in sr.INFRASTRUCTURE_STOP_REASONS
    assert sr.outcome_status(ENV_FAULT) == "aborted"
    assert sr.outcome_status(ENV_FAULT) != "failed"


def test_a_refused_argv_is_the_harness_faulting_not_the_model():
    """An argument the installed parser never had is a harness fault, so it reads like one."""
    assert ARGV_INVALID in sr.INFRASTRUCTURE_STOP_REASONS
    assert sr.outcome_status(ARGV_INVALID) == "aborted"


@pytest.mark.parametrize("reason", sorted(sr.INFRASTRUCTURE_STOP_REASONS))
def test_an_infrastructure_terminal_survives_being_written_to_the_state(reason):
    """The state's closed vocabulary admits every one of these terminals.

    Asserting on the constants alone cannot see this: ``set_stop_reason`` maps
    anything the vocabulary does not list to ``"unknown"``, and ``"unknown"``
    is a failure. A terminal missing from the vocabulary therefore reaches the
    report as a verdict about the model no matter what this module says about
    it, and nothing but a write through the state notices.
    """
    state = SharedState()

    written = state.set_stop_reason(reason)

    assert written == reason
    assert state.stop_reason == reason
    assert sr.outcome_status(state.stop_reason) == "aborted"


def test_the_new_category_is_consulted_by_the_function_that_derives_the_outcome():
    """A category no derivation reads is a category that changes nothing.

    The model-gate set is the counterexample the codebase already carries: it
    names the stage a session reached and is deliberately not consulted here.
    """
    for reason in sr.INFRASTRUCTURE_STOP_REASONS:
        assert sr.outcome_status(reason) == "aborted"
        assert _outcome_status(reason) == "aborted"
        assert derive_outcome_status(reason) == "aborted"


def test_both_consumers_derive_the_outcome_the_same_way():
    """One mapping, so a live snapshot and an exported outcome cannot disagree."""
    reasons = (
        *sr.SUCCESS_STOP_REASONS,
        *sr.ABORTED_STOP_REASONS,
        *sr.INFRASTRUCTURE_STOP_REASONS,
        *sr.MODEL_GATE_STOP_REASONS,
        "baseline_failed",
        "server_argv_invalid",
    )
    for reason in reasons:
        assert _outcome_status(reason) == derive_outcome_status(reason) == sr.outcome_status(reason)


def test_every_classified_terminal_is_one_the_state_machine_can_actually_write():
    """A classified reason outside the vocabulary is a rule for a dead terminal.

    ``SharedState.set_stop_reason`` refuses anything outside
    ``STOP_REASON_VOCAB``, so a name only this module knows can never reach it.
    It is not inert either: it reads as a live rule, and the terminal it
    silently stops covering keeps being classified by the fallthrough.
    """
    from hyperloom.orchestrator.phases.machine_state import STOP_REASON_VOCAB

    classified = (
        sr.SUCCESS_STOP_REASONS | sr.ABORTED_STOP_REASONS | sr.INFRASTRUCTURE_STOP_REASONS | sr.MODEL_GATE_STOP_REASONS
    )
    assert not (classified - STOP_REASON_VOCAB)
