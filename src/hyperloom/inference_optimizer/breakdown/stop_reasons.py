# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a terminal ``stop_reason`` says about how a session ended."""

from __future__ import annotations

from hyperloom.orchestrator.bringup.argv_preflight import ARGV_INVALID
from hyperloom.orchestrator.bringup.env_preflight import ENV_FAULT
from hyperloom.orchestrator.supervisor.watch import DIED_STOP_REASON, WEDGED_STOP_REASON

#: Terminals that mean the run optimized and closed normally.
SUCCESS_STOP_REASONS: frozenset[str] = frozenset(
    {
        "target_reached",
        "global_converged",
        "time_exhausted",
        "max_ticks",
        "sweep_done",
        # The model asking to close early. A run whose infrastructure actually
        # failed carries baseline_failed, crash_threshold_exceeded, policy_loop
        # or signal instead, so this value marks a normal closeout; the close
        # collector keeps the escalation flag either way.
        "robustness_escalated",
    }
)

#: Terminals where something outside the optimization ended the run.
ABORTED_STOP_REASONS: frozenset[str] = frozenset({"signal", "user_stop_requested"})

#: Terminals about the machine or the harness rather than the model: a host
#: that cannot run the combo, an argv the installed parser refuses, a bring-up
#: round that expired unreaped, or a supervisor that found the coordinator dead
#: or its tick wedged.
INFRASTRUCTURE_STOP_REASONS: frozenset[str] = frozenset({ENV_FAULT, ARGV_INVALID, DIED_STOP_REASON, WEDGED_STOP_REASON})

#: Terminals the model gate reached before the loop started. Verdicts about the
#: model, so absent from :data:`INFRASTRUCTURE_STOP_REASONS`.
MODEL_GATE_STOP_REASONS: frozenset[str] = frozenset(
    {
        "model_context_window_too_small",
        "model_config_incompatible",
        "unsupported_model_arch",
    }
)


def outcome_status(stop_reason: str) -> str:
    """Map a terminal ``stop_reason`` onto the outcome vocabulary.

    Args:
        stop_reason: The session's terminal stop reason; empty while it runs.

    Returns:
        str: ``completed`` when the run closed normally, ``aborted`` when
        something other than a verdict ended it -- including a fault in the
        host -- and ``failed`` otherwise.
    """
    if stop_reason in SUCCESS_STOP_REASONS:
        return "completed"
    if not stop_reason or stop_reason in ABORTED_STOP_REASONS or stop_reason in INFRASTRUCTURE_STOP_REASONS:
        return "aborted"
    return "failed"


__all__ = [
    "ABORTED_STOP_REASONS",
    "INFRASTRUCTURE_STOP_REASONS",
    "MODEL_GATE_STOP_REASONS",
    "SUCCESS_STOP_REASONS",
    "outcome_status",
]
