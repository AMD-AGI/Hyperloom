# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a terminal ``stop_reason`` says about how a session ended."""

from __future__ import annotations

from hyperloom.orchestrator.bringup.argv_preflight import ARGV_INVALID
from hyperloom.orchestrator.bringup.env_preflight import ENV_FAULT

# Historical reports retain these terminal reasons after the producer is retired.
DIED_STOP_REASON = "supervisor_coordinator_died"
WEDGED_STOP_REASON = "supervisor_tick_stalled"
SUPERVISOR_RESTART_REASON = "supervisor_restart_requested"

#: Terminals that mean the run optimized and closed normally.
SUCCESS_STOP_REASONS: frozenset[str] = frozenset(
    {
        "target_reached",
        "global_converged",
        "time_exhausted",
        "max_ticks",
        "sweep_done",
        # The model asking to close early. A run whose infrastructure actually
        # failed carries baseline_failed or signal instead, so this value marks
        # a normal closeout; the close collector keeps the escalation flag
        # either way.
        "robustness_escalated",
    }
)

#: Terminals where something outside the optimization ended the run.
ABORTED_STOP_REASONS: frozenset[str] = frozenset({"signal"})

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


def outcome_status(stop_reason: str, baseline_tput: float) -> str:
    """Map a terminal ``stop_reason`` onto the outcome vocabulary.

    Args:
        stop_reason: The session's terminal stop reason; empty while it runs.
        baseline_tput: The session's baseline throughput. A run that closed on
            a success-shaped reason without one measured nothing, so it reads
            as failed rather than completed.

    Returns:
        str: ``completed`` when the run closed normally on a measured
        baseline, ``aborted`` when something other than a verdict ended it --
        including a fault in the host -- and ``failed`` otherwise.
    """
    if stop_reason in SUCCESS_STOP_REASONS:
        return "completed" if baseline_tput > 0 else "failed"
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
