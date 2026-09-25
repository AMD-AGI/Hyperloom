# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a terminal ``stop_reason`` says about how a session ended."""

from __future__ import annotations

#: The argv the installed CLI parser refuses (a bring-up bad-args terminal).
ARGV_INVALID = "server_argv_invalid"

#: The host cannot run the combo (a bring-up environment-fault terminal).
ENV_FAULT = "environment_fault"

# Historical reports retain these terminal reasons after the producer is retired.
DIED_STOP_REASON = "supervisor_coordinator_died"
WEDGED_STOP_REASON = "supervisor_tick_stalled"
SUPERVISOR_RESTART_REASON = "supervisor_restart_requested"

#: AgentX is on but its benchmark client (aiperf) is missing or is not the
#: pinned build, and the runtime install could not supply it. An
#: environment/supply gap, not a code gap: nothing downstream can author its
#: way out of it, so the run halts on the FIRST occurrence instead of
#: spending the budget in the enablement lane.
AGENTX_PREFLIGHT_STOP_REASON: str = "agentx_client_unavailable"

#: A patch lifecycle owed the framework tree a revert and could not complete it,
#: so the tree still holds patches the session never measured against. Two
#: independent recoveries -- the integrate sentinel and the kernel stack
#: checkpoint -- halt on this, and both refuse to continue rather than measure a
#: tree whose contents they cannot account for.
PATCH_RECOVERY_INCOMPLETE_STOP_REASON: str = "patch_recovery_incomplete"

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

#: The closed vocabulary a ``stop_reason`` field is validated against
#: (:func:`is_valid_stop_reason`, :meth:`SharedState.set_stop_reason`).
#: PolicyGate rejects any stop_reason outside it, so a typo on either side
#: would silently degrade into "the run did not stop" rather than into an
#: error anyone sees.
STOP_REASON_VOCAB: frozenset[str] = frozenset(
    {
        "target_reached",
        "time_exhausted",
        "max_ticks",
        "baseline_failed",
        "emergency",
        "coordinator_exception",
        "signal",
        "unknown",
        "custom",
        "robustness_escalated",
        "prelude_baseline_failed",
        "prelude_cold_anchor_low_budget",
        "time_exhausted_during_prelude",
        "warm_replay_rollback_failed",
        "active_inferencex_checkout_missing",
        "no_kernel_skipped",
        "sweep_done",
        "sweep_failed",
        "framework_agent_phase_done",
        # R7: cyclic phase machine exhausted leverage across macro-cycles.
        "global_converged",
        # Context-window preflight: max_position_embeddings can't hold ISL+OSL.
        "model_context_window_too_small",
        # Model-arch preflight: multimodal/vision model unsupported.
        "unsupported_model_arch",
        # Pre-run model-config compatibility preflight: config.json is corrupt or declares RoPE scaling without a
        # max-position field (both crash at load).
        "model_config_incompatible",
        # Baseline arg-validation fast-exit: >=2 consecutive baseline attempts exited <30s on a bad CLI arg.
        "baseline_arg_error",
        # Enablement attempt cap: too many consecutive rounds bought no ground.
        # A bring-up that is still advancing is bounded by the run's wall clock.
        "enablement_attempts_exhausted",
        # The baseline could not produce an accuracy result even though the
        # accuracy test was expected to run (broken eval / missing quality
        # gate). Optimizing against an unvalidated baseline is unsafe, so the
        # run halts. Post-baseline accuracy failures REVERT the offending
        # change instead of stopping.
        "baseline_accuracy_failed",
        # Bring-up terminals: the host cannot run the combo, or the harness
        # composed an argument the installed parser does not have. Classified as
        # infrastructure by ``INFRASTRUCTURE_STOP_REASONS``.
        ENV_FAULT,
        ARGV_INVALID,
        # A bring-up round expired with nothing confirming its holder dead, so
        # it keeps excluding the machine.
        # The out-of-band supervisor found the coordinator's process gone; it
        # reaches a report through the terminal artifact the supervisor writes.
        DIED_STOP_REASON,
        # The out-of-band supervisor found the tick not advancing and the
        # coordinator did not answer the stop it was sent; it reaches a report
        # through the terminal artifact the supervisor writes.
        WEDGED_STOP_REASON,
        AGENTX_PREFLIGHT_STOP_REASON,
        # A restore obligation outlived the attempt that owed it: the framework
        # tree still carries patches, so every later measurement would be
        # attributed to a baseline that is not on disk. Deliberately absent from
        # INFRASTRUCTURE_STOP_REASONS -- the host is healthy and a person has to
        # settle the tree, which reads as a failure rather than an abort.
        PATCH_RECOVERY_INCOMPLETE_STOP_REASON,
    }
)


def is_valid_stop_reason(value: str) -> bool:
    """Return True when ``value`` is a member of :data:`STOP_REASON_VOCAB`."""
    return (value or "").strip() in STOP_REASON_VOCAB


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
    "AGENTX_PREFLIGHT_STOP_REASON",
    "ARGV_INVALID",
    "DIED_STOP_REASON",
    "ENV_FAULT",
    "INFRASTRUCTURE_STOP_REASONS",
    "MODEL_GATE_STOP_REASONS",
    "PATCH_RECOVERY_INCOMPLETE_STOP_REASON",
    "STOP_REASON_VOCAB",
    "SUCCESS_STOP_REASONS",
    "SUPERVISOR_RESTART_REASON",
    "WEDGED_STOP_REASON",
    "is_valid_stop_reason",
    "outcome_status",
]
