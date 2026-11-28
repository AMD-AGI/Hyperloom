# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one ``escalate_strategy_change`` hint the Coordinator defers, and why it refuses the rest."""

from __future__ import annotations

from ..phases.machine_state import PHASE_FRAMEWORK_AGENT
from ..policy.denial import PolicyDenied
from ..state.shared_state import ESCALATE_HINT_SKIP_TO_KERNEL


def escalate_hint_denial(hint: str, phase: str) -> PolicyDenied | None:
    """Return why ``hint`` is refused to its sender in ``phase``, or ``None`` when it is deferred.

    Args:
        hint: The non-empty ``next_action_hint``.
        phase: The session's current phase, upper-case; empty before the first phase.
    """
    if hint != ESCALATE_HINT_SKIP_TO_KERNEL:
        return PolicyDenied(
            f"escalate_strategy_change hint {hint!r} is not accepted",
            rule="escalate_hint_not_in_vocab",
            hint="skip_to_kernel, sent in FRAMEWORK_AGENT, is the one hint; the run closes itself.",
        )
    if phase != PHASE_FRAMEWORK_AGENT:
        # Only the FRAMEWORK_AGENT exit reads it, and the transition into that phase clears it.
        return PolicyDenied(
            f"next_action_hint={hint!r} advances FRAMEWORK_AGENT only; the session is in {phase or '<none>'}",
            rule="escalate_hint_phase",
            hint=f"No hint advances {phase or 'this phase'}: the Coordinator exits it on the gate its phase goal names.",
        )
    return None
