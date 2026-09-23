# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep a Critic-approved ``integrate_patch`` alive across phase boundaries."""

from __future__ import annotations

from typing import Any

from ..policy.gate import INTEGRATE_PATCH_PERMISSIVE_VERDICTS, patch_verdict_subject
from .machine_state import PHASE_CLOSE


def spare_approved_integrate_patch_on_phase_transition(
    *,
    target_phase: str,
    kind: str,
    params: dict[str, Any],
    shared_state: Any,
) -> bool:
    """Return True to leave a queued, Critic-approved ``integrate_patch`` alive.

    ``integrate_patch`` is allowed in FRAMEWORK_AGENT and not in KERNEL_AGENT, so
    the transition between them used to bulk-cancel every one still queued. An
    approved patch waits behind whatever holds the benchmark lane, and on
    2026-09-22 that wait was an explore grid: the Olmo-3-7B hybrid-SWA patch was
    approved, queued, and cancelled 57 minutes later without ever running. The
    authoring and the review had already been paid for; the cancel threw both
    away.

    Deny-list, as for the GEAK rebench: only ``CLOSE`` kills it, because a phase
    set that grows would otherwise quietly reintroduce the cancel for any phase
    missing from an allow-list. Surviving the boundary is enough on its own --
    ``PolicyGate.validate_dispatched_task`` skips phase compatibility precisely
    so queued work is not refused after a transition, and it still replays the
    Critic gate.

    Only a task with a permissive verdict on record is spared. One without is
    either unreviewed or refused, and keeping it alive would only have it
    rejected at dispatch instead.

    Args:
        target_phase: The phase being entered.
        kind: The queued task's kind.
        params: The queued task's params.
        shared_state: The ``SharedState`` holding the verdict ledger.

    Returns:
        bool: Whether to keep the task queued.
    """
    if (target_phase or "").strip().upper() == PHASE_CLOSE:
        return False
    if (kind or "").strip() != "integrate_patch":
        return False
    subject = patch_verdict_subject(params or {})
    if not subject:
        return False
    verdict = shared_state.get_specialist_patch_verdict(subject)
    return str(verdict or "").strip().lower() in INTEGRATE_PATCH_PERMISSIVE_VERDICTS
