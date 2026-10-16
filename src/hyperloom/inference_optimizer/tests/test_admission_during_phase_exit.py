# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A phase that has decided to leave admits nothing new, instead of enqueueing what it will cancel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.policy.gate import PolicyDenied


async def _no_queued():
    return []


def _dispatcher(*, frozen: bool) -> DispatcherCollaborator:
    """A dispatcher whose only live parts are the freeze flag and the gates below it."""
    dispatcher = DispatcherCollaborator.__new__(DispatcherCollaborator)
    dispatcher.admission_frozen = frozen
    dispatcher._coord = SimpleNamespace(tasks=SimpleNamespace(queued=_no_queued), action_registry={})
    dispatcher._phase_denial_for_action = lambda _a: None
    dispatcher._sequence_denial_for_action = lambda _a: None
    dispatcher.time_budget_denial_for_action = lambda _a, **_kw: None
    dispatcher.expected_cost_minutes_for = lambda _kind, _params=None: 12.0
    return dispatcher


def test_a_phase_waiting_to_leave_refuses_the_round_it_would_have_cancelled():
    """Seventeen minutes of a FRAMEWORK_AGENT phase went this way.

    The transition to KERNEL_AGENT was decided but held on one running
    explore, so ``_drain_for_transition`` re-cleared the queue every tick
    while admission kept saying yes. Eleven rounds were proposed, reviewed,
    admitted and cancelled within half a second each.
    """
    denied = asyncio.run(_dispatcher(frozen=True).admission_denial_for_action("explore"))

    assert denied is not None
    assert denied.rule == "phase_transition_pending"
    assert "leaving" in str(denied)
    # The agent is told to wait rather than re-propose; a bare refusal reads as "never".
    assert "propose again" in str(denied.hint)


def test_the_freeze_outranks_the_gates_below_it():
    """A frozen phase is not a budget question, so it answers before the clock is read."""
    dispatcher = _dispatcher(frozen=True)
    dispatcher.time_budget_denial_for_action = lambda _a, **_kw: PolicyDenied("too slow", rule="phase_time_budget")

    denied = asyncio.run(dispatcher.admission_denial_for_action("explore"))

    assert denied is not None and denied.rule == "phase_transition_pending"


def test_a_phase_that_is_staying_admits_as_before():
    assert asyncio.run(_dispatcher(frozen=False).admission_denial_for_action("explore")) is None


def test_the_freeze_lifts_with_the_flag():
    """The phase machine clears the flag the moment the transition commits, and admission follows."""
    dispatcher = _dispatcher(frozen=True)

    assert asyncio.run(dispatcher.admission_denial_for_action("explore")) is not None

    dispatcher.admission_frozen = False

    assert asyncio.run(dispatcher.admission_denial_for_action("explore")) is None
