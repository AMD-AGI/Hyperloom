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


# A grid to measure, so the explore-grid gate admission ends on stays out of these.
_ONE_VARIANT = {"grid": [{"name": "v", "extra_server_args": "--x"}]}


def _dispatcher(*, frozen: bool) -> DispatcherCollaborator:
    """A dispatcher whose only live parts are the freeze flag and the gates below it."""
    dispatcher = DispatcherCollaborator.__new__(DispatcherCollaborator)
    dispatcher.admission_frozen = frozen
    dispatcher._coord = SimpleNamespace(
        tasks=SimpleNamespace(queued=_no_queued),
        shared_state=SimpleNamespace(find_gap=lambda _cid: None),
        action_registry={},
    )
    dispatcher._phase_denial_for_action = lambda _a: None
    dispatcher._sequence_denial_for_action = lambda _a: None
    dispatcher.time_budget_denial_for_action = lambda _a, **_kw: None
    return dispatcher


def test_a_phase_waiting_to_leave_refuses_new_rounds_until_the_flag_lifts():
    dispatcher = _dispatcher(frozen=True)
    # The freeze answers before the clock is read.
    dispatcher.time_budget_denial_for_action = lambda _a, **_kw: PolicyDenied("too slow", rule="phase_time_budget")

    denied = asyncio.run(dispatcher.admission_denial_for_action("explore", params=_ONE_VARIANT))

    assert denied is not None and denied.rule == "phase_transition_pending"
    assert "propose again" in str(denied.hint)

    dispatcher.admission_frozen = False
    dispatcher.time_budget_denial_for_action = lambda _a, **_kw: None

    assert asyncio.run(dispatcher.admission_denial_for_action("explore", params=_ONE_VARIANT)) is None
