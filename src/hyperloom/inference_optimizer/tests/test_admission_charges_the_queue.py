# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Admitting a round prices it against the queue's committed minutes, not the bare phase clock."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.policy.gate import PolicyDenied


def _dispatcher(*, queued_kinds: tuple[str, ...], phase_minutes_left: float) -> DispatcherCollaborator:
    """A dispatcher whose only live parts are the queue and the time gate."""
    seen: dict[str, float] = {}

    def time_gate(action: str, *, params=None, already_committed_minutes: float = 0.0) -> PolicyDenied | None:
        seen["charged"] = already_committed_minutes
        if 12.0 + already_committed_minutes > phase_minutes_left:
            return PolicyDenied(f"{action} does not fit", rule="phase_time_budget")
        return None

    dispatcher = DispatcherCollaborator.__new__(DispatcherCollaborator)
    # The phase is staying; a phase on its way out refuses before any of this (test_admission_during_phase_exit).
    dispatcher.admission_frozen = False
    # ``tasks`` is derived from the coordinator, so the fake queue lives there.
    dispatcher._coord = SimpleNamespace(tasks=SimpleNamespace(queued=lambda: _queued(queued_kinds)))
    dispatcher._phase_denial_for_action = lambda _a: None
    dispatcher._sequence_denial_for_action = lambda _a: None
    dispatcher.time_budget_denial_for_action = time_gate
    dispatcher.expected_cost_minutes_for = lambda _kind: 12.0
    # No lanes on record: every queued task is treated as contending.
    dispatcher._coord.action_registry = {}
    dispatcher._seen = seen
    return dispatcher


async def _queued(kinds: tuple[str, ...]):
    return [SimpleNamespace(kind=k) for k in kinds]


def test_an_empty_queue_charges_nothing():
    d = _dispatcher(queued_kinds=(), phase_minutes_left=30.0)

    assert asyncio.run(d.admission_denial_for_action("explore")) is None
    assert d._seen["charged"] == 0.0


def test_a_round_that_fits_alone_is_refused_once_the_queue_is_charged():
    """Two observed phases admitted thirteen and fourteen explore rounds and measured two and one.

    Each fitted the phase clock on its own; together they never could.
    """
    d = _dispatcher(queued_kinds=("explore", "explore"), phase_minutes_left=30.0)

    denied = asyncio.run(d.admission_denial_for_action("explore"))

    assert denied is not None
    assert denied.rule == "phase_time_budget"
    assert d._seen["charged"] == 24.0


def test_a_phase_with_room_for_the_queue_and_one_more_still_admits():
    d = _dispatcher(queued_kinds=("explore",), phase_minutes_left=40.0)

    assert asyncio.run(d.admission_denial_for_action("explore")) is None


def test_an_earlier_gate_still_wins():
    """Phase and sequence gates are about what may run at all, not what it costs."""
    d = _dispatcher(queued_kinds=(), phase_minutes_left=999.0)
    d._phase_denial_for_action = lambda _a: PolicyDenied("wrong phase", rule="phase_action")

    denied = asyncio.run(d.admission_denial_for_action("explore"))

    assert denied is not None and denied.rule == "phase_action"


def _with_lanes(d, registry: dict) -> None:
    """Give the dispatcher a lane map; the registry is read through the coordinator."""
    d._coord.action_registry = registry


def test_a_queue_on_another_lane_does_not_delay_this_round():
    """A specialist holds research_lane; an explore round needs server_lifecycle and benchmark_lane.

    Summed regardless of lane, two queued specialists refuse an explore round
    the phase can in fact run alongside them.
    """
    d = _dispatcher(queued_kinds=("specialist", "specialist"), phase_minutes_left=20.0)
    _with_lanes(
        d,
        {
            "explore": SimpleNamespace(requires_lanes=("server_lifecycle", "benchmark_lane")),
            "specialist": SimpleNamespace(requires_lanes=("research_lane",)),
        },
    )

    assert asyncio.run(d.admission_denial_for_action("explore")) is None
    assert d._seen["charged"] == 0.0


def test_a_queue_on_the_same_lane_still_counts():
    d = _dispatcher(queued_kinds=("explore", "explore"), phase_minutes_left=20.0)
    _with_lanes(d, {"explore": SimpleNamespace(requires_lanes=("server_lifecycle", "benchmark_lane"))})

    assert asyncio.run(d.admission_denial_for_action("explore")) is not None
    assert d._seen["charged"] == 24.0
