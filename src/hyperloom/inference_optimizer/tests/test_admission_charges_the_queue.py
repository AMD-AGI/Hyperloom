# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Admitting a round prices it against the queue's committed minutes, not the bare phase clock."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.policy.gate import PolicyDenied

# A grid to measure, so the explore-grid gate admission ends on stays out of these.
_ONE_VARIANT = {"grid": [{"name": "v", "extra_server_args": "--x"}]}
_BENCH = ("server_lifecycle", "benchmark_lane")


def _dispatcher(*, queued_kinds: tuple[str, ...], phase_minutes_left: float, lanes: dict) -> DispatcherCollaborator:
    """A dispatcher whose only live parts are the queue and the time gate; every catalogued round costs 12 minutes."""
    seen: dict[str, float] = {}

    def time_gate(action: str, *, params=None, already_committed_minutes: float = 0.0) -> PolicyDenied | None:
        seen["charged"] = already_committed_minutes
        if 12.0 + already_committed_minutes > phase_minutes_left:
            return PolicyDenied(f"{action} does not fit", rule="phase_time_budget")
        return None

    async def queued():
        return [SimpleNamespace(kind=k) for k in queued_kinds]

    dispatcher = DispatcherCollaborator.__new__(DispatcherCollaborator)
    dispatcher.admission_frozen = False
    dispatcher._coord = SimpleNamespace(
        tasks=SimpleNamespace(queued=queued),
        shared_state=SimpleNamespace(find_gap=lambda _cid: None, baseline_runtime_sec=0.0),
        action_registry={
            kind: SimpleNamespace(typical_runtime_min=12.0, requires_lanes=held) for kind, held in lanes.items()
        },
    )
    dispatcher._phase_denial_for_action = lambda _a: None
    dispatcher._sequence_denial_for_action = lambda _a: None
    dispatcher.time_budget_denial_for_action = time_gate
    dispatcher._seen = seen
    return dispatcher


@pytest.mark.parametrize(
    ("queued_kinds", "minutes_left", "lanes", "denied", "charged"),
    [
        pytest.param((), 30.0, {"explore": ()}, False, 0.0, id="empty-queue"),
        pytest.param(("explore", "explore"), 30.0, {"explore": ()}, True, 24.0, id="fits-alone-not-after-queue"),
        pytest.param(("explore",), 40.0, {"explore": ()}, False, 12.0, id="room-for-queue-and-one"),
        pytest.param(
            ("specialist", "specialist"),
            20.0,
            {"explore": _BENCH, "specialist": ("research_lane",)},
            False,
            0.0,
            id="other-lane-not-charged",
        ),
        pytest.param(("explore", "explore"), 20.0, {"explore": _BENCH}, True, 24.0, id="same-lane-charged"),
    ],
)
def test_the_queue_on_the_rounds_lanes_is_charged(queued_kinds, minutes_left, lanes, denied, charged):
    d = _dispatcher(queued_kinds=queued_kinds, phase_minutes_left=minutes_left, lanes=lanes)

    result = asyncio.run(d.admission_denial_for_action("explore", params=_ONE_VARIANT))

    assert (result is not None) is denied
    assert d._seen["charged"] == charged


def test_an_earlier_gate_still_wins():
    d = _dispatcher(queued_kinds=(), phase_minutes_left=999.0, lanes={})
    d._phase_denial_for_action = lambda _a: PolicyDenied("wrong phase", rule="phase_action")

    denied = asyncio.run(d.admission_denial_for_action("explore", params=_ONE_VARIANT))

    assert denied is not None and denied.rule == "phase_action"
