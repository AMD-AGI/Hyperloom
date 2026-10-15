# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration prompt says which of the phase's actions its remaining budget can still pay for."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hyperloom.orchestrator.loop.conversation import ConversationCollaborator
from hyperloom.orchestrator.policy.gate import PolicyDenied


def _collaborator(*, phase: str, unaffordable: set[str], queued: tuple = ()) -> ConversationCollaborator:
    """A collaborator whose dispatcher refuses exactly ``unaffordable``."""

    def denial(action: str, *, already_committed_minutes: float = 0.0, **_kw: object) -> PolicyDenied | None:
        if action in unaffordable or already_committed_minutes >= 30.0:
            return PolicyDenied(f"{action} denied", rule="phase_time_budget")
        return None

    class _Tasks:
        async def queued(self):
            return list(queued)

    collaborator = ConversationCollaborator.__new__(ConversationCollaborator)
    # ``shared_state`` is derived from the coordinator, so the fake lives there.
    collaborator._coord = SimpleNamespace(
        shared_state=SimpleNamespace(phase=phase),
        dispatcher=SimpleNamespace(
            time_budget_denial_for_action=denial,
            tasks=_Tasks(),
            expected_cost_minutes_for=lambda kind: 12.0,
        ),
    )
    return collaborator


def test_an_action_the_phase_can_still_fund_is_listed_plainly():
    line = asyncio.run(_collaborator(phase="FRAMEWORK_AGENT", unaffordable=set())._affordable_actions_line())

    assert line.startswith("=== Affordable now ===")
    assert "explore" in line
    assert "unaffordable" not in line


def test_an_action_priced_above_what_is_left_is_marked():
    """Twelve explore rounds were proposed into a phase that could fund two; ten were dropped at dispatch."""
    line = asyncio.run(_collaborator(phase="FRAMEWORK_AGENT", unaffordable={"explore"})._affordable_actions_line())

    assert "explore(unaffordable)" in line


def test_a_phase_with_no_proposable_action_renders_nothing():
    assert asyncio.run(_collaborator(phase="NOT_A_PHASE", unaffordable=set())._affordable_actions_line()) == ""


def test_a_run_without_a_dispatcher_renders_nothing():
    """Composed before the dispatcher exists, the prompt simply omits the line rather than failing."""
    collaborator = ConversationCollaborator.__new__(ConversationCollaborator)
    collaborator._coord = SimpleNamespace(shared_state=SimpleNamespace(phase="FRAMEWORK_AGENT"), dispatcher=None)

    assert asyncio.run(collaborator._affordable_actions_line()) == ""


def test_a_queue_that_has_already_spoken_for_the_phase_marks_the_action():
    """The case the admission gate alone cannot see.

    A FRAMEWORK_AGENT phase proposed thirteen explore rounds and measured four:
    by the fifth, the queue held more minutes than the phase had left, yet each
    round priced on its own still fitted. Nine waited and were dropped in one
    batch when the budget ran out.
    """
    line = asyncio.run(
        _collaborator(
            phase="FRAMEWORK_AGENT",
            unaffordable=set(),
            queued=("explore", "explore", "explore"),  # 3 x 12 min = 36 min already spoken for
        )._affordable_actions_line()
    )

    assert "explore(unaffordable)" in line
    assert "queue already holds ~36 min" in line


def test_an_empty_queue_does_not_mention_committed_work():
    line = asyncio.run(_collaborator(phase="FRAMEWORK_AGENT", unaffordable=set())._affordable_actions_line())

    assert "queue already holds" not in line
