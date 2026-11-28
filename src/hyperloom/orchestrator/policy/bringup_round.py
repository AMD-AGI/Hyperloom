# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline waits while a bring-up round another task holds is open."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .denial import PolicyDenied

if TYPE_CHECKING:
    from ..state.round_store import RoundStore

__all__ = ["ENABLEMENT_ROUND_IN_FLIGHT", "bringup_round_denial"]

ENABLEMENT_ROUND_IN_FLIGHT = "enablement_round_in_flight"

_BASELINE_ACTION_NAME = "baseline"


async def bringup_round_denial(action_name: str, rounds: RoundStore, *, task_id: str = "") -> PolicyDenied | None:
    """Refuse a baseline while a bring-up round holds the machine, as the round store stands at the call.

    A second bring-up fights the first for the same cards and ports, and a
    specialist rewriting the framework underneath the round leaves the anchor
    describing neither the old stack nor the new one. No lane carries this:
    between the round's legs every serving lane is free. The round's own
    holder is admitted: it already won the acquire ``RoundStore.open`` decides.

    Args:
        action_name: The delegated action, or the kind of the queued task about to run.
        rounds: The session's round store.
        task_id: The queued task's id; empty for a delegate, which has no task yet.

    Returns:
        A :class:`PolicyDenied` when a round is open and this attempt is not its holder, else ``None``.
    """
    if str(action_name or "").strip() != _BASELINE_ACTION_NAME:
        return None
    held = await rounds.held()
    if held is None or (task_id and task_id == held.holder_task_id):
        return None
    return PolicyDenied(
        (
            f"baseline: bring-up round {held.round_id!r} "
            f"(holder={held.holder_task_id!r}) holds the machine while its lease is live"
        ),
        rule=ENABLEMENT_ROUND_IN_FLIGHT,
        hint="Let the round settle; a second bring-up would fight it for the same cards and ports.",
    )
