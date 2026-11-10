# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which Agent profile each slot of a tree round runs on, and the search actor that runs it.

Under a split relay the slot positions of a round take the profiles that can run in turn, and each round starts one
profile further along, so over the campaign every profile plans and implements an equal share of the expansions. The
positions are those of the selector's answer, which is told the order before it answers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kernelforge.workflows.forge_loop.tree.model import Slot


def round_profile_cycle(profile_ids: Sequence[str], iteration: int) -> tuple[str, ...]:
    """The profiles a round's slot positions take in turn, starting one further along each iteration."""
    if not profile_ids:
        return ()
    offset = (iteration - 1) % len(profile_ids)
    return (*profile_ids[offset:], *profile_ids[:offset])


def profile_at(cycle: Sequence[str], position: int) -> str:
    """The profile of the slot at 0-based ``position`` of the selector's answer; empty when the round has none."""
    return cycle[position % len(cycle)] if cycle else ""


@dataclass(frozen=True)
class SlotActors:
    """The search actor each slot of a round runs on: its own profile's when this round has it, else the round's."""

    default: Any
    by_profile: Mapping[str, Any] = field(default_factory=dict)

    def of(self, slot: Slot) -> Any:
        return self.by_profile.get(slot.agent_profile_id, self.default)


__all__ = ["SlotActors", "profile_at", "round_profile_cycle"]
