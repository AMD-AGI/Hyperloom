# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The node quarantine list, kept in the notebook's ``NodeHealth`` and read back when the toolbox starts (design 4.5).

A node enters the list when an attempt on it fails for an infrastructure reason, or when an unclear failure did not
reproduce on the rerun. Quarantine only widens ``avoid_nodes``; it never blocks a test: on a single node, avoiding it
still creates the sandbox there and the result is marked same-node (design 4.9.3).
"""

from __future__ import annotations

from collections.abc import Iterable

from hyperloom_multibranch.contracts.notebook.records import QuarantineEvent
from hyperloom_multibranch.contracts.notebook.vocab import QuarantineAction
from hyperloom_multibranch.contracts.toolbox import NodeHealth


class Quarantine:
    def __init__(self, health: NodeHealth) -> None:
        self._health = health
        self._nodes: set[str] = set()

    async def load(self) -> None:
        """Read the list the notebook holds; called once when the toolbox starts."""

        self._nodes = {event.node for event in await self._health.quarantined() if not event.gpu_ids}

    async def cards(self) -> tuple[QuarantineEvent, ...]:
        """The card-level events the notebook holds (cards lost for good), not the whole-node ones."""

        return tuple(event for event in await self._health.quarantined() if event.gpu_ids)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._nodes)

    async def add(self, node: str, reason: str, request_ids: Iterable[str] = ()) -> bool:
        """Quarantine ``node``; ``False`` when it already was (nothing is written twice)."""

        if node in self._nodes:
            return False
        await self._health.record_quarantine(
            QuarantineEvent(
                node=node, action=QuarantineAction.QUARANTINE, reason=reason, request_ids=tuple(request_ids)
            )
        )
        self._nodes.add(node)
        return True

    async def add_cards(self, node: str, gpu_ids: Iterable[int], reason: str) -> None:
        """Record cards of ``node`` that are lost for good; the node itself stays in service (one event, not a node
        quarantine: :meth:`load` and :attr:`nodes` ignore it)."""

        await self._health.record_quarantine(
            QuarantineEvent(
                node=node, action=QuarantineAction.QUARANTINE, reason=reason, gpu_ids=tuple(sorted(gpu_ids))
            )
        )

    async def lift(self, node: str, reason: str) -> bool:
        if node not in self._nodes:
            return False
        lost = [event for event in await self.cards() if event.node == node]
        await self._health.record_quarantine(QuarantineEvent(node=node, action=QuarantineAction.LIFT, reason=reason))
        for event in lost:  # lifting a node returns its card sets in the notebook; cards lost for good are not returned
            await self._health.record_quarantine(event)
        self._nodes.discard(node)
        return True
