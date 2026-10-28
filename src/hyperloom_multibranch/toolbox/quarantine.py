# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The node quarantine list, kept in the notebook's ``NodeHealth`` and read back when the toolbox starts (design 4.5).

A node enters the list when an attempt on it fails with an infrastructure failure the executor classified, or when a
second unclear failure that did not reproduce on its rerun happens on it within the session (the toolbox's lifetime;
the count is not persisted). A single non-reproducing unclear failure does not quarantine a node. Quarantine only widens
``avoid_nodes``; it never blocks a test: on a single node, avoiding it still creates the sandbox there and the result is
marked same-node (design 4.9.3), so a toolbox with one node counts the failures of its node (:meth:`count_failure`) and
never quarantines it.

A failure that says nothing about its node (``attempts.NODE_BLAMELESS_INFRA_REASONS``) never quarantines it, and a node
quarantine the notebook holds from such a failure (one recorded before that rule) is not read back: :meth:`load` leaves
it out, so the node is in service again without an operator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from hyperloom_multibranch.contracts.notebook.records import QuarantineEvent
from hyperloom_multibranch.contracts.notebook.vocab import QuarantineAction
from hyperloom_multibranch.contracts.toolbox import NodeHealth
from hyperloom_multibranch.toolbox.attempts import node_blameless


class Quarantine:
    def __init__(self, health: NodeHealth) -> None:
        self._health = health
        self._nodes: set[str] = set()
        self._suspected: set[str] = set()
        self._failures: Counter[str] = Counter()

    async def load(self) -> None:
        """Read the list the notebook holds, less the node quarantines a blameless failure recorded; called once when the
        toolbox starts."""

        events = await self._health.quarantined()
        self._nodes = {event.node for event in events if not event.gpu_ids and not node_blameless(event.reason)}

    async def cards(self) -> tuple[QuarantineEvent, ...]:
        """The card-level events the notebook holds (cards lost for good), not the whole-node ones."""

        return tuple(event for event in await self._health.quarantined() if event.gpu_ids)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._nodes)

    @property
    def failures(self) -> Mapping[str, int]:
        """How many failures were reported against each node this session, quarantined or not."""

        return dict(self._failures)

    def count_failure(self, node: str) -> int:
        """Count one failure reported against ``node`` and return how many it has now."""

        self._failures[node] += 1
        return self._failures[node]

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

    async def add_suspect(self, node: str, reason: str, request_ids: Iterable[str] = ()) -> bool:
        """Count an unclear failure that did not reproduce on ``node``; its second one quarantines it."""

        if node not in self._suspected:
            self._suspected.add(node)
            return False
        return await self.add(node, reason, request_ids)

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
