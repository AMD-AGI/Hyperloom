# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for Coordinator collaborators.

A collaborator holds a back-reference to its owning ``Coordinator`` and
delegates every attribute it does not define itself back to that coordinator
via ``__getattr__``. Method bodies keep using ``self.shared_state`` /
``self.tasks`` / ``self.bus`` and sibling ``self._foo`` calls, which resolve
onto the coordinator; state rebinds land there via ``self._coord.<attr> = ...``.

Not every collaborator owns a phase — ``enablement`` is pumped on every tick
regardless of ``state.phase`` — so the phase-bound flavour is the subclass
(:class:`hyperloom.orchestrator.phases.base.PhaseHandler`), not the base.

Imports nothing first-party, so any package may depend on it.
"""

from __future__ import annotations

from typing import Any


class CoordinatorCollaborator:
    """An object that borrows the Coordinator's attributes for its own methods."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_coord"), name)
