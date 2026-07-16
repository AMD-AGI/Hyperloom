# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for per-phase handler collaborators.

Each phase's coordinator methods live in a ``PhaseHandler`` subclass that holds
a back-reference to its owning ``Coordinator`` and delegates every attribute it
does not define itself back to that coordinator via ``__getattr__``. Method
bodies keep using ``self.shared_state`` / ``self.bus`` / sibling ``self._foo``
calls, which resolve onto the coordinator. State rebinds land on the coordinator
via ``self._coord.<attr> = ...``.
"""

from __future__ import annotations

from typing import Any


class PhaseHandler:
    """A coordinator collaborator that owns one phase's methods."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_coord"), name)
