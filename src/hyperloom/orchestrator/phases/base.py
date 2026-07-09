# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Base class for per-phase handler collaborators.

Each phase's coordinator methods are extracted into a ``PhaseHandler`` subclass
that holds a back-reference to its owning ``Coordinator`` and delegates every
attribute it does not define itself back to that coordinator via ``__getattr__``
(the ``IntentRouter``/``ResultRecorder`` pattern). Method bodies keep using
``self.shared_state`` / ``self.bus`` / sibling ``self._foo`` calls, which resolve
onto the coordinator (and, for methods now owned by other collaborators, onward
via the coordinator's own ``__getattr__`` delegation).

State rebinds (``self.<attr> = ...``) in the extracted bodies are rewritten to
``self._coord.<attr> = ...`` so they land on the coordinator, not the (stateless)
handler.
"""

from __future__ import annotations

from typing import Any


class PhaseHandler:
    """A coordinator collaborator that owns one phase's methods."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_coord"), name)
