# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DelegatedTask state machine constants, free of any storage dependency.

Held apart from :mod:`task_registry` because that module imports
``bus.resource_lock``, so anything in ``bus`` that needs to ask whether a task
has ended would close an import cycle. The states are a vocabulary, not
behaviour, and nothing here touches a database.
"""

from __future__ import annotations

#: Successor states each task state may move to; a state with none is terminal.
TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "failed": frozenset(),
    "succeeded": frozenset(),
    "cancelled": frozenset(),
}

#: States a task cannot leave. A task in one of these will never resume, which
#: is what lets a holder's lane row be judged abandoned rather than merely late.
TERMINAL_STATES = frozenset(state for state, outgoing in TRANSITIONS.items() if not outgoing)
