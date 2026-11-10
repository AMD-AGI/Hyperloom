# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``events`` socket: progress events a run emits for whoever is watching it."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ProgressEvents(Protocol):
    """Receives the run's progress events (``hyperloom.*`` types).

    Not the breakdown recorder's ``EventSink``, which records a session's own history.
    """

    def emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        """Hand over one event; never raises into the run."""

    def flush(self) -> None:
        """Make everything emitted so far durable."""
