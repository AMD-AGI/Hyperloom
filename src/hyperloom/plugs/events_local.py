# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local implementation of the ``events`` socket: a debug log line per event, nothing else."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from hyperloom.plugs.socket import PlugContext

log = logging.getLogger("hyperloom.events")


class LocalProgressEvents:
    """Writes each event to the debug log, so a local run behaves exactly as it did before the socket existed."""

    def emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        """Log one event."""
        log.debug("event %s %s", event_type, dict(data))

    def flush(self) -> None:
        """Nothing is buffered."""


def create(context: PlugContext) -> LocalProgressEvents:
    """Factory the built-in registration points at."""
    return LocalProgressEvents()
