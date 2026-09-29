# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hold a lane-deferred kernel ``integrate`` for the coordinator to retry, and bound the wait.

The KERNEL pipeline task holds the benchmark lanes for as long as it runs, so an ``integrate``
for a KEEP it produced is answered ``deferred: lanes_busy`` until it finishes. The request used
to be dropped with that answer and the orchestrator was told to re-send it every turn; one
session re-sent the same one 338 times over 3.8 hours. Parked here instead, it is re-dispatched
by the coordinator on the first tick its lanes are free. A KEEP still deferred after
``max_attempts`` deferrals or ``max_minutes`` stops holding KERNEL open: its pending record is
marked, and it stays queued for the drain at SWEEP entry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

RETRY_ENV = "HYPERLOOM_INTEGRATE_RETRY"
MAX_ATTEMPTS_ENV = "HYPERLOOM_INTEGRATE_DEFER_MAX_ATTEMPTS"
MAX_MINUTES_ENV = "HYPERLOOM_INTEGRATE_DEFER_MAX_MIN"
DEFAULT_MAX_ATTEMPTS = 30
DEFAULT_MAX_MINUTES = 60.0


def _env_number(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name) or default))
    except ValueError:
        return default


def integrate_kernel_id(payload: dict[str, Any]) -> str:
    """The kernel an ``integrate`` request names, from its params or its top level."""
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    return str(params.get("kernel_id") or payload.get("kernel_id") or "").strip()


@dataclass
class ParkedIntegrate:
    """One deferred ``integrate`` waiting for its lanes."""

    source: str
    payload: dict[str, Any]
    kernel_id: str
    lanes: list[str]
    first_deferred: float
    attempts: int = 1
    dispatches: int = field(default=0, compare=False)


class DeferredIntegrates:
    """The parked ``integrate`` requests, one per kernel id."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_attempts: int | None = None,
        max_minutes: float | None = None,
    ) -> None:
        if enabled is None:
            enabled = (os.environ.get(RETRY_ENV) or "1").strip().lower() not in ("0", "false", "no", "off")
        self.enabled = enabled
        self.max_attempts = int(
            _env_number(MAX_ATTEMPTS_ENV, DEFAULT_MAX_ATTEMPTS) if max_attempts is None else max_attempts
        )
        self.max_minutes = _env_number(MAX_MINUTES_ENV, DEFAULT_MAX_MINUTES) if max_minutes is None else max_minutes
        self._parked: dict[str, ParkedIntegrate] = {}

    def park(self, source: str, payload: dict[str, Any], lanes: list[str], now: float) -> ParkedIntegrate | None:
        """Record one more deferral; the first starts the wait clock, later ones only count."""
        kernel_id = integrate_kernel_id(payload)
        if not self.enabled or not kernel_id:
            return None
        entry = self._parked.get(kernel_id)
        if entry is None:
            entry = ParkedIntegrate(source, dict(payload), kernel_id, list(lanes), now)
            self._parked[kernel_id] = entry
        else:
            entry.attempts += 1
            entry.source, entry.payload, entry.lanes = source, dict(payload), list(lanes)
        return entry

    def expiry_reason(self, entry: ParkedIntegrate, now: float) -> str:
        """Why the wait on ``entry`` is over, or ``""`` while it may continue."""
        if self.max_attempts and entry.attempts >= self.max_attempts:
            return f"deferred_{entry.attempts}_times"
        waited_min = (now - entry.first_deferred) / 60.0
        if self.max_minutes and waited_min >= self.max_minutes:
            return f"deferred_for_{int(waited_min)}_min"
        return ""

    def entries(self) -> list[ParkedIntegrate]:
        return list(self._parked.values())

    def get(self, kernel_id: str) -> ParkedIntegrate | None:
        return self._parked.get(kernel_id)

    def discard(self, kernel_id: str) -> None:
        self._parked.pop(kernel_id, None)

    def __len__(self) -> int:
        return len(self._parked)


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_MINUTES",
    "MAX_ATTEMPTS_ENV",
    "MAX_MINUTES_ENV",
    "RETRY_ENV",
    "DeferredIntegrates",
    "ParkedIntegrate",
    "integrate_kernel_id",
]
