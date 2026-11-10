# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the run's code emits a progress event: one call, to whichever implementation fills the ``events`` socket.

An event with no ``events`` socket to take it is logged at debug level and the run goes on; the implementation owns
its own write failures.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping

from hyperloom.plugs.errors import PlugError
from hyperloom.plugs.host import active

log = logging.getLogger(__name__)

EVENT_PHASE_CHANGED = "hyperloom.phase_changed"
EVENT_CANDIDATE_SETTLED = "hyperloom.candidate_settled"
EVENT_BEST_UPDATED = "hyperloom.best_updated"
EVENT_BUDGET = "hyperloom.budget"
EVENT_WARNING = "hyperloom.warning"

#: At most one ``budget`` event per this many seconds.
BUDGET_INTERVAL_SEC = 300.0


class _Throttle:
    """Lets one call through per ``interval`` seconds, whichever thread asks."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self.last: float | None = None

    def due(self) -> bool:
        """Whether the caller may go ahead; if so, the interval starts now."""
        now = time.monotonic()
        with self._lock:
            if self.last is not None and now - self.last < self._interval:
                return False
            self.last = now
            return True


_budget = _Throttle(BUDGET_INTERVAL_SEC)


def emit(event_type: str, data: Mapping[str, Any]) -> None:
    """Hand one event to the run's ``events`` implementation, if one is started."""
    try:
        active().get("events").emit(event_type, data)  # type: ignore[attr-defined]
    except PlugError:
        log.debug("could not emit %s", event_type, exc_info=True)


def warn(code: str, message: str) -> None:
    """Emit ``hyperloom.warning`` for a degradation the run carried on through."""
    emit(EVENT_WARNING, {"code": code, "message": message})


def phase_changed(from_phase: str, to_phase: str, macro_cycle: int) -> None:
    """Emit ``hyperloom.phase_changed``; the phase names are the emitter's own strings."""
    emit(EVENT_PHASE_CHANGED, {"from": from_phase or None, "to": to_phase, "macro_cycle": int(macro_cycle or 0)})


def candidate_settled(task_id: str, action: str, decision: str, gain_pct: float | None = None) -> None:
    """Emit ``hyperloom.candidate_settled`` for an action that got its verdict."""
    emit(
        EVENT_CANDIDATE_SETTLED,
        {"task_id": task_id, "action": action or None, "decision": decision, "gain_pct": gain_pct},
    )


def best_updated(value: float, state: Any, request_id: str | None) -> None:
    """Emit ``hyperloom.best_updated``; ``gain_pct`` is against ``state.baseline_tput``, null without a positive one."""
    baseline = float(state.baseline_tput or 0.0)
    gain = round((value / baseline - 1.0) * 100.0, 3) if baseline > 0 else None
    emit(
        EVENT_BEST_UPDATED,
        {
            "metric": "output_throughput",
            "value": value,
            "gain_pct": gain,
            "patch_count": len(state.optimization_stack),
            "request_id": request_id or None,
        },
    )


def emit_budget(state: Any) -> bool:
    """Emit ``hyperloom.budget`` for ``state`` unless one went out within :data:`BUDGET_INTERVAL_SEC`.

    ``tokens`` and ``gpu_sec`` are null: the legacy flow does not keep a running total of either. Reports whether
    an event went out.
    """
    if not _budget.due():
        return False
    emit(
        EVENT_BUDGET,
        {
            "elapsed_min": round(state.elapsed_minutes(), 2),
            "max_min": float(state.max_minutes or 0),
            "tokens": None,
            "gpu_sec": None,
        },
    )
    return True
