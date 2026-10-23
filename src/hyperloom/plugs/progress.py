# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the run's code emits a progress event: one call, to whichever implementation fills the ``events`` socket.

Best effort by construction: an event that cannot be emitted is logged at debug level and the run goes on.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Mapping

from hyperloom.plugs.host import active

log = logging.getLogger(__name__)

EVENT_PHASE_CHANGED = "hyperloom.phase_changed"
EVENT_CANDIDATE_SETTLED = "hyperloom.candidate_settled"
EVENT_BEST_UPDATED = "hyperloom.best_updated"
EVENT_BUDGET = "hyperloom.budget"
EVENT_WARNING = "hyperloom.warning"

#: At most one ``budget`` event per this many seconds.
BUDGET_INTERVAL_SEC = 300.0

_budget_lock = threading.Lock()
_budget_last: float | None = None


def emit(event_type: str, data: Mapping[str, Any]) -> None:
    """Hand one event to the run's ``events`` implementation; never raises."""
    try:
        active().get("events").emit(event_type, data)  # type: ignore[attr-defined]
    except Exception:
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


def best_updated(value: float, baseline: float, patch_count: int, request_id: str | None) -> None:
    """Emit ``hyperloom.best_updated``; ``gain_pct`` is against ``baseline`` and null without a positive one."""
    gain = round((value / baseline - 1.0) * 100.0, 3) if baseline and baseline > 0 else None
    emit(
        EVENT_BEST_UPDATED,
        {
            "metric": "output_throughput",
            "value": value,
            "gain_pct": gain,
            "patch_count": int(patch_count),
            "request_id": request_id or None,
        },
    )


def emit_budget(*, elapsed_min: float, max_min: float, tokens: int | None = None, gpu_sec: float | None = None) -> bool:
    """Emit ``hyperloom.budget`` unless one went out within :data:`BUDGET_INTERVAL_SEC`; report whether it did."""
    global _budget_last
    now = time.monotonic()
    with _budget_lock:
        if _budget_last is not None and now - _budget_last < BUDGET_INTERVAL_SEC:
            return False
        _budget_last = now
    emit(EVENT_BUDGET, {"elapsed_min": elapsed_min, "max_min": max_min, "tokens": tokens, "gpu_sec": gpu_sec})
    return True
