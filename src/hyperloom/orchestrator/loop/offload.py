# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run blocking work on a worker thread under a deadline the caller controls."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypeVar

from hyperloom.common.deadline import Deadline, seconds_until

log = logging.getLogger(__name__)

__all__ = [
    "offload",
]

#: Longest an offloaded call may hold its awaiter when no deadline bounds it.
OFFLOAD_UNBOUNDED_CAP_SEC: float = 120.0

T = TypeVar("T")


async def offload(
    work: Callable[[], T],
    *,
    deadline: Deadline | None,
    label: str,
    default: Any = None,
) -> T | Any:
    """Run ``work`` on a thread, returning ``default`` if it outlasts ``deadline``.

    Args:
        work: The blocking callable. Must bound its own I/O; it keeps running
            after a timeout here and nothing can stop it.
        deadline: When the caller stops waiting. ``None`` waits up to
            :data:`OFFLOAD_UNBOUNDED_CAP_SEC`.
        label: Short name for the work, used in the timeout warning.
        default: Returned when the wait ends first, or when ``work`` raises.

    Returns:
        The call's result, or ``default`` if it timed out or raised.
    """
    budget = seconds_until(deadline, unbounded_cap=OFFLOAD_UNBOUNDED_CAP_SEC)
    if budget <= 0.0:
        log.info("offload: skipping %s; its deadline had already passed", label)
        return default
    try:
        return await asyncio.wait_for(asyncio.to_thread(work), timeout=budget)
    except asyncio.TimeoutError:
        log.warning(
            "offload: %s outlasted its %.0fs deadline; the tick moved on and the thread was left to finish",
            label,
            budget,
        )
        return default
    except Exception:  # noqa: BLE001 — returning ``default`` on a raise is the contract
        log.exception("offload: %s raised", label)
        return default
