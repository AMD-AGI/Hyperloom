# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run blocking work on a worker thread under a deadline the caller controls."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, TypeVar

from hyperloom.common.deadline import Deadline

log = logging.getLogger(__name__)

__all__ = [
    "offload",
]

T = TypeVar("T")


async def offload(work: Callable[[], T], *, deadline: Deadline, label: str) -> T | None:
    """Run ``work`` on a thread, returning ``None`` if it outlasts ``deadline``; its exceptions propagate.

    Args:
        work: The blocking callable. Must bound its own I/O; it keeps running
            after a timeout here and nothing can stop it.
        deadline: When the caller stops waiting.
        label: Short name for the work, used in the timeout warning.

    Returns:
        The call's result, or ``None`` if it timed out.
    """
    budget = deadline.remaining()
    if budget <= 0.0:
        log.info("offload: skipping %s; its deadline had already passed", label)
        return None
    try:
        return await asyncio.wait_for(asyncio.to_thread(work), timeout=budget)
    except asyncio.TimeoutError:
        log.warning(
            "offload: %s outlasted its %.0fs deadline; the tick moved on and the thread was left to finish",
            label,
            budget,
        )
        return None
