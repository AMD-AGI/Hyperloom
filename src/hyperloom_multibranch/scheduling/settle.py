# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Awaiting a peer call whose failure must not stop the caller's next step, as every step of a stop must run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def settled(work: Awaitable[T]) -> T | BaseException:
    """``work``'s result, or the exception it raised; cancelling the caller still cancels ``work`` and propagates."""

    (outcome,) = await asyncio.gather(work, return_exceptions=True)
    return outcome


def failure_text(error: BaseException) -> str:
    """How a failed step is named in the stop decisions and the stop report."""

    return f"{type(error).__name__}: {error}"


__all__ = ["failure_text", "settled"]
