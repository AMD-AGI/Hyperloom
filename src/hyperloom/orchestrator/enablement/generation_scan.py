# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which idempotency-key generations an enablement task row opening tries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationScan:
    """Which idempotency-key generations a row opening tries.

    Attributes:
        key_for: Builds the idempotency key for a generation number.
        generation: The generation to try first.
        attempts: How many generations to try before giving up this pass.
    """

    key_for: Callable[[int], str]
    generation: int
    attempts: int = 2
