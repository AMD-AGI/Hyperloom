# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GPU a campaign works on: one card, or a pool it moves through when someone else takes its card.

Every child process inherits ``HIP_VISIBLE_DEVICES``. Right before the work that uses the device, a card whose used
memory is above a threshold counts as taken; the campaign keeps its card while it is free, moves to a free card of
its pool otherwise, and waits while every card of the pool is taken.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

DEVICE_ENV = "HIP_VISIBLE_DEVICES"
# rocm-smi must see every card, not only the ones this process was given.
_VISIBILITY_ENVS = (DEVICE_ENV, "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


@dataclass(frozen=True)
class GpuChoice:
    """Which card to start on, which cards to move to, and when a card counts as taken."""

    gpu: str = ""
    pool: tuple[str, ...] = ()
    free_vram_mb: int = 1024
    poll_sec: int = 60

    def pin(self) -> None:
        """Make the starting card the one every child process sees; no card leaves the inherited device alone."""
        if self.gpu:
            os.environ[DEVICE_ENV] = self.gpu


def parse_cards(text: str) -> list[str]:
    """``0,1, 3`` as card indices; blank entries are dropped."""
    return [card.strip() for card in (text or "").split(",") if card.strip()]


def free_cards(threshold_bytes: int) -> dict[str, bool] | None:
    """Each card's index and whether its used memory is below ``threshold_bytes``; None when rocm-smi cannot say."""
    env = {key: value for key, value in os.environ.items() if key not in _VISIBILITY_ENVS}
    try:
        shown = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"], env=env, capture_output=True, text=True, timeout=60
        )
        data = json.loads(shown.stdout[shown.stdout.index("{") : shown.stdout.rindex("}") + 1])
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    free = {}
    for card, fields in data.items():
        match = re.fullmatch(r"card(\d+)", card)
        used = [value for key, value in fields.items() if "used" in key.lower()] if isinstance(fields, dict) else []
        if match and used:
            try:
                free[match.group(1)] = int(float(used[0])) < threshold_bytes
            except (TypeError, ValueError):
                continue
    return free or None


async def free_card(
    choice: GpuChoice,
    current: str,
    *,
    on_wait: Callable[[list[str]], None],
    probe: Callable[[int], dict[str, bool] | None] = free_cards,
) -> str:
    """The card to work on now: ``current`` while it is free, else the first free card of the pool, waiting while
    every one is taken. Without a pool, or when the cards cannot be read, ``current``."""
    if not choice.pool:
        return current
    while True:
        free = probe(choice.free_vram_mb * 1024 * 1024)
        if free is None:
            return current
        for card in [current, *(card for card in choice.pool if card != current)]:
            if free.get(card):
                return card
        on_wait([card for card in choice.pool if not free.get(card)])
        await asyncio.sleep(choice.poll_sec)
