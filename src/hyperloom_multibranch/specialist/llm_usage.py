# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one agent CLI launch cost, read from the result event its stream-json log ends with.

Claude Code writes that event when its model finishes its turn. A launch killed at its deadline, lost or stopped
writes none, and codex reports no cost, so such a launch's spend is not known here (board #66).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: The prompt-token counters of a Claude ``usage``: its ``input_tokens`` is only the uncached share.
_PROMPT_COUNTERS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
_RESULT = "result"


@dataclass(frozen=True)
class LaunchCost:
    """The CLI's own ``total_cost_usd`` for a launch, its prompt tokens (cached ones included), its output tokens,
    and the CLI session id it ran under."""

    cost_usd: float
    input_tokens: int
    output_tokens: int
    session_id: str


def launch_cost(process_log: Path) -> LaunchCost | None:
    """The last result event of a Claude stream-json log; ``None`` when the log is missing or has none."""
    try:
        lines = process_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if f'"{_RESULT}"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("type") != _RESULT:
            continue
        cost = row.get("total_cost_usd")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            continue
        usage = row.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        prompt = sum(_count(usage, key) for key in _PROMPT_COUNTERS)
        return LaunchCost(float(cost), prompt, _count(usage, "output_tokens"), str(row.get("session_id") or ""))
    return None


def _count(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


__all__ = ["LaunchCost", "launch_cost"]
