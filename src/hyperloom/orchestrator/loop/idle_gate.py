# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skip an orchestration turn whose prompt carries nothing new since the last turn that ran.

A stuck session re-sends a near-identical prompt every tick: only the clock, the tick number, the model's own
previous summary and fresh ids on repeated inbox messages change. :func:`normalize_prompt` removes exactly those, so
two prompts with the same digest offer the model no new information. A heartbeat still lets one turn through at
least every ``heartbeat_sec`` so time-based decisions are never starved.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

IDLE_TICK_GATE_ENV = "HYPERLOOM_IDLE_TICK_GATE"
IDLE_TICK_HEARTBEAT_ENV = "HYPERLOOM_IDLE_TICK_HEARTBEAT_SEC"
DEFAULT_HEARTBEAT_SEC = 900.0

# Clock and budget lines keep their keys and flags (closing_phase, reloop feasibility); only the numbers move.
_CLOCK_LINE = re.compile(r"^\s*(budget\s*:|reloop\s*:|time\s*:|elapsed=)")
_SELF_SUMMARY = re.compile(r"^\s*current_action=")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TICK = re.compile(r"\btick[= ]\d+")
_MESSAGE_ID = re.compile(r"\bseq=\d+\s+msg_id=[0-9a-f]+|\bmsg_id=[0-9a-f]{16,}")
_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:min|sec|s)\b")


def normalize_prompt(prompt: str) -> str:
    """Return ``prompt`` without the parts that change every tick without informing a decision."""
    out: list[str] = []
    seen_messages: set[str] = set()
    for line in prompt.splitlines():
        if _SELF_SUMMARY.match(line):
            continue
        if _CLOCK_LINE.match(line):
            line = _NUMBER.sub("N", line)
        line = _TICK.sub("tick=N", line)
        line = _TIMESTAMP.sub("TS", line)
        line = _DURATION.sub("T", line)
        if _MESSAGE_ID.search(line):
            line = _MESSAGE_ID.sub("msg_id=X", line)
            if line in seen_messages:
                continue
            seen_messages.add(line)
        out.append(line)
    return "\n".join(out)


def prompt_digest(prompt: str) -> str:
    """Stable digest of the normalized prompt."""
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


@dataclass
class IdleTickGate:
    """Per-agent memory of the last prompt that reached the model."""

    enabled: bool = True
    heartbeat_sec: float = DEFAULT_HEARTBEAT_SEC
    skipped: int = 0
    _last_digest: str = field(default="", repr=False)
    _last_sent_at: float = field(default=0.0, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> IdleTickGate:
        """Build the gate from ``HYPERLOOM_IDLE_TICK_GATE`` (default on) and the heartbeat env."""
        source = os.environ if env is None else env
        enabled = str(source.get(IDLE_TICK_GATE_ENV, "1")).strip().lower() not in {"0", "false", "no", "off"}
        try:
            heartbeat = float(source.get(IDLE_TICK_HEARTBEAT_ENV, DEFAULT_HEARTBEAT_SEC))
        except (TypeError, ValueError):
            heartbeat = DEFAULT_HEARTBEAT_SEC
        return cls(enabled=enabled, heartbeat_sec=max(0.0, heartbeat))

    def should_skip(self, prompt: str, now: float) -> bool:
        """True when ``prompt`` matches the last prompt sent and the heartbeat is not due."""
        if not self.enabled or not self._last_digest:
            return False
        if self.heartbeat_sec and now - self._last_sent_at >= self.heartbeat_sec:
            return False
        if prompt_digest(prompt) != self._last_digest:
            return False
        self.skipped += 1
        return True

    def record_sent(self, prompt: str, now: float) -> None:
        """Remember ``prompt`` as the last one the model saw."""
        self._last_digest = prompt_digest(prompt)
        self._last_sent_at = now


__all__ = [
    "DEFAULT_HEARTBEAT_SEC",
    "IDLE_TICK_GATE_ENV",
    "IDLE_TICK_HEARTBEAT_ENV",
    "IdleTickGate",
    "normalize_prompt",
    "prompt_digest",
]
