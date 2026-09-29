# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skip an orchestration LLM turn when nothing it could act on has changed since the last one.

While a long task holds the serving / benchmark lanes the orchestrator has nothing to do, yet
the coordinator used to ask it every tick. In one KERNEL phase (Qwen3.5-122B, 631 ticks) the
replies were "no action taken, by design" or a re-sent ``integrate`` answered
``deferred: lanes_busy``, and each such turn re-read ~220k context tokens.

The gate compares the tick prompt, stripped of what moves on its own (clocks, the tick
counter, message ids, bare acknowledgements), with the prompt as it stood right after the
last real turn. Identical means the turn would see exactly what the previous one saw, so it
is skipped; any new message, outcome, task state or phase change reopens it, and a heartbeat
bounds how long the orchestrator can stay silent.
"""

from __future__ import annotations

import hashlib
import os
import re

GATE_ENV = "HYPERLOOM_IDLE_TICK_GATE"
HEARTBEAT_ENV = "HYPERLOOM_IDLE_TICK_HEARTBEAT_SEC"
DEFAULT_HEARTBEAT_SEC = 300.0

# Lines whose only change between ticks is the clock or the tick counter. ``agent_last_active``
# ages on every skipped tick, so keeping it would reopen the gate one tick after it closes.
_CLOCK_LINE = re.compile(
    r"^\s*(budget|reloop|time|entered)\s*:.*$"
    r"|^\s*elapsed=[\d.]+min\b.*$"
    r"|^\s*tick=\d+\b.*$"
    r"|^\s*agent_last_active=.*$",
    re.M,
)
# Per-message identifiers that differ even when the message says the same thing.
_MESSAGE_IDS = re.compile(r"\bseq=\d+ |\bmsg_id=[0-9a-f]{8,} |'in_reply_to': '[0-9a-f]{8,}',? ?")
# Acknowledgements that carry no information (the critic posts one every tick).
_BARE_ACK = re.compile(
    r"^.*topic=(observation|heartbeat) payload=\{'topic': '(observation|heartbeat)', "
    r"'body_md': 'ok(?: \([a-z_]+\))?'\}\s*$",
    re.M,
)
_INBOX_HEADER = re.compile(r"^=== Inbox for (\w+)(?: \(newest last\))? ===$", re.M)
_EMPTY_INBOX = re.compile(r"^\(no new messages\)$", re.M)


def normalize_prompt(prompt: str) -> str:
    """The prompt with self-moving parts removed; equal outputs mean nothing decidable changed."""
    text = _BARE_ACK.sub("", prompt)
    text = _CLOCK_LINE.sub("", text)
    text = _MESSAGE_IDS.sub("", text)
    # An inbox holding only acknowledgements reads the same as an empty one.
    text = _INBOX_HEADER.sub(r"=== Inbox for \1 ===", text)
    text = _EMPTY_INBOX.sub("", text)
    return "\n".join(line for line in text.splitlines() if line.strip())


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _env_enabled() -> bool:
    return (os.environ.get(GATE_ENV) or "1").strip().lower() not in ("0", "false", "no", "off")


def _env_heartbeat() -> float:
    try:
        return max(0.0, float(os.environ.get(HEARTBEAT_ENV) or DEFAULT_HEARTBEAT_SEC))
    except ValueError:
        return DEFAULT_HEARTBEAT_SEC


class IdleTickGate:
    """Tracks the state the orchestrator last saw and says when a new turn would be redundant."""

    def __init__(self, *, enabled: bool | None = None, heartbeat_sec: float | None = None) -> None:
        self.enabled = _env_enabled() if enabled is None else enabled
        self.heartbeat_sec = _env_heartbeat() if heartbeat_sec is None else heartbeat_sec
        self._digests: frozenset[str] = frozenset()
        self._last_call: float | None = None
        self.skipped = 0
        self.streak = 0

    def should_skip(self, prompt: str, now: float) -> bool:
        """True when ``prompt`` shows nothing new since the last turn and the heartbeat is not due."""
        if not self.enabled or not self._digests or self._last_call is None:
            return False
        if now - self._last_call >= self.heartbeat_sec:
            return False
        if prompt_digest(prompt) not in self._digests:
            return False
        self.skipped += 1
        self.streak += 1
        return True

    def record_turn(self, prompt_after: str, now: float, *, prompt_before: str | None = None) -> None:
        """Remember what the last real turn saw and the state it left behind (intents applied, cursor advanced)."""
        seen = {prompt_digest(prompt_after)}
        if prompt_before is not None:
            seen.add(prompt_digest(prompt_before))
        self._digests = frozenset(seen)
        self._last_call = now
        self.streak = 0

    def reset(self) -> None:
        """Forget the last turn, so the next tick always reaches the model."""
        self._digests = frozenset()
        self._last_call = None
        self.streak = 0


__all__ = [
    "DEFAULT_HEARTBEAT_SEC",
    "GATE_ENV",
    "HEARTBEAT_ENV",
    "IdleTickGate",
    "normalize_prompt",
    "prompt_digest",
]
