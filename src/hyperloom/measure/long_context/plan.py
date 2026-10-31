# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the instruction sheet asks the runner to do for the long-context check.

The sheet names the staged prompt set by its key and by the hash of its prompts file, the token length its prompts were
built for, how many tokens an answer may have, how many requests are in flight, how long the whole check may take, and why
the toolbox asked for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hyperloom.measure.docs import SHA256_RE, DocumentError, Reader
from hyperloom.measure.short_prompts.plan import KEY_RE

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PHASE_CAP_SEC",
    "LONG_CONTEXT_DIR",
    "MAX_LENGTH",
    "MIN_LENGTH",
    "OUTPUTS_FILE",
    "REPLY_HEADROOM",
    "LongContextPlan",
]

#: The directory, below a test's output directory, that holds the check's raw outputs; the signed bundle lists them.
LONG_CONTEXT_DIR = "long_context"
OUTPUTS_FILE = "outputs.json"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_CONCURRENCY = 4
#: The check's own time limit: prompts not answered by then end the check as incomplete, not the test.
DEFAULT_PHASE_CAP_SEC = 2400.0
#: The server must serve the prompt and its answer: a launch that serves less than the prompt length and this is refused.
REPLY_HEADROOM = 128
MIN_LENGTH = 1024
MAX_LENGTH = 1 << 21
_MAX_CONCURRENCY = 64
_MAX_TRIGGER_LINES = 32
_MAX_TRIGGER_CHARS = 200


def _trigger_line(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TRIGGER_CHARS:
        raise DocumentError(f"{path}: expected a non-empty string of at most {_MAX_TRIGGER_CHARS} characters")
    return value


def _in_range(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


@dataclass(frozen=True)
class LongContextPlan:
    """One test's long-context check: ``prompt_set`` is the key of the staged entry, ``prompts_sha256`` the hash of its
    prompts file and ``length`` the token length its prompts were built for. Each prompt is answered greedily with at most
    ``max_new_tokens`` tokens, ``concurrency`` requests at a time, and the check ends, incomplete, after ``phase_cap_sec``.
    ``trigger`` says why the check was asked for."""

    prompt_set: str
    prompts_sha256: str
    length: int
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    concurrency: int = DEFAULT_CONCURRENCY
    phase_cap_sec: float = DEFAULT_PHASE_CAP_SEC
    trigger: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not KEY_RE.fullmatch(self.prompt_set):
            raise DocumentError(f"long context: {self.prompt_set!r} is not a prompt set key")
        if not SHA256_RE.fullmatch(self.prompts_sha256):
            raise DocumentError("long context: prompts_sha256 is not a sha256")
        if not _in_range(self.length, MIN_LENGTH, MAX_LENGTH):
            raise DocumentError(f"long context: length must be {MIN_LENGTH} to {MAX_LENGTH}, got {self.length!r}")
        if not _in_range(self.max_new_tokens, 1, REPLY_HEADROOM):
            raise DocumentError(f"long context: max_new_tokens must be 1 to {REPLY_HEADROOM}")
        if not _in_range(self.concurrency, 1, _MAX_CONCURRENCY):
            raise DocumentError(f"long context: concurrency must be 1 to {_MAX_CONCURRENCY}")
        if not math.isfinite(self.phase_cap_sec) or self.phase_cap_sec <= 0:
            raise DocumentError("long context: phase_cap_sec must be a positive number")
        if len(self.trigger) > _MAX_TRIGGER_LINES:
            raise DocumentError(f"long context: at most {_MAX_TRIGGER_LINES} trigger lines")
        for index, line in enumerate(self.trigger):
            _trigger_line(line, f"long context: trigger[{index}]")

    @property
    def needed_context(self) -> int:
        """The least context a launch must serve for this check: a prompt of ``length`` tokens and its answer."""
        return self.length + REPLY_HEADROOM

    def to_document(self) -> dict[str, Any]:
        return {
            "prompt_set": self.prompt_set,
            "prompts_sha256": self.prompts_sha256,
            "length": self.length,
            "max_new_tokens": self.max_new_tokens,
            "concurrency": self.concurrency,
            "phase_cap_sec": self.phase_cap_sec,
            "trigger": list(self.trigger),
        }

    @classmethod
    def parse(cls, value: Any, path: str) -> LongContextPlan:
        r = Reader(value, path)
        plan = cls(
            prompt_set=r.text("prompt_set", pattern=KEY_RE) or "",
            prompts_sha256=r.text("prompts_sha256", pattern=SHA256_RE) or "",
            length=r.integer("length", minimum=MIN_LENGTH) or MIN_LENGTH,
            max_new_tokens=r.integer("max_new_tokens", minimum=1) or 1,
            concurrency=r.integer("concurrency", minimum=1) or 1,
            phase_cap_sec=r.number("phase_cap_sec", minimum=1) or 1.0,
            trigger=r.sequence("trigger", _trigger_line),
        )
        r.done()
        return plan
