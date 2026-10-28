# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the instruction sheet asks the runner to do for the short-prompt consistency check.

The sheet names the staged prompt set by its key and by the hash of its prompts file, how many tokens each prompt may
generate, how long the check may take, and the lines of the recipe change that made the toolbox ask for it (so a signed
sheet says why the check ran). The test's own server answers, after its rounds; the check boots no server of its own.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from hyperloom.measure.docs import SHA256_RE, DocumentError, Reader

__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PHASE_CAP_SEC",
    "KEY_RE",
    "OUTPUTS_FILE",
    "SHORT_PROMPTS_DIR",
    "ShortPromptPlan",
    "miss_probability",
]

#: The directory, below a test's output directory, that holds the check's raw outputs; the signed bundle lists them.
SHORT_PROMPTS_DIR = "short_prompts"
OUTPUTS_FILE = "outputs.json"
DEFAULT_MAX_NEW_TOKENS = 32
#: The check's own time limit: a server that stops answering ends the check as incomplete, not the test.
DEFAULT_PHASE_CAP_SEC = 1200.0
_MAX_NEW_TOKENS = 256
_MAX_TRIGGER_LINES = 32
_MAX_TRIGGER_CHARS = 200
KEY_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
#: Keys a sheet written while the toolbox stored each check's outputs by recipe may carry; read and left out.
_FORMER_FIELDS = ("recipe", "parent_recipe")
#: The share of server processes a defect of the kind this check looks for shows in (the measured case: one in three).
_DEFECT_SHARE = 1 / 3


def miss_probability(processes: int) -> float:
    """The chance that ``processes`` launches (each a test's own server) all miss a defect that shows in one launch in
    three."""
    return (1 - _DEFECT_SHARE) ** processes


def _trigger_line(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TRIGGER_CHARS:
        raise DocumentError(f"{path}: expected a non-empty string of at most {_MAX_TRIGGER_CHARS} characters")
    return value


@dataclass(frozen=True)
class ShortPromptPlan:
    """One test's short-prompt check: ``prompt_set`` is the key of the staged entry and ``prompts_sha256`` the hash of its
    prompts file; the test's own server, after its measured rounds, answers every prompt with at most ``max_new_tokens``
    greedy tokens, and the check ends, incomplete, after ``phase_cap_sec``. ``trigger`` says why the check was asked for."""

    prompt_set: str
    prompts_sha256: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    phase_cap_sec: float = DEFAULT_PHASE_CAP_SEC
    trigger: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not KEY_RE.fullmatch(self.prompt_set):
            raise DocumentError(f"short prompts: {self.prompt_set!r} is not a prompt set key")
        if not SHA256_RE.fullmatch(self.prompts_sha256):
            raise DocumentError("short prompts: prompts_sha256 is not a sha256")
        if isinstance(self.max_new_tokens, bool) or not 1 <= self.max_new_tokens <= _MAX_NEW_TOKENS:
            raise DocumentError(f"short prompts: max_new_tokens must be 1 to {_MAX_NEW_TOKENS}")
        if not math.isfinite(self.phase_cap_sec) or self.phase_cap_sec <= 0:
            raise DocumentError("short prompts: phase_cap_sec must be a positive number")
        if len(self.trigger) > _MAX_TRIGGER_LINES:
            raise DocumentError(f"short prompts: at most {_MAX_TRIGGER_LINES} trigger lines")
        for index, line in enumerate(self.trigger):
            _trigger_line(line, f"short prompts: trigger[{index}]")

    def to_document(self) -> dict[str, Any]:
        return {
            "prompt_set": self.prompt_set,
            "prompts_sha256": self.prompts_sha256,
            "max_new_tokens": self.max_new_tokens,
            "phase_cap_sec": self.phase_cap_sec,
            "trigger": list(self.trigger),
        }

    @classmethod
    def parse(cls, value: Any, path: str) -> ShortPromptPlan:
        r = Reader(value, path)
        plan = cls(
            prompt_set=r.text("prompt_set", pattern=KEY_RE) or "",
            prompts_sha256=r.text("prompts_sha256", pattern=SHA256_RE) or "",
            max_new_tokens=r.integer("max_new_tokens", minimum=1) or 1,
            phase_cap_sec=r.number("phase_cap_sec", minimum=1) or 1.0,
            trigger=r.sequence("trigger", _trigger_line),
        )
        for name in _FORMER_FIELDS:
            r.optional(name)
        r.done()
        return plan
