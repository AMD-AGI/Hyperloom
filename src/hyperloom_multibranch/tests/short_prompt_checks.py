# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Short-prompt consistency checks as a test result carries them, for the rule gate's tests."""

from __future__ import annotations

from hyperloom.contracts.consistency import (
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
    OutputCounts,
    OutputDifference,
    ShortPromptConsistency,
)

_DIFFERENCE = OutputDifference(1, "s05", "sequential", 7, 1, "abc", "aXY")


def short_prompt_check(*mismatches: int, planned: int | None = None) -> ShortPromptConsistency:
    """A check whose completed processes answer ``mismatches`` prompts each differently from their consensus; it is
    incomplete when fewer than ``planned`` processes completed, and compares nothing below two."""
    planned = planned or max(len(mismatches), 1)
    counts = OutputCounts(mismatches, _DIFFERENCE if any(mismatches) else None) if len(mismatches) >= 2 else None
    complete = len(mismatches) == planned
    return ShortPromptConsistency(
        status=SHORT_PROMPTS_COMPLETE if complete else SHORT_PROMPTS_INCOMPLETE,
        reason="" if complete else "a process did not answer every prompt",
        prompt_set_sha256="a" * 64,
        prompts=64,
        max_new_tokens=32,
        trigger=("purpose baseline",),
        processes_planned=planned,
        processes=tuple(range(len(mismatches))),
        miss_probability_one_third=0.088,
        compared_by="tokens" if counts is not None else "",
        self_consistency=counts,
        outputs="short_prompts/outputs.json",
    )
