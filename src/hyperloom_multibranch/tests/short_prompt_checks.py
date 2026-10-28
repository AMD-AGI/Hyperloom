# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Short-prompt consistency checks as a test result carries them, for the rule gate's tests."""

from __future__ import annotations

from hyperloom.contracts.consistency import (
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
    OutputCounts,
    OutputDifference,
    PooledConsistency,
    ShortPromptConsistency,
)

_DIFFERENCE = OutputDifference(1, "s05", "sequential", 7, 1, "abc", "aXY")


def short_prompt_check(*mismatches: int, complete: bool = True) -> ShortPromptConsistency:
    """A check whose recipe's pool holds processes that answer ``mismatches`` prompts each differently from their
    consensus (the pool compares nothing below two); ``complete`` says whether the test's own server answered every
    prompt."""
    counts = OutputCounts(mismatches, _DIFFERENCE if any(mismatches) else None) if len(mismatches) >= 2 else None
    names = tuple(f"tests/t{index}/attempt-1#0" for index in range(len(mismatches)))
    pooled = PooledConsistency("sha256:" + "1" * 64, names, "tokens" if counts is not None else "", counts)
    return ShortPromptConsistency(
        status=SHORT_PROMPTS_COMPLETE if complete else SHORT_PROMPTS_INCOMPLETE,
        reason="" if complete else "the test's own server did not answer every prompt",
        prompt_set_sha256="a" * 64,
        prompts=64,
        max_new_tokens=32,
        trigger=("purpose baseline",),
        processes=(0,) if complete else (),
        miss_probability_one_third=2 / 3 if complete else 1.0,
        compared_by="tokens" if complete else "",
        outputs="short_prompts/outputs.json",
        pooled=pooled,
    )
