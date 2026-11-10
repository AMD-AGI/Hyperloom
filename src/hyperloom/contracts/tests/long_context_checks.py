# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long-context checks as a test result carries them, for the tests of the bar and of the rule gate that applies it."""

from __future__ import annotations

from hyperloom.contracts.long_context import (
    LONG_CONTEXT_COMPLETE,
    LONG_CONTEXT_INCOMPLETE,
    PROMPT_ERROR,
    PROMPT_OK,
    LongContextPrompt,
    LongContextResult,
    LongContextRoot,
    LongContextTask,
)

SHA = "a" * 64
TASK = "niah_single"


def long_context_check(
    n: int, passed: int, *, errored: int = 0, sha: str = SHA, root: LongContextRoot | None = None
) -> LongContextResult:
    """A check of ``n`` prompts, the first ``passed`` of them passed and the last ``errored`` of them not answered, judged
    after ``root``."""
    prompts = tuple(
        LongContextPrompt(
            f"p-{index}",
            TASK,
            PROMPT_ERROR if index >= n - errored else PROMPT_OK,
            index < passed,
            0 if index >= n - errored else 2000,
            1.0,
        )
        for index in range(n)
    )
    return LongContextResult(
        status=LONG_CONTEXT_INCOMPLETE if errored else LONG_CONTEXT_COMPLETE,
        reason=f"{errored} of {n} prompts were not answered" if errored else "",
        prompt_set_sha256=sha,
        length=2048,
        max_new_tokens=128,
        concurrency=4,
        served_context=4096,
        n=n,
        passed=passed,
        score=passed / n,
        tasks=(LongContextTask(TASK, n, passed, passed / n),),
        prompts=prompts,
        seconds=10.0,
        prefill_tokens=2000 * (n - errored),
        root=root,
    )
