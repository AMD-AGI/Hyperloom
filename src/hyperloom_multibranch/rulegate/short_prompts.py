# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt consistency rule: a candidate's processes may disagree with each other no more than the session's
baseline processes did.

The floor is the inconsistency the root's BASELINE tests measured; a candidate whose complete check is more inconsistent
than that shows the defect. A session without a floor, a candidate without a check, and a check that is incomplete or
compared fewer than two processes refuse nothing: the check stays recorded on the test's result.
"""

from __future__ import annotations

from collections.abc import Iterable

from hyperloom.contracts.consistency import SHORT_PROMPTS_COMPLETE
from hyperloom.contracts.measurement import TestResult

SHORT_PROMPTS_INCONSISTENT = "short_prompts_inconsistent"


def baseline_floor(baselines: Iterable[TestResult]) -> int | None:
    """The largest inconsistency any of ``baselines`` measured; ``None`` when none compared two processes.

    Every baseline measures the same recipe, so the most any of them shows is what that recipe does on its own.
    """
    checks = (result.short_prompt_consistency for result in baselines)
    measured = [check.inconsistency for check in checks if check is not None and check.inconsistency is not None]
    return max(measured, default=None)


def exceeds_floor(result: TestResult, floor: int | None) -> bool:
    check = result.short_prompt_consistency
    if floor is None or check is None or check.status != SHORT_PROMPTS_COMPLETE or check.inconsistency is None:
        return False
    return check.inconsistency > floor


__all__ = ["SHORT_PROMPTS_INCONSISTENT", "baseline_floor", "exceeds_floor"]
