# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Placeholder nominator, kept in its own file so it can be replaced wholesale.

It exists to make the contract runnable end to end, not to be good at picking
kernels: it trusts Hyperloom's already-resolved rows and never looks at the
trace. A real nominator parses the trace, resolves the rows Hyperloom could not,
and splits the budget on evidence rather than evenly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernelforge.nomination import Candidate, NominationRequest, Target


def nominate_from_candidates(
    request: NominationRequest,
    candidates: list[Candidate],
) -> list[Target]:
    """Take the hottest already-resolved, unrejected rows and split the budget evenly.

    Args:
        request: The lane brief, for the budget and the ceiling.
        candidates: Rows from the hot-kernel list.

    Returns:
        At most ``request.max_kernels`` targets, hottest first. Empty when no row
        is eligible, which is a valid outcome and not an error.
    """
    from kernelforge.nomination import Target

    eligible = [candidate for candidate in candidates if candidate.is_resolved and not candidate.rejected]
    eligible.sort(key=lambda candidate: candidate.gpu_pct, reverse=True)
    picked = eligible[: request.max_kernels]
    if not picked:
        return []
    share = request.lane_budget_sec // len(picked)
    return [
        Target(
            kernel_name=candidate.kernel_name,
            source_file=candidate.source_file,
            budget_sec=share,
            gpu_pct=candidate.gpu_pct,
            reason="placeholder nominator: hottest resolved candidate",
        )
        for candidate in picked
    ]
