# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0 honours the contract and, until the body routes a call to it, decides nothing."""

from __future__ import annotations

from hyperloom.brain import create_brain
from hyperloom.brain_api import (
    Candidate,
    ConfigDelta,
    Decision,
    ProposeAsk,
    ReviewAsk,
    SearchState,
    Workload,
    check_proposal,
    check_review,
    validate_brain,
)

_STATE = SearchState(Workload("m", "sglang"), baseline_tput=1.0, current_tput=1.0, keep_threshold_pct=1.0)


def test_v0_honours_the_contract() -> None:
    assert validate_brain(create_brain({})).name == "v0"


async def test_v0_proposes_nothing_and_defers_every_review() -> None:
    brain = create_brain({})
    ask = ProposeAsk(domain="attention")
    assert check_proposal(ask, await brain.propose(_STATE, ask, None)).candidates == ()
    review_ask = ReviewAsk((Candidate("a", ConfigDelta("--a")), Candidate("b", ConfigDelta("--b"))))
    review = check_review(review_ask, await brain.review(_STATE, review_ask, None))
    assert {v.decision for v in review.verdicts} == {Decision.DEFER}
