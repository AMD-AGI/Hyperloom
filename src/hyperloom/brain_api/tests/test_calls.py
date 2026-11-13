# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a propose or review answer must hold before the body acts on it."""

from __future__ import annotations

import pytest

from hyperloom.brain_api import (
    Candidate,
    ConfigDelta,
    Decision,
    PatchFile,
    Proposal,
    ProposeAsk,
    Review,
    ReviewAsk,
    Verdict,
    WireError,
    check_proposal,
    check_review,
)


def _config(name: str) -> Candidate:
    return Candidate(name, ConfigDelta(f"--{name}"))


def _patch(name: str) -> Candidate:
    return Candidate(name, patches=(PatchFile("python/x.py", "--- a\n+++ b\n"),))


def test_an_answer_within_the_ask_passes() -> None:
    proposal = Proposal((_config("a"), _patch("b")), confidence=0.5)
    assert check_proposal(ProposeAsk(domain="attention", max_candidates=2, patches=True), proposal) is proposal


@pytest.mark.parametrize(
    ("ask", "candidates", "match"),
    [
        (ProposeAsk(domain="x", max_candidates=1), (_config("a"), _config("b")), "allows 1"),
        (ProposeAsk(domain="x"), (_config("a"), _config("a")), "duplicate"),
        (ProposeAsk(domain="x", patches=False), (_patch("a"),), "patches"),
    ],
)
def test_an_answer_outside_the_ask_is_refused(ask, candidates, match) -> None:
    with pytest.raises(WireError, match=match):
        check_proposal(ask, Proposal(candidates))


def test_an_ask_needs_a_domain_or_a_task_and_room_for_a_candidate() -> None:
    with pytest.raises(WireError, match="domain or a task"):
        ProposeAsk()
    with pytest.raises(WireError, match="at least 1"):
        ProposeAsk(task="free", max_candidates=0)
    with pytest.raises(WireError, match=r"\[0, 1\]"):
        Proposal(confidence=1.5)


def test_a_review_holds_one_verdict_per_candidate() -> None:
    ask = ReviewAsk((_config("a"), _config("b")))
    review = Review((Verdict("b", Decision.REJECT), Verdict("a", Decision.APPROVE, predicted_gain_pct=2.0)))
    assert check_review(ask, review) is review
    for verdicts in [
        (Verdict("a", Decision.APPROVE),),
        (Verdict("a", Decision.APPROVE),) * 2 + (Verdict("b", Decision.DEFER),),
    ]:
        with pytest.raises(WireError, match="covers"):
            check_review(ask, Review(verdicts))


def test_a_review_ask_needs_distinct_candidates() -> None:
    with pytest.raises(WireError, match="at least one"):
        ReviewAsk(())
    with pytest.raises(WireError, match="duplicate"):
        ReviewAsk((_config("a"), _config("a")))
