# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: today's specialist and Critic decisions, moved behind the contract one call at a time.

Until the body routes a call here, v0 decides nothing: it proposes no candidate and defers every review.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import (
    CONTRACT_VERSION,
    Decision,
    Ports,
    Proposal,
    ProposeAsk,
    Review,
    ReviewAsk,
    SearchState,
    Verdict,
    eval_point,
)


class BrainV0:
    """The brain that reproduces the current behaviour."""

    name = "v0"
    contract_version = CONTRACT_VERSION

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @eval_point("main.propose")
    async def propose(self, state: SearchState, ask: ProposeAsk, ports: Ports) -> Proposal:
        """No specialist has moved yet."""
        return Proposal(summary="v0 does not propose yet")

    @eval_point("main.review")
    async def review(self, state: SearchState, ask: ReviewAsk, ports: Ports) -> Review:
        """The Critic has not moved yet."""
        return Review(tuple(Verdict(c.name, Decision.DEFER, reason="v0 does not review yet") for c in ask.candidates))
