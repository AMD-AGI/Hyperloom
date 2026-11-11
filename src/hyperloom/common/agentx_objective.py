# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX keep rule's facts: the objectives a session is graded on, the bar each must clear and the guards it
may not pay for that gain with. Single-lineage and multibranch grading both read them from here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    TOTAL_TOKEN_THROUGHPUT,
    parse_intvty_noise_pct,
)


@dataclass(frozen=True)
class Guard:
    """An axis a candidate must hold within the noise band of its parent, and the refusal that says it did not."""

    refusal: str
    metric: str

    @property
    def name(self) -> str:
        return _TITLES[self.metric]


class AgentXObjective(str, Enum):
    """The axis an AgentX session is graded on: the round metric its bar reads (:attr:`metric`), named in words
    (:attr:`title`) and told to the LLM roles with what moves it (:attr:`description`), the gain that clears it
    (:attr:`bar_pct`), the refusal that says a candidate did not (:attr:`below_threshold`), and the axes it may not pay
    for that gain with (:attr:`guards`), each held within :attr:`guard_band_pct` of the parent. The error-rate bar
    holds under either."""

    P50_INTERACTIVITY = "p50_interactivity"
    TOTAL_TOKEN_THROUGHPUT = "total_token_throughput"

    @property
    def metric(self) -> str:
        return _METRICS[self]

    @property
    def title(self) -> str:
        return _TITLES[self.metric]

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]

    @property
    def bar_pct(self) -> float:
        return _BAR_PCT

    @property
    def below_threshold(self) -> str:
        return _BELOW_THRESHOLD[self]

    @property
    def guards(self) -> tuple[Guard, ...]:
        return _GUARDS[self]

    @property
    def guard_band_pct(self) -> float:
        return parse_intvty_noise_pct()


P50_BELOW_THRESHOLD = "p50_gain_below_threshold"
TOTAL_BELOW_THRESHOLD = "total_throughput_gain_below_threshold"
_METRICS = {
    AgentXObjective.P50_INTERACTIVITY: GRADED_INTVTY_P50,
    AgentXObjective.TOTAL_TOKEN_THROUGHPUT: TOTAL_TOKEN_THROUGHPUT,
}
_BELOW_THRESHOLD = {
    AgentXObjective.P50_INTERACTIVITY: P50_BELOW_THRESHOLD,
    AgentXObjective.TOTAL_TOKEN_THROUGHPUT: TOTAL_BELOW_THRESHOLD,
}
_TITLES = {
    GRADED_INTVTY_P50: "P50 interactivity",
    GRADED_INTVTY: "P90 interactivity",
    TOTAL_TOKEN_THROUGHPUT: "total token throughput",
}
_DESCRIPTIONS = {
    AgentXObjective.P50_INTERACTIVITY: (
        "the median request's per-output-token speed end to end, decode and time to first token together. Faster "
        "decode, attention and MoE kernels, a lower TTFT and a higher speculative-decoding acceptance move it."
    ),
    AgentXObjective.TOTAL_TOKEN_THROUGHPUT: (
        "the tokens, input and output, the server completes per second at the session's concurrency. Capacity moves "
        "it: batching and scheduling, prefill efficiency, KV-cache capacity and prefix-cache hits, fewer stalls. It "
        "moves only at high concurrency: at low concurrency the trace's tool and think time paces the requests, and a "
        "faster server barely moves it."
    ),
}
# A property of the objective, not of how far into a session a round lands, so no decaying session threshold applies.
_BAR_PCT = 2.0
_P90_GUARD = Guard("p90_regressed", GRADED_INTVTY)
# Output throughput guards neither: it follows the trace window's output-length mix, not the server.
_GUARDS = {
    AgentXObjective.P50_INTERACTIVITY: (_P90_GUARD, Guard("total_regressed", TOTAL_TOKEN_THROUGHPUT)),
    AgentXObjective.TOTAL_TOKEN_THROUGHPUT: (_P90_GUARD, Guard("p50_regressed", GRADED_INTVTY_P50)),
}
#: The refusals that say only that a candidate did not clear its objective's bar, under any objective.
BELOW_THRESHOLD = frozenset(_BELOW_THRESHOLD.values())

__all__ = [
    "BELOW_THRESHOLD",
    "P50_BELOW_THRESHOLD",
    "TOTAL_BELOW_THRESHOLD",
    "AgentXObjective",
    "Guard",
]
