# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a request pays per input token before the GPU sees it, read off the requests a run already recorded.

On a cached prefix the GPU does almost no prefill work, so time-to-first-token should barely move with how long
the prompt is. When it moves anyway, the cost is in front of the engine -- rendering the template, tokenising the
whole prompt again, pickling it to the scheduler -- and it is paid per input token whether or not the KV for those
tokens is already resident.

Measured on a GLM-5.3 AgentX run with a 97.9% prefix-cache hit: TTFT runs 225 ms at 25k of input and 1414 ms at
625k, about 198 ms per 100k. The hand-tuned stack took the same workload's front end down to roughly 7 ms per
100k by encoding the prompt in parallel and caching tokenised segments, which was its single largest step.

The GPU trace cannot show this. In that same run the device was 99.4% busy inside every step and the steps covered
98.8% of the capture: there is no idle to find, because the cost is paid before a step exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.orchestrator.measurement.intvty_samples import metric_values, profiling_records

#: Requests below this cannot carry a slope worth reading.
MIN_REQUESTS: int = 60

#: Input span, in tokens, a run must cover before a slope over it means anything.
MIN_ISL_SPAN: float = 50_000.0

#: Slope, in ms per 100k input tokens, above which the front end is worth a reader's attention. The hand-tuned
#: GLM-5.3 stack reached about 7 ms per 100k after its host-side work, so a run well above that is paying for
#: something that has been removed before.
FLAG_MS_PER_100K: float = 20.0

_TTFT = "time_to_first_token"
_ISL = "input_sequence_length"


def _paired(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """``(input tokens, ttft ms)`` for requests that reported both."""
    out: list[tuple[float, float]] = []
    for metrics in records:
        isl = metric_values([metrics], _ISL)
        ttft = metric_values([metrics], _TTFT)
        if isl and ttft:
            out.append((isl[0], ttft[0]))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def input_cost(workspace: str | Path | None) -> dict[str, Any] | None:
    """How much time-to-first-token grows per 100k input tokens, or ``None`` when the run cannot say.

    Read off two halves rather than a least-squares fit: TTFT carries a long tail of queued requests, and a median
    on each side of the input range is not moved by it. The halves are split at the median input length, so both
    hold the same number of requests whatever the workload's shape.
    """
    pairs = _paired(profiling_records(workspace))
    if len(pairs) < MIN_REQUESTS:
        return None
    pairs.sort()
    midpoint = len(pairs) // 2
    low, high = pairs[:midpoint], pairs[midpoint:]
    low_isl, high_isl = _median([p[0] for p in low]), _median([p[0] for p in high])
    span = high_isl - low_isl
    if span < MIN_ISL_SPAN:
        return None
    low_ttft, high_ttft = _median([p[1] for p in low]), _median([p[1] for p in high])
    slope = (high_ttft - low_ttft) / span * 100_000.0
    return {
        "ms_per_100k_tokens": round(slope, 1),
        "requests": len(pairs),
        "low_isl": int(low_isl),
        "low_ttft_ms": round(low_ttft, 1),
        "high_isl": int(high_isl),
        "high_ttft_ms": round(high_ttft, 1),
        "flagged": bool(slope >= FLAG_MS_PER_100K),
    }


def render_input_cost(cost: dict[str, Any] | None) -> list[str]:
    """Render the per-input-token cost, and what it means when the prefix is cached."""
    if not cost:
        return []
    line = (
        f"- front-end cost     : `{cost['ms_per_100k_tokens']:+.1f} ms` of TTFT per 100k input tokens "
        f"({cost['low_ttft_ms']:.0f} ms at {cost['low_isl'] // 1000}k → {cost['high_ttft_ms']:.0f} ms at "
        f"{cost['high_isl'] // 1000}k, {cost['requests']} requests)"
    )
    if not cost["flagged"]:
        return [line]
    return [
        line + " ⚠",
        "  *On a cached prefix the GPU does little work for those tokens, so growth this size is paid in front "
        "of the engine — template rendering, re-tokenising the whole prompt, pickling it to the scheduler.*",
    ]


__all__ = [
    "FLAG_MS_PER_100K",
    "MIN_ISL_SPAN",
    "MIN_REQUESTS",
    "input_cost",
    "render_input_cost",
]
