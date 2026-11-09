# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's GPU energy over its measured window, and whether that energy can be trusted.

The method and the reason codes are InferenceX's (``infx/results/power/single_node.py``), so a Hyperloom round and an
InferenceX run report the same figure for the same samples: each card's power is linearly interpolated at the
window's boundaries and integrated by trapezoid between them; a card whose samples do not bracket the window, or
leave a gap longer than :data:`MAX_SAMPLE_GAP_S` across it, has no trustworthy energy. The p75 and p90 power are
time-weighted quantiles of the summed curve, so the sampling cadence cannot bias them.

The cards' hardware energy accumulators are read at the window's boundaries and compared with the integral over the
same span; like InferenceX, a disagreement beyond :data:`ACCUMULATOR_TOLERANCE` is a warning and does not invalidate
the integral, which is the figure the samples support.

A window is a list of intervals on the monotonic clock: an inference_suite round has one measured interval, a GEAK
replay one per replica it boots.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ACCUMULATOR_TOLERANCE",
    "ENERGY_METHOD",
    "MAX_SAMPLE_GAP_S",
    "EnergyInputs",
    "TokenCounts",
    "Window",
    "counter_delta",
    "energy_block",
    "integrate",
]

MAX_SAMPLE_GAP_S = 3.0
ACCUMULATOR_TOLERANCE = 0.05
ENERGY_METHOD = "per_gpu_trapezoid_linear_boundary_interpolation"
_QUANTILES = {"p75": 0.75, "p90": 0.9}
#: InferenceX's precision: watts, joules and seconds to the milli, per-token and ratio figures to the micro.
_DIGITS = 3
_FINE_DIGITS = 6
_COUNT_MISMATCH = "expected_gpu_count_mismatch"
_BISECTIONS = 60
#: Reasons that leave the energy itself valid and null only the per-token figure they name.
_TOKEN_REASONS = {
    "successful_queries": "invalid_successful_query_count",
    "input_tokens": "invalid_input_token_count",
    "output_tokens": "invalid_output_token_count",
}

Series = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class TokenCounts:
    """What the measured window served, as the load's report counts it; ``None`` for a count it does not keep."""

    successful_queries: int | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class Window:
    """The measured window: monotonic intervals, with the wall-clock instants of its first start and last end."""

    phase: str
    intervals: tuple[tuple[float, float], ...]
    start_unix: float
    end_unix: float
    source: str

    @property
    def duration_s(self) -> float:
        return sum(end - start for start, end in self.intervals)

    def to_document(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "start_unix": round(self.start_unix, _DIGITS),
            "end_unix": round(self.end_unix, _DIGITS),
            "duration_s": round(self.duration_s, _DIGITS),
            "intervals": len(self.intervals),
            "source": self.source,
            "method": ENERGY_METHOD,
        }


@dataclass(frozen=True)
class EnergyInputs:
    """Everything the energy of one window is computed from.

    ``power`` and ``accumulator`` hold every sampled card's ``(mono, W)`` and ``(mono, J)`` readings, oldest first, a
    failed power read as ``None``. ``scope`` is the cards whose energy is the round's; ``serving`` the cards found
    holding a model. When ``expected_scope`` says the scope was configured rather than found, every scope card must be
    serving; a card outside the scope that holds a model is another server's and does not count.
    """

    window: Window | None
    power: Mapping[int, Sequence[tuple[float, float | None]]]
    accumulator: Mapping[int, Series]
    scope: tuple[int, ...]
    serving: tuple[int, ...]
    expected_scope: bool
    external_server: bool
    aborted: bool


def _append(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _interpolate(samples: Series, at: float) -> float:
    times = [t for t, _ in samples]
    right = bisect.bisect_left(times, at)
    if right < len(samples) and math.isclose(samples[right][0], at, rel_tol=0.0, abs_tol=1e-9):
        return samples[right][1]
    left_t, left_p = samples[right - 1]
    right_t, right_p = samples[right]
    return left_p + (at - left_t) / (right_t - left_t) * (right_p - left_p)


def _clipped(samples: Series, start: float, end: float) -> list[tuple[float, float]]:
    inner = [(t, p) for t, p in samples if start < t < end]
    return [(start, _interpolate(samples, start)), *inner, (end, _interpolate(samples, end))]


def _bracketed(samples: Series, start: float, end: float) -> bool:
    return len(samples) >= 2 and samples[0][0] <= start and samples[-1][0] >= end


def integrate(samples: Series, start: float, end: float) -> float | None:
    """The trapezoid integral of ``samples`` over ``[start, end]``; ``None`` when they do not bracket it."""
    if not _bracketed(samples, start, end) or end <= start:
        return None
    return sum((t1 - t0) * (p0 + p1) / 2.0 for (t0, p0), (t1, p1) in itertools.pairwise(_clipped(samples, start, end)))


def counter_delta(readings: Series, start: float, end: float, wrap: float | None = None) -> float | None:
    """How much a cumulative counter advanced over ``[start, end]``, interpolated at both ends.

    A counter that steps back has wrapped at ``wrap``; without one, the step back is a reset and the delta unread.
    """
    unwrapped: list[tuple[float, float]] = []
    offset = 0.0
    for (t, value), previous in zip(readings, [None, *readings]):
        if previous is not None and value < previous[1]:
            if wrap is None:
                return None
            offset += wrap
        unwrapped.append((t, value + offset))
    if not _bracketed(unwrapped, start, end):
        return None
    return _interpolate(unwrapped, end) - _interpolate(unwrapped, start)


def _max_gap(samples: Series, start: float, end: float) -> float:
    times = [t for t, _ in samples]
    left = bisect.bisect_right(times, start) - 1
    right = bisect.bisect_left(times, end)
    relevant = times[max(left, 0) : right + 1]
    return max((b - a for a, b in itertools.pairwise(relevant)), default=0.0)


@dataclass
class _Card:
    integral_j: float | None = 0.0
    accumulator_j: float | None = 0.0
    accumulator_span_j: float | None = 0.0
    max_gap_s: float = 0.0
    samples: int = 0
    accumulator_missing: str | None = None

    def to_document(self) -> dict[str, Any]:
        return {
            "integral_j": _round(self.integral_j, _DIGITS),
            "accumulator_delta_j": _round(self.accumulator_j, _DIGITS),
            "max_gap_s": round(self.max_gap_s, _DIGITS),
            "samples": self.samples,
        }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _card(
    readings: Sequence[tuple[float, float | None]], accumulator: Series, window: Window, reasons: list[str]
) -> _Card:
    samples = [(t, p) for t, p in readings if p is not None]
    card = _Card()
    for start, end in window.intervals:
        if any(p is None and start <= t <= end for t, p in readings):
            _append(reasons, "invalid_power_sample")
        card.samples += sum(1 for t, _ in samples if start <= t <= end)
        if len(samples) < 2:
            _append(reasons, "insufficient_power_samples")
            card.integral_j = None
            continue
        if not _bracketed(samples, start, end):
            _append(reasons, "benchmark_window_not_bracketed")
            card.integral_j = None
            continue
        card.max_gap_s = max(card.max_gap_s, _max_gap(samples, start, end))
        card.integral_j = _add(card.integral_j, integrate(samples, start, end))
        _add_accumulator(card, samples, accumulator, (start, end))
    if card.max_gap_s > MAX_SAMPLE_GAP_S:
        _append(reasons, "sampling_gap_exceeded")
    return card


def _add(total: float | None, value: float | None) -> float | None:
    return None if total is None or value is None else total + value


def _add_accumulator(card: _Card, samples: Series, accumulator: Series, interval: tuple[float, float]) -> None:
    """Add the accumulator's movement between the first reads at or after each boundary, and the integral over the
    same span, so the comparison is exact however late the reads came."""
    opened = next((read for read in accumulator if read[0] >= interval[0]), None)
    closed = next((read for read in accumulator if read[0] >= interval[1]), None)
    if opened is None or closed is None or closed[0] <= opened[0] or closed[1] < opened[1]:
        card.accumulator_j = card.accumulator_span_j = None
        card.accumulator_missing = (
            "missing_start_snapshot"
            if opened is None
            else "missing_end_snapshot"
            if closed is None or closed[0] <= opened[0]
            else "accumulator_went_back"
        )
        return
    card.accumulator_j = _add(card.accumulator_j, closed[1] - opened[1])
    card.accumulator_span_j = _add(card.accumulator_span_j, integrate(samples, opened[0], closed[0]))


def _segments(curves: Sequence[Series], start: float, end: float) -> list[tuple[float, float, float]]:
    """The summed curve over ``[start, end]`` as ``(low W, high W, seconds)`` linear pieces."""
    slopes: dict[float, float] = {start: 0.0, end: 0.0}
    total = 0.0
    for samples in curves:
        clipped = _clipped(samples, start, end)
        total += clipped[0][1]
        for (t0, p0), (t1, p1) in itertools.pairwise(clipped):
            slope = (p1 - p0) / (t1 - t0)
            slopes[t0] = slopes.get(t0, 0.0) + slope
            slopes[t1] = slopes.get(t1, 0.0) - slope
    pieces = []
    slope = 0.0
    for t0, t1 in itertools.pairwise(sorted(slopes)):
        slope += slopes[t0]
        following = total + slope * (t1 - t0)
        pieces.append((min(total, following), max(total, following), t1 - t0))
        total = following
    return pieces


def _quantile(pieces: Sequence[tuple[float, float, float]], quantile: float) -> float:
    lower, upper = min(low for low, _, _ in pieces), max(high for _, high, _ in pieces)
    target = sum(seconds for _, _, seconds in pieces) * quantile
    for _ in range(_BISECTIONS):
        value = lower + (upper - lower) / 2
        below = sum(
            seconds if value >= high else seconds * (value - low) / (high - low) if value > low else 0.0
            for low, high, seconds in pieces
        )
        upper, lower = (value, lower) if below >= target else (upper, value)
    return upper


def _scope_reasons(inputs: EnergyInputs, reasons: list[str]) -> None:
    if inputs.aborted:
        _append(reasons, "aborted")
    if not inputs.scope:
        _append(reasons, _COUNT_MISMATCH)
    if inputs.expected_scope and not set(inputs.scope) <= set(inputs.serving):
        _append(reasons, _COUNT_MISMATCH)
    missing = [gpu for gpu in inputs.scope if gpu not in inputs.power]
    if missing:
        _append(reasons, _COUNT_MISMATCH)


def _accumulator_check(cards: Mapping[int, _Card], warnings: list[str]) -> dict[str, Any] | None:
    for card in cards.values():
        if card.accumulator_missing is not None:
            _append(warnings, card.accumulator_missing)
    deltas = [card.accumulator_j for card in cards.values()]
    spans = [card.accumulator_span_j for card in cards.values()]
    accumulated = sum(value for value in deltas if value is not None)
    integrated = sum(value for value in spans if value is not None)
    if not cards or None in deltas or None in spans or accumulated <= 0:
        _append(warnings, "accumulator_check_unavailable")
        return None
    error = abs(accumulated - integrated) / accumulated
    if error > ACCUMULATOR_TOLERANCE:
        _append(warnings, "accumulator_mismatch")
    return {
        "accumulator_delta_j": round(accumulated, _DIGITS),
        "integrated_j": round(integrated, _DIGITS),
        "relative_error": round(error, _FINE_DIGITS),
        "tolerance": ACCUMULATOR_TOLERANCE,
        "within_tolerance": error <= ACCUMULATOR_TOLERANCE,
    }


def _power_metrics(
    window: Window, power: Mapping[int, Sequence[tuple[float, float | None]]], cards: Mapping[int, _Card]
) -> dict[str, float | None]:
    total = sum(card.integral_j or 0.0 for card in cards.values())
    n = len(cards)
    curves = [[(t, p) for t, p in power[gpu] if p is not None] for gpu in cards]
    pieces = [piece for start, end in window.intervals for piece in _segments(curves, start, end)]
    metrics: dict[str, float | None] = {
        "total_gpu_energy_j": round(total, _DIGITS),
        "avg_total_gpu_power_w": round(total / window.duration_s, _DIGITS),
        "avg_power_w": round(total / window.duration_s / n, _DIGITS),
    }
    for name, quantile in _QUANTILES.items():
        value = _quantile(pieces, quantile)
        metrics[f"{name}_total_gpu_power_w"] = round(value, _DIGITS)
        metrics[f"{name}_power_w"] = round(value / n, _DIGITS)
    return metrics


_POWER_KEYS = (
    "total_gpu_energy_j",
    "avg_power_w",
    "p75_power_w",
    "p90_power_w",
    "avg_total_gpu_power_w",
    "p75_total_gpu_power_w",
    "p90_total_gpu_power_w",
)
_PER_TOKEN_KEYS = {
    "successful_queries": "joules_per_successful_query",
    "input_tokens": "joules_per_input_token",
    "output_tokens": "joules_per_output_token",
}


def _per_token(energy: float | None, counts: TokenCounts | None, reasons: list[str]) -> dict[str, float | None]:
    values: dict[str, int | None] = {
        name: getattr(counts, name) if counts is not None else None for name in _PER_TOKEN_KEYS
    }
    out: dict[str, float | None] = {}
    for name, key in _PER_TOKEN_KEYS.items():
        count = values[name]
        valid = isinstance(count, int) and not isinstance(count, bool) and count > 0
        if not valid:
            _append(reasons, _TOKEN_REASONS[name])
        out[key] = round(energy / count, _FINE_DIGITS) if energy is not None and valid and count else None
    inputs, outputs = values["input_tokens"], values["output_tokens"]
    total = inputs + outputs if isinstance(inputs, int) and isinstance(outputs, int) and inputs + outputs > 0 else None
    out["joules_per_total_token"] = round(energy / total, _FINE_DIGITS) if energy is not None and total else None
    return out


def _counts_document(counts: TokenCounts | None) -> dict[str, int | None]:
    return {name: None if counts is None else getattr(counts, name) for name in _PER_TOKEN_KEYS}


def _external(inputs: EnergyInputs, counts: TokenCounts | None) -> dict[str, Any]:
    return {
        "window": None if inputs.window is None else inputs.window.to_document(),
        "per_gpu": {},
        **dict.fromkeys(_POWER_KEYS),
        "accumulator_check": None,
        "counts": _counts_document(counts),
        **dict.fromkeys(_PER_TOKEN_KEYS.values()),
        "joules_per_total_token": None,
        "power_valid": 0,
        "reasons": ["external_server"],
        "warnings": [],
    }


def energy_block(inputs: EnergyInputs, counts: TokenCounts | None) -> dict[str, Any]:
    """``power.json``'s ``energy`` block: the window's GPU energy, its power, J per token and whether it is valid.

    ``power_valid`` is 1 when the energy is; a missing or unusable token count nulls only the per-token figure it
    feeds, and its reason is listed beside the others.
    """
    if inputs.external_server:
        return _external(inputs, counts)
    reasons: list[str] = []
    warnings: list[str] = []
    window = inputs.window
    if window is None or not window.intervals or window.duration_s <= 0:
        _append(reasons, "invalid_benchmark_window")
    _scope_reasons(inputs, reasons)
    cards: dict[int, _Card] = {}
    if window is not None and window.duration_s > 0:
        cards = {
            gpu: _card(inputs.power[gpu], inputs.accumulator.get(gpu, ()), window, reasons)
            for gpu in inputs.scope
            if gpu in inputs.power
        }
    valid = not reasons and window is not None
    check = _accumulator_check(cards, warnings) if valid else None
    metrics = _power_metrics(window, inputs.power, cards) if valid and window else dict.fromkeys(_POWER_KEYS)
    per_token = _per_token(metrics["total_gpu_energy_j"], counts, reasons)
    return {
        "window": None if window is None else window.to_document(),
        "per_gpu": {str(gpu): card.to_document() for gpu, card in cards.items()},
        **metrics,
        "accumulator_check": check,
        "counts": _counts_document(counts),
        **per_token,
        "power_valid": int(valid),
        "reasons": reasons,
        "warnings": warnings,
    }
