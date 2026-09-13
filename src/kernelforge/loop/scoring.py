# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single KEEP/REVERT policy and reported-result invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import statistics

KEEP_MEASUREMENT_COUNT = 3

# The KEEP bar, as the one-sided 95% Student-t critical value for the degrees of freedom the sigma estimate actually
# carries.
KEEP_T_CRITICAL: Mapping[int, float] = {2: 2.920, 5: 2.015, 8: 1.860}

# Floor under the margin, as a fraction of the incumbent, so a freak run of three near-identical measurements cannot
# drive the bar to zero.
KEEP_MIN_MARGIN_FRACTION = 0.001

# A scored case is called *dominant* when it supplies more of the objective's variance than every other scored case
# combined.
SIGMA_DOMINANCE_VARIANCE_SHARE = 0.5

# The second half of the pathology: the dominant contributor is also cheap, so its spread is an artifact of timing a
# 10 us dispatch rather than a property of the work the campaign is optimising.
SIGMA_DOMINANCE_WALL_SHARE_OF_EQUAL = 1.0

# An absolute ceiling under the equal-share rule, because the equal share stops meaning "cheap" as N falls.
SIGMA_DOMINANCE_WALL_SHARE_CAP = 0.25

# Extra measurements bought per re-measure round, and the number of rounds.
SIGMA_REMEASURE_BATCH = 3
SIGMA_REMEASURE_MAX_ROUNDS = 2

# The cheap in-session parity probe: bf16 with fp32 accumulation is not bit-exact, so this is judged on
# signal-to-noise rather than allclose.
DEFAULT_SNR_THRESHOLD_DB = 30.0

# The one description of the gate every kernel backend prompt renders, so an agent's self-check and forge's acceptance
# decision cannot drift apart.
CANONICAL_GATE_PROMPT = f"""\
SNR >= {DEFAULT_SNR_THRESHOLD_DB:g} dB is a fast pre-filter, NOT the gate. A KEEP is decided by the
task's own `compile_command` and then its `correctness_command`, both from its
`config.yaml`, which forge runs on every candidate it would otherwise accept,
whether the loop kept it or a warm start adopted it from the knowledge base, and
whose tolerances are the task's, not forge's. Run both yourself before you
propose a change: a candidate that clears SNR and fails either is reverted, and
the error it raised is the only thing that tells you what to fix. The
`compile_command` may build a different, smaller shape than the one you measure,
so a guard you add for the shape you tested can still reject it there."""


def measurement_sigma(measurement_scores: Sequence[float]) -> float | None:
    """Return the spread of one candidate's independent pristine-relative scores."""
    if len(measurement_scores) < 2:
        return None
    return statistics.stdev(float(score) for score in measurement_scores)


@dataclass(frozen=True)
class SigmaAttribution:
    """How one candidate's objective sigma splits across the cases it was scored on."""

    case_sigmas: Mapping[str, float]
    variance_shares: Mapping[str, float]
    wall_shares: Mapping[str, float]
    dominant_case: str | None
    sample_size: int

    @property
    def total_variance(self) -> float:
        """The objective variance the independent per-case model accounts for."""
        return sum(sigma * sigma for sigma in self.case_sigmas.values())


def attribute_sigma(
    case_series: Mapping[str, Sequence[float]],
    baseline_case_times: Mapping[str, float],
) -> SigmaAttribution | None:
    """Blame the objective's spread on the cases that produced it."""
    scored = [case_id for case_id in sorted(baseline_case_times) if case_id in case_series]
    if not scored or len(scored) != len(baseline_case_times):
        return None
    count = len(scored)
    sizes = {len(tuple(case_series[case_id])) for case_id in scored}
    if len(sizes) != 1 or min(sizes) < 2:
        return None
    sample_size = sizes.pop()

    case_sigmas: dict[str, float] = {}
    mean_times: dict[str, float] = {}
    for case_id in scored:
        baseline = float(baseline_case_times[case_id])
        times = [float(value) for value in case_series[case_id]]
        if baseline <= 0.0 or not math.isfinite(baseline):
            return None
        if any(not math.isfinite(t) or t <= 0.0 for t in times):
            return None
        terms = [baseline / t / count for t in times]
        case_sigmas[case_id] = statistics.stdev(terms)
        mean_times[case_id] = statistics.fmean(times)

    total_variance = sum(sigma * sigma for sigma in case_sigmas.values())
    total_time = sum(mean_times.values())
    if total_variance <= 0.0 or total_time <= 0.0:
        return None

    variance_shares = {case_id: (sigma * sigma) / total_variance for case_id, sigma in case_sigmas.items()}
    wall_shares = {case_id: mean_times[case_id] / total_time for case_id in scored}
    equal_share = min(
        SIGMA_DOMINANCE_WALL_SHARE_OF_EQUAL / count,
        SIGMA_DOMINANCE_WALL_SHARE_CAP,
    )
    dominant = max(variance_shares, key=lambda case_id: variance_shares[case_id])
    if variance_shares[dominant] <= SIGMA_DOMINANCE_VARIANCE_SHARE or wall_shares[dominant] >= equal_share:
        dominant = None
    return SigmaAttribution(
        case_sigmas=case_sigmas,
        variance_shares=variance_shares,
        wall_shares=wall_shares,
        dominant_case=dominant,
        sample_size=sample_size,
    )


def rescaled_sigma(
    observed_sigma: float,
    base: SigmaAttribution,
    extended: SigmaAttribution,
) -> float:
    """Re-estimate the objective's sigma from a larger per-case sample."""
    base_variance = base.total_variance
    extended_variance = extended.total_variance
    if base_variance <= 0.0 or extended_variance <= 0.0:
        return float(observed_sigma)
    return float(observed_sigma) * math.sqrt(extended_variance / base_variance)


def keep_t_critical(sample_size: int) -> float:
    """The one-sided 95% t value for a sigma estimated from ``sample_size`` samples."""
    df = int(sample_size) - 1
    earned = [key for key in KEEP_T_CRITICAL if key <= df]
    return KEEP_T_CRITICAL[max(earned)] if earned else KEEP_T_CRITICAL[min(KEEP_T_CRITICAL)]


def required_keep_speedup(
    best_mean_case_speedup: float,
    measurement_scores: Sequence[float],
    *,
    sigma: float | None = None,
    sigma_sample_size: int | None = None,
) -> float:
    """Return the mean pristine-relative score required for the next KEEP."""
    best = float(best_mean_case_speedup)
    floor = best * KEEP_MIN_MARGIN_FRACTION
    spread = measurement_sigma(measurement_scores) if sigma is None else float(sigma)
    count = len(measurement_scores)
    if spread is None or count < 2:
        return best + floor
    samples = count if sigma_sample_size is None else int(sigma_sample_size)
    standard_error = spread / math.sqrt(count)
    return best + max(keep_t_critical(samples) * standard_error, floor)


def passes_keep_threshold(
    measurement_scores: list[float],
    *,
    best_mean_case_speedup: float,
    sigma: float | None = None,
    sigma_sample_size: int | None = None,
) -> bool:
    """Require the mean of the independent scores to clear the threshold."""
    if len(measurement_scores) != KEEP_MEASUREMENT_COUNT:
        return False
    required = required_keep_speedup(
        best_mean_case_speedup,
        measurement_scores,
        sigma=sigma,
        sigma_sample_size=sigma_sample_size,
    )
    return statistics.fmean(measurement_scores) >= required


def aggregate_regression_detail(
    *,
    baseline_ms: float | None,
    best_ms: float | None,
    mean_case_speedup: float | None,
) -> str:
    """Name the contradiction when a claimed improvement is slower overall."""
    if not mean_case_speedup or float(mean_case_speedup) <= 1.0:
        return ""
    if baseline_ms is None or best_ms is None:
        return ""
    baseline = float(baseline_ms)
    best = float(best_ms)
    if best < baseline:
        return ""
    return (
        f"reported mean case speedup {float(mean_case_speedup):.6f}x but the "
        f"best raw mean {best:g} ms is not faster than the pristine baseline "
        f"{baseline:g} ms"
    )


def warm_start_improvement_flags(
    *,
    pristine_ms: float | None,
    best_ms: float | None,
    mean_case_speedup: float | None,
) -> dict[str, str | bool]:
    """Derive what a validated warm start may claim from what it measured."""
    aggregate_regression = aggregate_regression_detail(
        baseline_ms=pristine_ms,
        best_ms=best_ms,
        mean_case_speedup=mean_case_speedup,
    )
    improved = bool(mean_case_speedup and float(mean_case_speedup) > 1.0) and not aggregate_regression
    return {
        "aggregate_regression": aggregate_regression,
        "improved": improved,
        "total_improved": improved,
    }


def keep_score(measurement_scores: list[float]) -> float | None:
    """Persist the mean of the independent measurements as the best score."""
    return statistics.fmean(measurement_scores) if measurement_scores else None


def beats_current_best(
    score: float | None,
    *,
    best_mean_case_speedup: float | None,
) -> bool:
    """Whether a candidate was faster than the incumbent, threshold aside."""
    if score is None:
        return False
    incumbent = 1.0 if best_mean_case_speedup is None else float(best_mean_case_speedup)
    return float(score) > incumbent
