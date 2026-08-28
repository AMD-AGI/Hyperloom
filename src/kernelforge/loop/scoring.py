# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single KEEP/REVERT policy and reported-result invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import statistics

KEEP_MEASUREMENT_COUNT = 3

# The KEEP bar, as the one-sided 95% Student-t critical value for the degrees of
# freedom the sigma estimate actually carries. The bar sits
# ``t(df) * sigma / sqrt(n)`` over the incumbent -- the standard error of the
# mean of the ``n`` scores the protocol took, not the per-measurement spread.
#
# The constant this replaces, ``KEEP_MARGIN_SIGMAS = 3.0``, was derived as
# exactly this value: with KEEP_MEASUREMENT_COUNT measurements df = 2, and the
# one-sided 95% Student-t value is 2.920, rounded up. But it was then applied by
# `passes_keep_threshold` to *every* score rather than to their mean, against an
# incumbent that `keep_score` had itself taken as a minimum.
# Requiring ``min(scores) >= min(incumbent_scores) + 3 sigma`` asks, in
# expectation, for a true mean gain of 3 sigma: both minima carry the same
# downward offset and it cancels. The t test the 2.920 came from asks for
# 2.920 / sqrt(3) = 1.686 sigma. The rule as written was therefore 1.78x
# stricter than its own derivation, and that factor was nowhere recorded.
#
# The replay that set k = 3 could not have seen this, because both of its
# metrics are one-sided. It scored rules by the false-accept rate on a zero-gain
# candidate, which any stricter rule wins by construction; and it scored
# recoveries as "certainly real" by whether they were >= 3 sigma gains, so a
# 3 sigma rule was validated against 3 sigma as its own ground truth. Neither
# metric measures power, so neither could report a rule that was too strict.
#
# Replaying the 2026-08-24 batch -- 9 kernels, 70 REVERT_PERF decisions -- with
# the incumbent moved onto the mean as well, so both sides of the comparison are
# the same statistic: 27 of those 70 measured a faster mean than the incumbent
# and were rejected anyway, and 13 of the 27 clear the bar under this rule. The
# largest are +5.5% on dsa_sparse_mla (t = 4.06), +1.7% on gdn_linear_attn
# (t = 3.15) and +0.9% on gqa_sparse_attn_prefill (t = 3.53, df = 8); the
# smallest recovered is +0.25% at t = 5.24. Under the null those 27 would have
# yielded about 1.4 passes at this level, so what the change admits is
# overwhelmingly real gain rather than noise. Holding the incumbent at the
# minimum the old rule published -- the same replay without that correction --
# recovers 23, which is the optimistic bound and not the one claimed here.
#
# Tabulated per df rather than fixed at 2.920 because `rescaled_sigma` can
# estimate sigma from a larger per-case sample: SIGMA_REMEASURE_BATCH extra
# measurements per round over SIGMA_REMEASURE_MAX_ROUNDS rounds reach 6 and then
# 9 samples per case -- the extra benches run the whole suite, so every scored
# case reaches the same count and the df is exact rather than pooled. Charging a
# 9-sample estimate the df = 2 value would take the estimator's cost and leave
# its benefit on the table. Note what this is not: re-running the replay above
# with the df pinned at 2 recovers the same 13 candidates, so the table earns
# nothing on that batch and no claim here rests on it. It is correctness for a
# path that fired 10 times in 159 decisions, not a source of the gain.
KEEP_T_CRITICAL: Mapping[int, float] = {2: 2.920, 5: 2.015, 8: 1.860}

# Floor under the margin, as a fraction of the incumbent, so a freak run of three
# near-identical measurements cannot drive the bar to zero. Across the same 1007
# measurement groups the smallest relative sigma seen was 0.002% and the bottom
# 1% sit near 0.015%. The floor was originally set at 0.05% because that is where
# 3 sigma lands on that bottom 1%, leaving the rest adaptive.
#
# What the floor answers is not "is this gain real". Where it binds the candidate
# has already cleared the t test by a wide margin -- on a kernel repeating to
# 0.02%, a 0.1% gain is t = 8.7 -- so the question it settles is whether a gain
# that small is worth spending a KEEP on. That is a policy call, and 0.05% was
# too low a one: it let the loop ratchet on gains too small to matter and lock
# the campaign into the local optimum they came from.
#
# Raised to 0.1% on the 2026-08-24 batch, which reconstructs 100 bench decisions
# whose incumbent can be recovered from the logged bar. Relative sigma there runs
# min 0.009%, p25 0.076%, median 0.141%, p90 0.560%, max 2.280%, so the t term
# already covers the bulk of the distribution and the floor governs only the
# quiet tail. Sweeping it against that batch:
#
#     floor    candidates kept    vs 0.05%    decisions the floor decides
#     0.05%          76              --                9%
#     0.10%          76              +0               20%
#     0.20%          73              -3               39%
#     0.50%          64             -12               71%
#     1.00%          52             -24                --
#
# 0.1% doubles the floor's reach at zero cost -- no candidate in the batch landed
# a gain between 0.05% and 0.1% -- which is why the step stops there. Past 0.2%
# it starts refusing real gains, and by 0.5% the floor has taken over 71% of the
# decisions, which would retire the t test rather than back it up. Going higher
# needs its own replay across more than one batch; the zero-loss claim above is
# the only one this constant rests on, and 100 decisions is a thin basis for
# anything stronger.
#
# This is the one place the KEEP change is *stricter* than the rule it replaces,
# on top of the same 0.015% now drawing a noise term of t * sigma / sqrt(3) =
# 0.025% where it used to draw 3 * sigma = 0.045%. The band it governs -- roughly
# sub-0.06% relative sigma -- is where a claimed gain is least distinguishable
# from an unmodelled systematic, so holding the floor above it is the safe
# direction.
KEEP_MIN_MARGIN_FRACTION = 0.001

# A scored case is called *dominant* when it supplies more of the objective's
# variance than every other scored case combined. The threshold is a majority
# rather than a tuned number: with N scored cases an equal contribution is 1/N,
# and only above 1/2 does one case's spread decide the bar by itself, so it is
# also the only point at which re-measuring one case can move the bar. On the
# 2026-08 GQA campaign the 10 us `q61` case supplied a median 87% of the
# objective's sigma across the three-measurement groups of all 23 candidates,
# and the bar it drew ranged from 0.32% to 8.42% of the incumbent.
SIGMA_DOMINANCE_VARIANCE_SHARE = 0.5

# The second half of the pathology: the dominant contributor is also cheap, so
# its spread is an artifact of timing a 10 us dispatch rather than a property of
# the work the campaign is optimising. This threshold is derived rather than
# chosen -- the case carries less than its equal 1/N share of the suite's
# measured wall time, so it is expressed as a multiple of that equal share.
# `q61` sat at 5.7% of a three-case suite against an equal share of 33.3%,
# i.e. 0.17 of its share. A case that dominates sigma *and* carries the wall
# time is the objective's real noise and is deliberately left alone.
SIGMA_DOMINANCE_WALL_SHARE_OF_EQUAL = 1.0

# An absolute ceiling under the equal-share rule, because the equal share stops
# meaning "cheap" as N falls. On a two-case suite 1/N is 50%, so a case holding
# nearly half the suite's wall time would qualify as the cheap one and buy a
# re-measure costing nearly half a bench -- which is the opposite of the
# pathology the guard exists for, since at that size the spread is the work's
# and not a timing artifact. Two-case suites are not hypothetical: the 2026-08
# MoE benchmark is one.
#
# Across 968 variance-dominant candidates in the archive this refuses 20 of
# them, all on three-case suites where the dominant case sat between 25% and
# 33.3%. It refuses none of the 96 observed two-case candidates, because there
# the variance-dominant case is usually the *expensive* one -- median wall share
# 79.1% -- which the equal-share rule already refuses. So this is a bound on a
# reachable state rather than a fix for an observed one, and it is cheap: the
# candidates it declines are the ones whose re-measure would have cost the most.
SIGMA_DOMINANCE_WALL_SHARE_CAP = 0.25

# Extra measurements bought per re-measure round, and the number of rounds. The
# sample standard deviation of n samples has relative standard error
# 1/sqrt(2(n-1)): 50% at n = 3, 35% at n = 5, 25% at n = 9. Two rounds of three
# take the dominant case from 3 to 9 samples and halve the scatter of the
# estimator itself, which is what makes the bar a per-candidate lottery. A third
# round would move 25% to 21% for another whole-suite bench, so the loop stops
# at two. The bound is what makes the loop terminate: a pathologically noisy
# case cannot buy more than SIGMA_REMEASURE_MAX_ROUNDS benches.
#
# PROVISIONAL. Unlike KEEP_T_CRITICAL and KEEP_MIN_MARGIN_FRACTION above,
# this number has not been replayed against historical runs: the argument for
# it is the standard error above plus a cost ceiling, not a measured
# false-accept rate. It is the one constant here that trades campaign
# throughput for estimator quality -- a contender on a suite with a cheap
# dominant case pays up to 3x the bench cost of an ordinary iteration -- and
# the trade has never been measured. It needs the same replay treatment the
# margin constants carry before it can be called derived.
SIGMA_REMEASURE_BATCH = 3
SIGMA_REMEASURE_MAX_ROUNDS = 2

# The cheap in-session parity probe: bf16 with fp32 accumulation is not
# bit-exact, so this is judged on signal-to-noise rather than allclose. It is a
# pre-filter and a diagnostic only. A KEEP, and every other route to becoming
# this run's incumbent, is decided by the task's own correctness suite (see
# loop/canonical_correctness.py), whose tolerances differ per task and which no
# single global dB figure can stand in for.
DEFAULT_SNR_THRESHOLD_DB = 30.0

# The one description of the gate every kernel backend prompt renders, so an agent's
# self-check and forge's acceptance decision cannot drift apart.
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
    """Return the spread of one candidate's independent pristine-relative scores.

    The *sample* standard deviation, dividing by n-1. At n = 3 the population
    form divides by 3 instead of 2 and systematically understates the spread; a
    simulation using it produced a higher false-accept rate at k = 2 than at
    k = 3, which is incoherent. ``None`` when fewer than two measurements were
    taken, because then the spread was not measured at all.
    """
    if len(measurement_scores) < 2:
        return None
    return statistics.stdev(float(score) for score in measurement_scores)


@dataclass(frozen=True)
class SigmaAttribution:
    """How one candidate's objective sigma splits across the cases it was scored on.

    The objective is the equal-weight mean of per-case speedups, so measurement
    ``i`` scores ``s_i = (1/N) * sum_c baseline_c / t_(c,i)`` and each case
    contributes the term ``baseline_c / t_(c,i) / N`` additively. ``case_sigmas``
    is the sample spread of that term, so it is already in the units of the
    objective and comparable between cases of wildly different cost.

    ``variance_shares`` treats the cases as independent -- at three measurements
    an empirical covariance is not estimable, and the independent model is the
    conservative one for the question asked here, since correlated cases would
    only concentrate the blame further on whichever case moves most.
    """

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
    """Blame the objective's spread on the cases that produced it.

    ``case_series`` is one scored case's measured times across every
    independent measurement taken of this candidate, which is exactly what
    ``bench_result["measurements"][i]["case_times"]`` already carries; only
    cases present in ``baseline_case_times`` are considered, so unscored cases
    are excluded here for the same reason they are excluded from the mean.

    A case is named ``dominant_case`` when it clears both halves of the
    documented pathology: it supplies a majority of the objective's variance
    (:data:`SIGMA_DOMINANCE_VARIANCE_SHARE`) *and* it carries less than its
    equal share of the suite's measured wall time
    (:data:`SIGMA_DOMINANCE_WALL_SHARE_OF_EQUAL`, capped by
    :data:`SIGMA_DOMINANCE_WALL_SHARE_CAP` so a small suite's equal share
    cannot admit an expensive case). Both are required. A case that is noisy
    because it is the expensive one is the objective's real noise;
    re-measuring it buys nothing and costs the most.

    Returns ``None`` when the split cannot be established at all -- fewer than
    two measurements, a case missing from a measurement, a non-positive time,
    or a total variance of zero. The caller then keeps today's aggregate sigma,
    which is the whole no-op guarantee: attribution never *replaces* the
    measured sigma, it only says whether one case is worth re-measuring.
    """
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
    """Re-estimate the objective's sigma from a larger per-case sample.

    ``observed_sigma`` is what the KEEP measurements actually showed and stays
    the anchor: this returns it scaled by the square root of the ratio of the
    per-case variance the extended sample accounts for to the variance the
    original three measurements accounted for. Anchoring rather than
    substituting keeps whatever correlation the aggregate carried and means the
    result is the *same* statistic, estimated from more data, rather than a
    different one.

    The scale can be greater than one. Re-measuring reduces the sampling error
    of sigma, not the noise itself, so a case whose three-sample draw happened
    to be low comes back with a larger spread and a *higher* bar. That is the
    honest direction and is not suppressed: this changes how well the bar is
    estimated, never which side of it a rule sits on.
    """
    base_variance = base.total_variance
    extended_variance = extended.total_variance
    if base_variance <= 0.0 or extended_variance <= 0.0:
        return float(observed_sigma)
    return float(observed_sigma) * math.sqrt(extended_variance / base_variance)


def keep_t_critical(sample_size: int) -> float:
    """The one-sided 95% t value for a sigma estimated from ``sample_size`` samples.

    Falls back to the largest tabulated df at or below the one requested, so an
    unlisted sample size is charged a *more* conservative value than it earned
    rather than an interpolated one. Below the smallest tabulated df the sample
    does not support a t statistic at all and the df = 2 value stands.
    """
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
    """Return the mean pristine-relative score required for the next KEEP.

    The bar follows the candidate's own noise rather than a fixed step, because
    measurement noise varies by more than an order of magnitude between kernels:
    a 0.3% gain is certain on one that repeats to 0.022% and indistinguishable
    from noise on one that spreads over 0.281%. There is no upper bound on what
    a candidate may claim; the only question this asks is whether its mean
    out-measured the standard error of that mean.

    ``sigma`` overrides the estimate taken from ``measurement_scores`` alone,
    and ``sigma_sample_size`` reports how many samples per case that override
    was built from, so the critical value can be charged the df it earned. The
    override exists because three aggregate scores are a poor estimator of that
    same sigma when one cheap case supplies most of it, and
    :func:`rescaled_sigma` can estimate it from more measurements of that case.
    Omitting both reproduces the df = 2 bar over the protocol's own scores.

    The floor is unchanged and still relative to the incumbent, so a freak run
    of near-identical measurements cannot drive the bar to zero.
    """
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
    """Require the mean of the independent scores to clear the threshold.

    A one-sided Student-t test of the candidate's mean against the incumbent, at
    the critical value :data:`KEEP_T_CRITICAL` charges for the sigma estimate's
    own degrees of freedom. The comparison is mean against mean --
    :func:`keep_score` publishes the same statistic for the incumbent -- so a
    low draw is charged once, through the mean it lowers and the spread it
    widens, and not a second time by also standing in as the candidate's score.

    ``sigma`` and ``sigma_sample_size`` are forwarded to
    :func:`required_keep_speedup`. The scores that must clear the bar remain the
    ``KEEP_MEASUREMENT_COUNT`` scores the KEEP protocol took: any measurement
    bought to sharpen sigma informs the bar and is never itself admitted as
    evidence of a gain.
    """
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
    """Name the contradiction when a claimed improvement is slower overall.

    KEEP is decided on the equal-weight mean of per-case speedups while these
    are aggregate wall times, so a few winning cheap cases can outvote one
    collapsing expensive case and still score above 1.0. Returns "" when the two
    agree, when no improvement was claimed, or when either wall time is unknown.
    """
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
    """Derive what a validated warm start may claim from what it measured.

    A warm start publishes three artifacts -- the manifest, the caller's result
    JSON and the recovery checkpoint -- from one adoption, and only the manifest
    derived its badge from the aggregate invariant; the other two asserted an
    improvement outright. This is the single derivation all three share, and it
    is pure so the claim can be tested without a live campaign.
    """
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
    """Persist the mean of the independent measurements as the best score.

    This is the statistic :func:`passes_keep_threshold` tests and the incumbent
    it is tested against, so both sides of every later comparison are the same
    estimator. It was previously the minimum, which read as the conservative
    choice but was not one: incumbent and challenger both carried the same
    downward offset, so taking it raised no bar, while the challenger's own low
    draw was charged twice -- once as its score, and again through the sigma
    that set the margin above it.

    Every score reaching this policy is built by
    :func:`~kernelforge.mcp_server.tools.bench.calculate_mean_case_speedup`,
    which raises ``CaseCoverageError`` on a non-finite or non-positive timing
    before it divides. Nothing here re-checks finiteness: a NaN promoted into
    the incumbent would deadlock the run, and that invariant is what keeps one
    from being constructed.
    """
    return statistics.fmean(measurement_scores) if measurement_scores else None


def beats_current_best(
    score: float | None,
    *,
    best_mean_case_speedup: float | None,
) -> bool:
    """Whether a candidate was faster than the incumbent, threshold aside.

    Separates the two outcomes a REVERT conflates: a regression, and a real gain
    that landed inside the band between the incumbent and the threshold
    :func:`required_keep_speedup` sets above it. ``score`` is
    :func:`keep_score`, so a strict win here means the mean of the independent
    measurements beat the incumbent's own mean. An unmeasured candidate fails
    closed: it demonstrated nothing, so it cannot claim a gain.

    A missing incumbent is the pristine 1.0, the same reading the KEEP gate's
    callers apply. Treating it as "no gain possible" would blacklist exactly the
    candidates this separation exists for: before the first KEEP, every real
    gain over pristine is still below the threshold.
    """
    if score is None:
        return False
    incumbent = 1.0 if best_mean_case_speedup is None else float(best_mean_case_speedup)
    return float(score) > incumbent
