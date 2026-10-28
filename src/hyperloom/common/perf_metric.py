# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading: an E2E-normalised-interactivity objective guarded by per-chip throughput."""

from __future__ import annotations

from hyperloom.common.benchmark_config import configured_benchmark, is_magpie_agentx

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_float, env_str

INTVTY_V1 = "intvty_v1"

_AGENTX_ENV = "HYPERLOOM_AGENTX"

# The value ``SharedState.benchmark_mode`` carries for an AgentX session, stamped at seed so it outlives the shell
# that started the run.
_AGENTX_MODE = "agentx"

# The ``SharedState`` attribute holding the baseline's graded-axis snapshot.
_BASELINE_PERF_ATTR = "baseline_perf"

# Graded axis names, which are also the snapshot keys they are read from. InferenceX publishes a 2-D Pareto frontier
# with interactivity on x and per-chip throughput on y, and no fixed interactivity target, so trading one for the
# other moves a point along the frontier rather than violating a constraint.
GRADED_INTVTY = "e2e_norm_intvty_p90"
GRADED_INTVTY_P50 = "e2e_norm_intvty_p50"
GRADED_TOTAL = "total_throughput"
GRADED_OUTPUT = "output_throughput"

# Every percentile of the interactivity family. A consumer asking "was this graded on interactivity" must read this
# rather than one axis name, or it silently answers no the next time the graded percentile moves.
INTVTY_OBJECTIVES = (GRADED_INTVTY, GRADED_INTVTY_P50)

# The y axis InferenceX plots the frontier on. Reported, not graded: tensor parallelism is fixed for a session, so
# dividing both sides of a ratio by it leaves the guard's verdict unchanged.
GRADED_OUTPUT_PER_GPU = "output_tput_per_gpu"

# Comparability inputs: a pair is comparable only when both replayed a window of the same length, and a rate that
# rose because more requests failed is not a win.
GRADED_DURATION = "duration_seconds"
GRADED_ERROR_RATE = "request_error_rate"
# Requests left unserved (a transport failure or a zero-token reply) and requests sent over the measured rounds. A
# short reply the client counted failed is not unserved; its quality is the accuracy check's question.
GRADED_UNSERVED_REQUESTS = "graded_unserved_requests"
GRADED_REQUESTS_TOTAL = "graded_requests_total"

# A trace replay slices a different part of the corpus when the window moves, so the two rounds stop measuring the
# same work. Sized to catch a truncated round, not the few percent a full round drifts by.
DURATION_DRIFT_PCT = 5.0

_PERCENT = 100.0

# Extra failed requests a candidate may report over its anchor, in the rate's percentage points. aiperf counts a reply
# that carries no content as failed, and a corpus turn answered with a single token comes back with content under one
# configuration and without it under another, so with no slack that one turn refuses a real gain. Being a rate, the
# slack admits more failures in a longer or more concurrent round and none in a round of fewer than 200 requests (an
# MLPerf smoke run issues 150 samples). Percentiles are read over the same round's successful requests, so dropping a
# fraction f of them moves any percentile by at most f in rank, and the slack cannot buy a median gain anywhere near
# AGENTX_KEEP_P50_THRESHOLD_PCT.
ERROR_RATE_SLACK_PCT = 0.5

# The axes ``graded_axes_of`` can carry, for a consumer that must publish all of them including the ones a
# measurement did not supply. Absent and null are not the same fact: a recorder that omits an axis leaves a reader
# unable to tell an unmeasured axis from one the framework failed to report, and zero reads as "measured, and it
# was zero".
#
# Duration and error rate are members because they are decision inputs, not decoration: ``incomparability_reason``
# refuses a pair whose windows differ by more than ``DURATION_DRIFT_PCT`` or whose candidate dropped too many requests,
# and a published verdict that omits them cannot be re-derived from the record. The objective and its two guards
# are here for the same reason -- every input the verdict reads is recoverable from one block. The exception is the
# unserved and total request counts the guard prefers when both sides carry them: they join once a ``main`` producer
# stamps them, together with dropping the error-rate fallback in ``_failure_rates_pct``.
GRADED_AXIS_KEYS = (
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_TOTAL,
    GRADED_OUTPUT_PER_GPU,
    "input_throughput",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    GRADED_DURATION,
    GRADED_ERROR_RATE,
)

# Upstream reports run-to-run noise on this workload as 1-5% depending on the concurrency regime, so the band opens
# to the top of that range instead of rejecting movement upstream would call noise.
_DEFAULT_INTVTY_NOISE_PCT = 5.0

# The median bar is a property of the objective, so the session's decaying threshold does not apply to it.
AGENTX_KEEP_P50_THRESHOLD_PCT = 3.0

VERDICT_KEEP = "KEEP"
VERDICT_REVERT = "REVERT"


def agentx_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether CLI workload translation or a bound YAML selects AgentX."""
    return is_magpie_agentx(configured_benchmark(env)) or env_bool(_AGENTX_ENV, env=env)


def is_agentx_mode(benchmark_mode: Any) -> bool:
    """Whether a ``benchmark_mode`` names the agentic workload."""
    return str(benchmark_mode or "").strip().lower() == _AGENTX_MODE


def agentx_active(*, benchmark_mode: Any = "") -> bool:
    """Whether either persisted workload identity or the environment enables AgentX."""
    return agentx_enabled() or is_agentx_mode(benchmark_mode)


def intvty_grading_enabled(*, benchmark_mode: str = "") -> bool:
    """True when interactivity grading applies; ``benchmark_mode`` is a parameter to keep this module a leaf."""
    # Passing the mode matters: the env var describes only the shell that happens to be running, so a re-baseline or
    # integrate round in a subprocess would otherwise grade an agentic measurement on the synthetic axis.
    # The MLPerf harness publishes no per-request OSL/E2EL series, so its sessions grade on output throughput;
    # asking for interactivity there would only degrade every round. Adding that series upstream in
    # mlcommons/endpoints is future work, if MLPerf mandates grading on interactivity.
    from hyperloom.common.agentx_workload import is_mlperf_backend

    if is_mlperf_backend():
        return False
    raw = env_str("HYPERLOOM_PERF_METRIC").strip().lower()
    if raw:
        return raw == INTVTY_V1
    if agentx_enabled():
        return True
    return is_agentx_mode(benchmark_mode)


def intvty_serving_grading_enabled(*, scriptable: bool = False, benchmark_mode: str = "") -> bool:
    """Interactivity grading, limited to non-scriptable serving runs, which have no interactivity axis."""
    return intvty_grading_enabled(benchmark_mode=benchmark_mode) and not scriptable


def graded_metric_key(*, benchmark_mode: str = "") -> str:
    """The curve-row field a session's speedups are measured on."""
    if intvty_grading_enabled(benchmark_mode=benchmark_mode):
        return GRADED_INTVTY
    return GRADED_OUTPUT


def parse_intvty_noise_pct() -> float:
    """Noise band in percent from ``HYPERLOOM_PERF_NOISE_PCT``."""
    return env_float("HYPERLOOM_PERF_NOISE_PCT", _DEFAULT_INTVTY_NOISE_PCT)


def _positive(value: Any) -> float | None:
    """Coerce to a strictly positive float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coerced = float(value)
    return coerced if coerced > 0 else None


def _non_negative(value: Any) -> float | None:
    """Coerce to a float of zero or more, else None.

    Zero is a measured error rate, not a missing one, so ``_positive`` would read a flawless run as unreported.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coerced = float(value)
    return coerced if coerced >= 0 else None


_AGENTX_EVIDENCE = ("benchmark_backend", "benchmark_valid", "diagnostic_only", "workload_identity")


def perf_snapshot_from_mapping(source: Mapping[str, Any] | None) -> dict[str, float] | None:
    """The graded axes from a measurement or a ``current_best``; None unless objective and guards are all positive."""
    # Requiring all of them is what stops a lane half-applying the objective. A total that is absent, null or
    # non-positive coalesces to input plus output, the same fallback ``agentx.mapping`` applies.
    if not isinstance(source, Mapping):
        return None
    intvty = _positive(source.get(GRADED_INTVTY))
    intvty_p50 = _positive(source.get(GRADED_INTVTY_P50))
    duration = _positive(source.get(GRADED_DURATION)) or _positive(source.get("duration"))
    error_rate = _non_negative(source.get(GRADED_ERROR_RATE))
    inp = _positive(source.get("input_throughput"))
    out = _positive(source.get(GRADED_OUTPUT)) or _positive(source.get("tput"))
    total = _positive(source.get(GRADED_TOTAL)) or _positive(source.get("total_token_throughput"))
    if total is None and inp is not None and out is not None:
        total = inp + out
    if intvty is None or intvty_p50 is None or total is None:
        return None
    snap: dict[str, float] = {
        GRADED_INTVTY: intvty,
        GRADED_INTVTY_P50: intvty_p50,
        GRADED_TOTAL: total,
    }
    for key, value in (
        ("input_throughput", inp),
        (GRADED_OUTPUT, out),
        ("tpot_p90_ms", _positive(source.get("tpot_p90_ms"))),
        ("ttft_p50_ms", _positive(source.get("ttft_p50_ms"))),
        ("ttft_p90_ms", _positive(source.get("ttft_p90_ms"))),
        ("tpot_p50_ms", _positive(source.get("tpot_p50_ms"))),
        (GRADED_OUTPUT_PER_GPU, _positive(source.get(GRADED_OUTPUT_PER_GPU))),
        (GRADED_DURATION, duration),
        (GRADED_ERROR_RATE, error_rate),
        (GRADED_UNSERVED_REQUESTS, _non_negative(source.get(GRADED_UNSERVED_REQUESTS))),
        (GRADED_REQUESTS_TOTAL, _positive(source.get(GRADED_REQUESTS_TOTAL))),
    ):
        if value is not None:
            snap[key] = value
    snap.update({key: source[key] for key in _AGENTX_EVIDENCE if key in source})
    return snap


def output_tput_of(source: Mapping[str, Any] | None) -> float:
    """Output throughput from a measurement or a ``current_best``; 0.0 when absent."""
    if not isinstance(source, Mapping):
        return 0.0
    return float(_positive(source.get(GRADED_OUTPUT)) or _positive(source.get("tput")) or 0.0)


def axis_of(snapshot: Mapping[str, float] | None, key: str) -> float:
    """One axis from a perf snapshot; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get(key) or 0.0)


def intvty_of(snapshot: Mapping[str, float] | None) -> float:
    """Slow-tail interactivity from a perf snapshot; 0.0 when unavailable."""
    return axis_of(snapshot, GRADED_INTVTY)


def total_tput_of(snapshot: Mapping[str, float] | None) -> float:
    """Total token throughput from a perf snapshot; 0.0 when unavailable."""
    return axis_of(snapshot, GRADED_TOTAL)


def baseline_on_axis(state: Any, objective: str) -> float | None:
    """The session baseline read on *objective*'s axis, or None when that axis was not measured."""
    # A ledger divides by this, so an output baseline must never stand in for an interactivity objective.
    if objective in INTVTY_OBJECTIVES:
        value = axis_of(getattr(state, _BASELINE_PERF_ATTR, None), objective)
    else:
        value = float(getattr(state, "baseline_tput", 0.0) or 0.0)
    return value if value > 0 else None


def graded_axes_of(source: Mapping[str, Any] | None) -> dict[str, float]:
    """The graded axes *source* carries, for stamping onto a winner record."""
    # A KEEP's ``current_best`` becomes the next candidate's anchor, and an anchor missing an axis degrades the whole
    # session to output grading. Axes are absent rather than None so a partial record is not read as a measured zero.
    if not isinstance(source, Mapping):
        return {}
    axes: dict[str, float] = {}
    intvty = _positive(source.get(GRADED_INTVTY))
    if intvty is not None:
        axes[GRADED_INTVTY] = intvty
    total = _positive(source.get(GRADED_TOTAL)) or _positive(source.get("total_token_throughput"))
    if total is not None:
        axes[GRADED_TOTAL] = total
    for key in (
        "input_throughput",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        GRADED_INTVTY_P50,
        GRADED_OUTPUT_PER_GPU,
        GRADED_REQUESTS_TOTAL,
    ):
        value = _positive(source.get(key))
        if value is not None:
            axes[key] = value
    duration = _positive(source.get(GRADED_DURATION)) or _positive(source.get("duration"))
    if duration is not None:
        axes[GRADED_DURATION] = duration
    for key in (GRADED_ERROR_RATE, GRADED_UNSERVED_REQUESTS):
        value = _non_negative(source.get(key))
        if value is not None:
            axes[key] = value
    axes.update({key: source[key] for key in _AGENTX_EVIDENCE if key in source})
    return axes


def resolve_grading_anchor_perf(state: Any) -> tuple[dict[str, float] | None, str]:
    """Grading anchor: the current-best snapshot, falling back to the baseline; ``reason`` names any failure."""
    # A ``current_best`` that exists but carries no axes must not fall through to ``baseline_perf`` -- that would
    # anchor a candidate against a recipe it was never measured on.
    current_best = getattr(state, "current_best", None)
    if current_best:
        snap = perf_snapshot_from_mapping(current_best)
        if snap is not None:
            return snap, ""
        return None, "current_best_axes_missing"
    baseline_snap = perf_snapshot_from_mapping(getattr(state, _BASELINE_PERF_ATTR, None))
    if baseline_snap is not None:
        return baseline_snap, ""
    return None, "baseline_perf_missing"


def _within_band(candidate: float, anchor: float, band_pct: float) -> bool:
    """Whether *candidate* is not worse than *anchor* by more than the band."""
    if anchor <= 0:
        return True
    return candidate >= anchor * (1.0 - band_pct / _PERCENT)


def stamp_output_per_gpu(measurement: Any, tp: Any) -> None:
    """Derive the frontier's y axis onto *measurement* in place; a non-positive chip count leaves it unstamped.

    The chip count is the tensor-parallel degree, the same stand-in the roofline ceiling and the competitor gap
    already divide by. It undercounts a deployment that spreads over data or pipeline parallelism, disaggregated
    prefill, or several nodes; correcting it belongs with those callers, since a second denominator here would
    publish two different per-GPU figures for one session.
    """
    if not isinstance(measurement, dict):
        return
    chips = _positive(tp)
    out = _positive(measurement.get(GRADED_OUTPUT)) or _positive(measurement.get("tput"))
    if chips is None or out is None:
        return
    measurement[GRADED_OUTPUT_PER_GPU] = out / chips


def with_output_per_gpu(measurement: Mapping[str, Any], tp: Any) -> dict[str, Any]:
    """A copy of *measurement* carrying the per-GPU output, for a record written before its owner stamps it."""
    stamped = dict(measurement)
    stamp_output_per_gpu(stamped, tp)
    return stamped


def _unserved_rate_pct(side: Mapping[str, Any]) -> float | None:
    """The percentage of *side*'s requests the server left unserved; None when the side did not count them."""
    unserved = _non_negative(side.get(GRADED_UNSERVED_REQUESTS))
    total = _positive(side.get(GRADED_REQUESTS_TOTAL))
    if unserved is None or total is None:
        return None
    return _PERCENT * unserved / total


def _failure_rates_pct(candidate: Mapping[str, Any], anchor: Mapping[str, Any]) -> tuple[float, float] | None:
    """Both sides' failure rates in percent: unserved if both counted them, else all failed; None if unreported."""
    cand, ref = _unserved_rate_pct(candidate), _unserved_rate_pct(anchor)
    if cand is None or ref is None:
        cand, ref = _non_negative(candidate.get(GRADED_ERROR_RATE)), _non_negative(anchor.get(GRADED_ERROR_RATE))
    if cand is None or ref is None:
        return None
    return cand, ref


def incomparability_reason(candidate: Mapping[str, float], anchor: Mapping[str, float]) -> str:
    """Why the pair did not measure the same work, or "" when it did: equal-length windows, failures within the slack.

    Failures are the unserved requests when both sides counted them, so a short reply never refuses a pair. Fails
    closed on an unreported input. A truncated round still publishes plausible rates, so treating "no evidence" as
    "comparable" is what lets one KEEP on a window it never ran.
    """
    rates = _failure_rates_pct(candidate, anchor)
    ref_duration = axis_of(anchor, GRADED_DURATION)
    if rates is None or GRADED_DURATION not in candidate or ref_duration <= 0:
        return "comparability_inputs_missing"
    if abs(axis_of(candidate, GRADED_DURATION) / ref_duration - 1.0) * _PERCENT > DURATION_DRIFT_PCT:
        return "duration_drift"
    cand_rate, ref_rate = rates
    if cand_rate > ref_rate + ERROR_RATE_SLACK_PCT:
        return "extra_failed_requests"
    return ""


def holds_within_band(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    key: str,
    *,
    noise_pct: float | None = None,
) -> bool:
    """Whether candidate *key* holds within the noise band below *anchor*."""
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    return _within_band(axis_of(candidate, key), axis_of(anchor, key), band)


def _ceiling_veto(observed: Any, ceiling: float | None, *, exceeded: str, unmeasured: str) -> str:
    """``exceeded`` over the ceiling, ``unmeasured`` without a usable reading, else ""; "" when no ceiling is set.

    Fails closed on an unmeasured candidate: a constraint nobody measured is not one anybody satisfied.
    """
    if not ceiling or ceiling <= 0:
        return ""
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        return unmeasured
    value = float(observed)
    if not isfinite(value) or value <= 0:
        return unmeasured
    return exceeded if value > float(ceiling) else ""


def latency_veto_reason(observed_ms: Any, budget_ms: float) -> str:
    """Why the latency budget refuses this candidate, or "" when it does not.

    The budget is a ceiling on mean end-to-end latency, so unlike the gain gates
    it refuses a candidate whose throughput won: a lever that buys throughput by
    making each stream slower is exactly the case a throughput-only comparison
    selects for. Off entirely when *budget_ms* is not positive. Every lane copies
    ``e2el_mean_ms`` onto the dict it promotes.
    """
    return _ceiling_veto(observed_ms, budget_ms, exceeded="latency_budget_exceeded", unmeasured="latency_unmeasured")


_POWER_UNMEASURED = "power_unmeasured"
_POWER_BUDGET_EXCEEDED = "power_budget_exceeded"
_GPU_POWER_BUDGET_EXCEEDED = "gpu_power_budget_exceeded"
_POWER_VETO_REASONS = frozenset({_POWER_UNMEASURED, _POWER_BUDGET_EXCEEDED, _GPU_POWER_BUDGET_EXCEEDED})


def power_veto_reason(
    by_gpu_w: Any,
    *,
    total_budget_w: float = 0.0,
    per_gpu_budget_w: Mapping[str, float] | None = None,
) -> str:
    """Why the power budgets refuse this candidate, or "" when they do not.

    *by_gpu_w* is each serving card's mean power over the measured round, keyed by physical GPU id. *total_budget_w*
    caps their sum (``--max-power-w``); *per_gpu_budget_w* caps individual cards (``--max-per-gpu-power-w``). A card
    with no limit of its own is bound only by the total, and a limited card that did not serve this round is not
    judged on it. Off entirely when neither budget is set. Fails closed on a candidate with no per-card reading: a
    constraint nobody measured is not one anybody satisfied.
    """
    limits = {str(gpu): float(watts) for gpu, watts in (per_gpu_budget_w or {}).items() if watts and watts > 0}
    total = float(total_budget_w or 0.0)
    if total <= 0 and not limits:
        return ""
    if not isinstance(by_gpu_w, Mapping) or not by_gpu_w:
        return _POWER_UNMEASURED
    readings: dict[str, float] = {}
    for gpu, watts in by_gpu_w.items():
        if isinstance(watts, bool) or not isinstance(watts, (int, float)) or not isfinite(watts) or watts <= 0:
            return _POWER_UNMEASURED
        readings[str(gpu)] = float(watts)
    if any(gpu in readings and readings[gpu] > limit for gpu, limit in limits.items()):
        return _GPU_POWER_BUDGET_EXCEEDED
    if total > 0 and sum(readings.values()) > total:
        return _POWER_BUDGET_EXCEEDED
    return ""


def native_agentx_veto(source: Mapping[str, Any], state: Any) -> str:
    """Refuse diagnostic, invalid or differently configured AgentX evidence."""
    if getattr(state, "agentx_backend", "") != "magpie" and source.get("benchmark_backend") != "magpie_agentx":
        return ""
    if source.get("benchmark_valid") is not True or source.get("diagnostic_only") is not False:
        return "agentx_evidence_invalid"
    anchor = getattr(state, "current_best", None) or getattr(state, _BASELINE_PERF_ATTR, None) or {}
    identity = source.get("workload_identity")
    if not identity or identity != anchor.get("workload_identity"):
        return "agentx_workload_mismatch"
    return ""


def constraint_veto_reason(measurement: Any, state: Any) -> str:
    """The first session constraint *measurement* breaks, or ""; the one reader every KEEP decision uses."""
    source = measurement if isinstance(measurement, Mapping) else {}
    return (
        native_agentx_veto(source, state)
        or latency_veto_reason(source.get("e2el_mean_ms"), float(getattr(state, "latency_budget_ms", 0.0) or 0.0))
        or power_veto_reason(
            source.get("gpu_power_by_gpu_w"),
            total_budget_w=float(getattr(state, "power_budget_w", 0.0) or 0.0),
            per_gpu_budget_w=getattr(state, "power_budget_per_gpu_w", None),
        )
    )


#: Every measurement field a session constraint reads, with the mean power reported beside the per-card figures. A
#: carrier that hands a measurement to a KEEP decision copies them together, so no lane drops one.
CONSTRAINT_FIELDS = (
    "e2el_mean_ms",
    "gpu_power_avg_w",
    "gpu_power_by_gpu_w",
    "benchmark_backend",
    "benchmark_valid",
    "diagnostic_only",
    "workload_identity",
)


def constraint_fields(source: Any) -> dict[str, Any]:
    """:data:`CONSTRAINT_FIELDS` read off a mapping or an object; an absent field is ``None``."""
    if isinstance(source, Mapping):
        return {name: source.get(name) for name in CONSTRAINT_FIELDS}
    return {name: getattr(source, name, None) for name in CONSTRAINT_FIELDS}


@dataclass(frozen=True)
class GradedComparison:
    """A candidate, the figure it must beat, and the verdict on that pair.

    ``candidate`` and ``reference`` are both read on ``objective``. ``tput_*`` carry total throughput and are 0.0 off
    AgentX. ``degrade_reason`` names why the interactivity axis did not apply on a session that asked for it.
    ``veto_reason`` names a constraint that refused a candidate its throughput would otherwise have kept.
    ``refused_by`` names, comma-joined, every interactivity gate the candidate failed.
    """

    objective: str
    candidate: float
    reference: float
    verdict: str
    tput_candidate: float = 0.0
    tput_reference: float = 0.0
    degrade_reason: str = ""
    veto_reason: str = ""
    refused_by: str = ""

    @property
    def comparable(self) -> bool:
        """Whether both sides supplied the axes the session asked to be graded on.

        A degraded pair still carries an output-axis figure, which is a useful diagnostic but not the objective the
        session was configured for. Lanes that must not promote on a substitute axis read this rather than the
        verdict, so an axis-less measurement fails closed instead of scoring as an output win.
        """
        return not self.degrade_reason

    @property
    def vetoed_by(self) -> str:
        """The session constraint behind ``veto_reason``: ``power_budget`` or ``latency_budget``; ``""`` when none."""
        if not self.veto_reason:
            return ""
        return "power_budget" if self.veto_reason in _POWER_VETO_REASONS else "latency_budget"

    @property
    def graded_on_intvty(self) -> bool:
        """Whether the interactivity objective actually applied."""
        return self.objective in INTVTY_OBJECTIVES


@dataclass(frozen=True)
class GradingAnchor:
    """The reference a candidate is graded against in place of the session anchor; ``None`` keeps the session's."""

    perf: Mapping[str, float] | None = None
    tput: float | None = None


__all__ = [
    "AGENTX_KEEP_P50_THRESHOLD_PCT",
    "CONSTRAINT_FIELDS",
    "DURATION_DRIFT_PCT",
    "ERROR_RATE_SLACK_PCT",
    "GradedComparison",
    "GradingAnchor",
    "GRADED_AXIS_KEYS",
    "GRADED_DURATION",
    "GRADED_ERROR_RATE",
    "GRADED_INTVTY",
    "GRADED_INTVTY_P50",
    "GRADED_OUTPUT",
    "GRADED_OUTPUT_PER_GPU",
    "GRADED_REQUESTS_TOTAL",
    "GRADED_TOTAL",
    "GRADED_UNSERVED_REQUESTS",
    "INTVTY_OBJECTIVES",
    "INTVTY_V1",
    "VERDICT_KEEP",
    "VERDICT_REVERT",
    "agentx_active",
    "axis_of",
    "baseline_on_axis",
    "graded_axes_of",
    "graded_metric_key",
    "holds_within_band",
    "incomparability_reason",
    "intvty_grading_enabled",
    "intvty_of",
    "intvty_serving_grading_enabled",
    "is_agentx_mode",
    "constraint_fields",
    "constraint_veto_reason",
    "latency_veto_reason",
    "power_veto_reason",
    "output_tput_of",
    "parse_intvty_noise_pct",
    "perf_snapshot_from_mapping",
    "resolve_grading_anchor_perf",
    "stamp_output_per_gpu",
    "with_output_per_gpu",
    "total_tput_of",
]
