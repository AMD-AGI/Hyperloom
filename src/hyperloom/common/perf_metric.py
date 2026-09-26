# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading: an E2E-normalised-interactivity objective guarded by per-chip throughput."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_float, env_str

INTVTY_V1 = "intvty_v1"

_AGENTX_ENV = "HYPERLOOM_AGENTX"

# The value ``SharedState.benchmark_mode`` carries for an AgentX session, stamped at seed so it outlives the shell
# that started the run.
_AGENTX_MODE = "agentx"

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

# A trace replay slices a different part of the corpus when the window moves, so the two rounds stop measuring the
# same work. Sized to catch a truncated round, not the few percent a full round drifts by.
DURATION_DRIFT_PCT = 5.0

# The axes ``graded_axes_of`` can carry, for a consumer that must publish all of them including the ones a
# measurement did not supply. Absent and null are not the same fact: a recorder that omits an axis leaves a reader
# unable to tell an unmeasured axis from one the framework failed to report, and zero reads as "measured, and it
# was zero".
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
)

# Upstream reports run-to-run noise on this workload as 1-5% depending on the concurrency regime, so the band opens
# to the top of that range instead of rejecting movement upstream would call noise.
_DEFAULT_INTVTY_NOISE_PCT = 5.0

# The median bar is a property of the objective, so the session's decaying threshold does not apply to it.
AGENTX_KEEP_P50_THRESHOLD_PCT = 3.0

VERDICT_KEEP = "KEEP"
VERDICT_REVERT = "REVERT"


def agentx_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the AgentX benchmark wrapper is explicitly enabled."""
    return env_bool(_AGENTX_ENV, env=env)


def is_agentx_mode(benchmark_mode: Any) -> bool:
    """Whether a ``benchmark_mode`` names the agentic workload."""
    return str(benchmark_mode or "").strip().lower() == _AGENTX_MODE


def agentx_active(*, benchmark_mode: Any = "") -> bool:
    """Whether either persisted workload identity or the environment enables AgentX."""
    return env_bool(_AGENTX_ENV) or is_agentx_mode(benchmark_mode)


def intvty_grading_enabled(*, benchmark_mode: str = "") -> bool:
    """True when interactivity grading applies; ``benchmark_mode`` is a parameter to keep this module a leaf."""
    # Passing the mode matters: the env var describes only the shell that happens to be running, so a re-baseline or
    # integrate round in a subprocess would otherwise grade an agentic measurement on the synthetic axis.
    raw = env_str("HYPERLOOM_PERF_METRIC").strip().lower()
    if raw:
        return raw == INTVTY_V1
    if env_bool(_AGENTX_ENV):
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
    ):
        if value is not None:
            snap[key] = value
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
    ):
        value = _positive(source.get(key))
        if value is not None:
            axes[key] = value
    duration = _positive(source.get(GRADED_DURATION)) or _positive(source.get("duration"))
    if duration is not None:
        axes[GRADED_DURATION] = duration
    error_rate = _non_negative(source.get(GRADED_ERROR_RATE))
    if error_rate is not None:
        axes[GRADED_ERROR_RATE] = error_rate
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
    baseline_snap = perf_snapshot_from_mapping(getattr(state, "baseline_perf", None))
    if baseline_snap is not None:
        return baseline_snap, ""
    return None, "baseline_perf_missing"


def _within_band(candidate: float, anchor: float, band_pct: float) -> bool:
    """Whether *candidate* is not worse than *anchor* by more than the band."""
    if anchor <= 0:
        return True
    return candidate >= anchor * (1.0 - band_pct / 100.0)


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


def rounds_are_comparable(candidate: Mapping[str, float], anchor: Mapping[str, float]) -> bool:
    """Whether the pair measured the same work: equal-length windows and no extra failed requests.

    Fails closed on an unreported input. A truncated round still publishes plausible rates, so treating "no
    evidence" as "comparable" is what lets one KEEP on a window it never ran.
    """
    for side in (candidate, anchor):
        if not all(key in side for key in (GRADED_DURATION, GRADED_ERROR_RATE)):
            return False
    ref_duration = axis_of(anchor, GRADED_DURATION)
    if ref_duration <= 0:
        return False
    if abs(axis_of(candidate, GRADED_DURATION) / ref_duration - 1.0) * 100.0 > DURATION_DRIFT_PCT:
        return False
    return axis_of(candidate, GRADED_ERROR_RATE) <= axis_of(anchor, GRADED_ERROR_RATE)


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


def latency_veto_reason(observed_ms: Any, budget_ms: float) -> str:
    """Why the latency budget refuses this candidate, or "" when it does not.

    The budget is a ceiling on mean end-to-end latency, so unlike the gain gates
    it refuses a candidate whose throughput won: a lever that buys throughput by
    making each stream slower is exactly the case a throughput-only comparison
    selects for. Off entirely when *budget_ms* is not positive.

    Fails closed on an unmeasured candidate — a constraint nobody measured is not
    one anybody satisfied — which is why every lane copies ``e2el_mean_ms`` onto
    the dict it promotes.
    """
    if not budget_ms or budget_ms <= 0:
        return ""
    if isinstance(observed_ms, bool) or not isinstance(observed_ms, (int, float)):
        return "latency_unmeasured"
    observed = float(observed_ms)
    if not isfinite(observed) or observed <= 0:
        return "latency_unmeasured"
    return "latency_budget_exceeded" if observed > float(budget_ms) else ""


@dataclass(frozen=True)
class GradedComparison:
    """A candidate, the figure it must beat, and the verdict on that pair.

    ``candidate`` and ``reference`` are both read on ``objective``. ``tput_*`` carry total throughput and are 0.0 off
    AgentX. ``degrade_reason`` names why the interactivity axis did not apply on a session that asked for it.
    ``veto_reason`` names a constraint that refused a candidate its throughput would otherwise have kept.
    """

    objective: str
    candidate: float
    reference: float
    verdict: str
    tput_candidate: float = 0.0
    tput_reference: float = 0.0
    degrade_reason: str = ""
    veto_reason: str = ""

    @property
    def comparable(self) -> bool:
        """Whether both sides supplied the axes the session asked to be graded on.

        A degraded pair still carries an output-axis figure, which is a useful diagnostic but not the objective the
        session was configured for. Lanes that must not promote on a substitute axis read this rather than the
        verdict, so an axis-less measurement fails closed instead of scoring as an output win.
        """
        return not self.degrade_reason

    @property
    def graded_on_intvty(self) -> bool:
        """Whether the interactivity objective actually applied."""
        return self.objective in INTVTY_OBJECTIVES


__all__ = [
    "AGENTX_KEEP_P50_THRESHOLD_PCT",
    "DURATION_DRIFT_PCT",
    "GradedComparison",
    "GRADED_AXIS_KEYS",
    "GRADED_DURATION",
    "GRADED_ERROR_RATE",
    "GRADED_INTVTY",
    "GRADED_INTVTY_P50",
    "GRADED_OUTPUT",
    "GRADED_OUTPUT_PER_GPU",
    "GRADED_TOTAL",
    "INTVTY_OBJECTIVES",
    "INTVTY_V1",
    "VERDICT_KEEP",
    "VERDICT_REVERT",
    "agentx_active",
    "axis_of",
    "graded_axes_of",
    "graded_metric_key",
    "holds_within_band",
    "intvty_grading_enabled",
    "intvty_of",
    "intvty_serving_grading_enabled",
    "is_agentx_mode",
    "latency_veto_reason",
    "output_tput_of",
    "parse_intvty_noise_pct",
    "perf_snapshot_from_mapping",
    "resolve_grading_anchor_perf",
    "rounds_are_comparable",
    "stamp_output_per_gpu",
    "total_tput_of",
]
