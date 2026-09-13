# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading: an E2E-normalised-interactivity objective guarded by per-chip throughput."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_str

INTVTY_V1 = "intvty_v1"

# Read by name because ``common/`` must not import the orchestrator, where ``agentx_enabled`` lives.
_AGENTX_ENV = "HYPERLOOM_AGENTX"

# The value ``SharedState.benchmark_mode`` carries for an AgentX session, stamped at seed so it outlives the shell
# that started the run.
_AGENTX_MODE = "agentx"

# Graded axis names, which are also the snapshot keys they are read from. InferenceX publishes a 2-D Pareto frontier
# with interactivity on x and per-chip throughput on y, and no fixed interactivity target, so trading one for the
# other moves a point along the frontier rather than violating a constraint.
GRADED_INTVTY = "e2e_norm_intvty_p90"
GRADED_TOTAL = "total_throughput"
GRADED_OUTPUT = "output_throughput"

# Upstream reports run-to-run noise on this workload as 1-5% depending on the concurrency regime, so the band opens
# to the top of that range instead of rejecting movement upstream would call noise.
_DEFAULT_INTVTY_NOISE_PCT = 5.0

# Floor under ``keep_threshold_pct`` for AgentX: the slow-tail percentile's own variance is unmeasured, so the
# default 1% threshold sits inside the noise band.
AGENTX_KEEP_THRESHOLD_FLOOR_PCT = 2.0

# RECORDED exists because a point that loses at the measured concurrency can still be the frontier winner at another
# rung, so discarding it costs more than storing it.
VERDICT_KEEP = "KEEP"
VERDICT_REVERT = "REVERT"
VERDICT_RECORDED = "RECORDED"


def is_agentx_mode(benchmark_mode: Any) -> bool:
    """Whether a ``benchmark_mode`` names the agentic workload."""
    return str(benchmark_mode or "").strip().lower() == _AGENTX_MODE


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
    raw = env_str("HYPERLOOM_PERF_NOISE_PCT").strip()
    if not raw:
        return _DEFAULT_INTVTY_NOISE_PCT
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_INTVTY_NOISE_PCT


def _positive(value: Any) -> float | None:
    """Coerce to a strictly positive float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coerced = float(value)
    return coerced if coerced > 0 else None


def perf_snapshot_from_mapping(source: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Both graded axes from a measurement or a ``current_best``; None unless both are positive."""
    # Returning None unless both are present is what stops a lane half-applying the objective. A total that is
    # absent, null or non-positive coalesces to input plus output, the same fallback ``agentx.mapping`` applies.
    if not isinstance(source, Mapping):
        return None
    intvty = _positive(source.get(GRADED_INTVTY))
    inp = _positive(source.get("input_throughput"))
    out = _positive(source.get(GRADED_OUTPUT)) or _positive(source.get("tput"))
    total = _positive(source.get(GRADED_TOTAL)) or _positive(source.get("total_token_throughput"))
    if total is None and inp is not None and out is not None:
        total = inp + out
    if intvty is None or total is None:
        return None
    snap: dict[str, float] = {GRADED_INTVTY: intvty, GRADED_TOTAL: total}
    for key, value in (
        ("input_throughput", inp),
        (GRADED_OUTPUT, out),
        ("tpot_p90_ms", _positive(source.get("tpot_p90_ms"))),
    ):
        if value is not None:
            snap[key] = value
    return snap


def output_tput_of(source: Mapping[str, Any] | None) -> float:
    """Output throughput from a measurement or a ``current_best``; 0.0 when absent."""
    if not isinstance(source, Mapping):
        return 0.0
    return float(_positive(source.get(GRADED_OUTPUT)) or _positive(source.get("tput")) or 0.0)


def intvty_of(snapshot: Mapping[str, float] | None) -> float:
    """Interactivity from a perf snapshot; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get(GRADED_INTVTY) or 0.0)


def total_tput_of(snapshot: Mapping[str, float] | None) -> float:
    """Total token throughput from a perf snapshot; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get(GRADED_TOTAL) or 0.0)


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
    for key in ("input_throughput", "tpot_p90_ms"):
        value = _positive(source.get(key))
        if value is not None:
            axes[key] = value
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


def passes_intvty_gate(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    *,
    noise_pct: float | None = None,
) -> bool:
    """Whether candidate interactivity holds within the band below *anchor*."""
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    return _within_band(intvty_of(candidate), intvty_of(anchor), band)


def passes_tput_guard(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    *,
    noise_pct: float | None = None,
) -> bool:
    """Whether candidate throughput holds within the band below *anchor*."""
    # ``total_throughput`` is the raw aggregate; a caller comparing configurations of differing tensor-parallel
    # degree must normalise by the chip count first.
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    return _within_band(total_tput_of(candidate), total_tput_of(anchor), band)


@dataclass(frozen=True)
class GradedComparison:
    """A candidate, the figure it must beat, and the verdict on that pair.

    ``candidate`` and ``reference`` are both read on ``objective``. ``tput_*`` carry the guard axis and are 0.0 off
    AgentX. ``degrade_reason`` names why the interactivity axis did not apply on a session that asked for it.
    """

    objective: str
    candidate: float
    reference: float
    verdict: str
    tput_candidate: float = 0.0
    tput_reference: float = 0.0
    degrade_reason: str = ""

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
        return self.objective == GRADED_INTVTY


__all__ = [
    "AGENTX_KEEP_THRESHOLD_FLOOR_PCT",
    "GradedComparison",
    "GRADED_INTVTY",
    "GRADED_OUTPUT",
    "GRADED_TOTAL",
    "INTVTY_V1",
    "VERDICT_KEEP",
    "VERDICT_RECORDED",
    "VERDICT_REVERT",
    "graded_axes_of",
    "graded_metric_key",
    "intvty_grading_enabled",
    "intvty_of",
    "intvty_serving_grading_enabled",
    "is_agentx_mode",
    "output_tput_of",
    "parse_intvty_noise_pct",
    "passes_intvty_gate",
    "passes_tput_guard",
    "perf_snapshot_from_mapping",
    "resolve_grading_anchor_perf",
    "total_tput_of",
]
