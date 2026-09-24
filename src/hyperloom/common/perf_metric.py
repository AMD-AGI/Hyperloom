# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading and the canonical metrics carried by its measurements."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_str, is_truthy

INTVTY_V1 = "intvty_v1"

_AGENTX_ENV = "HYPERLOOM_AGENTX"

# The value ``SharedState.benchmark_mode`` carries for an AgentX session, stamped at seed so it outlives the shell
# that started the run.
_AGENTX_MODE = "agentx"

# Canonical AgentX fields. P50 is the promotion objective; P90 and output per
# GPU remain the official chart axes.
GRADED_INTVTY = "e2e_intvty_p50"
GRADED_INTVTY_P90 = "e2e_intvty_p90"
GRADED_OUTPUT_PER_GPU = "output_tput_per_gpu"
GRADED_TOTAL = "total_throughput"
GRADED_OUTPUT = "output_throughput"

# The user-facing AgentX measurement fields. Session Breakdown publishes every
# key, using null for a field the benchmark did not report.
GRADED_AXIS_KEYS = (
    GRADED_INTVTY,
    GRADED_INTVTY_P90,
    GRADED_OUTPUT_PER_GPU,
    "ttft_p50_ms",
    "ttft_p90_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
)

# Upstream reports run-to-run noise on this workload as 1-5% depending on the concurrency regime, so the band opens
# to the top of that range instead of rejecting movement upstream would call noise.
_DEFAULT_INTVTY_NOISE_PCT = 5.0

# AgentX promotion policy. These are fixed business thresholds rather than
# caller-specific tuning knobs.
AGENTX_KEEP_THRESHOLD_FLOOR_PCT = 3.0
AGENTX_P90_MIN_DELTA_PCT = -5.0
AGENTX_OUTPUT_MIN_DELTA_PCT = -5.0
AGENTX_DURATION_MAX_ABS_DELTA_PCT = 5.0

# ``RECORDED`` remains part of the shared verdict vocabulary for historical
# records and non-AgentX consumers. The current AgentX policy emits only KEEP
# or REVERT.
VERDICT_KEEP = "KEEP"
VERDICT_REVERT = "REVERT"
VERDICT_RECORDED = "RECORDED"


def agentx_policy_config() -> dict[str, Any]:
    """Serializable definition of the fixed AgentX all-of promotion policy."""
    return {
        "mode": "all_of",
        "anchor_priority": ["current_best", "baseline"],
        "submission_valid": True,
        "request_error_rate_max": "anchor",
        "accuracy_passed": True,
        "duration_max_abs_delta_pct": AGENTX_DURATION_MAX_ABS_DELTA_PCT,
        "e2e_intvty_p50_min_delta_pct": AGENTX_KEEP_THRESHOLD_FLOOR_PCT,
        "e2e_intvty_p90_min_delta_pct": AGENTX_P90_MIN_DELTA_PCT,
        "output_tput_per_gpu_min_delta_pct": AGENTX_OUTPUT_MIN_DELTA_PCT,
    }


def agentx_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the AgentX benchmark wrapper is explicitly enabled."""
    raw = (env or os.environ).get(_AGENTX_ENV, "")
    return is_truthy(raw)


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
    return coerced if math.isfinite(coerced) and coerced > 0 else None


def _nonnegative(value: Any) -> float | None:
    """Coerce to a finite non-negative float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coerced = float(value)
    return coerced if math.isfinite(coerced) and coerced >= 0 else None


def _first_positive(source: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first positive value found under *keys*."""
    for key in keys:
        value = _positive(source.get(key))
        if value is not None:
            return value
    return None


def _agentx_values(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize one AgentX measurement while accepting the previous flat names."""
    if not isinstance(source, Mapping):
        return {}
    return {
        GRADED_INTVTY: _first_positive(source, GRADED_INTVTY, "e2e_norm_intvty_p50"),
        GRADED_INTVTY_P90: _first_positive(source, GRADED_INTVTY_P90, "e2e_norm_intvty_p90"),
        GRADED_OUTPUT_PER_GPU: _first_positive(source, GRADED_OUTPUT_PER_GPU),
        "duration_s": _first_positive(source, "duration_s", "duration_seconds", "duration"),
        "request_error_rate": _nonnegative(source.get("request_error_rate")),
        "submission_valid": source.get("submission_valid")
        if isinstance(source.get("submission_valid"), bool)
        else None,
        "accuracy_passed": (
            source.get("accuracy_passed")
            if isinstance(source.get("accuracy_passed"), bool)
            else source.get("accuracy_pass")
            if isinstance(source.get("accuracy_pass"), bool)
            else None
        ),
        "ttft_p50_ms": _first_positive(source, "ttft_p50_ms", "median_ttft_ms"),
        "ttft_p90_ms": _first_positive(source, "ttft_p90_ms", "p90_ttft_ms"),
        "tpot_p50_ms": _first_positive(source, "tpot_p50_ms", "median_tpot_ms"),
        "tpot_p90_ms": _first_positive(source, "tpot_p90_ms", "p90_tpot_ms"),
    }


def perf_snapshot_from_mapping(source: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the complete AgentX anchor snapshot, or None when a required value is missing."""
    values = _agentx_values(source)
    required = (
        GRADED_INTVTY,
        GRADED_INTVTY_P90,
        GRADED_OUTPUT_PER_GPU,
        "duration_s",
        "request_error_rate",
    )
    if any(values.get(key) is None for key in required):
        return None
    snap: dict[str, Any] = {}
    for key, value in values.items():
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


def intvty_p90_of(snapshot: Mapping[str, Any] | None) -> float:
    """P90 E2E normalized interactivity; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(_agentx_values(snapshot).get(GRADED_INTVTY_P90) or 0.0)


def output_tput_per_gpu_of(snapshot: Mapping[str, Any] | None) -> float:
    """Per-GPU output throughput; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(_agentx_values(snapshot).get(GRADED_OUTPUT_PER_GPU) or 0.0)


def total_tput_of(snapshot: Mapping[str, float] | None) -> float:
    """Total token throughput from a perf snapshot; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get(GRADED_TOTAL) or 0.0)


def graded_axes_of(source: Mapping[str, Any] | None) -> dict[str, float]:
    """The user-facing AgentX metrics *source* carries."""
    values = _agentx_values(source)
    axes: dict[str, float] = {}
    for key in GRADED_AXIS_KEYS:
        value = values.get(key)
        if value is not None:
            axes[key] = float(value)
    return axes


def agentx_snapshot_of(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """All AgentX values needed to grade the next candidate, when available."""
    return {key: value for key, value in _agentx_values(source).items() if value is not None}


def resolve_grading_anchor_perf(state: Any) -> tuple[dict[str, Any] | None, str]:
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
    """Whether candidate per-GPU output throughput holds within the band."""
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    return _within_band(output_tput_per_gpu_of(candidate), output_tput_per_gpu_of(anchor), band)


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
    anchor_source: str = ""
    checks: dict[str, bool] | None = None
    deltas_pct: dict[str, float | None] | None = None
    failed_checks: tuple[str, ...] = ()
    candidate_evidence: dict[str, Any] | None = None
    reference_evidence: dict[str, Any] | None = None

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

    def policy_evidence(self) -> dict[str, Any]:
        """JSON-friendly AgentX policy evidence for Session Breakdown."""
        return {
            "anchor_source": self.anchor_source or None,
            "checks": dict(self.checks or {}),
            "deltas_pct": dict(self.deltas_pct or {}),
            "failed_checks": list(self.failed_checks),
            "candidate": dict(self.candidate_evidence or {}),
            "anchor": dict(self.reference_evidence or {}),
            "verdict": self.verdict,
        }


__all__ = [
    "AGENTX_KEEP_THRESHOLD_FLOOR_PCT",
    "GradedComparison",
    "GRADED_AXIS_KEYS",
    "GRADED_INTVTY",
    "GRADED_INTVTY_P90",
    "GRADED_OUTPUT",
    "GRADED_OUTPUT_PER_GPU",
    "GRADED_TOTAL",
    "INTVTY_V1",
    "VERDICT_KEEP",
    "VERDICT_RECORDED",
    "VERDICT_REVERT",
    "agentx_active",
    "agentx_policy_config",
    "agentx_snapshot_of",
    "graded_axes_of",
    "graded_metric_key",
    "intvty_grading_enabled",
    "intvty_of",
    "intvty_p90_of",
    "intvty_serving_grading_enabled",
    "is_agentx_mode",
    "output_tput_of",
    "output_tput_per_gpu_of",
    "parse_intvty_noise_pct",
    "passes_intvty_gate",
    "passes_tput_guard",
    "perf_snapshot_from_mapping",
    "resolve_grading_anchor_perf",
    "total_tput_of",
]
