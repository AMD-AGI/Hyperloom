# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading: a total-token-throughput objective under an interactivity gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_str

COMPOSITE_V1 = "composite_v1"

# Read by name because ``common/`` must not import the orchestrator, where ``agentx_enabled`` lives.
_AGENTX_ENV = "HYPERLOOM_AGENTX"

# The value ``SharedState.benchmark_mode`` carries for an AgentX session, stamped at seed so it outlives the shell
# that started the run.
_AGENTX_MODE = "agentx"


def is_agentx_mode(benchmark_mode: Any) -> bool:
    """Whether a ``benchmark_mode`` names the agentic workload."""
    return str(benchmark_mode or "").strip().lower() == _AGENTX_MODE


# Upstream reports run-to-run noise on this workload as 1-5% depending on the concurrency regime, so the veto band
# opens to the top of that range instead of rejecting movement upstream would call noise.
_DEFAULT_INTVTY_NOISE_PCT = 5.0


def total_tput_grading_enabled(*, benchmark_mode: str = "") -> bool:
    """True when total-token-throughput grading applies."""
    raw = env_str("HYPERLOOM_PERF_METRIC").strip().lower()
    if raw:
        return raw == COMPOSITE_V1
    if env_bool(_AGENTX_ENV):
        return True
    return is_agentx_mode(benchmark_mode)


GRADED_TOTAL = "total_throughput"
GRADED_OUTPUT = "output_throughput"


def graded_metric_key(*, benchmark_mode: str = "") -> str:
    """The curve-row field a session's speedups are measured on."""
    if total_tput_grading_enabled(benchmark_mode=benchmark_mode):
        return "total_token_throughput"
    return GRADED_OUTPUT


def total_tput_serving_grading_enabled(*, scriptable: bool = False, benchmark_mode: str = "") -> bool:
    """Total-token-throughput grading, limited to non-scriptable serving runs."""
    return total_tput_grading_enabled(benchmark_mode=benchmark_mode) and not scriptable


def parse_intvty_noise_pct() -> float:
    """Interactivity veto band in percent from ``HYPERLOOM_PERF_NOISE_PCT``."""
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
    """Extract the graded pair, carrying the reported axes when present."""
    if not isinstance(source, Mapping):
        return None
    inp = _positive(source.get("input_throughput"))
    out = _positive(source.get("output_throughput")) or _positive(source.get("tput"))
    total = _positive(source.get("total_throughput")) or _positive(source.get("total_token_throughput"))
    if total is None and inp is not None and out is not None:
        total = inp + out
    intv = _positive(source.get("intvty_p90"))
    if total is None or intv is None:
        return None
    snap: dict[str, float] = {"total_throughput": total, "intvty_p90": intv}
    for key, value in (
        ("input_throughput", inp),
        ("output_throughput", out),
        ("tpot_p90_ms", _positive(source.get("tpot_p90_ms"))),
    ):
        if value is not None:
            snap[key] = value
    return snap


def output_tput_of(source: Mapping[str, Any] | None) -> float:
    """Output throughput from a measurement or a ``current_best``; 0.0 when absent."""
    if not isinstance(source, Mapping):
        return 0.0
    return float(_positive(source.get("output_throughput")) or _positive(source.get("tput")) or 0.0)


def graded_axes_of(source: Mapping[str, Any] | None) -> dict[str, float]:
    """The graded axes *source* actually carries, for stamping onto a winner record."""
    if not isinstance(source, Mapping):
        return {}
    axes: dict[str, float] = {}
    total = _positive(source.get("total_throughput")) or _positive(source.get("total_token_throughput"))
    if total is not None:
        axes["total_throughput"] = total
    for key in ("input_throughput", "tpot_p90_ms", "intvty_p90"):
        value = _positive(source.get(key))
        if value is not None:
            axes[key] = value
    return axes


@dataclass(frozen=True)
class GradedComparison:
    """A candidate and the figure it must beat, both read off one axis."""

    objective: str
    candidate: float
    reference: float
    vetoed: bool = False
    degrade_reason: str = ""

    @property
    def comparable(self) -> bool:
        """Whether both measurements provide the axes required for a performance verdict."""
        return not self.degrade_reason

    @property
    def graded_on_total(self) -> bool:
        """Whether the total-token-throughput objective actually applied."""
        return self.objective == GRADED_TOTAL


def total_tput_of(snapshot: Mapping[str, float] | None) -> float:
    """Total token throughput from a perf snapshot; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get("total_throughput") or 0.0)


def resolve_grading_anchor_perf(state: Any) -> tuple[dict[str, float] | None, str]:
    """Total-axis grading anchor: current-best snapshot, falling back to baseline."""
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


def passes_intvty_gate(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    *,
    noise_pct: float | None = None,
) -> bool:
    """Veto: intvty p90 must not regress past the noise band below *anchor*."""
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    anchor_intv = float(anchor.get("intvty_p90") or 0.0)
    cand_intv = float(candidate.get("intvty_p90") or 0.0)
    return cand_intv >= anchor_intv * (1.0 - band / 100.0)


__all__ = [
    "COMPOSITE_V1",
    "GradedComparison",
    "GRADED_OUTPUT",
    "GRADED_TOTAL",
    "graded_axes_of",
    "graded_metric_key",
    "is_agentx_mode",
    "output_tput_of",
    "parse_intvty_noise_pct",
    "passes_intvty_gate",
    "perf_snapshot_from_mapping",
    "resolve_grading_anchor_perf",
    "total_tput_grading_enabled",
    "total_tput_of",
    "total_tput_serving_grading_enabled",
]
