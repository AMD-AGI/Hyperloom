# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading: a total-token-throughput objective under an interactivity gate.

Mirrors how InferenceX ranks an AgentX submission. Upstream never collapses the
axes into one weighted number: it sweeps a concurrency ladder (the ``conc-list``
search space in ``configs/*-master.yaml``), keeps TTFT and inter-token-latency
percentiles separately, and compares throughput per chip *at a fixed
interactivity target*. Interactivity is a constraint there, not a term in the
objective, so trading it away for throughput is not a result upstream can
express.

Grading therefore has two parts. Total token throughput is the objective, graded
with the same :func:`hyperloom.common.gain_math.gain_pct` every other executor
uses, against the same ``keep_threshold_pct``. An interactivity p90 regression
past the noise band vetoes a candidate before its throughput is read.

Total tokens rather than output tokens because that is upstream's numerator, and
prefill dominates an agentic replay: the canonical corpus averages ~114k prompt
tokens against ~810 output tokens per request, so output-only grading optimises
about 1% of the token budget.

Default-on for AgentX runs and off otherwise; ``HYPERLOOM_PERF_METRIC`` overrides
either way. Serving only; scriptable frameworks keep output-throughput grading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_str

COMPOSITE_V1 = "composite_v1"

# Read by name because ``common/`` must not import the orchestrator, where
# ``agentx_enabled`` lives. Both resolve ``common.env._TRUE_TOKENS``.
_AGENTX_ENV = "HYPERLOOM_AGENTX"

# Upstream reports run-to-run noise on this workload as 1-5% depending on the
# concurrency regime, so the veto band opens to the top of that range instead of
# rejecting movement upstream would call noise.
_DEFAULT_INTVTY_NOISE_PCT = 5.0


def total_tput_grading_enabled() -> bool:
    """True when total-token-throughput grading applies.

    Reads ``HYPERLOOM_PERF_METRIC`` first; if set, returns True only when it
    equals ``composite_v1``. When unset, follows ``HYPERLOOM_AGENTX``.
    An explicit value wins in both directions.
    """
    raw = env_str("HYPERLOOM_PERF_METRIC").strip().lower()
    if raw:
        return raw == COMPOSITE_V1
    return env_bool(_AGENTX_ENV)


def total_tput_serving_grading_enabled(*, scriptable: bool = False) -> bool:
    """Total-token-throughput grading, limited to non-scriptable serving runs.

    A scriptable framework reports an image-quality gate rather than token
    throughput, so it has no total axis and keeps output grading. ``scriptable``
    is a parameter because ``hyperloom.common`` must not import the framework
    registry; ``shared_state.framework_is_scriptable`` resolves it.
    """
    return total_tput_grading_enabled() and not scriptable


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
    """Extract the graded pair, carrying the reported axes when present.

    Returns None unless both graded quantities are positive. A total that is
    absent, null or non-positive coalesces to input plus output, the same
    fallback :mod:`hyperloom.inference_optimizer.agentx.mapping` applies when
    aiperf omits it -- persisted records carry explicit nulls for axes a
    framework never measured.
    """
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


GRADED_TOTAL = "total_throughput"
GRADED_OUTPUT = "output_throughput"


def output_tput_of(source: Mapping[str, Any] | None) -> float:
    """Output throughput from a measurement or a ``current_best``; 0.0 when absent."""
    if not isinstance(source, Mapping):
        return 0.0
    return float(_positive(source.get("output_throughput")) or _positive(source.get("tput")) or 0.0)


def graded_axes_of(source: Mapping[str, Any] | None) -> dict[str, float]:
    """The graded axes *source* actually carries, for stamping onto a winner record.

    A KEEP's ``current_best`` has to carry the axes of the measurement it was
    promoted on: the next candidate anchors against it, and an anchor missing
    an axis degrades the whole session to output grading. Absent rather than
    ``None`` for an axis that was not measured, so a partial record is not
    mistaken for a measured zero.
    """
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
    """A candidate and the figure it must beat, both read off one axis.

    Attributes:
        objective: ``GRADED_TOTAL`` or ``GRADED_OUTPUT`` -- the axis BOTH
            ``candidate`` and ``reference`` were read on.
        candidate: The measured candidate on that axis; ``0.0`` when absent.
        reference: The anchor or baseline it is graded against, same axis.
        vetoed: The interactivity constraint rejected the candidate. Only set
            on ``GRADED_TOTAL``; the constraint belongs to that objective.
        degrade_reason: Why the total axis did not apply on a session that
            asked for it; ``""`` when it applied or was never requested.
    """

    objective: str
    candidate: float
    reference: float
    vetoed: bool = False
    degrade_reason: str = ""

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
    """Total-axis grading anchor: current-best snapshot, falling back to baseline.

    - ``current_best`` non-empty and axes present: return its snapshot.
    - ``current_best`` non-empty but axes absent: return
      ``(None, "current_best_axes_missing")``.  Must not fall through to
      ``baseline_perf`` — that would anchor a candidate against a recipe it
      was never measured on.
    - ``current_best`` empty: snapshot ``baseline_perf``; failure returns
      ``(None, "baseline_perf_missing")``.

    Returns:
        ``(snapshot, reason)`` where ``reason`` is an empty string on success
        and a short tag when no usable anchor exists.
    """
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
    """Veto: intvty p90 must not regress past the noise band below *anchor*.

    Both ``candidate`` and ``anchor`` must come from ``perf_snapshot_from_mapping``,
    which returns ``None`` unless ``intvty_p90`` is strictly positive. Callers
    that respect this contract will never reach this function with a zero or
    missing axis.
    """
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
    "output_tput_of",
    "parse_intvty_noise_pct",
    "passes_intvty_gate",
    "perf_snapshot_from_mapping",
    "resolve_grading_anchor_perf",
    "total_tput_grading_enabled",
    "total_tput_of",
    "total_tput_serving_grading_enabled",
]
