# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX grading: E2E normalised interactivity objective guarded by per-chip throughput.

InferenceX ranks an AgentX submission on a 2-D Pareto frontier whose axes are:
  x = E2E normalised interactivity P90  (tok/s/user, slow-tail)
  y = token throughput per chip         (total tok/s / num_chips)

It sweeps a ``conc-list`` ladder and keeps TTFT / ITL / TPOT separately; it
never collapses the axes into one weighted number; and it has no fixed
interactivity *target* — interactivity is the frontier's x-axis, so trading it
for throughput simply moves a point along the frontier rather than violating a
constraint.

E2E normalised interactivity P90 is defined per-request:
  r_i = E2EL_i / OSL_i           (seconds per output token)
  interactivity_P90 = 1 / P90({r_i})   (output tok/s/user, slow tail)

Upstream takes the percentile in seconds-per-token *before* inverting to
preserve the slow-tail interpretation (``MODELS.md:78``).  In aiperf's export
``e2e_output_token_throughput`` is the per-request rate ``OSL / E2EL_s`` with
``LARGER_IS_BETTER``; its P90 is therefore the *fastest* decile, not the
slowest.  The slow tail is **P10** of that rate, which equals ``1/P90(ratio)``.

Local KEEP rule (fixed concurrency, no ladder):
  KEEP     — interactivity gain >= keep_threshold_pct AND tok/s/chip not worse
             beyond the noise band
  REVERT   — both axes worse
  RECORDED — neither dominates (measured and stored; not promoted to stack)

Default-on for AgentX runs and off otherwise; ``HYPERLOOM_PERF_METRIC``
overrides either way.  Serving only; scriptable frameworks keep output-tput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.common.env import env_bool, env_str

# Replaces the retired ``composite_v1`` knob.
INTVTY_V1 = "intvty_v1"

# Read by name because ``common/`` must not import the orchestrator, where
# ``agentx_enabled`` lives.  Both resolve ``common.env._TRUE_TOKENS``.
_AGENTX_ENV = "HYPERLOOM_AGENTX"

# The value ``SharedState.benchmark_mode`` carries for an AgentX session,
# stamped at seed so it outlives the shell that started the run.
_AGENTX_MODE = "agentx"


def is_agentx_mode(benchmark_mode: Any) -> bool:
    """Whether a ``benchmark_mode`` names the agentic workload."""
    return str(benchmark_mode or "").strip().lower() == _AGENTX_MODE


# Default noise band for both axes.  Calibrated on upstream run-to-run reports
# of 1-5%; the slow-tail variance is unmeasured — tracked as debt.
_DEFAULT_INTVTY_NOISE_PCT = 5.0

# Minimum KEEP threshold applied to AgentX sessions so a 1% bump inside the
# noise band is never promoted.  Underlying variance is unmeasured (debt).
AGENTX_KEEP_THRESHOLD_FLOOR_PCT = 2.0


def intvty_grading_enabled(*, benchmark_mode: str = "") -> bool:
    """True when E2E-normalised-interactivity grading applies.

    ``HYPERLOOM_PERF_METRIC`` wins first; if set, returns True only when it
    equals ``intvty_v1``.  When unset, AgentX decides by either signal: the
    ambient ``HYPERLOOM_AGENTX`` or the session's persisted ``benchmark_mode``.

    ``benchmark_mode`` is a parameter so ``hyperloom.common`` need not import
    the orchestrator.  It is stamped at seed precisely so it survives a
    restart; the env var only describes the shell that happens to be running.
    Mirrors ``_workload_envs.agentx_active``.
    """
    raw = env_str("HYPERLOOM_PERF_METRIC").strip().lower()
    if raw:
        return raw == INTVTY_V1
    if env_bool(_AGENTX_ENV):
        return True
    return is_agentx_mode(benchmark_mode)


# Keep ``total_tput_grading_enabled`` as a deprecated alias so existing callers
# continue to compile.  Remove after every call site migrates.
def total_tput_grading_enabled(*, benchmark_mode: str = "") -> bool:  # noqa: D401
    """Deprecated: use :func:`intvty_grading_enabled`."""
    return intvty_grading_enabled(benchmark_mode=benchmark_mode)


GRADED_INTVTY = "e2e_norm_intvty_p90"  # the primary objective axis
GRADED_TOTAL = "total_throughput"  # secondary guard axis (tok/s, unnormalized)
GRADED_OUTPUT = "output_throughput"  # synthetic-mode axis


def graded_metric_key(*, benchmark_mode: str = "") -> str:
    """The curve-row field a session's speedups are measured on.

    Returns the primary interactivity field for AgentX and output throughput
    for synthetic, matching what the conc-sweep plotter reads.
    """
    if intvty_grading_enabled(benchmark_mode=benchmark_mode):
        return GRADED_INTVTY
    return GRADED_OUTPUT


def intvty_serving_grading_enabled(*, scriptable: bool = False, benchmark_mode: str = "") -> bool:
    """Interactivity grading limited to non-scriptable serving runs."""
    return intvty_grading_enabled(benchmark_mode=benchmark_mode) and not scriptable


# Deprecated alias.
def total_tput_serving_grading_enabled(*, scriptable: bool = False, benchmark_mode: str = "") -> bool:  # noqa: D401
    """Deprecated: use :func:`intvty_serving_grading_enabled`."""
    return intvty_serving_grading_enabled(scriptable=scriptable, benchmark_mode=benchmark_mode)


def parse_intvty_noise_pct() -> float:
    """Noise band in percent from ``HYPERLOOM_PERF_NOISE_PCT`` (default 5)."""
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
    """Extract the graded pair from a measurement or ``current_best`` record.

    For AgentX the pair is
    ``(e2e_norm_intvty_p90, total_throughput)``.  Returns ``None`` unless both
    are strictly positive; persisted records carry explicit nulls for axes a
    framework never measured.

    For compatibility the snapshot also carries ``output_throughput``,
    ``input_throughput``, and ``tpot_p90_ms`` when present.
    """
    if not isinstance(source, Mapping):
        return None
    intvty = _positive(source.get("e2e_norm_intvty_p90"))
    inp = _positive(source.get("input_throughput"))
    out = _positive(source.get("output_throughput")) or _positive(source.get("tput"))
    total = _positive(source.get("total_throughput")) or _positive(source.get("total_token_throughput"))
    if total is None and inp is not None and out is not None:
        total = inp + out
    if intvty is None or total is None:
        return None
    snap: dict[str, float] = {"e2e_norm_intvty_p90": intvty, "total_throughput": total}
    for key, value in (
        ("input_throughput", inp),
        ("output_throughput", out),
        ("tpot_p90_ms", _positive(source.get("tpot_p90_ms"))),
    ):
        if value is not None:
            snap[key] = value
    return snap


def output_tput_of(source: Mapping[str, Any] | None) -> float:
    """Output throughput from a measurement or ``current_best``; 0.0 when absent."""
    if not isinstance(source, Mapping):
        return 0.0
    return float(_positive(source.get("output_throughput")) or _positive(source.get("tput")) or 0.0)


def graded_axes_of(source: Mapping[str, Any] | None) -> dict[str, float]:
    """Axes a KEEP record must carry so the next candidate anchors correctly.

    Absent rather than ``None`` for unmeasured axes so a partial record is not
    mistaken for a measured zero.
    """
    if not isinstance(source, Mapping):
        return {}
    axes: dict[str, float] = {}
    intvty = _positive(source.get("e2e_norm_intvty_p90"))
    if intvty is not None:
        axes["e2e_norm_intvty_p90"] = intvty
    total = _positive(source.get("total_throughput")) or _positive(source.get("total_token_throughput"))
    if total is not None:
        axes["total_throughput"] = total
    for key in ("input_throughput", "tpot_p90_ms", "output_throughput"):
        value = _positive(source.get(key))
        if value is not None:
            axes[key] = value
    return axes


def intvty_of(snapshot: Mapping[str, float] | None) -> float:
    """E2E-normalised interactivity P90 from a perf snapshot; 0.0 when absent."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get("e2e_norm_intvty_p90") or 0.0)


def total_tput_of(snapshot: Mapping[str, float] | None) -> float:
    """Total token throughput from a perf snapshot; 0.0 when unavailable."""
    if not isinstance(snapshot, Mapping):
        return 0.0
    return float(snapshot.get("total_throughput") or 0.0)


def resolve_grading_anchor_perf(state: Any) -> tuple[dict[str, float] | None, str]:
    """Interactivity-axis grading anchor: current-best falling back to baseline.

    - ``current_best`` non-empty and axes present: return its snapshot.
    - ``current_best`` non-empty but axes absent:
      ``(None, "current_best_axes_missing")``.  Must not fall through to
      ``baseline_perf`` — that would anchor against a recipe never measured.
    - ``current_best`` empty: snapshot ``baseline_perf``; failure returns
      ``(None, "baseline_perf_missing")``.

    Returns:
        ``(snapshot, reason)`` where ``reason`` is ``""`` on success and a
        short tag when no usable anchor exists.
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
    """True when candidate interactivity is not worse than anchor within the band.

    Called by the 2-D domination check: a candidate that improves interactivity
    but is within the noise band on the anchor does not trigger REVERT.  Both
    arguments must come from :func:`perf_snapshot_from_mapping` so
    ``e2e_norm_intvty_p90`` is guaranteed positive.
    """
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    anchor_intv = float(anchor.get("e2e_norm_intvty_p90") or 0.0)
    cand_intv = float(candidate.get("e2e_norm_intvty_p90") or 0.0)
    return cand_intv >= anchor_intv * (1.0 - band / 100.0)


def passes_tput_guard(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    *,
    noise_pct: float | None = None,
) -> bool:
    """True when candidate per-chip throughput is not worse than anchor within the band.

    The second axis of the 2-D domination check.  Uses the same noise band as
    the interactivity gate; ``total_throughput`` is the raw aggregate — the
    caller must normalise by ``tp`` before comparing configurations with
    different tensor-parallel degree.
    """
    band = float(noise_pct if noise_pct is not None else parse_intvty_noise_pct())
    anchor_total = float(anchor.get("total_throughput") or 0.0)
    cand_total = float(candidate.get("total_throughput") or 0.0)
    if anchor_total <= 0:
        return True  # no reference — cannot regress
    return cand_total >= anchor_total * (1.0 - band / 100.0)


# Outcome literals for the 2-D verdict.
VERDICT_KEEP = "KEEP"
VERDICT_REVERT = "REVERT"
VERDICT_RECORDED = "RECORDED"  # neither dominates — store but do not promote


@dataclass(frozen=True)
class GradedComparison:
    """A candidate and the figure it must beat, both read off one axis.

    Under AgentX the primary axis is ``GRADED_INTVTY``; the secondary guard
    axis is ``GRADED_TOTAL`` and is carried as ``tput_candidate`` /
    ``tput_reference``.  Under synthetic workloads ``objective`` is
    ``GRADED_OUTPUT`` and the AgentX fields are zero.

    Attributes:
        objective: The axis KEEP/REVERT is decided on.
        candidate: The measured candidate on ``objective``; 0.0 when absent.
        reference: The anchor on ``objective``; 0.0 when absent.
        verdict: ``KEEP``, ``REVERT``, or ``RECORDED`` (AgentX 2-D rule).
        tput_candidate: Candidate ``total_throughput`` (guard axis); 0.0 when N/A.
        tput_reference: Reference ``total_throughput``; 0.0 when N/A.
        degrade_reason: Why the AgentX axis did not apply; ``""`` when it did.
    """

    objective: str
    candidate: float
    reference: float
    verdict: str = VERDICT_REVERT
    tput_candidate: float = 0.0
    tput_reference: float = 0.0
    degrade_reason: str = ""

    @property
    def graded_on_total(self) -> bool:
        """Whether the AgentX interactivity objective actually applied."""
        return self.objective == GRADED_INTVTY

    @property
    def vetoed(self) -> bool:
        """True when the candidate was rejected (REVERT).

        Kept for backwards-compat with callers that used ``graded.vetoed``
        under the old single-axis rule.
        """
        return self.verdict == VERDICT_REVERT


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
    "total_tput_grading_enabled",
    "total_tput_of",
    "total_tput_serving_grading_enabled",
]
