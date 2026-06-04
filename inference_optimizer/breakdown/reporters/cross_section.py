"""Cross-section fact synthesis used by the executive summary + LLM
prompt.

Every fact this module surfaces is computed deterministically from the
``session_breakdown.json`` dict (and the renderer outputs already
produced for each section). The LLM is forbidden from inventing
numbers; the user prompt only contains:

* the ``key_facts`` from each renderer,
* the ``decisions`` from each renderer,
* the ``GlobalFacts`` produced here.

If you find yourself wanting the LLM to "figure something out across
sections" (gain attribution, capability gating, data-quality flags),
add it here instead. The numerical guard rails are this module's
responsibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .base import RenderedSection

__all__ = ["GlobalFacts", "build_global_facts"]


@dataclass(frozen=True)
class GlobalFacts:
    """One-shot fact pack that the LLM uses to build the executive summary.

    Field naming follows the dashboard / handover-doc taxonomy so an
    LLM that has never seen this codebase still understands what
    "kernel_pipeline_funnel" or "attribution_method" mean.
    """

    headline: str                       # 1-line "baseline X → final Y = +Z%"
    stop_reason: str
    elapsed_minutes: float | None
    objective: dict[str, Any]
    workload_summary: str               # "DeepSeek-R1 vllm fp8 tp=8 conc=64 isl=osl=1024"
    gain_attribution_lines: list[str]   # "100% via 1 explore KEEP (flag_x)"
    capabilities_not_attempted: list[str]
    capabilities_kept: list[str]
    kernel_pipeline_funnel: dict[str, int]   # detected/recommended/optimized/adopted/...
    data_quality_flags: list[str]
    attribution_method: str             # "validated" | "best-effort reconstructed" | "missing"

    def as_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_pct(num: Any, denom: Any) -> float | None:
    try:
        n, d = float(num), float(denom)
    except (TypeError, ValueError):
        return None
    if d == 0:
        return None
    return (n - d) / d * 100.0


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _workload_summary(workload: dict[str, Any]) -> str:
    model = workload.get("model_name") or "(unknown-model)"
    fw = workload.get("framework") or "?"
    prec = workload.get("precision") or "?"
    tp = workload.get("tp")
    conc = workload.get("conc")
    isl = workload.get("isl")
    osl = workload.get("osl")
    return (
        f"{model} {fw} {prec} tp={tp} conc={conc} isl={isl} osl={osl}"
    )


def _gain_attribution_lines(
    breakdown: dict[str, Any],
) -> tuple[list[str], str]:
    """Compute per-source gain attribution + which method we used.

    Priority:
    1. ``attribution.source_breakdown.*_pct_of_total`` when populated
       and the sum is non-zero (validated split).
    2. ``final.action_path`` walked sequentially when only one entry
       exists (single-source attribution is unambiguous).
    3. ``optimization_stack`` reconstruction (best-effort) — flagged.
    """
    attribution = breakdown.get("attribution") or {}
    sb = attribution.get("source_breakdown") or {}
    total = _to_float(sb.get("validated_total_pct"))
    sources = {
        "explore":  _to_float(sb.get("explore_pct_of_total")),
        "geak":     _to_float(sb.get("geak_pct_of_total")),
        "oob":      _to_float(sb.get("oob_pct_of_total")),
        "sweep":    _to_float(sb.get("sweep_pct_of_total")),
    }
    # Archived sessions can still carry the old split. Keep it only when
    # the canonical explore bucket is absent so current reports do not
    # double-count the same work.
    if not sources["explore"]:
        sources["backends"] = _to_float(sb.get("backends_pct_of_total"))
        sources["params"] = _to_float(sb.get("params_pct_of_total"))
    nonzero = {k: v for k, v in sources.items() if v and v != 0}
    if nonzero and total:
        lines = [
            f"{k}: {v:.2f}% of total (={(v/total*100):.0f}% share of {total:.2f}%)"
            for k, v in sorted(nonzero.items(), key=lambda kv: -kv[1])
        ]
        return lines, "validated"

    # Fall back to final.action_path single-source statement.
    final = breakdown.get("final") or {}
    path = final.get("action_path") or []
    gain_v = _to_float(final.get("cumulative_gain_pct_validated"))
    if len(path) == 1 and gain_v is not None and gain_v > 0:
        entry = str(path[0])
        action = entry.split(":")[0]
        return (
            [f"100% via 1 {action} KEEP ({entry})"],
            "single-source (inferred from final.action_path)",
        )
    if len(path) > 1 and gain_v is not None and gain_v > 0:
        return (
            [f"{gain_v:.2f}% spread across {len(path)} stack entries: " +
             ", ".join(str(p) for p in path)],
            "best-effort reconstructed from optimization_stack",
        )
    if gain_v in (None, 0.0):
        return ([], "missing")
    return (
        [f"{gain_v:.2f}% from unknown source"],
        "best-effort reconstructed",
    )


def _kernel_funnel(breakdown: dict[str, Any]) -> dict[str, int]:
    kl = breakdown.get("kernel_lifecycle") or {}
    return {
        "detected":    len(kl.get("detected") or []),
        "recommended": len(kl.get("recommended") or []),
        "optimized":   len(kl.get("optimized") or []),
        "adopted":     len(kl.get("adopted") or []),
        "partial":     len(kl.get("partial") or []),
        "reverted":    len(kl.get("reverted") or []),
        "rejected":    len(kl.get("rejected") or []),
    }


def _data_quality_flags(
    breakdown: dict[str, Any],
    rendered: list[RenderedSection],
) -> list[str]:
    """Collect data-quality warnings from every renderer + a few global
    cross-section checks the renderers can't easily see.

    Returns de-duplicated flags (renderer-level warnings sometimes
    overlap with global-level checks, which used to produce 3 copies
    of the same telemetry note).
    """
    flags: list[str] = []
    seen: set[str] = set()

    def _push(line: str) -> None:
        if line in seen:
            return
        seen.add(line)
        flags.append(line)

    for sec in rendered:
        # Suppress warnings on sections we'll drop from the report
        # anyway — no point flagging them globally if the user never
        # sees the section.
        if sec.skipped:
            continue
        for w in sec.warnings:
            _push(f"[{sec.section_id}] {w}")

    # Global cross-section checks.
    if (breakdown.get("attribution") or {}).get("notes"):
        for n in breakdown["attribution"]["notes"]:
            _push(f"[attribution] {n}")
    cap = breakdown.get("capability_summary") or {}
    val = cap.get("validate_stack") or {}
    if val.get("status") == "not_attempted":
        # Validated cumulative gain still gets reported elsewhere, so be
        # explicit that this archived action did not re-run in-session.
        _push(
            "[legacy validate_stack] never ran — cumulative_gain_pct_validated "
            "comes from state, not a final archived-action re-run."
        )
    return flags


def _capabilities_split(
    breakdown: dict[str, Any],
) -> tuple[list[str], list[str]]:
    cap = breakdown.get("capability_summary") or {}
    kept = []
    not_attempted = []
    for name, v in cap.items():
        status = (v or {}).get("status") or "not_attempted"
        if status == "kept":
            kept.append(name)
        elif status == "not_attempted":
            not_attempted.append(name)
    return sorted(kept), sorted(not_attempted)


def _headline(breakdown: dict[str, Any]) -> str:
    b = (breakdown.get("baseline") or {}).get("throughput_tok_s_per_gpu")
    f = (breakdown.get("final") or {}).get("throughput_tok_s_per_gpu")
    g = (breakdown.get("final") or {}).get("cumulative_gain_pct_validated")
    if b and f and g is not None:
        sign = "+" if g > 0 else ""
        return f"baseline {b:.2f} → final {f:.2f} tok/s/gpu = {sign}{g:.2f}% validated gain"
    if b and not f:
        return f"baseline {b:.2f} tok/s/gpu (no validated final)"
    return "no validated throughput recorded"


def build_global_facts(
    breakdown: dict[str, Any],
    rendered: list[RenderedSection],
) -> GlobalFacts:
    workload = breakdown.get("workload") or {}
    session = breakdown.get("session") or {}
    attribution_lines, attribution_method = _gain_attribution_lines(breakdown)
    kept, not_attempted = _capabilities_split(breakdown)
    return GlobalFacts(
        headline=_headline(breakdown),
        stop_reason=str(session.get("stop_reason") or ""),
        elapsed_minutes=_to_float(session.get("elapsed_minutes")),
        objective=dict(workload.get("objective") or {}),
        workload_summary=_workload_summary(workload),
        gain_attribution_lines=attribution_lines,
        capabilities_not_attempted=not_attempted,
        capabilities_kept=kept,
        kernel_pipeline_funnel=_kernel_funnel(breakdown),
        data_quality_flags=_data_quality_flags(breakdown, rendered),
        attribution_method=attribution_method,
    )
