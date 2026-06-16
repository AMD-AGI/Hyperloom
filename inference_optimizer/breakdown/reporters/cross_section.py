# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Cross-section fact synthesis for the executive summary + LLM prompt.

All facts are computed deterministically here (the LLM never invents
numbers); any cross-section reasoning belongs in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .base import RenderedSection

__all__ = ["GlobalFacts", "build_global_facts"]


@dataclass(frozen=True)
class GlobalFacts:
    """One-shot fact pack the LLM uses to build the executive summary."""

    headline: str                       # 1-line "baseline X → final Y = +Z%"
    stop_reason: str
    elapsed_minutes: float | None
    objective: dict[str, Any]
    workload_summary: str               # "DeepSeek-R1 vllm fp8 tp=8 conc=64 isl=osl=1024"
    gain_attribution_lines: list[str]   # "100% via 1 backends KEEP (flag_x)"
    capabilities_not_attempted: list[str]
    capabilities_kept: list[str]
    kernel_pipeline_funnel: dict[str, int]   # detected/recommended/optimized/adopted/...
    data_quality_flags: list[str]
    attribution_method: str             # "validated" | "best-effort reconstructed" | "missing"

    def as_prompt_dict(self) -> dict[str, Any]:
        """Serialize the fact pack to a plain dict for the LLM prompt.

        Returns:
            dict[str, Any]: All fields of this dataclass as a JSON-friendly
                mapping (via ``dataclasses.asdict``).
        """
        return asdict(self)


def _safe_pct(num: Any, denom: Any) -> float | None:
    """Compute a signed percentage change, tolerating bad inputs.

    Args:
        num (Any): The new value (numerator); coerced to float.
        denom (Any): The baseline value (denominator); coerced to float.

    Returns:
        float | None: ``(num - denom) / denom * 100`` as a percent, or
            ``None`` if either value is non-numeric or the denominator is 0.
    """
    try:
        n, d = float(num), float(denom)
    except (TypeError, ValueError):
        return None
    if d == 0:
        return None
    return (n - d) / d * 100.0


def _to_float(v: Any) -> float | None:
    """Coerce a value to float, returning ``None`` instead of raising.

    Args:
        v (Any): The value to coerce.

    Returns:
        float | None: The float value, or ``None`` if ``v`` is ``None`` or
            cannot be converted.
    """
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _workload_summary(workload: dict[str, Any]) -> str:
    """Build a compact one-line description of the workload.

    Args:
        workload (dict[str, Any]): The ``workload`` section of the breakdown,
            with keys such as ``model_name``, ``framework``, ``precision``,
            ``tp``, ``conc``, ``isl`` and ``osl``.

    Returns:
        str: A single line like ``"DeepSeek-R1 vllm fp8 tp=8 conc=64 ..."``,
            using placeholders for any missing fields.
    """
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
    """Compute per-source gain attribution + the method used.

    Priority: validated ``source_breakdown`` split, then single-entry
    ``final.action_path``, then best-effort ``optimization_stack``.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        A tuple of the human-readable attribution lines and a label
        describing the method used to derive them.
    """
    attribution = breakdown.get("attribution") or {}
    sb = attribution.get("source_breakdown") or {}
    total = _to_float(sb.get("validated_total_pct"))
    sources = {
        "backends": _to_float(sb.get("backends_pct_of_total")),
        "params":   _to_float(sb.get("params_pct_of_total")),
        "explore":  _to_float(sb.get("explore_pct_of_total")),
        "geak":     _to_float(sb.get("geak_pct_of_total")),
        "oob":      _to_float(sb.get("oob_pct_of_total")),
        "sweep":    _to_float(sb.get("sweep_pct_of_total")),
    }
    nonzero = {k: v for k, v in sources.items() if v and v != 0}
    if nonzero and total:
        lines = [
            f"{k}: {v:.2f}% of total (={(v/total*100):.0f}% share of {total:.2f}%)"
            for k, v in sorted(nonzero.items(), key=lambda kv: -kv[1])
        ]
        return lines, "validated"

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
    """Count kernels at each stage of the optimization lifecycle.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        dict[str, int]: Counts keyed by lifecycle stage (``detected``,
            ``recommended``, ``optimized``, ``adopted``, ``partial``,
            ``reverted`` and ``rejected``).
    """
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
    """Collect de-duplicated data-quality warnings from renderers + global cross-section checks.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.
        rendered: Rendered sections whose warnings are folded in.

    Returns:
        A de-duplicated list of data-quality flag strings.
    """
    flags: list[str] = []
    seen: set[str] = set()

    def _push(line: str) -> None:
        """Append ``line`` to ``flags`` once, de-duplicating via ``seen``.

        Args:
            line (str): The flag text to record.
        """
        if line in seen:
            return
        seen.add(line)
        flags.append(line)

    for sec in rendered:
        # Skip dropped sections; no point flagging what the user won't see.
        if sec.skipped:
            continue
        for w in sec.warnings:
            _push(f"[{sec.section_id}] {w}")

    if (breakdown.get("attribution") or {}).get("notes"):
        for n in breakdown["attribution"]["notes"]:
            _push(f"[attribution] {n}")
    cap = breakdown.get("capability_summary") or {}
    val = cap.get("validate_stack") or {}
    if val.get("status") == "not_attempted":
        # Be explicit that this archived action did not re-run in-session.
        _push(
            "[legacy validate_stack] never ran — cumulative_gain_pct_validated "
            "comes from state, not a final archived-action re-run."
        )
    return flags


def _capabilities_split(
    breakdown: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Split capabilities into those kept vs. never attempted.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        tuple[list[str], list[str]]: A sorted list of capability names with
            status ``"kept"`` and a sorted list with status ``"not_attempted"``.
    """
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
    """Build the one-line baseline→final throughput headline.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        str: A line such as ``"baseline X → final Y tok/s/gpu = +Z% validated
            gain"``, or a fallback message when validated throughput is missing.
    """
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
    """Assemble the deterministic :class:`GlobalFacts` pack for the LLM.

    This is the single entry point that combines the workload summary,
    headline, gain attribution, capability split, kernel funnel and
    data-quality flags into one fact pack.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.
        rendered (list[RenderedSection]): The already-rendered sections, used
            to gather per-section data-quality warnings.

    Returns:
        GlobalFacts: The populated, frozen fact pack.
    """
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
