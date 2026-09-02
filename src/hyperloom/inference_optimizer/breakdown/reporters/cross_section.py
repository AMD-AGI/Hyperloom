# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-section fact synthesis for the executive summary + LLM prompt.

All facts are computed deterministically here (the LLM never invents
numbers); any cross-section reasoning belongs in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from hyperloom.common.coerce import to_float

from .base import RenderedSection, as_dict

__all__ = ["GlobalFacts", "build_global_facts"]


@dataclass(frozen=True)
class GlobalFacts:
    """One-shot fact pack the LLM uses to build the executive summary."""

    headline: str  # 1-line "baseline X → final Y = +Z%"
    stop_reason: str
    elapsed_minutes: float | None
    objective: dict[str, Any]
    workload_summary: str  # "DeepSeek-R1 vllm fp8 tp=8 conc=64 isl=osl=1024"
    gain_attribution_lines: list[str]  # "explore: 5.00% of total (=100% share of 5.00%)"
    capabilities_not_attempted: list[str]
    capabilities_kept: list[str]
    kernel_pipeline_funnel: dict[str, int]  # detected/recommended/optimized/adopted/...
    data_quality_flags: list[str]
    attribution_method: str  # "validated" | "unattributed" | "unattributed (stack listed for reference, not verified as KEEP)" | "missing"

    def as_prompt_dict(self) -> dict[str, Any]:
        """Serialize the fact pack to a plain dict for the LLM prompt.

        Returns:
            dict[str, Any]: All fields of this dataclass as a JSON-friendly
                mapping (via ``dataclasses.asdict``).
        """
        return asdict(self)


def _workload_summary(workload: dict[str, Any]) -> str:
    """Build a compact one-line description of the workload.

    Args:
        workload (dict[str, Any]): The ``workload`` section of the breakdown,
            with keys such as ``model_name``, ``framework_name``, ``precision``,
            ``tp``, ``conc``, ``isl`` and ``osl``.

    Returns:
        str: A single line like ``"DeepSeek-R1 vllm fp8 tp=8 conc=64 ..."``,
            using placeholders for any missing fields.
    """
    model = workload.get("model_name") or "(unknown-model)"
    fw = workload.get("framework_name") or "?"
    prec = workload.get("precision") or "?"
    tp = workload.get("tp")
    conc = workload.get("conc")
    isl = workload.get("isl")
    osl = workload.get("osl")
    return f"{model} {fw} {prec} tp={tp} conc={conc} isl={isl} osl={osl}"


def _gain_attribution_lines(
    breakdown: dict[str, Any],
) -> tuple[list[str], str]:
    """Compute per-source gain attribution + the method used.

    Priority: canonical ``optimizations.summary_by_source``. Legacy
    ``attribution.source_breakdown`` remains a read-only fallback for archived
    breakdowns supplied directly to the reporter.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        A tuple of the human-readable attribution lines and a label
        describing the method used to derive them.
    """
    optimizations = as_dict(breakdown.get("optimizations"))
    summary = as_dict(optimizations.get("summary_by_source"))
    validation = as_dict(optimizations.get("validation"))
    canonical_sources = {
        source: to_float(bucket.get("total_gain_pct")) for source, bucket in summary.items() if isinstance(bucket, dict)
    }
    canonical_nonzero = {source: gain for source, gain in canonical_sources.items() if gain and gain != 0}
    canonical_total = sum(canonical_nonzero.values())
    if canonical_nonzero and canonical_total:
        lines = [
            f"{source}: {gain:.2f}% of total (={(gain / canonical_total * 100):.0f}% share of {canonical_total:.2f}%)"
            for source, gain in sorted(
                canonical_nonzero.items(),
                key=lambda item: -item[1],
            )
        ]
        method = validation.get("method")
        return lines, str(method) if isinstance(method, str) and method else "missing"

    attribution = as_dict(breakdown.get("attribution"))
    sb = as_dict(attribution.get("source_breakdown"))
    total = to_float(sb.get("validated_total_pct"))
    sources = {
        "backends": to_float(sb.get("backends_pct_of_total")),
        "params": to_float(sb.get("params_pct_of_total")),
        "explore": to_float(sb.get("explore_pct_of_total")),
        "replay_warm_recipe": to_float(sb.get("replay_warm_recipe_pct_of_total")),
        "geak": to_float(sb.get("geak_pct_of_total")),
        "sweep": to_float(sb.get("sweep_pct_of_total")),
    }
    nonzero = {k: v for k, v in sources.items() if v and v != 0}
    if nonzero and total:
        lines = [
            f"{k}: {v:.2f}% of total (={(v / total * 100):.0f}% share of {total:.2f}%)"
            for k, v in sorted(nonzero.items(), key=lambda kv: -kv[1])
        ]
        return lines, "validated"

    final = as_dict(breakdown.get("final"))
    path = final.get("action_path") or []
    gain_v = to_float(final.get("cumulative_gain_pct_validated"))
    # No validated per-source split. We must NOT claim a KEEP or "100% via"
    # from optimization_stack alone: action_path is built from the final stack,
    # which can include seeded / warm-replayed entries that were never a real
    # this-session KEEP. Surface the gain as unattributed and list the stack
    # entries only for reference (never as adoption evidence).
    if path and gain_v is not None and gain_v > 0:
        n = len(path)
        listed = ", ".join(str(p) for p in path)
        return (
            [
                f"{gain_v:.2f}% total gain, source unattributed "
                f"(no validated source_breakdown); optimization_stack lists "
                f"{n} entr{'y' if n == 1 else 'ies'} for reference: {listed}"
            ],
            "unattributed (stack listed for reference, not verified as KEEP)",
        )
    if gain_v in (None, 0.0):
        return ([], "missing")
    return (
        [f"{gain_v:.2f}% total gain, source unattributed"],
        "unattributed",
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
    kl = as_dict(breakdown.get("kernel_lifecycle"))
    return {
        "detected": len(kl.get("detected") or []),
        "recommended": len(kl.get("recommended") or []),
        "optimized": len(kl.get("optimized") or []),
        "adopted": len(kl.get("adopted") or []),
        "partial": len(kl.get("partial") or []),
        "reverted": len(kl.get("reverted") or []),
        "rejected": len(kl.get("rejected") or []),
    }


# Sections whose renderer is registered but whose data nothing produces: no
# collector, exporter key or recorder fragment ever fills them. They are always
# skipped, so surfacing that as a data-quality flag would tell the reader this
# run came up short when the truth is that the feature was never wired up --
# and four permanent flags would drown the ones that do describe the session.
# ``test_breakdown_report_integrity`` recomputes this set from the real
# exporter + recorder key space, so wiring a producer up (or adding another
# dead section) fails the suite instead of silently drifting.
_SECTIONS_WITHOUT_PRODUCER = frozenset(
    {
        "data_provenance",
        "decision_journal",
        "kernel_decision_path",
        "kernel_profiling",
    }
)


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
        if sec.skipped:
            if sec.section_id in _SECTIONS_WITHOUT_PRODUCER:
                continue
            # A dropped section still carries evidence: "this never ran" is a
            # data-quality fact, and discarding it lets a reader mistake an
            # untested area for a clean one. Prefer the renderer's warnings,
            # fall back to its key facts, and state the absence either way.
            # Both, not either: a renderer that logged a warning may still
            # carry the key fact that explains it, and ``or`` would drop it.
            evidence = [*sec.warnings, *sec.key_facts]
            for line in evidence:
                _push(f"[{sec.section_id}] skipped: {line}")
            if not evidence:
                _push(f"[{sec.section_id}] skipped: no data recorded this session.")
            continue
        for w in sec.warnings:
            _push(f"[{sec.section_id}] {w}")

    optimizations = as_dict(breakdown.get("optimizations"))
    validation = as_dict(optimizations.get("validation"))
    attribution = as_dict(breakdown.get("attribution"))
    notes = validation.get("notes") or attribution.get("notes") or []
    if notes:
        for n in notes:
            _push(f"[attribution] {n}")
    cap = as_dict(breakdown.get("capability_summary"))
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
    cap = as_dict(breakdown.get("capability_summary"))
    kept = []
    not_attempted = []
    for name, v in cap.items():
        status = as_dict(v).get("status") or "not_attempted"
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
    from ... import framework_registry

    fw = as_dict(breakdown.get("workload")).get("framework_name")
    b = to_float(as_dict(breakdown.get("baseline")).get("throughput_tok_s_per_gpu"))
    f = to_float(as_dict(breakdown.get("final")).get("throughput_tok_s_per_gpu"))
    g = to_float(as_dict(breakdown.get("final")).get("cumulative_gain_pct_validated"))
    if b and f and g is not None:
        sign = "+" if g > 0 else ""
        return (
            f"baseline {framework_registry.format_primary_metric(fw, b, precision=2)} → "
            f"final {framework_registry.format_primary_metric(fw, f, precision=2)} "
            f"= {sign}{g:.2f}% validated gain"
        )
    if b and not f:
        return f"baseline {framework_registry.format_primary_metric(fw, b, precision=2)} (no validated final)"
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
    workload = as_dict(breakdown.get("workload"))
    session = as_dict(breakdown.get("session"))
    attribution_lines, attribution_method = _gain_attribution_lines(breakdown)
    kept, not_attempted = _capabilities_split(breakdown)
    return GlobalFacts(
        headline=_headline(breakdown),
        stop_reason=str(session.get("stop_reason") or ""),
        elapsed_minutes=to_float(session.get("elapsed_minutes")),
        objective=as_dict(workload.get("objective")),
        workload_summary=_workload_summary(workload),
        gain_attribution_lines=attribution_lines,
        capabilities_not_attempted=not_attempted,
        capabilities_kept=kept,
        kernel_pipeline_funnel=_kernel_funnel(breakdown),
        data_quality_flags=_data_quality_flags(breakdown, rendered),
        attribution_method=attribution_method,
    )
