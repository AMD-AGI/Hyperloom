# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-section fact synthesis for the executive summary + LLM prompt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from hyperloom.common.coerce import to_float

from .base import RenderedSection, as_dict, outcome_of, session_of, stop_reason_of, task_config_of, validation_of

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
    # "stack_ledger" | "unattributed" | "unattributed (stack listed for
    # reference, not verified as KEEP)" | "missing"
    attribution_method: str

    def as_prompt_dict(self) -> dict[str, Any]:
        """Serialize the fact pack to a plain dict for the LLM prompt."""
        return asdict(self)


def _workload_summary(workload: dict[str, Any]) -> str:
    """Build a compact one-line description of the workload."""
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

    Read off ``outcome.validation``, the stack ledger's own account of itself.
    A run whose ledger holds no measurable contribution falls back to naming
    the total as unattributed and listing the final stack for reference.

    Args:
        breakdown: The full ``session_breakdown.json`` dict.

    Returns:
        A tuple of the human-readable attribution lines and a label
        describing the method used to derive them.
    """
    validation = validation_of(breakdown)
    summary = as_dict(as_dict(validation.get("attribution")).get("by_source"))
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
        # Every contribution is measured against the session baseline, so the
        # split is the ledger's own arithmetic rather than one of several
        # attribution methods the section used to have to name.
        return lines, "stack_ledger"

    final = as_dict(outcome_of(breakdown).get("final"))
    path = final.get("action_path") or []
    gain_v = to_float(final.get("gain_pct"))
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

    Counted off the same per-kernel rollup the lifecycle section renders, so
    the funnel in the summary and the table below it cannot disagree.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        dict[str, int]: Counts keyed by lifecycle stage (``detected``,
            ``recommended``, ``optimized``, ``adopted``, ``partial``,
            ``reverted`` and ``rejected``).
    """
    from ._renderers._kernels import kernel_rows

    rows = kernel_rows(breakdown)
    decisions = [str(row.get("final_decision") or "") for row in rows]
    return {
        "detected": len(rows),
        "recommended": sum(1 for row in rows if row.get("selected_for_optimization")),
        "optimized": sum(1 for row in rows if row.get("geak") or row.get("forge")),
        "adopted": decisions.count("kept"),
        # A candidate a route measured and nothing gated: the win is real at
        # the micro level and was never ruled on end to end.
        "partial": decisions.count("attempted"),
        "reverted": decisions.count("reverted"),
        "rejected": decisions.count("rejected"),
    }


def _data_quality_flags(
    breakdown: dict[str, Any],
    rendered: list[RenderedSection],
) -> list[str]:
    """Collect de-duplicated data-quality warnings from renderers + global cross-section checks."""
    flags: list[str] = []
    seen: set[str] = set()

    def _push(line: str) -> None:
        """Append ``line`` to ``flags`` once, de-duplicating via ``seen``."""
        if line in seen:
            return
        seen.add(line)
        flags.append(line)

    for sec in rendered:
        if sec.skipped:
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

    for note in validation_of(breakdown).get("notes") or []:
        _push(f"[attribution] {note}")
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
    from ._renderers.capability_summary import capability_rows

    kept = []
    not_attempted = []
    for name, row in capability_rows(breakdown).items():
        status = str(row.get("status") or "not_attempted")
        if status == "kept":
            kept.append(name)
        elif status == "not_attempted":
            not_attempted.append(name)
    return sorted(kept), sorted(not_attempted)


def _headline(breakdown: dict[str, Any]) -> str:
    """Build the one-line baseline→final throughput headline."""
    from ... import framework_registry

    fw = task_config_of(breakdown).get("framework_name")
    outcome = outcome_of(breakdown)
    b = to_float(as_dict(outcome.get("baseline")).get("throughput_tok_s_per_gpu"))
    f = to_float(as_dict(outcome.get("final")).get("throughput_tok_s_per_gpu"))
    g = to_float(as_dict(outcome.get("final")).get("gain_pct"))
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
    workload = task_config_of(breakdown)
    session = session_of(breakdown)
    attribution_lines, attribution_method = _gain_attribution_lines(breakdown)
    kept, not_attempted = _capabilities_split(breakdown)
    return GlobalFacts(
        headline=_headline(breakdown),
        stop_reason=stop_reason_of(breakdown),
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
