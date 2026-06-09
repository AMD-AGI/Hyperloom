# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Top-level composer: run every renderer, optionally call the LLM, stitch to one markdown doc.

:func:`render_session_report` is the main API. The compose layer is
section-agnostic (walks :data:`base.REGISTRY`), and degrades to
deterministic-only output when ``llm_client`` is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .base import REGISTRY, RenderedSection
from .cross_section import GlobalFacts, build_global_facts
from .llm_prompt import SYSTEM_PROMPT, build_user_prompt, parse_llm_response

# Import every renderer module for its @register_renderer side effect.
# REGISTRY order is decoupled from layout; :data:`SECTION_GROUPS` drives it.
from ._renderers import (   # noqa: F401  (side-effect imports)
    session as _r_session,
    workload as _r_workload,
    baseline as _r_baseline,
    final as _r_final,
    capability_summary as _r_capability_summary,
    phase_timeline as _r_phase_timeline,
    kernel_lifecycle as _r_kernel_lifecycle,
    kernel_profiling as _r_kernel_profiling,
    kernel_decision_path as _r_kernel_decision_path,
    roofline as _r_roofline,
    invocations as _r_invocations,
    param_search as _r_param_search,
    decision_journal as _r_decision_journal,
    sweep as _r_sweep,
    critic_robustness as _r_critic_robustness,
    attribution as _r_attribution,
    source_files as _r_source_files,
    data_provenance as _r_data_provenance,
)


# Final report layout ``(group_title, [section_id, ...])``, mirroring the
# orchestration phases. ``telemetry`` is dropped (GPU monitor data is
# consistently broken on real wekafs sessions).
SECTION_GROUPS: list[tuple[str, list[str]]] = [
    ("Session & Workload",          ["session", "workload"]),
    ("Performance Results",          ["baseline", "final", "roofline", "attribution"]),
    ("Capability Search",            ["capability_summary",
                                      "param_search",
                                      "decision_journal",
                                      "sweep"]),
    ("Kernel Optimization",          ["kernel_lifecycle",
                                      "kernel_profiling",
                                      "kernel_decision_path",
                                      "geak_invocations",
                                      "oob_invocations",
                                      "critic_robustness"]),
    ("Run Trace",                    ["phase_timeline"]),
    # source_files lists synthesis inputs; data_provenance explains per-section coverage.
    ("Source Artifacts",             ["source_files", "data_provenance"]),
]

__all__ = [
    "LLMClient",
    "ComposeResult",
    "render_session_report",
]


class LLMClient(Protocol):
    """Minimal LLM client interface (``(system, user) -> str``); a Protocol so tests can mock it."""

    def complete(self, *, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class ComposeResult:
    """Full output of one compose run (``markdown`` plus debug/replay fields)."""

    markdown: str
    sections: list[RenderedSection]
    global_facts: GlobalFacts
    llm_user_prompt: str
    llm_raw_response: str
    used_llm: bool


def render_session_report(
    breakdown: dict[str, Any],
    *,
    llm_client: LLMClient | None = None,
) -> ComposeResult:
    """Render ``breakdown`` (a parsed session_breakdown.json) to markdown."""
    sections = [fn(breakdown) for _sid, fn in REGISTRY]
    global_facts = build_global_facts(breakdown, sections)
    user_prompt = build_user_prompt(sections, global_facts)

    llm_raw = ""
    narratives: dict[str, str] = {}
    exec_summary_llm = ""
    if llm_client is not None:
        try:
            llm_raw = llm_client.complete(system=SYSTEM_PROMPT, user=user_prompt)
            parsed = parse_llm_response(llm_raw)
            exec_summary_llm = parsed["executive_summary"]
            narratives = parsed["section_narratives"]
        except Exception as exc:  # noqa: BLE001
            llm_raw = f"<llm_error: {type(exc).__name__}: {exc}>"

    md = _stitch(
        sections=sections,
        global_facts=global_facts,
        llm_exec_summary=exec_summary_llm,
        llm_narratives=narratives,
        used_llm=llm_client is not None and not llm_raw.startswith("<llm_error"),
        breakdown=breakdown,
    )
    return ComposeResult(
        markdown=md,
        sections=sections,
        global_facts=global_facts,
        llm_user_prompt=user_prompt,
        llm_raw_response=llm_raw,
        used_llm=llm_client is not None,
    )


# Stitching
def _stitch(
    *,
    sections: list[RenderedSection],
    global_facts: GlobalFacts,
    llm_exec_summary: str,
    llm_narratives: dict[str, str],
    used_llm: bool,
    breakdown: dict[str, Any],
) -> str:
    session = breakdown.get("session") or {}
    title = f"# Hyperloom Session Report — {session.get('session_id') or '(no session_id)'}"

    parts: list[str] = [title, ""]

    parts.append("## Executive Summary")
    if used_llm and llm_exec_summary:
        parts.append(llm_exec_summary)
    else:
        parts.append(_deterministic_exec_summary(global_facts))
    parts.append("")

    parts.append("## Key Facts (deterministic)")
    parts.append(_render_global_facts_block(global_facts))
    parts.append("")

    by_id = {s.section_id: s for s in sections}
    for group_title, ids in SECTION_GROUPS:
        live = [by_id[sid] for sid in ids
                if sid in by_id and not by_id[sid].skipped]
        if not live:
            # All sections skipped → drop the H2 header too.
            continue
        parts.append(f"## {group_title}")
        parts.append("")
        for sec in live:
            parts.append(f"### {sec.title}")
            narrative = llm_narratives.get(sec.section_id, "").strip()
            if narrative:
                parts.append(narrative)
                parts.append("")
            if sec.markdown_block:
                parts.append(sec.markdown_block)
                parts.append("")
            if sec.warnings:
                parts.append("> **Data quality notes for this section**:")
                for w in sec.warnings:
                    parts.append(f"> - {w}")
                parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _deterministic_exec_summary(g: GlobalFacts) -> str:
    """Fallback exec summary when no LLM is configured / it failed; lists every data-quality flag."""
    out: list[str] = []
    out.append(f"- {g.headline} (stop_reason={g.stop_reason or 'unset'}, "
               f"elapsed={g.elapsed_minutes or 0:.0f}min, objective={g.objective}).")
    out.append(f"- Workload: {g.workload_summary}.")
    if g.gain_attribution_lines:
        out.append(f"- Gain attribution ({g.attribution_method}): "
                   + "; ".join(g.gain_attribution_lines))
    else:
        out.append(f"- No measurable gain recorded ({g.attribution_method}).")
    funnel = g.kernel_pipeline_funnel
    out.append(
        f"- Kernel pipeline: detected={funnel['detected']} → "
        f"recommended={funnel['recommended']} → optimized={funnel['optimized']} → "
        f"adopted={funnel['adopted']} (reverted={funnel['reverted']}, "
        f"rejected={funnel['rejected']}, partial={funnel['partial']})."
    )
    if g.capabilities_not_attempted:
        out.append("- Capabilities never invoked: "
                   + ", ".join(f"`{c}`" for c in g.capabilities_not_attempted) + ".")
    if g.data_quality_flags:
        out.append("- **Data quality flags**:")
        for f in g.data_quality_flags:
            out.append(f"  - {f}")
    return "\n".join(out)


def _render_global_facts_block(g: GlobalFacts) -> str:
    """Render :class:`GlobalFacts` as a compact key-value block for cross-checking the report."""
    funnel = g.kernel_pipeline_funnel
    out: list[str] = []
    out.append(f"- **Headline**: {g.headline}")
    out.append(f"- **Stop reason**: `{g.stop_reason or 'unset'}` · "
               f"elapsed `{g.elapsed_minutes or 0:.1f} min` · "
               f"objective `{g.objective}`")
    out.append(f"- **Workload**: {g.workload_summary}")
    out.append(
        f"- **Kernel pipeline**: detected={funnel['detected']} → "
        f"recommended={funnel['recommended']} → optimized={funnel['optimized']} → "
        f"adopted={funnel['adopted']} (partial={funnel['partial']}, "
        f"reverted={funnel['reverted']}, rejected={funnel['rejected']})"
    )
    out.append(f"- **Capabilities kept**: "
               + (", ".join(f"`{c}`" for c in g.capabilities_kept) or "none"))
    out.append(f"- **Capabilities not attempted**: "
               + (", ".join(f"`{c}`" for c in g.capabilities_not_attempted) or "none"))
    out.append(f"- **Gain attribution** ({g.attribution_method}):")
    if g.gain_attribution_lines:
        for line in g.gain_attribution_lines:
            out.append(f"  - {line}")
    else:
        out.append("  - (no attribution recorded)")
    # Data quality flags live in the Executive Summary; don't duplicate here.
    return "\n".join(out)
