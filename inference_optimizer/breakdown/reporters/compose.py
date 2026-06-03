"""Top-level composer: run every renderer + (optionally) call the LLM
+ stitch the result into a single markdown document.

Entry points:

* :func:`render_session_report` — the main programmatic API. Takes a
  parsed ``session_breakdown.json`` dict + an optional LLM client and
  returns the final markdown string.

Design notes:

* The compose layer never inspects per-section semantics; it just
  walks :data:`base.REGISTRY` in registration order, hands each
  renderer the breakdown dict, and stitches outputs together. This
  means new sections plug in with zero changes to compose.py.
* When ``llm_client`` is ``None`` the output still works — sections
  show only their deterministic markdown blocks, prefixed by the
  global executive-summary computed from :class:`GlobalFacts`. This
  is the path tests use, and is the path operators get when the LLM
  backend is misconfigured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .base import REGISTRY, RenderedSection
from .cross_section import GlobalFacts, build_global_facts
from .llm_prompt import SYSTEM_PROMPT, build_user_prompt, parse_llm_response

# Import every renderer module so its @register_renderer side effect
# runs. Renderer order in REGISTRY is decoupled from final report
# layout — :data:`SECTION_GROUPS` below dictates the H2/H3 grouping
# the reader actually sees.
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


# Final report layout: ``(group_title, [section_id, ...])``.
#
# Grouping mirrors the orchestration phases a hyperloom run actually
# goes through — Session/Workload setup, performance Results, the
# capability search loop (backends/params/sweep), kernel optimization
# (lifecycle + GEAK/OOB invocations), and bookkeeping (attribution +
# source files). ``telemetry`` is deliberately dropped from the report
# (its renderer is no longer imported above) because the on-disk GPU
# monitor data has been consistently broken on real wekafs sessions.
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
    # ``source_files`` lists the artifacts the breakdown was synthesized
    # from; ``data_provenance`` (right after) explains, per section,
    # which of those source artifacts the collector actually saw —
    # making it trivial to diagnose "why is X empty?" without leaving
    # the report.
    ("Source Artifacts",             ["source_files", "data_provenance"]),
]

__all__ = [
    "LLMClient",
    "ComposeResult",
    "render_session_report",
]


class LLMClient(Protocol):
    """Minimal LLM client interface the compose layer uses.

    Any callable mapping ``(system, user) -> str`` works, including a
    thin shim around :class:`ClaudeBackend`. Kept as a Protocol so we
    can swap in mocks for tests without importing the orchestrator.
    """

    def complete(self, *, system: str, user: str) -> str:
        """Run one completion and return the model's text.

        Args:
            system (str): The system prompt.
            user (str): The user message.

        Returns:
            str: The model's response text.
        """
        ...


@dataclass(frozen=True)
class ComposeResult:
    """Full output of one compose run.

    ``markdown`` is the user-facing report; the other fields are kept
    so callers can persist them alongside the report for debugging /
    replay (e.g. dump ``llm_user_prompt`` next to the report to
    diagnose hallucinations).
    """

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
    """Render ``breakdown`` (a parsed session_breakdown.json) to markdown.

    Runs every registered renderer, builds the deterministic
    :class:`GlobalFacts`, optionally calls the LLM for narrative prose,
    and stitches everything into a single report.

    Args:
        breakdown (dict[str, Any]): The parsed ``session_breakdown.json`` dict.
        llm_client (LLMClient | None): Optional LLM client for the narrative
            pass; when ``None`` (or when the call fails), only the
            deterministic output is produced.

    Returns:
        ComposeResult: The final markdown plus the intermediate artifacts
            (sections, global facts, prompt and raw LLM response) for replay
            and debugging.
    """
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


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------
def _stitch(
    *,
    sections: list[RenderedSection],
    global_facts: GlobalFacts,
    llm_exec_summary: str,
    llm_narratives: dict[str, str],
    used_llm: bool,
    breakdown: dict[str, Any],
) -> str:
    """Assemble the final markdown document from all rendered pieces.

    Lays out the title, executive summary (LLM or deterministic fallback),
    the deterministic key-facts block, and each section group with its
    optional LLM narrative, verbatim markdown block and data-quality notes.

    Args:
        sections (list[RenderedSection]): All renderer outputs.
        global_facts (GlobalFacts): The deterministic cross-section fact pack.
        llm_exec_summary (str): The LLM-written executive summary (may be empty).
        llm_narratives (dict[str, str]): Section-id keyed narrative paragraphs.
        used_llm (bool): Whether a successful LLM pass produced the narratives.
        breakdown (dict[str, Any]): The parsed ``session_breakdown.json`` dict.

    Returns:
        str: The complete report markdown, newline-terminated.
    """
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
            # All sections in this group are skipped → drop the H2 header
            # too. Avoids "## Sweep\n_skipped_" placeholder noise.
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
    """Fallback exec summary used when no LLM is configured / LLM failed.

    Intentionally terse and bullet-point-y so it's obviously the
    deterministic path (vs a polished LLM paragraph). Mentions every
    data-quality flag so silent issues are never hidden.

    Args:
        g (GlobalFacts): The deterministic cross-section fact pack.

    Returns:
        str: A newline-joined markdown bullet summary.
    """
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
    """Render :class:`GlobalFacts` as a compact key-value block so the
    raw computed facts are always inspectable in the report (and so
    the LLM-written exec summary can be cross-checked against them).

    Args:
        g (GlobalFacts): The deterministic cross-section fact pack.

    Returns:
        str: A newline-joined markdown key-value block.
    """
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
    # Data quality flags appear in the Executive Summary above —
    # don't duplicate them here (the Key Facts block is the LLM /
    # script-facing cross-check view, not a second human-facing
    # summary).
    return "\n".join(out)
