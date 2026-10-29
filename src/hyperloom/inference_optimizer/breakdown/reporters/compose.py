# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Top-level composer: run every renderer, optionally call the LLM, stitch to one markdown doc."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

from .base import REGISTRY, RenderedSection, fmt_pct, render_section, session_of
from .cross_section import GlobalFacts, build_global_facts
from .llm_prompt import SYSTEM_PROMPT, build_user_prompt, parse_llm_response

# Import every renderer module for its @register_renderer side effect.
from ._renderers import (
    session as _r_session,
    workload as _r_workload,
    baseline as _r_baseline,
    final as _r_final,
    enablement as _r_enablement,
    capability_summary as _r_capability_summary,
    phase_timeline as _r_phase_timeline,
    kernel_lifecycle as _r_kernel_lifecycle,
    roofline as _r_roofline,
    param_search as _r_param_search,
    attribution as _r_attribution,
    optimizations as _r_optimizations,
    critic as _r_critic,
)

log = logging.getLogger(__name__)


# Final report layout ``(group_title, [section_id, ...])``. ``telemetry`` is dropped.
SECTION_GROUPS: list[tuple[str, list[str]]] = [
    ("Session & Workload", ["session", "workload"]),
    ("Enablement", ["enablement"]),
    (
        "Performance Results",
        ["baseline", "final", "roofline", "optimizations", "attribution"],
    ),
    ("Capability Search", ["capability_summary", "param_search"]),
    ("Kernel Optimization", ["kernel_lifecycle"]),
    ("Review", ["critic"]),
    ("Run Trace", ["phase_timeline"]),
]

__all__ = [
    "LLMClient",
    "ComposeResult",
    "render_session_report",
    # Renderer submodules re-exported so static analysis does not flag them as unused.
    "_r_session",
    "_r_workload",
    "_r_baseline",
    "_r_final",
    "_r_enablement",
    "_r_capability_summary",
    "_r_phase_timeline",
    "_r_kernel_lifecycle",
    "_r_roofline",
    "_r_param_search",
    "_r_attribution",
    "_r_optimizations",
    "_r_critic",
]


class LLMClient(Protocol):
    """Minimal LLM client interface (``(system, user) -> str``); a Protocol so tests can mock it."""

    def complete(self, *, system: str, user: str) -> str:
        """Run one completion and return the model's text."""


@dataclass(frozen=True)
class ComposeResult:
    """Full output of one compose run (``markdown`` plus debug/replay fields)."""

    markdown: str
    sections: list[RenderedSection]
    global_facts: GlobalFacts
    llm_user_prompt: str
    llm_raw_response: str
    used_llm: bool


@dataclass(frozen=True)
class Narration:
    """What the optional LLM pass wrote; ``used`` is whether a successful pass produced it."""

    exec_summary: str = ""
    narratives: dict[str, str] = field(default_factory=dict)
    used: bool = False


def _narration_failures() -> tuple[type[BaseException], ...]:
    """What a failed narration call or a malformed reply raises; an SDK not yet imported cannot have raised."""
    sdk_errors = (("httpx", "HTTPError"), ("openai", "OpenAIError"), ("claude_agent_sdk", "ClaudeSDKError"))
    loaded = tuple(getattr(sys.modules[mod], name) for mod, name in sdk_errors if mod in sys.modules)
    return (OSError, RuntimeError, ValueError, TimeoutError, AttributeError, *loaded)


def render_session_report(
    breakdown: dict[str, Any],
    *,
    llm_client: LLMClient | None = None,
) -> ComposeResult:
    """Render ``breakdown`` (a parsed session_breakdown.json) to markdown."""
    sections = [render_section(sid, fn, breakdown) for sid, fn in REGISTRY]
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
        except _narration_failures() as exc:
            log.warning("LLM narration failed: %s", exc, exc_info=True)
            llm_raw = f"<llm_error: {type(exc).__name__}: {exc}>"

    used_llm = llm_client is not None and not str(llm_raw).startswith("<llm_error")
    md = _stitch(
        sections=sections,
        global_facts=global_facts,
        narration=Narration(exec_summary=exec_summary_llm, narratives=narratives, used=used_llm),
        breakdown=breakdown,
    )
    return ComposeResult(
        markdown=md,
        sections=sections,
        global_facts=global_facts,
        llm_user_prompt=user_prompt,
        llm_raw_response=llm_raw,
        used_llm=used_llm,
    )


# Stitching
def _stitch(
    *,
    sections: list[RenderedSection],
    global_facts: GlobalFacts,
    narration: Narration,
    breakdown: dict[str, Any],
) -> str:
    """Assemble the final markdown document from all rendered pieces.

    Lays out the title, executive summary (LLM or deterministic fallback),
    the deterministic key-facts block, and each section group with its
    optional LLM narrative, verbatim markdown block and data-quality notes.

    Args:
        sections (list[RenderedSection]): All renderer outputs.
        global_facts (GlobalFacts): The deterministic cross-section fact pack.
        narration (Narration): The LLM-written executive summary (may be empty)
            and section-id keyed narrative paragraphs.
        breakdown (dict[str, Any]): The parsed ``session_breakdown.json`` dict.

    Returns:
        str: The complete report markdown, newline-terminated.
    """
    session = session_of(breakdown)
    title = f"# Hyperloom Session Report — {session.get('session_id') or '(no session_id)'}"

    parts: list[str] = [title, ""]

    parts.append("## Executive Summary")
    if narration.used and narration.exec_summary:
        parts.append(narration.exec_summary)
        # The system prompt asks the model to surface every data-quality flag, but a prompt is a request.
        flag_lines = _data_quality_flag_lines(global_facts)
        if flag_lines:
            parts.append("")
            parts.extend(flag_lines)
    else:
        parts.append(_deterministic_exec_summary(global_facts))
    parts.append("")

    parts.append("## Key Facts (deterministic)")
    parts.append(_render_global_facts_block(global_facts))
    parts.append("")

    by_id = {s.section_id: s for s in sections}
    for group_title, ids in SECTION_GROUPS:
        live = [by_id[sid] for sid in ids if sid in by_id and not by_id[sid].skipped]
        if not live:
            # All sections skipped → drop the H2 header too.
            continue
        parts.append(f"## {group_title}")
        parts.append("")
        for sec in live:
            parts.append(f"### {sec.title}")
            narrative = narration.narratives.get(sec.section_id, "").strip()
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


_UNSET = "unset"


def _deterministic_exec_summary(g: GlobalFacts) -> str:
    """Fallback exec summary when no LLM is configured / it failed; lists every data-quality flag."""
    out: list[str] = []
    out.append(
        f"- {g.headline} (stop_reason={g.stop_reason or _UNSET}, "
        f"{_elapsed_label(g)}={g.elapsed_minutes or 0:.0f}min, graded_on={g.graded_on or _UNSET}, "
        f"stop target {g.stop_target or _UNSET})."
    )
    out.append(f"- Workload: {g.workload_summary}.")
    if g.gain_attribution_lines:
        out.append(f"- Gain attribution ({g.attribution_method}): " + "; ".join(g.gain_attribution_lines))
    else:
        out.append(f"- No measurable gain recorded ({g.attribution_method}).")
    out.append(f"- Variants: {_variant_funnel_text(g)}.")
    out.append(f"- Kernel pipeline: {_kernel_funnel_text(g)}.")
    if g.capabilities_not_attempted:
        out.append("- Capabilities never invoked: " + ", ".join(f"`{c}`" for c in g.capabilities_not_attempted) + ".")
    out.extend(_data_quality_flag_lines(g))
    return "\n".join(out)


def _elapsed_label(g: GlobalFacts) -> str:
    """``elapsed(total)`` when the figure spans every leg of the session, else ``elapsed``."""
    return "elapsed(total)" if g.elapsed_is_total else "elapsed"


def _variant_funnel_text(g: GlobalFacts) -> str:
    """The variant funnel as one line, shared by the executive summary and the key facts."""
    funnel = g.variant_funnel
    runs = funnel["bench_runs"]
    if not runs:
        return "none benched"
    rate = funnel["keep_rate_pct"]
    rate_text = f" ({rate:.1f}%)" if rate is not None else ""
    text = (
        f"{_count(runs, 'bench run')} over {_count(funnel['distinct'], 'distinct change')} → "
        f"kept {funnel['kept']}{rate_text} · reverted {funnel['reverted']} · failed {funnel['failed']}"
    )
    shipped: list[str] = []
    if funnel["final_stack"] is not None:
        shipped.append(f"final stack {_count(funnel['final_stack'], 'change')}")
    if funnel["validated_gain_pct"] is not None:
        shipped.append(f"validated {fmt_pct(funnel['validated_gain_pct'], plus=True)}")
    return f"{text} → {', '.join(shipped)}" if shipped else text


def _kernel_funnel_text(g: GlobalFacts) -> str:
    """The kernel pipeline funnel as one line, shared by the executive summary and the key facts."""
    funnel = g.kernel_pipeline_funnel
    return (
        f"detected={funnel['detected']} → recommended={funnel['recommended']} → "
        f"attempted={funnel['attempted']} → adopted={funnel['adopted']} "
        f"(reverted={funnel['reverted']}, rejected={funnel['rejected']}, partial={funnel['partial']})"
    )


def _count(n: int, noun: str) -> str:
    """``n`` followed by ``noun``, pluralised unless ``n`` is one."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _data_quality_flag_lines(g: GlobalFacts) -> list[str]:
    """Render the data-quality flags as markdown bullets."""
    if not g.data_quality_flags:
        return []
    lines = ["- **Data quality flags**:"]
    lines.extend(f"  - {flag}" for flag in g.data_quality_flags)
    return lines


def _render_global_facts_block(g: GlobalFacts) -> str:
    """Render :class:`GlobalFacts` as a compact key-value block for cross-checking the report."""
    out: list[str] = []
    out.append(f"- **Headline**: {g.headline}")
    out.append(
        f"- **Stop reason**: `{g.stop_reason or _UNSET}` · "
        f"{_elapsed_label(g)} `{g.elapsed_minutes or 0:.1f} min` · "
        f"graded on `{g.graded_on or _UNSET}` · stop target `{g.stop_target or _UNSET}`"
    )
    out.append(f"- **Workload**: {g.workload_summary}")
    out.append(f"- **Variants**: {_variant_funnel_text(g)}")
    out.append(f"- **Kernel pipeline**: {_kernel_funnel_text(g)}")
    out.append("- **Capabilities kept**: " + (", ".join(f"`{c}`" for c in g.capabilities_kept) or "none"))
    out.append(
        "- **Capabilities not attempted**: " + (", ".join(f"`{c}`" for c in g.capabilities_not_attempted) or "none")
    )
    out.append(f"- **Gain attribution** ({g.attribution_method}):")
    if g.gain_attribution_lines:
        for line in g.gain_attribution_lines:
            out.append(f"  - {line}")
    else:
        out.append("  - (no attribution recorded)")
    # Data quality flags live in the Executive Summary; don't duplicate here.
    return "\n".join(out)
