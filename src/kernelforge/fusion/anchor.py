# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Anchored discovery: resolve an operator-named kernel and read its fusion neighbourhood."""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .discover import (
    COMPUTE_CATEGORIES,
    LlmFn,
    _output_schema_block,
    fusion_constraints,
    kernel_names_from_trace,
    parse_discovered_recipes,
    render_repo_scope_brief,
    render_source_files,
    stream_ordered_kernels,
)
from .models import Recipe
from .vllm_passes import PassState

log = logging.getLogger("forge_fusion")

# How far the compute-bounded span around the anchor is allowed to reach before it stops being one fusible chain.
_MAX_SPAN = 8

# A shape shared by no more than half the launches is a coin flip, not a pattern, and fusing it buys nothing
# repeatable.
_MIN_CONSISTENCY = 0.5

_BOUNDARY = "<none>"


class AnchorResolutionError(ValueError):
    """The named kernel could not be located in the trace."""


@dataclass(frozen=True)
class KernelAnchor:
    """The kernel an operator named."""

    name: str


@dataclass
class Slot:
    """What runs on one side of the anchor, in the anchor's dominant pattern."""

    category: str
    names: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "kernels": [{"name": n, "count": c} for n, c in self.names]}


@dataclass
class AnchorReport:
    """Everything the trace says about one named kernel and what surrounds it."""

    name: str
    category: str
    occurrences: int
    total_us: float
    avg_us: float
    share: float
    signature: str
    consistency: float
    before: Optional[Slot]
    after: Optional[Slot]
    span: list[dict[str, str]]
    patterns: list[tuple[str, int]]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "occurrences": self.occurrences,
            "total_us": round(self.total_us, 3),
            "avg_us": round(self.avg_us, 3),
            "share": round(self.share, 6),
            "signature": self.signature,
            "consistency": round(self.consistency, 4),
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "span": self.span,
            "patterns": [{"signature": s, "count": c} for s, c in self.patterns],
            "warnings": self.warnings,
        }


def collapse_whitespace(name: str) -> str:
    """Normalize a kernel name for comparison.

    Trace names are single-line, but the operator pastes them out of a viewer that
    wraps. Only whitespace is touched: template parameters and mangling decide which
    kernel this is, so nothing else may be normalized away.
    """
    return re.sub(r"\s+", " ", str(name or "")).strip()


def _slot_from(events: list[dict[str, Any]]) -> Optional[Slot]:
    if not events:
        return None
    counts = Counter(str(event["name"]) for event in events)
    return Slot(
        category=str(events[0]["category"]),
        names=sorted(counts.items(), key=lambda item: (-item[1], item[0])),
    )


def _span_around(ordered: list[dict[str, Any]], index: int) -> list[dict[str, str]]:
    """The compute-bounded chain the anchor sits in, compute anchors included."""
    start = index
    while start > 0 and index - start < _MAX_SPAN:
        start -= 1
        if ordered[start]["category"] in COMPUTE_CATEGORIES:
            break
    end = index
    while end < len(ordered) - 1 and end - index < _MAX_SPAN:
        end += 1
        if ordered[end]["category"] in COMPUTE_CATEGORIES:
            break
    return [
        {"category": str(ordered[i]["category"]), "name": str(ordered[i]["name"]), "is_anchor": i == index}
        for i in range(start, end + 1)
    ]


def resolve_anchor(trace_path: str | Path, anchor: KernelAnchor) -> AnchorReport:
    """Locate the named kernel in the trace and summarize what runs around it."""
    streams, total_kernel_us = stream_ordered_kernels(trace_path)
    if not streams:
        raise AnchorResolutionError(f"no GPU kernel events in {trace_path}: nothing to anchor on")

    wanted = collapse_whitespace(anchor.name)
    if not wanted:
        raise AnchorResolutionError("empty kernel name")

    sites: list[tuple[tuple[Any, Any], int]] = []
    for key, ordered in streams.items():
        for index, event in enumerate(ordered):
            if collapse_whitespace(event["name"]) == wanted:
                sites.append((key, index))
    if not sites:
        raise AnchorResolutionError(_miss_message(trace_path, wanted))

    warnings: list[str] = []
    occurrences = len(sites)
    resolved_name = str(streams[sites[0][0]][sites[0][1]]["name"])
    category = str(streams[sites[0][0]][sites[0][1]]["category"])
    total_us = sum(float(streams[k][i]["dur"]) for k, i in sites)

    neighbours: dict[str, list[tuple[tuple[Any, Any], int]]] = defaultdict(list)
    for key, index in sites:
        ordered = streams[key]
        prev_cat = ordered[index - 1]["category"] if index > 0 else _BOUNDARY
        next_cat = ordered[index + 1]["category"] if index + 1 < len(ordered) else _BOUNDARY
        neighbours[f"{prev_cat} -> {category} -> {next_cat}"].append((key, index))

    ranked = sorted(neighbours.items(), key=lambda item: (-len(item[1]), item[0]))
    signature, dominant_sites = ranked[0]
    consistency = len(dominant_sites) / occurrences

    # The span is rendered for one launch; the most common neighbourhood's first site is the one that represents it.
    representative = dominant_sites[0]
    before_events = [streams[k][i - 1] for k, i in dominant_sites if i > 0]
    after_events = [streams[k][i + 1] for k, i in dominant_sites if i + 1 < len(streams[k])]

    if consistency <= _MIN_CONSISTENCY:
        warnings.append(
            f"the neighbourhood is unstable: the most common shape covers only {consistency:.1%} of "
            f"{occurrences} launches, so any fusion here applies to a minority of them"
        )
    if category in COMPUTE_CATEGORIES:
        warnings.append(
            f"the named kernel is itself a {category} kernel; fusion collapses the launch-bound tail around "
            "a compute kernel rather than the compute kernel itself"
        )
    if len({key for key, _ in sites}) > 1:
        warnings.append("the named kernel runs on more than one stream; neighbours are read per stream")

    return AnchorReport(
        name=resolved_name,
        category=category,
        occurrences=occurrences,
        total_us=total_us,
        avg_us=total_us / occurrences,
        share=(total_us / total_kernel_us) if total_kernel_us > 0 else 0.0,
        signature=signature,
        consistency=consistency,
        before=_slot_from(before_events),
        after=_slot_from(after_events),
        span=_span_around(streams[representative[0]], representative[1]),
        patterns=[(sig, len(hits)) for sig, hits in ranked],
        warnings=warnings,
    )


def _miss_message(trace_path: str | Path, wanted: str) -> str:
    """Explain a miss with the closest names the trace actually holds."""
    pool = kernel_names_from_trace(trace_path, top_n=400)
    collapsed = {collapse_whitespace(name): name for name in pool}
    close = difflib.get_close_matches(wanted, list(collapsed), n=3, cutoff=0.5)
    if not close:
        close = [name for name in collapsed if wanted.lower() in name.lower()][:3]
    lines = [f"no kernel in {trace_path} is named:", f"  {wanted[:200]}"]
    if close:
        lines.append("closest names in this trace (the full name is required, not a fragment):")
        lines.extend(f"  {collapsed[name][:200]}" for name in close)
    else:
        lines.append("no similar name was found; check the trace and the rank you are reading.")
    return "\n".join(lines)


def describe_anchor(report: AnchorReport) -> str:
    """Render the anchor evidence block that goes into the discovery prompt."""
    lines = [
        "## The fusion anchor (named by the operator, not chosen by you)",
        f"name: {report.name}",
        (
            f"category: {report.category}  launches: {report.occurrences}  "
            f"total: {report.total_us:.1f}us  avg: {report.avg_us:.2f}us  "
            f"share of kernel time: {report.share * 100:.2f}%"
        ),
        "",
        f"Dominant neighbourhood over all {report.occurrences} launches, by category:",
        f"  {report.signature}   ({report.patterns[0][1]}/{report.occurrences} = {report.consistency:.1%})",
    ]

    def slot_lines(title: str, slot: Optional[Slot]) -> list[str]:
        if slot is None:
            return [f"  {title}: nothing (the anchor is at a stream boundary here)"]
        out = [f"  {title} ({slot.category}), distinct kernels:"]
        out.extend(f"    {count:6d}x  {name}" for name, count in slot.names[:6])
        return out

    lines += slot_lines("immediately before", report.before)
    lines += slot_lines("immediately after", report.after)

    lines.append("")
    lines.append("Compute-bounded span of a representative launch (ANCHOR marked):")
    for item in report.span:
        mark = "  <== ANCHOR" if item["is_anchor"] else ""
        lines.append(f"  {item['category']:11s} {item['name']}{mark}")

    if len(report.patterns) > 1:
        lines.append("")
        lines.append("Other observed neighbourhoods:")
        for sig, count in report.patterns[1:5]:
            lines.append(f"  {sig}   ({count}/{report.occurrences} = {count / report.occurrences:.1%})")
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {text}" for text in report.warnings)
    return "\n".join(lines)


def anchor_trace_evidence(report: AnchorReport) -> dict[str, Any]:
    """The anchor's recorded launches, as ground truth for later stages.

    ``describe_anchor`` renders this for the discovery agent, but discovery's answer
    is prose: it names source symbols, and those names are what every later stage
    builds on. Carrying the kernel names forward lets the harness author CHECK the
    reference it picked instead of trusting the symbol names it was handed.
    """
    return {
        "anchor": report.name,
        "before": [name for name, _ in (report.before.names if report.before else [])],
        "after": [name for name, _ in (report.after.names if report.after else [])],
        # The anchor's position travels as the flag the span was built with rather
        # than as a name comparison: the same kernel can legitimately appear twice
        # in one span, and only one of those occurrences is the anchor.
        "span": [{"name": item["name"], "is_anchor": bool(item["is_anchor"])} for item in report.span],
    }


def build_anchored_discovery_prompt(
    *,
    model_type: str,
    framework: str,
    source_files: Sequence[str],
    report: AnchorReport,
    shapes: dict[str, Any],
    repo_scope: bool = False,
    repo_root: str = "",
) -> str:
    """Assemble a discovery prompt whose target is fixed and whose fusion is not."""
    if repo_scope:
        source_block = render_repo_scope_brief(
            repo_root,
            source_files,
            model_type=model_type,
            framework=framework,
        )
        # Under repo scope an unreachable neighbour is a search result, not a fact:
        # the file that issues it exists somewhere in the tree, so "I could not
        # reach it" has to mean "I looked and it is not there".
        reach = (
            "The kernels above are issued by code that EXISTS in this repository. Find\n"
            "the file(s) that issue them and list every one your fusion would edit.\n"
            "Only shrink the chain after searching has shown the rest is unreachable,\n"
            "and say what you searched for when you do."
        )
    else:
        source_block = render_source_files(source_files, model_type=model_type, framework=framework)
        reach = (
            "If the anchor's neighbours are not reachable from this source file, say so\n"
            "by proposing the largest fusion that IS reachable and including the anchor."
        )
    read_verb = "Explore the repository described below" if repo_scope else "Read the source below"
    return f"""You are analyzing the DECODE path of a {framework} model (`model_type={model_type}`)
to find a SOURCE-LEVEL KERNEL FUSION built around ONE kernel the operator named.
Analyze only; do not edit anything. Return your answer as JSON (schema below).

## What is fixed and what is yours to decide
The anchor kernel below is FIXED: the operator picked it out of the trace and every
proposal you return must be a fusion that includes it. What to fuse it WITH is yours
to decide, from the neighbourhood evidence and the model source.

The lever is launch count: each tiny op is a separate kernel launch and HBM
round-trip, so collapsing the anchor together with the work adjacent to it removes
launches and round-trips. Fusing into the prologue of the compute kernel after the
anchor is the preferred direction; the epilogue of the one before it is in scope
only when that kernel is not a tuned library call (see the constraints below).

{describe_anchor(report)}

Representative decode shapes: {shapes}

## Your task
{read_verb} and propose the fusion (or at most 2 alternatives) that
collapses this anchor into its neighbours. Name the exact call site you would
replace. {reach}

{fusion_constraints(repo_scope=repo_scope)}
- The anchor kernel must be part of every proposal. A proposal that does not
  include it answers a question nobody asked.

{_output_schema_block(model_type, repo_scope=repo_scope)}

{source_block}
"""


def discover_anchored_recipes(
    *,
    model_type: str,
    framework: str,
    source_file: str,
    shapes: dict[str, Any],
    report: AnchorReport,
    llm_fn: LlmFn,
    category_shares: Optional[dict[str, float]] = None,
    pass_probe: Optional[Callable[[str], PassState]] = None,
    framework_root: str = "",
    repo_scope: bool = False,
    repo_root: str = "",
) -> list[Recipe]:
    """Propose fusions built around one named kernel.

    Unlike :func:`discover.discover_recipes` this does not consult the diagnosis
    verdict: the operator named a kernel, which overrides a trace-wide judgement
    that there was nothing worth fusing.

    ``repo_scope`` turns ``source_file`` into a mere entry point: nothing is
    embedded, the whole repository is proposable, and a proposal may name several
    files. That is the difference between needing to know where the chain lives
    and being able to go and find out.
    """
    in_scope = [source_file] if source_file and Path(source_file).is_file() else []
    if not in_scope and not repo_scope:
        log.warning("anchored discovery: model source unreadable (%s); cannot propose a fusion", source_file)
        return []
    if repo_scope:
        log.info(
            "anchored discovery: repo scope over %s (entry point: %s)",
            repo_root or "the working directory",
            ", ".join(Path(p).name for p in in_scope) or "none resolved",
        )

    prompt = build_anchored_discovery_prompt(
        model_type=model_type,
        framework=framework,
        source_files=in_scope,
        report=report,
        shapes=shapes,
        repo_scope=repo_scope,
        repo_root=repo_root,
    )
    recipes = parse_discovered_recipes(
        llm_fn(prompt),
        model_type=model_type,
        framework=framework,
        source_file=source_file,
        shapes=shapes,
        category_shares=category_shares,
        pass_probe=pass_probe,
        framework_root=framework_root,
        explicit_target=True,
        in_scope_files=in_scope,
        repo_scope=repo_scope,
        repo_root=repo_root,
    )
    evidence = anchor_trace_evidence(report)
    for recipe in recipes:
        recipe.trace_kernels = dict(evidence)
    log.info(
        "anchored discovery proposed %d fusion(s) around %s: %s",
        len(recipes),
        report.category,
        ", ".join(r.pattern_id for r in recipes),
    )
    return recipes


__all__ = [
    "AnchorReport",
    "AnchorResolutionError",
    "KernelAnchor",
    "Slot",
    "anchor_trace_evidence",
    "build_anchored_discovery_prompt",
    "collapse_whitespace",
    "describe_anchor",
    "discover_anchored_recipes",
    "resolve_anchor",
]
