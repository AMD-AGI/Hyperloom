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
    render_source_files,
    stream_ordered_kernels,
)
from .models import Recipe
from .vllm_passes import PassState

log = logging.getLogger("forge_fusion")

# How far the compute-bounded span around the anchor is allowed to reach before it stops being one fusible chain.
_MAX_SPAN = 8

# Operators read nanoseconds out of a trace viewer; kineto records microseconds. Conversion happens here so every
# timestamp the operator sees or types is nanoseconds.
_NS_PER_US = 1000

# A pinned timestamp is a reference, not a key: viewers round, so the nearest launch is taken and only a gap wider
# than this is worth telling the operator about.
_TS_TOLERANCE_NS = 1000

# A shape shared by no more than half the launches is a coin flip, not a pattern, and fusing it buys nothing
# repeatable.
_MIN_CONSISTENCY = 0.5

_BOUNDARY = "<none>"


class AnchorResolutionError(ValueError):
    """The named kernel could not be located in the trace."""


@dataclass(frozen=True)
class KernelAnchor:
    """The kernel an operator named, plus the launch they were looking at."""

    name: str
    ts_ns: Optional[int] = None


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
    pinned_ts_ns: Optional[int] = None
    pinned_is_dominant: bool = True
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
            "pinned_ts_ns": self.pinned_ts_ns,
            "pinned_is_dominant": self.pinned_is_dominant,
            "warnings": self.warnings,
        }


def collapse_whitespace(name: str) -> str:
    """Normalize a kernel name for comparison.

    Trace names are single-line, but the operator pastes them out of a viewer that
    wraps. Only whitespace is touched: template parameters and mangling decide which
    kernel this is, so nothing else may be normalized away.
    """
    return re.sub(r"\s+", " ", str(name or "")).strip()


def _event_ts_ns(event: dict[str, Any]) -> int:
    """The launch time in nanoseconds, the unit the operator works in."""
    return round(float(event["ts"]) * _NS_PER_US)


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

    pinned_site: Optional[tuple[tuple[Any, Any], int]] = None
    pinned_ts_ns: Optional[int] = None
    if anchor.ts_ns is not None:
        wanted_ns = int(anchor.ts_ns)
        pinned_site = min(sites, key=lambda site: abs(_event_ts_ns(streams[site[0]][site[1]]) - wanted_ns))
        pinned_ts_ns = _event_ts_ns(streams[pinned_site[0]][pinned_site[1]])
        drift = abs(pinned_ts_ns - wanted_ns)
        if drift > _TS_TOLERANCE_NS:
            launch_ns = [_event_ts_ns(streams[k][i]) for k, i in sites]
            note = (
                f"--fuse-kernel-ts {wanted_ns} matched no launch exactly; using the nearest at "
                f"{pinned_ts_ns} ({drift} ns away)"
            )
            if not min(launch_ns) <= wanted_ns <= max(launch_ns):
                note += f"; this kernel runs between {min(launch_ns)} and {max(launch_ns)} ns"
                # The one mistake this unit invites, named rather than left as an unexplained 1000x miss.
                if min(launch_ns) <= wanted_ns * _NS_PER_US <= max(launch_ns):
                    note += " -- the value looks like microseconds, and this flag takes nanoseconds"
            warnings.append(note)

    representative = pinned_site if pinned_site is not None else dominant_sites[0]
    pinned_is_dominant = representative in dominant_sites
    if pinned_site is not None and not pinned_is_dominant:
        warnings.append(
            f"the launch at ts={pinned_ts_ns} ns is not in the dominant pattern ({signature}); "
            "the aggregate below still describes every launch"
        )

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
        pinned_ts_ns=pinned_ts_ns,
        pinned_is_dominant=pinned_is_dominant,
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

    if report.pinned_ts_ns is not None:
        lines.append(f"  the launch you referenced: ts={report.pinned_ts_ns} ns")
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
) -> str:
    """Assemble a discovery prompt whose target is fixed and whose fusion is not."""
    multi_file = len(source_files) > 1
    source_block = render_source_files(source_files, model_type=model_type, framework=framework)
    reach = (
        "If the anchor's neighbours are not reachable from ANY of these files, say so\n"
        "by proposing the largest fusion that IS reachable and including the anchor."
        if multi_file
        else "If the anchor's neighbours are not reachable from this source file, say so\n"
        "by proposing the largest fusion that IS reachable and including the anchor."
    )
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
Read the source below and propose the fusion (or at most 2 alternatives) that
collapses this anchor into its neighbours. Name the exact call site you would
replace. {reach}

{fusion_constraints(multi_file)}
- The anchor kernel must be part of every proposal. A proposal that does not
  include it answers a question nobody asked.

{_output_schema_block(model_type, multi_file=multi_file)}

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
    extra_source_files: Sequence[str] = (),
) -> list[Recipe]:
    """Propose fusions built around one named kernel.

    Unlike :func:`discover.discover_recipes` this does not consult the diagnosis
    verdict: the operator named a kernel, which overrides a trace-wide judgement
    that there was nothing worth fusing.

    ``extra_source_files`` are shown alongside ``source_file`` and are equally
    proposable. The anchor's chain frequently lives in one of them -- the model
    file often reaches it through a single opaque call whose operands are not
    local names there -- and a prompt showing only the model file forces the
    answer to be a smaller chain that happens to be local to it.
    """
    in_scope: list[str] = []
    for path in [source_file, *extra_source_files]:
        if path and path not in in_scope and Path(path).is_file():
            in_scope.append(path)
    if not in_scope:
        log.warning("anchored discovery: model source unreadable (%s); cannot propose a fusion", source_file)
        return []
    if len(in_scope) > 1:
        log.info(
            "anchored discovery: %d in-scope file(s): %s",
            len(in_scope),
            ", ".join(Path(p).name for p in in_scope),
        )

    prompt = build_anchored_discovery_prompt(
        model_type=model_type,
        framework=framework,
        source_files=in_scope,
        report=report,
        shapes=shapes,
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
