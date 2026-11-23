# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Section 4a of the specialist prompt: the TraceLens roofline evidence of the latest snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.common.tracelens_report import system_metric_text

from ..specialists.prompt_inputs import SpecialistPromptInputs

# GPU share below which re-dispatching an op to another backend is not worth the
# equivalence work; keeps per-op glue out of the substitution directive.
_VENDOR_SUBSTITUTION_MIN_GPU_PCT = 5.0


def vendor_substitution_candidates(hot_kernels: Any) -> list[dict[str, Any]]:
    """Select hot ``aten::`` ops worth re-dispatching to a different backend.

    An ``aten::`` name means the op still runs through PyTorch's own dispatch, so
    an alternative backend is available to the call site. A vendor kernel that is
    already in the trace under its own name has nothing left to swap.

    Args:
        hot_kernels: The ``hot_kernels_top15`` rows from the roofline evidence.

    Returns:
        Qualifying rows ordered by descending GPU share.
    """
    if not isinstance(hot_kernels, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in hot_kernels:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name.startswith("aten::"):
            continue
        gpu_pct = entry.get("gpu_pct")
        if not isinstance(gpu_pct, (int, float)) or float(gpu_pct) < _VENDOR_SUBSTITUTION_MIN_GPU_PCT:
            continue
        rows.append(entry)
    rows.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
    return rows


def _vendor_substitution_directive(hot_kernels: Any) -> list[str]:
    """Render the backend-substitution directive for hot ATen ops (empty when none qualify)."""
    candidates = vendor_substitution_candidates(hot_kernels)
    if not candidates:
        return []
    rows = [
        "**Backend substitution — these ops still dispatch through PyTorch:**",
        "",
        "| name | gpu_pct | category | call site |",
        "|---|---:|---|---|",
    ]
    for entry in candidates:
        gpu_pct = float(entry.get("gpu_pct") or 0.0)
        rows.append(
            f"| {entry.get('name')} | {gpu_pct:.2f}% | "
            f"{entry.get('kernel_category') or ''} | {entry.get('source_file') or ''} |"
        )
    rows.extend(
        [
            "",
            "You may not rewrite the body of a kernel PyTorch owns. You MAY change "
            "which kernel the **call site** dispatches to, and when the call site is "
            "in the tree you are optimizing that is an ordinary source rewrite — "
            "same class as any other switch you author, behind its own default-off "
            "environment switch, with the original path kept as the fallback.",
            "",
            "Section 7 lists the accelerator libraries installed in this container. "
            "Read them to find an entry point matching the op's semantics and "
            "tensor layout, and verify equivalence on the real shapes before "
            "proposing it. An op holding a large share of device time is worth more "
            "than anything you can win from the glue around it.",
        ]
    )
    return rows


_ANALYSIS_MD_READ_GUIDE = (
    "Use the `Read` tool on this path for the full TraceLens report (~10-20 KB). All section headings are stable: "
    "`## Executive Summary` / `## Top Operations` / `## Compute Kernel Optimizations` / "
    "`## Kernel Fusion Opportunities` / `## System-Level Optimizations` / `## Recommendations`."
)


def _fusion_findings_rows(findings: Any) -> list[str]:
    """Render TraceLens' unfused kernel runs; empty when the report has none."""
    if not findings:
        return []
    rows = [
        "**Kernel fusion opportunities (TraceLens, largest projected saving first):**",
        "",
        "| pattern | operation | projected E2E saving | launcher |",
        "|---|---|---:|---|",
    ]
    for finding in findings:
        pct = finding.get("projected_e2e_pct")
        pct_str = f"{float(pct):.2f}%" if isinstance(pct, (int, float)) else "—"
        launchers = sorted(
            {m["kernel_launcher_path"] for m in finding.get("members") or [] if m["kernel_launcher_path"]}
        )
        rows.append(
            f"| {finding.get('kind') or '—'} | {finding.get('operation') or ''} | {pct_str} | {'; '.join(launchers)} |"
        )
    rows += [
        "",
        "A pattern here whose fused path the framework already ships (attention backend, fused add/norm/quant, "
        "RoPE+KV store, activation-and-mul, compiler fusion passes) is a server-arg or env variant to propose; "
        "a pattern still listed with that path enabled needs a new fused kernel.",
        "",
    ]
    return rows


# A TraceLens resolution runs to 2000 characters; the head names the lever, the full text stays in analysis.md.
_SYSTEM_RESOLUTION_CHARS = 300


def _system_findings_rows(findings: Any) -> list[str]:
    """Render TraceLens' system bottlenecks with their metrics and fix; empty when the report has none."""
    if not findings:
        return []
    rows = ["**System bottlenecks (TraceLens, report priority: largest share of GPU time first):**", ""]
    for finding in findings:
        node = f" [{finding['node']}]" if finding.get("node") else ""
        metrics = "; ".join(system_metric_text(row) for row in finding.get("metrics") or [])
        rows.append(f"- P{finding.get('priority') or '?'} {finding.get('operation') or ''}{node}: {metrics}")
        resolution = flatten_for_prompt(str(finding.get("resolution") or ""))
        if resolution:
            rows.append(f"  fix: {resolution[:_SYSTEM_RESOLUTION_CHARS]}")
    rows += [
        "",
        "High GPU Idle Time points at CUDA-graph capture, batch size and async scheduling; Exposed Collective "
        "Communication at parallel layout and compute/communication overlap; H2D / D2H / D2D Memory Copy Overhead at "
        "pinned memory, non-blocking copies, keeping tensors on device and dropping redundant copies. Where the "
        "framework exposes the lever as a server arg or env, propose it as an `explore` variant; where the fix names a "
        "source change at a call site (a host read such as `.item()`, a redundant `.contiguous()` or clone), patch "
        "that call site. Neither is a kernel rewrite. A family with no finding here is one TraceLens did not flag on "
        f"this trace. Each `fix` is cut to {_SYSTEM_RESOLUTION_CHARS} characters; the full text is under "
        "`## System-Level Optimizations` in analysis.md.",
        "",
    ]
    return rows


def roofline_evidence_section(inp: SpecialistPromptInputs) -> list[str]:
    """Render the ROOFLINE EVIDENCE section from ``inp.roofline_evidence``;
    empty evidence renders a heading + ``(none)`` placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``roofline_evidence``).

    Returns:
        The rendered roofline-evidence section lines.
    """
    rows = ["## 4a. ROOFLINE EVIDENCE", ""]
    ev = inp.roofline_evidence
    if not ev:
        rows.append(
            "(none — no fresh roofline snapshot has been recorded yet. "
            "The Coordinator auto-enqueues `roofline` at the end of "
            "PRELUDE and again after every 10% watermark crossing; if "
            "you are seeing this, the snapshot is still in-flight.)"
        )
        return rows

    snap_id = ev.get("roofline_snapshot_id")
    if snap_id is not None:
        rows.append(f"**TraceLens snapshot #{snap_id}**")
        rows.append("")

    summary = ev.get("executive_summary")
    if summary:
        rows.append("**Executive Summary:**")
        for label, key in (
            ("Compute %", "compute_pct"),
            ("Idle %", "idle_pct"),
            ("Exposed Comm %", "comm_pct"),
            ("Exposed memcpy %", "memcpy_pct"),
            ("Top bottleneck", "top_bottleneck"),
        ):
            val = summary.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)):
                rows.append(f"- {label}: {float(val):.1f}%")
            else:
                rows.append(f"- {label}: {val}")
        rows.append("")

    hot = ev.get("hot_kernels_top15")
    if hot:
        rows.append("**Top hot kernels (kernel_id | name | gpu_pct | bottleneck | source_file):**")
        rows.append("")
        rows.append("| kernel_id | name | gpu_pct | bottleneck | source_file |")
        rows.append("|---|---|---:|---|---|")
        for k in hot:
            kid = str(k.get("kernel_id") or "")
            name = str(k.get("name") or "")
            gpu_pct = k.get("gpu_pct")
            gpu_pct_str = f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float)) else "—"
            bottleneck = str(k.get("bottleneck") or "")
            src = str(k.get("source_file") or "")
            rows.append(f"| `{kid}` | {name} | {gpu_pct_str} | {bottleneck} | {src} |")
        rows.append("")

    directive = _vendor_substitution_directive(hot)
    if directive:
        rows.extend(directive)
        rows.append("")

    rows.extend(_fusion_findings_rows(ev.get("fusion_findings")))
    rows.extend(_system_findings_rows(ev.get("system_findings")))

    analysis_path = ev.get("analysis_md_path")
    if analysis_path:
        is_md = Path(analysis_path).suffix == ".md"
        guide = _ANALYSIS_MD_READ_GUIDE if is_md else "Use the `Read` tool on this path for the full TraceLens report."
        rows += [f"**Full {'analysis.md' if is_md else 'TraceLens report'} path:** `{analysis_path}`", "", guide]
    return rows
