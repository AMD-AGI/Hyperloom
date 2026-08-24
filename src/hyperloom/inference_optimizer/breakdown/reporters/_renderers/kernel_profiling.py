# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel profiling renderer.

Surfaces profile / TraceLens runs: launch args, artifact paths and
parsed top-k kernels. Artifact locations are reported as recorded; this
renderer reads no files.
"""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_kv_list, md_table, register_renderer

_MAX_KERNEL_ROWS = 15


def _kernel_rows(kernels: list[dict[str, Any]]) -> list[list[Any]]:
    """Build table rows for the top-k profiled kernels.

    Args:
        kernels (list[dict[str, Any]]): Parsed top-kernel records.

    Returns:
        list[list[Any]]: Up to ``_MAX_KERNEL_ROWS`` rows of
            ``[kernel_id, name, gpu_pct, duration_us, bottleneck]``.
    """
    rows: list[list[Any]] = []
    for k in kernels[:_MAX_KERNEL_ROWS]:
        rows.append(
            [
                k.get("kernel_id") or "—",
                (k.get("name") or "")[:60],
                k.get("gpu_pct"),
                k.get("duration_us"),
                k.get("bottleneck") or "—",
            ]
        )
    return rows


@register_renderer("kernel_profiling")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the kernel-profiling section from recorded profile runs.

    Lists each profile / TraceLens run with its launch args, artifact
    paths and top-k kernel table. Skipped when no runs were recorded.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section, or a skipped placeholder when
            there are no profiling runs.
    """
    runs = breakdown.get("kernel_profiling") or []

    if not runs:
        return RenderedSection(
            section_id="kernel_profiling",
            title="Kernel Profiling",
            key_facts=["No profile or TraceLens runs recorded this session."],
            markdown_block="",
            warnings=[],
            skipped=True,
        )

    tools: set[str] = set()
    for run in runs:
        outputs = run.get("outputs") or {}
        tool = outputs.get("tool")
        if tool:
            tools.add(str(tool))

    facts: list[str] = [
        f"{len(runs)} profiling run(s): tools={', '.join(sorted(tools)) or 'unknown'}.",
    ]

    parts: list[str] = []
    for run in runs:
        run_id = run.get("run_id") or "—"
        outputs = run.get("outputs") or {}
        artifacts = run.get("artifacts") or {}
        launch = run.get("launch") or {}
        tool = outputs.get("tool") or "—"
        parts.append(f"**Run `{run_id}`** (task={run.get('task_id') or '—'}, tool={tool})")
        parts.append(
            md_kv_list(
                [
                    ("ts", run.get("ts")),
                    ("framework", run.get("framework")),
                    ("profile_config", run.get("profile_config_path")),
                    ("framework_args", (launch.get("framework_args") or "")[:160]),
                    ("framework_args_source", launch.get("framework_args_source")),
                    ("benchmark_report", artifacts.get("benchmark_report_path")),
                    ("tracelens_status_json", artifacts.get("tracelens_status_json")),
                    ("kernel_summary_csv", artifacts.get("kernel_summary_csv")),
                    ("trace_paths", len(artifacts.get("trace_paths") or [])),
                ]
            )
        )
        summary = outputs.get("analysis_summary")
        if summary:
            parts.append(f"- **analysis_summary**: {str(summary)[:500]}")
        kernels = outputs.get("top_kernels") or []
        if kernels:
            parts.append("")
            parts.append(
                md_table(
                    ["kernel_id", "name", "gpu_pct", "duration_us", "bottleneck"],
                    _kernel_rows(kernels),
                )
            )
            if len(kernels) > _MAX_KERNEL_ROWS:
                parts.append(f"_Showing top {_MAX_KERNEL_ROWS} of {len(kernels)} kernels._")
        parts.append("")

    return RenderedSection(
        section_id="kernel_profiling",
        title="Kernel Profiling",
        key_facts=facts,
        markdown_block="\n".join(parts).strip(),
        decisions=[],
        warnings=[],
        skipped=False,
    )
