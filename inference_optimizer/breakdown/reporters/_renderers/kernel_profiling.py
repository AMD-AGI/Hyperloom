"""Kernel profiling renderer (v1.1).

Surfaces profile / TraceLens runs: launch args, artifact paths,
parsed top-k kernels, and (verbose only) the last 40 lines of
tracelens CLI logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import RenderedSection, md_kv_list, md_table, register_renderer

_CLI_LOG_TAIL_LINES = 40
_MAX_KERNEL_ROWS = 15


def _session_root(breakdown: dict[str, Any]) -> Path | None:
    """Resolve the on-disk session root from the breakdown.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        Path | None: The session directory as a :class:`~pathlib.Path`, or
            ``None`` when ``session.session_dir`` is absent.
    """
    session = breakdown.get("session") or {}
    raw = session.get("session_dir")
    if not raw:
        return None
    return Path(str(raw))


def _read_log_tail(session_root: Path | None, rel_path: str | None) -> str:
    """Read the last few lines of a log file relative to the session root.

    Args:
        session_root (Path | None): The session directory, or ``None``.
        rel_path (str | None): Path to the log file relative to the session
            root, or ``None``.

    Returns:
        str: Up to ``_CLI_LOG_TAIL_LINES`` trailing lines of the file, or an
            empty string when inputs are missing or the file is unreadable.
    """
    if not session_root or not rel_path:
        return ""
    path = session_root / rel_path
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if len(lines) <= _CLI_LOG_TAIL_LINES:
        return "\n".join(lines)
    return "\n".join(lines[-_CLI_LOG_TAIL_LINES:])


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
        rows.append([
            k.get("kernel_id") or "—",
            (k.get("name") or "")[:60],
            k.get("gpu_pct"),
            k.get("duration_us"),
            k.get("bottleneck") or "—",
        ])
    return rows


@register_renderer("kernel_profiling")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the kernel-profiling section from recorded profile runs.

    Lists each profile / TraceLens run with its launch args, artifact
    paths and top-k kernel table, optionally including a CLI log tail when
    the detail level is verbose. Skipped when no runs were recorded.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered section, or a skipped placeholder when
            there are no profiling runs.
    """
    runs = breakdown.get("kernel_profiling") or []
    detail_level = str(breakdown.get("detail_level") or "standard")
    session_root = _session_root(breakdown)

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
        parts.append(md_kv_list([
            ("ts", run.get("ts")),
            ("framework", run.get("framework")),
            ("profile_config", run.get("profile_config_path")),
            ("framework_args", (launch.get("framework_args") or "")[:160]),
            ("framework_args_source", launch.get("framework_args_source")),
            ("benchmark_report", artifacts.get("benchmark_report_path")),
            ("tracelens_status_json", artifacts.get("tracelens_status_json")),
            ("kernel_summary_csv", artifacts.get("kernel_summary_csv")),
            ("trace_paths", len(artifacts.get("trace_paths") or [])),
        ]))
        summary = outputs.get("analysis_summary")
        if summary:
            parts.append(f"- **analysis_summary**: {str(summary)[:500]}")
        kernels = outputs.get("top_kernels") or []
        if kernels:
            parts.append("")
            parts.append(md_table(
                ["kernel_id", "name", "gpu_pct", "duration_us", "bottleneck"],
                _kernel_rows(kernels),
            ))
            if len(kernels) > _MAX_KERNEL_ROWS:
                parts.append(f"_Showing top {_MAX_KERNEL_ROWS} of {len(kernels)} kernels._")
        log_rel = artifacts.get("tracelens_log")
        if detail_level == "verbose" and log_rel:
            tail = _read_log_tail(session_root, str(log_rel))
            if tail:
                parts.append("")
                parts.append(
                    f"<details><summary>CLI log tail ({_CLI_LOG_TAIL_LINES} lines max): "
                    f"`{log_rel}`</summary>"
                )
                parts.append("")
                parts.append("```")
                parts.append(tail)
                parts.append("```")
                parts.append("")
                parts.append("</details>")
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
