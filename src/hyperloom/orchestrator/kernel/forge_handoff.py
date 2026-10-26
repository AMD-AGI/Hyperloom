# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the Markdown handoff from Hyperloom to KernelForge."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import is_secret_shaped_env_name, redact_secret_values
from hyperloom.common.io import WriteOptions, atomic_write_text
from hyperloom.common.kernel_source_contract import SOURCE_RESOLUTION_FILENAME
from hyperloom.inference_optimizer.roofline_snapshot import extract_workload_summary
from hyperloom.inference_optimizer.session.session_paths import forge_handoff_dir, jit_cache_dir
from hyperloom.orchestrator.kernel.campaign_baseline import campaign_repositories
from hyperloom.orchestrator.kernel.gemm_shape_coverage import parse_aiter_shape_lookups, traced_gemm_shapes
from hyperloom.orchestrator.measurement.jit_variants import render_jit_variants, variant_counts
from kernelforge.foldable_launches import foldable_launches, render_foldable_launches
from kernelforge.graph_coverage import graph_coverage, render_graph_coverage
from kernelforge.trace_events import resolve_capture
from kernelforge.serving_windows import hot_kernels_by_class, render_hot_kernels_by_class
from hyperloom.orchestrator.measurement.runtime_findings import (
    AITER_TUNED_MISS,
    load_runtime_findings,
    render_runtime_findings,
)

WORKLOAD_FILENAME = "workload.md"
SERVING_CONTEXT_FILENAME = "serving-context.md"
TRACE_EVIDENCE_FILENAME = "trace-evidence.md"


def _display(value: Any) -> str:
    text = str(value if value not in (None, "") else "not available")
    return redact_secret_values(text)


def _absolute_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return str(Path(raw).expanduser().resolve(strict=False))


def _workload_context(state: Any) -> dict[str, Any]:
    context = state.current_profile_workload_context()
    return dict(context) if isinstance(context, Mapping) else {}


def build_workload_md(state: Any) -> str:
    """Render the active workload without deriving optimization candidates."""
    context = _workload_context(state)
    fields = (
        ("Model name", getattr(state, "model_name", "")),
        ("Model path", context.get("model_path") or getattr(state, "model_path", "")),
        ("Model class", getattr(state, "model_class", "")),
        ("Precision", context.get("precision") or getattr(state, "precision", "")),
        ("Tensor parallelism", context.get("tp") or getattr(state, "tp", 0)),
        ("Expert parallelism", getattr(state, "ep", 0)),
        ("Input sequence length", context.get("isl") or getattr(state, "isl", 0)),
        ("Output sequence length", context.get("osl") or getattr(state, "osl", 0)),
        ("Concurrency", context.get("conc") or getattr(state, "conc", 0)),
        ("Maximum model length", context.get("max_model_len") or getattr(state, "max_model_len", 0)),
    )
    lines = ["# Workload", ""]
    lines.extend(f"- **{label}:** `{_display(value)}`" for label, value in fields)
    return "\n".join(lines) + "\n"


def _environment_overrides(
    context: Mapping[str, Any],
    env_spec: Mapping[str, Any],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (
        context.get("extra_envs"),
        (env_spec.get("config") or {}).get("extra_envs") if isinstance(env_spec.get("config"), Mapping) else None,
    ):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            name = str(key or "").strip()
            if not name or is_secret_shaped_env_name(name):
                continue
            merged[name] = redact_secret_values(str(value))
    return dict(sorted(merged.items()))


def _fenced_section(heading: str, *body: str) -> list[str]:
    """A Markdown section, opened by a blank line, whose content is one fenced text block."""
    return ["", heading, "", "```text", *([line for line in body if line] or ["not available"]), "```"]


def _bullet_section(heading: str, *body: str) -> list[str]:
    """A Markdown section, opened by a blank line, whose content is a bullet list."""
    return ["", heading, "", *([f"- {line}" for line in body if line] or ["- not available"])]


def build_serving_context_md(
    state: Any,
    env_spec: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None,
) -> str:
    """Render the framework, serving arguments, and environment overrides."""
    context = _workload_context(state)
    spec = env_spec or {}
    config = spec.get("config") or {}
    resolved_args = str(config.get("server_launch_flags") or context.get("server_args") or "").strip()
    # The launch config, then the promoted best, then the profile's own serving block: first one that names them wins.
    extra_sources = (config, getattr(state, "current_best", None) or {}, context.get("serving_config") or {})
    extra_args = str(next((s["extra_server_args"] for s in extra_sources if s.get("extra_server_args")), "")).strip()
    envs = _environment_overrides(context, spec)
    # Taken from the seal that just ran rather than read back from Git: the
    # commit it returned is the base every rewrite of that repository diffs
    # from, and re-reading HEAD would both cost a subprocess per repository
    # and leave a window in which the two could disagree.
    sealed = {str(repo): getattr(baseline, "commit", "") for repo, baseline in (baselines or {}).items()}
    source_roots = tuple(sorted(sealed)) or tuple(map(str, campaign_repositories(state)))

    lines = [
        "# Serving Context",
        "",
        f"- **Framework:** `{_display(context.get('framework') or getattr(state, 'framework', ''))}`",
        f"- **Framework version:** `{_display(getattr(state, 'framework_version', ''))}`",
        f"- **Launch recipe:** `{_display(_absolute_path(spec.get('launch_recipe') or getattr(state, 'baseline_config_path', '')))}`",
        f"- **Overlay Python path:** `{_display(_absolute_path(spec.get('overlay_pythonpath')))}`",
        # Reported with the commit each one currently sits at, which the caller
        # sealed just before this was written. That object id is the base every
        # rewrite of this repository diffs from, so a task naming a path Git does
        # not carry there is answerable from this document alone.
        *_bullet_section(
            "## Source Repositories",
            *(f"`{root}` @ `{sealed.get(root) or 'unknown'}`" for root in source_roots),
        ),
        *_fenced_section("## Resolved Server Arguments", redact_secret_values(resolved_args)),
        *_fenced_section("## Additional Server Arguments", redact_secret_values(extra_args)),
        *_fenced_section(
            "## Environment Variable Overrides",
            *(f"{key}={value}" for key, value in envs.items()),
        ),
        *_bullet_section(
            "## Unset Environment Variables",
            *(f"`{_display(value)}`" for value in context.get("unset_envs") or ()),
        ),
    ]
    return "\n".join(lines) + "\n"


def _evidence_line(label: str, value: Any) -> str:
    path_text = _absolute_path(value)
    if not path_text:
        return f"- **{label}:** not provided"
    status = "available" if Path(path_text).exists() else "missing"
    return f"- **{label}:** `{path_text}` ({status})"


def build_trace_evidence_md(state: Any, *, session_dir: Path | None = None) -> str:
    """Render absolute paths to existing trace and TraceLens artifacts, and what the session recompiled."""
    analysis = getattr(state, "last_trace_analyze", None)
    analysis = analysis if isinstance(analysis, Mapping) else {}
    candidates_path = _absolute_path(analysis.get("candidates_path"))
    source_resolution = str(Path(candidates_path).parent / SOURCE_RESOLUTION_FILENAME) if candidates_path else ""
    evidence = (
        ("Profile raw trace", getattr(state, "last_profile_trace", "")),
        ("TraceLens input trace", analysis.get("trace_input")),
        ("TraceLens steady-state trace", analysis.get("steady_state_trace")),
        ("TraceLens analysis", analysis.get("analysis_md_path")),
        ("Kernel candidates", candidates_path),
        ("Kernel source resolution", source_resolution),
        ("Kernel roofline", analysis.get("kernel_roofline_path")),
    )
    lines = ["# Trace Evidence", ""]
    lines.extend(_evidence_line(label, value) for label, value in evidence)

    warnings = analysis.get("trace_health_warnings")
    lines.extend(["", "## Trace Health Warnings", ""])
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            if isinstance(warning, Mapping):
                code = str(warning.get("code") or "warning")
                message = str(warning.get("message") or warning.get("detail") or "")
                lines.append(f"- **{_display(code)}:** {_display(message)}")
            else:
                lines.append(f"- {_display(warning)}")
    else:
        lines.append("- none")

    lines.extend(
        ["", "## Exposed Memcpy", "", "```text", _exposed_memcpy_line(analysis.get("analysis_md_path")), "```"]
    )
    lines.extend(["", "## Runtime Findings", ""])
    measurement = getattr(state, "current_best_measurement", None)
    lines.extend(["```text", render_runtime_findings(measurement) if measurement else "not available", "```"])
    lines.extend(["", "## Hot GEMMs Missing Tuned Config", ""])
    profiled = {"launch_evidence_path": str(getattr(state, "last_profile_launch_evidence_path", "") or "")}
    lines.extend(["```text", *_tuned_miss_lines(candidates_path, profiled), "```"])
    lines.extend(["", "## Recompiled Kernels", ""])
    lines.extend(_jit_variant_lines(session_dir))
    capture = resolve_capture(getattr(state, "last_profile_trace", ""))
    lines.extend(["", "## Hot Kernels by Step Class", ""])
    lines.extend(_hot_kernel_lines(capture))
    lines.extend(["", "## Launch-Bound Kernels", ""])
    lines.extend(_foldable_lines(capture))
    lines.extend(["", "## Graph Coverage", ""])
    lines.extend(_graph_coverage_lines(capture))
    return "\n".join(lines) + "\n"


def _graph_coverage_lines(capture: str) -> list[str]:
    """Which step classes replay a captured graph and which still launch every kernel."""
    rendered = render_graph_coverage(graph_coverage(capture))
    return rendered or ["- the capture records no kernel-launch calls, so graph coverage cannot be read from it"]


def _foldable_lines(capture: str) -> list[str]:
    """Kernels costing a launch rather than their work, with the neighbour each could fold into."""
    rendered = render_foldable_launches(foldable_launches(capture))
    return rendered or ["- no launch-bound kernel crossed the reporting floor in any step class"]


def _hot_kernel_lines(capture: str) -> list[str]:
    """Per-step-class kernel rankings read straight off the raw capture, without TraceLens."""
    by_class = hot_kernels_by_class(capture)
    if not by_class:
        return [
            "- the raw capture carries no `step[...]` annotations, so no step class could be ranked "
            "(a capture taken outside the serving loop looks like this)"
        ]
    return render_hot_kernels_by_class(by_class)


def _jit_variant_lines(session_dir: Path | None) -> list[str]:
    """Kernels this session recompiled per call shape, which cost engine stalls no kernel rewrite can recover."""
    rows = variant_counts(jit_cache_dir(session_dir)) if session_dir is not None else []
    if not rows:
        return ["- none above the reporting floor, or the session's JIT cache is not readable"]
    return [
        *render_jit_variants(rows),
        "",
        "Each first compile stalls the whole engine. A kernel whose variants differ in one constant is "
        "specialising on a runtime quantity: making that argument runtime rather than `constexpr` removes the "
        "recompiles without touching what the kernel computes.",
    ]


def _exposed_memcpy_line(analysis_md_path: Any) -> str:
    """Exposed memcpy share read beside analysis.md, which omits it."""
    path_text = _absolute_path(analysis_md_path)
    memcpy_pct = extract_workload_summary(path_text)["memcpy_pct"] if path_text else None
    return "not available" if memcpy_pct is None else f"exposed_memcpy={memcpy_pct}% of GPU time"


def _tuned_miss_lines(candidates_path: str, profiled: Mapping[str, Any]) -> list[str]:
    """Name the candidate GEMM rows whose traced shapes the profiled run's server log reported as untuned.

    Prefill M depends on batching, so only the run that produced the trace can match its shapes.
    """
    findings = load_runtime_findings(profiled)
    entry = next((f for f in (findings or {}).get("findings", []) if f["rule_id"] == AITER_TUNED_MISS), None)
    if entry is None or not candidates_path or not Path(candidates_path).is_file():
        return ["not available"]
    if entry["status"] == "unknown":
        return [f"unknown: {entry['reason']}"]
    if entry["status"] == "not_detected":
        return ["none: the profiled server log reports no tuned-config miss"]
    missed, _hit = parse_aiter_shape_lookups(Path(findings["log_path"]).read_text(encoding="utf-8", errors="replace"))
    rows = json.loads(Path(candidates_path).read_text(encoding="utf-8")).get("hot_kernels", [])
    lines = []
    for row in rows:
        shapes = sorted({shape for shape, _calls in traced_gemm_shapes(row) if shape in missed})
        if shapes:
            dims = ", ".join(f"M={m} N={n} K={k}" for m, n, k in shapes)
            lines.append(
                f"- {_display(row.get('kernel_id'))} {_display(row.get('name'))} gpu_pct={row.get('gpu_pct')}: {dims}"
            )
    if not lines:
        lines.append(f"- no hot GEMM shape matches the {len(missed)} missed shape(s)")
    lines.append("Rows not listed: unknown; a tuned-config hit is only logged under AITER_LOG_TUNED_CONFIG.")
    return lines


def write_forge_handoff(
    session_dir: Path,
    state: Any,
    *,
    env_spec: Mapping[str, Any] | None = None,
    handoff_dir: Path | None = None,
    baselines: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one Forge handoff and return its directory."""
    if handoff_dir is None:
        handoff_dir = forge_handoff_dir(
            Path(session_dir),
            int(getattr(state, "macro_cycle", 0) or 0),
        )
    handoff_dir = Path(handoff_dir)
    documents = {
        WORKLOAD_FILENAME: build_workload_md(state),
        SERVING_CONTEXT_FILENAME: build_serving_context_md(state, env_spec, baselines),
        TRACE_EVIDENCE_FILENAME: build_trace_evidence_md(state, session_dir=Path(session_dir)),
    }
    for filename, text in documents.items():
        atomic_write_text(
            handoff_dir / filename,
            text,
            options=WriteOptions(make_parents=True, fsync=True, fsync_dir=True, mode=0o600),
        )
    return handoff_dir


__all__ = [
    "SERVING_CONTEXT_FILENAME",
    "TRACE_EVIDENCE_FILENAME",
    "WORKLOAD_FILENAME",
    "build_serving_context_md",
    "build_trace_evidence_md",
    "build_workload_md",
    "write_forge_handoff",
]
