# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Project a :class:`KernelContext` onto the Markdown handoff KernelForge reads."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.io import atomic_write_text

from .kernel_context import ArtifactRef, KernelContext

WORKLOAD_FILENAME = "workload.md"
SERVING_CONTEXT_FILENAME = "serving-context.md"
TRACE_EVIDENCE_FILENAME = "trace-evidence.md"


def _display(value: object) -> str:
    return str(value if value not in (None, "", 0) else "not available")


def build_workload_md(context: KernelContext) -> str:
    """Render the active workload without deriving optimization candidates."""
    workload = context.workload
    fields = (
        ("Model name", workload.model_name),
        ("Model path", workload.model_path),
        ("Model class", workload.model_class),
        ("Precision", workload.precision),
        ("Quantization", workload.quant_type),
        # The Controller's task contract requires a normalized ``identity.gpu``
        # and forge-loop derives ``--gpu-target`` from it, so withholding this
        # left the analysis agent inferring the accelerator it is tuning for.
        ("GPU", workload.gpu_type),
        ("Tensor parallelism", workload.tp),
        ("Expert parallelism", workload.ep),
        ("Input sequence length", workload.isl),
        ("Output sequence length", workload.osl),
        ("Concurrency", workload.conc),
        ("Maximum model length", workload.max_model_len),
    )
    lines = ["# Workload", ""]
    lines.extend(f"- **{label}:** `{_display(value)}`" for label, value in fields)
    return "\n".join(lines) + "\n"


def build_serving_context_md(context: KernelContext) -> str:
    """Render the framework, serving arguments, and environment overrides."""
    serving = context.serving
    lines = [
        "# Serving Context",
        "",
        f"- **Framework:** `{_display(serving.framework)}`",
        f"- **Framework version:** `{_display(serving.framework_version)}`",
        f"- **Launch recipe:** `{_display(serving.launch_recipe)}`",
        f"- **Overlay Python path:** `{_display(serving.overlay_pythonpath)}`",
        "",
        "## Source Repositories",
        "",
    ]
    if serving.source_repo_roots:
        lines.extend(f"- `{root}`" for root in serving.source_repo_roots)
    else:
        lines.append("- not available")
    lines.extend(
        [
            "",
            "## Resolved Server Arguments",
            "",
            "```text",
            serving.server_args or "not available",
            "```",
            "",
            "## Additional Server Arguments",
            "",
            "```text",
            serving.extra_server_args or "not available",
            "```",
            "",
            "## Environment Variable Overrides",
            "",
            "```text",
        ]
    )
    lines.extend(f"{key}={value}" for key, value in serving.extra_envs.items())
    if not serving.extra_envs:
        lines.append("not available")
    lines.extend(
        [
            "```",
            "",
            "## Unset Environment Variables",
            "",
        ]
    )
    if serving.unset_envs:
        lines.extend(f"- `{_display(value)}`" for value in serving.unset_envs)
    else:
        lines.append("- not available")
    return "\n".join(lines) + "\n"


def _evidence_line(label: str, ref: ArtifactRef) -> str:
    if not ref.path:
        return f"- **{label}:** not provided"
    return f"- **{label}:** `{ref.path}` ({'available' if ref.available else 'missing'})"


def build_trace_evidence_md(context: KernelContext) -> str:
    """Render absolute paths to existing trace and TraceLens artifacts."""
    evidence = context.evidence
    entries = (
        ("Profile raw trace", evidence.profile_trace),
        ("TraceLens input trace", evidence.trace_input),
        ("TraceLens steady-state trace", evidence.steady_state_trace),
        ("TraceLens analysis", evidence.analysis_md),
        ("Kernel candidates", evidence.kernel_candidates),
        ("Kernel source resolution", evidence.kernel_source_resolution),
        ("Kernel roofline", evidence.kernel_roofline),
        # Both were resolved for the GEMM lane alone. The analysis agent is told
        # to cross-check TraceLens against serving logs and to derive shape
        # cases, and could do neither without being told where these are.
        ("Serving log", evidence.server_log),
        ("Trace shape manifest", evidence.shape_manifest),
    )
    lines = ["# Trace Evidence", ""]
    lines.extend(_evidence_line(label, ref) for label, ref in entries)

    lines.extend(["", "## Trace Health Warnings", ""])
    if evidence.trace_health_warnings:
        for warning in evidence.trace_health_warnings:
            code = _display(warning.get("code") or "warning")
            message = _display(warning.get("message") or warning.get("detail") or "")
            lines.append(f"- **{code}:** {message}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_forge_handoff(context: KernelContext, handoff_dir: Path) -> Path:
    """Atomically write one Forge handoff and return its directory.

    ``handoff_dir`` rides inside the controller output that consumes it, so a
    second attempt within one macro cycle cannot overwrite the evidence the
    first one was given.
    """
    handoff_dir = Path(handoff_dir)
    documents = {
        WORKLOAD_FILENAME: build_workload_md(context),
        SERVING_CONTEXT_FILENAME: build_serving_context_md(context),
        TRACE_EVIDENCE_FILENAME: build_trace_evidence_md(context),
    }
    for filename, text in documents.items():
        atomic_write_text(
            handoff_dir / filename,
            text,
            make_parents=True,
            fsync=True,
            fsync_dir=True,
            mode=0o600,
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
