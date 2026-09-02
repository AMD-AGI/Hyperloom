# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the Markdown handoff from Hyperloom to KernelForge."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import is_secret_shaped_env_name, redact_secret_values
from hyperloom.common.io import atomic_write_text
from hyperloom.inference_optimizer.session.session_paths import forge_handoff_dir

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
    try:
        context = state.current_profile_workload_context()
    except Exception:
        context = {}
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


def build_serving_context_md(state: Any, env_spec: Mapping[str, Any] | None = None) -> str:
    """Render the framework, serving arguments, and environment overrides."""
    context = _workload_context(state)
    spec = dict(env_spec) if isinstance(env_spec, Mapping) else {}
    config = spec.get("config") if isinstance(spec.get("config"), Mapping) else {}
    current_best = getattr(state, "current_best", None)
    current_best = current_best if isinstance(current_best, Mapping) else {}
    serving_config = context.get("serving_config")
    serving_config = serving_config if isinstance(serving_config, Mapping) else {}

    resolved_args = str(config.get("server_launch_flags") or context.get("server_args") or "").strip()
    extra_args = str(
        config.get("extra_server_args")
        or current_best.get("extra_server_args")
        or serving_config.get("extra_server_args")
        or ""
    ).strip()
    envs = _environment_overrides(context, spec)
    unset_envs = context.get("unset_envs") if isinstance(context.get("unset_envs"), list) else []

    lines = [
        "# Serving Context",
        "",
        f"- **Framework:** `{_display(context.get('framework') or getattr(state, 'framework', ''))}`",
        f"- **Framework version:** `{_display(getattr(state, 'framework_version', ''))}`",
        f"- **Launch recipe:** `{_display(_absolute_path(spec.get('launch_recipe') or getattr(state, 'baseline_config_path', '')))}`",
        f"- **Overlay Python path:** `{_display(_absolute_path(spec.get('overlay_pythonpath')))}`",
        "",
        "## Resolved Server Arguments",
        "",
        "```text",
        redact_secret_values(resolved_args) if resolved_args else "not available",
        "```",
        "",
        "## Additional Server Arguments",
        "",
        "```text",
        redact_secret_values(extra_args) if extra_args else "not available",
        "```",
        "",
        "## Environment Variable Overrides",
        "",
        "```text",
    ]
    lines.extend(f"{key}={value}" for key, value in envs.items())
    if not envs:
        lines.append("not available")
    lines.extend(
        [
            "```",
            "",
            "## Unset Environment Variables",
            "",
        ]
    )
    if unset_envs:
        lines.extend(f"- `{_display(value)}`" for value in unset_envs)
    else:
        lines.append("- not available")
    return "\n".join(lines) + "\n"


def _evidence_line(label: str, value: Any) -> str:
    path_text = _absolute_path(value)
    if not path_text:
        return f"- **{label}:** not provided"
    status = "available" if Path(path_text).exists() else "missing"
    return f"- **{label}:** `{path_text}` ({status})"


def build_trace_evidence_md(state: Any) -> str:
    """Render absolute paths to existing trace and TraceLens artifacts."""
    analysis = getattr(state, "last_trace_analyze", None)
    analysis = analysis if isinstance(analysis, Mapping) else {}
    candidates_path = _absolute_path(analysis.get("candidates_path"))
    source_resolution = str(Path(candidates_path).parent / "kernel_source_resolution.json") if candidates_path else ""
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
    return "\n".join(lines) + "\n"


def write_forge_handoff(
    session_dir: Path,
    state: Any,
    *,
    env_spec: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one macro cycle's Forge handoff and return its directory."""
    handoff_dir = forge_handoff_dir(
        Path(session_dir),
        int(getattr(state, "macro_cycle", 0) or 0),
    )
    documents = {
        WORKLOAD_FILENAME: build_workload_md(state),
        SERVING_CONTEXT_FILENAME: build_serving_context_md(state, env_spec),
        TRACE_EVIDENCE_FILENAME: build_trace_evidence_md(state),
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
