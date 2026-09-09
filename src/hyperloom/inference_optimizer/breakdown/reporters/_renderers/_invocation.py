# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared invocation-block renderer for baseline / final sections."""

from __future__ import annotations

from typing import Any

__all__ = ["render_invocation_block"]


# Cap for the framework_args echo so long commands don't blow the column width.
_FRAMEWORK_ARGS_MAX = 200
_ENVS_MAX_DISPLAY = 12

#: What a source label means for a reader who wants the flags and did not get
#: them. Only the labels that report an absence need one.
_SOURCE_HINTS = {
    "unknown": "  (extraction failed; try server.log or config yaml)",
    "unrecorded": "  (the measurement recorded no launch flags)",
}


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _format_envs(envs: dict[str, Any] | None) -> str:
    """Format environment variables as a compact, capped string."""
    if not isinstance(envs, dict) or not envs:
        return ""
    items = sorted(envs.items())
    if len(items) <= _ENVS_MAX_DISPLAY:
        return ", ".join(f"{k}={v}" for k, v in items)
    shown = items[:_ENVS_MAX_DISPLAY]
    extra = len(items) - _ENVS_MAX_DISPLAY
    return ", ".join(f"{k}={v}" for k, v in shown) + f", ... +{extra} more"


def render_invocation_block(
    invocation: Any,
    session_image: Any,
) -> str:
    """Render an ``### Invocation`` markdown block, or \"\" when absent/empty."""
    if not isinstance(invocation, dict):
        return ""
    framework_args = str(invocation.get("framework_args") or "").strip()
    framework_args_source = str(invocation.get("framework_args_source") or "").strip()
    extra_envs = invocation.get("extra_envs")
    config_path = invocation.get("config_path")
    server_log_path = invocation.get("server_log_path")
    # The run directory the measurement was taken in, which is where its
    # config and server log live. Recorded in place of the two paths by
    # producers that name the directory rather than walking it for them.
    workspace = invocation.get("workspace")

    has_anything = (
        framework_args or (isinstance(extra_envs, dict) and extra_envs) or config_path or server_log_path or workspace
    )
    if not has_anything:
        return ""

    image_display = (
        str(session_image).strip() if isinstance(session_image, str) and session_image.strip() else "(not configured)"
    )
    lines = ["### Invocation"]
    lines.append(f"- **image**: {image_display}")
    if config_path:
        lines.append(f"- **config**: `{config_path}`")
    if framework_args:
        lines.append(f"- **command**: `{_truncate(framework_args, _FRAMEWORK_ARGS_MAX)}`")
    if framework_args_source:
        suffix = _SOURCE_HINTS.get(framework_args_source, "")
        lines.append(f"- **source**: {framework_args_source}{suffix}")
    envs_str = _format_envs(extra_envs if isinstance(extra_envs, dict) else None)
    if envs_str:
        lines.append(f"- **envs**: `{envs_str}`")
    if server_log_path:
        lines.append(f"- **server log**: `{server_log_path}`")
    if workspace:
        lines.append(f"- **workspace**: `{workspace}`")
    return "\n".join(lines)
