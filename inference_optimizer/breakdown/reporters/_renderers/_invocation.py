# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared invocation-block renderer for baseline / final sections.

Centralised so both renderers emit the identical five-line block. The
``_`` prefix keeps it out of ``compose.py``'s auto-imported renderer walk
(it registers no renderer).
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_invocation_block"]


# Hard cap for the framework_args echo so long commands don't blow the column width.
_FRAMEWORK_ARGS_MAX = 200
_ENVS_MAX_DISPLAY = 12


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _format_envs(envs: dict[str, Any] | None) -> str:
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
    """Render an ``### Invocation`` markdown block, or "" when absent/empty."""
    if not isinstance(invocation, dict):
        return ""
    framework_args = str(invocation.get("framework_args") or "").strip()
    framework_args_source = str(invocation.get("framework_args_source") or "").strip()
    extra_envs = invocation.get("extra_envs")
    config_path = invocation.get("config_path")
    server_log_path = invocation.get("server_log_path")

    has_anything = (
        framework_args
        or (isinstance(extra_envs, dict) and extra_envs)
        or config_path
        or server_log_path
    )
    if not has_anything:
        return ""

    image_display = (
        str(session_image).strip()
        if isinstance(session_image, str) and session_image.strip()
        else "(not configured)"
    )
    lines = ["### Invocation"]
    lines.append(f"- **image**: {image_display}")
    if config_path:
        lines.append(f"- **config**: `{config_path}`")
    if framework_args:
        lines.append(f"- **command**: `{_truncate(framework_args, _FRAMEWORK_ARGS_MAX)}`")
    # Lineage label under ``command`` shows where the echoed string came from.
    if framework_args_source:
        suffix = (
            "  (extraction failed; try server.log or config yaml)"
            if framework_args_source == "unknown"
            else ""
        )
        lines.append(f"- **source**: {framework_args_source}{suffix}")
    envs_str = _format_envs(extra_envs if isinstance(extra_envs, dict) else None)
    if envs_str:
        lines.append(f"- **envs**: `{envs_str}`")
    if server_log_path:
        lines.append(f"- **server log**: `{server_log_path}`")
    return "\n".join(lines)
