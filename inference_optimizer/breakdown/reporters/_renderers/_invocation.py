# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared invocation-block renderer for baseline / final sections.

Centralised here because both renderers need the exact same five-line
markdown block (``image``, ``config``, ``command``, ``envs``, ``server
log``). Keeping the layout identical across sections makes diffs across
two reports easy to scan.

Filename starts with ``_`` to keep it out of the auto-imported renderer
list (``compose.py`` walks ``_renderers/*.py`` looking for
``@register_renderer`` decorators; this module has none).
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_invocation_block"]


# Hard cap for the framework_args echo. Real launch commands fit
# comfortably in ~250 chars; pathologically long ones (10+ args, weka
# paths) get truncated so they don't blow the markdown column width.
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
    """Render an ``### Invocation`` markdown block (or "" when absent).

    Returns the empty string when ``invocation`` is missing / has no
    fields populated, so callers can ``if inv_md: ...`` to skip the
    sub-section gracefully on V1 breakdown JSONs that predate the
    invocation field.
    """
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
    # Lineage label sits directly under ``command`` so an operator can
    # see at a glance whether the echoed string came from a parsed
    # ``Server arguments:`` line, a literal python invocation, the
    # config yaml, or no source at all (extraction failed).
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
