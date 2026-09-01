"""Canonical PR Monitor URLs derived from the KB Store service endpoint."""

from __future__ import annotations

import os
from collections.abc import Mapping


KB_STORE_URL_ENV = "KB_STORE_URL"
PR_MONITOR_MOUNT = "pr-monitor"


def kb_store_url(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the normalized KB Store service URL, or ``""`` when unset."""

    source = os.environ if env is None else env
    return str(value if value is not None else source.get(KB_STORE_URL_ENV, "")).strip().rstrip("/")


def pr_monitor_base_url(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the co-hosted PR Monitor service root."""

    base = kb_store_url(value, env=env)
    return f"{base}/{PR_MONITOR_MOUNT}" if base else ""


def pr_monitor_rest_url(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return ``<KB_STORE_URL>/pr-monitor/v1``."""

    base = pr_monitor_base_url(value, env=env)
    return f"{base}/v1" if base else ""


def pr_monitor_mcp_url(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return ``<KB_STORE_URL>/pr-monitor/mcp/`` with the required slash."""

    base = pr_monitor_base_url(value, env=env)
    return f"{base}/mcp/" if base else ""


__all__ = [
    "KB_STORE_URL_ENV",
    "PR_MONITOR_MOUNT",
    "kb_store_url",
    "pr_monitor_base_url",
    "pr_monitor_mcp_url",
    "pr_monitor_rest_url",
]
