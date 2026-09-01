"""Canonical PR Monitor URLs derived from the KB Store service endpoint."""

from __future__ import annotations

import os
from collections.abc import Mapping


KB_STORE_URL_ENV = "KB_STORE_URL"
PR_MONITOR_ENABLED_ENV = "HYPERLOOM_PR_MONITOR_ENABLED"
PR_MONITOR_MOUNT = "pr-monitor"
DEFAULT_KB_STORE_URL = "https://global.primus-safe.amd.com/knowledge-base"


def pr_monitor_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether runtime preflight left PR Monitor enabled."""

    source = os.environ if env is None else env
    value = str(source.get(PR_MONITOR_ENABLED_ENV, "")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def kb_store_url(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the KB Service URL used by PR Monitor.

    Local Recipe mode falls back to the default KB Service so PR discovery works
    without extra configuration. Remote Recipe mode stays strict: its write
    credentials must include an explicit URL and token.
    """

    source = os.environ if env is None else env
    configured = str(value if value is not None else source.get(KB_STORE_URL_ENV, "")).strip().rstrip("/")
    if configured:
        return configured
    mode = str(source.get("KNOWLEDGE_STORE_MODE") or "local").strip().lower()
    return DEFAULT_KB_STORE_URL if mode == "local" else ""


def pr_monitor_base_url(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the co-hosted PR Monitor service root."""

    if not pr_monitor_enabled(env):
        return ""
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
    "DEFAULT_KB_STORE_URL",
    "KB_STORE_URL_ENV",
    "PR_MONITOR_ENABLED_ENV",
    "PR_MONITOR_MOUNT",
    "kb_store_url",
    "pr_monitor_base_url",
    "pr_monitor_enabled",
    "pr_monitor_mcp_url",
    "pr_monitor_rest_url",
]
