# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Environment-variable knobs for the trace subsystem."""

from __future__ import annotations

import os

# Master switch for live Langfuse push.
ENV_LANGFUSE_ENABLE = "HYPERLOOM_LANGFUSE_ENABLE"

# Langfuse connection credentials (official langfuse SDK variable names).
ENV_LANGFUSE_HOST = "LANGFUSE_HOST"
ENV_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"

# SDK batch-flush cadence (official langfuse SDK variable names).
ENV_LANGFUSE_FLUSH_INTERVAL = "LANGFUSE_FLUSH_INTERVAL"
_DEFAULT_FLUSH_INTERVAL = "1"

_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var with the project-standard token vocabulary."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return default


def langfuse_live_enabled() -> bool:
    """Report whether the live-Langfuse master switch is on (default off)."""
    return env_flag(ENV_LANGFUSE_ENABLE, default=False)


def langfuse_credentials() -> dict[str, str]:
    """Return the three Langfuse connection vars that are set (stripped)."""
    out: dict[str, str] = {}
    for key in (ENV_LANGFUSE_HOST, ENV_LANGFUSE_PUBLIC_KEY, ENV_LANGFUSE_SECRET_KEY):
        val = (os.environ.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def langfuse_credentials_complete() -> bool:
    """True iff all three Langfuse connection vars are present and non-empty."""
    return len(langfuse_credentials()) == 3


def apply_flush_defaults() -> None:
    """Seed the SDK's auto-flush cadence before the client is built."""
    os.environ.setdefault(ENV_LANGFUSE_FLUSH_INTERVAL, _DEFAULT_FLUSH_INTERVAL)


__all__ = [
    "ENV_LANGFUSE_ENABLE",
    "ENV_LANGFUSE_FLUSH_INTERVAL",
    "ENV_LANGFUSE_HOST",
    "ENV_LANGFUSE_PUBLIC_KEY",
    "ENV_LANGFUSE_SECRET_KEY",
    "apply_flush_defaults",
    "env_flag",
    "langfuse_credentials",
    "langfuse_credentials_complete",
    "langfuse_live_enabled",
]
