# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Environment-variable knobs for the trace subsystem.

Centralizes bool parsing for the trace package so the Langfuse toggle reads
identically everywhere. Owns only the trace-related names, prefixed
``HYPERLOOM_*`` for cross-component escape hatches.
"""

from __future__ import annotations

import os

# Master switch for live Langfuse push. Default OFF: Langfuse is an opt-in
# parallel sink alongside the always-written local jsonl ledger.
ENV_LANGFUSE_ENABLE = "HYPERLOOM_LANGFUSE_ENABLE"

# Langfuse connection credentials (official langfuse SDK variable names).
ENV_LANGFUSE_HOST = "LANGFUSE_HOST"
ENV_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"

# SDK batch-flush cadence (official langfuse SDK variable names). Tighten the
# flush interval to 1s by default so a hard-killed session loses at most ~1s of
# observations; ``flush_at`` stays at the SDK default (512) so steady-state
# traffic still batches normally.
ENV_LANGFUSE_FLUSH_INTERVAL = "LANGFUSE_FLUSH_INTERVAL"
_DEFAULT_FLUSH_INTERVAL = "1"

_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var with the project-standard token vocabulary.

    Accepts ``1/true/yes/on`` (True) and ``0/false/no/off`` (False),
    case-insensitive. An unset or unrecognized value returns ``default``.

    Args:
        name: Environment variable name to read.
        default: Value returned when the var is unset or unrecognized.

    Returns:
        The parsed boolean, or ``default`` when not recognized.
    """
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
    """Report whether the live-Langfuse master switch is on (default off).

    Returns:
        True when the master switch env var is enabled.
    """
    return env_flag(ENV_LANGFUSE_ENABLE, default=False)


def langfuse_credentials() -> dict[str, str]:
    """Return the three Langfuse connection vars that are set (stripped).

    Missing / blank vars are omitted; callers treat an incomplete set as
    "not configured" and degrade to a no-op.

    Returns:
        Mapping of env var name to stripped value for each set variable.
    """
    out: dict[str, str] = {}
    for key in (ENV_LANGFUSE_HOST, ENV_LANGFUSE_PUBLIC_KEY, ENV_LANGFUSE_SECRET_KEY):
        val = (os.environ.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def langfuse_credentials_complete() -> bool:
    """True iff all three Langfuse connection vars are present and non-empty.

    Returns:
        True when all three connection vars are set and non-empty.
    """
    return len(langfuse_credentials()) == 3


def apply_flush_defaults() -> None:
    """Seed the SDK's auto-flush cadence before the client is built.

    Must run before the first ``get_client()`` call, since the SDK reads
    ``LANGFUSE_FLUSH_INTERVAL`` / ``LANGFUSE_FLUSH_AT`` only when it lazily
    constructs the singleton client. Uses ``setdefault`` so an operator-supplied
    value always wins.
    """
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
