# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Environment-variable knobs for the trace subsystem.

Centralizes the (otherwise scattered) bool parsing for the trace package so
the Langfuse toggle reads identically everywhere. Hyperloom has no global
``env.py``; the convention is documented in
``docs/CONFIGURATION_REFERENCE.md`` and prefixed ``HYPERLOOM_*`` for
cross-component escape hatches. This module owns only the trace-related
names.
"""

from __future__ import annotations

import os

# Master switch for live Langfuse push. Default OFF: the local jsonl ledger
# is always written; Langfuse is an opt-in parallel sink. Documented in
# CONFIGURATION_REFERENCE.md §9.
ENV_LANGFUSE_ENABLE = "HYPERLOOM_LANGFUSE_ENABLE"

# Langfuse connection credentials (official langfuse SDK variable names).
ENV_LANGFUSE_HOST = "LANGFUSE_HOST"
ENV_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"

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


__all__ = [
    "ENV_LANGFUSE_ENABLE",
    "ENV_LANGFUSE_HOST",
    "ENV_LANGFUSE_PUBLIC_KEY",
    "ENV_LANGFUSE_SECRET_KEY",
    "env_flag",
    "langfuse_credentials",
    "langfuse_credentials_complete",
    "langfuse_live_enabled",
]
