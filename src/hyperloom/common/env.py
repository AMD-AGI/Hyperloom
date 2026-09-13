# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Environment-variable readers (canonical ``env_*``)."""

from __future__ import annotations

import os

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
# Canonical "off" vocabulary.
_FALSE_TOKENS = frozenset({"", "0", "false", "no", "off"})


def is_truthy(value: object, *, default: bool = False) -> bool:
    """Interpret an already-read *value* as a boolean flag."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return default


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean env var."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_TOKENS


def env_int(name: str, default: int = 0) -> int:
    """Read an integer env var."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float = 0.0) -> float:
    """Read a float env var."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(name: str, default: str = "") -> str:
    """Read a stripped string env var."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def forge_explicitly_enabled() -> bool:
    """Whether per-kernel forge is opted in."""
    return env_str("KERNEL_OPT_BACKEND_ORDER").lower() == "forge"


__all__ = [
    "is_truthy",
    "env_bool",
    "env_int",
    "env_float",
    "env_str",
    "forge_explicitly_enabled",
]
