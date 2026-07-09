# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Environment-variable readers (canonical ``env_*``).

Project-standard boolean token vocabulary (``1/true/yes/on`` → ``True``,
case-insensitive) plus int/str/float readers with safe fallbacks. Stdlib-only
so any package may depend on it without an import cycle.

Divergent readers intentionally NOT delegated here (kept local by design):

* ``ci/optimize_submit._env_truthy`` — additionally accepts ``y``.
* ``orchestrator/roofline_ceiling._env_int`` — reads from a *dict* mapping, not
  from ``os.environ``.
* ``orchestrator/trace/trace_env.env_flag`` — also treats ``0/false/no/off`` as
  an explicit ``False`` vocabulary (superset semantics).
"""

from __future__ import annotations

import os

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean env var.

    Returns ``default`` when the variable is unset; otherwise ``True`` when the
    value (stripped, lower-cased) is one of ``1/true/yes/on`` and ``False`` for
    any other set value.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_TOKENS


def env_int(name: str, default: int = 0) -> int:
    """Read an integer env var.

    Args:
        name: Environment variable name.
        default: Value returned when unset, blank, or not a valid integer.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float = 0.0) -> float:
    """Read a float env var.

    Args:
        name: Environment variable name.
        default: Value returned when unset, blank, or not a valid float.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(name: str, default: str = "") -> str:
    """Read a stripped string env var.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The stripped value, or ``default`` when unset.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


__all__ = ["env_bool", "env_int", "env_float", "env_str"]
