# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Environment-variable readers (canonical ``env_*``).

Project-standard boolean token vocabulary (``1/true/yes/on`` → ``True``,
case-insensitive) plus int/str/float readers with safe fallbacks. Stdlib-only
so any package may depend on it without an import cycle.

Divergent readers intentionally NOT delegated here (kept local by design):

* ``orchestrator/kernel/roofline_ceiling._env_int`` — reads from a *dict*
  mapping, not from ``os.environ``.
* ``orchestrator/trace/trace_env.env_flag`` — an unrecognised or empty set
  value falls back to ``default`` rather than being classified ``False``.
"""

from __future__ import annotations

import os

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
# Canonical "off" vocabulary. The empty string is an explicit false token (a
# blank/whitespace-only value is never "on"); an unset value (``None``) or an
# unrecognised token falls back to the caller's ``default`` instead.
_FALSE_TOKENS = frozenset({"", "0", "false", "no", "off"})


def is_truthy(value: object, *, default: bool = False) -> bool:
    """Interpret an already-read *value* as a boolean flag.

    Unlike :func:`env_bool` (which reads ``os.environ`` by name), this takes a
    value directly — a task-param, a dict entry, or a pre-read env string —
    reusing the shared token vocabulary.

    Classification (case-insensitive, whitespace-stripped):

    * ``bool`` values return themselves.
    * ``1``/``true``/``yes``/``on`` → ``True``.
    * ``0``/``false``/``no``/``off`` and the empty string → ``False``.
    * ``None`` (unset) or any *unrecognised* token → ``default``.

    ``default`` is what distinguishes the two legacy idioms this replaces: a
    strict "affirmative-token only" reader passes ``default=False`` (unknown →
    ``False``), while an "off-set" reader — truthy unless explicitly one of the
    off tokens — passes ``default=True`` (unknown → ``True``).

    Args:
        value: The value to interpret (``bool``, ``str``, ``int``, ``None`` …).
        default: Returned when *value* is ``None`` or an unrecognised token.

    Returns:
        The value's boolean interpretation.
    """
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


def forge_explicitly_enabled() -> bool:
    """Whether per-kernel forge is opted in.

    Returns:
        True for an exact ``KERNEL_OPT_BACKEND_ORDER=forge``; every other
        value leaves GEAK owning the whole kernel phase.
    """
    return env_str("KERNEL_OPT_BACKEND_ORDER").lower() == "forge"


def nomination_auto_enabled() -> bool:
    """Whether forge self-nomination (``--auto``) drives the KERNEL rewrite lane.

    A single opt-in env, off by default, mirroring
    :func:`forge_explicitly_enabled`: it reads one variable, touches no
    SharedState schema, and when unset the KERNEL phase serializes byte-for-byte
    as it did before the nomination contract landed. Set to route ``auto=true``
    through forge's own kernel selection instead of the Hyperloom selector.

    Returns:
        True for any truthy ``HYPERLOOM_FORGE_NOMINATION_AUTO``.
    """
    return env_bool("HYPERLOOM_FORGE_NOMINATION_AUTO", default=False)


__all__ = [
    "is_truthy",
    "env_bool",
    "env_int",
    "env_float",
    "env_str",
    "forge_explicitly_enabled",
    "nomination_auto_enabled",
]
