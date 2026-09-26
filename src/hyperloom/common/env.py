# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Environment-variable readers (canonical ``env_*``).

An environment value the reader cannot interpret raises :class:`EnvValueError`
rather than falling back to the caller's default. A typo in a boolean pin
(``ture``) or a unit left on a number (``30s``) used to be indistinguishable
from leaving the variable unset, so a run could silently execute the opposite
configuration from the one the operator wrote and report success. The default
still answers the one question it can answer honestly -- "the operator said
nothing" -- and an unreadable value is a configuration error at the boundary
that read it.

Two readers are deliberately outside that contract. :func:`is_truthy`
interprets a value somebody else already read -- an LLM-authored Intent
parameter, an operator's grid ``extra_envs`` entry -- where an unrecognised
token is data to fall back on, not a configuration error that should abort the
run. :func:`env_flag` is the opt-in lenient variant its callers already chose.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
# Canonical "off" vocabulary. The empty string is an explicit off token: a
# variable blanked on the command line is not the same as one never set.
_FALSE_TOKENS = frozenset({"", "0", "false", "no", "off"})

_BOOL_VOCABULARY = "1/true/yes/on or 0/false/no/off (or blank for off)"


class EnvValueError(ValueError):
    """A configuration value could not be interpreted by the reader that read it.

    Raised instead of silently returning the caller's default, so a malformed
    pin fails at the boundary that read it rather than at whatever measurement
    later depends on it.
    """


def _parse_bool(value: object) -> bool | None:
    """The shared boolean vocabulary; ``None`` when the token is not in it."""
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def is_truthy(value: object, *, default: bool = False) -> bool:
    """Interpret an already-read *value* as a boolean flag."""
    if value is None:
        return default
    parsed = _parse_bool(value)
    return default if parsed is None else parsed


def env_bool(name: str, default: bool = False, *, env: Mapping[str, str] | None = None) -> bool:
    """Read a boolean env var.

    Args:
        name: Environment variable to read.
        default: Returned when the variable is unset.
        env: Environment to read from; ``os.environ`` when omitted. A grid
            variant builds the environment it is about to run under before that
            environment exists as a process.

    Returns:
        The boolean the variable spells.

    Raises:
        EnvValueError: The variable is set to an unrecognised token.
    """
    raw = (os.environ if env is None else env).get(name)
    if raw is None:
        return default
    parsed = _parse_bool(raw)
    if parsed is None:
        raise EnvValueError(f"{name}={raw!r} is not a boolean; expected {_BOOL_VOCABULARY}")
    return parsed


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var, falling back to *default* for unrecognised values.

    Unlike ``env_bool``, a value that is neither a true-token nor a false-token
    returns *default* rather than raising.
    """
    raw = os.environ.get(name)
    return is_truthy(raw, default=default)


def env_int(name: str, default: int = 0) -> int:
    """Read an integer env var.

    Args:
        name: Environment variable to read.
        default: Returned when the variable is unset or blank.

    Returns:
        The integer the variable spells.

    Raises:
        EnvValueError: The variable is set to a value that is not an integer.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise EnvValueError(f"{name}={raw!r} is not an integer") from exc


def env_float(name: str, default: float = 0.0) -> float:
    """Read a float env var.

    Args:
        name: Environment variable to read.
        default: Returned when the variable is unset or blank.

    Returns:
        The float the variable spells.

    Raises:
        EnvValueError: The variable is set to a value that is not a number.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise EnvValueError(f"{name}={raw!r} is not a number") from exc


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
    "EnvValueError",
    "is_truthy",
    "env_bool",
    "env_flag",
    "env_int",
    "env_float",
    "env_str",
    "forge_explicitly_enabled",
]
