# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Sub-agent-local copy of Hyperloom's payload-rename compat shim.

``extra_sglang_args`` was renamed to ``extra_server_args``; reading either
key keeps ``repeated_payload`` signal hashing stable across legacy events.
"""

from __future__ import annotations

import warnings
from typing import Any


CANONICAL_KEY: str = "extra_server_args"
LEGACY_KEY: str = "extra_sglang_args"

_DEPRECATION_MESSAGE: str = (
    f"robustness-agent observed an envelope with the legacy payload "
    f"key {LEGACY_KEY!r}; the canonical name is {CANONICAL_KEY!r}. "
    f"The legacy alias is read-only and will be removed in the next "
    f"Hyperloom release."
)


def _coerce_str(value: Any) -> str:
    """Coerce an arbitrary value to a string.

    Args:
        value (Any): The value to coerce.

    Returns:
        str: ``""`` for ``None``, the value unchanged when already a
        string, otherwise ``str(value)``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def read_extra_server_args(payload: dict, *, default: str = "") -> str:
    """Read ``extra_server_args`` from a payload dict with a one-release
    read-only fallback to the legacy ``extra_sglang_args`` key.

    Emits a ``DeprecationWarning`` when only the legacy key is present.

    Args:
        payload (dict): The payload dict to read from.
        default (str): Value returned when neither key is present.

    Returns:
        str: The canonical value, the legacy value, or ``default``.
    """
    if CANONICAL_KEY in payload:
        return _coerce_str(payload[CANONICAL_KEY])
    if LEGACY_KEY in payload:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        return _coerce_str(payload[LEGACY_KEY])
    return default


__all__ = [
    "CANONICAL_KEY",
    "LEGACY_KEY",
    "read_extra_server_args",
]
