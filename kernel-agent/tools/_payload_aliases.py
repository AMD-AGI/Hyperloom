# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Sub-agent-local copy of Hyperloom's payload-rename compat shim.

Mirrors :mod:`inference_optimizer.compat.payload_aliases`; remove in lockstep with it.
"""

from __future__ import annotations

import warnings
from typing import Any


CANONICAL_KEY: str = "extra_server_args"
LEGACY_KEY: str = "extra_sglang_args"

_DEPRECATION_MESSAGE: str = (
    f"kernel-agent received an envelope with the legacy payload key "
    f"{LEGACY_KEY!r}; the canonical name is {CANONICAL_KEY!r}. The "
    f"legacy alias is read-only and will be removed in the next "
    f"Hyperloom release."
)


def _coerce_str(value: Any) -> str:
    """Coerce an arbitrary payload value to a string.

    Args:
        value (Any): The value pulled from the payload dict.

    Returns:
        str: ``value`` unchanged when it is already a string, an empty
            string when it is None, otherwise ``str(value)``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def read_extra_server_args(payload: dict, *, default: str = "") -> str:
    """Read ``extra_server_args`` with a fallback to legacy ``extra_sglang_args``.

    The canonical key wins silently; a legacy-only payload emits one
    ``DeprecationWarning``; if neither key is present, ``default`` is returned.

    Args:
        payload: The payload dict to read from.
        default: Value returned when neither key is present.

    Returns:
        The coerced string value for the server args.
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
