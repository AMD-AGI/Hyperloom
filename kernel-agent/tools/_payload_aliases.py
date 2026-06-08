# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Sub-agent-local copy of Hyperloom's payload-rename compat shim.

Mirrors :mod:`inference_optimizer.compat.payload_aliases` from the
main Hyperloom orchestrator. Sub-agents are intentionally standalone
Python packages (see ``framework_agent.repo_map`` for the same
isolation pattern), so this shim is duplicated here rather than
imported across package boundaries.

The payload-surface field ``extra_sglang_args`` was renamed to
``extra_server_args``. The kernel-agent runtime is invoked over a
JSON envelope by the Coordinator; this helper lets the runtime
tolerate envelopes still carrying the legacy name (e.g. from a
predating Hyperloom release or a saved KB record being replayed)
while emitting only the canonical name on response.

Removal target: in lockstep with Hyperloom's own compat helper.
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
    """Read ``extra_server_args`` from a payload dict with a one-release
    read-only fallback to the legacy ``extra_sglang_args`` key.

    Same contract as Hyperloom's compat helper:

    1. Canonical key present -> return its coerced value, no warning.
    2. Legacy key present (canonical absent) -> emit one
       ``DeprecationWarning`` and return the coerced legacy value.
    3. Neither key present -> return ``default``.

    Args:
        payload (dict): The decoded JSON envelope passed to the runtime.
        default (str): Value returned when neither key is present.
            Keyword-only. Defaults to an empty string.

    Returns:
        str: The coerced value of the canonical key, the legacy key, or
            ``default``, per the resolution order above.
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
