# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Sub-agent-local copy of Hyperloom's payload-rename compat shim.

Mirrors :mod:`inference_optimizer.compat.payload_aliases` from the
main Hyperloom orchestrator. The robustness-agent is a separately
packaged Python project, so the shim is duplicated here rather than
imported across package boundaries — this matches the existing
isolation pattern (``framework_agent.repo_map``).

The payload-surface field ``extra_sglang_args`` was renamed to
``extra_server_args``. The ``repeated_payload`` signal hashes
per-family payload projections to detect same-fingerprint retries; if
a legacy event arrives mid-streak the helper lets the signal still
produce a stable hash by reading either key transparently.

Removal target: in lockstep with Hyperloom's own compat helper.
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
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def read_extra_server_args(payload: dict, *, default: str = "") -> str:
    """Read ``extra_server_args`` from a payload dict with a one-release
    read-only fallback to the legacy ``extra_sglang_args`` key."""
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
