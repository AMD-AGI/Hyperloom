# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Canonical UTC ISO-8601 timestamp helpers (``_time``).

Relocated from ``hyperloom.orchestrator._time`` (P2.1); that re-export shim
was removed in P2.7 once all callers were updated to import directly from
here. Replaces the ~14 ``_now_iso`` copies plus the ``_utc_now_compact`` /
``_iso_z`` / ISO-parse variants. Stdlib-only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso(timespec: str = "microseconds", *, z_suffix: bool = False) -> str:
    """Current UTC time, ISO-8601. *timespec* → isoformat; *z_suffix* renders ``Z``."""
    ts = datetime.now(timezone.utc).isoformat(timespec=timespec)
    if z_suffix:
        ts = ts.replace("+00:00", "Z")
    return ts


def utc_now_compact() -> str:
    """Current UTC time as a compact ``YYYYMMDDTHHMMSSZ`` id timestamp.

    Byte-identical to the legacy ``session/manifest._utc_now_compact`` used to
    build session ids.

    Returns:
        The current UTC time formatted as ``%Y%m%dT%H%M%SZ``.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_z(ts: Any) -> str:
    """Normalise any ISO-8601 timestamp to canonical second-precision ``...Z`` UTC.

    Naive timestamps are assumed UTC; aware ones are converted to UTC. Returns
    ``""`` for empty input, or the original string when it cannot be parsed.

    Args:
        ts: An ISO-8601 timestamp value (any suffix), or ``None``.

    Returns:
        The canonical ``...Z`` UTC string, ``""`` for empty input, or the
        original string when unparseable.
    """
    if ts is None:
        return ""
    s = str(ts).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["now_iso", "utc_now_compact", "iso_z"]
