"""Canonical UTC ISO-8601 timestamp helper (replaces ~14 ``_now_iso`` copies)."""

from __future__ import annotations

from datetime import datetime, timezone


def now_iso(timespec: str = "microseconds", *, z_suffix: bool = False) -> str:
    """Current UTC time, ISO-8601. *timespec* → isoformat; *z_suffix* renders ``Z``."""
    ts = datetime.now(timezone.utc).isoformat(timespec=timespec)
    if z_suffix:
        ts = ts.replace("+00:00", "Z")
    return ts


__all__ = ["now_iso"]
