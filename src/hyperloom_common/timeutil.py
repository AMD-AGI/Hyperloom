# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical UTC ISO-8601 timestamp helpers (``hyperloom_common.timeutil``). Stdlib-only."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso(timespec: str = "microseconds", *, z_suffix: bool = False) -> str:
    """Current UTC time, ISO-8601. *timespec* → isoformat; *z_suffix* renders ``Z``."""
    ts = datetime.now(timezone.utc).isoformat(timespec=timespec)
    if z_suffix:
        ts = ts.replace("+00:00", "Z")
    return ts


def epoch_iso(epoch_s: float) -> str:
    """Unix seconds as a UTC microsecond ISO-8601 string, the ledger ``ts`` format ``now_iso`` also produces."""
    return datetime.fromtimestamp(epoch_s, timezone.utc).isoformat(timespec="microseconds")


def elapsed_ms(start: float, end: float | None) -> int | None:
    """Whole milliseconds from *start* to *end* (seconds), clamped at zero; ``None`` while *end* is unset."""
    return None if end is None else max(0, int((end - start) * 1000))


def utc_now_compact() -> str:
    """Current UTC time as a compact ``YYYYMMDDTHHMMSSZ`` id timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_z(ts: Any) -> str:
    """Normalise any ISO-8601 timestamp to canonical second-precision ``...Z`` UTC."""
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


def format_exc_brief(exc: "BaseException", limit: "int | None" = None) -> str:
    """Render an exception as ``"TypeName: message"``, optionally truncated."""
    msg = str(exc)
    if limit is not None:
        msg = msg[:limit]
    return f"{type(exc).__name__}: {msg}"


__all__ = ["now_iso", "epoch_iso", "elapsed_ms", "utc_now_compact", "iso_z", "format_exc_brief"]
