"""Canonical UTC ISO-8601 timestamp helper (replaces ~14 ``_now_iso`` copies).

tree-reform.MD §7/P2.1: the implementation moved to
:mod:`hyperloom.common.timeutil`. This module re-exports it so existing
``from ._time import now_iso`` call sites keep working during the migration
window. TODO(tree-reform): update importers to ``hyperloom.common.timeutil``
and drop this shim (P2.7).
"""

from __future__ import annotations

from hyperloom.common.timeutil import now_iso

__all__ = ["now_iso"]
