"""Throughput percentage-gain helpers (one home per edge contract).

tree-reform.MD §7/P2.1: the implementation moved to
:mod:`hyperloom.common.gain_math`. This module re-exports it so existing
``from .gain_math import ...`` call sites keep working during the migration
window. TODO(tree-reform): update importers to ``hyperloom.common.gain_math``
and drop this shim (P2.7).
"""

from __future__ import annotations

from hyperloom.common.gain_math import (
    gain_pct,
    gain_pct_or_zero,
    incremental_gain_pct,
)

__all__ = ["gain_pct", "gain_pct_or_zero", "incremental_gain_pct"]
