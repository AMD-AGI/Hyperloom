"""Shared safe-JSON helpers with one precise swallowed-exception set.

tree-reform.MD §7/P2.1: the implementation moved to
:mod:`hyperloom.common.jsonio`. This module re-exports it so existing
``from ._json_io import ...`` call sites keep working during the migration
window. TODO(tree-reform): update importers to ``hyperloom.common.jsonio`` and
drop this shim (P2.7).
"""

from __future__ import annotations

from hyperloom.common.jsonio import extract_first_json_with_key, read_json

__all__ = ["read_json", "extract_first_json_with_key"]
