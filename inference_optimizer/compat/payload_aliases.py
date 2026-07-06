# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-only deprecation aliases for renamed payload fields.

tree-reform.MD §7/P2.1: the canonical implementation moved to
:mod:`hyperloom.common.payload_aliases`. This module re-exports it so existing
``inference_optimizer.compat.payload_aliases`` call sites keep working during
the migration window (``extra_sglang_args`` → ``extra_server_args``). The names
are re-exported (not re-wrapped) so ``read_extra_server_args``'s ``stacklevel=3``
still reports the caller's caller unchanged.

TODO(tree-reform): update importers to ``hyperloom.common.payload_aliases`` and
drop this shim once the ``extra_sglang_args`` window closes (P2.7).
"""

from __future__ import annotations

from hyperloom.common.payload_aliases import (
    CANONICAL_KEY,
    LEGACY_KEY,
    migrate_legacy_key_in_place,
    read_extra_server_args,
)

__all__ = [
    "CANONICAL_KEY",
    "LEGACY_KEY",
    "read_extra_server_args",
    "migrate_legacy_key_in_place",
]
