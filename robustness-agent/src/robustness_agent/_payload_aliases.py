# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Sub-agent-local copy of Hyperloom's payload-rename compat shim.

tree-reform.MD §7/P2.1: the canonical implementation now lives in
:mod:`hyperloom.common.payload_aliases`; this module re-exports it (previously a
hand-maintained duplicate). Reading either key keeps ``repeated_payload`` signal
hashing stable across legacy events; behaviour is pinned by
``test_payload_aliases_shim.py``. TODO(tree-reform): import
``hyperloom.common.payload_aliases`` directly and drop this shim (P2.7).
"""

from __future__ import annotations

from hyperloom.common.payload_aliases import (
    CANONICAL_KEY,
    LEGACY_KEY,
    read_extra_server_args,
)

__all__ = [
    "CANONICAL_KEY",
    "LEGACY_KEY",
    "read_extra_server_args",
]
