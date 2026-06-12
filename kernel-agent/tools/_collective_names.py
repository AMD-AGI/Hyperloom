# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Collective-kernel name detection.

Name-pattern fallback for multi-GPU collectives TraceLens missed (r24
``custom_allreduce`` regression); false positives are cheap so we bias toward them.
"""

from __future__ import annotations

import re

# Canonical collective op tokens (full-verb match, rejecting bare "reduce"),
# applied to the normalised lowercase kernel name.
_COLLECTIVE_TOKEN_PATTERNS = [
    re.compile(r"(?:^|_)all_?reduce(?:_|$)"),
    re.compile(r"(?:^|_)all_?gather(?:_|$)"),
    re.compile(r"(?:^|_)reduce_?scatter(?:_|$)"),
    re.compile(r"(?:^|_)all_?to_?all(?:_|$)"),
    re.compile(r"(?:^|_)broadcast(?:_|$)"),
    # Vendor comms libs (nccl_/rccl_/ncclx_) only ship collectives + send/recv.
    re.compile(r"(?:^|_)n?cc?lx?(?:_|$)"),
]

_NORMALISE_DELIMS = re.compile(r"[\W]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalise_kernel_name(name: str) -> str:
    """Lowercase + underscore-delimit *name* so substring tests are stable."""
    if not name:
        return ""
    s = _CAMEL_BOUNDARY.sub("_", str(name))
    s = _NORMALISE_DELIMS.sub("_", s)
    s = s.strip("_").lower()
    return s


def kernel_name_implies_multigpu(name: str) -> bool:
    """Return True iff *name* matches a known collective-op pattern (leaf-name only)."""
    norm = _normalise_kernel_name(name)
    if not norm:
        return False
    for pat in _COLLECTIVE_TOKEN_PATTERNS:
        if pat.search(norm):
            return True
    return False


__all__ = ["kernel_name_implies_multigpu", "_normalise_kernel_name"]
