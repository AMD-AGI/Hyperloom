"""Collective-kernel name detection.

Some hot-kernel candidates surfaced by TraceLens are multi-GPU
collectives (`all_reduce` / `all_gather` / `reduce_scatter` /
`broadcast` / etc.) but TraceLens occasionally fails to flag
``is_multigpu=True`` or ``num_gpus_recommended >= 2`` for them — the r24
``custom_allreduce`` regression hit this exact gap. When the upstream
flag is missing, GEAK gets dispatched against a kernel that requires
``torchrun --nproc>=2``, the GEAK sub-agent framework spawns nested
``ray.remote(num_gpus=1)`` for patch validation, and every test inside
fails with ``HIP error: invalid device ordinal``. The kernel then
returns PARTIAL on every attempt (no measurable speedup) and the
orchestrator never retires it.

This module provides a name-pattern fallback. False positives only cost
us GEAK on a single-GPU op; claude / codex still run, so the trade-off
is asymmetric in our favour.
"""

from __future__ import annotations

import re

# Canonical collective op tokens. Each pattern checks for the full verb
# (we reject substring matches of just "reduce", which would overmatch
# unrelated kernels like ``reduce_sum`` / ``reduce_max``). Patterns are
# applied to a normalised lowercase form of the kernel name where
# camelCase / hyphens / dots are folded to underscores, so all of
# ``custom_allreduce``, ``CustomAllReduce``, ``rccl.AllGather``,
# ``triton-all-to-all-fwd`` resolve to the same token shape.
_COLLECTIVE_TOKEN_PATTERNS = [
    re.compile(r"(?:^|_)all_?reduce(?:_|$)"),
    re.compile(r"(?:^|_)all_?gather(?:_|$)"),
    re.compile(r"(?:^|_)reduce_?scatter(?:_|$)"),
    re.compile(r"(?:^|_)all_?to_?all(?:_|$)"),
    re.compile(r"(?:^|_)broadcast(?:_|$)"),
    # Vendor collective libs: anything starting with nccl_ / rccl_ /
    # ncclx_ is a comms primitive (NCCL/RCCL only ship collectives +
    # send/recv). GEAK can't handle either class.
    re.compile(r"(?:^|_)n?cc?lx?(?:_|$)"),
]

_NORMALISE_DELIMS = re.compile(r"[\W]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalise_kernel_name(name: str) -> str:
    """Lowercase + underscore-delimit *name* so substring tests are stable.

    Folds camelCase boundaries, hyphens, dots, and other non-word
    characters to single underscores, then strips leading/trailing
    underscores and lowercases the result.

    Examples:
      ``CustomAllReduce``  → ``custom_all_reduce``
      ``rccl.AllGather``   → ``rccl_all_gather``
      ``triton-all-to-all`` → ``triton_all_to_all``

    Args:
        name (str): Raw kernel name in any casing/delimiter style.

    Returns:
        str: The normalised lowercase, underscore-delimited form, or
            an empty string when *name* is falsy.
    """
    if not name:
        return ""
    s = _CAMEL_BOUNDARY.sub("_", str(name))
    s = _NORMALISE_DELIMS.sub("_", s)
    s = s.strip("_").lower()
    return s


def kernel_name_implies_multigpu(name: str) -> bool:
    """Return True iff *name* matches a known collective-op pattern.

    Intended as a fallback for cases where TraceLens did not surface
    ``is_multigpu=True`` / ``num_gpus_recommended >= 2``. The check is
    leaf-name only — it does not chase dispatch wrappers.

    Args:
        name (str): Kernel name to test. Normalised internally before
            pattern matching.

    Returns:
        bool: True if the normalised name matches a collective-op token
            (all_reduce / all_gather / reduce_scatter / all_to_all /
            broadcast / NCCL-RCCL prefix); False otherwise.
    """
    norm = _normalise_kernel_name(name)
    if not norm:
        return False
    for pat in _COLLECTIVE_TOKEN_PATTERNS:
        if pat.search(norm):
            return True
    return False


__all__ = ["kernel_name_implies_multigpu", "_normalise_kernel_name"]
