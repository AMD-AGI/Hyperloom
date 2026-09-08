# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How wide, how cheap and how long a KB warm start is allowed to search.

Two independent warm starts read these: the forge loop's
(:mod:`kernelforge.knowledge.experience_integration`) and the FlyDSL rewrite
path's (:mod:`kernelforge.rewrite_by_flydsl.kb`). They address different
identities -- the ``producer`` dimension is ``forge-loop`` for one and ``flydsl``
for the other, so their records never mix -- but the search they perform is the
same shape, and two copies of these numbers would drift.

Each value is a module default an environment variable may override, which is
how the rest of the knowledge package is configured.
"""

from __future__ import annotations

import os

#: How many best-ranked prior solutions a warm start reads. The store caps a
#: ranked page at 100.
DEFAULT_TOP_K = 10

#: The lowest claimed speedup worth spending a trial on. A record claiming less
#: than this is a port that lost badly to the source baseline; measuring it costs
#: a compile, a correctness suite and a benchmark, and adopting it would start
#: the run from a kernel several times slower than the implementation it is
#: supposed to replace. Filtered candidates are not offered as prompt reference
#: material either: a catastrophic port teaches the author nothing that pays for
#: the tokens.
#:
#: The claim is what is filtered on, and a claim is not comparable across tasks:
#: it was computed over whatever cases the producing task scored. So this is a
#: coarse screen for catastrophe, not a ranking. Choosing between the survivors
#: is what measurement is for.
DEFAULT_MIN_CLAIMED_SPEEDUP = 0.3

#: Wall-clock ceiling on the whole candidate search, in seconds. Each candidate
#: costs a compile plus a correctness suite plus a benchmark, which on the
#: heaviest kernels measured here is around 15 minutes on a cold cache, so an
#: unbounded search over a well-populated identity can spend hours before the
#: agent has made its first edit. On expiry the search stops and adopts the best
#: candidate it has already measured.
DEFAULT_BUDGET_SEC = 1800


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def top_k() -> int:
    """How many ranked candidates to read."""
    return _positive_int("FORGE_KB_WARMSTART_TOP_K", DEFAULT_TOP_K)


def min_claimed_speedup() -> float:
    """The claimed speedup below which a candidate is not tried at all."""
    return _positive_float("FORGE_KB_WARMSTART_MIN_SPEEDUP", DEFAULT_MIN_CLAIMED_SPEEDUP)


def budget_sec() -> float:
    """Wall-clock ceiling on the candidate search."""
    return _positive_float("FORGE_KB_WARMSTART_BUDGET_SEC", DEFAULT_BUDGET_SEC)


def below_floor(claimed_speedup: float | None) -> bool:
    """Whether a candidate's claim puts it under the floor.

    An unrecorded claim is not under the floor: nothing was claimed, so nothing
    is contradicted, and the candidate is worth a measurement on its own terms.
    """
    if claimed_speedup is None:
        return False
    return float(claimed_speedup) < min_claimed_speedup()
