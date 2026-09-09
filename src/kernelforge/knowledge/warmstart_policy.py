# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How wide, how cheap and how long a KB warm start is allowed to search.

Two independent warm starts read these: the forge loop's and the FlyDSL rewrite path's. They address different
identities, so their records never mix, but the search is the same shape and two copies of these numbers would drift.
"""

from __future__ import annotations

import os

#: How many best-ranked prior solutions a warm start reads. More than one, so a champion that fails to apply still
#: leaves something to fall back to, and so a claim that does not survive measurement can lose to one that does. Every
#: candidate read may be measured, so this also bounds the trials; :data:`DEFAULT_BUDGET_SEC` bounds their wall time.
DEFAULT_TOP_K = 10

#: The lowest claimed speedup worth spending a trial on. Below it a record is a port that lost badly, and it is not
#: offered as prompt reference material either. The claim is not comparable across tasks -- it was computed over
#: whatever cases the producing task scored -- so this screens for catastrophe; measurement ranks the survivors.
DEFAULT_MIN_CLAIMED_SPEEDUP = 0.3

#: Wall-clock ceiling on the whole candidate search, in seconds. One candidate is a compile plus a correctness suite
#: plus a benchmark, around 15 minutes on the heaviest kernel measured here, so an unbounded search over a
#: well-populated identity spends hours before the agent's first edit. On expiry the best already measured is adopted.
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

    An unrecorded claim is not under it: nothing was claimed, so nothing is contradicted.
    """
    if claimed_speedup is None:
        return False
    return float(claimed_speedup) < min_claimed_speedup()
