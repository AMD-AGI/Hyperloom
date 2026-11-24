# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a candidate and the tests its parent's value comes from met the shared compiled-code cache in different
states.

A warm cache moves aiter's and Triton's compiles out of the server's boot, which is before anything is measured; a
kernel the server first builds after warm-up, for a shape only the measured window asks for, is the one compile a warm
cache moves out of the measured window. A candidate whose patch gives it a key of its own meets a cold cache once while
its parent was measured warm, so its judgement says so (:attr:`~..contracts.judgement.Judgement.cache_states_differ`)
and a reader can weigh a small gain with that in mind. The verdict is not changed by it.
"""

from __future__ import annotations

from collections.abc import Sequence

from hyperloom.contracts.measurement import TestResult

__all__ = ["NOT_USED", "cache_states_differ"]

#: The state of a kind a test with cache states did not use at all.
NOT_USED = "not_used"


def cache_states_differ(candidates: Sequence[TestResult], parents: Sequence[TestResult]) -> tuple[str, ...]:
    """Each kind whose states differ between ``candidates`` and ``parents``, as ``"<kind>: <candidate states> vs
    <parent states>"``; none when either side has no test that states its cache (a test of a session without the cache,
    or one recorded before the states were kept): what was not stated is not called different."""
    candidate, parent = _states(candidates), _states(parents)
    if not candidate or not parent:
        return ()
    differing = []
    for kind in sorted({kind for states in (*candidate, *parent) for kind in states}):
        mine = sorted({states.get(kind, NOT_USED) for states in candidate})
        theirs = sorted({states.get(kind, NOT_USED) for states in parent})
        if mine != theirs:
            differing.append(f"{kind}: {'/'.join(mine)} vs {'/'.join(theirs)}")
    return tuple(differing)


def _states(results: Sequence[TestResult]) -> list[dict[str, str]]:
    return [{state.kind: state.state for state in result.jit_cache} for result in results if result.jit_cache]
