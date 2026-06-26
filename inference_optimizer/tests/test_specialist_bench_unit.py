# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the specialist worktree git helpers.

The legacy ``run_bench`` micro-bench surface has been removed (GPU specialists
now run real serving / benchmark / autotune loops on their own leased cards),
so only the worktree git helpers remain here.
"""

from __future__ import annotations


from inference_optimizer.orchestrator import specialist_bench as sb


# ---- envelopes ----


def test_error_and_ok():
    e = sb._error("bad", x=1)
    assert e == {"ok": False, "reason": "bad", "x": 1}
    assert sb._ok() == {"ok": True}
    assert sb._ok({"a": 2}) == {"ok": True, "a": 2}
