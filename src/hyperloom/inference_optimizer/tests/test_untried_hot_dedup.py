# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the k001/k002 identity dedup in untried_hot_reusable_kernels.

When a trace carries no ``task_groups`` metadata, every hot-kernel row degenerates
to its own synthetic id. One physical CK GEMM then shows up under several ids
(k001/k002) with identical (source_file, name, gpu_pct). Before the fix, rejecting
the id that was actually attempted (k001) left its twin (k002) forever "untried",
so kernel_work_pending() never went False and KERNEL_AGENT spun until the wall
cap. The identity dedup collapses the twins so a single rejection retires both.
"""
from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.kernel import _kernel_decisions as kd


def _state(hot, *, rejected=(), attempts=None):
    return SimpleNamespace(
        last_trace_analyze={"hot_kernels_top15": hot, "task_groups": []},
        optimization_stack=[],
        rejected_kernel_ids=list(rejected),
        kernel_opt_attempts=attempts or {},
        kernel_opt_task_attempts=None,  # populated by _ensure_kernel_task_state
    )


def _hot(kid, *, name, src="model.py", gpu_pct=47.8):
    return {
        "kernel_id": kid,
        "name": name,
        "source_file": src,
        "gpu_pct": gpu_pct,
        "reusable_native_kernel": True,
    }


def test_twin_ids_collapse_when_one_is_rejected():
    # k001 and k002 are the SAME kernel (identical src/name/gpu_pct). k001 was
    # attempted and rejected; k002 must NOT resurface as untried.
    hot = [_hot("k001", name="gemm_a8w8"), _hot("k002", name="gemm_a8w8")]
    attempts = {
        "k001": {
            "kernel_id": "k001",
            "current_kernel_id": "k001",
            "last_source_file": "model.py",
            "task_group_key": "",
            "rejected_reason": "slower than baseline",
        }
    }
    state = _state(hot, rejected=["k001"], attempts=attempts)

    untried = kd.untried_hot_reusable_kernels(state, min_gpu_pct=1.0, top_n=10)
    assert untried == []


def test_distinct_kernels_are_not_collapsed():
    # Different name => different identity => NOT the same kernel. Rejecting k001
    # must leave the genuinely-distinct k002 still owing an attempt.
    hot = [_hot("k001", name="gemm_a8w8"), _hot("k002", name="attention_fwd")]
    attempts = {
        "k001": {
            "kernel_id": "k001",
            "current_kernel_id": "k001",
            "last_source_file": "model.py",
            "task_group_key": "",
            "rejected_reason": "slower than baseline",
        }
    }
    state = _state(hot, rejected=["k001"], attempts=attempts)

    untried = kd.untried_hot_reusable_kernels(state, min_gpu_pct=1.0, top_n=10)
    assert untried == ["k002"]


def test_no_attempts_all_untried_but_deduped():
    # No attempts at all: both twins collapse to a single untried entry (one
    # kernel, one attempt owed), not two.
    hot = [_hot("k001", name="gemm_a8w8"), _hot("k002", name="gemm_a8w8")]
    state = _state(hot)

    untried = kd.untried_hot_reusable_kernels(state, min_gpu_pct=1.0, top_n=10)
    assert len(untried) == 1
    assert untried[0] in {"k001", "k002"}
