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
        kernel_opt_task_attempts=dict(attempts or {}),
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


def test_an_op_fanout_sibling_retires_with_its_representative():
    # Op fanout: two different operations over one file. The batch filter
    # dispatches the stronger one and reports the other as
    # ``opfanout_merged_into``, which means no backend ever saw it and so writes
    # no ledger row of its own. The representative's row records the merge, and
    # that is what retires the sibling: without it k002 owes an attempt no
    # dispatch will ever make, kernel_work_pending() never goes False, and
    # KERNEL redispatches the entry batch every tick while ``report`` stays
    # forbidden -- the same spin the identity dedup closes, reached through the
    # dispatcher instead of through synthetic ids. The identity key cannot see
    # it: op fanout is by definition several *different* operations, so name and
    # gpu_pct both differ.
    hot = [
        _hot("k001", name="gemm_a8w8", gpu_pct=9.0),
        _hot("k002", name="attention_fwd", gpu_pct=7.0),
    ]
    attempts = {
        "k001": {
            "kernel_id": "k001",
            "current_kernel_id": "k001",
            "last_source_file": "model.py",
            "task_group_key": "",
            "attempts": 1,
            "opfanout_collapsed_ids": ["k001", "k002"],
        }
    }
    state = _state(hot, attempts=attempts)

    untried = kd.untried_hot_reusable_kernels(state, min_gpu_pct=1.0, top_n=10)
    assert untried == []


def test_a_sibling_the_representative_did_not_cover_still_owes_an_attempt():
    # The retirement follows the recorded merge, not the shared file. Two ops in
    # one file that the dispatcher never merged are two units of work, which is
    # what several real-session regressions in test_shared_state_kernel_opt.py
    # depend on.
    hot = [
        _hot("k001", name="gemm_a8w8", gpu_pct=9.0),
        _hot("k002", name="attention_fwd", gpu_pct=7.0),
    ]
    attempts = {
        "k001": {
            "kernel_id": "k001",
            "current_kernel_id": "k001",
            "last_source_file": "model.py",
            "task_group_key": "",
            "attempts": 1,
        }
    }
    state = _state(hot, attempts=attempts)

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


def test_geometry_only_shape_excluded_from_untried():
    # A reusable kernel with a resolved source but shape_dispatchable=False
    # (geometry-only provenance) fails the kernel-opt gate, so it must never
    # enter the untried queue and spin KERNEL_AGENT.
    geom = _hot("k001", name="combine_kernel")
    geom["shape_dispatchable"] = False
    ok = _hot("k002", name="gemm_a8w8")
    ok["shape_dispatchable"] = True
    state = _state([geom, ok])

    untried = kd.untried_hot_reusable_kernels(state, min_gpu_pct=1.0, top_n=10)
    assert untried == ["k002"]


def test_missing_shape_dispatchable_field_stays_untried():
    # TraceLens path never emits shape_dispatchable; absent field must be treated
    # as dispatchable so the main path is not regressed.
    hot = [_hot("k001", name="gemm_a8w8")]  # no shape_dispatchable key
    state = _state(hot)

    untried = kd.untried_hot_reusable_kernels(state, min_gpu_pct=1.0, top_n=10)
    assert untried == ["k001"]
