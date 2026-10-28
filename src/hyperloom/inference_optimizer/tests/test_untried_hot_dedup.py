# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the k001/k002 identity dedup in untried_hot_reusable_kernels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel import kernel_decisions as kd
from hyperloom.orchestrator.state import kernel_decision_settings as kds


@pytest.fixture(autouse=True)
def _loose_gate(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "1.0")
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_GATE_TOP_N", "10")


def _state(hot, *, integrate_attempts=None, pending=None, controller=None):
    return SimpleNamespace(
        last_trace_analyze={"hot_kernels_top15": hot, "task_groups": []},
        optimization_stack=[],
        kernel_integrate_attempts=dict(integrate_attempts or {}),
        pending_kernel_integrations=dict(pending or {}),
        kernel_rewrite_controller_result=dict(controller or {}),
    )


def _hot(kid, *, name, src="model.py", gpu_pct=47.8):
    return {
        "kernel_id": kid,
        "name": name,
        "source_file": src,
        "gpu_pct": gpu_pct,
        "reusable_native_kernel": True,
    }


def _rejected_integrate(kid, *, src="model.py"):
    return {
        f"{kid}|/tmp/{kid}.patch|": {
            "kernel_id": kid,
            "task_group_key": "",
            "target_file": src,
            "patch_path": f"/tmp/{kid}.patch",
            "attempt_count": 1,
            "rejected": {"reason": "revert_decision"},
        }
    }


def test_twin_ids_collapse_when_one_was_gated():
    # k001 and k002 are the SAME kernel (identical src/name/gpu_pct). k001's patch was gated end to end and
    # rejected; k002 must NOT resurface as untried.
    hot = [_hot("k001", name="gemm_a8w8"), _hot("k002", name="gemm_a8w8")]
    state = _state(hot, integrate_attempts=_rejected_integrate("k001"))

    untried = kd.untried_hot_reusable_kernels(state)
    assert untried == []


def test_distinct_kernels_are_not_collapsed():
    # Different name => different identity => NOT the same kernel.
    hot = [_hot("k001", name="gemm_a8w8"), _hot("k002", name="attention_fwd")]
    state = _state(hot, integrate_attempts=_rejected_integrate("k001"))

    untried = kd.untried_hot_reusable_kernels(state)
    assert untried == ["k002"]


def test_an_attempt_on_another_source_does_not_retire_a_reassigned_ordinal():
    # Trace ordinals are reassigned on every profile: an integrate verdict for k001 on another file is not
    # evidence about the kernel k001 names now.
    hot = [_hot("k001", name="gemm_a8w8")]
    state = _state(hot, integrate_attempts=_rejected_integrate("k001", src="other.py"))

    untried = kd.untried_hot_reusable_kernels(state)
    assert untried == ["k001"]


def test_a_queued_patch_counts_as_tried():
    hot = [_hot("k001", name="gemm_a8w8")]
    pending = {
        "kernel-integration:abc": {
            "integration_id": "kernel-integration:abc",
            "kernel_id": "k001",
            "task_group_key": "",
            "source_file": "model.py",
            "status": "pending",
        }
    }
    state = _state(hot, pending=pending)

    assert kd.untried_hot_reusable_kernels(state) == []


def test_a_forge_nomination_retires_its_kernel_by_symbol():
    # A nominated sibling patch is keyed by the kernel symbol Forge rewrote, never by the trace ordinal.
    hot = [_hot("k001", name="FusedMoeKernel")]
    pending = {
        "kernel-integration:def": {
            "integration_id": "kernel-integration:def",
            "task_key": "forge_rewrite:fused_moe_kernel",
            "kernel_id": "fused_moe_kernel",
            "task_group_key": "",
            "source_file": "/repo/fused_moe.py",
            "status": "rejected",
        }
    }
    state = _state(hot, pending=pending)

    assert kd.untried_hot_reusable_kernels(state) == []


def test_an_operator_the_rewrite_controller_published_counts_as_tried():
    # The controller's integration reverted the patch, so nothing reached the stack, but the operator was tried.
    hot = [_hot("k001", name="_fwd_grouped_kernel_stage1")]
    controller = {
        "macro_cycle": 0,
        "status": "completed",
        "integration": {
            "results": [
                {
                    "operator_id": "kernel:forge-loop:fwd_grouped_kernel_stage1:sglang:0.5.17:triton:mi355x",
                    "status": "reverted_no_gain",
                }
            ]
        },
    }
    state = _state(hot, controller=controller)

    assert kd.untried_hot_reusable_kernels(state) == []


def test_no_attempts_all_untried_but_deduped():
    # No attempts at all: both twins collapse to a single untried entry (one kernel, one attempt owed), not two.
    hot = [_hot("k001", name="gemm_a8w8"), _hot("k002", name="gemm_a8w8")]
    state = _state(hot)

    untried = kd.untried_hot_reusable_kernels(state)
    assert len(untried) == 1
    assert untried[0] in {"k001", "k002"}


def test_forge_loop_integration_retires_its_hot_kernel_by_operator_name():
    # Reproduces a real session: forge-loop integrated "_fwd_grouped_kernel_stage1" via
    # kernel_rewrite_controller, landing an optimization_stack entry keyed by the long-form recipe id
    # ("kernel:forge-loop:<operator>:<framework>:<framework_version>:<backend>:<gpu>"), never sharing a
    # kernel_id/source_file with the roofline trace's k001.
    # Without operator-name reconciliation, k001 falsely resurfaces as untried despite being the exact
    # kernel that landed the session's validated gain.
    hot = [
        _hot(
            "k001",
            name="_fwd_grouped_kernel_stage1",
            src="/sgl-workspace/aiter/op_tests/triton_tests/utils/mla_decode_ref.py",
        )
    ]
    state = _state(hot)
    state.optimization_stack = [
        {
            "action": "integrate",
            "kernel_id": "kernel:forge-loop:fwd_grouped_kernel_stage1:sglang:0.5.17:triton:mi355x",
            "target_file": "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/decode_attention.py",
            "ts": "2026-09-21T18:30:06.535892+00:00",
        }
    ]

    untried = kd.untried_hot_reusable_kernels(state)
    assert untried == []


def test_hot_kernel_floor_falls_back_on_an_unparseable_override(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT", "not-a-number")
    assert kds.resolve_hot_kernel_min_gpu_pct() == kds._DEFAULT_HOT_KERNEL_MIN_GPU_PCT
