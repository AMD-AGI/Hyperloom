# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the multi-KEEP integrate queue helpers and the untried-hot gate."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.state.shared_state import SharedState

from .conftest import seed_kernel_keep as _seed_keep


@pytest.fixture
def state() -> SharedState:
    return SharedState()


# Invariant 1: empty kernel_id is a no-op
# Invariant 2: KEEP wins; non-KEEP never overwrites a pending KEEP
# Vendor-playbook KEEPs (e.g. mori dispatch/combine) must never auto-deploy.
# next_pending_keep_kernel_id queue semantics
def test_next_pending_keep_drains_in_micro_speedup_order(state: SharedState):
    """KEEPs on different source_files drain highest-micro-first as the stack fills."""
    _seed_keep(state, "k001", decision="KEEP", micro=2.5, source_file="/p/file_a.py", artifact="/t/a1.py")
    _seed_keep(state, "k009", decision="KEEP", micro=4.13, source_file="/p/file_b.py", artifact="/t/b9.py")
    _seed_keep(state, "k015", decision="KEEP", micro=3.2, source_file="/p/file_c.py", artifact="/t/c15.py")

    # Round 1: strongest first.
    assert state.next_pending_keep_kernel_id() == "k009"
    assert state.pending_keep_kernel_ids() == ["k009", "k015", "k001"]
    assert state.has_keep_pending_integrate is True

    # Simulate integrate k009 KEEP -> writes to optimization_stack.
    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k009",
            "target_file": "/p/file_b.py",
            "tput": 4500.0,
        }
    )

    # Round 2: next-strongest.
    assert state.next_pending_keep_kernel_id() == "k015"
    assert state.pending_keep_kernel_ids() == ["k015", "k001"]

    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k015",
            "target_file": "/p/file_c.py",
            "tput": 4620.0,
        }
    )

    # Round 3: last one.
    assert state.next_pending_keep_kernel_id() == "k001"

    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k001",
            "target_file": "/p/file_a.py",
            "tput": 4700.0,
        }
    )

    # Drained.
    assert state.next_pending_keep_kernel_id() == ""
    assert state.pending_keep_kernel_ids() == []
    assert state.has_keep_pending_integrate is False


def test_next_pending_keep_skips_same_source_file_after_integrate(state: SharedState):
    """Whole-file overwrite: a queued KEEP on an already-integrated source_file is dropped."""
    _seed_keep(state, "k001", decision="KEEP", micro=3.0, source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py")
    # Different kernel, same file -- weaker.
    _seed_keep(state, "k003", decision="KEEP", micro=2.0, source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py")
    _seed_keep(state, "k009", decision="KEEP", micro=4.13, source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py")

    # Strongest per file; k003 collapses away (shares moe_op.py with stronger k001).
    queue = state.pending_keep_kernel_ids()
    assert queue == ["k009", "k001"], queue
    assert "k003" not in queue

    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k001",
            "target_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
            "tput": 4500.0,
        }
    )
    assert state.next_pending_keep_kernel_id() == "k009"
    assert state.pending_keep_kernel_ids() == ["k009"]
    assert "k003" not in state.pending_keep_kernel_ids()


def test_next_pending_keep_excludes_rejected_and_integrated(state: SharedState):
    """Both rejected_kernel_ids and optimization_stack entries gate kernels out of the queue."""
    _seed_keep(state, "k001", decision="KEEP", micro=1.5, source_file="/p/a.py")
    _seed_keep(state, "k009", decision="KEEP", micro=4.0, source_file="/p/b.py")

    state.rejected_kernel_ids.append("k009")

    assert state.next_pending_keep_kernel_id() == "k001"

    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k001",
            "target_file": "/p/a.py",
            "tput": 4400.0,
        }
    )
    assert state.next_pending_keep_kernel_id() == ""
    assert state.has_keep_pending_integrate is False


def test_kernel_opt_attempts_count_property(state: SharedState):
    assert state.kernel_opt_attempts_count == 0
    _seed_keep(state, "k001", decision="KEEP", micro=1.5)
    _seed_keep(state, "k001", decision="REVERT", micro=0.9)  # same kid
    _seed_keep(state, "k002", decision="PARTIAL", micro=1.0)
    assert state.kernel_opt_attempts_count == 2


# failure_count + max_failures = 1 retirement.
def test_resolve_kernel_opt_max_failures_defaults_and_env(monkeypatch):
    from hyperloom.orchestrator.state.kernel_decision_settings import (
        _DEFAULT_KERNEL_OPT_MAX_FAILURES,
    )
    from hyperloom.orchestrator.state.shared_state import resolve_kernel_opt_max_failures

    monkeypatch.delenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", raising=False)
    assert resolve_kernel_opt_max_failures() == _DEFAULT_KERNEL_OPT_MAX_FAILURES

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "4")
    assert resolve_kernel_opt_max_failures() == 4

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "0")
    assert resolve_kernel_opt_max_failures() == 1

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "bad")
    assert resolve_kernel_opt_max_failures() == _DEFAULT_KERNEL_OPT_MAX_FAILURES


# untried_hot_reusable_kernels report gate.
def _set_trace(state: SharedState, *, hot_kernels, task_groups=None):
    state.last_trace_analyze = {
        "hot_kernels": hot_kernels,
        "task_groups": task_groups or [],
    }


def test_untried_hot_kernels_returns_only_reusable_above_threshold(state: SharedState):
    _set_trace(
        state,
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {
                "kernel_id": "k003",
                "gpu_pct": 4.5,
                "reusable_native_kernel": True,
                "source_file": "/p/rmsnorm.py",
            },  # just BELOW the 5% threshold
            {
                "kernel_id": "k004",
                "gpu_pct": 5.5,
                "reusable_native_kernel": True,
                "source_file": "/p/rmsnorm.py",
            },  # just ABOVE the 5% threshold
            {
                "kernel_id": "k006",
                "gpu_pct": 15.0,
                "reusable_native_kernel": False,
                "source_file": "/aten/mm.py",
            },  # aten not reusable
        ],
    )
    untried = state.untried_hot_reusable_kernels()
    assert set(untried) == {"k001", "k002", "k004"}
    assert "k003" not in untried  # below _DEFAULT_HOT_KERNEL_MIN_GPU_PCT (5.0)
    assert "k006" not in untried  # non-reusable


def test_untried_hot_kernels_reproduces_log1_session_164910Z(state: SharedState):
    """Replay of a real trace: 2 of its reusable hot kernels report untried.

    The gpu_pct values below are verbatim from the recorded session and must NOT
    be tuned to the gate. Under the 5% ``_DEFAULT_HOT_KERNEL_MIN_GPU_PCT`` k001
    (23.7), k002 (37.3) and k004 (9.7) clear it; k005 (2.8) and k003 (1.3) are
    below it and k006/k007 are non-reusable. k004 is the case the old 10% gate
    dropped: a real hot kernel that no wrapper-free operator in this trace could
    have reached.
    """
    _set_trace(
        state,
        hot_kernels=[
            {
                "kernel_id": "k001",
                "gpu_pct": 23.7,
                "reusable_native_kernel": True,
                "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
            },
            {
                "kernel_id": "k002",
                "gpu_pct": 37.3,
                "reusable_native_kernel": True,
                "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
            },
            {
                "kernel_id": "k003",
                "gpu_pct": 1.3,
                "reusable_native_kernel": True,
                "source_file": "/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
            },
            {
                "kernel_id": "k004",
                "gpu_pct": 9.7,
                "reusable_native_kernel": True,
                "source_file": "/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
            },
            {
                "kernel_id": "k005",
                "gpu_pct": 2.8,
                "reusable_native_kernel": True,
                "source_file": "/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
            },
            {"kernel_id": "k006", "gpu_pct": 15.6, "reusable_native_kernel": False, "source_file": ""},
            {"kernel_id": "k007", "gpu_pct": 9.7, "reusable_native_kernel": False, "source_file": ""},
        ],
    )
    untried = state.untried_hot_reusable_kernels()
    assert set(untried) == {"k001", "k002", "k004"}
    assert "k003" not in untried  # 1.3% sits below the 5% gate
    assert "k006" not in untried  # non-reusable despite 15.6%
    assert untried[0] == "k002"  # strongest-first


def test_untried_hot_kernels_vendor_playbook_group_gated_on_aggregate(state: SharedState):
    """mori's dispatch (7%) + combine (5%) must clear the gate together.

    Neither member clears the 10% default threshold alone, but
    _apply_vendor_operator_playbook_grouping() (tracelens_analysis.py) stamps
    vendor_playbook_aggregate_gpu_pct=12.0 on both, since the pair is deliberately dispatched
    as one forge-loop session (see KernelForge PR #88 / the mori vendor
    playbook). Regression for a real gap: the gate used to compare each row's
    own gpu_pct, so a split load like this was silently dropped as
    below_min_gpu_pct on both members despite clearing the floor combined.
    """
    _set_trace(
        state,
        hot_kernels=[
            {
                "kernel_id": "k010",
                "gpu_pct": 7.0,
                "vendor_playbook_aggregate_gpu_pct": 12.0,
                "reusable_native_kernel": True,
                "source_file": "/opt/venv/.../mori_ep_config.py",
                "name": "mori::EpDispatchCombineOp::dispatch",
            },
            {
                "kernel_id": "k011",
                "gpu_pct": 5.0,
                "vendor_playbook_aggregate_gpu_pct": 12.0,
                "reusable_native_kernel": True,
                "source_file": "/opt/venv/.../mori_ep_config.py",
                "name": "mori::EpDispatchCombineOp::combine",
            },
        ],
    )
    untried = state.untried_hot_reusable_kernels()
    # Both members carry the group's full aggregate and must both clear the
    # gate. No task_groups metadata is supplied here, and the two rows do
    # NOT share (source_file, name) -- names differ (`::dispatch` vs
    # `::combine`) -- so neither the group-key dedup nor the identity-dedup
    # fallback collapses them into one; both remain distinct, separately
    # gated rows.
    assert set(untried) == {"k010", "k011"}, "vendor-playbook group must not be dropped as below_min_gpu_pct"


def test_untried_hot_kernels_vendor_playbook_floor_still_applies(state: SharedState):
    """A playbook's min_gpu_pct_floor is a floor on the *threshold*, not a
    bypass: an aggregate that clears a loosened env override but not the
    playbook's own floor must still be gated out."""
    _set_trace(
        state,
        hot_kernels=[
            {
                "kernel_id": "k010",
                "gpu_pct": 2.0,
                "vendor_playbook_aggregate_gpu_pct": 3.0,
                "vendor_playbook_min_gpu_pct_floor": 10.0,
                "reusable_native_kernel": True,
                "source_file": "/opt/venv/.../mori_ep_config.py",
                "name": "mori::EpDispatchCombineOp::dispatch",
            },
        ],
    )
    # A caller loosening the env default to 1.0% must not let this in: the
    # playbook's own floor (10.0) still applies.
    untried = state.untried_hot_reusable_kernels(min_gpu_pct=1.0)
    assert untried == []


def test_untried_hot_kernels_vendor_playbook_gate_survives_real_projection(state: SharedState):
    """Regression for PR #1191 tech-lead finding: the aggregate/floor gate
    was a no-op on the production path because ``untried_hot_reusable_kernels()``
    reads ``hot_kernels_top15`` (SharedState._build_hot_kernel_summaries()'s
    projected ``summary_entry``, an explicit key whitelist) in preference to
    raw ``hot_kernels``, and that whitelist dropped
    ``vendor_playbook_aggregate_gpu_pct`` / ``vendor_playbook_min_gpu_pct_floor``
    / ``vendor_playbook_group_id`` / ``patch_strategy`` entirely.

    ``_set_trace()`` (used by the sibling tests above) assigns
    ``last_trace_analyze`` directly and never populates ``hot_kernels_top15``,
    so those tests fall through to the raw, unprojected ``hot_kernels`` and
    cannot catch this -- this test goes through the real
    ``record_trace_analyze()`` entry point instead, exactly like a live
    trace-analyze result would.
    """
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace.json"},
        {
            "status": "ok",
            "hot_kernels": [
                {
                    "kernel_id": "k010",
                    "gpu_pct": 7.0,
                    "vendor_playbook_aggregate_gpu_pct": 12.0,
                    "vendor_playbook_group_id": "mori_ep_dispatch_combine",
                    "patch_strategy": "vendor_playbook",
                    "reusable_native_kernel": True,
                    "source_file": "/opt/venv/.../mori_ep_config.py",
                    "name": "mori::EpDispatchCombineOp::dispatch",
                },
                {
                    "kernel_id": "k011",
                    "gpu_pct": 5.0,
                    "vendor_playbook_aggregate_gpu_pct": 12.0,
                    "vendor_playbook_group_id": "mori_ep_dispatch_combine",
                    "patch_strategy": "vendor_playbook",
                    "reusable_native_kernel": True,
                    "source_file": "/opt/venv/.../mori_ep_config.py",
                    "name": "mori::EpDispatchCombineOp::combine",
                },
            ],
        },
    )
    # The projection must actually carry the fields through -- this is the
    # exact assertion that fails without the _build_hot_kernel_summaries() fix.
    projected = {row["kernel_id"]: row for row in state.last_trace_analyze["hot_kernels_top15"]}
    assert projected["k010"]["vendor_playbook_aggregate_gpu_pct"] == 12.0
    assert projected["k010"]["patch_strategy"] == "vendor_playbook"
    assert projected["k010"]["vendor_playbook_group_id"] == "mori_ep_dispatch_combine"

    # Split-load pass-through direction: neither member clears the 10%
    # default alone (7%, 5%), but the pair's aggregate (12%) must.
    untried = state.untried_hot_reusable_kernels()
    assert set(untried) == {"k010", "k011"}, (
        "aggregate gate must not degrade to bare gpu_pct on the real "
        "record_trace_analyze() -> hot_kernels_top15 production path"
    )


def test_untried_hot_kernels_vendor_playbook_floor_still_applies_via_real_projection(
    state: SharedState,
):
    """Unsafe-direction counterpart of the test above: a playbook's own
    ``min_gpu_pct_floor`` must still block dispatch through the real
    projection path, even when the caller has loosened
    ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT``. Before the projection fix this
    degraded to the bare (loosened) threshold, letting a below-floor group
    burn a whole forge-loop session."""
    state.record_trace_analyze(
        {"trace_input": "/tmp/trace.json"},
        {
            "status": "ok",
            "hot_kernels": [
                {
                    "kernel_id": "k010",
                    "gpu_pct": 2.0,
                    "vendor_playbook_aggregate_gpu_pct": 3.0,
                    "vendor_playbook_min_gpu_pct_floor": 10.0,
                    "reusable_native_kernel": True,
                    "source_file": "/opt/venv/.../mori_ep_config.py",
                    "name": "mori::EpDispatchCombineOp::dispatch",
                },
            ],
        },
    )
    projected = state.last_trace_analyze["hot_kernels_top15"][0]
    assert projected["vendor_playbook_min_gpu_pct_floor"] == 10.0

    untried = state.untried_hot_reusable_kernels(min_gpu_pct=1.0)
    assert untried == [], (
        "the playbook's own floor must survive the real projection path "
        "and still block dispatch even under a loosened env override"
    )


def test_untried_hot_kernels_collapses_by_task_group(state: SharedState):
    """task_group dedup: same AST function -> one slot."""
    _set_trace(
        state,
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    untried = state.untried_hot_reusable_kernels()
    assert len(untried) == 1  # shared task_group -> one slot
    assert untried[0] == "k002"  # highest-gpu_pct member


def test_untried_hot_kernels_skips_when_any_group_member_attempted(state: SharedState):
    """An attempted group member marks the whole task_group tried."""
    _set_trace(
        state,
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
        ],
        task_groups=[
            {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
        ],
    )
    _seed_keep(
        state,
        "k002",
        decision="REVERT",
        micro=0.0,
        source_file="/p/moe_op.py",
        task_group_key="k002",
    )
    untried = state.untried_hot_reusable_kernels()
    assert untried == []


def test_untried_hot_kernels_skips_when_source_file_integrated(state: SharedState):
    """Once an integrate touches a source file, no further KEEP on it is meaningful."""
    _set_trace(
        state,
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            # Both are comfortably above the 10% gate, so the only thing that can
            # drop k001 below is the integrate on its source_file.
            {"kernel_id": "k009", "gpu_pct": 12.0, "reusable_native_kernel": True, "source_file": "/p/rmsnorm.py"},
        ],
    )
    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k001",
            "target_file": "/p/moe_op.py",
            "tput": 4500.0,
        }
    )
    untried = state.untried_hot_reusable_kernels()
    assert untried == ["k009"]


def test_untried_hot_kernels_caps_at_top_n(state: SharedState, monkeypatch):
    """Even with 15 reusable hot kernels, the gate only demands top-N."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_GATE_TOP_N", "3")
    hot = [
        {
            "kernel_id": f"k{i:03d}",
            "gpu_pct": 30.0 - i * 1.5,
            "reusable_native_kernel": True,
            "source_file": f"/p/f{i}.py",
        }
        for i in range(15)
    ]
    _set_trace(state, hot_kernels=hot)
    untried = state.untried_hot_reusable_kernels()
    assert len(untried) == 3


# record_kernel_integrate_result distinguishes integration faults from
# genuine gate REVERTs and gives faults an independent bounded retry budget.
def _integrate_result(
    kernel_id: str,
    *,
    decision: str | None = None,
    status: str = "ok",
    error_class: str | None = None,
    patch_path: str = "",
    target_file: str = "",
    gain_pct: float | None = None,
    new_tput: float | None = None,
) -> dict:
    """Build an integrate E2E result envelope (kernel integrate path)."""
    return {
        "status": status,
        "decision": decision,
        "kernel_id": kernel_id,
        "patch_path": patch_path or f"/tmp/{kernel_id}_opt.py",
        "target_file": target_file,
        "error_class": error_class,
        "gain_pct": gain_pct,
        "new_tput": new_tput,
    }


def test_integrate_fault_does_not_consume_revert_quota(state: SharedState):
    """An integration fault marks the entry retryable, never entering rejected."""
    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="failed",
            error_class="rebaseline_exception",
        ),
    )
    assert entry is not None
    assert entry["fault_count"] == 1
    assert entry["verdict_attempt_count"] == 0
    assert entry.get("retryable") is True
    assert "rejected" not in entry
    assert state.rejected_kernel_patches == []
    assert "k001" not in state.rejected_kernel_ids


@pytest.mark.parametrize("error_class", ["session_time_exhausted", "orchestrator_cancelled"])
def test_a_run_stopped_integrate_does_not_consume_revert_quota(state: SharedState, error_class):
    """A patch the run never measured must not be counted as one that lost."""
    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="NEEDS_REVIEW",
            status="failed",
            error_class=error_class,
        ),
    )
    assert entry is not None
    assert entry["verdict_attempt_count"] == 0
    assert entry.get("retryable") is True
    assert state.rejected_kernel_patches == []


def test_integrate_attempt_is_stamped_with_macro_cycle(state: SharedState):
    state.macro_cycle = 2
    entry = state.record_kernel_integrate_result(
        _integrate_result("k001", decision="KEEP", gain_pct=1.0),
    )
    assert entry is not None
    assert entry["attempts"][-1]["cycle"] == 2


def test_integrate_fault_rejected_after_budget_exhausted(state: SharedState):
    """The first fault stays retryable; the second exhausts the budget and rejects."""
    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="failed",
            error_class="bench_exception",
        ),
    )
    assert entry.get("retryable") is True
    assert "rejected" not in entry

    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="failed",
            error_class="bench_exception",
        ),
    )
    assert entry.get("retryable") is not True
    assert entry["rejected"]["reason"] == "fault_attempts_exhausted_2"
    assert "k001" in state.rejected_kernel_ids


def test_integrate_genuine_revert_rejects_immediately(state: SharedState):
    """A real gate REVERT (no fault error_class) is terminal on the first attempt."""
    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="ok",
            gain_pct=-3.0,
        ),
    )
    assert entry["fault_count"] == 0
    assert entry["verdict_attempt_count"] == 1
    assert entry.get("retryable") is not True
    assert entry["rejected"]["reason"] == "revert_decision"
    assert "k001" in state.rejected_kernel_ids


def test_integrate_bare_apply_fault_is_retryable_without_error_class(
    state: SharedState,
):
    """A status=failed/decision=REVERT envelope with NO top-level error_class
    must be treated as a retryable fault, not a genuine REVERT."""
    entry = state.record_kernel_integrate_result(
        # NOTE: no error_class — mirrors the bare handler envelope.
        _integrate_result("k001", decision="REVERT", status="failed"),
    )
    assert entry is not None
    assert entry["fault_count"] == 1
    assert entry["verdict_attempt_count"] == 0
    assert entry.get("retryable") is True
    assert "rejected" not in entry
    assert state.rejected_kernel_patches == []
    assert "k001" not in state.rejected_kernel_ids


def test_integrate_keep_is_terminal_and_not_rejected(state: SharedState):
    """A KEEP returns without rejecting or marking retryable."""
    entry = state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="KEEP",
            status="ok",
            gain_pct=5.0,
            new_tput=105.0,
        ),
    )
    assert "rejected" not in entry
    assert entry.get("retryable") is not True
    assert state.rejected_kernel_patches == []


def test_pending_keep_includes_kernel_with_unexhausted_fault(state: SharedState):
    """A kernel whose only integrate attempt is an un-exhausted fault stays queueable."""
    _seed_keep(state, "k001", decision="KEEP", micro=3.0, source_file="/p/a.py", artifact="/tmp/k001_opt.py")
    assert state.pending_keep_kernel_ids() == ["k001"]

    # An integration fault must NOT remove it from the pending queue (retryable).
    state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="failed",
            error_class="apply_failed",
            patch_path="/tmp/k001_opt.py",
        ),
    )
    assert "k001" not in state._kernel_ids_with_integrate_attempts()
    assert state.pending_keep_kernel_ids() == ["k001"]


def test_pending_keep_drops_kernel_after_fault_budget_exhausted(state: SharedState):
    """Once the fault budget is spent and the kernel is rejected, it leaves the queue."""
    _seed_keep(state, "k001", decision="KEEP", micro=3.0, source_file="/p/a.py", artifact="/tmp/k001_opt.py")
    for _ in range(3):
        state.record_kernel_integrate_result(
            _integrate_result(
                "k001",
                decision="REVERT",
                status="failed",
                error_class="apply_failed",
                patch_path="/tmp/k001_opt.py",
            ),
        )
    assert "k001" in state.rejected_kernel_ids
    assert state.pending_keep_kernel_ids() == []


def test_pending_keep_drops_kernel_on_genuine_revert(state: SharedState):
    """A real gate REVERT on the integrate attempt removes the kernel immediately."""
    _seed_keep(state, "k001", decision="KEEP", micro=3.0, source_file="/p/a.py", artifact="/tmp/k001_opt.py")
    state.record_kernel_integrate_result(
        _integrate_result(
            "k001",
            decision="REVERT",
            status="ok",
            gain_pct=-3.0,
            patch_path="/tmp/k001_opt.py",
        ),
    )
    assert "k001" in state._kernel_ids_with_integrate_attempts()
    assert state.pending_keep_kernel_ids() == []
