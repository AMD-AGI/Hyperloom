# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for SharedState.record_kernel_opt invariants + the multi-KEEP integrate queue helpers."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


def _ok_result(
    kernel_id: str,
    decision: str,
    micro: float,
    source_file: str = "",
    artifact: str = "",
) -> dict:
    """Build a kernel_optimization_handler-shaped result dict."""
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "source_file": source_file,
        "proposal": {"decision": decision, "reasons": []},
        "verification": {
            "micro_speedup": micro,
            "best_artifact_path": artifact,
            "compile_passed": True,
            "correctness_passed": True,
        },
    }


@pytest.fixture
def state() -> SharedState:
    return SharedState()


# Invariant 1: empty kernel_id is a no-op
def test_record_kernel_opt_empty_kernel_id_is_noop_after_keep(state: SharedState):
    """A metadata-less failure must NOT clobber a previously-recorded KEEP."""
    keep = _ok_result(
        "k009",
        "KEEP",
        micro=4.13,
        source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
        artifact="/tmp/k009_patch.py",
    )
    state.record_kernel_opt(keep)
    assert state.last_kernel_opt["kernel_id"] == "k009"
    assert state.last_kernel_opt["decision"] == "KEEP"

    # Coordinator's batch handler exception path wraps as a bare failure dict.
    failed_wrap = {
        "status": "failed",
        "error_class": "handler_exception",
        "error": "TimeoutExpired(['python3', 'kernel_optimization.py', ...], 5580)",
    }
    state.record_kernel_opt(failed_wrap)

    assert state.last_kernel_opt["kernel_id"] == "k009", "empty-kernel_id failure must not overwrite a pending KEEP"
    assert state.last_kernel_opt["decision"] == "KEEP"
    assert state.kernel_opt_attempts_count == 1, "no kernel_id => no attempts ledger update"


def test_record_kernel_opt_empty_kernel_id_noop_on_blank_state(state: SharedState):
    """No prior data + empty kernel_id => still a no-op (no spurious stub written)."""
    state.record_kernel_opt({"status": "failed", "error": "transport"})
    assert state.last_kernel_opt == {}
    assert state.kernel_opt_attempts == {}


def test_record_kernel_opt_no_eligible_kernels_skip_is_captured(state: SharedState):
    """An empty-queue skip is stashed as a non-failure breadcrumb."""
    state.record_kernel_opt(
        {
            "status": "skipped",
            "reason": "no_eligible_kernels",
            "kernels_considered": 7,
            "message": "no eligible kernels to optimize (...)",
        }
    )
    skip = state.last_kernel_opt_dispatch_skip
    assert skip["reason"] == "no_eligible_kernels"
    assert skip["kernels_considered"] == 7
    assert skip["ts"]
    # A non-failure skip must not pollute the per-kernel ledgers or last_kernel_opt.
    assert state.last_kernel_opt == {}
    assert state.kernel_opt_attempts == {}
    assert state.rejected_kernel_ids == []


def test_record_kernel_opt_plain_failure_does_not_set_dispatch_skip(state: SharedState):
    """A regular failure (no no_eligible_kernels reason) leaves dispatch-skip empty."""
    state.record_kernel_opt({"status": "failed", "error": "missing 'kernel_id' in payload"})
    assert state.last_kernel_opt_dispatch_skip == {}


# Invariant 2: KEEP wins; non-KEEP never overwrites a pending KEEP
def test_record_kernel_opt_keep_survives_later_revert(state: SharedState):
    """A later REVERT on a different kernel must not displace an un-integrated KEEP."""
    state.record_kernel_opt(
        _ok_result(
            "k009",
            "KEEP",
            4.13,
            source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "REVERT",
            0.95,
            source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        )
    )

    assert state.last_kernel_opt["kernel_id"] == "k009"
    assert state.last_kernel_opt["decision"] == "KEEP"
    # The REVERT was still ledgered against k001 (and retired it).
    assert "k001" in state.kernel_opt_attempts
    assert state.kernel_opt_attempts["k001"]["last_decision"] == "REVERT"
    assert "k001" in state.rejected_kernel_ids


def test_record_kernel_opt_keep_always_overrides_prev_keep(state: SharedState):
    """Two KEEPs in succession => the second wins; the earlier KEEP stays queueable."""
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            2.0,
            source_file="/path/moe_op.py",
            artifact="/tmp/k001.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k009",
            "KEEP",
            4.13,
            source_file="/path/rmsnorm.py",
            artifact="/tmp/k009.py",
        )
    )

    assert state.last_kernel_opt["kernel_id"] == "k009"
    assert state.last_kernel_opt["micro_speedup"] == 4.13
    assert state.kernel_opt_attempts["k001"]["last_decision"] == "KEEP"
    assert state.kernel_opt_attempts["k009"]["last_decision"] == "KEEP"


def test_record_kernel_opt_nonkeep_overwrites_when_prev_already_integrated(state: SharedState):
    """An already-integrated KEEP is no longer pending, so a non-KEEP may overwrite it."""
    state.record_kernel_opt(
        _ok_result(
            "k009",
            "KEEP",
            4.13,
            source_file="/path/rmsnorm.py",
        )
    )
    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k009",
            "target_file": "/path/rmsnorm.py",
            "tput": 4500.0,
        }
    )

    state.record_kernel_opt(
        _ok_result(
            "k004",
            "PARTIAL",
            0.8,
            source_file="/path/moe_op.py",
        )
    )
    assert state.last_kernel_opt["kernel_id"] == "k004", (
        "k009 already integrated => no longer pending => k004 PARTIAL may overwrite"
    )


# next_pending_keep_kernel_id queue semantics
def test_next_pending_keep_drains_in_micro_speedup_order(state: SharedState):
    """KEEPs on different source_files drain highest-micro-first as the stack fills."""
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            2.5,
            source_file="/p/file_a.py",
            artifact="/t/a1.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k009",
            "KEEP",
            4.13,
            source_file="/p/file_b.py",
            artifact="/t/b9.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k015",
            "KEEP",
            3.2,
            source_file="/p/file_c.py",
            artifact="/t/c15.py",
        )
    )

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
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k003",
            "KEEP",
            2.0,  # different kernel, same file -- weaker
            source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k009",
            "KEEP",
            4.13,
            source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
        )
    )

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
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            1.5,
            source_file="/p/a.py",
        )
    )
    state.record_kernel_opt(
        _ok_result(
            "k009",
            "KEEP",
            4.0,
            source_file="/p/b.py",
        )
    )

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
    state.record_kernel_opt(_ok_result("k001", "KEEP", 1.5))
    state.record_kernel_opt(_ok_result("k001", "REVERT", 0.9))  # same kid
    state.record_kernel_opt(_ok_result("k002", "PARTIAL", 1.0))
    assert state.kernel_opt_attempts_count == 2


# PR-C: failure_count + max_failures = 1 retirement
def _failed_result(
    kernel_id: str, *, status: str = "failed", error_class: str = "subtask_exception", source_file: str = ""
) -> dict:
    return {
        "status": status,
        "kernel_id": kernel_id,
        "source_file": source_file,
        "error_class": error_class,
        "error": "simulated",
    }


def test_record_kernel_opt_failure_count_increments_on_status_failed(state: SharedState):
    state.record_kernel_opt(
        _failed_result(
            "k001",
            status="failed",
            source_file="/p/a.py",
        )
    )
    e = state.kernel_opt_attempts["k001"]
    assert e["failure_count"] == 1
    assert e["last_status"] == "failed"


def test_record_kernel_opt_one_failure_does_not_retire_kernel_by_default(state: SharedState):
    """A transient backend/infra failure gets one retry before retirement."""
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" not in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["failure_count"] == 1
    assert not state.kernel_opt_attempts["k001"].get("rejected_reason")


def test_record_kernel_opt_second_failure_retires_kernel_by_default(state: SharedState):
    """Two backend/infra failures retire the kernel when no KEEP appears."""
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["rejected_reason"] == "max_failures_2_without_keep"


def test_record_kernel_opt_revert_retires_immediately(state: SharedState):
    state.record_kernel_opt(_ok_result("k001", "REVERT", 0.9, source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["rejected_reason"] == "revert_decision"


def test_record_kernel_opt_keep_resets_failure_count(state: SharedState):
    """A later KEEP clears the failure streak so the kernel is usable again."""
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    # A subsequent KEEP clears the streak.
    state.record_kernel_opt(_ok_result("k001", "KEEP", 4.0, source_file="/p/a.py"))
    e = state.kernel_opt_attempts["k001"]
    assert e["failure_count"] == 0
    assert e["last_decision"] == "KEEP"


def test_record_kernel_opt_max_failures_env_override(state: SharedState, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "1")
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids


def test_resolve_kernel_opt_max_failures_defaults_and_env(monkeypatch):
    from inference_optimizer.orchestrator.shared_state import (
        _DEFAULT_KERNEL_OPT_MAX_FAILURES,
        resolve_kernel_opt_max_failures,
    )

    monkeypatch.delenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", raising=False)
    assert resolve_kernel_opt_max_failures() == _DEFAULT_KERNEL_OPT_MAX_FAILURES

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "4")
    assert resolve_kernel_opt_max_failures() == 4

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "0")
    assert resolve_kernel_opt_max_failures() == 1

    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "bad")
    assert resolve_kernel_opt_max_failures() == _DEFAULT_KERNEL_OPT_MAX_FAILURES


# PR-C: untried_hot_reusable_kernels report gate
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
                "gpu_pct": 1.2,
                "reusable_native_kernel": True,
                "source_file": "/p/rmsnorm.py",
            },  # below 3% threshold
            {"kernel_id": "k004", "gpu_pct": 9.7, "reusable_native_kernel": True, "source_file": "/p/rmsnorm.py"},
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
    assert "k003" not in untried  # below 3% threshold
    assert "k006" not in untried  # non-reusable


def test_untried_hot_kernels_reproduces_log1_session_164910Z(state: SharedState):
    """Real numbers from session 20260522T164910Z: 3 reusable hot kernels report untried."""
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
    assert set(untried) >= {"k001", "k002", "k004"}
    assert untried[0] == "k002"  # strongest-first


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
    state.record_kernel_opt(_failed_result("k002", source_file="/p/moe_op.py"))
    untried = state.untried_hot_reusable_kernels()
    assert untried == []


def test_untried_hot_kernels_skips_when_source_file_integrated(state: SharedState):
    """Once an integrate touches a source file, no further KEEP on it is meaningful."""
    _set_trace(
        state,
        hot_kernels=[
            {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True, "source_file": "/p/moe_op.py"},
            {"kernel_id": "k009", "gpu_pct": 4.5, "reusable_native_kernel": True, "source_file": "/p/rmsnorm.py"},
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


# Issue 3: record_kernel_integrate_result distinguishes integration faults from
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
    """A status=failed/decision=REVERT envelope with NO top-level error_class.

    This is the exact shape ``integrate_handler`` emits on the apply-failed and
    "re-baseline did not succeed" paths. Keying fault detection on ``decision``
    would misread it as a genuine REVERT and discard the patch on the first
    environment hiccup; it must instead be a retryable fault.
    """
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
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
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
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
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
    state.record_kernel_opt(
        _ok_result(
            "k001",
            "KEEP",
            3.0,
            source_file="/p/a.py",
            artifact="/tmp/k001_opt.py",
        )
    )
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
