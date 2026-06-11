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
        "k009", "KEEP", micro=4.13,
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

    assert state.last_kernel_opt["kernel_id"] == "k009", \
        "empty-kernel_id failure must not overwrite a pending KEEP"
    assert state.last_kernel_opt["decision"] == "KEEP"
    assert state.kernel_opt_attempts_count == 1, \
        "no kernel_id => no attempts ledger update"


def test_record_kernel_opt_empty_kernel_id_noop_on_blank_state(state: SharedState):
    """No prior data + empty kernel_id => still a no-op (no spurious stub written)."""
    state.record_kernel_opt({"status": "failed", "error": "transport"})
    assert state.last_kernel_opt == {}
    assert state.kernel_opt_attempts == {}


# Invariant 2: KEEP wins; non-KEEP never overwrites a pending KEEP
def test_record_kernel_opt_keep_survives_later_revert(state: SharedState):
    """A later REVERT on a different kernel must not displace an un-integrated KEEP."""
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.13,
        source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k001", "REVERT", 0.95,
        source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
    ))

    assert state.last_kernel_opt["kernel_id"] == "k009"
    assert state.last_kernel_opt["decision"] == "KEEP"
    # The REVERT was still ledgered against k001 (and retired it).
    assert "k001" in state.kernel_opt_attempts
    assert state.kernel_opt_attempts["k001"]["last_decision"] == "REVERT"
    assert "k001" in state.rejected_kernel_ids


def test_record_kernel_opt_keep_always_overrides_prev_keep(state: SharedState):
    """Two KEEPs in succession => the second wins; the earlier KEEP stays queueable."""
    state.record_kernel_opt(_ok_result(
        "k001", "KEEP", 2.0,
        source_file="/path/moe_op.py", artifact="/tmp/k001.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.13,
        source_file="/path/rmsnorm.py", artifact="/tmp/k009.py",
    ))

    assert state.last_kernel_opt["kernel_id"] == "k009"
    assert state.last_kernel_opt["micro_speedup"] == 4.13
    assert state.kernel_opt_attempts["k001"]["last_decision"] == "KEEP"
    assert state.kernel_opt_attempts["k009"]["last_decision"] == "KEEP"


def test_record_kernel_opt_nonkeep_overwrites_when_prev_already_integrated(state: SharedState):
    """An already-integrated KEEP is no longer pending, so a non-KEEP may overwrite it."""
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.13,
        source_file="/path/rmsnorm.py",
    ))
    state.optimization_stack.append({
        "action": "integrate",
        "kernel_id": "k009",
        "target_file": "/path/rmsnorm.py",
        "tput": 4500.0,
    })

    state.record_kernel_opt(_ok_result(
        "k004", "PARTIAL", 0.8,
        source_file="/path/moe_op.py",
    ))
    assert state.last_kernel_opt["kernel_id"] == "k004", \
        "k009 already integrated => no longer pending => k004 PARTIAL may overwrite"


# next_pending_keep_kernel_id queue semantics
def test_next_pending_keep_drains_in_micro_speedup_order(state: SharedState):
    """KEEPs on different source_files drain highest-micro-first as the stack fills."""
    state.record_kernel_opt(_ok_result(
        "k001", "KEEP", 2.5, source_file="/p/file_a.py", artifact="/t/a1.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.13, source_file="/p/file_b.py", artifact="/t/b9.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k015", "KEEP", 3.2, source_file="/p/file_c.py", artifact="/t/c15.py",
    ))

    # Round 1: strongest first.
    assert state.next_pending_keep_kernel_id() == "k009"
    assert state.pending_keep_kernel_ids() == ["k009", "k015", "k001"]
    assert state.has_keep_pending_integrate is True

    # Simulate integrate k009 KEEP -> writes to optimization_stack.
    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k009",
        "target_file": "/p/file_b.py", "tput": 4500.0,
    })

    # Round 2: next-strongest.
    assert state.next_pending_keep_kernel_id() == "k015"
    assert state.pending_keep_kernel_ids() == ["k015", "k001"]

    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k015",
        "target_file": "/p/file_c.py", "tput": 4620.0,
    })

    # Round 3: last one.
    assert state.next_pending_keep_kernel_id() == "k001"

    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001",
        "target_file": "/p/file_a.py", "tput": 4700.0,
    })

    # Drained.
    assert state.next_pending_keep_kernel_id() == ""
    assert state.pending_keep_kernel_ids() == []
    assert state.has_keep_pending_integrate is False


def test_next_pending_keep_skips_same_source_file_after_integrate(state: SharedState):
    """Whole-file overwrite: a queued KEEP on an already-integrated source_file is dropped."""
    state.record_kernel_opt(_ok_result(
        "k001", "KEEP", 3.0,
        source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k003", "KEEP", 2.0,  # different kernel, same file -- weaker
        source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.13,
        source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
    ))

    # Strongest per file; k003 collapses away (shares moe_op.py with stronger k001).
    queue = state.pending_keep_kernel_ids()
    assert queue == ["k009", "k001"], queue
    assert "k003" not in queue

    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001",
        "target_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        "tput": 4500.0,
    })
    assert state.next_pending_keep_kernel_id() == "k009"
    assert state.pending_keep_kernel_ids() == ["k009"]
    assert "k003" not in state.pending_keep_kernel_ids()


def test_next_pending_keep_excludes_rejected_and_integrated(state: SharedState):
    """Both rejected_kernel_ids and optimization_stack entries gate kernels out of the queue."""
    state.record_kernel_opt(_ok_result(
        "k001", "KEEP", 1.5, source_file="/p/a.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.0, source_file="/p/b.py",
    ))

    state.rejected_kernel_ids.append("k009")

    assert state.next_pending_keep_kernel_id() == "k001"

    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001",
        "target_file": "/p/a.py", "tput": 4400.0,
    })
    assert state.next_pending_keep_kernel_id() == ""
    assert state.has_keep_pending_integrate is False


def test_kernel_opt_attempts_count_property(state: SharedState):
    assert state.kernel_opt_attempts_count == 0
    state.record_kernel_opt(_ok_result("k001", "KEEP", 1.5))
    state.record_kernel_opt(_ok_result("k001", "REVERT", 0.9))  # same kid
    state.record_kernel_opt(_ok_result("k002", "PARTIAL", 1.0))
    assert state.kernel_opt_attempts_count == 2


# PR-C: failure_count + max_failures = 1 retirement
def _failed_result(kernel_id: str, *, status: str = "failed",
                   error_class: str = "subtask_exception",
                   source_file: str = "") -> dict:
    return {
        "status": status,
        "kernel_id": kernel_id,
        "source_file": source_file,
        "error_class": error_class,
        "error": "simulated",
    }


def test_record_kernel_opt_failure_count_increments_on_status_failed(state: SharedState):
    state.record_kernel_opt(_failed_result(
        "k001", status="failed", source_file="/p/a.py",
    ))
    e = state.kernel_opt_attempts["k001"]
    assert e["failure_count"] == 1
    assert e["last_status"] == "failed"


def test_record_kernel_opt_one_failure_retires_kernel(state: SharedState):
    """PR-C max_failures=1: one completed-ladder-without-KEEP retires the kernel."""
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["rejected_reason"].startswith(
        "max_failures_"
    )


def test_record_kernel_opt_revert_retires_immediately(state: SharedState):
    state.record_kernel_opt(_ok_result("k001", "REVERT", 0.9,
                                       source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["rejected_reason"] == "revert_decision"


def test_record_kernel_opt_keep_resets_failure_count(state: SharedState):
    """A later KEEP clears the failure streak so the kernel is usable again."""
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids
    # A subsequent KEEP clears the streak.
    state.rejected_kernel_ids.remove("k001")
    state.record_kernel_opt(_ok_result("k001", "KEEP", 4.0,
                                       source_file="/p/a.py"))
    e = state.kernel_opt_attempts["k001"]
    assert e["failure_count"] == 0
    assert e["last_decision"] == "KEEP"


def test_record_kernel_opt_max_failures_env_override(state: SharedState, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "2")
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" not in state.rejected_kernel_ids
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids


# PR-C: untried_hot_reusable_kernels report gate
def _set_trace(state: SharedState, *, hot_kernels, task_groups=None):
    state.last_trace_analyze = {
        "hot_kernels": hot_kernels,
        "task_groups": task_groups or [],
    }


def test_untried_hot_kernels_returns_only_reusable_above_threshold(state: SharedState):
    _set_trace(state, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k003", "gpu_pct": 1.2, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},  # below 3% threshold
        {"kernel_id": "k004", "gpu_pct": 9.7, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
        {"kernel_id": "k006", "gpu_pct": 15.0, "reusable_native_kernel": False,
         "source_file": "/aten/mm.py"},  # aten not reusable
    ])
    untried = state.untried_hot_reusable_kernels()
    assert set(untried) == {"k001", "k002", "k004"}
    assert "k003" not in untried  # below 3% threshold
    assert "k006" not in untried  # non-reusable


def test_untried_hot_kernels_reproduces_log1_session_164910Z(state: SharedState):
    """Real numbers from session 20260522T164910Z: 3 reusable hot kernels report untried."""
    _set_trace(state, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 23.7, "reusable_native_kernel": True,
         "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.3, "reusable_native_kernel": True,
         "source_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py"},
        {"kernel_id": "k003", "gpu_pct": 1.3, "reusable_native_kernel": True,
         "source_file": "/sgl-workspace/aiter/aiter/ops/rmsnorm.py"},
        {"kernel_id": "k004", "gpu_pct": 9.7, "reusable_native_kernel": True,
         "source_file": "/sgl-workspace/aiter/aiter/ops/rmsnorm.py"},
        {"kernel_id": "k005", "gpu_pct": 2.8, "reusable_native_kernel": True,
         "source_file": "/sgl-workspace/aiter/aiter/ops/rmsnorm.py"},
        {"kernel_id": "k006", "gpu_pct": 15.6, "reusable_native_kernel": False,
         "source_file": ""},
        {"kernel_id": "k007", "gpu_pct": 9.7, "reusable_native_kernel": False,
         "source_file": ""},
    ])
    untried = state.untried_hot_reusable_kernels()
    assert set(untried) >= {"k001", "k002", "k004"}
    assert untried[0] == "k002"  # strongest-first


def test_untried_hot_kernels_collapses_by_task_group(state: SharedState):
    """task_group dedup: same AST function -> one slot."""
    _set_trace(state, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ], task_groups=[
        {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
    ])
    untried = state.untried_hot_reusable_kernels()
    assert len(untried) == 1  # shared task_group -> one slot
    assert untried[0] == "k002"  # highest-gpu_pct member


def test_untried_hot_kernels_skips_when_any_group_member_attempted(state: SharedState):
    """An attempted group member marks the whole task_group tried."""
    _set_trace(state, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ], task_groups=[
        {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
    ])
    state.record_kernel_opt(_failed_result("k002", source_file="/p/moe_op.py"))
    untried = state.untried_hot_reusable_kernels()
    assert untried == []


def test_untried_hot_kernels_skips_when_source_file_integrated(state: SharedState):
    """Once an integrate touches a source file, no further KEEP on it is meaningful."""
    _set_trace(state, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k009", "gpu_pct": 4.5, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001",
        "target_file": "/p/moe_op.py", "tput": 4500.0,
    })
    untried = state.untried_hot_reusable_kernels()
    assert untried == ["k009"]


def test_untried_hot_kernels_caps_at_top_n(state: SharedState, monkeypatch):
    """Even with 15 reusable hot kernels, the gate only demands top-N."""
    monkeypatch.setenv("HYPERLOOM_KERNEL_OPT_GATE_TOP_N", "3")
    hot = [
        {"kernel_id": f"k{i:03d}", "gpu_pct": 30.0 - i * 1.5,
         "reusable_native_kernel": True, "source_file": f"/p/f{i}.py"}
        for i in range(15)
    ]
    _set_trace(state, hot_kernels=hot)
    untried = state.untried_hot_reusable_kernels()
    assert len(untried) == 3
