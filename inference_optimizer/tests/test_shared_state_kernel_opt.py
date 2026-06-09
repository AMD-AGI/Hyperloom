"""Tests for SharedState.record_kernel_opt overwrite invariants + the
multi-KEEP integrate queue helpers (next_pending_keep_kernel_id,
pending_keep_kernel_ids, has_keep_pending_integrate).

These tests pin down two regression-prone behaviours that drove the
Qwen3-30B-A3B-Base session (20260522T093903Z) to leave a 4.13x KEEP
on the floor:

1. Streaming batch sub-results write to ``last_kernel_opt`` in
   completion order, not micro_speedup order. Without the "KEEP wins,
   non-KEEP never overwrites a pending KEEP" invariant, a late
   REVERT / TimeoutExpired sibling would erase an earlier KEEP and
   the integrate gate would never reopen.

2. The Coordinator's batch handler exception path wraps timeouts as
   ``{"status": "failed", "error_class": "handler_exception"}`` -- a
   metadata-less dict with NO ``kernel_id``. record_kernel_opt must
   no-op on such inputs instead of clobbering ``last_kernel_opt`` to
   an empty stub.

And one design-level guarantee: with the next_pending_keep_kernel_id
queue draining in micro_speedup-descending order, multiple KEEPs
across different source_files all get a chance to integrate, while
same source_file KEEPs collapse to the strongest one (because
``apply_kernel_patch`` is whole-file overwrite and the second integrate
would silently clobber the first).
"""

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


# ---------------------------------------------------------------------------
# Invariant 1: empty kernel_id is a no-op
# ---------------------------------------------------------------------------
def test_record_kernel_opt_empty_kernel_id_is_noop_after_keep(state: SharedState):
    """A metadata-less failure (batch handler exception wrap) must NOT
    clobber a previously-recorded KEEP. Otherwise the integrate gate
    keyed off ``last_kernel_opt.decision`` would never reopen."""
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
    """No prior data + empty kernel_id => still a no-op (no spurious
    {kernel_id:'', decision:''} stub written)."""
    state.record_kernel_opt({"status": "failed", "error": "transport"})
    assert state.last_kernel_opt == {}
    assert state.kernel_opt_attempts == {}


# ---------------------------------------------------------------------------
# Invariant 2: KEEP wins; non-KEEP never overwrites a pending KEEP
# ---------------------------------------------------------------------------
def test_record_kernel_opt_keep_survives_later_revert(state: SharedState):
    """Batch streaming order is completion order, not micro order. A
    later REVERT on a different kernel must not displace an earlier
    KEEP that hasn't been integrated yet."""
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
    # But the REVERT was still ledgered against k001 (and retired it).
    assert "k001" in state.kernel_opt_attempts
    assert state.kernel_opt_attempts["k001"]["last_decision"] == "REVERT"
    assert "k001" in state.rejected_kernel_ids


def test_record_kernel_opt_keep_always_overrides_prev_keep(state: SharedState):
    """Two KEEPs in succession => the second one wins (we want the
    strongest pending KEEP to surface to last_kernel_opt). The earlier
    KEEP is still queueable via kernel_opt_attempts."""
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
    # Both KEEPs are queueable.
    assert state.kernel_opt_attempts["k001"]["last_decision"] == "KEEP"
    assert state.kernel_opt_attempts["k009"]["last_decision"] == "KEEP"


def test_record_kernel_opt_nonkeep_overwrites_when_prev_already_integrated(state: SharedState):
    """If the previously-pending KEEP has already been integrated (i.e.
    it's now on optimization_stack), it is no longer "pending", so a
    new non-KEEP result IS allowed to overwrite last_kernel_opt."""
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.13,
        source_file="/path/rmsnorm.py",
    ))
    # Simulate the Coordinator's _record_integrate_keep landing k009.
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


# ---------------------------------------------------------------------------
# next_pending_keep_kernel_id queue semantics
# ---------------------------------------------------------------------------
def test_next_pending_keep_drains_in_micro_speedup_order(state: SharedState):
    """Multiple KEEPs on DIFFERENT source_files: queue returns highest
    micro first, then after integrate writes its entry to the stack the
    queue returns the next-highest."""
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
    """``apply_kernel_patch`` is whole-file overwrite. Once an integrate
    entry on the stack covers source_file X, any other queued KEEP on
    X must be dropped (otherwise the second integrate would silently
    clobber the first patch)."""
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

    # Before integrating anything: queue sees the strongest per file
    # (k001 on moe_op, k009 on rmsnorm); k003 is collapsed away because
    # it shares moe_op.py with the stronger k001.
    queue = state.pending_keep_kernel_ids()
    assert queue == ["k009", "k001"], queue
    assert "k003" not in queue

    # Integrate k001 (moe_op.py). Now k003 is also conflict_blocked at
    # the stack level (target_file matched), and stays out of the queue.
    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001",
        "target_file": "/sgl-workspace/aiter/aiter/ops/moe_op.py",
        "tput": 4500.0,
    })
    assert state.next_pending_keep_kernel_id() == "k009"
    assert state.pending_keep_kernel_ids() == ["k009"]
    assert "k003" not in state.pending_keep_kernel_ids()


def test_next_pending_keep_excludes_rejected_and_integrated(state: SharedState):
    """Both rejected_kernel_ids (e.g. PARTIAL streak retired) AND
    optimization_stack entries gate kernel_ids out of the queue."""
    state.record_kernel_opt(_ok_result(
        "k001", "KEEP", 1.5, source_file="/p/a.py",
    ))
    state.record_kernel_opt(_ok_result(
        "k009", "KEEP", 4.0, source_file="/p/b.py",
    ))

    # External rejection (e.g. integrate REVERT'd it).
    state.rejected_kernel_ids.append("k009")

    # k001 still queueable.
    assert state.next_pending_keep_kernel_id() == "k001"

    # k001 lands on stack.
    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001",
        "target_file": "/p/a.py", "tput": 4400.0,
    })
    # Now nothing pending (k009 rejected, k001 integrated).
    assert state.next_pending_keep_kernel_id() == ""
    assert state.has_keep_pending_integrate is False


def test_kernel_opt_attempts_count_property(state: SharedState):
    assert state.kernel_opt_attempts_count == 0
    state.record_kernel_opt(_ok_result("k001", "KEEP", 1.5))
    state.record_kernel_opt(_ok_result("k001", "REVERT", 0.9))  # same kid
    state.record_kernel_opt(_ok_result("k002", "PARTIAL", 1.0))
    assert state.kernel_opt_attempts_count == 2  # unique kernels


# ---------------------------------------------------------------------------
# PR-C: failure_count + max_failures = 1 retirement
# ---------------------------------------------------------------------------
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
    """PR-C max_failures=1: one completed-ladder-without-KEEP retires.

    Operator decision: GEAK->Claude->Codex runs once; if every backend
    fails to produce a KEEP, the kernel "cannot be optimized" and must
    not be re-dispatched. Without this, Qwen3-30B-A3B-Base 164405Z
    burned 8h re-running the same k002/k004 GEAK->Claude->Codex chain
    every time the LLM proposed a fresh run_optimization batch.
    """
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
    """An earlier failure must not retire a kernel once a later KEEP
    arrives -- e.g. GEAK timeout that streamed first, then Claude KEEP
    that streamed second within the same _run_kernel_backend_sequence
    aggregation."""
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    # failed once, retired
    assert "k001" in state.rejected_kernel_ids
    # but a subsequent KEEP (e.g. operator resumed, re-dispatched
    # explicitly) clears the streak so the kernel is usable again from
    # the queue's perspective.
    state.rejected_kernel_ids.remove("k001")
    state.record_kernel_opt(_ok_result("k001", "KEEP", 4.0,
                                       source_file="/p/a.py"))
    e = state.kernel_opt_attempts["k001"]
    assert e["failure_count"] == 0
    assert e["last_decision"] == "KEEP"


def test_record_kernel_opt_max_failures_env_override(state: SharedState, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES", "2")
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    # First failure -> not yet retired with env=2
    assert "k001" not in state.rejected_kernel_ids
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids


def _ray_failed_result(kernel_id: str, *, source_file: str = "/p/a.py") -> dict:
    return {
        "status": "failed",
        "kernel_id": kernel_id,
        "source_file": source_file,
        "error_class": "ray_transient",
        "error": "ray submission failed: LocalRayletDiedError",
        "attempts": [{
            "stdout_tail": "ray submission failed: LocalRayletDiedError: raylet died",
            "status": "failed",
        }],
        "proposal": {"decision": "REVERT", "reasons": [
            "backend dispatch failed: no usable backend attempt",
        ]},
    }


def test_record_kernel_opt_ray_transient_does_not_retire(state: SharedState):
    state.record_kernel_opt(_ray_failed_result("k001"))
    assert "k001" not in state.rejected_kernel_ids
    entry = state.kernel_opt_attempts["k001"]
    assert entry.get("ray_transient_failures") == 1
    assert entry.get("attempts", 0) == 0
    assert entry.get("failure_count", 0) == 0


def test_clear_ray_transient_kernel_rejections(state: SharedState):
    state.record_kernel_opt(_failed_result("k001", source_file="/p/a.py"))
    assert "k001" in state.rejected_kernel_ids
    state.kernel_opt_attempts["k001"]["last_ray_transient"] = True
    state.kernel_opt_attempts["k001"]["ray_transient_failures"] = 1
    cleared = state.clear_ray_transient_kernel_rejections()
    assert cleared == ["k001"]
    assert "k001" not in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["ray_transient_failures"] == 0


def test_record_kernel_opt_ray_transient_dispatch_revert_never_retires(state: SharedState):
    """Run13: 3x REVERT/micro=0 from Ray dispatch must not reject k001."""
    for _ in range(3):
        state.record_kernel_opt(_ray_failed_result("k001"))
    assert "k001" not in state.rejected_kernel_ids
    entry = state.kernel_opt_attempts["k001"]
    assert entry.get("ray_transient_failures") == 3
    assert entry.get("last_decision") == "REVERT"


def test_reset_ray_transient_kernel_counters(state: SharedState):
    state.record_kernel_opt(_ray_failed_result("k001"))
    assert state.kernel_opt_attempts["k001"]["ray_transient_failures"] == 1
    reset = state.reset_ray_transient_kernel_counters()
    assert reset == ["k001"]
    assert state.kernel_opt_attempts["k001"]["ray_transient_failures"] == 0


# ---------------------------------------------------------------------------
# PR-C: untried_hot_reusable_kernels report gate
# ---------------------------------------------------------------------------
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
    """Real numbers from /wekafs/users/.../20260522T164910Z/.../kernel_candidates.json.

    That session emitted ``report`` at tick=8 with zero kernel_opt
    attempts despite k001=23.7%, k002=37.3%, k004=9.7% all reusable.
    The new gate should report 3 untried hot kernels and block report.
    """
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
    # k001/k002/k004 must all be flagged untried -- log1 dropped these
    assert set(untried) >= {"k001", "k002", "k004"}
    # Sorted strongest-first (k002 37% > k001 24% > k004 9.7%)
    assert untried[0] == "k002"


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
    # Both kernels share a task_group -> only one slot reported
    assert len(untried) == 1
    # And it picks the highest-gpu_pct member (k002)
    assert untried[0] == "k002"


def test_untried_hot_kernels_skips_when_any_group_member_attempted(state: SharedState):
    """If k002 of group [k001,k002] has been attempted, k001 is also
    considered tried (same AST function -> same patch target)."""
    _set_trace(state, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ], task_groups=[
        {"primary_kernel_id": "k002", "kernel_ids": ["k001", "k002"]},
    ])
    # Mark k002 as attempted
    state.record_kernel_opt(_failed_result("k002", source_file="/p/moe_op.py"))
    untried = state.untried_hot_reusable_kernels()
    assert untried == []


def test_untried_hot_kernels_skips_when_source_file_integrated(state: SharedState):
    """Whole-file overwrite: once an integrate touches a source file,
    no further KEEP on the same file is meaningful."""
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
    assert untried == ["k009"]  # k001's whole file is integrated


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
