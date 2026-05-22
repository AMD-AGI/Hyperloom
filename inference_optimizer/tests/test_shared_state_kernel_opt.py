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
