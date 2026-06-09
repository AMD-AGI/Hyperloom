"""PART C — DEFER_TO_E2E routing + bounding in SharedState.

A ``DEFER_TO_E2E`` kernel_opt result (high-impact + correctness-passed kernel
whose micro-score was inconclusive/low) is routed to integrate through the
SAME promotion queue as KEEP so the integrate E2E ``gain_pct`` becomes the
authoritative KEEP/REVERT signal -- but BOUNDED by the existing
max_e2e_attempts (E2E) ledger so we never blow up expensive E2E runs. These
tests pin: DEFER is recorded (and never auto-rejected at the micro stage),
queued like KEEP, bounded by the integrate-attempt budget, protected from
clobber, and that normal KEEP / correctness-fail (REVERT) behaviour is intact.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


@pytest.fixture
def state() -> SharedState:
    return SharedState()


def _defer_result(kernel_id: str, *, micro: float = 1.0, source_file: str = "",
                  fallback: str = "REVERT", artifact: str = "") -> dict:
    """A kernel_optimization_handler result carrying a DEFER_TO_E2E proposal."""
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "source_file": source_file,
        "disposition": "DEFER_TO_E2E",
        "high_impact": True,
        "proposal": {
            "decision": "DEFER_TO_E2E",
            "fallback_decision": fallback,
            "defer_to_e2e": True,
            "reasons": ["inconclusive micro"],
        },
        "verification": {
            "micro_speedup": micro,
            "best_artifact_path": artifact,
            "compile_passed": True,
            "correctness_passed": True,
        },
    }


def _keep_result(kernel_id: str, *, micro: float, source_file: str = "") -> dict:
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "source_file": source_file,
        "proposal": {"decision": "KEEP", "reasons": []},
        "verification": {
            "micro_speedup": micro,
            "best_artifact_path": "",
            "compile_passed": True,
            "correctness_passed": True,
        },
    }


# ---------------------------------------------------------------------------
# DEFER is recorded + observable + NOT auto-rejected at the micro stage.
# ---------------------------------------------------------------------------
def test_defer_recorded_and_not_rejected(state: SharedState) -> None:
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    entry = state.kernel_opt_attempts["k001"]
    assert entry["last_decision"] == "DEFER_TO_E2E"
    assert entry["deferred_to_e2e"] is True
    assert entry["defer_fallback_decision"] == "REVERT"
    # A DEFER is NOT a micro-stage rejection (unlike REVERT).
    assert "k001" not in state.rejected_kernel_ids
    # last_kernel_opt reflects the deferred candidate.
    assert state.last_kernel_opt["kernel_id"] == "k001"
    assert state.last_kernel_opt["decision"] == "DEFER_TO_E2E"


# ---------------------------------------------------------------------------
# DEFER is routed to integrate via the KEEP promotion queue.
# ---------------------------------------------------------------------------
def test_defer_is_queued_for_integrate(state: SharedState) -> None:
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    assert state.next_pending_keep_kernel_id() == "k001"
    assert state.pending_keep_kernel_ids() == ["k001"]
    assert state.has_keep_pending_integrate is True


def test_keep_outranks_defer_then_both_drain(state: SharedState) -> None:
    """KEEP (higher micro) and DEFER coexist in the queue; KEEP is the
    stronger lever and drains first."""
    state.record_kernel_opt(_keep_result("k009", micro=4.0, source_file="/p/b.py"))
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    assert state.pending_keep_kernel_ids() == ["k009", "k001"]
    assert state.next_pending_keep_kernel_id() == "k009"
    # k009 integrated -> DEFER candidate is next.
    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k009", "target_file": "/p/b.py", "tput": 1.0,
    })
    assert state.next_pending_keep_kernel_id() == "k001"


# ---------------------------------------------------------------------------
# DEFER is BOUNDED by the existing max_e2e_attempts ledger.
# ---------------------------------------------------------------------------
def test_integrate_attempts_for_kernel_counts_across_patch_keys(state: SharedState) -> None:
    state.kernel_integrate_attempts["k001|/p/a.py|"] = {
        "kernel_id": "k001", "attempt_count": 2, "attempts": [{}, {}],
    }
    state.kernel_integrate_attempts["k001|/p/a.py|--flag"] = {
        "kernel_id": "k001", "attempt_count": 1, "attempts": [{}],
    }
    state.kernel_integrate_attempts["other|/p/c.py|"] = {
        "kernel_id": "other", "attempt_count": 9, "attempts": [],
    }
    assert state._integrate_attempts_for_kernel("k001") == 2
    assert state._integrate_attempts_for_kernel("missing") == 0


def test_defer_dropped_when_e2e_budget_exhausted(state: SharedState) -> None:
    """Once a DEFER candidate has spent its max_e2e_attempts (default 3) E2E
    runs without a KEEP, it drops out of the queue -> falls back."""
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    assert state.next_pending_keep_kernel_id() == "k001"  # budget available

    # 3 integrate attempts recorded => budget exhausted (default max = 3).
    state.kernel_integrate_attempts["k001|/p/a.py|"] = {
        "kernel_id": "k001", "attempt_count": 3, "attempts": [{}, {}, {}],
    }
    assert state._defer_e2e_budget_exhausted("k001") is True
    assert state.next_pending_keep_kernel_id() == ""
    assert state.pending_keep_kernel_ids() == []
    assert state.has_keep_pending_integrate is False


def test_defer_budget_below_max_still_queued(state: SharedState) -> None:
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    state.kernel_integrate_attempts["k001|/p/a.py|"] = {
        "kernel_id": "k001", "attempt_count": 2, "attempts": [{}, {}],
    }
    assert state._defer_e2e_budget_exhausted("k001") is False
    assert state.next_pending_keep_kernel_id() == "k001"


def test_defer_budget_env_override(state: SharedState, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_MAX_E2E_ATTEMPTS", "1")
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    state.kernel_integrate_attempts["k001|/p/a.py|"] = {
        "kernel_id": "k001", "attempt_count": 1, "attempts": [{}],
    }
    assert state._defer_e2e_budget_exhausted("k001") is True
    assert state.next_pending_keep_kernel_id() == ""


# ---------------------------------------------------------------------------
# DEFER does not break KEEP / REVERT behaviour.
# ---------------------------------------------------------------------------
def test_normal_keep_still_routes(state: SharedState) -> None:
    state.record_kernel_opt(_keep_result("k009", micro=4.0, source_file="/p/b.py"))
    assert state.next_pending_keep_kernel_id() == "k009"
    assert "k009" not in state.rejected_kernel_ids


def test_revert_still_rejected_and_not_queued(state: SharedState) -> None:
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k002", "source_file": "/p/x.py",
        "proposal": {"decision": "REVERT", "reasons": []},
        "verification": {"micro_speedup": 0.8, "compile_passed": True,
                         "correctness_passed": True},
    })
    assert "k002" in state.rejected_kernel_ids
    assert state.next_pending_keep_kernel_id() == ""


def test_pending_defer_survives_later_revert_sibling(state: SharedState) -> None:
    """A pending DEFER must not be clobbered in last_kernel_opt by a later
    REVERT on a different kernel (same protection a pending KEEP gets)."""
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    state.record_kernel_opt({
        "status": "ok", "kernel_id": "k002", "source_file": "/p/x.py",
        "proposal": {"decision": "REVERT", "reasons": []},
        "verification": {"micro_speedup": 0.8, "compile_passed": True,
                         "correctness_passed": True},
    })
    assert state.last_kernel_opt["kernel_id"] == "k001"
    assert state.last_kernel_opt["decision"] == "DEFER_TO_E2E"
    # the DEFER candidate is still queued for integrate.
    assert state.next_pending_keep_kernel_id() == "k001"


def test_defer_excluded_once_integrated(state: SharedState) -> None:
    state.record_kernel_opt(_defer_result("k001", micro=1.0, source_file="/p/a.py"))
    state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k001", "target_file": "/p/a.py", "tput": 1.0,
    })
    assert state.next_pending_keep_kernel_id() == ""
