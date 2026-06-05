"""Coordinator hard-gate regression tests for the post-classify pipeline.

After the deletion of the in-loop ``setup`` / ``classify`` actions, two
Coordinator-enforced action-layer gates remain (see plan
``prep-actions-hard-gates`` and the follow-ups ``remove-pmc-hard-gate``
and ``remove-select-kernels-action-gate``):

* ``target_analysis`` — fires whenever
  ``$SESSION_DIR/target_analysis/target_baseline.json`` is missing
  (independent of ``--compare-against-gpu``; with the flag unset the
  executor still runs and writes a structured
  ``reason='no_target_gpu_configured'`` marker JSON to satisfy the gate).
* ``integrate`` — fires when ``last_kernel_opt.decision == "KEEP"`` and
  the kernel_id has not yet been integrated into ``optimization_stack``
  (and is not on ``rejected_kernel_ids``).

``trace_analyze`` no longer has a sequencing-policy gate at either
layer. The action-layer gate was removed first; the request-layer
gate on ``run_optimization`` has now been demoted to a data-contract
check inside ``run_optimization_handler`` (a missing / stale
candidates artifact surfaces as a structured handler error visible to
the LLM, not a policy pre-deny).

These tests exercise each remaining gate's open / closed transitions
plus the matching ``_sequence_denial_for_action`` deny / allow pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_intent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _backends_full() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


def _write_baseline_json(session_dir: Path) -> Path:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return path


def _seed_post_baseline(coord: Coordinator) -> None:
    """Open every earlier gate so the test can isolate the gate under test.

    Includes writing the target_baseline.json marker — the
    ``target_analysis`` gate now fires unconditionally and would otherwise
    mask the downstream gates these tests target. Also populates
    ``last_trace_analyze`` matching the trace so the analyze prerequisite
    is satisfied for tests targeting the integrate / stack gates.
    """
    _write_baseline_json(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/x.json",
    }


# ===========================================================================
# target_analysis is no longer sequence-gated (only baseline-first remains)
# ===========================================================================
def test_baseline_allowed_without_target_analysis(session_dir):
    """The target_analysis-first deny was removed: ``baseline`` is no
    longer blocked on a missing target_baseline.json."""
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    assert coord._sequence_denial_for_action("baseline") is None
    assert coord._sequence_denial_for_action("target_analysis") is None


def test_baseline_first_still_blocks_other_actions(session_dir):
    """The baseline-first invariant remains: with baseline_tput == 0 and
    no target_analysis on disk, ``explore`` is denied for baseline (not
    target_analysis)."""
    coord = Coordinator(session_dir, backends=_backends_full())
    denied = coord._sequence_denial_for_action("explore")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "baseline must run first" in str(denied)
    # baseline + target_analysis themselves pass.
    assert coord._sequence_denial_for_action("baseline") is None
    assert coord._sequence_denial_for_action("target_analysis") is None


# ===========================================================================
# integrate gate
# ===========================================================================
def test_integrate_gate_inactive_without_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    # No last_kernel_opt -> gate closed.
    assert coord._kernel_opt_keep_pending() == ""


def _seed_kernel_opt_state(coord, *, kernel_id: str, decision: str,
                            micro: float = 1.5,
                            source_file: str = "/p/dummy.py",
                            artifact: str = "/tmp/dummy.py") -> None:
    """Mimic the streaming-record write path (PR-B) so the integrate gate,
    which now reads ``kernel_opt_attempts`` via
    :meth:`SharedState.next_pending_keep_kernel_id`, fires from a
    realistic state shape (not just a bare ``last_kernel_opt`` stub).
    """
    coord.shared_state.record_kernel_opt({
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
    })


def test_integrate_gate_inactive_when_decision_not_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(coord, kernel_id="k-1", decision="REVERT")
    assert coord._kernel_opt_keep_pending() == ""


def test_integrate_gate_fires_when_keep_pending(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(
        coord, kernel_id="k-rmsnorm", decision="KEEP", micro=4.13,
        source_file="/sgl-workspace/aiter/aiter/ops/rmsnorm.py",
    )
    assert coord._kernel_opt_keep_pending() == "k-rmsnorm"


def test_integrate_gate_clears_when_already_in_optimization_stack(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(
        coord, kernel_id="k-rmsnorm", decision="KEEP",
        source_file="/p/rmsnorm.py",
    )
    coord.shared_state.optimization_stack = [
        {"action": "integrate", "kernel_id": "k-rmsnorm",
         "target_file": "/p/rmsnorm.py"},
    ]
    # The integrate KEEP is now on the stack, so the integrate gate
    # closes.
    assert coord._kernel_opt_keep_pending() == ""


def test_integrate_gate_clears_when_kernel_already_rejected(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(coord, kernel_id="k-bad", decision="KEEP")
    coord.shared_state.rejected_kernel_ids = ["k-bad"]
    assert coord._kernel_opt_keep_pending() == ""


def test_pending_keep_no_longer_blocks_other_actions(session_dir):
    """The KEEP-forces-integrate deny was removed: a pending kernel_opt
    KEEP no longer blocks explore / sweep. ``_kernel_opt_keep_pending``
    still reports the fact (surfaced in the report's completeness
    annotations), but integrate is now the LLM's call."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(
        coord, kernel_id="k-rmsnorm", decision="KEEP",
        source_file="/p/rmsnorm.py",
    )
    assert coord._kernel_opt_keep_pending() == "k-rmsnorm"
    for action in ("explore", "sweep", "integrate", "report"):
        assert coord._sequence_denial_for_action(action) is None, (
            f"{action!r} must not be sequence-denied by a pending KEEP"
        )


# ---------------------------------------------------------------------------
# PR-C: hot-kernel report gate (reproduces log1 session 164910Z bug)
# ---------------------------------------------------------------------------
def _seed_trace_analyze(coord, *, hot_kernels, task_groups=None):
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "hot_kernels": hot_kernels,
        "task_groups": task_groups or [],
    }


def test_report_allowed_when_hot_reusable_kernels_untried(session_dir):
    """The hot_kernel_unfinished report gate was removed: ``report`` is
    allowed even with untried reusable hot kernels. The report's
    completeness annotations surface the untried set instead of a deny."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 23.7, "reusable_native_kernel": True,
         "source_file": "/sgl/aiter/ops/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.3, "reusable_native_kernel": True,
         "source_file": "/sgl/aiter/ops/moe_op.py"},
    ])
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = 1
    coord.shared_state.explore_attempts = [{"variant_name": "x"}]
    assert coord._sequence_denial_for_action("report") is None
    # The fact is still observable for the report annotation.
    assert coord.shared_state.untried_hot_reusable_kernels()


def test_mission_summary_surfaces_untried_hot_kernels(session_dir):
    """The mission summary surfaces the untried reusable hot kernels as a
    neutral fact so Orchestration sees them without an enforced checklist.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    summary = coord.shared_state.to_mission_summary()
    assert "untried_hot_kernels" in summary
    assert "k001" in summary
    assert "k002" in summary
    # Highest gpu_pct first.
    assert summary.find("k002") < summary.find("k001"), summary


def test_report_always_allowed_regardless_of_hot_kernels(session_dir):
    """``report`` is never sequence-denied for hot-kernel reasons now —
    whether the kernels are reusable, sub-threshold, or untried."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 31.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = 1
    assert coord._sequence_denial_for_action("report") is None


# ===========================================================================
# trace_analyze gate — DEMOTED. Action-layer gate has been removed; the
# request-layer gate on ``run_optimization`` remains as the only enforcement
# point. Reverse regressions guard against the action-layer gate coming back
# AND assert the request-layer gate stays in place.
# ===========================================================================
def test_trace_analyze_gate_does_not_block_explore_actions(session_dir):
    """With ``last_profile_trace`` set and ``last_trace_analyze`` cache
    empty, ``_sequence_denial_for_action`` must NOT deny any explore
    action with ``trace_analyze must run first``."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    # Cache deliberately empty; would have triggered the old gate.
    s.last_trace_analyze = {}
    for action in ("explore", "sweep", "report", "profile", "roofline"):
        denied = coord._sequence_denial_for_action(action)
        if denied is None:
            continue
        assert "trace_analyze must run first" not in str(denied), (
            f"{action!r} hit the removed trace_analyze action-layer "
            f"gate: {denied!s}"
        )


def test_run_optimization_request_no_longer_blocked_by_stale_trace_analyze(
    session_dir,
):
    """The request-layer trace_analyze prerequisite was demoted: a stale
    ``last_trace_analyze`` cache no longer pre-denies ``run_optimization``
    at the policy boundary. The data dependency is now enforced inside
    ``run_optimization_handler`` so a missing candidates artifact returns
    a structured handler error rather than a sequence-denial loop."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {}
    assert coord._sequence_denial_for_request("kernel", "run_optimization") is None


def test_trace_analyze_request_itself_passes(session_dir):
    """``trace_analyze`` REQUEST itself bypasses
    ``_sequence_denial_for_request``'s prerequisite check (it IS the
    prerequisite). Only the baseline prerequisite still applies."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {}
    assert coord._sequence_denial_for_request("kernel", "trace_analyze") is None


def test_run_optimization_handler_reports_missing_trace_analyze(session_dir):
    """With no ``candidates_path`` in the payload and an empty
    ``last_trace_analyze`` cache, the handler returns a structured
    ``missing_trace_analyze`` error so the LLM can see why the request
    failed and re-run ``trace_analyze``."""
    import asyncio

    from inference_optimizer.orchestrator.kernel_request_handlers import (
        run_optimization_handler,
    )

    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {}
    s.save(session_dir)

    result = asyncio.run(
        run_optimization_handler({"kernel_id": "k001"}, session_dir=session_dir),
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_trace_analyze"


def test_legacy_select_kernels_request_kind_no_longer_recognised(session_dir):
    """The pre-M4 ``select_kernels`` request kind was removed in this
    branch (dispatch table + back-compat alias). The
    ``_sequence_denial_for_request`` carve-out for ``trace_analyze`` no
    longer applies to ``select_kernels``; since ``get_handler`` returns
    None for the dropped kind, the request short-circuits with no
    denial (no handler, no prerequisite chain to evaluate), and the
    LLM-side prompt template wording in
    ``inference_optimizer/orchestrator/system_prompts/`` is the
    canonical source of truth telling the model to emit
    ``kind='trace_analyze'`` instead."""
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    # The legacy kind has no handler now → request-prerequisite gate
    # short-circuits to None (the unknown-kind path is the LLM's
    # responsibility to avoid).
    from inference_optimizer.orchestrator.kernel_request_handlers import (
        get_handler,
    )
    assert get_handler("select_kernels") is None
    assert get_handler("trace_analyze") is not None


def test_trace_analyze_gate_clears_run_opt_request_when_cache_fresh(session_dir):
    """Once ``last_trace_analyze.trace_input`` matches the current
    ``last_profile_trace``, the request-layer gate clears and
    ``run_optimization`` is allowed through."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/cands.json",
    }
    assert coord._sequence_denial_for_request("kernel", "run_optimization") is None


def test_closing_phase_denies_non_report_proposals():
    state = SharedState(closing_phase=True)
    gate = PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state,
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
            ),
        )
    assert exc.value.rule == "closing_phase_only_report"

    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "report", "predicted_gain_pct": 0.0},
        ),
    )
