# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator hard-gate regression tests for the post-classify pipeline."""

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


# Fixtures
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
    """Open every earlier gate (incl. the target_baseline.json marker) to isolate the gate under test."""
    _write_baseline_json(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/x.json",
    }


# target_analysis is no longer sequence-gated (only baseline-first remains)
def test_baseline_allowed_without_target_analysis(session_dir):
    """``baseline`` is no longer blocked on a missing target_baseline.json."""
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    assert coord._sequence_denial_for_action("baseline") is None
    assert coord._sequence_denial_for_action("target_analysis") is None


def test_baseline_first_still_blocks_other_actions(session_dir):
    """With baseline_tput == 0, ``explore`` is denied for baseline (not target_analysis)."""
    coord = Coordinator(session_dir, backends=_backends_full())
    denied = coord._sequence_denial_for_action("explore")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "baseline must run first" in str(denied)
    assert coord._sequence_denial_for_action("baseline") is None
    assert coord._sequence_denial_for_action("target_analysis") is None


# integrate gate
def test_integrate_gate_inactive_without_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    assert coord._kernel_opt_keep_pending() == ""


def _seed_kernel_opt_state(coord, *, kernel_id: str, decision: str,
                            micro: float = 1.5,
                            source_file: str = "/p/dummy.py",
                            artifact: str = "/tmp/dummy.py") -> None:
    """Mimic the streaming-record write path (PR-B) so the integrate gate fires realistically."""
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
    assert coord._kernel_opt_keep_pending() == ""


def test_integrate_gate_clears_when_kernel_already_rejected(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(coord, kernel_id="k-bad", decision="KEEP")
    coord.shared_state.rejected_kernel_ids = ["k-bad"]
    assert coord._kernel_opt_keep_pending() == ""


def test_pending_keep_no_longer_blocks_other_actions(session_dir):
    """A pending kernel_opt KEEP no longer blocks explore / sweep; integrate is the LLM's call."""
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


# PR-C: hot-kernel report gate (reproduces log1 session 164910Z bug)
def _seed_trace_analyze(coord, *, hot_kernels, task_groups=None):
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "hot_kernels": hot_kernels,
        "task_groups": task_groups or [],
    }


def test_report_allowed_when_hot_reusable_kernels_untried(session_dir):
    """``report`` is allowed even with untried reusable hot kernels."""
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
    assert coord.shared_state.untried_hot_reusable_kernels()


def test_mission_summary_surfaces_untried_hot_kernels(session_dir):
    """The mission summary surfaces the untried reusable hot kernels as a neutral fact."""
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
    assert summary.find("k002") < summary.find("k001"), summary  # highest gpu_pct first


def test_report_always_allowed_regardless_of_hot_kernels(session_dir):
    """``report`` is never sequence-denied for hot-kernel reasons."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 31.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = 1
    assert coord._sequence_denial_for_action("report") is None


# trace_analyze gate — DEMOTED to a data-contract check in the handler.
def test_trace_analyze_gate_does_not_block_explore_actions(session_dir):
    """An empty ``last_trace_analyze`` cache must not action-layer-deny explore actions."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
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
    """A stale ``last_trace_analyze`` cache no longer pre-denies ``run_optimization``."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {}
    assert coord._sequence_denial_for_request("kernel", "run_optimization") is None


def test_trace_analyze_request_itself_passes(session_dir):
    """A ``trace_analyze`` REQUEST bypasses its own prerequisite check; baseline still applies."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_trace_analyze = {}
    assert coord._sequence_denial_for_request("kernel", "trace_analyze") is None


def test_run_optimization_handler_reports_missing_trace_analyze(session_dir):
    """No candidates_path + empty cache → handler returns ``missing_trace_analyze``."""
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
    """The pre-M4 ``select_kernels`` request kind was removed; ``get_handler`` returns None."""
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    from inference_optimizer.orchestrator.kernel_request_handlers import (
        get_handler,
    )
    assert get_handler("select_kernels") is None
    assert get_handler("trace_analyze") is not None


def test_trace_analyze_gate_clears_run_opt_request_when_cache_fresh(session_dir):
    """A fresh ``last_trace_analyze`` cache clears the request-layer gate."""
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
