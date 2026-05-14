"""Coordinator hard-gate regression tests for the post-classify pipeline.

After the deletion of the in-loop ``setup`` / ``classify`` actions, three
extra Coordinator-enforced gates were introduced (see plan
``prep-actions-hard-gates``):

* ``target_analysis`` — fires only when ``--compare-against-gpu`` is set
  and ``$SESSION_DIR/target_analysis/target_baseline.json`` is missing.
* ``pmc_roofline`` — fires when ``last_profile_trace`` exists but
  ``last_profile_pmc_summary`` is empty.
* ``integrate`` — fires when ``last_kernel_opt.decision == "KEEP"`` and
  the kernel_id has not yet been integrated into ``optimization_stack``
  (and is not on ``rejected_kernel_ids``).

These tests exercise each gate's open / closed transitions plus the
matching ``_sequence_denial_for_action`` deny / allow pairs.
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
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied
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


def _seed_post_baseline(coord: Coordinator) -> None:
    """Open every earlier gate so the test can isolate the gate under test."""
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_profile_pmc_summary = "/tmp/pmc.json"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/x.json",
    }


def _write_baseline_json(session_dir: Path) -> Path:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return path


# ===========================================================================
# target_analysis gate
# ===========================================================================
def test_target_analysis_gate_inactive_when_compare_unset(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    # No --compare-against-gpu -> gate is dormant; baseline is still TODO 1
    todo = coord._required_next_step()
    assert "target_analysis" not in todo
    assert "baseline is required now" in todo


def test_target_analysis_gate_fires_when_compare_set_and_json_missing(session_dir):
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    todo = coord._required_next_step()
    assert "TODO 0/6" in todo
    assert "target_analysis is required now" in todo
    assert "b300" in todo


def test_target_analysis_gate_clears_after_baseline_json_written(session_dir):
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    _write_baseline_json(session_dir)
    todo = coord._required_next_step()
    assert "target_analysis" not in todo
    # baseline TODO takes precedence next.
    assert "baseline is required now" in todo


def test_target_analysis_denial_blocks_baseline_when_gate_open(session_dir):
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    denied = coord._sequence_denial_for_action("baseline")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "target_analysis must run first" in str(denied)
    # target_analysis itself is allowed through.
    assert coord._sequence_denial_for_action("target_analysis") is None


def test_target_analysis_denial_clears_after_baseline_json_written(session_dir):
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    _write_baseline_json(session_dir)
    # baseline gate now applies (baseline_tput is 0). target_analysis no
    # longer blocks; the next gate down (baseline) is what speaks up.
    denied = coord._sequence_denial_for_action("backends")
    assert isinstance(denied, PolicyDenied)
    assert "baseline must run first" in str(denied)
    # target_analysis -> not in sequence_actions if compare is unset, but
    # here it IS in the set; the gate has just been satisfied so it
    # should also pass through cleanly.
    assert coord._sequence_denial_for_action("baseline") is None


# ===========================================================================
# pmc_roofline gate
# ===========================================================================
def test_pmc_roofline_gate_inactive_without_profile_trace(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    coord.shared_state.baseline_tput = 100.0
    todo = coord._required_next_step()
    # profile is the next required step, not pmc_roofline.
    assert "TODO 2/6" in todo
    assert "profile is required now" in todo


def test_pmc_roofline_gate_fires_when_trace_set_pmc_missing(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    todo = coord._required_next_step()
    assert "TODO 3/6" in todo
    assert "pmc_roofline is required now" in todo


def test_pmc_roofline_gate_clears_when_pmc_summary_set(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_profile_pmc_summary = "/tmp/pmc.json"
    todo = coord._required_next_step()
    # pmc gate cleared; select_kernels is now the next required step.
    assert "pmc_roofline" not in todo
    assert "TODO 4/6" in todo
    assert "select_kernels is required now" in todo


def test_pmc_roofline_denial_blocks_explore_actions(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    for action in ("backends", "params", "sweep", "report", "profile"):
        denied = coord._sequence_denial_for_action(action)
        assert isinstance(denied, PolicyDenied), (
            f"{action!r} should be denied while pmc_roofline is required"
        )
        assert denied.rule == "execution_order"
        assert "pmc_roofline must run before" in str(denied)
    # pmc_roofline + validate_stack always allowed.
    assert coord._sequence_denial_for_action("pmc_roofline") is None
    assert coord._sequence_denial_for_action("validate_stack") is None


def test_pmc_roofline_gate_skipped_in_no_kernel_mode(session_dir):
    """When the kernel role is absent, the pmc gate must be inactive."""
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    role_registry = {
        k: v for k, v in default_role_registry().items() if k != "kernel"
    }
    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    backends = {
        "orchestration": MockBackend(silent, name="orch"),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }
    coord = Coordinator(
        session_dir, backends=backends, role_registry=role_registry,
    )
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    # pmc_roofline gate is kernel-only; without kernel role, no TODO.
    assert coord._required_next_step() == ""
    assert coord._sequence_denial_for_action("backends") is None


# ===========================================================================
# integrate gate
# ===========================================================================
def test_integrate_gate_inactive_without_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    # No last_kernel_opt -> gate closed.
    assert coord._kernel_opt_keep_pending() == ""
    assert coord._required_next_step() == ""


def test_integrate_gate_inactive_when_decision_not_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_kernel_opt = {
        "kernel_id": "k-1", "decision": "REVERT",
    }
    assert coord._kernel_opt_keep_pending() == ""


def test_integrate_gate_fires_when_keep_pending(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_kernel_opt = {
        "kernel_id": "k-rmsnorm", "decision": "KEEP",
    }
    assert coord._kernel_opt_keep_pending() == "k-rmsnorm"
    todo = coord._required_next_step()
    assert "TODO 5/6" in todo
    assert "integrate is required now" in todo
    assert "k-rmsnorm" in todo


def test_integrate_gate_clears_when_already_in_optimization_stack(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_kernel_opt = {
        "kernel_id": "k-rmsnorm", "decision": "KEEP",
    }
    coord.shared_state.optimization_stack = [
        {"action": "integrate", "kernel_id": "k-rmsnorm"},
    ]
    assert coord._kernel_opt_keep_pending() == ""
    # The integrate entry counts as an unvalidated KEEP -> validate_stack
    # gate fires next, not integrate.
    todo = coord._required_next_step()
    assert "integrate is required now" not in todo
    assert "validate_stack required" in todo


def test_integrate_gate_clears_when_kernel_already_rejected(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_kernel_opt = {
        "kernel_id": "k-bad", "decision": "KEEP",
    }
    coord.shared_state.rejected_kernel_ids = ["k-bad"]
    assert coord._kernel_opt_keep_pending() == ""
    assert coord._required_next_step() == ""


def test_integrate_denial_blocks_explore_but_allows_safe_actions(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    coord.shared_state.last_kernel_opt = {
        "kernel_id": "k-rmsnorm", "decision": "KEEP",
    }
    # Explore actions denied
    for action in ("backends", "params", "sweep"):
        denied = coord._sequence_denial_for_action(action)
        assert isinstance(denied, PolicyDenied), (
            f"{action!r} should be denied while integrate is required"
        )
        assert denied.rule == "execution_order"
        assert "integrate must run first" in str(denied)
        assert "k-rmsnorm" in (denied.hint or "")
    # integrate / validate_stack / report bypass the gate
    assert coord._sequence_denial_for_action("integrate") is None
    assert coord._sequence_denial_for_action("validate_stack") is None
    assert coord._sequence_denial_for_action("report") is None
