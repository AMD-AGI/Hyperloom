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

Two former gates have been demoted; this file holds reverse regressions
for both:

* ``pmc_roofline`` is now opt-in advisory enrichment for ``kernel_opt``
  only and never blocks any other action.
* ``select_kernels`` is now enforced ONLY at the REQUEST layer for
  ``run_optimization`` (see ``_sequence_denial_for_request``). Action-
  layer explore actions (``params`` / ``backends`` / ``sweep`` /
  ``report``) are never gated on a fresh ``last_select_kernels`` cache.

These tests exercise each remaining gate's open / closed transitions
plus the matching ``_sequence_denial_for_action`` deny / allow pairs,
and the reverse regressions for the two demoted gates.
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


def _write_baseline_json(session_dir: Path) -> Path:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return path


def _seed_post_baseline(coord: Coordinator) -> None:
    """Open every earlier gate so the test can isolate the gate under test.

    Includes writing the target_baseline.json marker — the
    ``target_analysis`` gate now fires unconditionally and would otherwise
    mask the downstream gates these tests target.
    """
    _write_baseline_json(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_profile_pmc_summary = "/tmp/pmc.json"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/x.json",
    }


# ===========================================================================
# target_analysis gate
# ===========================================================================
def test_target_analysis_gate_fires_when_compare_unset_and_json_missing(session_dir):
    """Gate fires unconditionally on missing target_baseline.json — even
    without --compare-against-gpu — because the executor still runs and
    writes a 'no_target_gpu_configured' marker JSON."""
    coord = Coordinator(session_dir, backends=_backends_full())
    todo = coord._required_next_step()
    assert "TODO 0/4" in todo
    assert "target_analysis is required now" in todo
    assert "no_target_gpu_configured" in todo
    assert "baseline is required now" not in todo


def test_target_analysis_gate_fires_when_compare_set_and_json_missing(session_dir):
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    todo = coord._required_next_step()
    assert "TODO 0/4" in todo
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


def test_target_analysis_gate_clears_after_marker_json_written_unset(session_dir):
    """A 'no_target_gpu_configured' marker JSON is enough to satisfy the
    gate; the executor writes one even without --compare-against-gpu."""
    coord = Coordinator(session_dir, backends=_backends_full())
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "skipped",
            "reason": "no_target_gpu_configured",
        }),
        encoding="utf-8",
    )
    todo = coord._required_next_step()
    assert "target_analysis is required now" not in todo
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


def test_target_analysis_denial_blocks_baseline_when_compare_unset(session_dir):
    """Without --compare-against-gpu the denial still fires; the hint
    just changes to mention the marker JSON."""
    coord = Coordinator(session_dir, backends=_backends_full())
    denied = coord._sequence_denial_for_action("baseline")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "target_analysis must run first" in str(denied)
    assert "no_target_gpu_configured" in (denied.hint or "")
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
    # target_analysis -> the gate has just been satisfied so it should
    # also pass through cleanly.
    assert coord._sequence_denial_for_action("baseline") is None


# ===========================================================================
# pmc_roofline gate — REMOVED. PMC is now opt-in advisory enrichment for
# `kernel_opt` and never gates any other action. The tests below are
# reverse regressions guarding against the gate ever coming back.
# ===========================================================================
def test_pmc_roofline_gate_does_not_fire_when_pmc_missing(session_dir):
    """With ``last_profile_trace`` set and ``last_profile_pmc_summary``
    empty, ``_required_next_step()`` must NOT mention ``pmc_roofline``.
    The PMC hard-gate has been removed. With no ``kernel_opt`` KEEP
    pending and no unvalidated stack KEEPs, the chain has reached its
    end and the required step is empty."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    todo = coord._required_next_step()
    assert "pmc_roofline" not in todo
    assert "select_kernels" not in todo
    assert todo == ""


def test_pmc_roofline_gate_does_not_block_explore_actions(session_dir):
    """``_sequence_denial_for_action`` must not deny any action with the
    reason ``pmc_roofline must run before ...``. With both the PMC and
    the action-layer ``select_kernels`` hard-gates removed, no explore
    action should be denied at all when only the cache is empty."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    for action in ("backends", "params", "sweep", "report", "profile",
                   "pmc_roofline", "validate_stack"):
        denied = coord._sequence_denial_for_action(action)
        if denied is None:
            continue
        assert "pmc_roofline must run before" not in str(denied), (
            f"{action!r} hit the removed PMC hard-gate: {denied!s}"
        )


def test_pmc_summary_present_does_not_change_required_next_step(session_dir):
    """Sanity: setting ``last_profile_pmc_summary`` to a value must NOT
    change ``_required_next_step()`` because PMC is no longer part of
    the TODO chain. The chain is empty either way."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    todo_without_pmc = coord._required_next_step()
    s.last_profile_pmc_summary = "/tmp/pmc.json"
    todo_with_pmc = coord._required_next_step()
    assert todo_without_pmc == todo_with_pmc
    assert todo_with_pmc == ""


def test_pmc_roofline_gate_skipped_in_no_kernel_mode(session_dir):
    """In no-kernel mode the entire kernel-pipeline section of TODOs is
    skipped, so ``_required_next_step`` is empty and explore actions
    pass through ``_sequence_denial_for_action`` with no PMC mention.
    This is a regression that should hold both before and after the
    PMC hard-gate removal."""
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
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
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
    assert "TODO 3/4" in todo
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


# ===========================================================================
# select_kernels gate — DEMOTED. Action-layer gate has been removed; the
# request-layer gate on ``run_optimization`` remains as the only enforcement
# point. Reverse regressions guard against the action-layer gate coming back
# AND assert the request-layer gate stays in place.
# ===========================================================================
def test_select_kernels_gate_does_not_block_explore_actions(session_dir):
    """With ``last_profile_trace`` set and ``last_select_kernels`` cache
    empty, ``_sequence_denial_for_action`` must NOT deny any explore
    action with ``select_kernels must run first``."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    # Cache deliberately empty; would have triggered the old gate.
    s.last_select_kernels = {}
    for action in ("backends", "params", "sweep", "report", "profile",
                   "pmc_roofline", "validate_stack"):
        denied = coord._sequence_denial_for_action(action)
        if denied is None:
            continue
        assert "select_kernels must run first" not in str(denied), (
            f"{action!r} hit the removed select_kernels action-layer "
            f"gate: {denied!s}"
        )


def test_select_kernels_gate_does_not_appear_in_required_next_step(session_dir):
    """``_required_next_step()`` must not surface a select_kernels TODO
    even when ``last_select_kernels`` is stale."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {}
    todo = coord._required_next_step()
    assert "select_kernels" not in todo
    # No other gate is open in this state, so the chain is empty.
    assert todo == ""


def test_select_kernels_gate_still_blocks_run_optimization_request(session_dir):
    """The request-layer gate on ``run_optimization`` MUST still fire
    when ``last_select_kernels`` is stale. This is the sole remaining
    enforcement point that keeps ``kernel_opt`` from running without
    candidates."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {}
    denied = coord._sequence_denial_for_request("kernel", "run_optimization")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "execution_order"
    assert "select_kernels must run first" in str(denied)


def test_select_kernels_request_itself_passes(session_dir):
    """``select_kernels`` REQUEST itself bypasses
    ``_sequence_denial_for_request``'s prerequisite check (it IS the
    prerequisite). Baseline + profile prerequisites still apply."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {}
    assert coord._sequence_denial_for_request("kernel", "select_kernels") is None


def test_select_kernels_gate_clears_run_opt_request_when_cache_fresh(session_dir):
    """Once ``last_select_kernels.trace_input`` matches the current
    ``last_profile_trace``, the request-layer gate clears and
    ``run_optimization`` is allowed through."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/cands.json",
    }
    assert coord._sequence_denial_for_request("kernel", "run_optimization") is None
