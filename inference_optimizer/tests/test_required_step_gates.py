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

* ``pmc_roofline`` has been physically removed (PMC counter / rocprof
  collection is no longer part of the action catalogue — the F1
  composite ``roofline`` action superseded it). The reverse-regression
  tests below assert no leftover ``pmc_roofline must run before …``
  denial fires for any other action.
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
    ``last_select_kernels`` matching the trace so the P3 analyze gate
    (TODO 3/5) is open by default; tests targeting integrate / validate_stack
    don't care about the analyze gate and would otherwise be masked by it.
    """
    _write_baseline_json(coord.session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
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
    assert "TODO 0/5" in todo
    assert "target_analysis is required now" in todo
    assert "no_target_gpu_configured" in todo
    assert "baseline is required now" not in todo


def test_target_analysis_gate_fires_when_compare_set_and_json_missing(session_dir):
    coord = Coordinator(
        session_dir, backends=_backends_full(),
        compare_against_gpu="b300",
    )
    todo = coord._required_next_step()
    assert "TODO 0/5" in todo
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
    # KB_gaps/Dead-C — exercise the gate via the v0.8 canonical
    # ``explore`` action (the legacy ``backends`` name is denied earlier
    # at PolicyGate ``action_deprecated`` and never reaches this layer).
    denied = coord._sequence_denial_for_action("explore")
    assert isinstance(denied, PolicyDenied)
    assert "baseline must run first" in str(denied)
    # target_analysis -> the gate has just been satisfied so it should
    # also pass through cleanly.
    assert coord._sequence_denial_for_action("baseline") is None


# ===========================================================================
# pmc_roofline gate — REMOVED (the action itself was retired in favour
# of the F1 composite ``roofline``). The reverse regressions below
# guard against any leftover hard-gate ever coming back.
# ===========================================================================
def test_no_pmc_roofline_mention_in_required_next_step(session_dir):
    """``_required_next_step`` must never name the retired
    ``pmc_roofline`` action."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/x.json",
    }
    todo = coord._required_next_step()
    assert "pmc_roofline" not in todo
    assert todo == ""


def test_no_pmc_roofline_denial_for_any_action(session_dir):
    """No surviving rule may produce a denial whose message contains
    ``pmc_roofline must run before …``."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    for action in ("explore", "sweep", "report", "profile", "roofline"):
        denied = coord._sequence_denial_for_action(action)
        if denied is None:
            continue
        assert "pmc_roofline must run before" not in str(denied), (
            f"{action!r} still hits a removed pmc_roofline gate: {denied!s}"
        )


def test_no_pmc_roofline_action_in_sequence_actions(session_dir):
    """The Coordinator's ``sequence_actions`` set must not contain
    ``pmc_roofline``; otherwise an LLM proposing the retired name would
    receive a sequence denial instead of the canonical
    ``unknown_action`` PolicyGate denial.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    # Calling _sequence_denial_for_action with a name not in the set
    # short-circuits to ``None`` regardless of state.
    assert coord._sequence_denial_for_action("pmc_roofline") is None


# ===========================================================================
# integrate gate
# ===========================================================================
def test_integrate_gate_inactive_without_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    # No last_kernel_opt -> gate closed.
    assert coord._kernel_opt_keep_pending() == ""
    assert coord._required_next_step() == ""


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
    todo = coord._required_next_step()
    # P3 renumbered the integrate gate from 3/4 to 4/5 (analyze is the
    # new 3/5). `_seed_post_baseline` populates last_select_kernels so
    # the analyze gate is satisfied; the integrate gate is what fires.
    assert "TODO 4/5" in todo
    assert "integrate is required now" in todo
    assert "k-rmsnorm" in todo


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
    # The integrate entry counts as an unvalidated KEEP -> the
    # stack-rebench gate fires next, not integrate. v0.8 M3 +
    # KB_gaps/Gap-10: the rebench is inlined into ``explore``; the
    # TODO surfaces with that wording instead of the deprecated
    # ``validate_stack``.
    todo = coord._required_next_step()
    assert "integrate is required now" not in todo
    assert "stack rebench required" in todo


def test_integrate_gate_clears_when_kernel_already_rejected(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(coord, kernel_id="k-bad", decision="KEEP")
    coord.shared_state.rejected_kernel_ids = ["k-bad"]
    assert coord._kernel_opt_keep_pending() == ""
    assert coord._required_next_step() == ""


def test_integrate_denial_blocks_explore_but_allows_safe_actions(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_kernel_opt_state(
        coord, kernel_id="k-rmsnorm", decision="KEEP",
        source_file="/p/rmsnorm.py",
    )
    # v0.8 M3 / KB_gaps/Dead-C — the canonical EXPLORE-phase actions
    # (``explore`` + ``sweep``) must be denied while ``integrate`` is
    # required. Legacy ``backends`` / ``params`` / ``validate_stack`` are
    # already denied earlier at the PolicyGate ``action_deprecated`` rule
    # so they never reach this sequence gate.
    for action in ("explore", "sweep"):
        denied = coord._sequence_denial_for_action(action)
        assert isinstance(denied, PolicyDenied), (
            f"{action!r} should be denied while integrate is required"
        )
        assert denied.rule == "execution_order"
        assert "integrate must run first" in str(denied)
        assert "k-rmsnorm" in (denied.hint or "")
    # integrate / report bypass the integrate gate; legacy
    # ``validate_stack`` no longer appears in ``sequence_actions`` and
    # short-circuits early. ``report``'s own PR-C hot-kernel gate is
    # separately covered below.
    assert coord._sequence_denial_for_action("integrate") is None
    assert coord._sequence_denial_for_action("report") is None
    assert coord._sequence_denial_for_action("validate_stack") is None


# ---------------------------------------------------------------------------
# PR-C: hot-kernel report gate (reproduces log1 session 164910Z bug)
# ---------------------------------------------------------------------------
def _seed_trace_analyze(coord, *, hot_kernels, task_groups=None):
    coord.shared_state.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "hot_kernels": hot_kernels,
        "task_groups": task_groups or [],
    }


def test_report_denied_when_hot_reusable_kernels_untried(session_dir):
    """Repro of session 20260522T164910Z (log1): the LLM jumped
    straight to ``report`` despite k001=23.7% / k002=37.3% / k004=9.7%
    being reusable hot kernels with zero attempts. The new gate must
    deny report and surface the untried set in the hint.

    Requires N19c to be unlocked so the hot_kernel_unfinished rule
    activates -- simulated here via cheap-exhausted (last_delta < EPS).
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 23.7, "reusable_native_kernel": True,
         "source_file": "/sgl/aiter/ops/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.3, "reusable_native_kernel": True,
         "source_file": "/sgl/aiter/ops/moe_op.py"},
        {"kernel_id": "k004", "gpu_pct": 9.7, "reusable_native_kernel": True,
         "source_file": "/sgl/aiter/ops/rmsnorm.py"},
    ])
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = 1
    coord.shared_state.backends_attempts = [{"variant_name": "x"}]
    coord.shared_state.last_cheap_delta_gain = 0.05  # below EPS

    denied = coord._sequence_denial_for_action("report")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "hot_kernel_unfinished"
    # Hint must name the untried kernels so the LLM can act.
    assert "k001" in (denied.hint or "")
    assert "k002" in (denied.hint or "")
    assert "k004" in (denied.hint or "")


def test_report_allowed_after_every_hot_kernel_tried(session_dir):
    """Once every reusable hot kernel has either KEEP/REVERT/PARTIAL
    or has been retired via max_failures, the gate opens. Mix of
    integrate-stack + rejected_kernel_ids should both count as 'tried'.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    # k001 retired via max_failures.
    coord.shared_state.rejected_kernel_ids = ["k001"]
    coord.shared_state.kernel_opt_attempts = {
        "k001": {
            "attempts": 1, "failure_count": 1,
            "last_decision": "", "last_status": "failed",
        },
    }
    # k002 was integrated and validated (so the unvalidated-KEEPs
    # gate doesn't independently block report).
    coord.shared_state.optimization_stack.append({
        "action": "integrate", "kernel_id": "k002",
        "target_file": "/p/rmsnorm.py", "tput": 4500.0,
    })
    coord.shared_state.cumulative_gain_validated_stack_len = len(
        coord.shared_state.optimization_stack
    )

    # Hot-kernel gate must NOT fire (everything tried). Other gates
    # may still apply -- assert specifically on the rule we care about.
    denied = coord._sequence_denial_for_action("report")
    if denied is not None:
        assert denied.rule != "hot_kernel_unfinished", denied


def test_required_next_step_surfaces_untried_hot_kernels(session_dir):
    """``_required_next_step`` should also surface the TODO 4a line so
    Orchestration sees the explicit list when no KEEP is pending.

    Requires N19c unlocked (cheap exhausted) -- otherwise the TODO is
    intentionally hidden to avoid the death-spiral covered by the
    test below.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 24.0, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 37.0, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},
    ])
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = 1
    coord.shared_state.backends_attempts = [{"variant_name": "x"}]
    coord.shared_state.last_cheap_delta_gain = 0.05  # below EPS
    todo = coord._required_next_step()
    assert "TODO 4a/5" in todo
    # Highest gpu_pct first
    assert todo.find("k002") < todo.find("k001"), todo


def test_report_gate_inactive_when_no_reusable_hot_kernel_above_threshold(
    session_dir,
):
    """If the trace only has aten::mm or sub-3% rmsnorms, the gate
    correctly allows report -- nothing more for kernel_opt to do."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 15.0, "reusable_native_kernel": False,
         "source_file": "/aten/mm.py"},  # aten not reusable
        {"kernel_id": "k002", "gpu_pct": 2.5, "reusable_native_kernel": True,
         "source_file": "/p/rmsnorm.py"},  # below 3% threshold
    ])
    assert coord._sequence_denial_for_action("report") is None


# ---------------------------------------------------------------------------
# PR-C death-spiral guard: hot_kernel_unfinished must yield to N19c
# (reproduces session 20260523T014653Z bug)
# ---------------------------------------------------------------------------
def _seed_post_cheap_round(coord, *, snapshot_id=1, last_delta=0.77):
    """Mimic 'cheap action ran once'. On this branch
    ``_kernel_opt_unlocked`` reads the F3-5 ``gain_per_stack_entry``
    ledger (window=3, EPSILON=0.5%) instead of main's v0.6
    ``backends_attempts`` + ``last_cheap_delta_gain``, so we seed the
    last-3 deltas at ``last_delta`` and flip the
    ``gain_driven_kernel_opt`` toggle on.

    With ``last_delta=0.77`` (>= 0.5) the gate is CLOSED; with
    ``last_delta=0.1`` (< 0.5) the gate is OPEN.
    """
    coord.shared_state.last_trace_analyze = dict(
        coord.shared_state.last_trace_analyze or {}
    )
    coord.shared_state.last_trace_analyze["roofline_snapshot_id"] = snapshot_id
    coord.shared_state.gain_driven_kernel_opt = True
    coord.shared_state.gain_per_stack_entry = [
        {"delta_pct": float(last_delta)} for _ in range(3)
    ]


def test_report_gate_yields_when_kernel_opt_locked_by_n19c(session_dir):
    """20260523T014653Z death-spiral repro:

    1. Cheap round produced +0.77% delta (> EPSILON=0.3%)
       -> N19c locks kernel_opt
    2. PR-C's untried_hot_reusable_kernels has k001/k002/k005/k008
       -> hot_kernel_unfinished previously denied `report`
    3. LLM tried run_optimization -> N19c denied execution_order
    4. 10 consecutive execution_order denials -> policy_loop auto-stop

    Fix: report's hot_kernel_unfinished rule must yield when
    ``_kernel_opt_unlocked()`` returns False. The LLM can then either
    propose another cheap round (preferred -- cheap is still earning)
    or emit ``report`` if no cheap actions are left.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 31.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
        {"kernel_id": "k002", "gpu_pct": 47.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    _seed_post_cheap_round(coord, snapshot_id=1, last_delta=0.77)

    # N19c is locked (cheap still earning)
    assert coord._kernel_opt_unlocked() is False
    # ... so hot_kernel_unfinished must NOT deny report
    denied = coord._sequence_denial_for_action("report")
    if denied is not None:
        assert denied.rule != "hot_kernel_unfinished", denied


def test_required_next_step_hides_todo_4a_when_kernel_opt_locked(session_dir):
    """Symmetric to the report-gate yield: TODO 4a must not push the
    LLM toward `run_optimization` if N19c will just reject it."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 31.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    _seed_post_cheap_round(coord, snapshot_id=1, last_delta=0.77)

    todo = coord._required_next_step()
    assert "TODO 4a" not in todo, (
        f"TODO 4a leaked while N19c locks kernel_opt: {todo!r}"
    )


def test_report_gate_fires_again_after_cheap_exhausts(session_dir):
    """Once cheap delta falls below EPSILON, N19c unlocks and PR-C
    re-activates -- gate must fire again."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 31.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    # First: cheap still earning -> gate yields
    _seed_post_cheap_round(coord, snapshot_id=1, last_delta=0.77)
    denied_1 = coord._sequence_denial_for_action("report")
    assert denied_1 is None or denied_1.rule != "hot_kernel_unfinished"

    # Then: cheap exhausted (delta drops below F3-5 epsilon=0.5%) ->
    # gate fires
    coord.shared_state.gain_per_stack_entry = [
        {"delta_pct": 0.1} for _ in range(3)
    ]
    assert coord._kernel_opt_unlocked() is True
    denied_2 = coord._sequence_denial_for_action("report")
    assert isinstance(denied_2, PolicyDenied)
    assert denied_2.rule == "hot_kernel_unfinished"


def test_report_gate_active_with_escape_hatch(session_dir, monkeypatch):
    """ALLOW_EARLY_KERNEL_OPT bypasses N19c -> kernel_opt is
    immediately dispatchable -> hot_kernel_unfinished fires even
    without a prior cheap round."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "1")
    coord = Coordinator(session_dir, backends=_backends_full())
    _seed_post_baseline(coord)
    _seed_trace_analyze(coord, hot_kernels=[
        {"kernel_id": "k001", "gpu_pct": 31.9, "reusable_native_kernel": True,
         "source_file": "/p/moe_op.py"},
    ])
    # No cheap round at all; escape hatch unlocks kernel_opt.
    assert coord._kernel_opt_unlocked() is True
    denied = coord._sequence_denial_for_action("report")
    assert isinstance(denied, PolicyDenied)
    assert denied.rule == "hot_kernel_unfinished"


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
    for action in ("explore", "sweep", "report", "profile", "roofline"):
        denied = coord._sequence_denial_for_action(action)
        if denied is None:
            continue
        assert "select_kernels must run first" not in str(denied), (
            f"{action!r} hit the removed select_kernels action-layer "
            f"gate: {denied!s}"
        )


def test_analyze_gate_surfaces_select_kernels_todo_when_cache_stale(session_dir):
    """When ``last_profile_trace`` is set but ``last_select_kernels`` is
    empty/stale, ``_required_next_step()`` surfaces a TODO 3/5 (analyze)
    guidance prompt telling the LLM to emit a `select_kernels` REQUEST
    before any kernel_opt cycle.

    NB: This is a GUIDANCE-only gate. ``_sequence_denial_for_action``
    still does NOT block explore actions (params/backends/sweep) on a
    stale cache — see ``test_select_kernels_gate_does_not_block_explore_actions``
    which remains valid. The TODO only adds an LLM-visible prompt; it
    does not add a new action-layer denial.
    """
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {}
    todo = coord._required_next_step()
    assert "TODO 3/5" in todo
    assert "analyze is required now" in todo
    assert "select_kernels" in todo
    assert "trace_input" in todo


def test_analyze_gate_clears_when_cache_matches_trace(session_dir):
    """Once ``last_select_kernels.trace_input`` matches the current
    ``last_profile_trace``, the P3 analyze gate clears and the chain
    falls through to the next guard (integrate / validate_stack /
    empty)."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/cands.json",
    }
    todo = coord._required_next_step()
    assert "analyze is required now" not in todo
    # No other gate is open in this state, so the chain is empty.
    assert todo == ""


def test_analyze_gate_clears_when_only_trace_analyze_populated(session_dir):
    """Repro: M4 renamed the cache field ``last_select_kernels`` ->
    ``last_trace_analyze``. The production ``select_kernels`` handler
    populates ONLY ``last_trace_analyze`` (canonical), leaving the legacy
    ``last_select_kernels`` empty. The TODO 3/5 analyze gate MUST read the
    canonical field so it clears after select_kernels runs; otherwise the
    orchestration loops re-emitting ``select_kernels`` forever and
    ``kernel_opt`` never fires (observed: every session stalled at
    kernel_opt_attempts=0)."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {}  # legacy field stays empty post-M4
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "candidates_path": "/tmp/cands.json",
        "reusable_native_kernel_ids": ["k001", "k002"],
    }
    todo = coord._required_next_step()
    assert "analyze is required now" not in todo, (
        "TODO 3/5 select_kernels loop: the analyze gate read the legacy "
        f"last_select_kernels instead of canonical last_trace_analyze: {todo!r}"
    )


def test_all_reusable_kernels_rejected_reads_canonical_trace_analyze(session_dir):
    """Companion to the analyze-gate fix: ``_all_reusable_kernels_rejected``
    must read the canonical ``last_trace_analyze`` cache. Reading the legacy
    ``last_select_kernels`` (empty post-M4) made it return True (vacuously
    'all rejected'), which would let the KERNEL phase wind down / skip
    kernel_opt entirely once the select_kernels TODO loop was fixed."""
    coord = Coordinator(session_dir, backends=_backends_full())
    _write_baseline_json(session_dir)
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {}  # legacy field empty post-M4
    s.last_trace_analyze = {
        "trace_input": "/tmp/profile.tar.gz",
        "reusable_native_kernel_ids": ["k001", "k002"],
    }
    # No kernel rejected yet -> NOT all rejected (k001/k002 still tryable).
    assert coord._all_reusable_kernels_rejected() is False
    # After both rejected -> all rejected.
    s.rejected_kernel_ids = ["k001", "k002"]
    assert coord._all_reusable_kernels_rejected() is True


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
    # Main M4 renamed the prerequisite from ``select_kernels`` to
    # ``trace_analyze``; both names refer to the same request kind
    # via the back-compat alias, so accept either wording.
    assert (
        "trace_analyze must run first" in str(denied)
        or "select_kernels must run first" in str(denied)
    )


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
