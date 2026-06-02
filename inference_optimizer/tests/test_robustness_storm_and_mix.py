"""PR-A8 (Arbor-into-Hyperloom): Robustness storm detector + intervention-mix.

The data primitives Robustness consumes to detect:

1. **Specialist storm**: many specialists dispatched in one EXPLORE
   entry with no actionable result. Tracked via
   ``SharedState.explore_specialist_dispatched_count`` (bumped per
   completed specialist task; reset on phase entry).
2. **Config-only streak**: consecutive EXPLORE rounds that only
   landed config tweaks (no code patches). Tracked via
   ``SharedState.consecutive_config_only_rounds`` (advanced on
   ``explore`` KEEPs, reset on ``integrate_patch`` KEEPs).
3. **Intervention-mix ledger**: per-KEEP entries of
   ``{change_type, action, task_id, delta_pct, ts}`` so Robustness
   can render an audit trail in its prompt context.

PR-A8's job is to land these primitives. The Robustness reactor
itself is a subprocess runtime (see ``robustness_agent.runtime.cli``)
that reads SharedState through its prompt — modifying that runtime
is out of scope for this PR; we only verify the read surface is
correctly populated.
"""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# 1. Intervention-mix ledger basics
# ---------------------------------------------------------------------------
def test_intervention_mix_ledger_appends_entries():
    s = SharedState()
    assert s.intervention_mix == []
    s.record_intervention(
        change_type="config",
        action="explore",
        task_id="t1",
        delta_pct=2.3,
    )
    assert len(s.intervention_mix) == 1
    entry = s.intervention_mix[0]
    assert entry["change_type"] == "config"
    assert entry["action"] == "explore"
    assert entry["task_id"] == "t1"
    assert entry["delta_pct"] == 2.3
    assert "ts" in entry


def test_intervention_mix_normalises_change_type_casing():
    s = SharedState()
    s.record_intervention(change_type="CONFIG", action="explore")
    s.record_intervention(change_type="  code_patch  ", action="integrate_patch")
    assert s.intervention_mix[0]["change_type"] == "config"
    assert s.intervention_mix[1]["change_type"] == "code_patch"


# ---------------------------------------------------------------------------
# 2. consecutive_config_only_rounds counter
# ---------------------------------------------------------------------------
def test_consecutive_config_only_advances_on_explore_keep():
    s = SharedState()
    assert s.consecutive_config_only_rounds == 0
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    assert s.consecutive_config_only_rounds == 1
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    assert s.consecutive_config_only_rounds == 2


def test_consecutive_config_only_resets_on_code_patch_keep():
    s = SharedState()
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="config", action="explore", task_id="t2")
    assert s.consecutive_config_only_rounds == 2
    s.record_intervention(
        change_type="code_patch", action="integrate_patch", task_id="t3",
    )
    assert s.consecutive_config_only_rounds == 0


def test_consecutive_config_only_ignores_unknown_change_type():
    """An unrecognised change_type leaves the counter unchanged."""
    s = SharedState()
    s.record_intervention(change_type="config", action="explore", task_id="t1")
    s.record_intervention(change_type="other", action="recover", task_id="t2")
    assert s.consecutive_config_only_rounds == 1


# ---------------------------------------------------------------------------
# 3. explore_specialist_dispatched_count
# ---------------------------------------------------------------------------
def test_specialist_dispatch_counter_increments():
    s = SharedState()
    assert s.explore_specialist_dispatched_count == 0
    assert s.bump_specialist_dispatched() == 1
    assert s.bump_specialist_dispatched(3) == 4


def test_specialist_dispatch_counter_resets():
    s = SharedState()
    s.bump_specialist_dispatched(5)
    s.reset_specialist_dispatched()
    assert s.explore_specialist_dispatched_count == 0


# ---------------------------------------------------------------------------
# 4. Coordinator hook: explore KEEP → config; integrate_patch kept → code_patch
# ---------------------------------------------------------------------------
def test_coordinator_intervention_hook_records_config_for_explore():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-config")
    task = Task(
        task_id="explore-1",
        kind="explore",
        state="succeeded",
        params={},
        idempotency_key="explore-1",
    )
    result = {
        "status": "succeeded",
        "winners": [{"name": "var1", "gain_pct": 3.5}],
        "best_variant": {"name": "var1", "gain_pct": 3.5},
    }
    c._record_intervention_for_task(task, result)
    assert len(c.shared_state.intervention_mix) == 1
    assert c.shared_state.intervention_mix[0]["change_type"] == "config"
    assert c.shared_state.consecutive_config_only_rounds == 1


def test_coordinator_intervention_hook_skips_explore_with_no_winners():
    """Explore round with empty winners + no best_variant is a no-keep
    round: the contiguous config-KEEP counter does NOT advance, but B2
    records a ``config_attempt`` so repeated fruitless config rounds can
    still drive the intervention-mix ESCALATION toward a code patch."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-skip")
    task = Task(
        task_id="explore-2", kind="explore", state="succeeded",
        params={}, idempotency_key="explore-2",
    )
    result = {"status": "succeeded", "winners": [], "best_variant": None}
    c._record_intervention_for_task(task, result)
    # B2: a no-keep config round is recorded as a config_attempt ...
    assert len(c.shared_state.intervention_mix) == 1
    assert c.shared_state.intervention_mix[0]["change_type"] == "config_attempt"
    # ... but the contiguous config-KEEP counter must NOT advance.
    assert c.shared_state.consecutive_config_only_rounds == 0


def test_coordinator_intervention_hook_records_code_patch_for_integrate_kept():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-code")
    # Prime with two config rounds to verify the reset.
    c.shared_state.record_intervention(change_type="config", action="explore")
    c.shared_state.record_intervention(change_type="config", action="explore")
    assert c.shared_state.consecutive_config_only_rounds == 2

    task = Task(
        task_id="ip-1", kind="integrate_patch", state="succeeded",
        params={}, idempotency_key="ip-1",
    )
    result = {
        "status": "kept",
        "delta_pct": 5.4,
        "patches_applied": ["patches/001.patch"],
    }
    c._record_intervention_for_task(task, result)
    assert c.shared_state.intervention_mix[-1]["change_type"] == "code_patch"
    assert c.shared_state.intervention_mix[-1]["delta_pct"] == 5.4
    assert c.shared_state.consecutive_config_only_rounds == 0


def test_coordinator_intervention_hook_skips_integrate_with_non_kept_status():
    """integrate_patch with status reverted / apply_failed must NOT
    advance any counter."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.task_registry import Task

    c = Coordinator.__new__(Coordinator)
    c.shared_state = SharedState(session_id="pr-a8-noop")
    task = Task(
        task_id="ip-2", kind="integrate_patch", state="succeeded",
        params={}, idempotency_key="ip-2",
    )
    for status in ("reverted", "apply_failed", "rejected_by_critic",
                    "applied_no_bench", "no_patches"):
        c._record_intervention_for_task(task, {"status": status})
    assert c.shared_state.intervention_mix == []
