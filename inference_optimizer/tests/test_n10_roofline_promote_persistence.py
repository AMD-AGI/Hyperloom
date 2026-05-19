"""Roofline-v2 N10: Coordinator persists roofline executor SharedState mutations.

GPU-empirical post-fix (Qwen3-32B session 15:24-15:55 of 2026-05-19).

Bug observed: RooflineExecutor mutates `shared_state.last_profile_trace`
/ `last_profile_status` / `last_profile_args` / `last_trace_analyze`
inline during its sub-step orchestration. The mutations land on the
in-memory SharedState object (so the next-tick LLM prompt reflects
them), BUT `_promote_to_shared_state` previously had no `"roofline"`
branch — so `changed` stayed False and the standard
`if changed: self.shared_state.save(...)` tail never persisted the
mutations to `state.json`. Disk view stayed stale (snapshot_id=0,
analysis_md_len=0) while in-memory was correct.

N10 fix:
* `coordinator._promote_to_shared_state`: add `elif task_kind == "roofline":`
  branch that sets `audit_decision="promoted"` + `audit_extras` and
  flips `changed=True` to trigger the tail-save. Does NOT re-mutate
  any field — the executor remains the single writer.
* `coordinator._AUDIT_ACTIONS`: add "roofline".
* `shared_state._AUDIT_ACTIONS`: add "roofline".
* `shared_state._KEY_METRIC_MAP`: add ("roofline" → "snapshot_id").
* `SharedState`: add `last_roofline` dict field + `roofline_attempts`
  list field, mirroring the pattern used by every other audit action.

This test exercises the promote path with a synthesized roofline task
result + pre-set RooflineExecutor-style SharedState mutations to pin
that:

* `_promote_to_shared_state("roofline", ...)` triggers a save.
* `record_action_attempt("roofline", ...)` populates `last_roofline`
  + `roofline_attempts`.
* Default `roofline_attempts == []` / `last_roofline == {}` after
  construction (pin against accidental schema drift).
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
from inference_optimizer.orchestrator.coordinator import (
    _AUDIT_ACTIONS as COORDINATOR_AUDIT_ACTIONS,
    Coordinator,
)
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import (
    _AUDIT_ACTIONS as SHARED_STATE_AUDIT_ACTIONS,
    _KEY_METRIC_MAP,
    SharedState,
)
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


def _roofline_task(snapshot_id: int = 1) -> Task:
    return Task(
        task_id="t-rf-1",
        kind="roofline",
        state="running",
        params={"base_extra_args": "--mem-fraction-static=0.9"},
        idempotency_key=f"roofline-tick5-initial-{snapshot_id}",
    )


def _roofline_result(snapshot_id: int = 1) -> dict:
    """Mirrors what RooflineExecutor returns on success per design §8.4."""
    return {
        "status": "succeeded",
        "executed_at_iso": "2026-05-19T15:55:00+00:00",
        "snapshot_id": snapshot_id,
        "last_profile_trace": "/sessions/abc/runs/roofline/.../trace.json.gz",
        "analysis_md_path": "/sessions/abc/kernel-agent/runs/.../analysis.md",
        "profile_workspace": "/sessions/abc/runs/roofline/.../benchmark_sglang_...",
    }


# ---------------------------------------------------------------------------
# SharedState schema additions (default values + import surface)
# ---------------------------------------------------------------------------
def test_shared_state_has_roofline_audit_fields_by_default():
    """N10 prerequisite: SharedState must declare `last_roofline` +
    `roofline_attempts` so `record_action_attempt('roofline', ...)`
    has somewhere to write."""
    s = SharedState()
    assert hasattr(s, "last_roofline")
    assert hasattr(s, "roofline_attempts")
    assert s.last_roofline == {}
    assert s.roofline_attempts == []


def test_audit_actions_includes_roofline_in_both_modules():
    """Coordinator and SharedState audit sets must agree (the shared
    state _AUDIT_ACTIONS gates `record_action_attempt`; the coordinator
    set just documents the same membership inline for the audit
    bookkeeping branch)."""
    assert "roofline" in SHARED_STATE_AUDIT_ACTIONS
    assert "roofline" in COORDINATOR_AUDIT_ACTIONS


def test_key_metric_map_has_roofline_snapshot_id():
    """`record_action_attempt` reads the result dict via _KEY_METRIC_MAP
    to populate `key_metric` / `key_metric_kind` on each attempt; the
    natural progress metric for a composite roofline action is its
    `snapshot_id`."""
    assert "roofline" in _KEY_METRIC_MAP
    key, label = _KEY_METRIC_MAP["roofline"]
    assert key == "snapshot_id"
    assert label == "snapshot_id"


# ---------------------------------------------------------------------------
# record_action_attempt populates the rolling history
# ---------------------------------------------------------------------------
def test_record_action_attempt_populates_last_roofline_and_history():
    """Pin the `_AUDIT_ACTIONS` membership wires through the standard
    `record_action_attempt` recorder so the v1 audit-trail mechanism
    automatically covers roofline without bespoke code."""
    s = SharedState()
    s.record_action_attempt(
        action="roofline",
        task_id="t-rf-1",
        status="succeeded",
        decision="promoted",
        result={"snapshot_id": 3},
        extras={"analysis_md_path": "/p/analysis.md"},
    )
    assert s.last_roofline
    assert s.last_roofline.get("status") == "succeeded"
    assert s.last_roofline.get("decision") == "promoted"
    assert s.last_roofline.get("key_metric") == 3
    assert s.last_roofline.get("key_metric_kind") == "snapshot_id"
    # `extras` is a nested dict inside the attempt entry (v0 schema)
    assert s.last_roofline.get("extras", {}).get("analysis_md_path") == "/p/analysis.md"
    assert len(s.roofline_attempts) == 1


def test_record_action_attempt_caps_roofline_history():
    """Same `_DEFAULT_ATTEMPTS_HISTORY` cap (20) other audit actions
    have applies to roofline."""
    s = SharedState()
    for i in range(25):
        s.record_action_attempt(
            action="roofline", task_id=f"t-{i}",
            status="succeeded", decision="promoted",
            result={"snapshot_id": i},
        )
    assert len(s.roofline_attempts) == 20
    # Newest last (id 24)
    assert s.roofline_attempts[-1].get("task_id") == "t-24"


# ---------------------------------------------------------------------------
# Coordinator._promote_to_shared_state persists via tail-save
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_promote_roofline_flips_changed_and_saves(session_dir):
    """Bug N10 was: RooflineExecutor mutated shared_state inline,
    but _promote_to_shared_state had no `roofline` branch → changed
    stayed False → no save() → state.json stayed stale.

    This test simulates the executor mutations + invokes the promote
    path + asserts that state.json reflects the mutations after the
    call. Pre-N10 this assertion would fail (state.json snapshot_id=0
    while in-memory snapshot_id=1).
    """
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state

    # Simulate the inline mutations RooflineExecutor performs in
    # action_executors/roofline.py during its sub-step orchestration.
    s.last_profile_trace = "/sessions/abc/runs/roofline/.../trace.json.gz"
    s.last_profile_status = "succeeded"
    s.last_profile_args = "--mem-fraction-static=0.9"
    s.last_trace_analyze = {
        "trace_input": "/sessions/abc/runs/roofline/.../trace.json.gz",
        "analysis_md_path": "/sessions/abc/.../analysis.md",
        "analysis_md_text": "# Executive Summary\nCompute 51%\n",
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
    }

    # Save baseline state.json BEFORE promote to capture pre-N10 view
    s.save(session_dir)
    pre = json.loads((session_dir / "state.json").read_text())
    assert pre.get("last_profile_trace", "") == s.last_profile_trace
    # This works because s.save() now persists the in-memory mutations.

    # Now invoke promote — N10 branch should set changed=True and
    # the tail-save should run. Mutate something extra so we can tell
    # the call actually re-saved:
    s.last_profile_trace = "/sessions/abc/.../NEW_trace.gz"
    await coord._promote_to_shared_state(
        "roofline",
        _roofline_result(snapshot_id=1),
        task=_roofline_task(),
    )
    post = json.loads((session_dir / "state.json").read_text())
    assert post.get("last_profile_trace") == "/sessions/abc/.../NEW_trace.gz", (
        "N10 _promote_to_shared_state 'roofline' branch must trigger "
        "the tail-save; otherwise the post-promote state.json would "
        "still show the pre-mutate trace path"
    )


@pytest.mark.asyncio
async def test_promote_roofline_records_audit_attempt(session_dir):
    """The audit-bookkeeping tail of `_promote_to_shared_state` calls
    `record_action_attempt`. Pin that this path actually populates
    `last_roofline` + `roofline_attempts` (so verify_roofline_v2.py
    can `_count_action_attempts(state, 'roofline')` non-zero)."""
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state
    # Pre-set the cache so audit_extras has the snapshot_id to surface
    s.last_trace_analyze = {
        "analysis_md_path": "/p/analysis.md",
        "roofline_snapshot_id": 1,
    }
    s.last_profile_trace = "/t/trace.gz"

    assert s.roofline_attempts == []
    await coord._promote_to_shared_state(
        "roofline",
        _roofline_result(snapshot_id=1),
        task=_roofline_task(),
    )
    assert len(s.roofline_attempts) == 1, (
        "roofline must enter _AUDIT_ACTIONS so record_action_attempt fires"
    )
    attempt = s.roofline_attempts[-1]
    assert attempt.get("status") == "succeeded"
    assert attempt.get("decision") == "promoted"
    assert attempt.get("task_id") == "t-rf-1"
    # Audit extras land in the `extras` nested dict (v0 schema)
    extras = attempt.get("extras") or {}
    assert extras.get("snapshot_id") == 1
    assert extras.get("analysis_md_path") == "/p/analysis.md"


@pytest.mark.asyncio
async def test_promote_roofline_does_not_remutate_state(session_dir):
    """N10 promote branch is read-only over SharedState — it only
    flips `changed=True` and emits audit_extras snapshot. The
    RooflineExecutor remains the single writer. Specifically:
    last_profile_trace / last_trace_analyze.analysis_md_text must NOT
    be touched by the promote branch."""
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state
    s.last_profile_trace = "/before/trace.gz"
    s.last_trace_analyze = {
        "analysis_md_text": "before report",
        "roofline_snapshot_id": 3,
    }
    s.last_profile_status = "succeeded"

    await coord._promote_to_shared_state(
        "roofline",
        _roofline_result(snapshot_id=3),
        task=_roofline_task(snapshot_id=3),
    )
    # Same values after promote — branch must not re-mutate
    assert s.last_profile_trace == "/before/trace.gz"
    assert s.last_trace_analyze["analysis_md_text"] == "before report"
    assert s.last_trace_analyze["roofline_snapshot_id"] == 3
    assert s.last_profile_status == "succeeded"


@pytest.mark.asyncio
async def test_promote_roofline_non_dict_result_short_circuits(session_dir):
    """Defensive: non-dict result short-circuits without raising
    (mirrors the pre-existing branches in _promote_to_shared_state)."""
    coord = Coordinator(session_dir, backends=_silent_backends())
    # No exception; no audit entry either
    await coord._promote_to_shared_state(
        "roofline", None, task=_roofline_task(),  # type: ignore[arg-type]
    )
    assert coord.shared_state.roofline_attempts == []
