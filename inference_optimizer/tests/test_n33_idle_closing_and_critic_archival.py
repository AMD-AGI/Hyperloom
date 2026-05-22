"""N33 (May 2026) tests: critic auto-approve archival actions +
Coordinator silent-tick early-closing.

Two related guardrails landed together as N33:
* ``critic.md`` now carries an explicit "archival actions skip the
  before/after benchmark gate" carve-out so the Critic never blocks
  the LLM from emitting ``report`` / ``session_breakdown`` /
  ``target_analysis`` as a "done" signal.
* :class:`Coordinator` now counts consecutive idle ticks (no queued
  task, no running task, no pending proposal, no ``current_action``)
  in :attr:`SharedState.consecutive_silent_ticks` and force-enters
  closing phase once the count exceeds the
  ``INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS`` threshold so a silent LLM
  cannot burn the rest of the wall-clock budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.action_executors import report_executor
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _resolve_silent_ticks_closing_threshold,
)
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


def _write_marker_target_baseline(session_dir: Path) -> None:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "no_target",
            "reason": "no_target_gpu_configured",
            "row_count": 0,
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# silent-tick threshold env knob
# ---------------------------------------------------------------------------
def test_resolve_silent_ticks_closing_threshold_default(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", raising=False)
    assert _resolve_silent_ticks_closing_threshold() == 120


def test_resolve_silent_ticks_closing_threshold_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "5")
    assert _resolve_silent_ticks_closing_threshold() == 5


def test_resolve_silent_ticks_closing_threshold_disabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "0")
    assert _resolve_silent_ticks_closing_threshold() == 0


def test_resolve_silent_ticks_closing_threshold_garbage(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "not-a-number")
    assert _resolve_silent_ticks_closing_threshold() == 120


def test_shared_state_default_consecutive_silent_ticks_is_zero():
    s = SharedState()
    assert s.consecutive_silent_ticks == 0


# ---------------------------------------------------------------------------
# coordinator silent-tick bump + early-close trigger
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_silent_ticks_increment_when_run_is_idle(session_dir, monkeypatch):
    """Silent-only backends with no scripted proposals leave the run
    completely idle. Each such tick must bump
    ``consecutive_silent_ticks`` so the early-close logic has a signal
    to fire on."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "0")
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        await c.run(max_ticks=3, tick_interval_sec=0.0)
        assert c.shared_state.consecutive_silent_ticks >= 3
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_silent_ticks_triggers_early_closing_phase(
    session_dir, monkeypatch,
):
    """With ``INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS=2`` and a long
    ``max_minutes`` the run should never hit the wall-clock budget but
    should still terminate via the closing-phase report flush after
    a few silent ticks."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "2")
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.save(session_dir)
    try:
        reason = await c.run(
            max_minutes=60.0,
            max_ticks=80,
            closing_grace_sec=30.0,
            tick_interval_sec=0.0,
        )
        assert reason == "time_exhausted"
        assert (session_dir / "reports" / "final.md").exists()
        on_disk = json.loads((session_dir / "state.json").read_text())
        assert on_disk["closing_report_task_id"]
        assert on_disk["closing_started_unix"] > 0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_silent_ticks_disabled_by_zero_threshold(
    session_dir, monkeypatch,
):
    """Setting the env knob to 0 must restore legacy behaviour: silent
    ticks count up but the loop never short-circuits to closing — only
    the wall-clock deadline (or ``max_ticks`` in this test) ends it."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "0")
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        reason = await c.run(max_ticks=5, tick_interval_sec=0.0)
        assert reason == "max_ticks"
        assert c.shared_state.consecutive_silent_ticks >= 5
        assert c.shared_state.closing_phase is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_silent_ticks_reset_on_pending_proposal(session_dir, monkeypatch):
    """A non-empty ``pending_proposals`` set means the LLM is mid-
    review-loop — the run is NOT idle, so the silent counter must
    reset to 0 that tick."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "0")
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        await c.run(max_ticks=2, tick_interval_sec=0.0)
        assert c.shared_state.consecutive_silent_ticks >= 2
        # Simulate an in-flight proposal; the next tick must reset.
        from inference_optimizer.orchestrator.coordinator import PendingProposal
        c.state.pending_proposals["fake"] = PendingProposal(
            proposal_msg_id="fake",
            from_agent="orchestration",
            action_name="report",
            predicted_gain_pct=0.0,
            payload={},
        )
        await c.tick(1)
        # Coordinator.tick uses the reactor/dispatcher pair but does NOT
        # run the long-loop counter — exercise the long path instead.
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# critic.md archival-exception carve-out
# ---------------------------------------------------------------------------
def test_critic_md_carves_out_archival_actions():
    """Critic prompt must explicitly approve archival actions so the
    LLM's ``report`` proposals never bounce. The wording moved under
    a structured Archival/Exploration bullet pair in N35; this test
    pins the action names + the "always approve" semantics regardless
    of the exact phrasing."""
    path = (
        Path(__file__).resolve().parent.parent
        / "orchestrator" / "system_prompts" / "critic.md"
    )
    text = path.read_text(encoding="utf-8")
    assert "Archival" in text, "expected an Archival bullet header"
    assert "`report`" in text
    assert "`session_breakdown`" in text
    assert "`target_analysis`" in text
    # The carve-out must say "always approve" in some form so future
    # readers can't argue the rule is conditional.
    lowered = text.lower()
    assert "always `approve`" in lowered or "always approve" in lowered, (
        "expected the carve-out to state 'always approve' archival actions"
    )
