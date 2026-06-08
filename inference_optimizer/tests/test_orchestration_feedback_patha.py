"""Path A feedback-fidelity tests (A1 / A2 / A3).

Covers the structured inbox formatter, the ``get_recent_outcomes`` pull
tool + its Coordinator-backed reader, and the ``run_action_now`` inline
fast-action tool (whitelist gating, flag off, and the happy path with a
stub executor + ``delegated_result`` audit emission).
"""

from __future__ import annotations

import asyncio

import pytest

from inference_optimizer.orchestrator.backends import MockBackend, ScriptedPlan
from inference_optimizer.orchestrator.backends.mcp_context_tools import (
    CONTEXT_TOOL_NAMES,
    CONTEXT_TOOL_SPECS,
    ContextProvider,
)
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _first_present,
    _format_inbox_event,
)
from inference_optimizer.orchestrator.message_bus import Message
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.paths import make_session_dir


# ---------------------------------------------------------------------------
# A1 — structured inbox formatter
# ---------------------------------------------------------------------------
def _msg(topic: str, payload: dict, seq: int = 7) -> Message:
    m = Message.new("coordinator", "*", topic, payload)
    m.seq = seq
    return m


def test_first_present_picks_first_non_none():
    d = {"b": None, "c": 3}
    assert _first_present(d, ("a", "b", "c")) == 3
    assert _first_present(d, ("a", "b")) is None
    assert _first_present("not a dict", ("a",)) is None


def test_format_delegated_result_surfaces_high_signal_fields():
    line = _format_inbox_event(_msg(
        "delegated_result",
        {
            "task_id": "t1", "kind": "explore", "state": "succeeded",
            "result": {"status": "ok", "kept": True, "gain_pct": 12.5,
                       "tokens_per_s": 3400.0},
            "error": None,
        },
    ))
    assert "topic=delegated_result" in line
    assert "kind='explore'" in line
    assert "state='succeeded'" in line
    assert "status='ok'" in line
    assert "kept=True" in line
    assert "gain=12.5" in line
    assert "tput=3400.0" in line


def test_format_delegated_result_with_error_is_truncated():
    line = _format_inbox_event(_msg(
        "delegated_result",
        {"task_id": "t1", "kind": "explore", "state": "failed",
         "result": {}, "error": "boom " * 200},
    ))
    assert "state='failed'" in line
    assert "error=" in line
    # 200-char cap on the error string.
    assert len(line) < 600


def test_format_review_verdict_and_denial():
    verdict = _format_inbox_event(_msg(
        "review_verdict",
        {"target_proposal_msg_id": "p9", "verdict": "approve",
         "reasoning": "looks good"},
    ))
    assert "verdict='approve'" in verdict
    assert "target='p9'" in verdict

    denial = _format_inbox_event(_msg(
        "policy_denial",
        {"action_name": "kernel_opt", "rule": "phase_incompatible",
         "hint": "not allowed here"},
    ))
    assert "action='kernel_opt'" in denial
    assert "rule='phase_incompatible'" in denial


def test_format_unknown_topic_falls_back_to_payload_dump():
    line = _format_inbox_event(_msg("send_message", {"topic": "hello"}))
    assert "topic=send_message" in line
    assert "payload=" in line


# ---------------------------------------------------------------------------
# Shared coordinator fixture
# ---------------------------------------------------------------------------
def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "kernel": MockBackend(silent, name="k"),
            "critic": MockBackend(silent, name="c"),
            "robustness": MockBackend(silent, name="r"),
        },
    )


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# ---------------------------------------------------------------------------
# A2 — get_recent_outcomes tool + reader
# ---------------------------------------------------------------------------
def test_get_recent_outcomes_tool_is_registered():
    assert "get_recent_outcomes" in CONTEXT_TOOL_NAMES
    assert "run_action_now" in CONTEXT_TOOL_NAMES
    spec_methods = {spec[0]: spec[3] for spec in CONTEXT_TOOL_SPECS}
    assert spec_methods["get_recent_outcomes"] == "recent_outcomes"
    assert spec_methods["run_action_now"] == "run_action_now"


def test_context_provider_recent_outcomes_not_wired_message():
    provider = ContextProvider(shared_state=object())
    assert "not wired" in provider.recent_outcomes()


def test_context_provider_recent_outcomes_delegates_to_reader():
    provider = ContextProvider(
        shared_state=object(),
        recent_outcomes_reader=lambda k: f"outcomes(top_k={k})",
    )
    assert provider.recent_outcomes(5) == "outcomes(top_k=5)"


@pytest.mark.asyncio
async def test_recent_outcomes_reader_projects_delegated_results(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c.bus.append_and_seq(Message.new(
            "coordinator", "*", "delegated_result",
            {"task_id": "t1", "kind": "explore", "state": "succeeded",
             "result": {"status": "ok", "gain_pct": 4.0}, "error": None},
        ))
        await c.bus.append_and_seq(Message.new(
            "coordinator", "*", "observation", {"note": "ignored_obs"},
        ))
        out = c._context_recent_outcomes_reader(top_k=5)
        assert "Recent action outcomes" in out
        assert "kind='explore'" in out
        assert "gain=4.0" in out
        # Non-outcome topics are excluded (only delegated_result /
        # review_verdict are projected).
        assert "ignored_obs" not in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_recent_outcomes_reader_empty(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        assert "no recent outcomes" in c._context_recent_outcomes_reader()
    finally:
        await c.stop()


# ---------------------------------------------------------------------------
# A3 — run_action_now inline fast-action
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inline_whitelist_picks_lane_light_registered_actions(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        async def _stub(ctx: RunnerContext) -> dict:
            return {"status": "ok"}

        # target_analysis is lane-light in the registry; register an
        # executor so it qualifies for the whitelist.
        c.sub.register_executor("target_analysis", _stub)
        wl = c._inline_action_whitelist()
        assert "target_analysis" in wl
        # Heavy, lane-holding actions never qualify.
        assert "explore" not in wl
        assert "kernel_opt" not in wl
        # Deny-listed kinds are excluded even if lane-light.
        assert "report" not in wl
        assert "session_breakdown" not in wl
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_action_now_sync_disabled_by_flag(session_dir, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS", "0")
    c = _silent_coordinator(session_dir)
    try:
        out = c._run_action_now_sync("target_analysis", {})
        assert "disabled" in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_action_now_sync_rejects_non_whitelisted(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        # explore holds lanes -> never inline-eligible.
        out = c._run_action_now_sync("explore", {})
        assert "not inline-eligible" in out
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_action_now_happy_path_emits_delegated_result(
    session_dir, monkeypatch,
):
    c = _silent_coordinator(session_dir)
    try:
        ran: dict = {}

        async def _stub(ctx: RunnerContext) -> dict:
            ran["called"] = True
            return {"status": "ok", "gain_pct": 1.5}

        c.sub.register_executor("inline_probe", _stub)
        # Focus this test on the inline mechanics: stub the whitelist +
        # PolicyGate (both independently covered elsewhere).
        monkeypatch.setattr(
            c, "_inline_action_whitelist", lambda: frozenset({"inline_probe"}),
        )
        monkeypatch.setattr(c.policy, "validate_intent", lambda *a, **k: None)
        monkeypatch.setattr(
            c, "_sequence_denial_for_action", lambda *a, **k: None,
        )

        out = await c._run_action_now("inline_probe", {"p": 1})
        assert ran.get("called") is True
        assert "inline run complete" in out
        assert "state='succeeded'" in out
        assert "gain=1.5" in out

        # A delegated_result audit event was published with inline=True.
        events = await c.bus.tail(topic="delegated_result")
        assert events
        last = events[-1]
        assert last.payload.get("inline") is True
        assert last.payload.get("kind") == "inline_probe"
        assert last.payload.get("state") == "succeeded"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_action_now_sync_bridges_to_coordinator_loop(
    session_dir, monkeypatch,
):
    """The sync bridge marshals the coroutine onto the captured
    coordinator loop and returns its rendered result."""
    c = _silent_coordinator(session_dir)
    try:
        async def _stub(ctx: RunnerContext) -> dict:
            return {"status": "ok"}

        c.sub.register_executor("inline_probe", _stub)
        monkeypatch.setattr(
            c, "_inline_action_whitelist", lambda: frozenset({"inline_probe"}),
        )
        monkeypatch.setattr(c.policy, "validate_intent", lambda *a, **k: None)
        monkeypatch.setattr(
            c, "_sequence_denial_for_action", lambda *a, **k: None,
        )
        # Capture the running loop the way Coordinator.run() does.
        c._coordinator_loop = asyncio.get_running_loop()

        # Run the blocking sync bridge in a worker thread so it can wait
        # on the coordinator loop (this loop) via run_coroutine_threadsafe.
        out = await asyncio.to_thread(
            c._run_action_now_sync, "inline_probe", {},
        )
        assert "inline run complete" in out
        assert "state='succeeded'" in out
    finally:
        await c.stop()
