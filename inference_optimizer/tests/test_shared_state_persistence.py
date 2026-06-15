# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P1-3 SharedState + Coordinator integration tests."""

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
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_full() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


# SharedState unit
def test_shared_state_defaults_blank():
    s = SharedState()
    assert s.session_id == ""
    assert s.baseline_tput == 0.0
    assert s.cumulative_gain == 0.0
    assert s.crash_count == 0
    assert s.pruned_families == []
    assert s.current_best == {}


def test_save_load_round_trip(tmp_path):
    s = SharedState(
        session_id="abc",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        baseline_tput=1840.0,
        cumulative_gain=12.5,
        pruned_families=["deep_kernel"],
        current_best={"action": "backends", "tput": 2010.0},
    )
    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert s2.session_id == "abc"
    assert s2.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert s2.baseline_tput == 1840.0
    assert s2.cumulative_gain == 12.5
    assert s2.pruned_families == ["deep_kernel"]
    assert s2.current_best == {"action": "backends", "tput": 2010.0}


def test_tick_exception_round_trip(tmp_path):
    s = SharedState(session_id="abc")
    entry = s.record_tick_exception(
        tick=7,
        stage="tick_body",
        agent="orchestration",
        exc_type="RuntimeError",
        message="boom",
        traceback_text="Traceback...\nRuntimeError: boom",
    )
    s.save(tmp_path)

    s2 = SharedState.load_or_init(tmp_path)
    assert s2.last_tick_exception == entry
    assert s2.last_tick_exception["tick"] == 7
    assert s2.last_tick_exception["stage"] == "tick_body"
    assert s2.last_tick_exception["agent"] == "orchestration"
    assert s2.last_tick_exception["type"] == "RuntimeError"


def test_load_or_init_returns_blank_when_missing(tmp_path):
    s = SharedState.load_or_init(tmp_path)
    assert s.session_id == ""
    assert (tmp_path / "state.json").exists() is False


def test_save_is_atomic(tmp_path):
    """Concurrent readers must never see a partial write."""
    s = SharedState(session_id="x")
    s.save(tmp_path)
    raw = (tmp_path / "state.json").read_text()
    parsed = json.loads(raw)
    assert parsed["session_id"] == "x"
    leftovers = list(tmp_path.glob(".state-*"))
    assert leftovers == []


def test_from_dict_drops_unknown_fields():
    raw = {"session_id": "s", "unknown_future_field": 42, "baseline_tput": 100.0}
    s = SharedState.from_dict(raw)
    assert s.session_id == "s"
    assert s.baseline_tput == 100.0
    assert s.last_tick_exception == {}
    assert not hasattr(s, "unknown_future_field")


def test_apply_changes_only_known_fields():
    s = SharedState()
    applied = s.apply_changes(
        {"current_action": "baseline", "bogus": 1, "cumulative_gain": 5.0},
        allow_core=True,
    )
    assert applied == {"current_action": "baseline", "cumulative_gain": 5.0}
    assert s.current_action == "baseline"
    assert s.cumulative_gain == 5.0


def test_add_pruned_family_idempotent():
    s = SharedState()
    assert s.add_pruned_family("deep_kernel") is True
    assert s.add_pruned_family("deep_kernel") is False
    assert s.pruned_families == ["deep_kernel"]


def test_is_pruned():
    s = SharedState(pruned_families=["long"])
    assert s.is_pruned("long")
    assert not s.is_pruned("prep")


def test_increment_crash_count():
    s = SharedState()
    assert s.increment_crash_count() == 1
    assert s.increment_crash_count(by=2) == 3
    assert s.crash_count == 3


def test_to_prompt_summary_contains_key_fields():
    s = SharedState(
        session_id="s1", model_name="Llama-3", baseline_tput=1840.0,
        cumulative_gain=10.0, current_action="backends",
        pruned_families=["deep_kernel"],
    )
    summary = s.to_prompt_summary()
    assert "s1" in summary
    assert "Llama-3" in summary
    assert "1840" in summary
    assert "10.0" in summary
    assert "backends" in summary
    assert "deep_kernel" in summary


# Coordinator × SharedState integration
@pytest.mark.asyncio
async def test_coordinator_loads_existing_shared_state(session_dir):
    """Coordinator.__init__ must pick up an existing state.json (resume hook)."""
    pre = SharedState(session_id="resumed", baseline_tput=2000.0,
                      pruned_families=["deep_kernel"])
    pre.save(session_dir)
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c.shared_state.session_id == "resumed"
        assert c.shared_state.baseline_tput == 2000.0
        assert c.shared_state.is_pruned("deep_kernel")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_prune_branch_persists(session_dir):
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        await c._handle_intent("robustness", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "deep_kernel", "reason": "3 fails"},
        ))
        # In-memory updated
        assert "deep_kernel" in c.shared_state.pruned_families
        # File on disk reflects it
        on_disk = json.loads((session_dir / "state.json").read_text())
        assert "deep_kernel" in on_disk["pruned_families"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_pruned_family_survives_coordinator_restart(session_dir):
    c1 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c1._handle_intent("robustness", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "long", "reason": "expensive"},
        ))
    finally:
        await c1.stop()

    # Restart — fresh Coordinator must still observe the prune
    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c2.shared_state.is_pruned("long")
        # The prune is advisory — proposals still reach the pending queue
        # with an advisory observation so the LLM can decide.
        await c2._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "long", "predicted_gain_pct": 5.0},
        ))
        assert c2.state.pending_proposals
        obs = await c2.bus.tail(topic="observation")
        assert any(
            m.payload.get("kind") == "proposal_pruned_advisory" for m in obs
        )
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_coordinator_update_state_persists_known_fields(session_dir):
    """Orchestration may write non-core fields; core fields are gated by PolicyGate's CORE_STATE_FIELDS."""
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        await c._handle_intent("orchestration", Intent(
            type=IntentType.UPDATE_STATE,
            payload={"changes": {"current_action": "baseline",
                                  "target_summary": "GEMM-bound 8B model"}},
        ))
        assert c.shared_state.current_action == "baseline"
        assert c.shared_state.target_summary == "GEMM-bound 8B model"
        on_disk = json.loads((session_dir / "state.json").read_text())
        assert on_disk["current_action"] == "baseline"
        assert on_disk["target_summary"] == "GEMM-bound 8B model"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_update_state_drops_unknown_fields(session_dir):
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        await c._handle_intent("orchestration", Intent(
            type=IntentType.UPDATE_STATE,
            payload={"changes": {"current_action": "baseline",
                                  "future_unknown_key": 42}},
        ))
        assert c.shared_state.current_action == "baseline"
        # The observation events should record the rejected key
        obs = await c.bus.tail(topic="observation", n=20)
        update_events = [m for m in obs if m.payload.get("kind") == "update_state"]
        assert update_events
        last = update_events[0]  # tail returns DESC
        assert "future_unknown_key" in last.payload["rejected"]
    finally:
        await c.stop()


# reference recipe fact-layer (resume survival + current_setting.sh render)
def test_reference_fields_survive_resume(tmp_path):
    """R3: reference_* fields persist through save → from_dict (resume)."""
    s = SharedState(session_id="t", model_name="m", model_path="/x/m")
    s.reference_server_args = "--block-size 128"
    s.reference_envs = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}
    s.reference_model = "minimaxm3"
    s.reference_source = "/recipes/minimaxm3_fp8_mi300x.sh"
    restored = SharedState.from_dict(s.to_dict())
    assert restored.reference_server_args == "--block-size 128"
    assert restored.reference_envs == {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}
    assert restored.reference_model == "minimaxm3"
    assert restored.reference_source == "/recipes/minimaxm3_fp8_mi300x.sh"


def test_save_renders_current_setting_sh(tmp_path, monkeypatch):
    """save() emits a re-parseable current_setting.sh from current_best."""
    monkeypatch.setenv("FRAMEWORK", "vllm")
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState(session_id="t", model_name="m", model_path="/x/m")
    s.current_best = {
        "extra_server_args": "--block-size 128 --attention-backend TRITON_ATTN",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
    }
    s.save(sd)
    out = sd / "current_setting.sh"
    assert out.exists()
    from inference_optimizer.reference_script import parse_reference_script
    r = parse_reference_script(str(out), framework="vllm")
    assert "--block-size 128" in r.server_args
    assert "TRITON_ATTN" in r.server_args
    assert r.envs.get("VLLM_ROCM_USE_AITER") == "1"


def test_save_no_current_setting_when_no_best(tmp_path):
    """No current_best → no current_setting.sh (0-degrade)."""
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState(session_id="t", model_name="m", model_path="/x/m")
    s.save(sd)
    assert not (sd / "current_setting.sh").exists()
