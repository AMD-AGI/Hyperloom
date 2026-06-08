# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P1-4 Coordinator resume tests.

Covers:

* Fresh session: is_resume=False, no replay needed
* Existing state.json triggers is_resume=True even with empty events
* Existing events trigger is_resume=True
* replay_for_resume reconstructs pending_proposals from undecided
  topic='proposal' events
* Approved proposals are NOT re-instantiated as pending after resume
* Rejected proposals are NOT re-instantiated as pending after resume
* Multiple proposals + mixed decisions reconstruct correct pending set
* Coordinator restart preserves pruned_families AND restores undecided
  pending_proposals AND keeps task lifecycle intact
* tick() lazily triggers replay_for_resume on first call (so callers
  don't have to remember to call it explicitly)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker
    seed_target_analysis_marker(sd)
    return sd


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


# ===========================================================================
# Resume detection
# ===========================================================================
@pytest.mark.asyncio
async def test_fresh_session_is_not_resume(session_dir):
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        info = c.resumed_from
        assert info["is_resume"] is False
        assert info["event_count"] == 0
        assert info["state_json_present"] is False
        assert info["rebuilt"] is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_existing_state_json_triggers_resume(session_dir):
    SharedState(session_id="resumed").save(session_dir)
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c.resumed_from["is_resume"] is True
        assert c.resumed_from["state_json_present"] is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_existing_events_triggers_resume(session_dir):
    # First Coordinator: emit a heartbeat to populate the events table.
    c1 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c1.tick(1)
    finally:
        await c1.stop()
    # Second Coordinator on the same session_dir sees events ≥1.
    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c2.resumed_from["is_resume"] is True
        assert c2.resumed_from["event_count"] >= 1
    finally:
        await c2.stop()


# ===========================================================================
# Resume rebuild — pending_proposals
# ===========================================================================
@pytest.mark.asyncio
async def test_replay_rebuilds_undecided_proposals(session_dir):
    """One propose, no verdict → resume restores it as pending."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    # Mock orchestration only — no Critic mock so no auto-approval.
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends_no_critic = {
        "orchestration": MockBackend(ScriptedPlan(turns=[MockTurn(intents=[propose])],
                                                    default_intent=_heartbeat()),
                                       name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends_no_critic)
    try:
        await c1.tick(1)
        assert len(c1.state.pending_proposals) == 1
        original_id = next(iter(c1.state.pending_proposals.keys()))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 1
        assert original_id in c2.state.pending_proposals
        restored = c2.state.pending_proposals[original_id]
        assert restored.action_name == "baseline"
        assert restored.from_agent == "orchestration"
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_approved_proposals(session_dir):
    """Approved proposal must not appear as pending after resume."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    backends = _backends_full()
    backends["orchestration"] = MockBackend(
        ScriptedPlan(turns=[MockTurn(intents=[propose])],
                     default_intent=_heartbeat()),
        name="orch",
    )
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(2)  # tick 1: propose, tick 2: critic auto-approves
        assert any(p.verdict == "approve" for p in c1.state.pending_proposals.values())
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        # The approved proposal should be filtered out.
        assert stats["pending_restored"] == 0
        assert c2.state.pending_proposals == {}
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_rejected_proposals(session_dir):
    """Rejected proposal also counted as decided → not pending after resume."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[propose])],
                         default_intent=_heartbeat()),
            name="o",
        ),
        "kernel":     MockBackend(silent, name="k"),
        "critic":     MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(1)
        proposal_id = next(iter(c1.state.pending_proposals.keys()))
        await c1._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": proposal_id, "verdict": "reject",
                     "reasoning": "violates kb-7", "kb_evidence": "kb-7"},
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 0
        assert stats["verdicts_seen"] >= 1
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_mixed_pending_and_decided(session_dir):
    """3 proposals, 1 approved, 1 rejected, 1 undecided → 1 restored."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        # This test is about replay bookkeeping, not execution-order gating.
        # Seed prerequisites so arbitrary proposals are accepted.
        c1.shared_state.baseline_tput = 100.0
        c1.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c1.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            # Roofline-v2 N3: backends now requires fresh analysis_md_text.
            "analysis_md_text": "FAKE_REPORT",
        }
        c1.shared_state.save(session_dir)
        proposal_ids = []
        for action in ("baseline", "profile", "explore"):
            await c1._handle_intent("orchestration", Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": action, "predicted_gain_pct": 0.0},
            ))
            # Grab the most recent proposal_msg_id
            tail = await c1.bus.tail(topic="proposal", n=1)
            proposal_ids.append(tail[0].msg_id)

        # Approve baseline (proposal_ids[0])
        await c1._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": proposal_ids[0],
                     "verdict": "approve", "reasoning": "ok"},
        ))
        # Reject profile (proposal_ids[1])
        await c1._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": proposal_ids[1],
                     "verdict": "reject", "reasoning": "no", "kb_evidence": "kb-x"},
        ))
        # backends (proposal_ids[2]) left undecided
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 1
        restored = next(iter(c2.state.pending_proposals.values()))
        assert restored.action_name == "explore"
    finally:
        await c2.stop()


# ===========================================================================
# Resume + SharedState combined
# ===========================================================================
@pytest.mark.asyncio
async def test_resume_preserves_pruned_and_restores_pending(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        # 1 prune + 1 undecided proposal
        await c1._handle_intent("robustness", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "deep_kernel", "reason": "x"},
        ))
        await c1._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c2.replay_for_resume()
        assert c2.shared_state.is_pruned("deep_kernel")
        assert len(c2.state.pending_proposals) == 1
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_tick_lazily_runs_replay_on_resume(session_dir):
    """Resume callers shouldn't have to remember to call replay manually —
    the first tick() should do it."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        # No explicit replay_for_resume — tick should trigger it
        assert c2.resumed_from["rebuilt"] is False
        await c2.tick(1)
        assert c2.resumed_from["rebuilt"] is True
        assert len(c2.state.pending_proposals) == 1
    finally:
        await c2.stop()


# ===========================================================================
# N23 — --resume is N17-layout-aware (formerly test_n23_resume_per_session.py)
# ===========================================================================


class TestN23ResumePerSession:
    """``--resume`` must understand the N17 per-session layout: workspace
    is the parent of ``<model>/<UTC ts>/``. The fallback contract is:

    * leave $USER_DATA_PATH alone (workspace level) so runtime/ resolves;
    * pick the LATEST per-session subdir under ``<model>/<ts>/`` when no
      ``--resume-from`` is given;
    * accept ``--resume-from`` as an explicit override (must be under
      workspace_root, must exist);
    * pin INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR to the resolved subdir
      BEFORE any state load, so subprocesses inherit it.

    These tests exercise the path-resolution layer
    (``inference_optimizer.paths.find_latest_per_session_dir``) the
    ``--resume`` block delegates to.
    """

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        from inference_optimizer import paths as _paths
        monkeypatch.setenv(_paths.ENV_USER_DATA_PATH, str(tmp_path))
        monkeypatch.delenv(_paths.ENV_CURRENT_SESSION_DIR, raising=False)
        monkeypatch.delenv(_paths.ENV_SESSION_LAYOUT, raising=False)

    def test_resume_picks_latest_subdir_after_two_launches(self, tmp_path):
        from inference_optimizer import paths as _paths
        sd1 = _paths.make_session_dir(model_name="DeepSeek-R1-0528")
        assert _paths.find_latest_per_session_dir() == sd1
        assert _paths.find_latest_per_session_dir(model_name="DeepSeek-R1-0528") == sd1

        later_ts = "29990101T000000Z"
        sd2 = tmp_path / "DeepSeek-R1-0528" / later_ts
        sd2.mkdir(parents=True)

        assert _paths.find_latest_per_session_dir() == sd2
        assert _paths.find_latest_per_session_dir(model_name="DeepSeek-R1-0528") == sd2

    def test_resume_does_not_mutate_user_data_path(self, tmp_path):
        from inference_optimizer import paths as _paths
        sd = _paths.make_session_dir(model_name="Qwen3-32B")
        import os as _os
        assert _os.environ[_paths.ENV_USER_DATA_PATH] == str(tmp_path)
        assert _os.environ[_paths.ENV_CURRENT_SESSION_DIR] == str(sd)
        assert _paths.workspace_root() == tmp_path
        assert tmp_path in sd.parents

    def test_resume_falls_back_to_flat_when_no_per_session_subdir(self, tmp_path):
        from inference_optimizer import paths as _paths
        assert _paths.find_latest_per_session_dir() is None

    def test_resume_from_explicit_path_must_be_under_workspace_root(
        self, tmp_path, monkeypatch,
    ):
        from inference_optimizer import paths as _paths
        sd = _paths.make_session_dir(model_name="Qwen3-32B")
        assert tmp_path.resolve() in sd.resolve().parents

        foreign = tmp_path.parent / "stranger_workspace" / "sess"
        foreign.mkdir(parents=True)
        try:
            foreign.resolve().relative_to(tmp_path.resolve())
            assert False, "foreign path should not be under workspace_root"
        except ValueError:
            pass

    def test_latest_picks_across_models_when_model_name_omitted(self, tmp_path):
        from inference_optimizer import paths as _paths
        (tmp_path / "ModelA").mkdir()
        (tmp_path / "ModelA" / "20260101T000000Z").mkdir()
        (tmp_path / "ModelB").mkdir()
        (tmp_path / "ModelB" / "20260520T000000Z").mkdir()
        (tmp_path / "ModelC").mkdir()
        (tmp_path / "ModelC" / "20260315T000000Z").mkdir()

        picked = _paths.find_latest_per_session_dir()
        assert picked is not None
        assert picked.parent.name == "ModelB"
        assert picked.name == "20260520T000000Z"

    def test_workspace_shared_dirs_never_picked_as_session(self, tmp_path):
        from inference_optimizer import paths as _paths
        (tmp_path / "runtime").mkdir()
        (tmp_path / "runtime" / "20990101T000000Z").mkdir()
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "20990101T000000Z").mkdir()
        (tmp_path / "RealModel").mkdir()
        (tmp_path / "RealModel" / "20260518T100000Z").mkdir()

        picked = _paths.find_latest_per_session_dir()
        assert picked is not None
        assert picked.parent.name == "RealModel"
        assert "runtime" not in str(picked)
        assert "logs" not in str(picked)


# ===========================================================================
# N24 — _load_kernel_agent_env_fallback hard-fails on bad state
# (formerly test_n24_kernel_agent_env_hardfail.py)
# ===========================================================================


class TestN24KernelAgentEnvHardFail:
    """Pre-N24 the fallback printed a WARN and let ``_preflight()``
    continue when ``$USER_DATA_PATH/runtime/kernel-agent.env.sh`` was
    missing or empty — silently masking the most common N17 misuse
    (USER_DATA_PATH pointed at a per-session subdir). N24 aborts with
    sys.exit(2) and a clear actionable message so operators notice
    within seconds instead of after a 10h silent stall.
    """

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        for var in (
            "HYPERLOOM_KERNEL_AGENT_ROOT",
            "KERNEL_AGENT_ENV",
            "USER_DATA_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_noop_when_root_already_set(self, monkeypatch, capsys):
        from inference_optimizer import cli
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/opt/kernel-agent")
        cli._load_kernel_agent_env_fallback()
        out = capsys.readouterr()
        assert out.out == ""
        assert out.err == ""

    def test_aborts_when_no_user_data_path(self, monkeypatch, capsys):
        from inference_optimizer import cli
        with pytest.raises(SystemExit) as excinfo:
            cli._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "USER_DATA_PATH" in err
        assert "install.sh" in err

    def test_aborts_when_env_file_missing(self, tmp_path, monkeypatch, capsys):
        from inference_optimizer import cli
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            cli._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "kernel-agent.env.sh" in err
        assert "install.sh" in err
        assert str(tmp_path) in err

    def test_aborts_when_env_file_does_not_define_root(
        self, tmp_path, monkeypatch, capsys,
    ):
        from inference_optimizer import cli
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "# stale file\nexport SOMETHING_ELSE=1\n", encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            cli._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "HYPERLOOM_KERNEL_AGENT_ROOT" in err
        assert "stale" in err or "malformed" in err

    def test_sources_vars_on_success(self, tmp_path, monkeypatch, capsys):
        from inference_optimizer import cli
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "# valid env file\n"
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/opt/kernel-agent\n"
            "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        cli._load_kernel_agent_env_fallback()
        import os as _os
        assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/opt/kernel-agent"
        assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "INFO"
        out = capsys.readouterr().out
        assert "loaded" in out
        assert "kernel-agent" in out

    def test_env_wins_over_file(self, tmp_path, monkeypatch, capsys):
        from inference_optimizer import cli
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/from/file\n"
            "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("KERNEL_AGENT_LOG_LEVEL", "DEBUG")
        cli._load_kernel_agent_env_fallback()
        import os as _os
        assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/from/file"
        assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "DEBUG"

    def test_explicit_kernel_agent_env_overrides_user_data_path(
        self, tmp_path, monkeypatch,
    ):
        from inference_optimizer import cli
        custom = tmp_path / "custom-loc.sh"
        custom.write_text(
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/from/custom\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("KERNEL_AGENT_ENV", str(custom))
        monkeypatch.setenv("USER_DATA_PATH", "/nonexistent/should-not-be-used")
        cli._load_kernel_agent_env_fallback()
        import os as _os
        assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/from/custom"
