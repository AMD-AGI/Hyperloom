# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator resume tests."""

from __future__ import annotations

import time

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.cli import preflight as cli_preflight
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_full() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "critic": MockCriticBackend(),
    }


@pytest.mark.asyncio
async def test_fresh_session_is_not_resume(session_dir):
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        info = c.writeback.resumed_from
        assert info == {"is_resume": False, "rebuilt": False}
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_existing_state_json_triggers_resume(session_dir):
    SharedState(session_id="resumed").save(session_dir)
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c.writeback.resumed_from["is_resume"] is True
    finally:
        await c.stop()


class TestAClosedSessionIsReopenedOnResume:
    """CLOSE has no way out, so a leg that loads it would tick in it to the end."""

    @pytest.mark.asyncio
    async def test_a_session_stopped_in_close_starts_the_next_leg_at_the_entrance(
        self,
        session_dir,
    ):
        SharedState(session_id="closed", phase="CLOSE").save(session_dir)

        coordinator = Coordinator(session_dir, backends=_backends_full())
        try:
            assert coordinator.shared_state.phase == "PRELUDE"
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_the_reopening_is_recorded_as_the_transition_it_is(self, session_dir):
        """A phase the run did not reach by working its way there needs saying so."""
        SharedState(session_id="closed", phase="CLOSE").save(session_dir)

        coordinator = Coordinator(session_dir, backends=_backends_full())
        try:
            latest = coordinator.shared_state.phase_history[-1]
        finally:
            await coordinator.stop()

        assert latest["from_phase"] == "CLOSE"
        assert latest["to_phase"] == "PRELUDE"
        assert latest["evidence"]["trigger"] == "resumed_from_close"

    @pytest.mark.asyncio
    async def test_the_earlier_legs_close_sequence_does_not_count_for_this_one(
        self,
        session_dir,
    ):
        """The flag means \"the sequencer already wrote the breakdown\"."""
        SharedState(session_id="closed", phase="CLOSE", close_sequence_done=True).save(session_dir)

        coordinator = Coordinator(session_dir, backends=_backends_full())
        try:
            assert coordinator.shared_state.close_sequence_done is False
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_a_session_stopped_anywhere_else_resumes_where_it_stopped(self, session_dir):
        """Only the phase with no exit is reopened; the rest can still make progress."""
        SharedState(session_id="mid", phase="FRAMEWORK_AGENT").save(session_dir)

        coordinator = Coordinator(session_dir, backends=_backends_full())
        try:
            assert coordinator.shared_state.phase == "FRAMEWORK_AGENT"
            assert coordinator.shared_state.phase_history == []
        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_the_reopened_leg_may_actually_measure_the_baseline_it_reopened_for(
        self,
        session_dir,
    ):
        """Reopening the phase is only half of it; the round has to be admissible."""
        SharedState(
            session_id="cold",
            phase="CLOSE",
            baseline_tput=1000.0,
            baseline_measure_round_dropped=True,
        ).save(session_dir)

        coordinator = Coordinator(session_dir, backends=_backends_full())
        try:
            assert coordinator.shared_state.phase == "PRELUDE"
            coordinator.policy.validate_intent(
                "orchestration",
                Intent(
                    type=IntentType.DELEGATE,
                    payload={"action_name": "baseline", "params": {}},
                ),
            )
        finally:
            await coordinator.stop()


@pytest.mark.asyncio
async def test_existing_events_triggers_resume(session_dir):
    c1 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c1.tick(1)
    finally:
        await c1.stop()
    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c2.writeback.resumed_from["is_resume"] is True
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_rebuilds_undecided_proposals(session_dir):
    """One propose, no verdict → resume restores it as pending."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends_no_critic = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[propose])], default_intent=_heartbeat()), name="o"
        ),
        "critic": MockBackend(silent, name="c"),
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
        await c2.writeback.replay_for_resume()
        assert list(c2.state.pending_proposals) == [original_id]
        restored = c2.state.pending_proposals[original_id]
        assert restored.action_name == "baseline"
        assert restored.from_agent == "orchestration"
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_approved_proposals(session_dir):
    """Approved proposal must not appear as pending after resume."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
    backends = _backends_full()
    backends["orchestration"] = MockBackend(
        ScriptedPlan(turns=[MockTurn(intents=[propose])], default_intent=_heartbeat()),
        name="orch",
    )
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(2)
        verdicts = await c1.bus.tail(topic="review_verdict", n=100)
        assert any(v.payload.get("verdict") == "approve" for v in verdicts)
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c2.writeback.replay_for_resume()
        assert c2.state.pending_proposals == {}
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_rejected_proposals(session_dir):
    """Rejected proposal also counted as decided → not pending after resume."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[propose])], default_intent=_heartbeat()),
            name="o",
        ),
        "critic": MockBackend(silent, name="c"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(1)
        proposal_id = next(iter(c1.state.pending_proposals.keys()))
        await c1.router.handle_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": proposal_id,
                    "verdict": "reject",
                    "reasoning": "violates kb-7",
                    "kb_evidence": "kb-7",
                },
            ),
        )
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c2.writeback.replay_for_resume()
        assert c2.state.pending_proposals == {}
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_mixed_pending_and_decided(session_dir):
    """3 proposals, 1 approved, 1 rejected, 1 undecided → 1 restored."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        # Seed prerequisites so arbitrary proposals are accepted.
        c1.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c1.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "analysis_md_text": "FAKE_REPORT",
        }
        c1.shared_state.save(session_dir)
        proposal_ids = []
        for action in ("baseline", "target_analysis", "explore"):
            if action == "explore":
                # Admission holds each proposal to its phase's actions; explore is FRAMEWORK_AGENT's.
                c1.shared_state.phase = "FRAMEWORK_AGENT"
            await c1.router.handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.PROPOSE_ACTION,
                    payload={"action_name": action, "predicted_gain_pct": 0.0},
                ),
            )
            tail = await c1.bus.tail(topic="proposal", n=1)
            proposal_ids.append(tail[0].msg_id)
            if action == "baseline":
                # profile/explore require baseline_tput > 0 (execution_order); the real baseline action would have set
                # this on completion.
                c1.shared_state.baseline_tput = 100.0

        await c1.router.handle_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": proposal_ids[0], "verdict": "approve", "reasoning": "ok"},
            ),
        )
        await c1.router.handle_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": proposal_ids[1],
                    "verdict": "reject",
                    "reasoning": "no",
                    "kb_evidence": "kb-x",
                },
            ),
        )
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c2.writeback.replay_for_resume()
        [restored] = c2.state.pending_proposals.values()
        assert restored.action_name == "explore"
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_resume_preserves_pruned_and_restores_pending(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"family": "deep_kernel", "reason": "x"},
            ),
        )
        await c1.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
            ),
        )
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c2.writeback.replay_for_resume()
        assert c2.shared_state.is_pruned("deep_kernel")
        assert len(c2.state.pending_proposals) == 1
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_tick_lazily_runs_replay_on_resume(session_dir):
    """The first tick() triggers replay so resume callers needn't call it manually."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
            ),
        )
    finally:
        await c1.stop()

    # Use a silent critic so the restored proposal is not auto-approved during tick.
    silent_backends = {
        "orchestration": MockBackend(silent, name="o2"),
        "critic": MockBackend(silent, name="c2"),
    }
    c2 = Coordinator(session_dir, backends=silent_backends)
    try:
        assert c2.writeback.resumed_from["rebuilt"] is False
        await c2.tick(1)
        assert c2.writeback.resumed_from["rebuilt"] is True
        assert len(c2.state.pending_proposals) == 1
    finally:
        await c2.stop()


class TestN23ResumePerSession:
    """``--resume-from`` addresses a session inside the N17 per-session layout."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.session import paths as _paths

        monkeypatch.setenv(_paths.ENV_USER_DATA_PATH, str(tmp_path))
        monkeypatch.delenv(_paths.ENV_CURRENT_SESSION_DIR, raising=False)

    def test_resume_does_not_mutate_user_data_path(self, tmp_path):
        from hyperloom.inference_optimizer.session import paths as _paths

        sd = _paths.make_session_dir(model_name="Qwen3-32B")
        import os as _os

        assert _os.environ[_paths.ENV_USER_DATA_PATH] == str(tmp_path)
        assert _os.environ[_paths.ENV_CURRENT_SESSION_DIR] == str(sd)
        assert _paths.workspace_root() == tmp_path
        assert tmp_path in sd.parents

    def test_resume_from_explicit_path_must_be_under_workspace_root(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hyperloom.inference_optimizer.session import paths as _paths

        sd = _paths.make_session_dir(model_name="Qwen3-32B")
        assert tmp_path.resolve() in sd.resolve().parents

        foreign = tmp_path.parent / "stranger_workspace" / "sess"
        foreign.mkdir(parents=True)
        try:
            foreign.resolve().relative_to(tmp_path.resolve())
            assert False, "foreign path should not be under workspace_root"
        except ValueError:
            pass

    @pytest.mark.parametrize(
        "argv",
        [
            ["optimize", "--resume"],
            # Historical automatic-resume invocations remain unsupported.
            ["optimize", "--resume", "--resume-from", "/tmp/sess"],
        ],
    )
    def test_no_session_can_be_resumed_without_naming_it(self, argv):
        """``--resume`` cannot start a run; it exits instead of choosing a session."""
        from hyperloom.inference_optimizer.cli.parser import _build_parser

        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args(argv)
        assert exc.value.code == 2


# _load_kernel_agent_env_fallback hard-fails on bad state


class TestN24KernelAgentEnvHardFail:
    """A missing/empty ``kernel-agent.env.sh`` aborts with sys.exit(2)."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        for var in (
            "KERNEL_AGENT_ENV",
            "USER_DATA_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_aborts_when_no_user_data_path(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli_preflight._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "USER_DATA_PATH" in err
        assert "install.sh" in err

    def test_aborts_when_env_file_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            cli_preflight._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "kernel-agent.env.sh" in err
        assert "install.sh" in err
        assert str(tmp_path) in err

    @pytest.mark.parametrize(
        "contents",
        ["# stale file\n", "# stale file\nexport SOMETHING_ELSE=1\n"],
        ids=["comments-only", "only-unsupported"],
    )
    def test_aborts_when_env_file_sets_no_supported_vars(
        self,
        tmp_path,
        monkeypatch,
        capsys,
        contents,
    ):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(contents, encoding="utf-8")
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            cli_preflight._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "malformed or stale" in err
        assert "install.sh" in err

    def test_sources_vars_on_success(self, tmp_path, monkeypatch, capsys):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "# valid env file\nexport KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        cli_preflight._load_kernel_agent_env_fallback()
        import os as _os

        assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "INFO"
        out = capsys.readouterr().out
        assert "loaded" in out
        assert "kernel-agent" in out

    def test_env_wins_over_file(self, tmp_path, monkeypatch, capsys):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("KERNEL_AGENT_LOG_LEVEL", "DEBUG")
        cli_preflight._load_kernel_agent_env_fallback()
        import os as _os

        assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "DEBUG"

    def test_credential_fallback_block_parses_without_warnings(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        """The installer emits credentials as a conditional block (#1169)."""
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            'if [ -n "${ANTHROPIC_API_KEY:-}" ]; then\n'
            "  [ \"${ANTHROPIC_API_KEY}\" = 'ak-install-time' ] || \\\n"
            "    echo '[kernel-agent] ANTHROPIC_API_KEY differs' >&2\n"
            "else\n"
            "  export ANTHROPIC_API_KEY='ak-install-time'\n"
            "fi\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import os as _os

        try:
            cli_preflight._load_kernel_agent_env_fallback()
            loaded_key = _os.environ.get("ANTHROPIC_API_KEY")
        finally:
            # The loader writes straight into os.environ, which monkeypatch cannot roll back; a leaked credential
            # reshapes later auth tests.
            _os.environ.pop("ANTHROPIC_API_KEY", None)

        assert loaded_key == "ak-install-time"
        assert "unsupported kernel-agent env key" not in capsys.readouterr().err

    def test_explicit_kernel_agent_env_overrides_user_data_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        custom = tmp_path / "custom-loc.sh"
        custom.write_text(
            "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("KERNEL_AGENT_ENV", str(custom))
        monkeypatch.setenv("USER_DATA_PATH", "/nonexistent/should-not-be-used")
        outcome = cli_preflight._load_kernel_agent_env_fallback()

        assert outcome["detail"]["env_file"] == str(custom)


# A stale/placeholder TRACELENS_ROOT is corrected from the installer-written env file; template placeholders are
# treated as unset.
class TestTracelensRootEnvCorrection:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        import os

        snapshot = dict(os.environ)
        for var in (
            "KERNEL_AGENT_ENV",
            "USER_DATA_PATH",
            "TRACELENS_ROOT",
            "MAGPIE_PATH",
        ):
            monkeypatch.delenv(var, raising=False)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(snapshot)

    def _write_env_file(self, tmp_path, tracelens_dir):
        runtime = tmp_path / "runtime"
        runtime.mkdir(exist_ok=True)
        (runtime / "kernel-agent.env.sh").write_text(
            f"export TRACELENS_ROOT='{tracelens_dir}'\n",
            encoding="utf-8",
        )

    def test_corrects_invalid_inherited_root_from_env_file(self, tmp_path, monkeypatch, capsys):
        """An inherited TRACELENS_ROOT that points nowhere is corrected from the file."""
        good = tmp_path / "deps" / "TraceLens"
        good.mkdir(parents=True)
        self._write_env_file(tmp_path, good)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "ghost" / "TraceLens"))

        outcome = cli_preflight._load_kernel_agent_env_fallback()

        import os as _os

        assert _os.environ["TRACELENS_ROOT"] == str(good)
        assert outcome["detail"]["corrected_keys"] == ["TRACELENS_ROOT"]
        assert "TRACELENS_ROOT" in capsys.readouterr().err

    def test_keeps_valid_inherited_root(self, tmp_path, monkeypatch):
        """A valid inherited TRACELENS_ROOT wins over the env file (env-wins)."""
        file_dir = tmp_path / "file" / "TraceLens"
        file_dir.mkdir(parents=True)
        inherited = tmp_path / "inherited" / "TraceLens"
        inherited.mkdir(parents=True)
        self._write_env_file(tmp_path, file_dir)
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("TRACELENS_ROOT", str(inherited))

        outcome = cli_preflight._load_kernel_agent_env_fallback()

        import os as _os

        assert _os.environ["TRACELENS_ROOT"] == str(inherited)
        assert outcome["detail"]["corrected_keys"] == []

    @pytest.mark.parametrize("override_kind", ["missing", "non-checkout", "empty"])
    def test_magpie_path_is_not_corrected(self, tmp_path, monkeypatch, override_kind):
        """TraceLens correction must not replace an explicit Magpie value, even an invalid one."""
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        magpie_dir = tmp_path / "installed-magpie"
        (magpie_dir / "Magpie").mkdir(parents=True)
        (magpie_dir / "Magpie" / "__init__.py").write_text("", encoding="utf-8")
        override = tmp_path / "not-a-magpie-checkout"
        if override_kind == "non-checkout":
            override.mkdir()
        selected = "" if override_kind == "empty" else str(override)
        (runtime / "kernel-agent.env.sh").write_text(
            f"export MAGPIE_PATH='{magpie_dir}'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("MAGPIE_PATH", selected)

        outcome = cli_preflight._load_kernel_agent_env_fallback()

        import os as _os

        assert _os.environ["MAGPIE_PATH"] == selected
        assert "MAGPIE_PATH" not in outcome["detail"]["corrected_keys"]

    def test_runtime_magpie_gapfill_reaches_child_imports(self, tmp_path, monkeypatch):
        """The runtime file's Magpie import root reaches child processes."""
        import os
        import subprocess
        import sys

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        magpie_dir = tmp_path / "runtime Magpie"
        (magpie_dir / "Magpie").mkdir(parents=True)
        (magpie_dir / "Magpie" / "__init__.py").write_text("RUNTIME_MARKER = 'installer-checkout'\n", encoding="utf-8")
        (runtime / "kernel-agent.env.sh").write_text(
            f"export MAGPIE_PATH='{magpie_dir}'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("PYTHONPATH", "")

        outcome = cli_preflight._load_kernel_agent_env_fallback()
        cli_preflight._derive_runtime_paths()

        assert os.environ["MAGPIE_PATH"] == str(magpie_dir)
        assert "MAGPIE_PATH" not in outcome["detail"]["corrected_keys"]
        child = subprocess.run(
            [sys.executable, "-c", "import Magpie; print(Magpie.RUNTIME_MARKER)"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "installer-checkout"

    def test_placeholder_path_to_your_is_unset(self):
        assert cli_preflight._is_placeholder_tracelens_path("/path/to/your/TraceLens") is True
        assert cli_preflight._is_placeholder_tracelens_path("<your-tracelens>") is True
        assert cli_preflight._is_placeholder_tracelens_path("/tmp/hyperloom/TraceLens") is False

    def test_check_root_exits_when_set_but_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "ghost"))
        with pytest.raises(SystemExit) as excinfo:
            cli_preflight._check_tracelens_root_exists()
        assert excinfo.value.code == 2

    def test_check_root_noop_when_valid_or_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRACELENS_ROOT", raising=False)
        cli_preflight._check_tracelens_root_exists()
        good = tmp_path / "TraceLens"
        good.mkdir()
        monkeypatch.setenv("TRACELENS_ROOT", str(good))
        cli_preflight._check_tracelens_root_exists()


@pytest.mark.asyncio
async def test_resume_preserves_custom_workload_paths(session_dir):
    """Coordinator reload must not drop the paths a custom resume re-exports."""
    SharedState(
        session_id="custom-resume",
        framework="custom",
        bypass_scripts_dir="/opt/worldplay/scripts",
        framework_repo_path="/opt/worldplay/src",
        benchmark_backend="bypass",
    ).save(session_dir)

    coordinator = Coordinator(session_dir, backends=_backends_full())
    try:
        state = coordinator.shared_state
        assert state.framework == "custom"
        assert state.bypass_scripts_dir == "/opt/worldplay/scripts"
        assert state.framework_repo_path == "/opt/worldplay/src"
        assert state.benchmark_backend == "bypass"
    finally:
        await coordinator.stop()


class TestASweepCancelledByRecoverIsRedispatchedOnResume:
    """``recover`` cancels the in-flight conc_sweep out of band, so nothing ever records its outcome.

    SWEEP's only work is that sweep, and its entry hook fires on a transition, not on a resume; without a fresh
    dispatch the resumed leg waits in SWEEP for a result no task will ever write.
    """

    #: The execution environment the resumed leg runs in, as ``recover`` is told it.
    _NEW_SCOPE = "0b0b0b0b-2222-4222-8222-000000000002:4026531002"

    @staticmethod
    async def _enter_sweep(session_dir, *, start: bool):
        from hyperloom.orchestrator.phases.machine import Transition

        first = Coordinator(session_dir, backends=_backends_full())
        try:
            first.shared_state.phase = "SWEEP"
            first.shared_state.baseline_tput = 800.0
            await first.phase_sweep.on_enter_sweep(
                Transition(from_phase="KERNEL_AGENT", to_phase="SWEEP", reason="test", evidence={})
            )
            [sweep] = await first.tasks.queued()
            assert sweep.kind == "conc_sweep"
            if start:
                # Running, holding its lanes under this environment, as the dispatcher leaves it.
                await first.locks.acquire_many(
                    list(sweep.requires_lanes),
                    holder_id=sweep.task_id,
                    task_id=sweep.task_id,
                    action="conc_sweep",
                    ttl_sec=600,
                )
                await first.tasks.transition(sweep.task_id, "running")
            first.shared_state.save(session_dir)
        finally:
            await first.stop()
        return sweep

    @staticmethod
    async def _resume(coordinator) -> None:
        """What a resumed leg does before its first tick admits anything: replay, then the first phase decision."""
        await coordinator.writeback.replay_resume_if_needed()
        await coordinator.phase_machine.advance_phase_if_needed()

    @staticmethod
    async def _live_sweeps(coordinator) -> list:
        live = (*await coordinator.tasks.queued(), *await coordinator.tasks.running())
        return [task for task in live if task.kind == "conc_sweep"]

    @pytest.mark.asyncio
    async def test_a_resume_after_recover_starts_the_sweep_again(self, session_dir):
        from hyperloom.common.owner_scope import local_owner_scope
        from hyperloom.inference_optimizer.session.resume_guard import confirm_scope_gone

        lost = await self._enter_sweep(session_dir, start=True)
        assert lost.requires_lanes, "the sweep must hold lanes for recover to find it"
        confirmed = confirm_scope_gone(
            session_dir,
            owner_scope=local_owner_scope(),
            reason="the machine is gone",
            current_scope=self._NEW_SCOPE,
        )
        assert confirmed["cancelled_tasks"] == [lost.task_id]

        resumed = Coordinator(session_dir, backends=_backends_full())
        try:
            assert (await resumed.tasks.get(lost.task_id)).state == "cancelled"
            assert resumed.shared_state.last_conc_sweep == {}
            await self._resume(resumed)

            live = await self._live_sweeps(resumed)
            assert [t.task_id for t in live] != [], "no conc_sweep was dispatched; SWEEP would wait out its grant"
            assert lost.task_id not in {t.task_id for t in live}
            assert resumed.shared_state.conc_sweep_granted_until_unix > time.time()
        finally:
            await resumed.stop()

    @pytest.mark.asyncio
    async def test_a_resume_with_the_sweep_still_queued_does_not_start_a_second_one(self, session_dir):
        queued = await self._enter_sweep(session_dir, start=False)

        resumed = Coordinator(session_dir, backends=_backends_full())
        try:
            await self._resume(resumed)
            assert [t.task_id for t in await self._live_sweeps(resumed)] == [queued.task_id]
        finally:
            await resumed.stop()

    @pytest.mark.asyncio
    async def test_a_queued_sweep_whose_grant_ran_out_while_the_session_was_down_is_owed_it_again(
        self,
        session_dir,
    ):
        """Otherwise the resumed leg's first decision leaves SWEEP on its spent budget and cancels the sweep unrun."""
        queued = await self._enter_sweep(session_dir, start=False)
        state = SharedState.load_or_init(session_dir)
        now = time.time()
        # A three-hour session whose SWEEP share is long spent, resumed after the grant ran out.
        state.max_minutes = 180
        state.start_ts = "2026-06-02T10:00:00+00:00"
        state.phase_budget_pct = {"SWEEP": 0.05, "CLOSE": 0.02}
        state.phase_started_unix = now - 600.0
        state.elapsed_charged_sec = 7000.0
        state.conc_sweep_granted_until_unix = now - 60.0
        state.save(session_dir)

        resumed = Coordinator(session_dir, backends=_backends_full())
        try:
            await self._resume(resumed)
            assert resumed.shared_state.phase == "SWEEP"
            assert [t.task_id for t in await self._live_sweeps(resumed)] == [queued.task_id]
            assert resumed.shared_state.conc_sweep_granted_until_unix >= now + queued.lease_ttl_sec - 60.0
        finally:
            await resumed.stop()

    @pytest.mark.asyncio
    async def test_a_resumed_leg_that_reaches_sweep_later_enters_it_only_through_the_transition(self, session_dir):
        """The resume check belongs to the process's first decision; a later SWEEP has its own entry hook."""
        SharedState(session_id="mid", phase="KERNEL_AGENT").save(session_dir)

        resumed = Coordinator(session_dir, backends=_backends_full())
        try:
            await resumed.phase_sweep.resume_unsettled_sweep()
            resumed.shared_state.phase = "SWEEP"
            await resumed.phase_sweep.resume_unsettled_sweep()
            assert await self._live_sweeps(resumed) == []
        finally:
            await resumed.stop()
