# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator resume tests.

Covers resume detection, ``replay_for_resume`` rebuilding undecided
pending_proposals, pruned_families preservation, lazy replay on the first
``tick()``, and reopening the phase machine for a session that stopped in CLOSE.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
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
        "robustness": MockRobustnessBackend(),
    }


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


class TestAClosedSessionIsReopenedOnResume:
    """CLOSE has no way out, so a leg that loads it would tick in it to the end.

    The machine's only terminal phase, and the run loop stops on ``stop_reason``
    rather than on the phase. A resumed leg that keeps CLOSE therefore spends its
    whole new clock in a phase admitting nothing but ``report``,
    ``session_breakdown`` and ``recover``. Every design that stops early on the
    promise of "resume with more budget" rests on this being reopened.
    """

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
        """The flag means "the sequencer already wrote the breakdown".

        Carried into a leg that then never reaches CLOSE, it silences the
        end-of-run safety net that would have written one, and the leg finishes
        with no breakdown at all.
        """
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
        """Reopening the phase is only half of it; the round has to be admissible.

        A cold anchor is a positive ``baseline_tput``, which is what the singleton
        rule refuses repeats on -- so the leg would reopen at PRELUDE, decline to
        finish while the mark is set, decline to close while the clock is healthy,
        and have the one round that clears the mark denied on its way in. This is
        the last link in the chain the whole cold-anchor design rests on, and
        nothing above it can tell whether it holds.
        """
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
        assert c2.resumed_from["is_resume"] is True
        assert c2.resumed_from["event_count"] >= 1
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
        "robustness": MockBackend(silent, name="r"),
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
        assert any(p.verdict == "approve" for p in c1.state.pending_proposals.values())
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 0
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
        "robustness": MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(1)
        proposal_id = next(iter(c1.state.pending_proposals.keys()))
        await c1._handle_intent(
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
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
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
        for action in ("baseline", "profile", "explore"):
            await c1._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.PROPOSE_ACTION,
                    payload={"action_name": action, "predicted_gain_pct": 0.0},
                ),
            )
            tail = await c1.bus.tail(topic="proposal", n=1)
            proposal_ids.append(tail[0].msg_id)
            if action == "baseline":
                # profile/explore require baseline_tput > 0 (execution_order);
                # the real baseline action would have set this on completion.
                c1.shared_state.baseline_tput = 100.0

        await c1._handle_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": proposal_ids[0], "verdict": "approve", "reasoning": "ok"},
            ),
        )
        await c1._handle_intent(
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
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 1
        restored = next(iter(c2.state.pending_proposals.values()))
        assert restored.action_name == "explore"
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_resume_preserves_pruned_and_restores_pending(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1._handle_intent(
            "robustness",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"family": "deep_kernel", "reason": "x"},
            ),
        )
        await c1._handle_intent(
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
        await c2.replay_for_resume()
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
        "robustness": MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1._handle_intent(
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
        assert c2.resumed_from["rebuilt"] is False
        await c2.tick(1)
        assert c2.resumed_from["rebuilt"] is True
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
            # The command line already-deployed robustness monitor copies send.
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
            "HYPERLOOM_KERNEL_AGENT_ROOT",
            "KERNEL_AGENT_ENV",
            "USER_DATA_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_noop_when_root_already_set(self, monkeypatch, capsys):
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/opt/kernel-agent")
        cli_preflight._load_kernel_agent_env_fallback()
        out = capsys.readouterr()
        assert out.out == ""
        assert out.err == ""

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

    def test_aborts_when_env_file_does_not_define_root(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "# stale file\nexport SOMETHING_ELSE=1\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            cli_preflight._load_kernel_agent_env_fallback()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "HYPERLOOM_KERNEL_AGENT_ROOT" in err
        assert "stale" in err or "malformed" in err

    def test_sources_vars_on_success(self, tmp_path, monkeypatch, capsys):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "# valid env file\n"
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/opt/kernel-agent\n"
            "export KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        cli_preflight._load_kernel_agent_env_fallback()
        import os as _os

        assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/opt/kernel-agent"
        assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "INFO"
        out = capsys.readouterr().out
        assert "loaded" in out
        assert "kernel-agent" in out

    def test_env_wins_over_file(self, tmp_path, monkeypatch, capsys):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/from/file\nexport KERNEL_AGENT_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("KERNEL_AGENT_LOG_LEVEL", "DEBUG")
        cli_preflight._load_kernel_agent_env_fallback()
        import os as _os

        assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/from/file"
        assert _os.environ["KERNEL_AGENT_LOG_LEVEL"] == "DEBUG"

    def test_credential_fallback_block_parses_without_warnings(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        """The installer emits credentials as a conditional block (#1169).

        The comparison line inside it contains ``=`` without being an
        assignment, so a parser that splits on ``=`` alone would invent a key
        and warn about it on every launch.
        """
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/opt/kernel-agent\n"
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
            # The loader writes straight into os.environ, which monkeypatch
            # cannot roll back; a leaked credential reshapes later auth tests.
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
            "export HYPERLOOM_KERNEL_AGENT_ROOT=/from/custom\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("KERNEL_AGENT_ENV", str(custom))
        monkeypatch.setenv("USER_DATA_PATH", "/nonexistent/should-not-be-used")
        cli_preflight._load_kernel_agent_env_fallback()
        import os as _os

        assert _os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] == "/from/custom"


# A stale/placeholder TRACELENS_ROOT is corrected from the installer-written env
# file; template placeholders are treated as unset.
class TestTracelensRootEnvCorrection:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        for var in (
            "HYPERLOOM_KERNEL_AGENT_ROOT",
            "KERNEL_AGENT_ENV",
            "USER_DATA_PATH",
            "TRACELENS_ROOT",
            "MAGPIE_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

    def _write_env_file(self, tmp_path, tracelens_dir):
        runtime = tmp_path / "runtime"
        runtime.mkdir(exist_ok=True)
        (runtime / "kernel-agent.env.sh").write_text(
            f"export HYPERLOOM_KERNEL_AGENT_ROOT=/opt/kernel-agent\nexport TRACELENS_ROOT='{tracelens_dir}'\n",
            encoding="utf-8",
        )

    def test_corrects_invalid_inherited_root_from_env_file(self, tmp_path, monkeypatch, capsys):
        """Root set + inherited TRACELENS_ROOT points nowhere → corrected from file."""
        good = tmp_path / "deps" / "TraceLens"
        good.mkdir(parents=True)
        self._write_env_file(tmp_path, good)
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/opt/kernel-agent")
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "ghost" / "TraceLens"))

        cli_preflight._load_kernel_agent_env_fallback()

        import os as _os

        assert _os.environ["TRACELENS_ROOT"] == str(good)
        assert "TRACELENS_ROOT" in capsys.readouterr().err

    def test_keeps_valid_inherited_root(self, tmp_path, monkeypatch):
        """A valid inherited TRACELENS_ROOT wins over the env file (env-wins)."""
        file_dir = tmp_path / "file" / "TraceLens"
        file_dir.mkdir(parents=True)
        inherited = tmp_path / "inherited" / "TraceLens"
        inherited.mkdir(parents=True)
        self._write_env_file(tmp_path, file_dir)
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/opt/kernel-agent")
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
        monkeypatch.setenv("TRACELENS_ROOT", str(inherited))

        cli_preflight._load_kernel_agent_env_fallback()

        import os as _os

        assert _os.environ["TRACELENS_ROOT"] == str(inherited)

    def test_magpie_path_is_not_corrected(self, tmp_path, monkeypatch):
        """MAGPIE_PATH is out of scope: a merely-existing non-checkout dir in the
        env file must NOT be promoted to an explicit MAGPIE_PATH override."""
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        magpie_dir = tmp_path / "not-a-magpie-checkout"
        magpie_dir.mkdir()
        (runtime / "kernel-agent.env.sh").write_text(
            f"export HYPERLOOM_KERNEL_AGENT_ROOT=/opt/kernel-agent\nexport MAGPIE_PATH='{magpie_dir}'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/opt/kernel-agent")
        monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

        cli_preflight._load_kernel_agent_env_fallback()

        import os as _os

        assert _os.environ.get("MAGPIE_PATH") is None

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
