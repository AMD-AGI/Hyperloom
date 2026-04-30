"""Tests for orchestrator/multi_cli/launcher.py — generic restart-loop pane."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.multi_cli.agent_card import (
    AgentCard,
    RestartPolicy,
)
from inference_optimizer.orchestrator.multi_cli.launcher import (
    DEFAULT_TMUX_SESSION,
    LOG_SUBDIR,
    LauncherError,
    MultiCLILauncher,
    WORK_SUBDIR,
)


def _stub_card(
    tmp_path: Path,
    name: str,
    *,
    backend: str = "claude",
    continue_flag: bool = True,
    extra: dict | None = None,
) -> AgentCard:
    card_dir = tmp_path / "agents" / name
    card_dir.mkdir(parents=True, exist_ok=True)
    prompt = card_dir / "system_prompt.md"
    prompt.write_text(f"# {name} stub prompt\n", encoding="utf-8")
    return AgentCard(
        name=name,
        role="executor" if backend == "claude" else "critic",
        backend=backend,
        card_path=card_dir / "agent_card.yaml",
        card_dir=card_dir,
        capabilities=("send_message",),
        allowed_modes=(),
        enabled=True,
        system_prompt="system_prompt.md",
        restart_policy=RestartPolicy(
            max_restarts=3,
            backoff_seconds=1,
            continue_flag=continue_flag,
        ),
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# stage()
# ---------------------------------------------------------------------------
def test_stage_writes_env_and_pane_scripts(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"executor": _stub_card(tmp_path, "executor", backend="claude")}
    launcher = MultiCLILauncher(
        session_dir=session, cards=cards,
        env={"MODEL_PATH": "/models/foo", "MAX_HOURS": "0.05"},
    )
    staged = launcher.stage()

    assert "executor" in staged
    agent = staged["executor"]
    assert agent.pane_script.is_file()
    assert agent.log_file.parent == session / LOG_SUBDIR
    assert agent.stop_file == session / "STOP_AGENT_executor"
    assert agent.inbox_path.is_file()  # pre-touched by launcher
    assert agent.outbox_path.is_file()

    env_file = session / WORK_SUBDIR / ".env"
    assert env_file.is_file()
    body = env_file.read_text()
    assert "SESSION_DIR=" in body
    assert "MODEL_PATH=/models/foo" in body or "MODEL_PATH='/models/foo'" in body
    assert "MAX_HOURS='0.05'" in body or "MAX_HOURS=0.05" in body


def test_stage_skips_disabled_agents(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    enabled = _stub_card(tmp_path, "executor")
    disabled = _stub_card(tmp_path, "critic", backend="codex")
    object.__setattr__(disabled, "enabled", False)
    cards = {"executor": enabled, "critic": disabled}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    assert set(staged.keys()) == {"executor"}


def test_stage_raises_when_no_cards(tmp_path: Path):
    launcher = MultiCLILauncher(session_dir=tmp_path / "s", cards={})
    with pytest.raises(LauncherError, match="no agent cards"):
        launcher.stage()


def test_pane_script_is_executable(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"executor": _stub_card(tmp_path, "executor")}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    mode = staged["executor"].pane_script.stat().st_mode
    assert mode & stat.S_IXUSR
    assert mode & stat.S_IRUSR


# ---------------------------------------------------------------------------
# Pane script body shape (per backend)
# ---------------------------------------------------------------------------
def test_claude_pane_emits_restart_loop(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"executor": _stub_card(tmp_path, "executor", backend="claude")}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    body = staged["executor"].pane_script.read_text()
    # restart-loop hallmarks
    assert "while [ ! -f \"$STOP_FILE\" ]" in body
    assert "claude --print" in body
    assert "--output-format stream-json --verbose" in body
    assert "--continue" in body  # restart_policy.continue_flag=True
    assert "--system-prompt" in body
    assert "--add-dir" in body
    # System prompt is base64-encoded inline
    assert "base64 -d" in body


def test_pane_isolation_per_agent_cwd_and_claude_config(tmp_path: Path):
    """Phase B — every staged agent must get its own per-pane cwd
    (``$WORK/<name>/``) and per-pane Claude config dir
    (``$WORK/<name>/.claude``). The pane script must ``cd`` there and
    export ``CLAUDE_CONFIG_DIR`` before invoking ``claude --print``."""
    session = tmp_path / "session"
    session.mkdir()
    cards = {
        "executor": _stub_card(tmp_path, "executor", backend="claude"),
        "critic": _stub_card(tmp_path, "critic", backend="claude"),
    }
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()

    exec_pane = session / WORK_SUBDIR / "executor"
    crit_pane = session / WORK_SUBDIR / "critic"
    assert exec_pane.is_dir() and (exec_pane / ".claude").is_dir()
    assert crit_pane.is_dir() and (crit_pane / ".claude").is_dir()
    assert exec_pane != crit_pane

    exec_body = staged["executor"].pane_script.read_text()
    crit_body = staged["critic"].pane_script.read_text()
    assert f'cd "{exec_pane}"' in exec_body
    assert f'cd "{crit_pane}"' in crit_body
    assert f'export CLAUDE_CONFIG_DIR="{exec_pane / ".claude"}"' in exec_body
    assert f'export CLAUDE_CONFIG_DIR="{crit_pane / ".claude"}"' in crit_body


def test_claude_pane_emits_turn_end_heartbeat_fallback(tmp_path: Path):
    """Phase E: when claude exits without writing any envelope to its
    outbox, the pane script must append a single send_message
    (topic=heartbeat) so the bus reflects the stall and triage can
    react. See standalone_agent_design §13.9.x — analysis-paralysis
    guard."""
    session = tmp_path / "session"
    session.mkdir()
    cards = {"executor": _stub_card(tmp_path, "executor", backend="claude")}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    body = staged["executor"].pane_script.read_text()
    # We capture line counts before/after the claude call.
    assert "OUTBOX_LINES_BEFORE=" in body
    assert "OUTBOX_LINES_AFTER=" in body
    # When unchanged the script appends a structured send_message envelope.
    assert "no_intent_emitted_this_turn" in body
    assert '\\"intent_type\\":\\"send_message\\"' in body
    assert '\\"topic\\":\\"heartbeat\\"' in body


def test_claude_pane_omits_continue_when_disabled(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"executor": _stub_card(tmp_path, "executor", backend="claude",
                                    continue_flag=False)}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    body = staged["executor"].pane_script.read_text()
    # Variable assignment uses an empty quoted string when continue is off.
    assert "CONTINUE_FLAG=''" in body or 'CONTINUE_FLAG=""' in body


def test_codex_pane_uses_conversation_log(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"critic": _stub_card(tmp_path, "critic", backend="codex",
                                  continue_flag=False,
                                  extra={"conversation_log": "history.jsonl"})}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    body = staged["critic"].pane_script.read_text()
    assert "codex --prompt-file" in body
    assert "history.jsonl" in body
    # No `claude --print --continue` invocation in a Codex pane (the
    # comment header may mention --continue while explaining the
    # workaround; we only care that the actual binary line lacks it).
    assert "claude --print" not in body


def test_mock_pane_just_writes_heartbeat(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"hb": _stub_card(tmp_path, "hb", backend="mock")}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    body = staged["hb"].pane_script.read_text()
    assert "intent_type" in body
    assert "heartbeat" in body
    # Mock launcher must NOT shell out to claude/codex.
    assert "claude --print" not in body
    assert "codex --prompt-file" not in body


# ---------------------------------------------------------------------------
# Mock pane end-to-end (no Claude/Codex required)
# ---------------------------------------------------------------------------
def test_mock_pane_actually_runs(tmp_path: Path):
    """Sanity check: the generated mock script writes a heartbeat envelope
    into outbox.jsonl when invoked."""
    session = tmp_path / "session"
    session.mkdir()
    cards = {"hb": _stub_card(tmp_path, "hb", backend="mock")}
    # Tight loop so the test finishes fast.
    object.__setattr__(cards["hb"], "restart_policy",
                       RestartPolicy(max_restarts=1, backoff_seconds=0,
                                     continue_flag=False))
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    res = subprocess.run(
        ["bash", str(staged["hb"].pane_script)],
        capture_output=True, text=True, timeout=10,
    )
    assert res.returncode == 0, res.stderr
    out = staged["hb"].outbox_path.read_text().strip().splitlines()
    assert len(out) >= 1
    import json
    env = json.loads(out[0])
    assert env["kind"] == "intent"
    assert env["intent_type"] == "send_message"
    assert env["from_agent"] == "hb"


# ---------------------------------------------------------------------------
# request_stop_all
# ---------------------------------------------------------------------------
def test_request_stop_all_drops_sentinels(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    cards = {
        "a": _stub_card(tmp_path, "a"),
        "b": _stub_card(tmp_path, "b", backend="codex"),
    }
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    staged = launcher.stage()
    launcher.request_stop_all(staged)
    assert (session / "STOP_AGENT_a").is_file()
    assert (session / "STOP_AGENT_b").is_file()


# ---------------------------------------------------------------------------
# launch() — when tmux missing
# ---------------------------------------------------------------------------
def test_launch_raises_when_tmux_missing(tmp_path: Path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    cards = {"executor": _stub_card(tmp_path, "executor")}
    launcher = MultiCLILauncher(session_dir=session, cards=cards)
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.multi_cli.launcher._have_tmux",
        lambda: False,
    )
    with pytest.raises(LauncherError, match="tmux"):
        launcher.launch()
