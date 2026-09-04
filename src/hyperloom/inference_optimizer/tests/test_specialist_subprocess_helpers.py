# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``specialists.subprocess_`` helpers: worktree pick/setup,
claude argv assembly, patch discovery, and done-file parse/unwrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _pick_worktree_base,
    _setup_worktree,
)


class _CP:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# -- _pick_worktree_base ---------------------------------------------------
def test_pick_worktree_base_none(tmp_path: Path) -> None:
    # directory without .git yields None
    (tmp_path / "plain").mkdir()
    assert _pick_worktree_base((str(tmp_path / "plain"), str(tmp_path / "absent"))) is None


def test_pick_worktree_base_finds_git(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    assert _pick_worktree_base(("/nonexistent", str(repo))) == repo


# -- _setup_worktree -------------------------------------------------------
def test_setup_worktree_reuses_existing(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    out, err = _setup_worktree(tmp_path, wt, "branch-x")
    assert out == wt and err == ""


def test_setup_worktree_spawn_failure(tmp_path: Path, monkeypatch) -> None:
    def _boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ss.subprocess, "run", _boom)
    out, err = _setup_worktree(tmp_path, tmp_path / "wt2", "b")
    assert out is None and "failed to spawn" in err


def test_setup_worktree_nonzero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _CP(1, "", "fatal: oops"))
    out, err = _setup_worktree(tmp_path, tmp_path / "wt3", "b")
    assert out is None and "rc=1" in err


def test_setup_worktree_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _CP(0))
    target = tmp_path / "wt4"
    out, err = _setup_worktree(tmp_path, target, "b")
    assert out == target and err == ""


# -- _build_claude_cmd -----------------------------------------------------
def _dispatcher(**cfg_over: Any) -> SpecialistSubprocessDispatcher:
    cfg = SpecialistSubprocessConfig(**cfg_over)
    return SpecialistSubprocessDispatcher(cfg)


def test_build_claude_cmd_full(tmp_path: Path) -> None:
    fw = tmp_path / "fw"
    fw.mkdir()
    d = _dispatcher(
        model="claude-opus-4-7",
        mcp_config_path="/cfg/mcp.json",
        framework_source_roots=(str(fw),),
        extra_claude_args=("--foo", "bar"),
    )
    wt = tmp_path / "wt"
    wt.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    sys_file = ws / "system_prompt.md"
    user_file = tmp_path / "p.txt"
    cmd = d._build_claude_cmd(
        system_prompt_file=sys_file,
        system_prompt="SYS",
        user_prompt_file=user_file,
        workspace=ws,
        worktree=wt,
        disallowed_tools=frozenset({"KillShell", "SlashCommand"}),
    )
    assert "--model" in cmd and "claude-opus-4-7" in cmd
    assert cmd[cmd.index("--system-prompt-file") + 1] == str(sys_file)
    assert "--mcp-config" in cmd and "/cfg/mcp.json" in cmd
    assert "--allowedTools" not in cmd
    assert "-p" not in cmd
    deny_idx = cmd.index("--disallowedTools") + 1
    denied = set(cmd[deny_idx].split(","))
    assert "KillShell" in denied and "SlashCommand" in denied
    assert str(wt) in cmd and str(ws) in cmd and str(fw) in cmd
    assert cmd[-2:] == ["--foo", "bar"]


def test_build_claude_cmd_minimal_no_model_no_mcp(tmp_path: Path) -> None:
    d = _dispatcher()
    ws = tmp_path / "ws"
    ws.mkdir()
    sys_file = ws / "system_prompt.md"
    cmd = d._build_claude_cmd(
        system_prompt_file=sys_file,
        system_prompt="SYS",
        user_prompt_file=tmp_path / "p.txt",
        workspace=ws,
        worktree=None,
    )
    assert "--model" not in cmd
    assert "--mcp-config" not in cmd
    assert "--allowedTools" not in cmd
    assert "-p" not in cmd
    assert "--agents" in cmd


def test_build_claude_cmd_injects_leaf_agents_when_task_allowed(tmp_path: Path) -> None:

    from hyperloom.orchestrator.specialists.leaf import LEAF_AGENT_NAME

    d = _dispatcher()
    ws = tmp_path / "ws"
    ws.mkdir()
    sys_file = ws / "system_prompt.md"
    cmd = d._build_claude_cmd(
        system_prompt_file=sys_file,
        system_prompt="SYS",
        user_prompt_file=tmp_path / "p.txt",
        workspace=ws,
        worktree=None,
    )
    agents_idx = cmd.index("--agents") + 1
    agents = json.loads(cmd[agents_idx])
    assert LEAF_AGENT_NAME in agents
    assert "Task" not in agents[LEAF_AGENT_NAME]["tools"]


# -- _build_hermes_cmd ----------------------------------------------------
def test_build_hermes_cmd_is_narrowed_and_uses_configured_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ss, "running_in_container", lambda: True)
    binary = tmp_path / "custom-hermes"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    d = _dispatcher(
        agent_backend="hermes",
        hermes_executable=str(binary),
        hermes_profile="isolated",
        hermes_provider="openai-codex",
        model="gpt-5.6-sol",
        hermes_external_sandbox=True,
    )

    cmd = d._build_hermes_cmd(system_prompt="SYS", user_prompt="USER")

    assert cmd[0] == str(binary)
    assert "--safe-mode" in cmd
    assert cmd[cmd.index("--toolsets") + 1] == "terminal,file"
    assert "--yolo" not in cmd
    assert cmd[-2] == "-z"


# -- _collect_patches ------------------------------------------------------
def test_collect_patches(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    (wt / "patches").mkdir(parents=True)
    (wt / "patches" / "a.patch").write_text("p", encoding="utf-8")
    (wt / "patches" / "b.diff").write_text("d", encoding="utf-8")
    (wt / "patches" / "ignore.txt").write_text("x", encoding="utf-8")
    ws = tmp_path / "ws"
    out, roots = SpecialistSubprocessDispatcher._collect_patches(wt, ws)
    names = sorted(Path(p).name for p in out)
    assert names == ["a.patch", "b.diff"]
    assert roots == {}


def test_collect_patches_none_worktree(tmp_path: Path) -> None:
    assert SpecialistSubprocessDispatcher._collect_patches(None, tmp_path) == ([], {})


# -- _read_done ------------------------------------------------------------
def test_read_done_missing(tmp_path: Path) -> None:
    assert SpecialistSubprocessDispatcher._read_done(tmp_path / "absent.json") is None


def test_read_done_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text("{bad", encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(p) is None


def test_read_done_non_dict(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(p) is None


def test_read_done_flat_dict(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text(json.dumps({"empty": True, "proposal_set": []}), encoding="utf-8")
    assert SpecialistSubprocessDispatcher._read_done(p) == {"empty": True, "proposal_set": []}


def test_read_done_unwraps_intent_envelope(tmp_path: Path) -> None:
    p = tmp_path / "done.json"
    p.write_text(
        json.dumps(
            {
                "intent_type": "specialist_done",
                "domain": "kernel_switch_specialist",
                "payload": {"proposal_set": [{"name": "v1"}], "empty": False},
            }
        ),
        encoding="utf-8",
    )
    out = SpecialistSubprocessDispatcher._read_done(p)
    assert out["domain"] == "kernel_switch_specialist"
    assert out["proposal_set"] == [{"name": "v1"}]
    assert "intent_type" not in out and "payload" not in out
