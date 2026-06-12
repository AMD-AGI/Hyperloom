# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for specialist_subprocess helpers: worktree pick/setup/teardown,
claude argv assembly, patch discovery, and done-file parse/unwrap."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator import specialist_subprocess as ss
from inference_optimizer.orchestrator.specialist_subprocess import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
    _pick_worktree_base,
    _setup_worktree,
    _teardown_worktree,
)


class _CP:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# -- _pick_worktree_base ---------------------------------------------------
def test_pick_worktree_base_none(tmp_path: Path) -> None:
    # directory without .git -> skipped -> None
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


# -- _teardown_worktree ----------------------------------------------------
def test_teardown_worktree_missing(tmp_path: Path) -> None:
    _teardown_worktree(tmp_path, tmp_path / "absent")  # must not raise


def test_teardown_worktree_removes(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / ".git").mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    calls: list[list[str]] = []

    def _run(cmd, *a, **k):
        calls.append(list(cmd))
        return _CP(0)

    monkeypatch.setattr(ss.subprocess, "run", _run)
    _teardown_worktree(base, wt)
    # git worktree remove was attempted, then rm -rf cleanup
    assert any("worktree" in c for c in calls)


def test_teardown_worktree_no_git_base_falls_back_to_rmtree(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "f.txt").write_text("x", encoding="utf-8")
    _teardown_worktree(None, wt)
    assert not wt.exists()


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
    cmd = d._build_claude_cmd(
        prompt_file=tmp_path / "p.txt",
        workspace=ws,
        worktree=wt,
        allowed_tools=("read_file", "emit_intent", "edit"),
    )
    assert "--model" in cmd and "claude-opus-4-7" in cmd
    assert "--mcp-config" in cmd and "/cfg/mcp.json" in cmd
    # emit_intent dropped from whitelist
    tools_idx = cmd.index("--allowedTools") + 1
    assert "emit_intent" not in cmd[tools_idx]
    assert "read_file" in cmd[tools_idx] and "edit" in cmd[tools_idx]
    # worktree appears before workspace in --add-dir order
    assert str(wt) in cmd and str(ws) in cmd and str(fw) in cmd
    # operator escape-hatch args appended verbatim
    assert cmd[-2:] == ["--foo", "bar"]


def test_build_claude_cmd_minimal_no_model_no_mcp(tmp_path: Path) -> None:
    d = _dispatcher()
    ws = tmp_path / "ws"
    ws.mkdir()
    cmd = d._build_claude_cmd(
        prompt_file=tmp_path / "p.txt",
        workspace=ws,
        worktree=None,
        allowed_tools=("emit_intent",),  # only the dropped tool -> no --allowedTools
    )
    assert "--model" not in cmd
    assert "--mcp-config" not in cmd
    assert "--allowedTools" not in cmd


# -- _collect_patches ------------------------------------------------------
def test_collect_patches(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    (wt / "patches").mkdir(parents=True)
    (wt / "patches" / "a.patch").write_text("p", encoding="utf-8")
    (wt / "patches" / "b.diff").write_text("d", encoding="utf-8")
    (wt / "patches" / "ignore.txt").write_text("x", encoding="utf-8")
    ws = tmp_path / "ws"  # no patches dir
    out = SpecialistSubprocessDispatcher._collect_patches(wt, ws)
    names = sorted(Path(p).name for p in out)
    assert names == ["a.patch", "b.diff"]


def test_collect_patches_none_worktree(tmp_path: Path) -> None:
    assert SpecialistSubprocessDispatcher._collect_patches(None, tmp_path) == []


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
        json.dumps({
            "intent_type": "specialist_done",
            "domain": "kernel_switch_specialist",
            "payload": {"proposal_set": [{"name": "v1"}], "empty": False},
        }),
        encoding="utf-8",
    )
    out = SpecialistSubprocessDispatcher._read_done(p)
    # outer keys merged with inner payload, envelope keys dropped
    assert out["domain"] == "kernel_switch_specialist"
    assert out["proposal_set"] == [{"name": "v1"}]
    assert "intent_type" not in out and "payload" not in out
