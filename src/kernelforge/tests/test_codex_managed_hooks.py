# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for Codex managed hook materialization and runner."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from kernelforge.agent_backends.base import AgentRunSpec
from kernelforge.agent_backends.codex_hook_runner import handle_pre_tool_use, handle_stop, main
from kernelforge.agent_backends.codex_managed_hooks import materialize_codex_managed_hooks
from kernelforge.kernel_rewrite_controller.opportunity_agent import _AnalysisToolGuard


def test_materialize_writes_config_and_state(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(staging),
        hooks=_AnalysisToolGuard(staging).hooks(),
    )
    codex_home = tmp_path / "codex_home"
    materialize_codex_managed_hooks(codex_home=codex_home, spec=spec)
    config = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert "[[hooks.PreToolUse]]" in config
    assert "[[hooks.Stop]]" in config
    state = json.loads((codex_home / "forge_guard_state.json").read_text(encoding="utf-8"))
    assert state["staging_root"] == str(staging.resolve())


def test_pre_tool_use_denies_shell(tmp_path: Path) -> None:
    state = {"staging_root": str(tmp_path), "deny_shell_tools": True}
    payload = {"tool_name": "shell", "tool_input": {"command": "echo hi"}}
    result = handle_pre_tool_use(payload, state)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pre_tool_use_denies_write_outside_staging(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    state = {"staging_root": str(staging), "deny_shell_tools": True}
    payload = {
        "tool_name": "apply_patch",
        "tool_input": {"file_path": str(tmp_path / "outside.txt")},
    }
    result = handle_pre_tool_use(payload, state)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_stop_blocks_when_pending_rejections_exist(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    draft = staging / "draft-a"
    draft.mkdir(parents=True)
    (draft / "task.json").write_text("{}", encoding="utf-8")
    (draft / "rejection.json").write_text(
        json.dumps({"reason": "missing operator_id"}),
        encoding="utf-8",
    )
    state_path = tmp_path / "forge_guard_state.json"
    state_path.write_text(
        json.dumps({"staging_root": str(staging), "max_stop_denials": 3, "stop_denials": 0}),
        encoding="utf-8",
    )
    result = handle_stop({}, json.loads(state_path.read_text()), state_path)
    assert result.get("decision") == "block"
    assert "draft-a" in result.get("reason", "")


def test_cli_main_pre_tool_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    state_path = tmp_path / "forge_guard_state.json"
    state_path.write_text(
        json.dumps({"staging_root": str(staging), "deny_shell_tools": True}),
        encoding="utf-8",
    )
    stdin = StringIO(json.dumps({"hook_event_name": "PreToolUse", "tool_name": "bash", "tool_input": {}}))
    monkeypatch.setattr("sys.stdin", stdin)
    assert main(["--state", str(state_path), "pre_tool_use"]) == 0
