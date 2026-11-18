# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Claude specialist killed before its ``result`` row still has its token spend booked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.env_safety import RunAs
from hyperloom.orchestrator.specialists.launch_env import SANDBOX_HOME, LaunchPlace, _point_into_sandbox
from hyperloom.orchestrator.specialists.trace_harvest import harvest_agent_trace

_SESSION_ID = "b5c5d762-e7a2-42d0-846c-6349c6be1fa8"
_ZEROED = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stream_log(tmp_path: Path, *, result: dict[str, Any] | None = None) -> Path:
    """What a CLI streamed through a gateway that zeroes per-message usage."""
    rows: list[dict[str, Any]] = [
        {"type": "system", "subtype": "init", "session_id": _SESSION_ID, "model": "gpt-6.1-sol"},
        {"type": "assistant", "message": {"id": "msg_1", "model": "gpt-6.1-sol", "usage": dict(_ZEROED)}},
        {"type": "assistant", "message": {"id": "msg_2", "model": "gpt-6.1-sol", "usage": dict(_ZEROED)}},
    ]
    if result is not None:
        rows.append({"type": "result", "usage": result})
    log = tmp_path / "process.log"
    _write_lines(log, rows)
    return log


def _cli_transcript(config_dir: Path) -> None:
    """The CLI's own session record: each message appears once per content block, the last entry is final."""

    def entry(message_id: str, usage: dict[str, int]) -> dict[str, Any]:
        return {"type": "assistant", "message": {"id": message_id, "usage": usage}}

    _write_lines(
        config_dir / "projects" / "-some-workspace" / f"{_SESSION_ID}.jsonl",
        [
            entry(
                "msg_1",
                {
                    "input_tokens": 2152,
                    "cache_creation_input_tokens": 12900,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 10,
                },
            ),
            entry(
                "msg_1",
                {
                    "input_tokens": 2152,
                    "cache_creation_input_tokens": 12900,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 217,
                },
            ),
            entry(
                "msg_2",
                {
                    "input_tokens": 30,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 15052,
                    "output_tokens": 80,
                },
            ),
        ],
    )


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path / "claude-config"


def _env(config_dir: Path) -> dict[str, str]:
    return {"CLAUDE_CONFIG_DIR": str(config_dir)}


def _assert_booked(trace: Any) -> None:
    assert trace.usage is not None
    assert trace.usage["input_tokens"] == 2152 + 30
    assert trace.usage["cache_creation_input_tokens"] == 12900 + 500
    assert trace.usage["cache_read_input_tokens"] == 15052
    assert trace.usage["output_tokens"] == 217 + 80
    assert trace.usage["model"] == "gpt-6.1-sol"
    assert [turn["output_tokens"] for turn in trace.turn_usages] == [217, 80]


def test_a_run_killed_before_its_result_row_is_booked_from_the_cli_session_record(
    tmp_path: Path, config_dir: Path
) -> None:
    _cli_transcript(config_dir)

    trace = harvest_agent_trace("claude", tmp_path, _stream_log(tmp_path), _env(config_dir))

    _assert_booked(trace)


def test_the_result_row_stays_authoritative_when_the_run_finished(tmp_path: Path, config_dir: Path) -> None:
    _cli_transcript(config_dir)
    result = {
        "input_tokens": 50632,
        "cache_read_input_tokens": 291392,
        "cache_creation_input_tokens": 0,
        "output_tokens": 7542,
    }

    trace = harvest_agent_trace("claude", tmp_path, _stream_log(tmp_path, result=result), _env(config_dir))

    assert trace.usage is not None
    assert trace.usage["input_tokens"] == 50632
    assert trace.usage["cache_read_input_tokens"] == 291392


def test_a_run_with_no_session_record_keeps_the_usage_its_stream_reported(tmp_path: Path, config_dir: Path) -> None:
    trace = harvest_agent_trace("claude", tmp_path, _stream_log(tmp_path), _env(config_dir))

    assert trace.usage is None
    assert [turn["input_tokens"] for turn in trace.turn_usages] == [0, 0]


def test_a_sandboxed_run_is_booked_from_the_config_dir_its_workspace_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_home = tmp_path / "app-home"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(app_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(app_home / ".claude"))
    env = {"HOME": str(app_home), "CLAUDE_CONFIG_DIR": str(app_home / ".claude")}
    _point_into_sandbox(LaunchPlace(workspace), env)
    assert env["CLAUDE_CONFIG_DIR"] == str(workspace / ".claude-config")
    _cli_transcript(Path(env["CLAUDE_CONFIG_DIR"]))

    trace = harvest_agent_trace("claude", workspace, _stream_log(tmp_path), env)

    _assert_booked(trace)


def test_a_sandboxed_run_without_a_config_dir_is_booked_from_the_home_its_workspace_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_home = tmp_path / "app-home"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(app_home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    env = {"HOME": str(app_home)}
    _point_into_sandbox(LaunchPlace(workspace), env)
    assert env["HOME"] == str(workspace / SANDBOX_HOME)
    _cli_transcript(workspace / SANDBOX_HOME / ".claude")

    trace = harvest_agent_trace("claude", workspace, _stream_log(tmp_path), env)

    _assert_booked(trace)


def test_a_run_as_another_account_is_booked_from_that_accounts_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_home = tmp_path / "app-home"
    account_home = tmp_path / "account-home"
    monkeypatch.setenv("HOME", str(app_home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    env = {"HOME": str(app_home)}
    RunAs(user="agent", uid=2000, gid=2000, groups=(2000,), home=str(account_home)).apply_env(env)
    _cli_transcript(account_home / ".claude")

    trace = harvest_agent_trace("claude", tmp_path, _stream_log(tmp_path), env)

    _assert_booked(trace)


def test_the_coordinators_own_config_dir_is_not_read_for_a_child_launched_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_home = tmp_path / "app-home"
    monkeypatch.setenv("HOME", str(app_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(app_home / ".claude"))
    _cli_transcript(app_home / ".claude")

    trace = harvest_agent_trace("claude", tmp_path, _stream_log(tmp_path), {"HOME": str(tmp_path / "elsewhere")})

    assert trace.usage is None
