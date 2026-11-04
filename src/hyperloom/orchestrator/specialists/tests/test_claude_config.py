# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session CLI configuration: Claude Code isolation, Codex tables, and where the token may live."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom.orchestrator.specialists.agent_runner import AgentConfigError, McpEndpoint
from hyperloom.orchestrator.specialists.tests._agent_fixtures import (
    TOKEN,
    TOKEN_ENV,
    make_runner,
    make_spec,
    read_all_files,
    wait_finished,
)

DUMP_ENV = (
    "import json, os, pathlib\n"
    "pathlib.Path('seen_env.json').write_text(json.dumps({k: os.environ.get(k) for k in "
    "['CLAUDE_CONFIG_DIR', 'CLAUDE_CODE_PLUGIN_DIRS', 'CODEX_HOME', 'HYPERLOOM_CODEX_MCP_ENV_0', '"
    + TOKEN_ENV
    + "']}))\n"
    "pathlib.Path('specialist_done.json').write_text('{}')\n"
)


async def run_to_end(tmp_path: Path, spec, runner=None):
    runner = runner or make_runner(tmp_path / "session")
    handle = await runner.launch(spec)
    status = await wait_finished(runner, handle)
    await runner.collect(handle)
    return runner, status


def seen_env(workspace: Path) -> dict[str, str | None]:
    return json.loads((workspace / "seen_env.json").read_text(encoding="utf-8"))


async def test_default_gives_each_session_its_own_config_dir_without_user_plugins(tmp_path: Path) -> None:
    for name in ("a", "b"):
        ws = tmp_path / name
        spec = make_spec(
            ws,
            task_id=name,
            code=DUMP_ENV,
            env={"CLAUDE_CONFIG_DIR": str(tmp_path / "user-config"), "CLAUDE_CODE_PLUGIN_DIRS": "/user/plugins"},
        )
        await run_to_end(tmp_path, spec, make_runner(tmp_path / f"s-{name}"))
        seen = seen_env(ws)
        assert seen["CLAUDE_CONFIG_DIR"] == str(ws / ".claude-config")
        assert seen["CLAUDE_CODE_PLUGIN_DIRS"] is None
        assert oct((ws / ".claude-config").stat().st_mode & 0o777) == "0o700"
        settings = json.loads((ws / ".claude-config" / "settings.json").read_text(encoding="utf-8"))
        assert settings == {"enabledPlugins": {}}
    assert seen_env(tmp_path / "a")["CLAUDE_CONFIG_DIR"] != seen_env(tmp_path / "b")["CLAUDE_CONFIG_DIR"]


async def test_inherit_keeps_the_callers_config_dir_but_never_plugin_dirs(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    spec = make_spec(
        ws,
        code=DUMP_ENV,
        claude_config_dir="inherit",
        env={"CLAUDE_CONFIG_DIR": str(tmp_path / "user-config"), "CLAUDE_CODE_PLUGIN_DIRS": "/user/plugins"},
    )
    await run_to_end(tmp_path, spec)
    seen = seen_env(ws)
    assert seen["CLAUDE_CONFIG_DIR"] == str(tmp_path / "user-config")
    assert seen["CLAUDE_CODE_PLUGIN_DIRS"] is None
    assert not (ws / ".claude-config").exists()


async def test_two_live_sessions_cannot_share_one_claude_config_dir(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    first = await runner.launch(make_spec(tmp_path / "w", task_id="one"))
    try:
        with pytest.raises(AgentConfigError, match="already used by a live session"):
            await runner.launch(make_spec(tmp_path / "w", task_id="two"))
    finally:
        await runner.cancel(first)
        await wait_finished(runner, first)
        await runner.collect(first)
    # Once the first one is over, a restart in the same workspace is allowed.
    again = await runner.launch(make_spec(tmp_path / "w", task_id="one"))
    await runner.cancel(again)
    await wait_finished(runner, again)
    await runner.collect(again)


async def test_claude_mcp_config_names_the_token_variable_and_never_its_value(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(ws))
    try:
        path = ws / "runtime" / "specialist_mcp.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {
            "mcpServers": {
                "tools": {
                    "type": "http",
                    "url": "http://127.0.0.1:9/mcp",
                    "headers": {"Authorization": "Bearer ${" + TOKEN_ENV + "}"},
                }
            }
        }
        assert oct(path.stat().st_mode & 0o777) == "0o600"
        argv = handle.session.proc.args
        assert argv[-2:] == ["--mcp-config", str(path)]
        assert TOKEN not in " ".join(argv)
    finally:
        await runner.cancel(handle)
        await wait_finished(runner, handle)
        await runner.collect(handle)


async def test_argv_that_already_carries_mcp_config_is_refused(tmp_path: Path) -> None:
    spec = make_spec(tmp_path / "w", argv=("claude", "--mcp-config", "/x.json"))
    with pytest.raises(AgentConfigError, match="already carries --mcp-config"):
        await make_runner(tmp_path / "session").launch(spec)


async def test_codex_tables_use_env_http_headers_and_the_token_reaches_only_the_environment(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    spec = make_spec(ws, code=DUMP_ENV, backend="codex")
    codex_home = ws / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text('developer_instructions = "be careful"\n', encoding="utf-8")
    await run_to_end(tmp_path, spec)
    text = (codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'developer_instructions = "be careful"' in text  # what the existing builder wrote is kept
    assert "[mcp_servers.tools]" in text
    assert 'url = "http://127.0.0.1:9/mcp"' in text
    assert '"Authorization" = "HYPERLOOM_CODEX_MCP_ENV_0"' in text
    assert TOKEN not in text
    assert oct((codex_home / "config.toml").stat().st_mode & 0o777) == "0o600"
    seen = seen_env(ws)
    assert seen["CODEX_HOME"] == str(codex_home)
    assert seen["HYPERLOOM_CODEX_MCP_ENV_0"] == "Bearer " + TOKEN


async def test_codex_relaunch_in_the_same_workspace_does_not_duplicate_tables(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    for _ in range(2):
        await run_to_end(tmp_path, make_spec(ws, code=DUMP_ENV, backend="codex"), make_runner(tmp_path / "session"))
        (ws / "specialist_done.json").unlink()  # the earlier result was consumed before the restart
    text = (ws / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert text.count("[mcp_servers.tools]") == 1


async def test_codex_refuses_a_server_the_config_already_defines(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    (ws / ".codex").mkdir(parents=True)
    (ws / ".codex" / "config.toml").write_text('[mcp_servers.tools]\nurl = "http://other"\n', encoding="utf-8")
    with pytest.raises(AgentConfigError, match="already defines MCP server 'tools'"):
        await make_runner(tmp_path / "session").launch(make_spec(ws, backend="codex"))


@pytest.mark.parametrize(
    ("spec_kwargs", "message"),
    [
        ({"env": {TOKEN_ENV: ""}}, "is not set in spec.env"),
        ({"deadline_unix": 1.0}, "already in the past"),
        ({"argv": ()}, "argv is empty"),
        ({"backend": "gemini"}, "is not claude or codex"),
        (
            {"mcp_servers": (McpEndpoint("bad name", "http://x", TOKEN_ENV),)},
            "is not a letter followed by",
        ),
        (
            {"mcp_servers": (McpEndpoint("a", "http://x", TOKEN_ENV), McpEndpoint("a", "http://y", TOKEN_ENV))},
            "duplicate MCP server names",
        ),
    ],
)
async def test_bad_specs_are_refused_before_anything_is_spawned(tmp_path: Path, spec_kwargs, message) -> None:
    spawned: list[object] = []
    runner = make_runner(tmp_path / "session", popen=lambda *a, **k: spawned.append(a))
    with pytest.raises(AgentConfigError, match=message):
        await runner.launch(make_spec(tmp_path / "w", **spec_kwargs))
    assert spawned == []


async def test_a_refused_launch_does_not_leave_the_claude_config_dir_claimed(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    bad = make_spec(tmp_path / "w", argv=("claude", "--mcp-config", "/x.json"))
    with pytest.raises(AgentConfigError):
        await runner.launch(bad)
    handle = await runner.launch(make_spec(tmp_path / "w"))  # the failed launch must not leave the dir claimed
    await runner.cancel(handle)
    await wait_finished(runner, handle)
    await runner.collect(handle)


async def test_agent_runtime_records_version_config_mode_and_plugins_from_the_init_message(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    code = (
        "import json, pathlib\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init', 'plugins': [{'name': 'p1'}]}), flush=True)\n"
        "pathlib.Path('specialist_done.json').write_text('{}')\n"
    )
    await run_to_end(tmp_path, make_spec(ws, code=code))
    runtime = json.loads((ws / "agent_runtime.json").read_text(encoding="utf-8"))
    assert runtime["version"] == "test-1.0"
    assert runtime["backend"] == "claude"
    assert runtime["config_dir_mode"] == "isolated"
    assert runtime["plugins"] == [{"name": "p1"}]
    assert runtime["mcp_servers"] == ["tools"]


async def test_agent_runtime_says_unknown_when_no_init_message_exists(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    await run_to_end(tmp_path, make_spec(ws, code="open('specialist_done.json','w').write('{}')"))
    assert json.loads((ws / "agent_runtime.json").read_text(encoding="utf-8"))["plugins"] == "unknown"


async def test_token_is_in_no_file_written_under_the_workspace_or_the_session(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    # The child prints the token to its own stdout: the log must be scrubbed on collect.
    code = (
        "import os, pathlib\n"
        f"print('leaked', os.environ['{TOKEN_ENV}'], flush=True)\n"
        f"pathlib.Path('child_saw_token').write_text(str(os.environ['{TOKEN_ENV}'] == '{TOKEN}'))\n"
        "pathlib.Path('specialist_done.json').write_text('{}')\n"
    )
    for backend in ("claude", "codex"):
        await run_to_end(tmp_path, make_spec(ws, code=code, backend=backend), make_runner(tmp_path / "session"))
        assert (ws / "child_saw_token").read_text(encoding="utf-8") == "True"  # it really was in the environment
        (ws / "child_saw_token").unlink()
        (ws / "specialist_done.json").unlink()
    for root in (ws, tmp_path / "session"):
        for path, data in read_all_files(root).items():
            assert TOKEN.encode() not in data, path
    assert b"[REDACTED]" in (ws / "process.log").read_bytes()


async def test_cancel_then_collect_leaves_no_process_behind(tmp_path: Path) -> None:
    runner = make_runner(tmp_path / "session")
    handle = await runner.launch(make_spec(tmp_path / "w"))
    pid = handle.pid
    await runner.cancel(handle)
    await runner.cancel(handle)  # idempotent
    status = await wait_finished(runner, handle)
    await runner.collect(handle)
    assert status.state == "exited"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_the_token_is_gone_from_the_log_before_the_pattern_redaction_copies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing redaction writes a temporary copy with default permissions; it must never see the token."""
    from hyperloom.orchestrator.specialists import agent_runner

    seen: list[bytes] = []
    real = agent_runner.redact_file_in_place

    def spy(path, **kwargs):
        seen.append(Path(path).read_bytes())
        real(path, **kwargs)

    monkeypatch.setattr(agent_runner, "redact_file_in_place", spy)
    ws = tmp_path / "w"
    code = f"import os\nprint(os.environ['{TOKEN_ENV}'], flush=True)\nopen('specialist_done.json', 'w').write('{{}}')\n"
    await run_to_end(tmp_path, make_spec(ws, code=code))
    assert seen and all(TOKEN.encode() not in data for data in seen)
    assert b"[REDACTED]" in (ws / "process.log").read_bytes()
    assert not list(ws.glob("process.log.*"))  # no temporary copy left behind


async def test_a_shared_codex_home_in_the_callers_env_is_replaced_by_the_task_private_one(tmp_path: Path) -> None:
    ws = tmp_path / "w"
    shared = tmp_path / "shared-codex"
    await run_to_end(tmp_path, make_spec(ws, code=DUMP_ENV, backend="codex", env={"CODEX_HOME": str(shared)}))
    assert seen_env(ws)["CODEX_HOME"] == str(ws / ".codex")
    assert not shared.exists()
