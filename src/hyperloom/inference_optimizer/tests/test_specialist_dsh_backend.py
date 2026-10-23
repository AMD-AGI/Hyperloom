# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist DeepSeek Harness (``dsh``) CLI integration."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.common import llm_config
from hyperloom.common.deadline import Deadline

import hyperloom.orchestrator.specialists.dsh_launch as dsh_launch
import hyperloom.orchestrator.specialists.subprocess_ as sp
from hyperloom.inference_optimizer.trace.parse_dsh_session import parse_dsh_session_trace

SpecialistAgentUnavailableError = sp.SpecialistAgentUnavailableError
SpecialistSubprocessConfig = sp.SpecialistSubprocessConfig
SpecialistSubprocessDispatcher = sp.SpecialistSubprocessDispatcher

_PROVIDER_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "LLM_GATEWAY_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
)

_OPENAI_ENV: dict[str, str] = {
    "OPENAI_BASE_URL": "https://gateway.invalid/Unified/v1",
    "OPENAI_API_KEY": "openai-side-key",
}

_DONE_PAYLOAD: dict[str, object] = {
    "gap_canonical_id": "gap.test.example",
    "domain": "research_scout_specialist",
    "proposal_set": [],
    "patches_written": [],
    "summary": "fake dsh specialist run",
    "confidence": 0.5,
}


def _session_rows(*, depth: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One session log: the header row the harness writes first, then ``rows``."""
    return [{"type": "session", "version": 3, "id": f"session-{depth}", "delegationDepth": depth}, *rows]


def _assistant(step: int, content: list[dict[str, Any]], usage: dict[str, int]) -> dict[str, Any]:
    return {
        "type": "assistant/message",
        "data": {"turn": 1, "step": step, "message": {"role": "assistant", "content": content}, "usage": usage},
    }


# The records a real ``dsh --profile headless`` run wrote for one shell call and a reply, trimmed to the fields read.
_ROOT_SESSION: list[dict[str, Any]] = _session_rows(
    depth=0,
    rows=[
        _assistant(
            1,
            [{"type": "tool-call", "id": "call_1", "name": "bash", "arguments": '{"command": "echo hi > hello.txt"}'}],
            {"inputTokens": 6542, "outputTokens": 67, "totalTokens": 7377, "cacheReadTokens": 768},
        ),
        {
            "type": "tool/call",
            "data": {"turn": 1, "step": 1, "callId": "call_1", "name": "bash", "arguments": '{"command": "echo hi"}'},
        },
        _assistant(
            2,
            [{"type": "text", "text": "DONE"}],
            {"inputTokens": 222, "outputTokens": 3, "totalTokens": 7393, "cacheReadTokens": 7168, "reasoningTokens": 0},
        ),
        {"type": "turn/end", "data": {"turn": 1, "reason": {"kind": "completed"}}},
    ],
)

# A subagent the root spawned: its spend and calls are the task's, its reply is not.
_CHILD_SESSION: list[dict[str, Any]] = _session_rows(
    depth=1,
    rows=[
        {
            "type": "tool/call",
            "data": {"turn": 1, "step": 1, "callId": "call_c", "name": "web_search", "arguments": '{"query": "x"}'},
        },
        _assistant(1, [{"type": "text", "text": "child reply"}], {"inputTokens": 10, "outputTokens": 5}),
    ],
)


def _write_session_log(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` one JSON line each, the way the harness appends its log as it runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _PatchLoader(yaml.SafeLoader):
    """Reads a dsh patch; the ``!!js`` expressions the harness evaluates stay as ``("js", source)``."""


_PatchLoader.add_multi_constructor(
    "tag:yaml.org,2002:js", lambda loader, _suffix, node: ("js", loader.construct_scalar(node))
)


def _patch_rows(text: str) -> dict[str, dict[str, Any]]:
    """The patch's rows by id; the ``insert`` rows of the MCP clients are listed under their own ids."""
    rows: dict[str, dict[str, Any]] = {}
    for row in yaml.load(text, Loader=_PatchLoader):
        for inserted in row.get("insert", []):
            rows[inserted["id"]] = inserted
        if "id" in row:
            rows[row["id"]] = row
    return rows


def _pin_provider_env(monkeypatch: pytest.MonkeyPatch, shape: dict[str, str]) -> None:
    for key in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in shape.items():
        monkeypatch.setenv(key, value)


def _select_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, llm_config.AGENT_BACKEND_DEEPSEEK)
    monkeypatch.setattr(llm_config, "deepseek_harness_sdk_installed", lambda: True)


_FAKE_DSH = """#!{python}
import json, os, sys
from pathlib import Path

home = Path(os.environ["DSH_HOME"])
patch = sys.argv[sys.argv.index("--patch") + 1]
record = {{
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "env": {{k: os.environ.get(k) for k in (
        "DSH_HOME", "DSH_PERMISSION_MODE", "DSH_TELEMETRY_MODE", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY",
    )}},
    "instructions": (home / "AGENTS.md").read_text(encoding="utf-8"),
    "patch": Path(patch).read_text(encoding="utf-8"),
}}
(home.parent / "fake_dsh_record.json").write_text(json.dumps(record), encoding="utf-8")
(home / "profiles" / "headless" / "node_modules").mkdir(parents=True, exist_ok=True)
for rel, rows in {sessions}:
    target = home / "sessions" / rel / "session.v3.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")
if {write_done}:
    Path(os.getcwd(), "specialist_done.json").write_text(json.dumps({done}), encoding="utf-8")
print("DONE")
sys.exit({exit_code})
"""


def _fake_dsh(
    tmp_path: Path,
    *,
    sessions: list[tuple[str, list[dict[str, Any]]]],
    write_done: bool = True,
    exit_code: int = 0,
) -> Path:
    path = tmp_path / "bin" / "dsh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _FAKE_DSH.format(
            python=sys.executable,
            sessions=repr(sessions),
            write_done=write_done,
            done=repr(_DONE_PAYLOAD),
            exit_code=exit_code,
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


async def _run(dispatcher: Any, tmp_path: Path, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "task_id": "t-dsh",
        "workspace": tmp_path / "workspace",
        "worktree": None,
        "worktree_base": None,
        "system_prompt": "be a research scout",
        "user_prompt": "find the gap",
        "disallowed_tools": frozenset({"KillShell"}),
        "deadline": Deadline.after(60.0),
    }
    kwargs.update(overrides)
    return await dispatcher.run(sp.SpecialistSpawn(**kwargs))


def _dispatcher(dsh: Path | str, **config: Any) -> Any:
    return SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=llm_config.AGENT_BACKEND_DEEPSEEK,
            dsh_executable=str(dsh),
            model=config.pop("model", "deepseek-v4-flash"),
            poll_interval_seconds=0.05,
            **config,
        )
    )


@pytest.mark.asyncio
async def test_a_deepseek_selection_runs_the_specialist_on_the_dsh_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting the harness spawns ``dsh`` headless and harvests its session log like the other CLIs."""
    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    dsh = _fake_dsh(
        tmp_path,
        sessions=[("ws/session-0", _ROOT_SESSION), ("ws/session-1", _CHILD_SESSION)],
    )

    result = await _run(_dispatcher(dsh), tmp_path)

    workspace = tmp_path / "workspace"
    assert result.exit_code == 0, f"error={result.error!r} log={Path(result.process_log_path).read_text()!r}"
    assert result.done_payload is not None
    assert result.done_payload["summary"] == "fake dsh specialist run"

    record = json.loads((workspace / "fake_dsh_record.json").read_text(encoding="utf-8"))
    patch_flag = record["argv"].index("--patch")
    assert record["argv"][:patch_flag] == ["--profile", "headless"]
    # The launcher consumes one ``--`` and the headless app the other, so the task reaches the model verbatim.
    assert record["argv"][patch_flag + 2 :] == ["--", "--", "find the gap"]
    assert record["cwd"] == str(workspace)
    assert record["instructions"] == "be a research scout"
    assert record["env"] == {
        "DSH_HOME": str(workspace / ".dsh"),
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DSH_TELEMETRY_MODE": "DISABLED",
        "DEEPSEEK_BASE_URL": _OPENAI_ENV["OPENAI_BASE_URL"],
        "DEEPSEEK_API_KEY": _OPENAI_ENV["OPENAI_API_KEY"],
    }
    rows = _patch_rows(record["patch"])
    assert rows["agent-default-model"]["config"] == {"provider": "deepseek-official", "model": "deepseek-v4-flash"}
    assert rows["session-title-llm"]["disabled"] is True
    assert rows["session-persistence-jsonl"]["config"]["compression"] == "none"

    trace = result.trace
    assert trace.usage == {
        "input_tokens": 6542 + 222 + 10,
        "output_tokens": 67 + 3 + 5,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": 768 + 7168,
        "reasoning_output_tokens": 0,
    }
    assert trace.response == "DONE"
    assert [call["tool"] for call in trace.tool_calls] == ["Bash", "WebSearch"]
    assert len(trace.turn_usages) == 3
    # The profile tree dsh materializes per DSH_HOME is megabytes per task; the session logs stay.
    assert not (workspace / ".dsh" / "profiles").exists()
    assert (workspace / ".dsh" / "sessions").is_dir()


_SILENT_DSH = """#!{python}
import json, os, sys, time
from pathlib import Path

log = Path(os.environ["DSH_HOME"]) / "sessions" / "ws" / "session-0" / "session.v3.jsonl"
log.parent.mkdir(parents=True, exist_ok=True)
for step in range(6):
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({{"step": step}}) + "\\n")
    time.sleep(0.4)
Path(os.getcwd(), "specialist_done.json").write_text(json.dumps({done}), encoding="utf-8")
"""


@pytest.mark.asyncio
async def test_a_dsh_run_that_writes_only_its_session_log_is_not_reaped_as_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dsh prints only its final message, so its step records are the only sign it is working."""
    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    dsh = tmp_path / "bin" / "dsh"
    dsh.parent.mkdir(parents=True)
    dsh.write_text(_SILENT_DSH.format(python=sys.executable, done=repr(_DONE_PAYLOAD)), encoding="utf-8")
    dsh.chmod(dsh.stat().st_mode | stat.S_IEXEC)

    result = await _run(_dispatcher(dsh, heartbeat_stale_seconds=1.0), tmp_path)

    assert result.error == "", result.error
    assert result.done_payload is not None


@pytest.mark.asyncio
async def test_a_failed_dsh_turn_reaches_the_task_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider failure the harness records on ``turn/end`` is the task's error, not a bare exit code."""
    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    failed = _session_rows(
        depth=0,
        rows=[
            {
                "type": "turn/end",
                "data": {
                    "turn": 1,
                    "reason": {"kind": "error", "error": {"code": "provider", "message": "429 rate limited"}},
                },
            }
        ],
    )
    dsh = _fake_dsh(tmp_path, sessions=[("ws/session-0", failed)], write_done=False, exit_code=1)

    result = await _run(_dispatcher(dsh), tmp_path)

    assert result.done_payload is None
    assert "dsh: provider: 429 rate limited" in (result.error or "")


@pytest.mark.asyncio
async def test_a_user_prompt_beyond_the_argv_limit_fails_the_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dsh`` headless reads its task only from argv, which the kernel caps per argument."""
    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    monkeypatch.setattr(sp.subprocess, "Popen", lambda *_a, **_k: pytest.fail("an oversized task must not spawn"))

    result = await _run(_dispatcher(_fake_dsh(tmp_path, sessions=[])), tmp_path, user_prompt="x" * 131072)

    assert result.done_payload is None
    assert "unavailable" in (result.error or "")
    assert "argument" in (result.error or "")


@pytest.mark.asyncio
async def test_a_missing_dsh_cli_fails_the_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    monkeypatch.setattr(sp, "resolve_dsh_executable", lambda explicit="": "")
    monkeypatch.setattr(sp.subprocess, "Popen", lambda *_a, **_k: pytest.fail("no dsh means no spawn"))

    result = await _run(_dispatcher(""), tmp_path)

    assert result.done_payload is None
    assert "no dsh CLI found" in (result.error or "")


@pytest.mark.asyncio
async def test_an_unconfigured_gateway_fails_the_dsh_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider_env(monkeypatch, {})
    monkeypatch.setattr(sp.subprocess, "Popen", lambda *_a, **_k: pytest.fail("no gateway means no spawn"))

    result = await _run(_dispatcher(_fake_dsh(tmp_path, sessions=[])), tmp_path)

    assert result.done_payload is None
    assert "OPENAI_BASE_URL" in (result.error or "")


def test_mcp_servers_become_dsh_client_rows_without_their_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Claude-style MCP config the CLI generates is wired into dsh; secret values stay in the child env."""
    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "pr_monitor": {
                        "type": "http",
                        "url": "http://pr-monitor.invalid/mcp",
                        "headers": {"Authorization": "Bearer ${PR_TOKEN}"},
                    },
                    "local": {"command": "/usr/bin/tool", "args": ["--serve"], "env": {"TOOL_KEY": "${TOOL_KEY}"}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PR_TOKEN", "pr-secret")
    monkeypatch.setenv("TOOL_KEY", "tool-secret")
    request = dsh_launch.DshLaunchRequest(
        executable="/usr/bin/dsh",
        model="deepseek-v4-flash",
        mcp_config_path=str(mcp_config),
        workspace=tmp_path / "workspace",
        system_prompt="sys",
        user_prompt="usr",
    )

    launch = dsh_launch.build_dsh_launch(request, dict(_OPENAI_ENV))
    cmd, env_additions = launch.cmd, launch.env_additions

    patch = Path(cmd[cmd.index("--patch") + 1]).read_text(encoding="utf-8")
    assert "pr-secret" not in patch and "tool-secret" not in patch
    rows = _patch_rows(patch)
    http = rows["mcp-pr_monitor"]["config"]
    assert (http["transport"], http["url"]) == ("streamable-http", "http://pr-monitor.invalid/mcp")
    stdio = rows["mcp-local"]["config"]
    assert (stdio["transport"], stdio["command"], stdio["args"]) == ("stdio", "/usr/bin/tool", ["--serve"])
    private = {k: v for k, v in env_additions.items() if k.startswith("HYPERLOOM_DSH_MCP_ENV_")}
    assert sorted(private.values()) == ["Bearer pr-secret", "tool-secret"]
    (header_ref,) = http["headers"].values()
    (env_ref,) = stdio["env"].values()
    assert {header_ref[1], env_ref[1]} == {f"process.env.{name}" for name in private}


def test_a_cut_off_dsh_session_log_still_yields_its_complete_records(tmp_path: Path) -> None:
    """A reaped run leaves a half-written final line; the records before it are still the run's."""
    home = tmp_path / ".dsh"
    log = home / "sessions" / "ws" / "session-0" / "session.v3.jsonl"
    _write_session_log(log, _ROOT_SESSION)
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "assistant/mess')

    trace = parse_dsh_session_trace(home)
    assert trace.usage is not None
    assert trace.usage["output_tokens"] == 67 + 3
    assert trace.response == "DONE"


def test_a_dsh_run_without_a_session_log_is_reported_not_read_as_zero_spend(tmp_path: Path) -> None:
    trace = parse_dsh_session_trace(tmp_path / ".dsh")
    assert trace.usage is None
    assert "no session log" in (trace.error or "")


def test_the_specialist_executor_pins_the_dsh_cli_for_a_deepseek_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boot-time config names the harness, its CLI and the harness model together."""
    import hyperloom.orchestrator.specialists.runner as runner_mod
    from hyperloom.inference_optimizer.cli.executors import _build_specialist_executor

    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    _select_deepseek(monkeypatch)
    monkeypatch.setattr(sp, "resolve_dsh_executable", lambda explicit="": "/opt/venv/bin/dsh")
    captured: dict[str, Any] = {}

    class _Runner:
        def __init__(self, *, subprocess_config: Any, **_kwargs: Any) -> None:
            captured["config"] = subprocess_config

    monkeypatch.setattr(runner_mod, "SpecialistRunner", _Runner)
    args = argparse.Namespace(
        claude_model="claude-opus-4-6",
        codex_model="gpt-5.5",
        specialist_model=None,
        specialist_max_turns=3,
        specialist_mcp_config=None,
    )

    _build_specialist_executor(args, session_dir=tmp_path, knowledge_plane=None)

    config = captured["config"]
    assert config.agent_backend == llm_config.AGENT_BACKEND_DEEPSEEK
    assert config.dsh_executable == "/opt/venv/bin/dsh"
    assert config.model == llm_config.resolve_forge_llm_model(llm_config.AGENT_BACKEND_DEEPSEEK)


def test_the_specialist_executor_refuses_to_start_without_a_dsh_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hyperloom.inference_optimizer.cli.executors import _build_specialist_executor

    _pin_provider_env(monkeypatch, _OPENAI_ENV)
    _select_deepseek(monkeypatch)
    monkeypatch.setattr(sp, "resolve_dsh_executable", lambda explicit="": "")
    args = argparse.Namespace(
        claude_model="claude-opus-4-6",
        codex_model="gpt-5.5",
        specialist_model=None,
        specialist_max_turns=3,
        specialist_mcp_config=None,
    )

    with pytest.raises(RuntimeError, match="dsh"):
        _build_specialist_executor(args, session_dir=tmp_path, knowledge_plane=None)
