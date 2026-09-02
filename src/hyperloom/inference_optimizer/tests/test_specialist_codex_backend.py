# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist Codex CLI and Agent SDK integration.

Pins the bug an OpenAI-only end-to-end run exposed: the specialist dispatcher
hard-wired the Claude CLI, so an OpenAI-only deployment spawned a runtime with
no credential. Every specialist task died with ``subprocess_exit_code:1`` after
the CLI reported ``Not logged in · Please run /login``, and the run continued
without its research-scout and static-recon specialists.

The dispatcher must spawn the Codex CLI when only the OpenAI side is
configured, and must keep spawning the Claude CLI for the Anthropic-only and
both-configured shapes. The same suite covers secure sandbox selection,
stdin/role transport, provider env wiring, MCP translation, structured errors,
model precedence, and true in-process Codex Agent SDK dispatch.
"""

from __future__ import annotations

import argparse
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import hyperloom.orchestrator.roles.codex_agent as codex_agent
import hyperloom.orchestrator.specialists.subprocess_ as sp
from hyperloom.orchestrator.trace import parse_usage as pu

AGENT_BACKEND_CLAUDE = sp.AGENT_BACKEND_CLAUDE
AGENT_BACKEND_CODEX = sp.AGENT_BACKEND_CODEX
SpecialistAgentUnavailableError = sp.SpecialistAgentUnavailableError
SpecialistSubprocessConfig = sp.SpecialistSubprocessConfig
SpecialistSubprocessDispatcher = sp.SpecialistSubprocessDispatcher
_build_specialist_env = sp._build_specialist_env
resolve_codex_executable = sp.resolve_codex_executable
resolve_specialist_agent_backend = sp.resolve_specialist_agent_backend

# Every provider-shape signal ``llm_config`` consults, so a test can pin an
# exact deployment shape instead of inheriting the developer's own gateway.
_PROVIDER_ENV_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "LLM_GATEWAY_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "HYPERLOOM_CODEX_SANDBOX_MODE",
)

_OPENAI_ONLY_ENV: dict[str, str] = {
    "OPENAI_BASE_URL": "https://gateway.invalid/Unified/v1",
    "OPENAI_API_KEY": "openai-side-key",
}
_ANTHROPIC_ONLY_ENV: dict[str, str] = {
    "ANTHROPIC_BASE_URL": "https://gateway.invalid/Anthropic",
    "ANTHROPIC_API_KEY": "anthropic-side-key",
}
_BOTH_CONFIGURED_ENV: dict[str, str] = {**_OPENAI_ONLY_ENV, **_ANTHROPIC_ONLY_ENV}

# The specialist_done.json payload both fake CLIs write on a successful run.
_DONE_PAYLOAD: dict[str, object] = {
    "gap_canonical_id": "gap.test.example",
    "domain": "research_scout_specialist",
    "proposal_set": [{"name": "codex_variant", "extra_args": "", "extra_envs": {}, "reason": "fake"}],
    "patches_written": [],
    "empty": False,
    "summary": "fake codex specialist run",
    "confidence": 0.5,
}

# The event stream ``codex exec --json`` emits, captured verbatim from a real
# Codex CLI turn against the gateway (one JSON object per line).
_CODEX_JSONL: tuple[str, ...] = (
    '{"type":"thread.started","thread_id":"019fe0ee-1a4e-7dc0-9f05-b5bfb0c7fb7f"}',
    '{"type":"turn.started"}',
    (
        '{"type":"item.started","item":{"id":"item_0","type":"command_execution",'
        '"command":"/bin/bash -lc \'echo SANDBOX_OK > proof.txt\'",'
        '"aggregated_output":"","exit_code":null,"status":"in_progress"}}'
    ),
    (
        '{"type":"item.completed","item":{"id":"item_0","type":"command_execution",'
        '"command":"/bin/bash -lc \'echo SANDBOX_OK > proof.txt\'",'
        '"aggregated_output":"","exit_code":0,"status":"completed"}}'
    ),
    '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"DONE"}}',
    (
        '{"type":"turn.completed","usage":{"input_tokens":24099,'
        '"cached_input_tokens":11648,"output_tokens":44,"reasoning_output_tokens":0}}'
    ),
)

# What the Claude CLI actually printed in the failing run, before exiting 1.
_CLAUDE_NOT_LOGGED_IN_JSONL: tuple[str, ...] = (
    (
        '{"type":"assistant","message":{"content":[{"type":"text",'
        '"text":"Not logged in \\u00b7 Please run /login"}]},'
        '"error":"authentication_failed","is_api_error_message":true}'
    ),
    ('{"is_error":true,"result":"Not logged in \\u00b7 Please run /login","terminal_reason":"api_error"}'),
)


def _pin_provider_env(monkeypatch: pytest.MonkeyPatch, shape: dict[str, str]) -> None:
    """Clear every provider signal, then set exactly ``shape``."""
    for key in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in shape.items():
        monkeypatch.setenv(key, value)


def _write_executable(path: Path, body: str) -> Path:
    """Write an executable shell script at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_agent_bin_dir(tmp_path: Path) -> Path:
    """Create a bin dir holding a failing fake ``claude`` and a working fake ``codex``.

    ``claude`` reproduces the unauthenticated run observed in production: it
    prints the ``Not logged in`` stream-json rows and exits 1 without writing
    ``specialist_done.json``. ``codex`` prints the real ``codex exec --json``
    event stream and completes the specialist contract.
    """
    bin_dir = tmp_path / "bin"
    claude_log = "\n".join(f"echo {json.dumps(line)}" for line in _CLAUDE_NOT_LOGGED_IN_JSONL)
    _write_executable(
        bin_dir / "claude",
        f"#!/usr/bin/env bash\nset -e\n{claude_log}\nexit 1\n",
    )
    codex_log = "\n".join(f"echo {json.dumps(line)}" for line in _CODEX_JSONL)
    _write_executable(
        bin_dir / "codex",
        "#!/usr/bin/env bash\nset -e\n"
        f"{codex_log}\n"
        f"cat > \"$PWD/specialist_done.json\" <<'EOF'\n{json.dumps(_DONE_PAYLOAD)}\nEOF\n"
        "exit 0\n",
    )
    return bin_dir


@pytest.mark.asyncio
async def test_openai_only_deployment_runs_the_specialist_on_the_codex_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reproduction: an OpenAI-only deployment must not be handed the Claude CLI.

    The live run spawned ``claude --print --output-format stream-json`` with no
    Anthropic credential in the environment at all, so the CLI answered
    ``Not logged in · Please run /login`` and exited 1. Both affected domains
    exhausted their retries and the session lost them silently.
    """
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")
    bin_dir = _fake_agent_bin_dir(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    workspace = tmp_path / "workspace"
    dispatcher = SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            claude_executable=str(bin_dir / "claude"),
            model="gpt-5.5",
            poll_interval_seconds=0.05,
        )
    )
    result = await dispatcher.run(
        task_id="t-openai-only",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="be a research scout",
        user_prompt="find the gap",
        disallowed_tools=frozenset(),
        max_turns=2,
        wall_budget_sec=60.0,
    )

    assert result.exit_code == 0, (
        "the specialist subprocess failed in an OpenAI-only deployment; "
        f"error={result.error!r} log={Path(result.process_log_path).read_text(encoding='utf-8')!r}"
    )
    assert result.done_payload is not None, "no specialist_done.json was harvested"
    assert result.done_payload["summary"] == "fake codex specialist run"
    # Token spend, reply text and shell calls all have to survive the swap.
    assert result.usage == {
        "input_tokens": 24099,
        "output_tokens": 44,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": 11648,
        "reasoning_output_tokens": 0,
    }
    assert result.response == "DONE"
    assert [call["tool"] for call in result.tool_calls] == ["Bash"]


@pytest.mark.asyncio
async def test_codex_home_is_per_task_and_outside_any_temp_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CODEX_HOME`` is redirected into the task workspace, not a temp dir.

    Per-task so concurrent specialists and the operator's own Codex state stay
    independent. Deliberately *not* a ``tempfile`` directory: the Codex CLI
    refuses to create its PATH helper binaries under one and runs on without
    them. The workspace lives under the session dir, so it is a real location on
    every deployment.
    """
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "codex",
        "#!/usr/bin/env bash\nset -e\n"
        'printf \'{"codex_home":"%s","proposal_set":[],"empty":true}\\n\' "$CODEX_HOME"'
        ' > "$PWD/specialist_done.json"\nexit 0\n',
    )
    workspace = tmp_path / "workspace"
    dispatcher = SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            codex_executable=str(bin_dir / "codex"),
            poll_interval_seconds=0.05,
        )
    )
    result = await dispatcher.run(
        task_id="t-codex-home",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=60.0,
    )

    assert result.done_payload is not None
    codex_home = result.done_payload["codex_home"]
    assert codex_home == str(workspace / ".codex")
    assert Path(codex_home).is_dir()
    # ``tempfile`` would have produced a path outside the session entirely.
    assert str(workspace) in codex_home


# ---------------------------------------------------------------------------
# Backend selection per credential shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape_name", "shape", "expected"),
    [
        ("openai_only", _OPENAI_ONLY_ENV, AGENT_BACKEND_CODEX),
        ("anthropic_only", _ANTHROPIC_ONLY_ENV, AGENT_BACKEND_CLAUDE),
        ("both_configured", _BOTH_CONFIGURED_ENV, AGENT_BACKEND_CLAUDE),
        # Nothing configured keeps the historical default, so a deployment that
        # authenticates the Claude CLI by other means (a logged-in CLI, Bedrock)
        # is untouched by this selection.
        ("unconfigured", {}, AGENT_BACKEND_CLAUDE),
    ],
)
def test_agent_backend_follows_the_credential_shape(
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    shape: dict[str, str],
    expected: str,
) -> None:
    """Only the shape that cannot drive Claude at all is redirected to Codex."""
    _pin_provider_env(monkeypatch, shape)
    assert resolve_specialist_agent_backend() == expected, shape_name


def _build_cmd(tmp_path: Path, **cfg_overrides: object) -> list[str]:
    """Assemble the agent argv the dispatcher would spawn for one specialist."""
    workspace = tmp_path / "workspace"
    worktree = workspace / "worktree"
    framework = tmp_path / "framework"
    for path in (workspace, worktree, framework):
        path.mkdir(parents=True, exist_ok=True)
    prompt_file = workspace / "prompt.md"
    prompt_file.write_text("USER_PROMPT_SENTINEL", encoding="utf-8")
    cfg_overrides.setdefault("framework_source_roots", (str(framework),))
    cfg = SpecialistSubprocessConfig(**cfg_overrides)
    dispatcher = SpecialistSubprocessDispatcher(cfg)
    backend = dispatcher._agent_backend()
    if backend == AGENT_BACKEND_CODEX:
        import os as _os

        base_env = {
            k: v
            for k, v in _os.environ.items()
            if k in sp._SPECIALIST_ENV_ALLOWLIST | sp._SPECIALIST_SECRET_ENV_ALLOWLIST
        }
        cmd, _ = dispatcher._build_codex_launch(
            prompt_file=prompt_file,
            workspace=workspace,
            worktree=worktree,
            system_prompt="SYSTEM_INSTRUCTION_SENTINEL",
            base_env=base_env,
            probe_sandbox=False,
        )
        return cmd
    return dispatcher._build_claude_cmd(
        system_prompt_file=workspace / "system_prompt.md",
        system_prompt="SYSTEM_INSTRUCTION_SENTINEL",
        user_prompt_file=prompt_file,
        workspace=workspace,
        worktree=worktree,
        disallowed_tools=frozenset(),
    )


def test_openai_only_deployment_builds_a_codex_exec_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Codex argv is the counterpart of the Claude stream-json invocation."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    cmd = _build_cmd(tmp_path, codex_executable="/usr/bin/codex", model="gpt-5.5")

    assert cmd[0] == "/usr/bin/codex"
    assert cmd[1] == "exec"
    # ``--json`` replaces ``--print --output-format stream-json``.
    assert "--json" in cmd
    assert "--output-format" not in cmd and "--print" not in cmd
    # A workspace-only task may run outside a git checkout.
    assert "--skip-git-repo-check" in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-5.5"
    assert cmd[-1] == "-"
    rendered_argv = " ".join(cmd)
    assert "USER_PROMPT_SENTINEL" not in rendered_argv
    assert "SYSTEM_INSTRUCTION_SENTINEL" not in rendered_argv

    workspace = tmp_path / "workspace"
    # ``-C`` is the write-isolated worktree; workspace + framework roots are added.
    assert cmd[cmd.index("-C") + 1] == str(workspace / "worktree")
    add_dirs = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "--add-dir"]
    assert add_dirs == [str(workspace), str(tmp_path / "framework")]
    codex_config = workspace / ".codex" / "config.toml"
    assert codex_config.stat().st_mode & 0o777 == 0o600
    config_text = codex_config.read_text(encoding="utf-8")
    assert 'developer_instructions = "SYSTEM_INSTRUCTION_SENTINEL"' in config_text
    assert "USER_PROMPT_SENTINEL" not in config_text


def test_codex_argv_reuses_the_sdk_gateway_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway wiring comes from ``codex_session``, so CLI and SDK cannot drift.

    The API key crosses as an env-var NAME, never a value, so the secret stays
    out of the spawned process's argv.
    """
    from hyperloom.common.codex_session import CODEX_PROVIDER_NAME, resolve_codex_provider_config

    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    cmd = _build_cmd(tmp_path, codex_executable="/usr/bin/codex")

    overrides = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "-c"]
    for expected in resolve_codex_provider_config().overrides:
        assert expected in overrides
    assert f'model_provider="{CODEX_PROVIDER_NAME}"' in overrides
    # A specialist must not carry memory between tasks (matches the SDK session).
    assert "features.memories=false" in overrides
    # The credential crosses as an env-var NAME; its value stays out of argv,
    # where any user on the host could read it out of ``ps``.
    assert any(o.endswith('env_key="OPENAI_API_KEY"') for o in overrides)
    assert _OPENAI_ONLY_ENV["OPENAI_API_KEY"] not in " ".join(cmd)


def test_codex_argv_uses_workspace_write_sandbox_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The secure default is explicit workspace-write, never silent bypass."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)

    default_cmd = _build_cmd(tmp_path, codex_executable="/usr/bin/codex")
    assert default_cmd[default_cmd.index("--sandbox") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in default_cmd


def test_codex_argv_bypass_requires_mode_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full access is available only when the operator selects bypass."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")
    bypass_cmd = _build_cmd(tmp_path, codex_executable="/usr/bin/codex")
    assert "--dangerously-bypass-approvals-and-sandbox" in bypass_cmd
    assert "--sandbox" not in bypass_cmd


def test_codex_specialist_rejects_read_only_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A specialist cannot satisfy its result contract in read-only mode."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "read-only")
    with pytest.raises(SpecialistAgentUnavailableError, match="read-only"):
        _build_cmd(tmp_path, codex_executable="/usr/bin/codex")


def test_codex_argv_appends_operator_escape_hatch_before_the_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``extra_codex_args`` must not displace the positional prompt."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    cmd = _build_cmd(
        tmp_path,
        codex_executable="/usr/bin/codex",
        extra_codex_args=("--ephemeral",),
    )
    assert cmd[-2] == "--ephemeral"
    assert cmd[-1] == "-"


@pytest.mark.parametrize("shape", [_ANTHROPIC_ONLY_ENV, _BOTH_CONFIGURED_ENV, {}])
def test_claude_argv_is_unchanged_for_every_non_openai_only_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: dict[str, str],
) -> None:
    """Codex is a fallback for one shape, not a new default anywhere else."""
    _pin_provider_env(monkeypatch, shape)
    cmd = _build_cmd(tmp_path, claude_executable="/usr/bin/claude", model="claude-opus-5")

    assert cmd[0] == "/usr/bin/claude"
    assert "--print" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5"
    assert cmd[cmd.index("--system-prompt-file") + 1].endswith("prompt.md")
    assert "exec" not in cmd and "--json" not in cmd
    workspace = tmp_path / "workspace"
    add_dirs = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "--add-dir"]
    # Unchanged order: worktree (writes), workspace (done.json), framework roots.
    assert add_dirs == [str(workspace / "worktree"), str(workspace), str(tmp_path / "framework")]


@pytest.mark.parametrize(
    ("pinned", "shape"),
    [
        (AGENT_BACKEND_CLAUDE, _OPENAI_ONLY_ENV),
        (AGENT_BACKEND_CODEX, _ANTHROPIC_ONLY_ENV),
    ],
)
def test_explicit_config_backend_overrides_the_credential_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pinned: str,
    shape: dict[str, str],
) -> None:
    """An explicitly configured backend wins over the shape probe.

    The CLI resolves the backend once at boot and pins it, so a dispatch cannot
    disagree with the executable and model that were chosen alongside it.
    """
    _pin_provider_env(monkeypatch, {**shape, **_OPENAI_ONLY_ENV} if pinned == AGENT_BACKEND_CODEX else shape)
    cmd = _build_cmd(
        tmp_path,
        agent_backend=pinned,
        claude_executable="/usr/bin/claude",
        codex_executable="/usr/bin/codex",
    )
    assert cmd[0] == f"/usr/bin/{pinned}"


def test_unknown_configured_backend_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd backend is a configuration error, not a silent Claude fallback."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    with pytest.raises(SpecialistAgentUnavailableError, match="agent_backend"):
        _build_cmd(tmp_path, agent_backend="gemini")


# ---------------------------------------------------------------------------
# Codex runtime resolution
# ---------------------------------------------------------------------------


def test_resolve_codex_executable_prefers_explicit_then_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit config wins, then ``codex`` on PATH, then the SDK's runtime."""
    assert resolve_codex_executable("/opt/custom/codex") == "/opt/custom/codex"

    bin_dir = tmp_path / "bin"
    on_path = _write_executable(bin_dir / "codex", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert resolve_codex_executable() == str(on_path)


def test_resolve_codex_executable_falls_back_to_the_sdk_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pod that never ran the npm install still finds the pinned runtime."""
    monkeypatch.setenv("PATH", "/nonexistent")
    resolved = resolve_codex_executable()
    # The SDK runtime is an install-time dependency of ``openai-codex``; when it
    # is absent the resolver reports that rather than guessing a name.
    if resolved:
        assert Path(resolved).exists()
        assert Path(resolved).name == "codex"


@pytest.mark.asyncio
async def test_missing_codex_runtime_fails_the_task_instead_of_spawning_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Codex runtime must be reported, never absorbed into a Claude spawn.

    The whole point of this fix is that a missing specialist runtime silently
    degraded the run; a deployment with no usable CLI has to say so.
    """
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")
    monkeypatch.setattr(
        "hyperloom.orchestrator.specialists.subprocess_.resolve_codex_executable",
        lambda explicit="": "",
    )

    monkeypatch.setattr(
        sp.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("a missing codex runtime must not fall back to a spawn"),
    )

    dispatcher = SpecialistSubprocessDispatcher(SpecialistSubprocessConfig(poll_interval_seconds=0.05))
    result = await dispatcher.run(
        task_id="t-no-codex",
        workspace=tmp_path / "workspace",
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=60.0,
    )

    assert result.done_payload is None
    assert "codex" in result.error
    assert "unavailable" in result.error


@pytest.mark.asyncio
async def test_unconfigured_codex_gateway_fails_the_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OpenAI key with no base URL cannot be expressed as a Codex provider.

    Codex provider setup needs an explicit gateway URL, so this shape is
    reported as a task failure rather than spawning a CLI that cannot route.
    """
    _pin_provider_env(monkeypatch, {"OPENAI_API_KEY": "openai-side-key"})
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")
    dispatcher = SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(codex_executable="/usr/bin/codex", poll_interval_seconds=0.05)
    )
    result = await dispatcher.run(
        task_id="t-no-gateway",
        workspace=tmp_path / "workspace",
        worktree=None,
        worktree_base=None,
        system_prompt="sys",
        user_prompt="usr",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=60.0,
    )
    assert result.done_payload is None
    assert "not configured" in result.error


def test_cli_refuses_to_boot_an_openai_only_run_without_a_codex_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI has no usable fallback in this shape, so it must not start.

    Degrading to the in-process backend would hand every specialist task the
    Claude runtime that has no credential — the exact silent loss this fixes.
    """
    from hyperloom.inference_optimizer.cli import executors

    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setattr(
        "hyperloom.orchestrator.specialists.subprocess_.resolve_codex_executable",
        lambda explicit="": "",
    )
    args = argparse.Namespace(
        claude_model="claude-opus-5",
        codex_model="gpt-5.5",
        specialist_model="",
        specialist_max_turns=2,
        specialist_per_turn_max_seconds=60.0,
        specialist_dispatch_mode="subprocess",
        specialist_mcp_config="",
    )
    with pytest.raises(RuntimeError, match="codex"):
        executors._build_specialist_executor(args, session_dir=tmp_path, knowledge_plane=None)


# ---------------------------------------------------------------------------
# Codex JSONL parsers
#
# Twins of the Claude ``stream-json`` parsers, exercised against the event
# stream a real ``codex exec --json`` turn emitted (``_CODEX_JSONL``). The
# Claude twins are covered in ``test_parse_usage_unit.py``.
# ---------------------------------------------------------------------------


@pytest.fixture
def codex_log(tmp_path: Path) -> Path:
    """A ``process.log`` holding the captured ``codex exec --json`` stream."""
    path = tmp_path / "process.log"
    path.write_text("\n".join(_CODEX_JSONL) + "\n", encoding="utf-8")
    return path


def test_parse_codex_usage_maps_onto_the_canonical_counters(codex_log: Path) -> None:
    """Codex's counter names differ from Anthropic's and must be translated.

    ``cached_input_tokens`` is a cache *read*; Codex has no cache-write counter,
    so ``cache_creation_input_tokens`` stays ``None`` and the collector can still
    tell "no cache concept" from "zero cache hits". Reasoning tokens ride along
    rather than being folded into the visible output count.
    """
    assert pu.parse_codex_jsonl_usage(codex_log) == {
        "input_tokens": 24099,
        "output_tokens": 44,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": 11648,
        "reasoning_output_tokens": 0,
    }


def test_parse_codex_usage_sums_across_turns(tmp_path: Path) -> None:
    """Codex reports per-turn counts, so a session total has to add them up."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":2,'
        '"output_tokens":3,"reasoning_output_tokens":1}}\n'
        "garbled\n"
        '{"type":"turn.completed","usage":{"input_tokens":20,"cached_input_tokens":5,'
        '"output_tokens":7,"reasoning_output_tokens":4}}\n',
        encoding="utf-8",
    )
    assert pu.parse_codex_jsonl_usage(log) == {
        "input_tokens": 30,
        "output_tokens": 10,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": 7,
        "reasoning_output_tokens": 5,
    }


def test_parse_codex_turn_usages_keeps_one_row_per_turn(tmp_path: Path) -> None:
    """Per-turn rows are what let a multi-turn subprocess be traced turn by turn."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":7}}\n',
        encoding="utf-8",
    )
    usages = pu.parse_codex_jsonl_turn_usages(log)
    assert [(u["input_tokens"], u["output_tokens"]) for u in usages] == [(10, 3), (20, 7)]


def test_parse_codex_response_reads_the_agent_message(codex_log: Path) -> None:
    assert pu.parse_codex_jsonl_response(codex_log) == "DONE"


def test_parse_codex_response_joins_multiple_agent_messages(tmp_path: Path) -> None:
    """Codex emits no consolidated final row, so the messages are joined."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"first"}}\n'
        '{"type":"item.completed","item":{"id":"i1","type":"reasoning","text":"hidden"}}\n'
        '{"type":"item.completed","item":{"id":"i2","type":"agent_message","text":"second"}}\n',
        encoding="utf-8",
    )
    assert pu.parse_codex_jsonl_response(log) == "first\nsecond"


def test_parse_codex_tool_calls_uses_the_claude_tool_names(codex_log: Path) -> None:
    """Shell calls land in the intel ledger under the name Claude runs use.

    ``item.started`` and ``item.completed`` describe one call, so the item id
    de-duplicates them instead of double counting.
    """
    calls = pu.parse_codex_jsonl_tool_calls(codex_log)
    assert calls == [{"tool": "Bash", "query": "/bin/bash -lc 'echo SANDBOX_OK > proof.txt'"}]


def test_parse_codex_tool_calls_records_an_in_flight_call(tmp_path: Path) -> None:
    """A run killed mid-command still reports the call that was in flight."""
    log = tmp_path / "process.log"
    log.write_text(_CODEX_JSONL[2] + "\n", encoding="utf-8")
    assert [c["tool"] for c in pu.parse_codex_jsonl_tool_calls(log)] == ["Bash"]


def test_parse_codex_tool_calls_maps_searches_and_edits(tmp_path: Path) -> None:
    """Web searches and file changes get their Claude-side names too."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"item.completed","item":{"id":"i0","type":"web_search","query":"rocm flash attn"}}\n'
        '{"type":"item.completed","item":{"id":"i1","type":"file_change",'
        '"changes":[{"path":"a.py","kind":"update"},{"path":"b.py","kind":"add"}]}}\n'
        '{"type":"item.completed","item":{"id":"i2","type":"mcp_tool_call",'
        '"server":"pr_monitor","arguments":{"query":"open prs"}}}\n',
        encoding="utf-8",
    )
    calls = pu.parse_codex_jsonl_tool_calls(log)
    assert [c["tool"] for c in calls] == ["WebSearch", "Edit", "mcp_tool_call"]
    assert calls[0]["query"] == "rocm flash attn"
    assert calls[1]["query"] == "a.py, b.py"


def test_parse_codex_tool_calls_records_unknown_item_types_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``item.type`` is an open set: an unmodelled type is kept, not swallowed."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"item.completed","item":{"id":"i0","type":"quantum_tool","command":"qq"}}\n',
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        calls = pu.parse_codex_jsonl_tool_calls(log)
    assert calls == [{"tool": "quantum_tool", "query": "qq"}]
    assert "quantum_tool" in caplog.text


def test_parse_codex_tool_calls_ignores_non_tool_items(tmp_path: Path) -> None:
    """Messages, reasoning and to-do updates are not tool calls."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"hi"}}\n'
        '{"type":"item.completed","item":{"id":"i1","type":"reasoning","text":"thinking"}}\n'
        '{"type":"item.completed","item":{"id":"i2","type":"todo_list","items":[]}}\n',
        encoding="utf-8",
    )
    assert pu.parse_codex_jsonl_tool_calls(log) == []


def test_codex_parsers_tolerate_a_missing_or_truncated_log(tmp_path: Path) -> None:
    """Tolerant by contract, like every parser in this module."""
    missing = tmp_path / "absent.log"
    assert pu.parse_codex_jsonl_usage(missing) is None
    assert pu.parse_codex_jsonl_response(missing) is None
    assert pu.parse_codex_jsonl_turn_usages(missing) == []
    assert pu.parse_codex_jsonl_tool_calls(missing) == []

    truncated = tmp_path / "truncated.log"
    truncated.write_text(
        "\n".join(_CODEX_JSONL[:4]) + '\n{"type":"turn.completed","usa',
        encoding="utf-8",
    )
    # The turn never completed, so there is no usage -- but the calls survive.
    assert pu.parse_codex_jsonl_usage(truncated) is None
    assert pu.parse_codex_jsonl_turn_usages(truncated) == []
    assert [c["tool"] for c in pu.parse_codex_jsonl_tool_calls(truncated)] == ["Bash"]


def test_codex_usage_returns_none_when_no_counters_are_reported(tmp_path: Path) -> None:
    """An empty or unrecognized usage block is not a zero-token turn."""
    log = tmp_path / "process.log"
    log.write_text(
        '{"type":"turn.completed"}\n{"type":"turn.completed","usage":{"unrelated":1}}\n',
        encoding="utf-8",
    )
    assert pu.parse_codex_jsonl_usage(log) is None
    assert pu.parse_codex_jsonl_turn_usages(log) == []


# ---------------------------------------------------------------------------
# Secure Codex specialist integration regressions
# ---------------------------------------------------------------------------


def _enable_codex_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select canonical bypass with both required operator opt-ins."""
    monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")

def _successful_codex_script(path: Path) -> Path:
    """Write a fake Codex CLI that consumes stdin and completes the contract."""
    return _write_executable(
        path,
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "cat >/dev/null\n"
        "cat > \"$PWD/specialist_done.json\" <<'EOF'\n"
        '{"proposal_set":[],"empty":true,"summary":"ok"}\n'
        "EOF\n",
    )


class _NeverStartedGpuLease:
    """Ray lease double that records an erroneous setup-time launch."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.kwargs: dict[str, Any] = {}

    def start_async(self, _cmd: list[str], **kwargs: Any) -> None:
        self.start_calls += 1
        self.kwargs = dict(kwargs)
        raise RuntimeError("setup failure must precede Ray submission")


def test_specialist_env_keeps_llm_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway-only credentials must survive the specialist secret allowlist."""
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_KEY", "gateway-key-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/Unified/v1")

    child_env = _build_specialist_env()

    assert child_env["LLM_GATEWAY_KEY"] == "gateway-key-value"


@pytest.mark.asyncio
async def test_codex_secret_opt_out_masks_parent_provider_secrets_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver fallback cannot reintroduce parent secrets after explicit opt-out."""

    gateway_secret = "parent-gateway-secret"
    header_secret = "parent-literal-header-secret"
    _pin_provider_env(
        monkeypatch,
        {
            "OPENAI_BASE_URL": "https://gateway.invalid/Unified/v1",
            "LLM_GATEWAY_KEY": gateway_secret,
            "OPENAI_CUSTOM_HEADERS": f"user: {header_secret}",
        },
    )
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV", "0")
    _enable_codex_bypass(monkeypatch)
    resolver_overlays: list[dict[str, str]] = []
    resolved_configs: list[Any] = []
    real_resolver = sp.resolve_codex_provider_config

    def _recording_resolver(**kwargs: Any) -> Any:
        resolver_overlays.append(dict(kwargs.get("env") or {}))
        resolved = real_resolver(**kwargs)
        resolved_configs.append(resolved)
        return resolved

    monkeypatch.setattr(sp, "resolve_codex_provider_config", _recording_resolver)
    popen_calls = 0

    def _recording_popen(*_args: Any, **_kwargs: Any) -> None:
        nonlocal popen_calls
        popen_calls += 1

    monkeypatch.setattr(sp.subprocess, "Popen", _recording_popen)
    lease = _NeverStartedGpuLease()
    workspace = tmp_path / "workspace"
    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable="/usr/bin/codex",
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id="secret-opt-out",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="system",
        user_prompt="user",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=10.0,
        gpu_lease=lease,
    )

    assert resolver_overlays
    assert resolver_overlays[0].get("LLM_GATEWAY_KEY") == ""
    assert resolver_overlays[0].get("OPENAI_CUSTOM_HEADERS") == ""
    assert resolver_overlays[0].get("OPENAI_API_KEY") == ""
    assert resolved_configs == []
    assert "not configured" in result.error
    assert gateway_secret not in result.error
    assert header_secret not in result.error
    assert not (workspace / ".codex" / "config.toml").exists()
    assert popen_calls == 0
    assert lease.start_calls == 0
    assert gateway_secret not in repr(lease.kwargs)
    assert header_secret not in repr(lease.kwargs)


@pytest.mark.asyncio
async def test_codex_child_receives_provider_env_without_secrets_or_prompt_in_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider additions and gateway credentials belong in env, never argv."""

    secret = "gateway-secret-value"
    header_value = "private-gateway-user"
    _pin_provider_env(
        monkeypatch,
        {
            "OPENAI_BASE_URL": "https://gateway.invalid/Unified/v1",
            "LLM_GATEWAY_KEY": secret,
            "OPENAI_CUSTOM_HEADERS": f"user: {header_value}",
        },
    )
    _enable_codex_bypass(monkeypatch)
    fake_codex = _successful_codex_script(tmp_path / "bin" / "codex")
    captured: dict[str, Any] = {}
    real_popen = subprocess.Popen

    def _recording_popen(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.Popen:
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        captured["stdin"] = kwargs["stdin"]
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(sp.subprocess, "Popen", _recording_popen)
    workspace = tmp_path / "workspace"
    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable=str(fake_codex),
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id="provider-env",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="SYSTEM_ROLE_SENTINEL",
        user_prompt="FULL_USER_PROMPT_SENTINEL",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=10.0,
    )

    assert result.exit_code == 0
    assert captured["env"]["LLM_GATEWAY_KEY"] == secret
    assert any(
        name.startswith("HYPERLOOM_CODEX_HTTP_HEADER_") and value == header_value
        for name, value in captured["env"].items()
    )
    rendered_argv = " ".join(captured["cmd"])
    assert secret not in rendered_argv
    assert header_value not in rendered_argv
    assert "FULL_USER_PROMPT_SENTINEL" not in rendered_argv
    assert "SYSTEM_ROLE_SENTINEL" not in rendered_argv
    assert captured["cmd"][-1] == "-"
    assert Path(captured["stdin"].name).read_text(encoding="utf-8") == "FULL_USER_PROMPT_SENTINEL"
    assert captured["stdin"].closed is True


@pytest.mark.asyncio
async def test_workspace_write_fails_closed_when_bwrap_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed containment probe must prevent the specialist from spawning."""

    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    fake_codex = _successful_codex_script(tmp_path / "bin" / "codex")
    monkeypatch.setattr(
        sp,
        "probe_codex_sandbox_capability",
        lambda **_kwargs: False,
        raising=False,
    )
    monkeypatch.setattr(
        sp.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Codex must not spawn after a failed sandbox probe"),
    )

    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable=str(fake_codex),
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id="probe-failure",
        workspace=tmp_path / "workspace",
        worktree=None,
        worktree_base=None,
        system_prompt="system",
        user_prompt="user",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=10.0,
    )

    assert "bubblewrap" in result.error
    assert "refusing to fall back" in result.error


class _RecordingGpuLease:
    """Minimal Ray lease double recording replace-env and file-backed stdin."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.started: dict[str, Any] = {}
        self._alive = False

    def start_async(self, cmd: list[str], **kwargs: Any) -> None:
        self.started = {"cmd": list(cmd), **kwargs}
        Path(kwargs["log_path"]).write_text("", encoding="utf-8")
        (self.workspace / "specialist_done.json").write_text(
            '{"proposal_set":[],"empty":true,"summary":"ray"}',
            encoding="utf-8",
        )

    def poll_started(self) -> int:
        return 4242

    def is_alive(self) -> bool:
        return self._alive

    def exit_code(self) -> int:
        return 0

    def stop(self) -> None:
        self._alive = False

    def close(self) -> None:
        self._alive = False


@pytest.mark.asyncio
async def test_ray_codex_launch_uses_replace_env_and_prompt_file_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray must receive the same filtered env and stdin transport as local Popen."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    _enable_codex_bypass(monkeypatch)
    workspace = tmp_path / "workspace"
    lease = _RecordingGpuLease(workspace)
    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable="/usr/bin/codex",
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id="ray-stdin",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="ray system",
        user_prompt="ray user prompt",
        disallowed_tools=frozenset(),
        max_turns=1,
        gpu_ids=(0,),
        wall_budget_sec=10.0,
        gpu_lease=lease,
    )

    assert result.exit_code == 0
    assert lease.started["env_mode"] == "replace"
    assert lease.started["stdin_path"] == str(workspace / "prompt.md")
    assert Path(lease.started["stdin_path"]).read_text(encoding="utf-8") == "ray user prompt"
    assert lease.started["cmd"][-1] == "-"


@pytest.mark.asyncio
async def test_codex_mcp_config_is_translated_without_credentials_in_config_or_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP and stdio MCP definitions retain transport/env through private env refs."""

    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    _enable_codex_bypass(monkeypatch)
    mcp_secret = "mcp-secret-value"
    monkeypatch.setenv("SOURCE_MCP_TOKEN", mcp_secret)
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "pr_monitor": {
                        "type": "http",
                        "url": "https://mcp.invalid/rpc",
                        "headers": {"Authorization": "Bearer ${SOURCE_MCP_TOKEN}"},
                    },
                    "local_tools": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["server.py", "--serve"],
                        "env": {
                            "MCP_TOKEN": "${SOURCE_MCP_TOKEN}",
                            "MCP_MODE": "strict",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    fake_codex = _successful_codex_script(tmp_path / "bin" / "codex")
    captured: dict[str, Any] = {}
    real_popen = subprocess.Popen

    def _recording_popen(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.Popen:
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(sp.subprocess, "Popen", _recording_popen)
    workspace = tmp_path / "workspace"
    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable=str(fake_codex),
            mcp_config_path=str(mcp_path),
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id="mcp-translation",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="mcp system",
        user_prompt="mcp user",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=10.0,
    )

    assert result.exit_code == 0
    config_text = (workspace / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "[mcp_servers.pr_monitor]" in config_text
    assert 'url = "https://mcp.invalid/rpc"' in config_text
    assert "[mcp_servers.local_tools]" in config_text
    assert 'command = "python"' in config_text
    assert 'args = ["server.py", "--serve"]' in config_text
    assert 'env_vars = ["MCP_TOKEN", "MCP_MODE"]' in config_text
    assert "[mcp_servers.pr_monitor.env_http_headers]" in config_text
    assert mcp_secret not in config_text
    assert mcp_secret not in " ".join(captured["cmd"])
    assert captured["env"]["MCP_TOKEN"] == mcp_secret
    assert captured["env"]["MCP_MODE"] == "strict"
    assert "Bearer " + mcp_secret in captured["env"].values()


@pytest.mark.parametrize(
    "collision_key",
    [
        "PATH",
        "CODEX_HOME",
        "OPENAI_API_KEY",
        "LLM_GATEWAY_KEY",
        "HYPERLOOM_CODEX_HTTP_HEADER_0",
    ],
)
@pytest.mark.asyncio
async def test_codex_mcp_env_rejects_control_and_provider_collisions_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_key: str,
) -> None:
    """MCP stdio env cannot overwrite child control or provider variables."""

    _pin_provider_env(
        monkeypatch,
        {
            "OPENAI_BASE_URL": "https://gateway.invalid/Unified/v1",
            "OPENAI_API_KEY": "provider-openai-key",
            "LLM_GATEWAY_KEY": "provider-gateway-key",
            "OPENAI_CUSTOM_HEADERS": "user: provider-header-value",
        },
    )
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("HYPERLOOM_CODEX_HTTP_HEADER_0", raising=False)
    _enable_codex_bypass(monkeypatch)
    mcp_path = tmp_path / "mcp-collision.json"
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "collision": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                        "env": {collision_key: "mcp-override"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    popen_calls = 0

    def _recording_popen(*_args: Any, **_kwargs: Any) -> None:
        nonlocal popen_calls
        popen_calls += 1

    monkeypatch.setattr(sp.subprocess, "Popen", _recording_popen)
    lease = _NeverStartedGpuLease()
    workspace = tmp_path / "workspace"
    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable="/usr/bin/codex",
            mcp_config_path=str(mcp_path),
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id=f"mcp-collision-{collision_key.lower()}",
        workspace=workspace,
        worktree=None,
        worktree_base=None,
        system_prompt="system",
        user_prompt="user",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=10.0,
        gpu_lease=lease,
    )

    assert collision_key in result.error
    assert "collision" in result.error.lower() or "reserved" in result.error.lower()
    assert not (workspace / ".codex" / "config.toml").exists()
    assert popen_calls == 0
    assert lease.start_calls == 0
    assert "mcp-override" not in repr(lease.kwargs)


def test_codex_mcp_env_accepts_identical_benign_existing_value(
    tmp_path: Path,
) -> None:
    """An identical non-reserved child value is safe to forward by name."""

    mcp_path = tmp_path / "mcp-benign.json"
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "type": "stdio",
                        "command": "python",
                        "env": {"SHARED_BENIGN_SETTING": "same-value"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    lines, additions = sp._codex_mcp_config(
        str(mcp_path),
        source={"SHARED_BENIGN_SETTING": "same-value"},
        child_env={"SHARED_BENIGN_SETTING": "same-value"},
        protected_env_names=frozenset(),
    )

    assert 'env_vars = ["SHARED_BENIGN_SETTING"]' in lines
    assert additions == {}


def test_untranslatable_codex_mcp_server_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown MCP transports cannot silently disappear from a Codex specialist."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps({"mcpServers": {"legacy": {"type": "sse", "url": "https://mcp.invalid/sse"}}}),
        encoding="utf-8",
    )
    with pytest.raises(SpecialistAgentUnavailableError, match="legacy"):
        _build_cmd(
            tmp_path,
            codex_executable="/usr/bin/codex",
            mcp_config_path=str(mcp_path),
        )


@pytest.mark.asyncio
async def test_codex_structured_failure_is_propagated_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structured turn failure is more useful than a bare exit-code label."""
    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    _enable_codex_bypass(monkeypatch)
    raw_secret = "sk-supersecret123"
    fake_codex = _write_executable(
        tmp_path / "bin" / "codex",
        "#!/usr/bin/env bash\n"
        "cat >/dev/null\n"
        "printf '%s\\n' "
        + repr(
            json.dumps(
                {
                    "type": "turn.failed",
                    "error": {"message": f"provider rejected API_KEY={raw_secret}"},
                }
            )
        )
        + "\nexit 7\n",
    )
    result = await SpecialistSubprocessDispatcher(
        SpecialistSubprocessConfig(
            agent_backend=AGENT_BACKEND_CODEX,
            codex_executable=str(fake_codex),
            poll_interval_seconds=0.01,
        )
    ).run(
        task_id="structured-error",
        workspace=tmp_path / "workspace",
        worktree=None,
        worktree_base=None,
        system_prompt="system",
        user_prompt="user",
        disallowed_tools=frozenset(),
        max_turns=1,
        wall_budget_sec=10.0,
    )

    assert result.exit_code == 7
    assert "provider rejected" in result.error
    assert raw_secret not in result.error
    assert "[REDACTED]" in result.error


def _specialist_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "claude_model": "claude-selected-model",
        "codex_model": "gpt-selected-model",
        "specialist_model": None,
        "specialist_max_turns": 3,
        "specialist_per_turn_max_seconds": 42.0,
        "specialist_dispatch_mode": "subprocess",
        "specialist_mcp_config": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _runner_from_executor(executor: Any) -> Any:
    """Recover the SpecialistRunner captured by the CLI adapter closure."""
    from hyperloom.orchestrator.specialists.runner import SpecialistRunner

    for cell in executor.__closure__ or ():
        if isinstance(cell.cell_contents, SpecialistRunner):
            return cell.cell_contents
    raise AssertionError("specialist executor did not capture a SpecialistRunner")


@pytest.mark.parametrize(
    ("shape", "expected_backend", "expected_model"),
    [
        (_OPENAI_ONLY_ENV, AGENT_BACKEND_CODEX, "gpt-selected-model"),
        (_ANTHROPIC_ONLY_ENV, AGENT_BACKEND_CLAUDE, "claude-selected-model"),
    ],
)
def test_selected_backend_uses_its_own_model_without_specialist_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: dict[str, str],
    expected_backend: str,
    expected_model: str,
) -> None:
    """Codex must not inherit a preflight-mutated Claude model."""
    import shutil

    from hyperloom.inference_optimizer.cli import executors

    _pin_provider_env(monkeypatch, shape)
    monkeypatch.setattr(sp, "resolve_codex_executable", lambda explicit="": "/usr/bin/codex")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    runner = _runner_from_executor(
        executors._build_specialist_executor(
            _specialist_args(),
            session_dir=tmp_path,
            knowledge_plane=None,
        )
    )

    assert runner.subprocess_config.agent_backend == expected_backend
    assert runner.subprocess_config.model == expected_model


@pytest.mark.parametrize("shape", [_OPENAI_ONLY_ENV, _ANTHROPIC_ONLY_ENV])
def test_explicit_specialist_model_overrides_the_selected_backend_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: dict[str, str],
) -> None:
    """The generic override remains authoritative for either selected backend."""
    import shutil

    from hyperloom.inference_optimizer.cli import executors

    _pin_provider_env(monkeypatch, shape)
    monkeypatch.setattr(sp, "resolve_codex_executable", lambda explicit="": "/usr/bin/codex")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    runner = _runner_from_executor(
        executors._build_specialist_executor(
            _specialist_args(specialist_model="specialist-override"),
            session_dir=tmp_path,
            knowledge_plane=None,
        )
    )
    assert runner.subprocess_config.model == "specialist-override"


def test_openai_only_inprocess_uses_codex_agent_sdk_not_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-only in-process dispatch must stay agentic without Claude fallback."""
    from hyperloom.inference_optimizer.cli import executors

    _pin_provider_env(monkeypatch, _OPENAI_ONLY_ENV)
    monkeypatch.setattr(
        executors,
        "ClaudeBackend",
        lambda **_kwargs: pytest.fail("OpenAI-only in-process dispatch must not construct ClaudeBackend"),
    )
    runner = _runner_from_executor(
        executors._build_specialist_executor(
            _specialist_args(specialist_dispatch_mode="inprocess"),
            session_dir=tmp_path,
            knowledge_plane=None,
        )
    )
    backend = runner.backend_factory(SimpleNamespace())

    assert isinstance(backend, codex_agent.CodexAgentBackend)
    assert backend.model == "gpt-selected-model"
    assert Path(backend.cwd).is_relative_to(tmp_path)
    assert Path(backend.cwd) in tuple(Path(root) for root in backend.writable_roots)


def test_anthropic_inprocess_keeps_claude_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new SDK backend changes only the OpenAI-only in-process shape."""
    from hyperloom.inference_optimizer.cli import executors

    _pin_provider_env(monkeypatch, _ANTHROPIC_ONLY_ENV)
    monkeypatch.setattr(executors, "ClaudeBackend", lambda **kwargs: ("claude", kwargs))
    runner = _runner_from_executor(
        executors._build_specialist_executor(
            _specialist_args(specialist_dispatch_mode="inprocess"),
            session_dir=tmp_path,
            knowledge_plane=None,
        )
    )
    backend_name, kwargs = runner.backend_factory(SimpleNamespace())
    assert backend_name == "claude"
    assert kwargs["model"] == "claude-selected-model"


@pytest.mark.asyncio
async def test_codex_agent_backend_preserves_roles_and_returns_validated_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Agent SDK receives developer/user roles separately and returns intents."""
    from hyperloom.common.codex_session import CodexSessionResult
    from hyperloom.inference_optimizer.protocol.intent import IntentType

    payload = {
        "gap_canonical_id": "gap.sdk",
        "domain": "serving_specialist",
        "proposal_set": [],
        "empty": True,
        "summary": "sdk result",
    }
    captured: dict[str, Any] = {}

    async def _fake_run_codex_turn(**kwargs: Any) -> CodexSessionResult:
        captured.update(kwargs)
        return CodexSessionResult(
            text=json.dumps(
                {
                    "intents": [
                        {
                            "intent_type": "specialist_done",
                            "payload": payload,
                        }
                    ]
                }
            ),
            usage={
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "reasoning_output_tokens": 2,
            },
            thread_id="thread-123",
        )

    monkeypatch.setattr(codex_agent, "run_codex_turn", _fake_run_codex_turn)
    backend = codex_agent.CodexAgentBackend(
        model="gpt-agent",
        cwd=tmp_path,
        writable_roots=(tmp_path,),
        call_timeout_s=17.0,
    )
    result = await backend.run(
        "SYSTEM_ROLE_SENTINEL\n---\nUSER_ROLE_SENTINEL",
        system_prompt="SYSTEM_ROLE_SENTINEL",
        max_turns=1,
    )

    assert captured["prompt"] == "USER_ROLE_SENTINEL"
    assert captured["developer_instructions"].startswith("SYSTEM_ROLE_SENTINEL")
    assert "USER_ROLE_SENTINEL" not in captured["developer_instructions"]
    assert captured["cwd"] == tmp_path
    assert captured["writable_roots"] == (tmp_path,)
    assert captured["model"] == "gpt-agent"
    assert captured["timeout_sec"] == 17.0
    assert result.intents[0].type is IntentType.SPECIALIST_DONE
    assert result.intents[0].payload == payload
    assert result.metadata["thread_id"] == "thread-123"
    assert result.metadata["input_tokens"] == 11
    assert result.metadata["cache_read_input_tokens"] == 3
    assert result.metadata["reasoning_output_tokens"] == 2
    assert result.metadata["error"] == ""


def test_specialist_model_help_describes_generic_selected_backend_override() -> None:
    """The help text must not claim the generic override is Claude-only."""
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    help_text = " ".join(subparsers.choices["optimize"].format_help().split())
    assert "selected specialist backend" in help_text
