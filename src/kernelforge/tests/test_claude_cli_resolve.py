# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for robust claude CLI resolution (RCA root cause 1): env override,
PATH discovery, and graceful fallback when the binary is absent."""

from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import (
    AgentHook,
    AgentHooks,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.claude import (
    ClaudeBackend,
    ClaudeUnavailableError,
    _prepare_claude_environment,
    _sdk_hooks,
    resolve_claude_cli,
)


def _make_exe(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_env_override_generic_agent_cli(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "claude")
    monkeypatch.setenv("FORGE_AGENT_CLI", exe)
    assert resolve_claude_cli() == exe


def test_explicit_runtime_cli_path(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "claude")
    monkeypatch.delenv("FORGE_AGENT_CLI", raising=False)
    assert resolve_claude_cli(exe) == exe


def test_env_override_ignored_when_not_executable(tmp_path, monkeypatch):
    # A non-existent override must not be returned; falls through to which/search,
    # ending at either a real executable on this host or the bare name.
    bad = str(tmp_path / "nope")
    monkeypatch.setenv("FORGE_AGENT_CLI", bad)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = resolve_claude_cli()
    assert result != bad
    assert result == "claude" or (os.path.isfile(result) and os.access(result, os.X_OK))


def test_path_discovery(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = _make_exe(bindir / "claude")
    monkeypatch.delenv("FORGE_AGENT_CLI", raising=False)
    monkeypatch.setenv("PATH", str(bindir))
    assert resolve_claude_cli() == exe


def test_resolve_returns_existing_or_bare(tmp_path, monkeypatch):
    # With no env override and a stripped PATH, the resolver returns either a
    # real existing executable (a common prefix on this host) or the bare name
    # "claude" as last resort -- never a stale path that does not exist.
    monkeypatch.delenv("FORGE_AGENT_CLI", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    result = resolve_claude_cli()
    assert result == "claude" or (os.path.isfile(result) and os.access(result, os.X_OK))


def test_claude_backend_maps_additional_directories(tmp_path):
    """Map provider-neutral read directories to Claude SDK add_dirs."""
    backend = object.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(
        provider="claude",
        model="claude-test",
    )
    extra = tmp_path / "read-only"
    spec = AgentRunSpec(
        system_prompt="Inspect references.",
        user_prompt="Prepare the driver.",
        cwd=str(tmp_path),
        additional_directories=[str(extra)],
    )

    options = backend._provider_options(spec)

    assert options["add_dirs"] == [str(extra)]


def test_claude_probe_checks_selected_model_with_configured_gateway(
    tmp_path,
    monkeypatch,
):
    backend = object.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(
        provider="claude",
        model="claude-opus-5",
        executable="/usr/bin/claude",
    )
    monkeypatch.setattr(backend, "preflight", lambda: None)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout='{"result":"OK"}',
            stderr="",
        )

    monkeypatch.setattr(
        "kernelforge.agent_backends.claude.subprocess.run",
        fake_run,
    )
    result = backend.probe(cwd=str(tmp_path))

    assert result.text == "OK"
    assert captured["command"][captured["command"].index("--model") + 1] == ("claude-opus-5")
    assert captured["kwargs"]["cwd"] == str(tmp_path)


def test_claude_probe_rejects_unsupported_model(tmp_path, monkeypatch):
    backend = object.__new__(ClaudeBackend)
    backend.runtime = AgentRuntimeConfig(
        provider="claude",
        model="claude-opus-5",
        executable="/usr/bin/claude",
    )
    monkeypatch.setattr(backend, "preflight", lambda: None)
    monkeypatch.setattr(
        "kernelforge.agent_backends.claude.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="model not available",
        ),
    )

    with pytest.raises(ClaudeUnavailableError, match="model not available"):
        backend.probe(cwd=str(tmp_path))


def test_claude_hook_mapping_is_environment_independent():
    """Translate populated and empty hook attributes deterministically."""
    callback = object()

    class FakeMatcher:
        """Record keyword arguments passed to the SDK matcher."""

        def __init__(self, **kwargs):
            """Store normalized matcher options."""
            self.kwargs = kwargs

    translated = _sdk_hooks(
        AgentHooks(
            pre_tool_use=[
                AgentHook(
                    matcher="Edit",
                    callback=callback,
                    timeout_sec=7,
                ),
            ],
            stop=[AgentHook(matcher="", callback=callback)],
        ),
        FakeMatcher,
    )

    assert set(translated) == {"PreToolUse", "Stop"}
    assert translated["PreToolUse"][0].kwargs == {
        "hooks": [callback],
        "matcher": "Edit",
        "timeout": 7,
    }
    assert translated["Stop"][0].kwargs == {"hooks": [callback]}


def test_prepare_claude_environment_keeps_the_operators_route(monkeypatch):
    """Apply the root sandbox flag and leave the operator's route alone.

    The CLI speaks the Anthropic protocol and owns its own path suffixes, so
    rewriting the route here would only hide misconfiguration.
    """

    def fake_geteuid() -> int:
        """Simulate a root process in any CI environment."""
        return 0

    monkeypatch.setattr(os, "geteuid", fake_geteuid)
    monkeypatch.delenv("IS_SANDBOX", raising=False)
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL",
        "https://gateway.example/llm-gateway/",
    )

    _prepare_claude_environment()

    assert os.environ["IS_SANDBOX"] == "1"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example/llm-gateway"


def test_prepare_claude_environment_drops_a_duplicated_version_suffix(monkeypatch):
    """The CLI appends /v1/messages, so a base already carrying /v1 404s.

    Measured against a LiteLLM proxy, which publishes its base that way: the
    doubled path comes back as "There's an issue with the selected model ... it
    may not exist or you may not have access to it", pointing at a model and a
    permission that were never the problem.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/llm-proxy/v1")

    _prepare_claude_environment()

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example/llm-proxy"


def test_prepare_claude_environment_expands_header_env_refs(monkeypatch):
    """The CLI reads this variable itself, so ${VAR} must be resolved first.

    Left alone, the reference text would travel as the header value and the
    gateway would reject a subscription key it never received.
    """
    monkeypatch.setenv("MY_SUB_KEY", "expanded-secret")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "Ocp-Apim-Subscription-Key: ${MY_SUB_KEY}\nuser: alice",
    )

    _prepare_claude_environment()

    assert os.environ["ANTHROPIC_CUSTOM_HEADERS"] == ("Ocp-Apim-Subscription-Key: expanded-secret\nuser: alice")


def test_prepare_claude_environment_leaves_plain_headers_alone(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "user: alice")
    _prepare_claude_environment()
    assert os.environ["ANTHROPIC_CUSTOM_HEADERS"] == "user: alice"


def test_prepare_claude_environment_rewrites_json_headers(monkeypatch):
    """The CLI understands only the newline form, so normalize JSON into it."""
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        '{"Ocp-Apim-Subscription-Key": "sub123", "user": "alice"}',
    )

    _prepare_claude_environment()

    assert os.environ["ANTHROPIC_CUSTOM_HEADERS"] == ("Ocp-Apim-Subscription-Key: sub123\nuser: alice")


def test_prepare_claude_environment_keeps_unparseable_headers(monkeypatch):
    """Nothing parses out, so leave the operator's value for the CLI to reject."""
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "no-colon-here")
    _prepare_claude_environment()
    assert os.environ["ANTHROPIC_CUSTOM_HEADERS"] == "no-colon-here"
