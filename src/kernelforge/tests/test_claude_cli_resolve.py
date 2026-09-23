# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pin Claude CLI selection and offline preflight validation, including invalid operator overrides."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("selection", ["runtime", "environment"])
@pytest.mark.parametrize("invalid", ["missing", "directory", "not-executable"])
def test_invalid_cli_pin_is_not_replaced_by_discovery(tmp_path, monkeypatch, selection, invalid):
    """Keep a bad operator pin selected so validation can report it."""
    good = _make_exe(tmp_path / "claude")
    bad_path = tmp_path / "bad-pin"
    if invalid == "directory":
        bad_path.mkdir()
    elif invalid == "not-executable":
        _make_exe(bad_path)
        real_access = os.access
        monkeypatch.setattr(os, "access", lambda path, mode: str(path) != str(bad_path) and real_access(path, mode))
    bad = str(bad_path)
    monkeypatch.setenv("FORGE_AGENT_CLI", bad if selection == "environment" else good)
    which = Mock(return_value=good)
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", which)

    assert resolve_claude_cli(bad if selection == "runtime" else "") == bad
    which.assert_not_called()


def test_path_discovery(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = _make_exe(bindir / "claude")
    monkeypatch.delenv("FORGE_AGENT_CLI", raising=False)
    monkeypatch.setenv("PATH", str(bindir))
    assert resolve_claude_cli() == exe


def test_resolve_returns_existing_or_bare(tmp_path, monkeypatch):
    # With no env override and a stripped PATH, the resolver returns either a real existing executable (a common
    # prefix on this host) or the bare name "claude" as last resort -- never a stale path that does not exist.
    monkeypatch.delenv("FORGE_AGENT_CLI", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    result = resolve_claude_cli()
    assert result == "claude" or (os.path.isfile(result) and os.access(result, os.X_OK))


@pytest.fixture
def preflight_backend(monkeypatch):
    """Construct normally but forbid any SDK query or model probe."""
    query = Mock(side_effect=AssertionError("preflight must not query the SDK"))
    monkeypatch.setattr("kernelforge.agent_backends.claude._load_claude_sdk", lambda: (query, Mock()))
    probe = Mock(side_effect=AssertionError("preflight must not probe a model"))
    monkeypatch.setattr(ClaudeBackend, "probe", probe)
    monkeypatch.delenv("FORGE_AGENT_CLI", raising=False)
    backend = ClaudeBackend(AgentRuntimeConfig(provider="claude", model="claude-test"))
    yield backend
    query.assert_not_called()
    probe.assert_not_called()


def test_preflight_rejects_missing_default_cli(preflight_backend, monkeypatch):
    resolver = Mock(return_value="claude")
    run = Mock(side_effect=AssertionError("a missing CLI must not be launched"))
    monkeypatch.setattr("kernelforge.agent_backends.claude.resolve_claude_cli", resolver)
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", lambda _name: None)
    monkeypatch.setattr("kernelforge.agent_backends.claude.Path.is_file", lambda _path: False)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ClaudeUnavailableError, match="Claude CLI is not executable: claude"):
        preflight_backend.preflight()

    resolver.assert_called_once_with("")
    run.assert_not_called()


@pytest.mark.parametrize("output", [b"other tool 1.0", b""])
def test_preflight_rejects_wrong_default_cli(tmp_path, monkeypatch, preflight_backend, output):
    exe = _make_exe(tmp_path / "claude")
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", lambda _name: exe)
    run = Mock(return_value=subprocess.CompletedProcess([exe, "--version"], 0, output, b""))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ClaudeUnavailableError, match="does not appear to be Claude") as exc:
        preflight_backend.preflight()

    assert exe in str(exc.value)
    run.assert_called_once_with([exe, "--version"], capture_output=True, timeout=10, check=False)


def test_preflight_rejects_default_cli_timeout(tmp_path, monkeypatch, preflight_backend):
    exe = _make_exe(tmp_path / "claude")
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", lambda _name: exe)
    timeout = subprocess.TimeoutExpired([exe, "--version"], 10)
    run = Mock(side_effect=timeout)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ClaudeUnavailableError, match="version check failed") as exc:
        preflight_backend.preflight()

    assert exc.value.__cause__ is timeout
    run.assert_called_once_with([exe, "--version"], capture_output=True, timeout=10, check=False)


@pytest.mark.parametrize("selection", ["default", "runtime", "runtime-command", "environment"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_preflight_accepts_only_selected_cli_version(tmp_path, monkeypatch, preflight_backend, selection, stream):
    exe = _make_exe(tmp_path / "selected-claude")
    other = _make_exe(tmp_path / "other-claude")
    if selection.startswith("runtime"):
        pin = exe if selection == "runtime" else "selected-claude"
        preflight_backend.runtime = replace(preflight_backend.runtime, executable=pin)
        monkeypatch.setenv("FORGE_AGENT_CLI", other)
    elif selection == "environment":
        monkeypatch.setenv("FORGE_AGENT_CLI", exe)
    which = Mock(side_effect=lambda name: exe if name == "selected-claude" or selection == "default" else other)
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", which)
    output = {"stdout": b"", "stderr": b""}
    output[stream] = b"2.1.0 (Claude Code)\n"
    run = Mock(return_value=subprocess.CompletedProcess([exe, "--version"], 0, **output))
    monkeypatch.setattr(subprocess, "run", run)

    assert preflight_backend.preflight() is None

    run.assert_called_once_with([exe, "--version"], capture_output=True, timeout=10, check=False)
    if selection in {"runtime", "environment"}:
        which.assert_not_called()


@pytest.mark.parametrize("selection", ["runtime", "environment"])
@pytest.mark.parametrize("invalid", ["missing", "directory", "not-executable"])
def test_preflight_rejects_bad_pin_despite_working_default(
    tmp_path, monkeypatch, preflight_backend, selection, invalid
):
    good = _make_exe(tmp_path / "claude")
    bad_path = tmp_path / "bad-pin"
    if invalid == "directory":
        bad_path.mkdir()
    elif invalid == "not-executable":
        _make_exe(bad_path)
        real_access = os.access
        monkeypatch.setattr(os, "access", lambda path, mode: str(path) != str(bad_path) and real_access(path, mode))
    bad = str(bad_path)
    if selection == "runtime":
        preflight_backend.runtime = replace(preflight_backend.runtime, executable=bad)
        monkeypatch.setenv("FORGE_AGENT_CLI", good)
    else:
        monkeypatch.setenv("FORGE_AGENT_CLI", bad)
    which = Mock(side_effect=lambda name: good if name == "claude" else None)
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", which)
    run = Mock(side_effect=AssertionError("a bad pin must not launch a different CLI"))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ClaudeUnavailableError, match="Claude CLI is not executable") as exc:
        preflight_backend.preflight()

    assert bad in str(exc.value)
    assert all(call.args[0] != "claude" for call in which.call_args_list)
    run.assert_not_called()


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError, OSError])
def test_preflight_wraps_native_launch_errors(tmp_path, monkeypatch, preflight_backend, explicit, error_type):
    exe = _make_exe(tmp_path / "claude")
    if explicit:
        preflight_backend.runtime = replace(preflight_backend.runtime, executable=exe)
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", lambda _name: exe)
    error = error_type("native launch failure")
    run = Mock(side_effect=error)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ClaudeUnavailableError, match="version check failed: native launch failure") as exc:
        preflight_backend.preflight()

    assert exc.value.__cause__ is error
    run.assert_called_once_with([exe, "--version"], capture_output=True, timeout=10, check=False)


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
def test_preflight_rejects_nonzero_version_even_if_it_mentions_claude(
    tmp_path, monkeypatch, preflight_backend, explicit
):
    exe = _make_exe(tmp_path / "claude")
    if explicit:
        preflight_backend.runtime = replace(preflight_backend.runtime, executable=exe)
    monkeypatch.setattr("kernelforge.agent_backends.claude.shutil.which", lambda _name: exe)
    run = Mock(return_value=subprocess.CompletedProcess([exe, "--version"], 7, b"Claude Code", b"loader failed"))
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ClaudeUnavailableError, match="does not appear to be Claude") as exc:
        preflight_backend.preflight()

    assert "loader failed" in str(exc.value)
    run.assert_called_once_with([exe, "--version"], capture_output=True, timeout=10, check=False)


def test_validate_runtime_needs_no_sdk_or_backend_instance(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "claude")
    runtime = AgentRuntimeConfig(provider="claude", model="claude-test", executable=exe)
    sdk = Mock(side_effect=AssertionError("CLI validation must not load the SDK"))
    prepare = Mock(side_effect=AssertionError("CLI validation must not rewrite the environment"))
    probe = Mock(side_effect=AssertionError("CLI validation must not probe a model"))
    monkeypatch.setattr("kernelforge.agent_backends.claude._load_claude_sdk", sdk)
    monkeypatch.setattr("kernelforge.agent_backends.claude._prepare_claude_environment", prepare)
    monkeypatch.setattr(ClaudeBackend, "probe", probe)
    run = Mock(return_value=subprocess.CompletedProcess([exe, "--version"], 0, b"Claude Code", b""))
    monkeypatch.setattr(subprocess, "run", run)

    assert ClaudeBackend.validate_runtime(runtime) is None

    sdk.assert_not_called()
    prepare.assert_not_called()
    probe.assert_not_called()
    run.assert_called_once_with([exe, "--version"], capture_output=True, timeout=10, check=False)


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
    """Apply the root sandbox flag and leave the operator's route alone."""

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
    """The CLI appends /v1/messages, so a base already carrying /v1 404s."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/llm-proxy/v1")

    _prepare_claude_environment()

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://gateway.example/llm-proxy"


def test_prepare_claude_environment_expands_header_env_refs(monkeypatch):
    """The CLI reads this variable itself, so ${VAR} must be resolved first."""
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
