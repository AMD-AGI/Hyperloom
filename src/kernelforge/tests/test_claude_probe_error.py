"""A failed Claude probe must report the cause, not whatever landed on stderr."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends import claude as claude_mod
from kernelforge.agent_backends.claude import ClaudeBackend, ClaudeUnavailableError


def _backend(monkeypatch, *, returncode: int, stdout: str, stderr: str) -> ClaudeBackend:
    """A ClaudeBackend whose CLI invocation returns exactly these three values."""
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.runtime = SimpleNamespace(
        provider="claude",
        model="glm-5-3",
        executable="",
        timeout_sec=60,
        reasoning_effort="low",
        options={},
    )
    backend.fallback_reason = ""
    backend.preflight = lambda: None
    monkeypatch.setattr(claude_mod, "resolve_claude_cli", lambda _: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr),
    )
    return backend


def test_probe_failure_reports_stdout_when_stderr_holds_only_a_warning(monkeypatch):
    """The actionable cause is on stdout; a benign stderr line must not hide it.

    The Claude CLI writes an ``unrecognized_model`` warning to stderr whenever the
    model is not one it knows by name -- including models that then serve the
    request perfectly well, because the warning comes from an internal
    session-title query rather than from the run itself. Reporting
    ``stderr or stdout`` means that always-present warning displaces whatever
    actually went wrong, so every failure on such a model reads as "the model is
    unrecognized" no matter its true cause.
    """
    backend = _backend(
        monkeypatch,
        returncode=1,
        stdout='{"result":"API Error: SSL certificate verification failed"}',
        stderr='[claude-code:unrecognized_model] {"model":"glm-5-3","query_source":"generate_session_title"}',
    )

    with pytest.raises(ClaudeUnavailableError) as excinfo:
        backend.probe(cwd=".")

    message = str(excinfo.value)
    assert "SSL certificate verification failed" in message, message
    # The warning is still worth surfacing -- it just must not be the whole story.
    assert "unrecognized_model" in message, message


def test_probe_failure_reports_stderr_when_stdout_is_empty(monkeypatch):
    """A CLI that fails before writing anything to stdout still reports its reason."""
    backend = _backend(
        monkeypatch,
        returncode=127,
        stdout="",
        stderr="claude: command not found",
    )

    with pytest.raises(ClaudeUnavailableError) as excinfo:
        backend.probe(cwd=".")

    assert "command not found" in str(excinfo.value)


def test_probe_failure_without_output_still_names_the_exit_code(monkeypatch):
    """A silent non-zero exit must not produce an empty, undiagnosable message."""
    backend = _backend(monkeypatch, returncode=9, stdout="", stderr="")

    with pytest.raises(ClaudeUnavailableError) as excinfo:
        backend.probe(cwd=".")

    assert "9" in str(excinfo.value)
