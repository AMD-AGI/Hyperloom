"""Tests for framework_agent.shell.render_template + run_command.

Hermetic - run_command uses ``/bin/echo`` / ``/bin/false`` so no
network/GPU is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from framework_agent.shell import render_template, run_command


# render_template --------------------------------------------------------


def test_render_template_substitutes_known_vars() -> None:
    """Placeholders matching variables are replaced; others raise."""
    out = render_template("python3 {bin}/x.py --out {dir}/o.json", {"bin": "/a", "dir": "/b"})
    assert out == "python3 /a/x.py --out /b/o.json"


def test_render_template_raises_on_unknown_var() -> None:
    """Unknown ``{var}`` placeholders should hard-fail."""
    with pytest.raises(ValueError, match="unknown variable"):
        render_template("echo {missing}", {"bin": "/a"})


def test_render_template_ignores_non_identifier_braces() -> None:
    """JSON-style ``{...}`` (no identifier) is preserved unchanged."""
    out = render_template("echo {{json: 1}}", {"x": "y"})
    assert out == "echo {{json: 1}}"


def test_render_template_shell_quote_neutralises_metachars() -> None:
    """shell_quote=True wraps values containing shell metacharacters in quotes."""
    out = render_template(
        "python3 {dir}/x.py",
        {"dir": "/tmp;rm -rf /"},
        shell_quote=True,
    )
    # The whole substituted value must be inside a single-quoted span so
    # the semicolon cannot start a new command.
    assert "/tmp;rm -rf /" in out
    assert "'/tmp;rm -rf /'" in out


def test_render_template_shell_quote_safe_path_unchanged() -> None:
    """shell_quote=True is a no-op for plain alphanumeric path values."""
    out = render_template("python3 {dir}/x.py", {"dir": "/tmp/foo"}, shell_quote=True)
    assert out == "python3 /tmp/foo/x.py"


# run_command ------------------------------------------------------------


def test_run_command_ok(tmp_path: Path) -> None:
    """A successful echo returns rc=0 and captures stdout tail."""
    result = run_command("echo", "echo hi", cwd=tmp_path, timeout_sec=5)
    assert result.returncode == 0
    assert result.ok is True
    assert "hi" in result.stdout_tail


def test_run_command_failure(tmp_path: Path) -> None:
    """A failing command propagates the non-zero rc and ok=False."""
    result = run_command("false", "false", cwd=tmp_path, timeout_sec=5)
    assert result.returncode != 0
    assert result.ok is False


def test_run_command_timeout(tmp_path: Path) -> None:
    """A command exceeding timeout flips timed_out=True and ok=False."""
    result = run_command("sleep", "sleep 3", cwd=tmp_path, timeout_sec=1)
    assert result.timed_out is True
    assert result.ok is False
    assert result.returncode == 124
