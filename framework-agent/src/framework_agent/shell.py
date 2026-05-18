"""Shell helpers for trusted framework exploration commands.

Pure subprocess + template rendering with no external deps. The
template renderer accepts an optional ``shell_quote`` flag that
wraps each substituted value in ``shlex.quote`` so an attacker who
managed to seed a candidate ref or path with shell metacharacters
cannot break out of the rendered command string. Callers that
render *paths* (not shell commands) keep ``shell_quote=False``.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from .models import CommandResult


def run_command(
    name: str,
    command: str,
    *,
    cwd: Path,
    timeout_sec: int,
) -> CommandResult:
    """Run a shell command with timeout, capture stdout/stderr tails."""
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return CommandResult(
            name=name,
            command=command,
            returncode=proc.returncode,
            stdout_tail=(proc.stdout or "")[-4000:],
            stderr_tail=(proc.stderr or "")[-4000:],
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=command,
            returncode=124,
            stdout_tail=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            stderr_tail=(exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            timed_out=True,
        )


def render_template(
    template: str,
    variables: dict[str, str],
    *,
    shell_quote: bool = False,
) -> str:
    """Render known ``{var}`` placeholders, raise on unknown placeholders.

    Does not touch JSON-style ``{...}`` braces that don't match the
    ``[A-Za-z_][A-Za-z0-9_]*`` identifier pattern.

    When ``shell_quote=True`` each substituted value is wrapped with
    :func:`shlex.quote`. Use this when the rendered string will be
    handed to a shell (e.g. ``subprocess.run(shell=True)``); the
    quoting is a no-op for plain alphanumeric / path-safe values
    and transparently neutralises shell metacharacters in any
    untrusted input.
    """
    rendered = template
    for key, value in variables.items():
        replacement = shlex.quote(value) if shell_quote else value
        rendered = rendered.replace("{" + key + "}", replacement)
    unknown = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)))
    if unknown:
        raise ValueError(
            "command template references unknown variable(s): "
            + ", ".join(repr(item) for item in unknown)
        )
    return rendered
