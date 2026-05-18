"""Shell helpers for trusted framework exploration commands.

Ported verbatim from zhenggong/framework-agent. Pure subprocess + template
rendering with no external deps.
"""

from __future__ import annotations

import subprocess
import re
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


def render_template(template: str, variables: dict[str, str]) -> str:
    """Render known ``{var}`` placeholders, raise on unknown placeholders.

    Does not touch JSON-style ``{...}`` braces that don't match the
    ``[A-Za-z_][A-Za-z0-9_]*`` identifier pattern.
    """
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{" + key + "}", value)
    unknown = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)))
    if unknown:
        raise ValueError(
            "command template references unknown variable(s): "
            + ", ".join(repr(item) for item in unknown)
        )
    return rendered
