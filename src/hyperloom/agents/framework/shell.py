# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shell helpers for trusted framework exploration commands."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from .models import CommandResult

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def run_command(
    name: str,
    command: str,
    *,
    cwd: Path,
    timeout_sec: int,
) -> CommandResult:
    """Run a shell command with timeout, capture stdout/stderr tails."""
    try:
        proc = subprocess.run(  # nosec B602 - framework commands are explicit operator/test configuration.
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
    """Render known ``{var}`` placeholders, raise on unknown placeholders."""
    unknown: list[str] = []

    def _replace(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in variables:
            unknown.append(name)
            return m.group(0)
        value = variables[name]
        return shlex.quote(value) if shell_quote else value

    rendered = _PLACEHOLDER_RE.sub(_replace, template)
    if unknown:
        raise ValueError(
            "command template references unknown variable(s): " + ", ".join(repr(item) for item in sorted(set(unknown)))
        )
    return rendered
