# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shell helpers for trusted framework exploration commands.

Pure subprocess + template rendering, no external deps. The renderer's
optional ``shell_quote`` flag wraps each substituted value in ``shlex.quote``
so a candidate ref/path seeded with shell metacharacters can't break out of
the command string. Callers rendering paths (not commands) keep it False.
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
    """Run a shell command with timeout, capture stdout/stderr tails.

    Args:
        name (str): Logical name for the command, echoed back in the result.
        command (str): The shell command line to execute.
        cwd (Path): Working directory the command runs in.
        timeout_sec (int): Hard timeout in seconds before the command is killed.

    Returns:
        CommandResult: Outcome holding the return code and the last 4000
            characters of stdout/stderr. On timeout, ``returncode`` is 124 and
            ``timed_out`` is True.
    """
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

    Leaves JSON-style ``{...}`` braces that don't match the
    ``[A-Za-z_][A-Za-z0-9_]*`` identifier pattern untouched.

    Args:
        template: Template string with ``{var}`` placeholders.
        variables: Mapping of placeholder names to replacement values.
        shell_quote: When True, wrap each value in :func:`shlex.quote` for
            shell-bound strings.

    Returns:
        The rendered string.

    Raises:
        ValueError: If any ``{identifier}`` placeholder is left unresolved.
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
