# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One way to run git.

A failed git command means the working tree is not what the caller believes it
is, so the default here is to raise rather than to return a result nobody
inspects. Callers that genuinely tolerate a non-zero exit -- probing whether a
ref exists, asking a detached HEAD for its branch -- say so with ``check=False``.

Every command runs in its own session and under a timeout, so a git wedged on a
lock is bounded instead of holding the run open.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
from pathlib import Path

# No local plumbing command on a large worktree comes close to this; anything
# that does is stuck rather than slow.
DEFAULT_TIMEOUT_SEC = 300.0


class GitError(subprocess.CalledProcessError):
    """A git command that exited non-zero, quoting git's own words."""

    def __str__(self) -> str:
        detail = self.stderr or self.output or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        command = " ".join(str(part) for part in self.cmd)
        return f"{command} failed ({self.returncode}): {detail.strip()[-400:]}"


def _kill_process_group(pid: int) -> None:
    """Take down the whole session, not just the direct child.

    Every command here is spawned with ``start_new_session``, so a helper an
    alias or a hook started is in the same group and would otherwise survive
    and hold the pipes open long after the caller gave up.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def _checked(
    completed: subprocess.CompletedProcess,
    check: bool,
) -> subprocess.CompletedProcess:
    """Raise when a non-zero exit is not one the caller asked to tolerate."""
    if check and completed.returncode != 0:
        raise GitError(
            completed.returncode,
            completed.args,
            completed.stdout,
            completed.stderr,
        )
    return completed


def git(
    *args: str,
    cwd: str | Path | None = None,
    check: bool = True,
    text: bool = True,
    timeout: float | None = DEFAULT_TIMEOUT_SEC,
    input: str | bytes | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one git command to completion, capturing both streams."""
    with subprocess.Popen(
        ["git", *args],
        cwd=None if cwd is None else str(cwd),
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        env=None if env is None else {**os.environ, **env},
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except BaseException:
            # ``subprocess.run`` would kill only git itself here, leaving an
            # alias or hook's own child holding the pipes it inherited.
            _kill_process_group(process.pid)
            raise
    return _checked(
        subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr),
        check,
    )


async def git_async(
    *args: str,
    cwd: str | Path | None = None,
    check: bool = True,
    timeout: float | None = DEFAULT_TIMEOUT_SEC,
) -> subprocess.CompletedProcess:
    """Await one git command, returning the same result shape as ``git``.

    Kept on the asyncio spawn path so a cancelled gather -- lane preparation
    giving up and removing the directory it was cloning into -- takes the git
    process down with it instead of racing the removal.
    """
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=None if cwd is None else str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        # Any exit without a finished communicate() leaves the child running.
        # Named by no type on purpose: which class wait_for raises for a
        # timeout changed in 3.11, and enumerating them once left the group
        # alive on 3.10 for the one case this exists to handle.
        _kill_process_group(process.pid)
        await process.wait()
        raise
    return _checked(
        subprocess.CompletedProcess(
            ["git", *args],
            process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        ),
        check,
    )
