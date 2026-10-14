# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one way the scheduling side asks git about a checkout: a read-only command whose output it needs."""

from __future__ import annotations

import subprocess
from pathlib import Path

GIT_TIMEOUT_SECONDS = 60.0


class GitFailed(RuntimeError):
    """git could not be run, did not finish in time, or exited with an error."""


def git_output(root: Path, *args: str) -> str:
    """The stripped standard output of ``git -C root args``; raises :class:`GitFailed` when it fails. Blocks."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        raise GitFailed(f"git {' '.join(args)} in {root} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitFailed(f"git {' '.join(args)} in {root} failed: {exc}") from exc
    return completed.stdout.strip()


__all__ = ["GIT_TIMEOUT_SECONDS", "GitFailed", "git_output"]
