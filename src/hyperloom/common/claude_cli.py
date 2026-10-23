# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locate the Claude CLI that Hyperloom's agents run."""

from __future__ import annotations

import os
import shutil

CLI_PATH_ENV = "HYPERLOOM_CLAUDE_CLI_PATH"
# The kernel-agent installer records the Claude CLI it resolved here (``kernel-agent.env.sh``).
INSTALLER_CLI_ENV = "GEAK_CLAUDE_BIN"


def resolve_claude_executable() -> str:
    """Resolve the Claude CLI Hyperloom's agents run, in-process and as subprocesses.

    Order: ``$HYPERLOOM_CLAUDE_CLI_PATH`` (the operator pin), then the installer's record in ``$GEAK_CLAUDE_BIN``,
    then ``claude`` on ``$PATH``. The installer may place the CLI outside ``$PATH`` (``~/.local/bin``).

    Returns:
        The resolved executable path, or ``""`` when none is executable.
    """
    for env_name in (CLI_PATH_ENV, INSTALLER_CLI_ENV):
        candidate = os.environ.get(env_name, "").strip()
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("claude") or ""
