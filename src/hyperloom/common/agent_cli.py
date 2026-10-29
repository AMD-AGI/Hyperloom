# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolution of the agent CLI executables (Claude, Codex) Hyperloom's agents run."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from common.llm_config import AGENT_BACKEND_CODEX

_CLI_PATH_ENV: str = "HYPERLOOM_CLAUDE_CLI_PATH"
# The kernel-agent installer records the Claude CLI it resolved here (``kernel-agent.env.sh``).
_INSTALLER_CLI_ENV: str = "GEAK_CLAUDE_BIN"


def resolve_claude_executable() -> str:
    """Resolve the Claude CLI Hyperloom's agents run, in-process and as subprocesses.

    Order: ``$HYPERLOOM_CLAUDE_CLI_PATH`` (the operator pin), then the installer's record in ``$GEAK_CLAUDE_BIN``,
    then ``claude`` on ``$PATH``. The installer may place the CLI outside ``$PATH`` (``~/.local/bin``).

    Returns:
        The resolved executable path, or ``""`` when none is executable.
    """
    for env_name in (_CLI_PATH_ENV, _INSTALLER_CLI_ENV):
        candidate = os.environ.get(env_name, "").strip()
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("claude") or ""


def resolve_codex_executable(explicit: str = "") -> str:
    """Resolve the Codex CLI a specialist subprocess should spawn.

    Order: an explicit path, then ``codex`` on ``$PATH`` (what the runtime
    container installs), then the version-pinned runtime shipped with the Codex
    SDK. The SDK runtime is last so an operator's own installation still wins,
    but present at all so a pod that never ran the npm install can still start a
    specialist.

    Args:
        explicit: Operator-configured path; returned as-is when non-empty.

    Returns:
        The resolved executable path, or ``""`` when no Codex runtime exists —
        the caller reports that instead of spawning a name that cannot run.
    """
    pinned = (explicit or "").strip()
    if pinned:
        return pinned
    on_path = shutil.which(AGENT_BACKEND_CODEX)
    if on_path:
        return on_path
    try:
        # Installed as the ``openai-codex`` SDK's own runtime dependency.
        from codex_cli_bin import bundled_codex_path  # type: ignore[import-not-found]

        bundled = Path(bundled_codex_path())
    except ImportError:
        return ""
    return str(bundled) if bundled.exists() else ""
