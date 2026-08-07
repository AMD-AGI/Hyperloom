# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent system prompts.

Markdown rule fragments plus two independent assemblers: :mod:`prompt_builder`
(the persistent Orchestration agent's system prompt, composed from
:class:`ActionMetadata`, the rules fragment and run-level parameters) and
:mod:`specialist_prompt_builder` (the ephemeral specialist sub-agent's
system/user prompt pair).
"""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom.inference_optimizer.session.session_paths import agent_prompt_snapshot

log = logging.getLogger(__name__)


def read_rules_fragment(path: Path | None) -> str:
    """Read a rules fragment (orchestration.md / critic.md), tolerating absence.

    Args:
        path (Path | None): Path to the rules fragment, or ``None`` to skip.

    Returns:
        str: The stripped fragment text, or an empty string when the path is
        ``None`` or unreadable.
    """
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_prompt_snapshot(
    session_dir: Path,
    role: str,
    body: str,
    *,
    phase: str = "",
) -> None:
    """Persist a role's effective system prompt for audit / drift inspection.

    Fail-soft: a failed write must never take down a phase transition.

    Args:
        session_dir (Path): The session root directory.
        role (str): The agent role the prompt belongs to.
        body (str): The prompt text as handed to the backend.
        phase (str): Pipeline phase this scope was built for; ``""`` writes the
            unsuffixed boot snapshot.
    """
    try:
        target = agent_prompt_snapshot(session_dir, role, phase=phase)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")
    except OSError:
        log.warning("prompt snapshot write failed for role=%s phase=%s", role, phase)
