# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent system prompts."""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom.inference_optimizer.session.session_paths import agent_prompt_snapshot

log = logging.getLogger(__name__)


def read_rules_fragment(path: Path | None) -> str:
    """Read a rules fragment (orchestration.md / critic.md), tolerating absence."""
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
    """Persist a role's effective system prompt for audit / drift inspection."""
    try:
        target = agent_prompt_snapshot(session_dir, role, phase=phase)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body or "(empty)", encoding="utf-8")
    except OSError:
        log.warning("prompt snapshot write failed for role=%s phase=%s", role, phase)
