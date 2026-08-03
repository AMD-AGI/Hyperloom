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

from pathlib import Path


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
