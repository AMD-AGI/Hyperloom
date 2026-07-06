# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Persistent-agent system prompts.

Markdown rule fragments plus :mod:`prompt_builder`, which composes the full
Orchestration system prompt from :class:`ActionMetadata`, the rules fragment,
and run-level parameters.
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
