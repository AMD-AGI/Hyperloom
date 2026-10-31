# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared LLM-transport stability env for spawned claude-CLI / SDK children."""

from __future__ import annotations

from typing import MutableMapping

__all__ = ["apply_llm_stability_env"]


def apply_llm_stability_env(env: MutableMapping[str, str]) -> None:
    """Inject client-side LLM-transport stability knobs into ``env``."""
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    env.setdefault("DISABLE_AUTOUPDATER", "1")
