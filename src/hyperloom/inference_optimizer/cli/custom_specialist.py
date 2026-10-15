# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator-defined specialist's launch inputs: the ``--custom-specialist-*`` pair and its only mode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyperloom.orchestrator.state.shared_state import SharedState

__all__ = ["CUSTOM_SPECIALIST_ONLY_ENV"]

_CUSTOM_SPECIALIST_PROMPT_MAX_BYTES = 16 * 1024
_CUSTOM_SPECIALIST_DESCRIPTION_MAX_CHARS = 200
# FRAMEWORK_AGENT runs the custom specialist and the bench of its output alone; persisted like the pair.
CUSTOM_SPECIALIST_ONLY_ENV = "HYPERLOOM_CUSTOM_SPECIALIST_ONLY"


def _load_custom_specialist_args(args: argparse.Namespace) -> None:
    """Validate the ``--custom-specialist-*`` pair and set ``args.custom_specialist_prompt``.

    Raises:
        ValueError: When only one flag is given or either value is unusable.
    """
    path = getattr(args, "custom_specialist_prompt_file", None)
    description = getattr(args, "custom_specialist_description", None)
    args.custom_specialist_prompt = ""
    if path is None and description is None:
        return
    if path is None or description is None:
        raise ValueError("--custom-specialist-prompt-file and --custom-specialist-description must be passed together")
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"--custom-specialist-prompt-file: cannot read {path}: {exc}") from exc
    if not text:
        raise ValueError(f"--custom-specialist-prompt-file: {path} is empty")
    if len(text.encode("utf-8")) > _CUSTOM_SPECIALIST_PROMPT_MAX_BYTES:
        raise ValueError(f"--custom-specialist-prompt-file: {path} exceeds {_CUSTOM_SPECIALIST_PROMPT_MAX_BYTES} bytes")
    description = description.strip()
    if not description or "\n" in description or "\r" in description:
        raise ValueError("--custom-specialist-description must be a single non-empty line")
    if len(description) > _CUSTOM_SPECIALIST_DESCRIPTION_MAX_CHARS:
        raise ValueError(
            f"--custom-specialist-description exceeds {_CUSTOM_SPECIALIST_DESCRIPTION_MAX_CHARS} characters"
        )
    args.custom_specialist_prompt = text
    args.custom_specialist_description = description


def _apply_custom_specialist_resume(args: argparse.Namespace, state: SharedState) -> None:
    """Flags passed on a resume replace the stored custom specialist; otherwise the stored one stays.

    The only-mode env turns only mode on and never off.
    """
    if getattr(args, "custom_specialist_only", False):
        state.custom_specialist_only = True
    prompt = getattr(args, "custom_specialist_prompt", "")
    if not prompt:
        return
    state.custom_specialist_prompt = prompt
    state.custom_specialist_description = args.custom_specialist_description
    print(f"  re-exported custom specialist: {state.custom_specialist_description}")


def _custom_specialist_conflict(args: argparse.Namespace, state: SharedState) -> str:
    """Return why a custom specialist from this launch or the stored state cannot run, or ``""``."""
    # --reset-state wipes the stored definition, so only a definition passed on this launch counts.
    stored = "" if getattr(args, "reset_state", False) else state.custom_specialist_prompt
    if not (getattr(args, "custom_specialist_prompt", "") or stored):
        if getattr(args, "custom_specialist_only", False):
            return f"{CUSTOM_SPECIALIST_ONLY_ENV} requires --custom-specialist-prompt-file and --custom-specialist-description"
        return ""
    if bool(getattr(args, "no_framework_agent", False)) or not state.framework_agent_phase_enabled:
        return "the FRAMEWORK_AGENT phase is disabled, so custom_specialist cannot be guaranteed to run"
    if int(getattr(args, "research_lane_capacity", 1) or 0) <= 0:
        return "--research-lane-capacity 0 builds no specialist executor"
    if getattr(args, "reset_state", False):
        return "--reset-state wipes the custom specialist definition; reset first, then launch with the flags"
    return ""
