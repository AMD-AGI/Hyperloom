# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tool hook that keeps a fusion author's direct edits on the files it was given."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from kernelforge.agent_backends.base import AgentHook, AgentHooks


def target_file_hooks(
    target_files: list[str],
    workdir: str,
    allows_new_path: Optional[Callable[[str], bool]] = None,
) -> AgentHooks | None:
    """Deny direct SDK edit tools outside the caller's writable path set."""
    if not target_files:
        return None
    root = Path(workdir).resolve()
    targets = {(Path(value) if Path(value).is_absolute() else root / value).resolve() for value in target_files}

    async def _deny_non_target(input_data: dict, _tool_use_id: Any, _context: Any) -> dict:
        tool_input = input_data.get("tool_input") or {}
        raw_path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path") or ""
        if raw_path:
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.resolve() in targets:
                return {}
            if allows_new_path is not None and allows_new_path(str(candidate)):
                return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Authoring may edit only the exact target files supplied by "
                    "forge-fuse, plus any new fused-kernel module the run "
                    "explicitly permits; tests, harness oracles, and unrelated "
                    "source are protected."
                ),
            }
        }

    return AgentHooks(
        pre_tool_use=[
            AgentHook(
                matcher="Edit|Write|MultiEdit|NotebookEdit",
                callback=_deny_non_target,
            )
        ]
    )


__all__ = ["target_file_hooks"]
