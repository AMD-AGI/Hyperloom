# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codex managed hook command entrypoint for Forge opportunity analysis guards."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from kernelforge.durable_io import atomic_write_text
from kernelforge.kernel_rewrite_controller.opportunity_agent import (
    _MAX_GREP_MATCHES,
    _MAX_READ_LINES,
    _bounded_result_cap,
)
from kernelforge.kernel_rewrite_controller.task_publisher import pending_rejections

_SHELL_TOOL_NAMES = frozenset(
    {
        "bash",
        "shell",
        "run_terminal_cmd",
        "terminal",
    },
)
_DISALLOWED_TOOL_RE = re.compile(r"(^task|shell|bash|agent)", re.IGNORECASE)
_WRITE_TOOL_NAMES = frozenset(
    {
        "edit",
        "write",
        "multiedit",
        "multi_edit",
        "notebookedit",
        "notebook_edit",
        "apply_patch",
        "applypatch",
    },
)
_INVESTIGATION_CAPS: dict[str, tuple[str, int]] = {
    "read": ("limit", _MAX_READ_LINES),
    "grep": ("head_limit", _MAX_GREP_MATCHES),
}


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def _staging_root(state: dict[str, Any]) -> Path:
    return Path(str(state.get("staging_root") or "")).resolve()


def _tool_input_path(tool_input: dict[str, Any]) -> Path | None:
    for key in ("file_path", "path", "notebook_path", "target", "file"):
        raw = tool_input.get(key)
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
    patch = tool_input.get("patch") or tool_input.get("patches")
    if isinstance(patch, str) and patch.strip():
        return None
    return None


def _resolve_under_staging(path: Path, staging: Path) -> bool:
    candidate = path if path.is_absolute() else staging / path
    try:
        candidate.resolve().relative_to(staging)
        return True
    except ValueError:
        return False


def _is_disallowed_tool(tool_name: str) -> bool:
    lowered = tool_name.strip().lower()
    if lowered in _SHELL_TOOL_NAMES:
        return True
    return _DISALLOWED_TOOL_RE.search(tool_name.strip()) is not None


def _is_write_tool(tool_name: str) -> bool:
    normalized = tool_name.strip().lower().replace("-", "_")
    return normalized in _WRITE_TOOL_NAMES


def _cap_investigation(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    field, ceiling = _INVESTIGATION_CAPS.get(tool_name.strip().lower(), ("", 0))
    if not field:
        return {}
    capped = _bounded_result_cap(tool_input.get(field), ceiling)
    if capped == tool_input.get(field):
        return {}
    updated = dict(tool_input)
    updated[field] = capped
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated,
        }
    }


def handle_pre_tool_use(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(payload.get("tool_name") or "").strip()
    if state.get("deny_shell_tools") and _is_disallowed_tool(tool_name):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Opportunity analysis is limited to direct read, search, and staging write tools."
                ),
            }
        }
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    cap = _cap_investigation(tool_name, tool_input)
    if cap:
        return cap
    if _INVESTIGATION_CAPS.get(tool_name.strip().lower()):
        return {}
    if not _is_write_tool(tool_name):
        return {}
    raw_path = _tool_input_path(tool_input)
    if raw_path is None:
        return {}
    staging = _staging_root(state)
    if _resolve_under_staging(raw_path, staging):
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Opportunity analysis may only write task.json and driver.py under the supplied staging directory."
            ),
        }
    }


def handle_stop(_payload: dict[str, Any], state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    staging = _staging_root(state)
    pending = pending_rejections(staging)
    max_denials = int(state.get("max_stop_denials") or 0)
    denials = int(state.get("stop_denials") or 0)
    if not pending or denials >= max_denials:
        return {}
    denials += 1
    state["stop_denials"] = denials
    _save_state(state_path, state)
    refusals = "\n".join(f"- {draft}: {reason}" for draft, reason in sorted(pending.items()))
    reason = (
        "The host refused these staged tasks, so they were never published:\n"
        f"{refusals}\n"
        "Each refusal is also written to rejection.json inside the draft's own directory. "
        "Correct the task.json the reason names and the host will revalidate it within a few seconds. "
        "If the operator should not be published at all, withdraw the draft by rewriting its task.json as "
        '{"withdrawn": "<why>"}. Do not stop with a draft neither fixed nor withdrawn '
        f"(attempt {denials} of {max_denials})."
    )
    return {"decision": "block", "reason": reason}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forge Codex managed hook runner")
    parser.add_argument("--state", required=True, help="Path to forge_guard_state.json")
    parser.add_argument("event", choices=("pre_tool_use", "stop"))
    args = parser.parse_args(argv)
    state_path = Path(args.state)
    state = _load_state(state_path)
    payload = json.load(sys.stdin)
    event_name = str(payload.get("hook_event_name") or "").strip()
    if args.event == "pre_tool_use" or event_name == "PreToolUse":
        result = handle_pre_tool_use(payload, state)
    else:
        result = handle_stop(payload, state, state_path)
    if result:
        sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
