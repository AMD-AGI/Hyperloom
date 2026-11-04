# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an Analysis session may touch: only its staging directory, with every workspace input held immutable."""

from __future__ import annotations

import re
import shlex
import time
from pathlib import Path

from kernelforge.agent_backends import AgentHook, AgentHooks

_WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
_BASH_WRITE_MARKERS = (
    " >",
    ">>",
    "sed -i",
    "perl -i",
    " tee ",
    " rm ",
    " mv ",
    " cp ",
)
_ROOT_FIND_RE = re.compile(r"""(?:^|[;&|]\s*)find\s+["']?/["']?(?:\s|$)""")


class AnalysisProtection:
    """Keep every workspace input immutable while staging remains writable."""

    def __init__(
        self,
        *,
        workspace: Path,
        staging_root: Path,
        protected_paths: tuple[Path, ...],
        deadline_monotonic: float,
    ) -> None:
        self.workspace = workspace.resolve()
        self.staging_root = staging_root.resolve()
        self.protected_paths = tuple(path.resolve() for path in protected_paths)
        self.deadline_monotonic = deadline_monotonic
        self._snapshots = {path: path.read_bytes() for path in self.protected_paths if path.is_file()}
        self._protected_basenames = {path.name for path in self.protected_paths}

    def hooks(self) -> AgentHooks:
        return AgentHooks(
            pre_tool_use=[
                AgentHook(
                    matcher="Edit|Write|MultiEdit|NotebookEdit",
                    callback=self._on_pre_write,
                ),
                AgentHook(
                    matcher="Bash",
                    callback=self._on_pre_bash,
                ),
            ]
        )

    async def _on_pre_write(self, input_data, _tool_use_id, _context) -> dict:
        if input_data.get("tool_name") not in _WRITE_TOOLS:
            return {}
        tool_input = input_data.get("tool_input") or {}
        raw_path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path") or ""
        if self._inside_staging(raw_path):
            return {}
        return self._deny("Analysis output may only be written inside the supplied staging directory.")

    async def _on_pre_bash(self, input_data, _tool_use_id, _context) -> dict:
        if input_data.get("tool_name") != "Bash":
            return {}
        command = str((input_data.get("tool_input") or {}).get("command") or "")
        if _ROOT_FIND_RE.search(command):
            return self._deny(
                "Unbounded root filesystem searches are forbidden. Use the exact "
                "staging and knowledge paths supplied in the Analysis request."
            )
        lowered = f" {command.lower()} "
        has_write_intent = any(marker in lowered for marker in _BASH_WRITE_MARKERS)
        names_protected = any(basename.lower() in lowered for basename in self._protected_basenames)
        if has_write_intent and names_protected:
            return self._deny("This command may modify immutable source, driver, or test inputs.")
        remaining = max(1, int(self.deadline_monotonic - time.monotonic()))
        try:
            tokens = shlex.split(command)
        except ValueError:
            return self._deny("Analysis Bash command could not be parsed safely.")
        if not tokens or Path(tokens[0]).name != "timeout":
            return self._deny(
                "Every Analysis Bash command must be bounded by the shared "
                "session deadline. Prefix it exactly with "
                f"`timeout --signal=TERM --kill-after=5s {remaining}s ...`."
            )
        if "--signal=TERM" not in tokens or "--kill-after=5s" not in tokens:
            return self._deny(
                "Analysis Bash commands must use both `--signal=TERM` and "
                "`--kill-after=5s` so child process groups terminate reliably."
            )
        duration = self._timeout_duration_sec(tokens)
        if duration is None:
            return self._deny("Analysis Bash timeout duration is missing or invalid.")
        if duration > remaining:
            return self._deny(
                f"Analysis Bash timeout {duration:.1f}s exceeds the shared session remaining time {remaining}s."
            )
        return {}

    @staticmethod
    def _timeout_duration_sec(tokens: list[str]) -> float | None:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", token)
            if not match:
                return None
            value = float(match.group(1))
            multiplier = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]
            duration = value * multiplier
            return duration if duration > 0 else None
        return None

    def _inside_staging(self, raw_path: str) -> bool:
        if not raw_path:
            return False
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.staging_root / path
        try:
            path.resolve().relative_to(self.staging_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _deny(reason: str) -> dict:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    def restore_and_report_changes(self) -> list[str]:
        changed = []
        for path, expected in self._snapshots.items():
            current = path.read_bytes() if path.is_file() else None
            if current == expected:
                continue
            changed.append(str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
        return changed


__all__ = ["AnalysisProtection"]
