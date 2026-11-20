# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist workspace file protocol: redacted transcript, heartbeat and done artifacts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from hyperloom_common.fs_utils import append_text_nofollow
from hyperloom_common.io import JsonFormat, WriteOptions, atomic_write_json
from hyperloom.common.env_safety import BENCHMARK_SECRET_ENV_NAMES, redact_secret_values
from hyperloom_common.timeutil import now_iso

from ..state.retest_view import scrub


_SECRET_ENV_NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            *BENCHMARK_SECRET_ENV_NAMES,
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_CUSTOM_HEADERS",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAW_API_KEY",
            "GITHUB_TOKEN",
            "HF_TOKEN",
            "HF_TOKEN_2",
            "HYPERLOOM_GIT_TOKEN",
            "HYPERLOOM_PR_CI_GH_TOKEN",
            "LLM_API_KEY",
            "LLM_GATEWAY_KEY",
            "OPENAI_API_KEY",
            "OPENAI_CUSTOM_HEADERS",
            "SAFE_API_KEY",
        }
    )
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<key>\b(?:"
    + "|".join(re.escape(name) for name in _SECRET_ENV_NAMES)
    + r")\b)(?P<sep>\s*(?:=|:)\s*)(?P<quote>['\"]?)(?!\[REDACTED\])"
    + r"(?P<value>[^\s,'\"\]}]+)(?P=quote)",
    re.IGNORECASE,
)


def _safe_redact(s: str) -> str:
    """Redact obvious secrets from a transcript line before writing to disk.

    Scans for known environment-variable secret names, Authorization/Bearer
    headers, and common token shapes, then masks the secret value while leaving
    enough surrounding context for debugging.

    Args:
        s (str): The raw transcript line that may contain secret material.

    Returns:
        str: The line with recognised secret values replaced by
            ``[REDACTED]``.
    """
    out = redact_secret_values(s)
    return _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group('key')}{m.group('sep')}{m.group('quote')}[REDACTED]{m.group('quote')}",
        out,
    )


def redact_transcript_value(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_redact(value)
    if isinstance(value, dict):
        return {str(k): redact_transcript_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_transcript_value(v) for v in value]
    if isinstance(value, tuple):
        return [redact_transcript_value(v) for v in value]
    return value


def transcript_path(workspace: Path | None) -> Path | None:
    """Return the ``transcript.jsonl`` path within the workspace.

    Args:
        workspace (Path | None): The per-task workspace directory.

    Returns:
        Path | None: The transcript path, or ``None`` when no workspace.
    """
    return (workspace / "transcript.jsonl") if workspace else None


def heartbeat_path(workspace: Path | None) -> Path | None:
    """Return the ``heartbeat.json`` path within the workspace.

    Args:
        workspace (Path | None): The per-task workspace directory.

    Returns:
        Path | None: The heartbeat path, or ``None`` when no workspace.
    """
    return (workspace / "heartbeat.json") if workspace else None


def done_path(workspace: Path | None) -> Path | None:
    """Return the ``specialist_done.json`` path within the workspace.

    Args:
        workspace (Path | None): The per-task workspace directory.

    Returns:
        Path | None: The done-artifact path, or ``None`` when no workspace.
    """
    return (workspace / "specialist_done.json") if workspace else None


def append_transcript(
    workspace: Path | None,
    turn: int,
    entry: dict[str, Any],
) -> None:
    """Append one JSON line to the workspace ``transcript.jsonl``.

    No-ops when no workspace is configured.

    Args:
        workspace (Path | None): The per-task workspace directory.
        turn (int): The turn index the entry belongs to.
        entry (dict[str, Any]): The transcript record to serialise.
    """
    path = transcript_path(workspace)
    if path is None:
        return
    safe_entry = redact_transcript_value(entry)
    line = json.dumps(
        {
            "turn": turn,
            "ts": now_iso(),
            **safe_entry,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    append_text_nofollow(path.parent, (path.name,), line + "\n")


def write_heartbeat(
    workspace: Path | None,
    *,
    turn: int,
    max_turns: int,
    status: str,
) -> None:
    """Atomically write the workspace ``heartbeat.json``.

    Writes to a temp file then ``os.replace``-s it into place so readers
    never observe a partial write. No-ops when no workspace is configured.

    Args:
        workspace (Path | None): The per-task workspace directory.
        turn (int): The current turn index.
        max_turns (int): The configured turn budget.
        status (str): A short lifecycle status string.
    """
    path = heartbeat_path(workspace)
    if path is None:
        return
    payload = {
        "ts": now_iso(),
        "ts_unix": time.time(),
        "turn": turn,
        "max_turns": max_turns,
        "status": status,
    }
    atomic_write_json(path, payload, fmt=JsonFormat(indent=None), options=WriteOptions())


def write_specialist_done(
    workspace: Path | None,
    payload: dict[str, Any],
) -> None:
    """Write the ``specialist_done.json`` artifact with a timestamp, group-readable (the retest runner reads it).

    Atomic (temp + ``os.replace``); every string is scrubbed of credentials. No-ops when no workspace is set.

    Args:
        workspace (Path | None): The per-task workspace directory.
        payload (dict[str, Any]): The ``specialist_done`` payload to persist.
    """
    path = done_path(workspace)
    if path is None:
        return
    atomic_write_json(path, scrub({"ts": now_iso(), **payload}), options=WriteOptions(group_read=True))
