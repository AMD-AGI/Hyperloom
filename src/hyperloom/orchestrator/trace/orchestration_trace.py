# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable diagnostics for orchestration turns."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from hyperloom.common.io import append_jsonl, atomic_write_json
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import (
    agent_mcp_setup_path,
    orchestration_turns_path,
)

from .conversation_trace import redact_secrets

log = logging.getLogger(__name__)

_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "turn_id",
        "ts",
        "tick",
        "phase",
        "outcome",
        "backend",
        "model",
        "sdk_name",
        "sdk_version",
        "cli_version",
        "gateway_endpoint",
        "request_id",
        "resume_requested",
        "previous_session_id_hash",
        "session_id_hash",
        "new_session",
        "max_turns",
        "timeout_sec",
        "reasoning_effort",
        "thinking",
        "prompt_sha256",
        "prompt_chars",
        "system_prompt_sha256",
        "system_prompt_chars",
        "allowed_tools",
        "mcp_servers",
        "emit_intent_registered",
        "messages",
        "result",
        "raw_text",
        "tool_blocks",
        "parse_errors",
        "usage",
        "stderr_tail",
        "sdk_boundary_error",
        "error_type",
        "error_message",
        "traceback",
    }
)


class OrchestrationTraceRowError(ValueError):
    """Raised when an orchestration trace row violates its schema."""


def _sha256(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_secrets(str(value))


@dataclass
class OrchestrationTurnRecord:
    """One orchestration backend invocation."""

    session_id: str
    turn_id: str
    tick: int | None
    phase: str | None
    outcome: str
    backend: str
    model: str | None
    sdk_name: str | None
    sdk_version: str | None
    cli_version: str | None
    gateway_endpoint: str | None
    request_id: str | None
    resume_requested: bool
    previous_session_id_hash: str | None
    session_id_hash: str | None
    new_session: bool | None
    max_turns: int | None
    timeout_sec: float | None
    reasoning_effort: str | None
    thinking: Any
    prompt: str
    system_prompt: str
    allowed_tools: list[str]
    mcp_servers: list[str]
    emit_intent_registered: bool
    messages: list[dict[str, Any]]
    result: str
    raw_text: str
    tool_blocks: list[dict[str, Any]]
    parse_errors: list[str]
    usage: dict[str, Any]
    stderr_tail: list[str]
    sdk_boundary_error: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Serialize a redacted append-only row."""
        return {
            "session_id": str(self.session_id),
            "turn_id": str(self.turn_id),
            "ts": now_iso(),
            "tick": self.tick,
            "phase": self.phase,
            "outcome": str(self.outcome),
            "backend": str(self.backend),
            "model": self.model,
            "sdk_name": self.sdk_name,
            "sdk_version": self.sdk_version,
            "cli_version": self.cli_version,
            "gateway_endpoint": _safe_value(self.gateway_endpoint),
            "request_id": self.request_id,
            "resume_requested": bool(self.resume_requested),
            "previous_session_id_hash": self.previous_session_id_hash,
            "session_id_hash": self.session_id_hash,
            "new_session": self.new_session,
            "max_turns": self.max_turns,
            "timeout_sec": self.timeout_sec,
            "reasoning_effort": self.reasoning_effort,
            "thinking": _safe_value(self.thinking),
            "prompt_sha256": _sha256(self.prompt),
            "prompt_chars": len(self.prompt),
            "system_prompt_sha256": _sha256(self.system_prompt),
            "system_prompt_chars": len(self.system_prompt),
            "allowed_tools": _safe_value(self.allowed_tools),
            "mcp_servers": _safe_value(self.mcp_servers),
            "emit_intent_registered": bool(self.emit_intent_registered),
            "messages": _safe_value(self.messages),
            "result": _safe_value(self.result),
            "raw_text": _safe_value(self.raw_text),
            "tool_blocks": _safe_value(self.tool_blocks),
            "parse_errors": _safe_value(self.parse_errors),
            "usage": _safe_value(self.usage),
            "stderr_tail": _safe_value(self.stderr_tail),
            "sdk_boundary_error": _safe_value(self.sdk_boundary_error),
            "error_type": self.error_type,
            "error_message": _safe_value(self.error_message),
            "traceback": _safe_value(self.traceback),
        }


def append_orchestration_turn(*, session_dir: Path, record: OrchestrationTurnRecord) -> None:
    """Append an orchestration turn without disrupting the coordinator."""
    row = record.to_row()
    if set(row) != _ROW_FIELDS:
        raise OrchestrationTraceRowError("orchestration_turns row violates closed schema")
    if not row["session_id"].strip():
        raise OrchestrationTraceRowError("orchestration_turns row requires session_id")
    try:
        append_jsonl(orchestration_turns_path(session_dir), row, make_parents=True, ensure_ascii=False, sort_keys=True)
    except OSError as exc:
        log.warning("orchestration_trace: append failed for session_id=%s: %r", record.session_id, exc)


def write_mcp_setup_once(*, session_dir: Path, setup: dict[str, Any]) -> None:
    """Persist the orchestration MCP setup once per session."""
    path = agent_mcp_setup_path(session_dir, "orchestration")
    if path.exists():
        return
    try:
        atomic_write_json(
            path,
            {"ts": now_iso(), "schema_version": 1, **_safe_value(setup)},
            ensure_ascii=False,
            trailing_newline=True,
            mode=0o600,
        )
    except OSError as exc:
        log.warning("orchestration_trace: MCP setup write failed for %s: %r", session_dir.name, exc)


_DATACLASS_FIELDS = frozenset(field.name for field in fields(OrchestrationTurnRecord))
assert (
    (_DATACLASS_FIELDS - {"prompt", "system_prompt"})
    | {"ts", "prompt_sha256", "prompt_chars", "system_prompt_sha256", "system_prompt_chars"}
    == _ROW_FIELDS
)


__all__ = [
    "OrchestrationTraceRowError",
    "OrchestrationTurnRecord",
    "append_orchestration_turn",
    "write_mcp_setup_once",
]
