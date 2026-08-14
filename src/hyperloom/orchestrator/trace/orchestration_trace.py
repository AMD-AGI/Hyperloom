# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-session orchestration MCP setup persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import (
    agent_mcp_setup_path,
)

from .conversation_trace import redact_secrets

log = logging.getLogger(__name__)


def _safe_value(value: Any) -> Any:
    """Recursively redact recognizable credential values in a JSON-compatible structure."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_secrets(str(value))


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


__all__ = [
    "write_mcp_setup_once",
]
