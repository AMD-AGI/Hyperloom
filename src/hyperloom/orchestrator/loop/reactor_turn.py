# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reactor-turn helpers: the turn's wall-clock timeout, LLM call and MCP setup records."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path

from hyperloom_common.env import env_float
from hyperloom.inference_optimizer.trace.llm_trace import LLMCallRecord, LLMCallSite, LLMTraceRowError, append_llm_call
from hyperloom.inference_optimizer.trace.orchestration_trace import write_mcp_setup_once
from ..collaborator import ORCHESTRATION_AGENT
from ..roles.base import Backend, BackendTurnResult
from ..state.shared_state import SharedState

log = logging.getLogger(__name__)

REACTOR_TURN_TIMEOUT_ENV = "INFERENCE_OPTIMIZER_REACTOR_TURN_TIMEOUT_SEC"
DEFAULT_REACTOR_TURN_TIMEOUT_SEC = 1800.0


def resolve_reactor_turn_timeout_sec() -> float:
    """Resolve the reactor turn's total wall-clock timeout."""
    value = env_float(REACTOR_TURN_TIMEOUT_ENV, default=DEFAULT_REACTOR_TURN_TIMEOUT_SEC)
    if value > 0.0 and math.isfinite(value):
        return value
    log.warning(
        "%s=%.1f is not a positive finite number; using default %.1fs",
        REACTOR_TURN_TIMEOUT_ENV,
        value,
        DEFAULT_REACTOR_TURN_TIMEOUT_SEC,
    )
    return DEFAULT_REACTOR_TURN_TIMEOUT_SEC


def trace_reactor_llm_call(
    session_dir: Path,
    shared_state: SharedState,
    result: BackendTurnResult,
    *,
    latency_ms: int | None = None,
) -> None:
    """Append one ``llm_calls.jsonl`` row for a reactor turn."""
    try:
        metadata = result.metadata or {}
        has_tokens = any(
            metadata.get(k) is not None
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        if not has_tokens:
            return
        site = LLMCallSite(
            session_id=session_dir.name,
            component=ORCHESTRATION_AGENT,
            role=ORCHESTRATION_AGENT,
            tick=int(shared_state.tick or 0),
            phase=(shared_state.phase or "") or None,
        )
        record = LLMCallRecord.from_metadata(site, metadata, latency_ms=latency_ms)
        append_llm_call(session_dir=session_dir, record=record)
    except LLMTraceRowError:
        log.debug("full-trace: reactor llm_call append failed", exc_info=True)


def trace_mcp_setup(session_dir: Path, backend: Backend) -> None:
    """Persist orchestration MCP setup once per session."""
    setup_getter = getattr(backend, "get_mcp_setup_diagnostic", None)
    if callable(setup_getter):
        setup = setup_getter()
        if isinstance(setup, dict):
            write_mcp_setup_once(session_dir=session_dir, setup=setup)


async def close_backends(backends: Mapping[str, Backend]) -> None:
    """Release every backend holding a live agent session."""
    for name, backend in list(backends.items()):
        closer = getattr(backend, "aclose", None)
        if not callable(closer):
            continue
        try:
            await closer()
        except (OSError, RuntimeError):
            log.exception("Coordinator: closing the %s backend failed", name)
