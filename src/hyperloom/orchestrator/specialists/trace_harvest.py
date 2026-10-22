# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read what a specialist's agent CLI recorded, with the parsers of the CLI that wrote it."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.llm_config import AGENT_BACKEND_CODEX, AGENT_BACKEND_DEEPSEEK
from hyperloom.inference_optimizer.trace.agent_trace import AgentTrace
from hyperloom.inference_optimizer.trace.context_events import record_stream_json_compactions
from hyperloom.inference_optimizer.trace.parse_usage import (
    parse_claude_stream_json_response,
    parse_claude_stream_json_tool_calls,
    parse_claude_stream_json_turn_usages,
    parse_claude_stream_json_usage,
    parse_codex_jsonl_error,
    parse_codex_jsonl_response,
    parse_codex_jsonl_tool_calls,
    parse_codex_jsonl_turn_usages,
    parse_codex_jsonl_usage,
)
from hyperloom.inference_optimizer.trace.request_events import record_stream_json_requests
from hyperloom.inference_optimizer.trace.tool_events import record_stream_json_tools

from .dsh_launch import harvest_dsh_trace


def harvest_agent_trace(backend: str, workspace: Path, process_log: Path) -> AgentTrace:
    """Read the run's trace with the parsers for the CLI that wrote it.

    Cumulative session token usage, the reply text that lands in
    conversations.jsonl, the intel/tool calls, and per-turn usage for
    fine-grained tracing (which falls back to ``usage`` when absent).
    """
    if backend == AGENT_BACKEND_DEEPSEEK:
        return harvest_dsh_trace(workspace)
    if backend == AGENT_BACKEND_CODEX:
        structured_error = parse_codex_jsonl_error(process_log)
        return AgentTrace(
            usage=parse_codex_jsonl_usage(process_log),
            response=parse_codex_jsonl_response(process_log),
            tool_calls=parse_codex_jsonl_tool_calls(process_log),
            turn_usages=parse_codex_jsonl_turn_usages(process_log),
            error=f"codex: {structured_error}" if structured_error else "",
        )
    record_stream_json_requests(process_log)
    record_stream_json_tools(process_log)
    record_stream_json_compactions(process_log)
    return AgentTrace(
        usage=parse_claude_stream_json_usage(process_log),
        response=parse_claude_stream_json_response(process_log),
        tool_calls=parse_claude_stream_json_tool_calls(process_log),
        turn_usages=parse_claude_stream_json_turn_usages(process_log),
    )
