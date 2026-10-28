# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolated, attributed Claude sessions and their transcripts for Ave."""

from __future__ import annotations

from contextlib import AsyncExitStack, aclosing
from dataclasses import asdict
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ave import AVE_BACKEND
from hyperloom_common.llm.llm_config import DEFAULT_CLAUDE_MODEL, claude_sdk_env_options

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions


async def query_result(prompt: dict, options: ClaudeAgentOptions, *, operation: str, transcript: Path) -> str:
    """Apply Ave's session policy and retain tool output, returning the final result."""
    from claude_agent_sdk import ClaudeSDKError, query

    options.model = options.model or os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
    options.system_prompt = (
        files("ave").joinpath(f"instructions/{operation}.md").read_text().split("-->", 1)[-1].strip()
    )
    options.allowed_tools = options.tools
    options.setting_sources = []
    options.strict_mcp_config = True
    options.env = claude_sdk_env_options(
        model=options.model, env=options.env or None, component=AVE_BACKEND, operation=operation
    ).get("env", options.env)
    try:
        async with AsyncExitStack() as stack:
            log = stack.enter_context(transcript.open("w", encoding="utf-8"))
            stream = await stack.enter_async_context(
                aclosing(query(prompt=json.dumps(prompt, default=str), options=options))
            )
            async for message in stream:
                log.write(json.dumps({"type": type(message).__name__, **asdict(message)}, default=str) + "\n")
                log.flush()
                if getattr(message, "is_error", False):
                    raise RuntimeError(f"Claude session failed: {message.result}")
                result = getattr(message, "result", None)
                if isinstance(result, str) and result:
                    return result
    except ClaudeSDKError as error:
        raise RuntimeError(f"Claude session failed: {error}") from error
    raise RuntimeError("Claude session ended without a result")
