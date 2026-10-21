# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claude Agent SDK results and provider failures for Ave optimization."""

from __future__ import annotations

from contextlib import aclosing, nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions


async def query_result(prompt: str, options: ClaudeAgentOptions, *, transcript: Path | None = None) -> str:
    """Return the session result and optionally record its messages, including tool output."""
    from claude_agent_sdk import ClaudeSDKError, query

    try:
        with transcript.open("w", encoding="utf-8") if transcript else nullcontext() as log:
            async with aclosing(query(prompt=prompt, options=options)) as stream:
                async for message in stream:
                    if log is not None:
                        record = {"type": type(message).__name__, **asdict(message)}
                        log.write(json.dumps(record, default=str) + "\n")
                        log.flush()
                    if getattr(message, "is_error", False):
                        raise RuntimeError(f"Claude session failed: {message.result}")
                    result = getattr(message, "result", None)
                    if isinstance(result, str) and result:
                        return result
    except ClaudeSDKError as error:
        raise RuntimeError(f"Claude session failed: {error}") from error
    raise RuntimeError("Claude session ended without a result")
