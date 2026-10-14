# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The production completion: it returns the reply text and turns every transport failure into one error.

``aanthropic_completion`` is the network boundary, so these tests replace it in place.
"""

from __future__ import annotations

import httpx
import pytest
from claude_agent_sdk import ClaudeSDKError

from hyperloom.common.llm_config import AnthropicMessageResult
from hyperloom_multibranch.scheduling import llm


async def test_the_completion_returns_the_reply_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def transport(**kwargs: object) -> AnthropicMessageResult:
        return AnthropicMessageResult(text="reply", stop_reason=None, usage=None)

    monkeypatch.setattr(llm, "aanthropic_completion", transport)

    assert await llm.anthropic_completion("m", component="c", operation="judge")("sys", "user") == "reply"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("anthropic messages status=529"),
        httpx.ReadTimeout("slow"),
        TimeoutError("slow"),
        OSError("reset"),
        ClaudeSDKError("cli exited"),
    ],
)
async def test_every_transport_failure_is_a_completion_failure(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    async def transport(**kwargs: object) -> AnthropicMessageResult:
        raise error

    monkeypatch.setattr(llm, "aanthropic_completion", transport)

    with pytest.raises(llm.CompletionFailed, match=type(error).__name__):
        await llm.anthropic_completion("m", component="c", operation="judge")("sys", "user")


async def test_a_bug_is_not_a_completion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def transport(**kwargs: object) -> AnthropicMessageResult:
        raise KeyError("bug")

    monkeypatch.setattr(llm, "aanthropic_completion", transport)

    with pytest.raises(KeyError):
        await llm.anthropic_completion("m", component="c", operation="judge")("sys", "user")
