# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The production completion: it returns the reply text with its token usage and turns every transport failure into
one error; the meter adds up the usage of the replies.

``aanthropic_completion`` is the network boundary, so these tests replace it in place.
"""

from __future__ import annotations

import httpx
import pytest
from claude_agent_sdk import ClaudeSDKError

from hyperloom.common.llm_config import AnthropicCompletionRequest, AnthropicMessageResult
from hyperloom_multibranch.scheduling import llm


def _answering(monkeypatch: pytest.MonkeyPatch, usage: object) -> list[AnthropicCompletionRequest]:
    asked: list[AnthropicCompletionRequest] = []

    async def transport(
        *, request: AnthropicCompletionRequest, env: object = None, component: str = "", operation: str = ""
    ) -> AnthropicMessageResult:
        asked.append(request)
        return AnthropicMessageResult(text="reply", stop_reason="end_turn", usage=usage)

    monkeypatch.setattr(llm, "aanthropic_completion", transport)
    return asked


async def test_the_completion_returns_the_reply_text_and_its_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _answering(
        monkeypatch,
        {"input_tokens": 40, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 60, "output_tokens": 25},
    )

    reply = await llm.anthropic_completion("m", component="c", operation="judge")("sys", "user")

    assert reply == llm.CompletionReply("reply", llm.CompletionUsage(input_tokens=1000, output_tokens=25))
    (request,) = asked
    assert (request.model, request.system, list(request.messages)) == (
        "m",
        "sys",
        [{"role": "user", "content": "user"}],
    )


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": None, "output_tokens": -1}])
async def test_a_reply_without_usable_counts_used_no_tokens(monkeypatch: pytest.MonkeyPatch, usage: object) -> None:
    _answering(monkeypatch, usage)

    reply = await llm.anthropic_completion("m", component="c", operation="judge")("sys", "user")

    assert reply == llm.CompletionReply("reply", llm.CompletionUsage())


async def test_the_meter_adds_up_the_tokens_of_every_completion_it_meters() -> None:
    meter = llm.CompletionMeter()
    failing = True

    async def judge(system: str, user: str) -> llm.CompletionReply:
        return llm.CompletionReply("j", llm.CompletionUsage(100, 10))

    async def resolver(system: str, user: str) -> llm.CompletionReply:
        if failing:
            raise llm.CompletionFailed("patch_resolve call failed")
        return llm.CompletionReply("r", llm.CompletionUsage(7, 3))

    with pytest.raises(llm.CompletionFailed):
        await meter.metered(resolver)("sys", "user")
    failing = False
    assert (await meter.metered(judge)("sys", "user")).text == "j"
    await meter.metered(resolver)("sys", "user")

    assert meter.used == llm.CompletionUsage(input_tokens=107, output_tokens=13)


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
