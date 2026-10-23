# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The production completion: it returns the reply text with its tokens, cached ones apart, and their cost from the
price sheet, refuses a model the sheet cannot price, and turns every transport failure into one error; a recorded
completion writes what each reply cost to the notebook as the scheduler's own spend.

``aanthropic_completion`` is the network boundary, so these tests replace it in place.
"""

from __future__ import annotations

import httpx
import pytest
from claude_agent_sdk import ClaudeSDKError

from hyperloom.common.llm_config import AnthropicCompletionRequest, AnthropicMessageResult
from hyperloom_multibranch.contracts.llm import CompletionFailed, CompletionReply
from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.notebook.records import LlmUsage, LlmUsageRecord
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.scheduling import llm

MODEL = "claude-sonnet-5-5"


def _answering(monkeypatch: pytest.MonkeyPatch, usage: object) -> list[AnthropicCompletionRequest]:
    asked: list[AnthropicCompletionRequest] = []

    async def transport(
        *, request: AnthropicCompletionRequest, env: object = None, component: str = "", operation: str = ""
    ) -> AnthropicMessageResult:
        asked.append(request)
        return AnthropicMessageResult(text="reply", stop_reason="end_turn", usage=usage)

    monkeypatch.setattr(llm, "aanthropic_completion", transport)
    return asked


async def test_the_completion_returns_the_reply_text_its_tokens_and_their_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _answering(
        monkeypatch,
        {"input_tokens": 40, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 60, "output_tokens": 25},
    )

    reply = await llm.anthropic_completion(MODEL, component="c", operation="judge")("sys", "user")

    tokens = TokenCounts(input=40, output=25, cache_read=900, cache_write=60)
    assert reply == CompletionReply("reply", tokens, price_of(MODEL).cost_usd(tokens))
    assert reply.cost_usd == pytest.approx((40 * 2 + 25 * 10 + 900 * 0.1 + 60 * 2.5) / 1e6)
    (request,) = asked
    assert (request.model, request.system, list(request.messages)) == (
        MODEL,
        "sys",
        [{"role": "user", "content": "user"}],
    )


async def test_cache_writes_split_by_lifetime_are_priced_at_their_own_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    _answering(
        monkeypatch,
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 300,
            "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
            "output_tokens": 5,
        },
    )

    reply = await llm.anthropic_completion(MODEL, component="c", operation="judge")("sys", "user")

    assert reply.tokens == TokenCounts(input=10, output=5, cache_write=100, cache_write_1h=200)
    assert reply.cost_usd == pytest.approx((10 * 2 + 5 * 10 + 100 * 2.5 + 200 * 4) / 1e6)


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": None, "output_tokens": -1}])
async def test_a_reply_without_usable_counts_used_no_tokens(monkeypatch: pytest.MonkeyPatch, usage: object) -> None:
    _answering(monkeypatch, usage)

    reply = await llm.anthropic_completion(MODEL, component="c", operation="judge")("sys", "user")

    assert reply == CompletionReply("reply", TokenCounts(), 0.0)


async def test_a_model_the_price_sheet_does_not_list_replies_with_its_tokens_and_no_cost(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _answering(monkeypatch, {"input_tokens": 40, "cache_read_input_tokens": 900, "output_tokens": 25})

    reply = await llm.anthropic_completion("glm-5-3", component="c", operation="judge")("sys", "user")

    assert reply == CompletionReply("reply", TokenCounts(input=40, output=25, cache_read=900), None)
    assert "judge runs on glm-5-3, which the price sheet does not list" in caplog.text


class _Usages:
    def __init__(self) -> None:
        self.recorded: list[LlmUsage] = []

    async def record_llm_usage(self, usage: LlmUsage) -> LlmUsageRecord:
        self.recorded.append(usage)
        return LlmUsageRecord(f"u{len(self.recorded)}", 0.0, usage)


async def test_a_recorded_completion_writes_each_reply_s_cost_as_the_scheduler_s_own() -> None:
    usages = _Usages()
    failing = True

    async def resolver(system: str, user: str) -> CompletionReply:
        if failing:
            raise CompletionFailed("patch_resolve call failed")
        return CompletionReply("r", TokenCounts(input=7, output=3, cache_read=50), 0.25)

    recorded = llm.recorded(resolver, usages, UsageSource.RESOLVER)
    with pytest.raises(CompletionFailed):
        await recorded("sys", "user")
    failing = False
    assert (await recorded("sys", "user")).text == "r"

    assert usages.recorded == [LlmUsage("", UsageSource.RESOLVER, 0.25, TokenCounts(input=7, output=3, cache_read=50))]


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

    with pytest.raises(CompletionFailed, match=type(error).__name__):
        await llm.anthropic_completion(MODEL, component="c", operation="judge")("sys", "user")


async def test_a_bug_is_not_a_completion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def transport(**kwargs: object) -> AnthropicMessageResult:
        raise KeyError("bug")

    monkeypatch.setattr(llm, "aanthropic_completion", transport)

    with pytest.raises(KeyError):
        await llm.anthropic_completion(MODEL, component="c", operation="judge")("sys", "user")
