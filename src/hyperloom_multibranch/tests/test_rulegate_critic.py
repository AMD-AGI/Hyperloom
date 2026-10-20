# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
from types import SimpleNamespace

import pytest

from hyperloom.common import llm_config
from hyperloom.contracts.measurement import NOT_REPRODUCED, REPRODUCED
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.notebook.records import LlmUsage
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.rulegate.critic import (
    CRITIC_MAX_TOKENS,
    CRITIC_TIMEOUT_SECONDS,
    CriticReply,
    CriticUnavailable,
    CriticVerdict,
    FixReview,
    LlmCritic,
    NoCritic,
    ReproVerdicts,
    anthropic_critic,
)
from hyperloom_multibranch.rulegate.fix_case import FixCase

WORKLOAD = "qwen3-8b on vllm, ISL 8192, OSL 1024, concurrency 64"
CHANGE = RecipeDelta(config=ConfigDelta(server_args="--kv-cache-dtype fp8"))
PATCH = "--- a/vllm/attention/fp8.py\n+++ b/vllm/attention/fp8.py\n+    scale = scale.clamp(max=448.0)\n"
CASE = FixCase(
    '#!/bin/bash\ncurl -sf "$HL_BASE_URL/v1/completions" | grep -q nan && exit 1\nexit 0\n', "fp8 scales overflow"
)
REACHABLE = json.dumps({"reachable": True, "reason": "8k prompts are the workload's own"})
TOKENS = TokenCounts(input=900, output=60, cache_read=2000)
COST_USD = 0.02
ANTHROPIC_USAGE = {"input_tokens": 900, "output_tokens": 60, "cache_read_input_tokens": 2000}


def _review(root: str = "") -> FixReview:
    slower = Judgement(Verdict.REGRESSED, -2.5, 2.0, ("r3", "r3b"), {"output_throughput": 975.0}, True)
    repro = ReproVerdicts(parent=REPRODUCED, candidate=NOT_REPRODUCED, root=root)
    return FixReview(CHANGE, {"patches/vllm/fp8.patch": PATCH}, CASE, WORKLOAD, slower, repro)


class _Model:
    """Answers each call with the next of ``answers``, each reply using :data:`TOKENS` that cost :data:`COST_USD`;
    an exception in them is raised instead."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, system: str, user: str) -> CriticReply:
        self.calls.append((system, user))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return CriticReply(answer, TOKENS, COST_USD)


class _Usages:
    def __init__(self) -> None:
        self.recorded: list[LlmUsage] = []

    async def record_llm_usage(self, usage: LlmUsage) -> None:
        self.recorded.append(usage)


def _critic(model: _Model, usages: _Usages | None = None) -> LlmCritic:
    return LlmCritic(model, _Usages() if usages is None else usages)


async def test_the_critic_sees_the_change_the_case_the_workload_and_the_verdicts():
    model = _Model(REACHABLE)

    verdict = await _critic(model).review(_review(root=REPRODUCED))

    assert verdict == CriticVerdict(True, "8k prompts are the workload's own")
    ((system, user),) = model.calls
    assert "reachable" in system and "JSON" in system
    for fact in (WORKLOAD, PATCH, CASE.repro_sh, CASE.diagnosis, "regressed", "-2.50%"):
        assert fact in user
    assert json.dumps(CHANGE.to_document(), sort_keys=True) in user
    assert f"parent: {REPRODUCED}" in user and f"candidate: {NOT_REPRODUCED}" in user
    assert f"root commit: {REPRODUCED}" in user


async def test_only_a_foundation_fix_shows_the_root_commit_s_verdict():
    model = _Model(REACHABLE)

    await _critic(model).review(_review())

    ((_system, user),) = model.calls
    assert "root commit" not in user


async def test_an_unreadable_answer_is_asked_again_once():
    model = _Model("I think it is reachable.", json.dumps({"reachable": False, "reason": "needs 1M-token prompts"}))

    assert await _critic(model).review(_review()) == CriticVerdict(False, "needs 1M-token prompts")
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "answer",
    [
        f"```json\n{REACHABLE}\n```",
        json.dumps({"reachable": "yes", "reason": "it is"}),
        json.dumps({"reachable": True, "reason": " "}),
        json.dumps({"reachable": True, "reason": "it is", "confidence": 0.9}),
        json.dumps([True, "it is"]),
    ],
)
async def test_an_answer_that_is_not_exactly_the_verdict_object_is_refused_after_one_retry(answer):
    model = _Model(answer, answer)

    verdict = await _critic(model).review(_review())

    assert not verdict.reachable and "no usable answer" in verdict.reason
    assert len(model.calls) == 2


async def test_a_critic_that_cannot_be_reached_refuses_the_fix_and_says_why():
    model = _Model(CriticUnavailable("RuntimeError: status=529"), CriticUnavailable("TimeoutError: read timed out"))

    verdict = await _critic(model).review(_review())

    assert not verdict.reachable
    assert "TimeoutError: read timed out" in verdict.reason


async def test_an_error_that_is_no_failed_call_is_not_taken_for_a_refusal():
    with pytest.raises(KeyError):
        await _critic(_Model(KeyError("bug"))).review(_review())


async def test_no_critic_refuses_every_fix():
    verdict = await NoCritic().review(_review())

    assert not verdict.reachable and "no critic" in verdict.reason


async def test_a_reply_is_recorded_as_the_critic_s_spend_with_no_branch_and_a_failed_call_records_nothing():
    usages = _Usages()

    await _critic(_Model(CriticUnavailable("RuntimeError: status=529"), REACHABLE), usages).review(_review())

    assert usages.recorded == [LlmUsage("", UsageSource.CRITIC, COST_USD, TOKENS)]


async def test_an_unreadable_reply_is_recorded_too():
    usages = _Usages()

    await _critic(_Model("yes", "yes"), usages).review(_review())

    assert usages.recorded == [LlmUsage("", UsageSource.CRITIC, COST_USD, TOKENS)] * 2


async def _ask_anthropic(monkeypatch, transport, model: str = "claude-x") -> CriticReply:
    monkeypatch.setattr(llm_config, "aanthropic_completion", transport)
    return await anthropic_critic(model, component="hyperloom-mb")("the system", "the user")


async def test_the_anthropic_critic_asks_the_configured_model(monkeypatch):
    asked = {}

    async def transport(*, request, env=None, component="", operation=""):
        asked.update(request=request, component=component, operation=operation)
        return SimpleNamespace(text=REACHABLE, usage=ANTHROPIC_USAGE)

    assert await _ask_anthropic(monkeypatch, transport) == CriticReply(REACHABLE, TOKENS, None)
    request = asked["request"]
    assert (request.model, request.system, asked["component"]) == ("claude-x", "the system", "hyperloom-mb")
    assert list(request.messages) == [{"role": "user", "content": "the user"}]
    assert (request.max_tokens, request.timeout_s) == (CRITIC_MAX_TOKENS, CRITIC_TIMEOUT_SECONDS)


async def test_an_anthropic_reply_on_a_listed_model_is_priced_from_the_price_sheet(monkeypatch):
    async def transport(*, request, env=None, component="", operation=""):
        return SimpleNamespace(text=REACHABLE, usage=ANTHROPIC_USAGE)

    reply = await _ask_anthropic(monkeypatch, transport, model="claude-opus-5-5")

    price = price_of("claude-opus-5-5")
    assert price is not None and reply == CriticReply(REACHABLE, TOKENS, price.cost_usd(TOKENS))


async def test_a_failed_anthropic_call_is_a_critic_unavailable(monkeypatch):
    async def transport(*, request, env=None, component="", operation=""):
        raise RuntimeError("anthropic messages status=529")

    with pytest.raises(CriticUnavailable, match="RuntimeError: anthropic messages status=529"):
        await _ask_anthropic(monkeypatch, transport)


async def test_an_anthropic_call_that_times_out_in_asyncio_is_a_critic_unavailable(monkeypatch):
    async def transport(*, request, env=None, component="", operation=""):
        raise asyncio.TimeoutError

    with pytest.raises(CriticUnavailable, match="TimeoutError"):
        await _ask_anthropic(monkeypatch, transport)
