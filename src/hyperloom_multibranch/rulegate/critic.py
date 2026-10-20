# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The review a correctness fix passes after its repro case proved it: is the defect's trigger reachable under the
target workload, so the fix matters for what the session serves?

The review only ever refuses by default: a critic that gives no usable answer after one retry, or no critic at all,
refuses the fix with the reason, and the candidate is judged on its performance alone. What each reply cost, an
unreadable one included, goes to the usage sink the critic is built on as the critic's spend, with no branch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.judgement import Judgement
from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.notebook.interfaces import UsageSink
from hyperloom_multibranch.contracts.notebook.records import LlmUsage
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.rulegate.fix_case import CASE_SCRIPT, DIAGNOSIS_FILE, FixCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CriticReply:
    """The reply text, its tokens and what they cost, ``None`` for a model the price sheet does not list."""

    text: str
    tokens: TokenCounts
    cost_usd: float | None


#: ``(system, user) -> reply``; raises :class:`CriticUnavailable` when the call fails.
CriticCompletion = Callable[[str, str], Awaitable[CriticReply]]

#: One call and one retry, for a failed call or an unreadable answer alike.
CRITIC_ATTEMPTS = 2
CRITIC_MAX_TOKENS = 1024
CRITIC_TIMEOUT_SECONDS = 120.0
_REACHABLE = "reachable"
_REASON = "reason"

_SYSTEM = f"""\
You review a correctness fix found while optimizing an LLM inference server for one workload. Its repro case \
({CASE_SCRIPT}, run against a live server) already showed the defect on the parent commit and not on the candidate. \
Decide one thing: is the defect's trigger reachable under the target workload? A trigger only an input, setting or \
load outside that workload reaches is not reachable. Judge from the diagnosis, the case and the change; do not judge \
the change's performance.
Answer with one JSON object and nothing else: {{"{_REACHABLE}": true or false, "{_REASON}": "one or two sentences"}}."""
_REVIEW = """\
Target workload: {workload}

Judgement of the candidate: {verdict}, {gain_pct:+.2f}% against its parent (bar {threshold_pct:.2f}%).
Repro case verdicts, on the parent: {parent}; on the candidate: {candidate}.{root}

{diagnosis_file}:
{diagnosis}

{case_script}:
{repro_sh}

The candidate's change, as a recipe delta:
{delta}

{patches}"""
_ROOT = " On the session's root commit: {root}."
_PATCH = "Patch {path}:\n{text}"
_NO_PATCH = "The change carries no patch."


@dataclass(frozen=True)
class ReproVerdicts:
    """The case's verdicts; ``root`` is the session root's, run for a foundation fix only."""

    parent: str
    candidate: str
    root: str = ""


@dataclass(frozen=True)
class FixReview:
    """What the critic sees of a fix: the change and its patch texts by path, the case, and how it was judged."""

    delta: RecipeDelta
    patches: Mapping[str, str]
    case: FixCase
    workload: str
    judgement: Judgement
    repro: ReproVerdicts


@dataclass(frozen=True)
class CriticVerdict:
    reachable: bool
    reason: str


class FixReviewer(Protocol):
    async def review(self, review: FixReview) -> CriticVerdict: ...


class CriticAnswerError(ValueError):
    """The critic's reply is not exactly the verdict object."""


class CriticUnavailable(RuntimeError):
    """The critic could not be asked or gave no reply."""


class NoCritic:
    """The reviewer of a session that configures none: every fix is refused, never approved unseen."""

    async def review(self, review: FixReview) -> CriticVerdict:
        return CriticVerdict(False, f"no critic is configured to review the fix of {review.judgement.request_ids[0]}")


class LlmCritic:
    """The critic over ``complete``; ``usage`` is the scheduler's, as only it may record spend with no branch."""

    def __init__(self, complete: CriticCompletion, usage: UsageSink) -> None:
        self._complete = complete
        self._usage = usage

    async def review(self, review: FixReview) -> CriticVerdict:
        user = _prompt(review)
        failure = ""
        for attempt in range(1, CRITIC_ATTEMPTS + 1):
            try:
                reply = await self._complete(_SYSTEM, user)
            except CriticUnavailable as exc:
                failure = str(exc)
            else:
                await self._usage.record_llm_usage(LlmUsage("", UsageSource.CRITIC, reply.cost_usd, reply.tokens))
                try:
                    return _verdict(reply.text)
                except CriticAnswerError as exc:
                    failure = str(exc)
            logger.warning("fix critic attempt %d of %d failed: %s", attempt, CRITIC_ATTEMPTS, failure)
        return CriticVerdict(False, f"the critic gave no usable answer: {failure}")


def _prompt(review: FixReview) -> str:
    judgement, repro = review.judgement, review.repro
    patches = [_PATCH.format(path=path, text=text) for path, text in sorted(review.patches.items())]
    return _REVIEW.format(
        workload=" ".join(review.workload.split()),
        verdict=judgement.verdict.value,
        gain_pct=judgement.gain_pct,
        threshold_pct=judgement.threshold_pct,
        parent=repro.parent,
        candidate=repro.candidate,
        root=_ROOT.format(root=repro.root) if repro.root else "",
        diagnosis_file=DIAGNOSIS_FILE,
        diagnosis=review.case.diagnosis.strip(),
        case_script=CASE_SCRIPT,
        repro_sh=review.case.repro_sh.strip(),
        delta=json.dumps(review.delta.to_document(), sort_keys=True),
        patches="\n\n".join(patches) or _NO_PATCH,
    )


def _verdict(reply: str) -> CriticVerdict:
    try:
        document = json.loads(reply)
    except ValueError as exc:
        raise CriticAnswerError(f"the answer is not JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {_REACHABLE, _REASON}:
        raise CriticAnswerError(f"the answer is not an object of exactly {_REACHABLE!r} and {_REASON!r}")
    reachable, reason = document[_REACHABLE], document[_REASON]
    if not isinstance(reachable, bool) or not isinstance(reason, str) or not reason.strip():
        raise CriticAnswerError(f"{_REACHABLE!r} must be true or false and {_REASON!r} a non-empty text")
    return CriticVerdict(reachable, reason.strip())


def anthropic_critic(model: str, *, component: str) -> CriticCompletion:
    """A :data:`CriticCompletion` over ``llm_config``'s Anthropic transport, tagged with ``component``, each reply
    priced at ``model``'s rates, or unpriced when the price sheet does not list it."""

    price = price_of(model)

    async def complete(system: str, user: str) -> CriticReply:
        # Imported when called: what the multi-branch command imports must not load the LLM stack with it.
        import httpx
        from claude_agent_sdk import ClaudeSDKError

        from hyperloom.common.llm_config import AnthropicCompletionRequest, aanthropic_completion

        # The HTTP transport raises RuntimeError on a bad status or body (LLMConfigError is one) and httpx errors on
        # the wire; the SDK transport raises the SDK's own errors and TimeoutError, which before Python 3.11 is not
        # asyncio's.
        transport_errors = (RuntimeError, OSError, TimeoutError, asyncio.TimeoutError, httpx.HTTPError, ClaudeSDKError)
        try:
            request = AnthropicCompletionRequest(
                model=model,
                messages=[{"role": "user", "content": user}],
                max_tokens=CRITIC_MAX_TOKENS,
                system=system,
                timeout_s=CRITIC_TIMEOUT_SECONDS,
            )
            result = await aanthropic_completion(request=request, component=component, operation="fix_critic")
        except transport_errors as exc:
            raise CriticUnavailable(f"{type(exc).__name__}: {exc}") from exc
        tokens = TokenCounts.from_anthropic_usage(result.usage if isinstance(result.usage, Mapping) else {})
        return CriticReply(result.text, tokens, None if price is None else price.cost_usd(tokens))

    return complete


__all__ = [
    "CRITIC_ATTEMPTS",
    "CRITIC_MAX_TOKENS",
    "CRITIC_TIMEOUT_SECONDS",
    "CriticAnswerError",
    "CriticCompletion",
    "CriticReply",
    "CriticUnavailable",
    "CriticVerdict",
    "FixReview",
    "FixReviewer",
    "LlmCritic",
    "NoCritic",
    "ReproVerdicts",
    "anthropic_critic",
]
