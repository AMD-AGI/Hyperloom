# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Greedy completions of a text prompt over the server's OpenAI-compatible HTTP API, for the long-context check.

The prompt is sent as it is (``/v1/completions``, no chat template) with temperature 0. The request, its time bound and its
refusals are those of ``measure.short_prompts.client``; this module reads what a long prompt needs from the answer, the
number of prompt tokens the server counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom.measure.short_prompts.client import CompletionError, post_json

__all__ = ["Answer", "complete_text"]


@dataclass(frozen=True)
class Answer:
    """One answer: the text, why the generation ended and the prompt tokens the server counted (``None`` when it did not
    say)."""

    text: str
    finish_reason: str
    prompt_tokens: int | None


def _count(usage: object) -> int | None:
    value = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def complete_text(base_url: str, model: str, text: str, *, max_new_tokens: int, timeout_sec: float) -> Answer:
    """The greedy completion of one prompt.

    Raises:
        CompletionError: The request failed, timed out or the answer holds no completion.
    """
    body = {"model": model, "prompt": text, "max_tokens": max_new_tokens, "temperature": 0, "top_p": 1, "stream": False}
    document = post_json(f"{base_url}/v1/completions", body, timeout_sec)
    try:
        choice = document["choices"][0]  # type: ignore[index]
        answer, reason = choice["text"], str(choice.get("finish_reason") or "")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise CompletionError("the server's answer holds no completion") from None
    if not isinstance(answer, str):
        raise CompletionError("the completion's text is not a string")
    return Answer(answer, reason, _count(document.get("usage")))
