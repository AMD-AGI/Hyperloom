# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Greedy completions over the server's OpenAI-compatible HTTP API, for the short-prompt check.

A prompt is sent as token ids (``/v1/completions`` takes a list of ints as ``prompt``) with temperature 0, so two
launches that compute the same thing return the same tokens. ``logprobs=1`` makes the server return the generated tokens
as strings beside the text; the check compares those when every output has them and the text otherwise. ``sequential``
prompts are sent one at a time, ``batched`` prompts all at once. Nothing here uses the proxy environment: the server is
on this machine.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from hyperloom.measure.short_prompts.prompts import BATCHED, SEQUENTIAL, Prompt

__all__ = ["Completion", "CompletionError", "complete", "discover_model", "post_json", "request_timeout", "run_prompts"]

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_JSON = {"Content-Type": "application/json"}
_UTF8 = "utf-8"
_MAX_REPLY_BYTES = 1 << 20
#: The longest one request may take, however much of the deadline is left.
REQUEST_TIMEOUT_SEC = 180.0
_PHASE_CAP = "the phase cap ended this process's questions"
_T = TypeVar("_T")


class CompletionError(Exception):
    """A request the server did not answer with a completion; the message is safe to put in a result."""


@dataclass(frozen=True)
class Completion:
    """One answer: the text, the generated tokens as the server printed them (``None`` when it returned none) and why the
    generation ended."""

    text: str
    tokens: tuple[str, ...] | None
    finish_reason: str


def request_timeout(deadline: float) -> float:
    """How long the next request may take: what is left until ``deadline`` (a ``time.monotonic()`` instant), at most
    :data:`REQUEST_TIMEOUT_SEC`.

    Raises:
        CompletionError: The deadline has passed.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CompletionError(_PHASE_CAP)
    return min(REQUEST_TIMEOUT_SEC, remaining)


def _bounded(timeout_sec: float, call: Callable[[], _T]) -> _T:
    """``call()``, which must finish within ``timeout_sec`` in all: a socket timeout only limits how long a read waits for
    the next bytes, so a server that trickles its answer would otherwise hold the caller past any deadline. A call that
    outlives it is left to its own thread (a daemon, which ends when the server is stopped or its socket times out).

    Raises:
        CompletionError: The call did not finish in time. Whatever the call raises is raised as it was.
    """
    outcome: list[tuple[bool, object]] = []

    def work() -> None:
        try:
            outcome.append((True, call()))
        except BaseException as exc:  # noqa: BLE001 - raised again, below, in the caller's thread
            outcome.append((False, exc))

    thread = threading.Thread(target=work, name="hl-short-prompt-request", daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if not outcome:
        raise CompletionError("the request did not finish within its time")
    succeeded, value = outcome[0]
    if succeeded:
        return value  # type: ignore[return-value]
    raise value  # type: ignore[misc]


def _read_json(source: urllib.request.Request | str, timeout_sec: float) -> object:
    with _OPENER.open(source, timeout=timeout_sec) as reply:
        return json.loads(reply.read(_MAX_REPLY_BYTES).decode(_UTF8))


def post_json(url: str, body: dict[str, object], timeout_sec: float) -> dict[str, object]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(_UTF8), headers=_JSON, method="POST")
    try:
        document = _bounded(timeout_sec, lambda: _read_json(request, timeout_sec))
    except urllib.error.HTTPError as exc:
        raise CompletionError(f"the server answered HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CompletionError(f"the request failed: {type(exc).__name__}") from None
    if not isinstance(document, dict):
        raise CompletionError("the server's answer is not a JSON object")
    return document


def discover_model(base_url: str, timeout_sec: float) -> str:
    """The id of the model the server serves, from ``/v1/models``: the name a request must carry on a server that checks it."""
    try:
        document = _bounded(timeout_sec, lambda: _read_json(f"{base_url}/v1/models", timeout_sec))
        return str(document["data"][0]["id"])  # type: ignore[index]
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise CompletionError(f"the server does not list its model: {type(exc).__name__}") from None


def _completion(document: dict[str, object]) -> Completion:
    try:
        choice = document["choices"][0]  # type: ignore[index]
        text = choice["text"]
        reason = str(choice.get("finish_reason") or "")
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise CompletionError("the server's answer holds no completion") from None
    if not isinstance(text, str):
        raise CompletionError("the completion's text is not a string")
    listed = tuple(tokens) if isinstance(tokens, list) and all(isinstance(t, str) for t in tokens) else None
    return Completion(text, listed, reason)


def complete(base_url: str, model: str, prompt: Prompt, *, max_new_tokens: int, timeout_sec: float) -> Completion:
    """The greedy completion of one prompt."""
    body = {
        "model": model,
        "prompt": list(prompt.token_ids),
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1,
        "logprobs": 1,
        "stream": False,
    }
    return _completion(post_json(f"{base_url}/v1/completions", body, timeout_sec))


def run_prompts(
    base_url: str, model: str, prompts: Sequence[Prompt], *, max_new_tokens: int, deadline: float
) -> dict[str, Completion]:
    """Every prompt's completion, by prompt id: the sequential ones in order, then the batched ones all at once, each
    request allowed what is left until ``deadline`` (a ``time.monotonic()`` instant) and at most ``REQUEST_TIMEOUT_SEC``.

    Raises:
        CompletionError: A request failed or the deadline passed; the outputs of the others are of no use to a check that
            needs all of them.
    """

    def one(prompt: Prompt) -> Completion:
        try:
            return complete(
                base_url, model, prompt, max_new_tokens=max_new_tokens, timeout_sec=request_timeout(deadline)
            )
        except CompletionError:
            if time.monotonic() >= deadline:
                raise CompletionError(
                    _PHASE_CAP
                ) from None  # a request cut off by the deadline is the cap's, not the server's
            raise

    answers: dict[str, Completion] = {}
    for prompt in (p for p in prompts if p.mode == SEQUENTIAL):
        answers[prompt.id] = one(prompt)
    batch = [p for p in prompts if p.mode == BATCHED]
    if batch:
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            for prompt, answer in zip(batch, pool.map(one, batch)):
                answers[prompt.id] = answer
    return answers
