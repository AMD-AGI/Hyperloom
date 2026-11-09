# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The context length a test's accuracy evaluation needs, read from the point the test runs.

The evaluation is lm_eval, started by InferenceX's ``run_lm_eval`` against the server the benchmark booted. It is told
the context it runs in as ``max_length``, which InferenceX takes from the point's ``MAX_MODEL_LEN``, and it sizes the
completion from that window (``max_tokens`` is ``max_length - 4096``, bounded again by the evaluation shim). It does not
check a prompt and its completion against the server's own limit (``tokenized_requests=False``), so a recipe whose launch
serves a shorter context (``--context-length`` on SGLang, ``--max-model-len`` on vLLM) is answered ``400`` for every
request, and the test ends with no accuracy score. ``max_length`` is the least context a request the evaluation may send
fits in, so a test that carries the evaluation is refused when its launch serves less.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from hyperloom.common.outside_facts import CONTEXT_LENGTH_FLAGS, ENV_MAX_MODEL_LEN
from hyperloom.contracts.recipe import RecipeError
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

#: The point variable InferenceX derives the evaluation's ``max_length`` from.
EVAL_CONTEXT_ENV = ENV_MAX_MODEL_LEN
#: The server flags that set the context length a server serves, by the framework that takes them.
CONTEXT_FLAGS = CONTEXT_LENGTH_FLAGS


class EvalContextError(RecipeError):
    """A recipe's launch serves a context shorter than the one its test's accuracy evaluation runs in."""


class ServedContext(NamedTuple):
    """The context length a launch sets, and the flag that sets it."""

    flag: str
    tokens: int


def eval_context(envs: Mapping[str, Any]) -> int | None:
    """The context length the evaluation of a point runs in, or ``None`` when the point does not state one."""

    text = str(envs.get(EVAL_CONTEXT_ENV, "")).strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def served_context(server_args: str) -> ServedContext | None:
    """The context flag the server arguments set, the last one winning as it does when the server parses its
    arguments; ``None`` when they set none."""

    parsed = tokenize_server_args_preserving_json(server_args)
    tokens = parsed[1] if parsed is not None else []
    found = None
    for index, token in enumerate(tokens):
        flag, equals, inline = token.partition("=")
        if flag not in CONTEXT_FLAGS:
            continue
        value = inline if equals else (tokens[index + 1] if index + 1 < len(tokens) else "")
        try:
            found = ServedContext(flag, int(value))
        except ValueError:
            found = None
    return found


def check_eval_context(envs: Mapping[str, Any], server_args: str) -> None:
    """Refuse a point whose launch serves less context than its evaluation runs in, naming the least it may serve.

    A launch that sets no context flag is left to the server's own default and the benchmark script's, which the
    evaluation already follows.
    """

    needed, served = eval_context(envs), served_context(server_args)
    if needed is None or served is None or served.tokens >= needed:
        return
    flag, tokens = served
    raise EvalContextError(
        f"the launch serves a context of {tokens} tokens ({flag} {tokens}), and this test's accuracy evaluation runs in "
        f"{needed} ({EVAL_CONTEXT_ENV}, the max_length it gives lm_eval and sizes its completion from), so the server "
        f"would answer every evaluation request with 400 and the gate could not validate the candidate. Serve at least "
        f"{needed} tokens: raise {flag} to {needed} or more, or leave it out"
    )
