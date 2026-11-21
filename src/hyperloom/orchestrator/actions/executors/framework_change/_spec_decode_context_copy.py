# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Refuse a context-copying speculative-decoding draft as a candidate on a synthetic benchmark.

The synthetic benchmark sends random-token prompts and holds every request to its full output length (``ignore_eos``).
The model is driven into repetition loops there, and a draft that proposes tokens by copying them from the context
accepts almost every one of them. One 12-hour vLLM session measured an ``ngram`` draft at 6.7-7.9 tokens per step on the
benchmark against 2.3-2.7 on the GSM8K eval the same server ran, and kept it at +95.7% when the gain that holds on real
text was about +40%.

Acceptance length differs from one dataset to another by design, so there is no principled threshold to hold a
benchmark's acceptance to. What is specific to the synthetic workload is that it rewards copying, so a candidate that
introduces or changes a context-copying draft is refused before it runs:

* vLLM ``--speculative-config`` ``method`` ``ngram``, ``ngram_gpu`` (prompt lookup) or ``suffix`` (suffix decoding), or a
  ``model`` of ``ngram`` / ``[ngram]`` with no method, which vLLM resolves to ``ngram``; the JSON object and its dotted
  ``--speculative-config.<key>`` flags alike;
* SGLang ``--speculative-algorithm NGRAM``.

Model-based drafts (MTP, EAGLE, a draft model, DFlash/DSpark) are not affected. Neither is a base that already copies:
the operator chose it, and a candidate that leaves its speculative settings as they are is graded against a base that
carries the same draft. AgentX sessions benchmark real datasets and are out of scope.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from typing import Any


__all__ = [
    "CONTEXT_COPY_REFUSAL",
    "context_copy_method",
    "context_copy_refusal",
]

#: Reason a refused candidate is recorded with.
CONTEXT_COPY_REFUSAL = "speculative_context_copy_on_synthetic"

#: vLLM's prompt-lookup method.
_NGRAM = "ngram"
#: vLLM ``SpeculativeConfig.method`` values that draft by copying from the context.
_VLLM_COPY_METHODS = frozenset({_NGRAM, "ngram_gpu", "suffix"})
#: vLLM ``SpeculativeConfig.model`` values it resolves to ``ngram`` when no method is given.
_VLLM_NGRAM_MODELS = frozenset({_NGRAM, f"[{_NGRAM}]"})
#: SGLang ``--speculative-algorithm`` values that draft by copying from the context (case-insensitive).
_SGLANG_COPY_ALGORITHMS = frozenset({"NGRAM"})

_VLLM_CONFIG_FLAG = "--speculative-config"
_SGLANG_ALGORITHM_FLAG = "--speculative-algorithm"
#: A ``key: value`` pair inside a ``--speculative-config`` value that is not valid JSON (a shell split strips the quotes).
_LOOSE_PAIR = re.compile(r"""(?<![\w])["']?(method|model)["']?\s*:\s*["']?([\w\[\]-]+)""")

#: A flag token, and the flags and env names that configure speculative decoding: vLLM ``--speculative-config`` (and
#: its dotted keys), SGLang ``--speculative-*``, ATOM ``--num-speculative-tokens`` / ``--spec-decode-*``, and any engine
#: env about speculation.
_FLAG = re.compile(r"--[A-Za-z0-9][\w.-]*")
_SPEC_FLAG = re.compile(r"^--(?:[\w.-]*speculative|spec-decode)", re.IGNORECASE)
_SPEC_ENV = re.compile(r"^(?:SGLANG|VLLM|TLLM|TRTLLM|ATOM)_\w*(?:SPEC|MTP|EAGLE|DRAFT|SIMULATE_ACC)", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _flag_values(tokens: list[str]) -> list[tuple[str, str]]:
    """``(flag, value)`` for every ``--flag value`` / ``--flag=value`` in order; a bare flag has an empty value."""
    pairs: list[tuple[str, str]] = []
    for i, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name, separator, attached = token.partition("=")
        if separator:
            pairs.append((name, attached))
        else:
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            pairs.append((name, "" if following.startswith("--") else following))
    return pairs


def _config_keys(value: str) -> dict[str, str]:
    """``method`` and ``model`` of one ``--speculative-config`` value; the value replaces the whole object."""
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = None
    if isinstance(parsed, Mapping):
        return {key: str(parsed[key]) for key in ("method", "model") if parsed.get(key) is not None}
    return dict(_LOOSE_PAIR.findall(value))


def _method_in(text: str) -> str:
    """The context-copying method one launch argv configures, the last setting of each flag winning like the engine."""
    vllm: dict[str, str] = {}
    sglang = ""
    for flag, value in _flag_values(_tokens(text)):
        name = flag.replace("_", "-").lower()
        if name == _VLLM_CONFIG_FLAG:
            vllm = _config_keys(value)
        elif name.startswith(f"{_VLLM_CONFIG_FLAG}."):
            key = name[len(_VLLM_CONFIG_FLAG) + 1 :]
            if key in ("method", "model"):
                vllm[key] = value
        elif name == _SGLANG_ALGORITHM_FLAG:
            sglang = value
    method = vllm.get("method", "").strip().lower()
    if method in _VLLM_COPY_METHODS:
        return method
    if not method and vllm.get("model", "").strip().lower() in _VLLM_NGRAM_MODELS:
        return _NGRAM
    return sglang.strip().upper() if sglang.strip().upper() in _SGLANG_COPY_ALGORITHMS else ""


def context_copy_method(argv: str, envs: Mapping[str, Any]) -> str:
    """The context-copying speculative-decoding method a launch configures, or ``""`` when it configures none.

    Read from the server argv and from every env value, since a server-args env (``EXTRA_VLLM_ARGS``) carries argv too.
    """
    for text in (argv, *(str(value) for value in envs.values())):
        method = _method_in(str(text or ""))
        if method:
            return method
    return ""


def _speculative_settings(argv: str, envs: Mapping[str, Any]) -> frozenset[str]:
    """Every speculative-decoding setting in a launch: each such flag with its value, and each such env with its value."""
    found: set[str] = set()
    for text in (argv, *(str(value) for value in envs.values())):
        tokens = str(text or "").split()
        for i, token in enumerate(tokens):
            if _FLAG.fullmatch(token.split("=", 1)[0]) and _SPEC_FLAG.search(token):
                following = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else ""
                found.add(f"{token} {following}".strip())
    found.update(f"{name}={value}" for name, value in envs.items() if _SPEC_ENV.search(str(name)))
    return frozenset(found)


def context_copy_refusal(base_argv: str, base_envs: Mapping[str, Any], argv: str, envs: Mapping[str, Any]) -> str:
    """Why a candidate's launch is refused on a synthetic benchmark, or ``""`` when it is not.

    Refused when the candidate launches a context-copying draft and its speculative settings differ from its base's:
    it introduces such a draft, switches to one, or tunes one the base already runs. A candidate that leaves a copying
    base's speculative settings alone, or that drops the draft, is not refused. Both launches are the materialized ones.
    """
    method = context_copy_method(argv, envs)
    if not method or _speculative_settings(base_argv, base_envs) == _speculative_settings(argv, envs):
        return ""
    return (
        f"speculative decoding method {method!r} drafts by copying from the context; on the synthetic benchmark "
        "(random prompts held to their full output length) the model falls into repetition loops that such a draft "
        "accepts almost entirely, so its throughput does not carry over to real text. Use a model-based draft "
        "(MTP, EAGLE, a draft model) or a real-dataset benchmark."
    )
