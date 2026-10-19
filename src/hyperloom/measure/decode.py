# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The greedy decode pass: a fixed prompt set decoded at the graded point, its output token ids kept in the bundle.

GSM8K is short, so the long-prefill and long-KV kernel paths of the graded point are never checked for correctness by
the accuracy round. The pass sends ``PROMPTS`` prompts of random token ids at the graded input length, from a fixed
seed, through the server's OpenAI-compatible ``/v1/completions`` (greedy: ``temperature`` 0, ``top_k`` 1) at the graded
concurrency, and records the token ids each one produced.

The record is ``decode/outputs.jsonl`` below the test's output directory, so the manifest lists it and the signature
covers it. Its first line is the header holding the pass identity: a digest over what decides the prompt set (seed,
prompt count, input length, output cap, token range) plus the sampling settings. Two passes with different identities
decoded different things and must not be compared; the lines after the header are ``{"prompt": i, "tokens": [...]}``.
A pass that fails records nothing: there is no output to guess.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.contracts.measurement import TestResult

__all__ = [
    "DECODE_DIR",
    "OUTPUTS_PATH",
    "DecodePass",
    "DecodePassError",
    "PassPlan",
    "parse_outputs",
    "plan_pass",
    "read_decode_outputs",
    "read_decode_pass",
    "run_pass",
    "vocab_size_of",
    "write_outputs",
]

DECODE_DIR = "decode"
OUTPUTS_PATH = f"{DECODE_DIR}/outputs.jsonl"
#: The fixed pass: distinct from the warmup and measured seeds, so parent and candidate decode the same prompts.
SEED = 20261019
PROMPTS = 32
MAX_TOKENS_CAP = 256
TEMPERATURE = 0
TOP_K = 1
#: The token ids prompts are drawn from: above the special tokens, below the ids every tokenizer has.
_TOKEN_FLOOR = 1000
_TOKEN_CEILING = 50000
_HEADER_KEY = "decode_pass"
_UTF8 = "utf-8"
_CHOICES = "choices"


class DecodePassError(Exception):
    """The pass did not produce outputs; the message is a short reason code the result's warning carries."""


class DecodePass(NamedTuple):
    """What a bundle holds: the pass identity and one tuple of output token ids per prompt, in prompt order."""

    identity: str
    outputs: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PassPlan:
    """The prompt set and sampling of one pass; two plans with the same :attr:`identity` decode the same prompts."""

    isl: int
    max_tokens: int
    token_low: int
    token_high: int
    seed: int = SEED
    prompts: int = PROMPTS

    @property
    def digest(self) -> str:
        decided = {
            "seed": self.seed,
            "prompts": self.prompts,
            "isl": self.isl,
            "max_tokens": self.max_tokens,
            "token_range": [self.token_low, self.token_high],
        }
        return hashlib.sha256(json.dumps(decided, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def identity(self) -> str:
        return f"sha256:{self.digest}|temperature={TEMPERATURE}|top_k={TOP_K}"

    def prompt_ids(self) -> list[list[int]]:
        rng = random.Random(self.seed)
        return [[rng.randrange(self.token_low, self.token_high) for _ in range(self.isl)] for _ in range(self.prompts)]


def plan_pass(isl: int, osl: int, vocab_size: int) -> PassPlan:
    """The plan for a point of input length ``isl`` and output length ``osl`` on a model of ``vocab_size`` tokens."""
    if isl < 1 or osl < 1 or vocab_size < 2:
        raise DecodePassError("point_has_no_isl_osl_or_vocab")
    high = min(vocab_size, _TOKEN_CEILING)
    return PassPlan(
        isl=isl, max_tokens=min(osl, MAX_TOKENS_CAP), token_low=min(_TOKEN_FLOOR, high // 2), token_high=high
    )


def vocab_size_of(model_dir: str | os.PathLike[str]) -> int:
    """``vocab_size`` from the model's ``config.json`` (top level, else its ``text_config``)."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding=_UTF8))
    except (OSError, ValueError):
        raise DecodePassError("model_config_unreadable") from None
    nested = config.get("text_config") if isinstance(config, dict) else None
    value = config.get("vocab_size") if isinstance(config, dict) else None
    value = value if value is not None else (nested or {}).get("vocab_size") if isinstance(nested, dict) else value
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecodePassError("model_config_has_no_vocab_size")
    return value


# -- running ---------------------------------------------------------------------------------------------------------

#: A request never goes through a proxy: the server is on this machine.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _tokens_of(body: Any) -> tuple[int, ...]:
    choices = body.get(_CHOICES) if isinstance(body, dict) else None
    ids = (
        choices[0].get("token_ids") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    )
    if not isinstance(ids, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in ids):
        raise DecodePassError("no_token_ids_in_response")
    return tuple(ids)


def _complete(url: str, payload: bytes, timeout_sec: float) -> tuple[int, ...]:
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout_sec) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise DecodePassError(f"http_{exc.code}") from None
    except (OSError, ValueError) as exc:  # a refused connection, a timeout, a body that is not JSON
        raise DecodePassError(f"request_failed_{type(exc).__name__}") from None
    return _tokens_of(body)


def run_pass(
    plan: PassPlan, *, base_url: str, model: str, concurrency: int, timeout_sec: float
) -> tuple[tuple[int, ...], ...]:
    """Decode every prompt of ``plan`` against ``base_url`` with at most ``concurrency`` requests in flight,
    the whole pass bounded by ``timeout_sec``.

    Raises:
        DecodePassError: A request failed or the response held no token ids; no partial result is returned.
    """
    url = f"{base_url.rstrip('/')}/v1/completions"
    payloads = [
        json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": plan.max_tokens,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "return_token_ids": True,
            }
        ).encode()
        for prompt in plan.prompt_ids()
    ]
    deadline = time.monotonic() + timeout_sec

    def decode(payload: bytes) -> tuple[int, ...]:
        left = deadline - time.monotonic()
        if left <= 0:
            raise DecodePassError("pass_timed_out")
        return _complete(url, payload, left)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(concurrency, plan.prompts))) as pool:
        futures = [pool.submit(decode, payload) for payload in payloads]
        try:
            return tuple(future.result() for future in futures)
        finally:
            for future in futures:
                future.cancel()  # a pass that failed leaves no queued request behind


# -- the record ------------------------------------------------------------------------------------------------------


def write_outputs(output_dir: Path, plan: PassPlan, outputs: tuple[tuple[int, ...], ...]) -> Path:
    """Write the record below ``output_dir`` (header line, then one line per prompt); return the file."""
    header = {
        _HEADER_KEY: {
            "identity": plan.identity,
            "seed": plan.seed,
            "prompts": plan.prompts,
            "isl": plan.isl,
            "max_tokens": plan.max_tokens,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        }
    }
    lines = [json.dumps(header)] + [json.dumps({"prompt": i, "tokens": list(t)}) for i, t in enumerate(outputs)]
    path = Path(output_dir) / OUTPUTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding=_UTF8)
    return path


def _bad(why: str) -> DecodePassError:
    return DecodePassError(f"{OUTPUTS_PATH}: {why}")


def parse_outputs(text: str) -> DecodePass:
    """The identity and outputs a record holds.

    Raises:
        DecodePassError: The header is missing, a line is not ``{"prompt": i, "tokens": [...]}`` in order, or the
            number of lines differs from the header's prompt count.
    """
    lines = text.splitlines()
    try:
        header = json.loads(lines[0])[_HEADER_KEY]
        identity, count = header["identity"], header["prompts"]
        rows = [json.loads(line) for line in lines[1:]]
    except (IndexError, KeyError, TypeError, ValueError):
        raise _bad("no readable header") from None
    if not isinstance(identity, str) or not identity or count != len(rows):
        raise _bad("the header does not describe the lines")
    outputs = []
    for index, row in enumerate(rows):
        tokens = row.get("tokens") if isinstance(row, dict) else None
        if (
            row.get("prompt") != index
            or not isinstance(tokens, list)
            or any(isinstance(t, bool) or not isinstance(t, int) for t in tokens)
        ):
            raise _bad(f"line {index + 1} is not prompt {index} with a list of token ids")
        outputs.append(tuple(tokens))
    return DecodePass(identity, tuple(outputs))


def read_decode_pass(result: TestResult, session_dir: str | os.PathLike[str]) -> DecodePass:
    """The pass a result's bundle holds, read from the landed copy below ``session_dir``.

    Raises:
        DecodePassError: The result names no pass, the file is missing or malformed, or its header's identity is not
            the identity the result records.
    """
    if not result.decode_outputs:
        raise DecodePassError("the result records no decode pass")
    root = Path(session_dir).resolve()
    path = (root / result.output_dir / result.decode_outputs).resolve()
    if root not in path.parents:
        raise _bad("the recorded path leaves the session directory")
    try:
        record = parse_outputs(path.read_text(encoding=_UTF8))
    except OSError:
        raise _bad("cannot be read") from None
    if record.identity != result.decode_identity:
        raise _bad(f"holds identity {record.identity!r}, the result records {result.decode_identity!r}")
    return record


def read_decode_outputs(result: TestResult, session_dir: str | os.PathLike[str]) -> tuple[tuple[int, ...], ...]:
    """The output token ids of each prompt of the result's decode pass (see :func:`read_decode_pass`)."""
    return read_decode_pass(result, session_dir).outputs
