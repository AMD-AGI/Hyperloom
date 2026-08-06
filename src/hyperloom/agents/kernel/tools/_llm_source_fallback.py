###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Last-resort semantic pick of a kernel's source file, from a fixed shortlist.

This is the final tier of source resolution, behind the curated dictionary, the
trace-derived launcher, and the name grep. It exists for models or frameworks
where all three come up empty -- not for any case seen so far.

Two properties keep it from becoming another source of nondeterminism, which
matters because an unresolved-launcher sentinel produced by an LLM is what broke
this pipeline in the first place:

* **Selection, never generation.** The model receives a shortlist gathered by a
  relaxed grep and may only return one of those exact strings. A path it invents
  is rejected outright.
* **Off unless asked.** Gated behind ``HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK``, so
  a normal run's resolution stays fully deterministic.

Every accepted answer is stamped ``source_resolution_method="llm_fallback"`` so
it can be audited apart from deterministic resolutions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

# An answer below this confidence is discarded: a coin-flip pick would send a
# backend at the wrong file and burn a whole optimization attempt.
_MIN_CONFIDENCE = 0.7

# Head of each shortlisted file shown to the model; enough to tell a kernel
# definition from a test or a dispatch shim.
_PREVIEW_LINES = 40
_PREVIEW_CHARS = 2000

_DEFAULT_TIMEOUT_SEC = 60.0

_ENV_FLAG = "HYPERLOOM_KERNEL_SOURCE_LLM_FALLBACK"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You identify which source file implements a GPU kernel. "
    "You are given the kernel symbol and a shortlist of candidate files. "
    "Choose the file that DEFINES the kernel body. Reject files that merely "
    "call it, test it, or dispatch to it, and reject a CPU implementation when "
    "the kernel runs on GPU. "
    'Answer with JSON only: {"source_file": "<one of the candidates, verbatim>", '
    '"confidence": <0..1>, "reason": "<one sentence>"}. '
    'If none of the candidates defines the kernel, return "source_file": "".'
)


def llm_fallback_enabled() -> bool:
    """Whether the operator opted into the LLM tier."""
    return str(os.environ.get(_ENV_FLAG, "")).strip().lower() in _TRUTHY


def _preview(path: str) -> str:
    """First lines of ``path``, for telling an implementation from a shim."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = "".join(next(fh, "") for _ in range(_PREVIEW_LINES))
    except OSError:
        return "<unreadable>"
    return head[:_PREVIEW_CHARS]


def _build_prompt(kernel_name: str, candidates: list[str]) -> str:
    """Render the shortlist with a preview of each file."""
    parts = [f"Kernel symbol: {kernel_name}", "", "Candidates:"]
    for index, path in enumerate(candidates, 1):
        parts.append(f"\n[{index}] {path}\n```\n{_preview(path)}\n```")
    return "\n".join(parts)


def _parse_answer(text: str) -> tuple[bool, str, float, str]:
    """Extract ``(parsed, source_file, confidence, reason)`` from a model reply.

    ``parsed`` separates "the reply was unreadable" from "the model answered that
    no candidate fits". Collapsing the two would report a malformed reply as a
    considered verdict and send triage the wrong way.
    """
    match = _JSON_BLOCK_RE.search(text or "")
    if not match:
        return False, "", 0.0, "no JSON object in reply"
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError) as exc:
        return False, "", 0.0, f"unparseable JSON: {exc}"
    if not isinstance(payload, dict):
        return False, "", 0.0, "JSON payload is not an object"
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return (
        True,
        str(payload.get("source_file") or "").strip(),
        confidence,
        str(payload.get("reason") or "").strip(),
    )


def _validate(
    picked: str,
    candidates: list[str],
    framework_roots: tuple[str, ...],
) -> tuple[bool, str]:
    """Check the pick against the shortlist, the filesystem and the roots."""
    if not picked:
        return False, "model reported no candidate defines the kernel"
    if picked not in candidates:
        # The whole point of a shortlist is that the answer comes from it.
        return False, f"path is not one of the candidates: {picked!r}"
    if not os.path.isfile(picked):
        return False, f"path does not exist: {picked!r}"
    if framework_roots:
        low = picked.lower()
        if not any(root.lower() in low for root in framework_roots if root):
            return False, f"path is outside every framework root: {picked!r}"
    return True, ""


def _complete(prompt: str, model: str, timeout_sec: float) -> str:
    """One chat completion against the configured OpenAI-compatible endpoint."""
    from openai import OpenAI  # noqa: PLC0415 - optional dependency

    from hyperloom.common.llm_config import openai_client_kwargs  # noqa: PLC0415

    client = OpenAI(**openai_client_kwargs())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        timeout=timeout_sec,
    )
    return str(response.choices[0].message.content or "")


def select_source_via_llm(
    kernel_name: str,
    candidates: list[str],
    *,
    framework_roots: tuple[str, ...] = (),
    model: str = "",
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    log: Callable[[str], None] | None = None,
    complete: Callable[[str, str, float], str] | None = None,
) -> tuple[str, float, str]:
    """Pick the file that defines ``kernel_name`` from ``candidates``.

    Args:
        kernel_name: The kernel symbol being resolved.
        candidates: Shortlist of on-disk paths from the relaxed grep. An empty
            list short-circuits: with nothing to choose from there is nothing to
            ask, and inventing a path is not allowed.
        framework_roots: Accepted path roots; a pick outside them is rejected.
        model: Chat model; defaults to ``$CLAUDE_MODEL``.
        timeout_sec: Per-call ceiling. There is no retry -- a failure here is
            advisory and the candidate simply stays unresolved.
        log: Optional ``callable(str)`` for diagnostics.
        complete: Injection point for the completion call (tests).

    Returns:
        ``(source_file, confidence, reason)``; ``source_file`` is ``""`` on any
        failure, including a low-confidence answer.
    """
    def _say(message: str) -> None:
        if callable(log):
            log(f"llm_source_fallback: {message}")

    if not llm_fallback_enabled():
        return "", 0.0, "disabled"
    if not kernel_name or not candidates:
        return "", 0.0, "no candidates to choose from"

    shortlist = [str(c) for c in candidates if str(c).strip()]
    caller = complete or _complete
    chosen_model = model or os.environ.get("CLAUDE_MODEL") or "claude-opus-4-6"
    try:
        reply = caller(_build_prompt(kernel_name, shortlist), chosen_model, timeout_sec)
    except Exception as exc:  # noqa: BLE001 - advisory tier, never fatal
        _say(f"call failed: {exc!r}")
        return "", 0.0, f"llm call failed: {exc!r}"

    parsed, picked, confidence, reason = _parse_answer(reply)
    if not parsed:
        _say(f"rejected: {reason}")
        return "", 0.0, reason
    ok, why = _validate(picked, shortlist, framework_roots)
    if not ok:
        _say(f"rejected: {why}")
        return "", confidence, why
    if confidence < _MIN_CONFIDENCE:
        _say(f"rejected: confidence {confidence:.2f} < {_MIN_CONFIDENCE}")
        return "", confidence, f"confidence {confidence:.2f} below {_MIN_CONFIDENCE}"

    _say(f"accepted {picked} (confidence={confidence:.2f})")
    return picked, confidence, reason


__all__ = ["llm_fallback_enabled", "select_source_via_llm"]
