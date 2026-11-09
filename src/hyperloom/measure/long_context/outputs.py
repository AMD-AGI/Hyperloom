# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The file the check leaves in the signed bundle: what the server answered, prompt by prompt.

``<output_dir>/long_context/outputs.json``::

    {"schema": 1, "prompt_set": KEY, "prompts_sha256": SHA, "length": L, "max_new_tokens": N, "concurrency": C,
     "needed_context": L+128, "served_context": N or null, "stopped": "", "boot_seconds": S, "query_seconds": S,
     "prompts": [{"id": ID, "task": T, "references": [..], "tokens": N, "status": "ok"|"error"|"not_run", "reason": "",
                  "text": "", "finish_reason": "", "prompt_tokens": N or null, "seconds": S}, ...]}

Every prompt of the set is listed, answered or not, with the references its answer is graded against, so the file alone is
enough to grade the check again. ``stopped`` says why the whole check ended early (a launch that serves too little context,
a server that did not come up, the phase cap). The toolbox reads the file from the landed bundle, so every field is read
strictly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.contracts.long_context import PROMPT_OK, PROMPT_STATUSES
from hyperloom.measure.docs import DocumentError, parse_json_object
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE, LongContextPlan

__all__ = [
    "OutputsError",
    "OutputsFile",
    "PromptEntry",
    "RunFacts",
    "clean_reason",
    "parse_outputs",
    "read_outputs",
    "write_outputs",
]

_SCHEMA = 1
_UTF8 = "utf-8"
_REASON_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_MAX_BYTES = 8 << 20
_TOP_KEYS = frozenset(
    (
        "schema prompt_set prompts_sha256 length max_new_tokens concurrency needed_context served_context stopped "
        "boot_seconds query_seconds prompts"
    ).split()
)
_PROMPT_KEYS = "id task references tokens status reason text finish_reason prompt_tokens seconds".split()


class OutputsError(Exception):
    """The bundle holds no usable outputs file; the message says what is wrong."""


@dataclass(frozen=True)
class PromptEntry:
    """One prompt of the set and what became of it; ``text`` is empty unless ``status`` is ``ok``."""

    id: str
    task: str
    references: tuple[str, ...]
    tokens: int
    status: str
    reason: str = ""
    text: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    seconds: float = 0.0

    def to_document(self) -> dict[str, Any]:
        document = {key: getattr(self, key) for key in _PROMPT_KEYS}
        return {**document, "references": list(self.references), "reason": clean_reason(self.reason)}


@dataclass(frozen=True)
class OutputsFile:
    prompt_set: str
    prompts_sha256: str
    length: int
    max_new_tokens: int
    concurrency: int
    needed_context: int
    served_context: int | None
    stopped: str
    boot_seconds: float
    query_seconds: float
    prompts: tuple[PromptEntry, ...]


def clean_reason(text: str) -> str:
    """``text`` on one bounded line, safe to put in a result."""
    return _CONTROL.sub(" ", text).strip()[:_REASON_CHARS]


@dataclass(frozen=True)
class RunFacts:
    """What the pass says of itself rather than of a prompt: the context the launch serves (when it states one), why the
    whole check ended early (or ``""``) and the seconds the server took to come up and to answer."""

    served_context: int | None
    stopped: str
    boot_seconds: float
    query_seconds: float


def write_outputs(output_dir: Path, plan: LongContextPlan, entries: Sequence[PromptEntry], facts: RunFacts) -> Path:
    """Write the file below ``output_dir``; the signed manifest then lists it."""
    document = {
        "schema": _SCHEMA,
        "prompt_set": plan.prompt_set,
        "prompts_sha256": plan.prompts_sha256,
        "length": plan.length,
        "max_new_tokens": plan.max_new_tokens,
        "concurrency": plan.concurrency,
        "needed_context": plan.needed_context,
        "served_context": facts.served_context,
        "stopped": clean_reason(facts.stopped),
        "boot_seconds": round(facts.boot_seconds, 3),
        "query_seconds": round(facts.query_seconds, 3),
        "prompts": [entry.to_document() for entry in entries],
    }
    directory = Path(output_dir) / LONG_CONTEXT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUTPUTS_FILE
    path.write_text(json.dumps(document, sort_keys=True), encoding=_UTF8)
    return path


def _is_int(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_seconds(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value < 1e9


def _prompt(value: Any, path: str) -> PromptEntry:
    if not isinstance(value, dict) or set(value) != set(_PROMPT_KEYS):
        raise OutputsError(f"{path}: expected the keys {sorted(_PROMPT_KEYS)}")
    refs, status, reached = value["references"], value["status"], value["prompt_tokens"]
    texts = (value["id"], value["task"], value["reason"], value["text"], value["finish_reason"])
    if not all(isinstance(t, str) for t in texts) or not value["id"]:
        raise OutputsError(f"{path}: id, task, reason, text and finish_reason are strings, the id not empty")
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) and r for r in refs):
        raise OutputsError(f"{path}: references must be a non-empty list of non-empty strings")
    if status not in PROMPT_STATUSES or not _is_int(value["tokens"], minimum=1) or not _is_seconds(value["seconds"]):
        raise OutputsError(f"{path}: a known status, a positive tokens and a non-negative seconds")
    if reached is not None and not _is_int(reached):
        raise OutputsError(f"{path}: prompt_tokens is null or a non-negative integer")
    if status != PROMPT_OK and value["text"]:
        raise OutputsError(f"{path}: a prompt that was not answered holds no text")
    return PromptEntry(
        texts[0], texts[1], tuple(refs), value["tokens"], status, *texts[2:], reached, float(value["seconds"])
    )


def parse_outputs(document: Mapping[str, Any]) -> OutputsFile:
    if set(document) != _TOP_KEYS or document["schema"] != _SCHEMA or not isinstance(document["prompts"], list):
        raise OutputsError(
            f"the outputs file holds the keys {sorted(document)}, not {sorted(_TOP_KEYS)} of schema {_SCHEMA}"
        )
    ints = [document[k] for k in ("length", "max_new_tokens", "concurrency", "needed_context")]
    texts = [document[k] for k in ("prompt_set", "prompts_sha256", "stopped")]
    served, spent = document["served_context"], [document["boot_seconds"], document["query_seconds"]]
    if not all(_is_int(i, minimum=1) for i in ints) or not all(isinstance(t, str) for t in texts):
        raise OutputsError("the outputs file's counts are positive integers and its names strings")
    if (served is not None and not _is_int(served)) or not all(_is_seconds(s) for s in spent):
        raise OutputsError("served_context is null or an integer, the seconds non-negative numbers")
    prompts = tuple(_prompt(item, f"prompts[{i}]") for i, item in enumerate(document["prompts"]))
    if not prompts or len({p.id for p in prompts}) != len(prompts):
        raise OutputsError("the outputs file lists no prompt, or a prompt id twice")
    return OutputsFile(*texts[:2], *ints, served, texts[2], float(spent[0]), float(spent[1]), prompts)


def read_outputs(directory: Path) -> OutputsFile:
    """The file a landed bundle holds below ``directory`` (the test's output directory).

    Raises:
        OutputsError: There is no such file, or it is not a well-formed outputs file.
    """
    path = Path(directory) / LONG_CONTEXT_DIR / OUTPUTS_FILE
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_BYTES:
            raise OutputsError(f"{LONG_CONTEXT_DIR}/{OUTPUTS_FILE} is larger than {_MAX_BYTES} bytes")
        document = parse_json_object(raw, what=f"{LONG_CONTEXT_DIR}/{OUTPUTS_FILE}")
    except (OSError, DocumentError) as exc:
        raise OutputsError(f"the bundle holds no readable {LONG_CONTEXT_DIR}/{OUTPUTS_FILE}: {exc}") from None
    return parse_outputs(document)
