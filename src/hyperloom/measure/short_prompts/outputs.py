# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The file the check leaves in the signed bundle: what each server process answered, prompt by prompt.

``<output_dir>/short_prompts/outputs.json``::

    {"schema": 2, "prompt_set": KEY, "prompts_sha256": SHA, "max_new_tokens": N,
     "prompts": [{"id": ID, "mode": MODE, "length": TOKENS}, ...],
     "processes": [{"index": 0, "status": "ok", "reason": "", "boot_seconds": S, "query_seconds": S,
                    "outputs": {"<prompt id>": {"text": T, "tokens": [..] or null, "finish_reason": R}}}, ...]}

The one process is the test's own server (index 0). A process that did not answer every prompt is ``failed`` with a
``reason`` and holds no outputs. The toolbox reads the file from the landed bundle, so every field is read strictly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure.docs import DocumentError, parse_json_object
from hyperloom.measure.short_prompts.client import Completion
from hyperloom.measure.short_prompts.compare import Output, Outputs, PromptInfo
from hyperloom.measure.short_prompts.plan import OUTPUTS_FILE, SHORT_PROMPTS_DIR, ShortPromptPlan

__all__ = [
    "FAILED",
    "OK",
    "OutputsError",
    "OutputsFile",
    "ProcessEntry",
    "output_document",
    "parse_output",
    "parse_outputs",
    "read_outputs",
    "read_outputs_document",
    "write_outputs",
]

OK = "ok"
FAILED = "failed"
#: Version 2 has no planned process count and no stop reason: the one process is the test's own server.
_SCHEMA = 2
_UTF8 = "utf-8"
_REASON_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_MAX_BYTES = 8 << 20


class OutputsError(Exception):
    """The bundle holds no usable outputs file; the message says what is wrong."""


@dataclass(frozen=True)
class ProcessEntry:
    """One server process: ``outputs`` holds every prompt's answer when ``status`` is ``ok`` and nothing otherwise."""

    index: int
    status: str
    reason: str
    boot_seconds: float
    query_seconds: float
    outputs: Outputs


@dataclass(frozen=True)
class OutputsFile:
    prompt_set: str
    prompts_sha256: str
    max_new_tokens: int
    prompts: tuple[PromptInfo, ...]
    processes: tuple[ProcessEntry, ...]

    @property
    def completed(self) -> dict[int, Outputs]:
        """The outputs of the processes that answered every prompt, by process index."""
        return {entry.index: entry.outputs for entry in self.processes if entry.status == OK}


def clean_reason(text: str) -> str:
    """``text`` on one bounded line, safe to put in a result."""
    return _CONTROL.sub(" ", text).strip()[:_REASON_CHARS]


def _entry_document(entry: ProcessEntry) -> dict[str, Any]:
    return {
        "index": entry.index,
        "status": entry.status,
        "reason": clean_reason(entry.reason),
        "boot_seconds": round(entry.boot_seconds, 3),
        "query_seconds": round(entry.query_seconds, 3),
        "outputs": {prompt_id: output_document(out) for prompt_id, out in entry.outputs.items()},
    }


def output_document(output: Output) -> dict[str, Any]:
    """One prompt's output as the file holds it."""
    return {
        "text": output.text,
        "tokens": None if output.tokens is None else list(output.tokens),
        "finish_reason": output.finish_reason,
    }


def completions_to_outputs(answers: Mapping[str, Completion]) -> dict[str, Output]:
    return {pid: Output(answer.text, answer.tokens, answer.finish_reason) for pid, answer in answers.items()}


def write_outputs(
    output_dir: Path,
    plan: ShortPromptPlan,
    prompts: Sequence[PromptInfo],
    entries: Sequence[ProcessEntry],
) -> Path:
    """Write the file below ``output_dir``; the signed manifest then lists it."""
    document = {
        "schema": _SCHEMA,
        "prompt_set": plan.prompt_set,
        "prompts_sha256": plan.prompts_sha256,
        "max_new_tokens": plan.max_new_tokens,
        "prompts": [{"id": p.id, "mode": p.mode, "length": p.length} for p in prompts],
        "processes": [_entry_document(entry) for entry in entries],
    }
    directory = Path(output_dir) / SHORT_PROMPTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / OUTPUTS_FILE
    path.write_text(json.dumps(document, sort_keys=True), encoding=_UTF8)
    return path


def parse_output(value: Any, path: str) -> Output:
    """One prompt's output read strictly from what :func:`output_document` made; ``path`` names it in a refusal.

    Raises:
        OutputsError: It is not such an output.
    """
    if not isinstance(value, dict) or set(value) != {"text", "tokens", "finish_reason"}:
        raise OutputsError(f"{path}: expected the keys text, tokens and finish_reason")
    tokens = value["tokens"]
    if tokens is not None and not (isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)):
        raise OutputsError(f"{path}.tokens: expected null or a list of strings")
    if not isinstance(value["text"], str) or not isinstance(value["finish_reason"], str):
        raise OutputsError(f"{path}: text and finish_reason are strings")
    return Output(value["text"], None if tokens is None else tuple(tokens), value["finish_reason"])


def _prompt_info(value: Any, path: str) -> PromptInfo:
    if not isinstance(value, dict) or set(value) != {"id", "mode", "length"}:
        raise OutputsError(f"{path}: expected the keys id, mode and length")
    prompt_id, mode, length = value["id"], value["mode"], value["length"]
    if (
        not isinstance(prompt_id, str)
        or not isinstance(mode, str)
        or isinstance(length, bool)
        or not isinstance(length, int)
    ):
        raise OutputsError(f"{path}: id and mode are strings, length an integer")
    return PromptInfo(prompt_id, mode, length)


def _entry(value: Any, path: str) -> ProcessEntry:
    keys = {"index", "status", "reason", "boot_seconds", "query_seconds", "outputs"}
    if not isinstance(value, dict) or set(value) != keys:
        raise OutputsError(f"{path}: expected the keys {sorted(keys)}")
    index, status = value["index"], value["status"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or status not in (OK, FAILED):
        raise OutputsError(f"{path}: a non-negative integer index and a status of ok or failed")
    seconds = [value["boot_seconds"], value["query_seconds"]]
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 <= s < 1e9 for s in seconds):
        raise OutputsError(f"{path}: boot_seconds and query_seconds are non-negative numbers")
    if not isinstance(value["outputs"], dict) or not isinstance(value["reason"], str):
        raise OutputsError(f"{path}: outputs is an object and reason a string")
    outputs = {pid: parse_output(out, f"{path}.outputs.{pid}") for pid, out in value["outputs"].items()}
    if (status == OK) == (not outputs):
        raise OutputsError(f"{path}: a process is ok exactly when it holds outputs")
    return ProcessEntry(index, status, value["reason"], float(seconds[0]), float(seconds[1]), outputs)


def read_outputs(directory: Path) -> OutputsFile:
    """The file a landed bundle holds below ``directory`` (the test's output directory).

    Raises:
        OutputsError: There is no such file, or it is not a well-formed outputs file.
    """
    return parse_outputs(read_outputs_document(directory))


def read_outputs_document(directory: Path) -> dict[str, Any]:
    """The JSON object of that file, as written; :func:`parse_outputs` checks it.

    Raises:
        OutputsError: There is no such file, or it is not a JSON object.
    """
    path = Path(directory) / SHORT_PROMPTS_DIR / OUTPUTS_FILE
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_BYTES:
            raise OutputsError(f"{SHORT_PROMPTS_DIR}/{OUTPUTS_FILE} is larger than {_MAX_BYTES} bytes")
        document = parse_json_object(raw, what=f"{SHORT_PROMPTS_DIR}/{OUTPUTS_FILE}")
    except (OSError, DocumentError) as exc:
        raise OutputsError(f"the bundle holds no readable {SHORT_PROMPTS_DIR}/{OUTPUTS_FILE}: {exc}") from None
    return document


def parse_outputs(document: Mapping[str, Any]) -> OutputsFile:
    keys = {"schema", "prompt_set", "prompts_sha256", "max_new_tokens", "prompts"}
    keys.add("processes")
    listed = isinstance(document.get("processes"), list) and isinstance(document.get("prompts"), list)
    if set(document) != keys or document["schema"] != _SCHEMA or not listed:
        raise OutputsError(
            f"the outputs file holds the keys {sorted(document)}, not {sorted(keys)} of schema {_SCHEMA}"
        )
    prompts = tuple(_prompt_info(item, f"prompts[{i}]") for i, item in enumerate(document["prompts"]))
    entries = tuple(_entry(item, f"processes[{i}]") for i, item in enumerate(document["processes"]))
    indices = [entry.index for entry in entries]
    tokens = document["max_new_tokens"]
    if len(set(indices)) != len(indices) or isinstance(tokens, bool) or not isinstance(tokens, int):
        raise OutputsError("the outputs file repeats a process index, or its max_new_tokens is not an integer")
    texts = (document["prompt_set"], document["prompts_sha256"])
    if not all(isinstance(t, str) for t in texts):
        raise OutputsError("the outputs file's prompt_set and prompts_sha256 are strings")
    return OutputsFile(texts[0], texts[1], tokens, prompts, entries)
