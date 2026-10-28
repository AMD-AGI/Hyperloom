# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt answers the toolbox pools per recipe, kept across a restart.

Every test that runs the short-prompt check answers it on its own server after its rounds (process 0), and a merge test,
a final candidate or the session's last root test whose recipe has fewer pooled processes than the pack's ``processes``
boots fresh ones up to that count (``short_prompts.with_extra_processes``). Every server boot is an independent launch, so the answers of every test of one
recipe (first test, confirm, merge test, final) are one sample of that recipe's launches: :func:`pooled_result` adds a
test's completed processes to the pool and compares every process pooled for its recipe.

The pool is keyed by the recipe's digest, the prompt set's hash and the token budget, and names each process by the test's
output directory and the process's index. It is a JSON-lines file below the session directory, one line per process,
flushed and fsynced like the submit log; a process already pooled is not added twice, and a torn last line, the trace of a
crash mid-write, is ignored.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.contracts.consistency import PooledConsistency
from hyperloom.contracts.measurement import TestResult
from hyperloom.measure.short_prompts.compare import Outputs, PromptInfo, compare_outputs
from hyperloom.measure.short_prompts.outputs import OutputsError, output_document, parse_output, read_outputs
from hyperloom_multibranch.toolbox.short_prompts import output_counts

__all__ = ["POOL_FILE", "PoolKey", "ShortPromptPool", "pooled_result"]

#: The pool's file below the session directory.
POOL_FILE = "short_prompt_pool.jsonl"
_RECIPE = "recipe_digest"
_PROMPTS = "prompts_sha256"
_TOKENS = "max_new_tokens"
_PROCESS = "process"
_OUTPUTS = "outputs"
_UTF8 = "utf-8"


@dataclass(frozen=True)
class PoolKey:
    """What makes two processes' answers comparable: the same recipe, the same prompts and the same token budget."""

    recipe_digest: str
    prompts_sha256: str
    max_new_tokens: int


class ShortPromptPool:
    """The answers of every completed process, by :class:`PoolKey` and process name, read back from ``path`` at start."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._answers: dict[PoolKey, dict[str, Outputs]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._replay()

    def count(self, key: PoolKey) -> int:
        """How many processes are pooled under ``key``."""
        return len(self._answers.get(key, {}))

    def answers(self, key: PoolKey) -> Mapping[str, Outputs]:
        """The pooled processes' answers under ``key``, by process name, in the order they were pooled."""
        return dict(self._answers.get(key, {}))

    def add(self, key: PoolKey, process: str, outputs: Outputs) -> None:
        """Pool ``process``'s ``outputs`` under ``key``, durably; a process already pooled stays as it was."""
        held = self._answers.setdefault(key, {})
        if process in held:
            return
        record = {
            _RECIPE: key.recipe_digest,
            _PROMPTS: key.prompts_sha256,
            _TOKENS: key.max_new_tokens,
            _PROCESS: process,
            _OUTPUTS: {prompt_id: output_document(out) for prompt_id, out in outputs.items()},
        }
        with self._path.open("a", encoding=_UTF8) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        held[process] = dict(outputs)

    def _replay(self) -> None:
        for line in self._path.read_text(encoding=_UTF8).splitlines():
            try:
                record = json.loads(line)
                key = PoolKey(str(record[_RECIPE]), str(record[_PROMPTS]), int(record[_TOKENS]))
                process = str(record[_PROCESS])
                outputs = {pid: parse_output(out, pid) for pid, out in record[_OUTPUTS].items()}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError, OutputsError):
                continue  # a torn write: only the last line can be one, and it was never acknowledged
            self._answers.setdefault(key, {}).setdefault(process, outputs)


def _pooled(pool: ShortPromptPool, key: PoolKey, prompts: tuple[PromptInfo, ...]) -> PooledConsistency:
    answers = pool.answers(key)
    comparison = compare_outputs(dict(enumerate(answers.values())), prompts)
    return PooledConsistency(
        key.recipe_digest,
        tuple(answers),
        comparison.compared_by if answers else "",
        output_counts(comparison.self_consistency),
    )


def pooled_result(pool: ShortPromptPool, recipe_digest: str, result: TestResult, session_dir: Path) -> TestResult:
    """``result`` with its short-prompt check's ``pooled`` comparison: the test's completed processes (read from its
    landed ``outputs.json`` below ``session_dir``) added to the pool of ``recipe_digest``, then every process pooled there
    compared. A result with no check, of a request logged without its recipe's digest, or whose outputs cannot be read
    (its own check then says why it is incomplete) is returned as it is."""
    check = result.short_prompt_consistency
    if check is None or not recipe_digest:
        return result
    try:
        found = read_outputs(session_dir / result.output_dir)
    except OutputsError:
        return result
    key = PoolKey(recipe_digest, found.prompts_sha256, found.max_new_tokens)
    for index, outputs in sorted(found.completed.items()):
        pool.add(key, f"{result.output_dir}#{index}", outputs)
    return replace(result, short_prompt_consistency=replace(check, pooled=_pooled(pool, key, found.prompts)))
