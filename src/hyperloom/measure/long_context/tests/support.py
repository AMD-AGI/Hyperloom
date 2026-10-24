# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the long-context tests stand in for: a tokenizer, a staged prompt set and a server that answers.

The server speaks ``/v1/models`` and ``/v1/completions`` over real HTTP on a free port. It *solves* each prompt the way a
model that retrieves perfectly would (it reads the needle, follows the chain, counts the words), except for the task
families in ``broken``, where it answers with something that holds no reference: a launch whose approximation loses what
sits far back. That makes a pass or a fail per task a fact of the fixture, not of the grading code under test.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import threading
import time
import zlib
from collections import Counter
from collections.abc import Callable, Collection
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from hyperloom.contracts.long_context import PROMPT_OK
from hyperloom.measure.long_context.outputs import PromptEntry, RunFacts, write_outputs
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE, LongContextPlan
from hyperloom.measure.long_context.rounds import run_long_context
from hyperloom.measure.long_context.staging import stage_entry
from hyperloom.measure.long_context.tasks import build_prompts
from hyperloom.measure.short_prompts.tests.support import LocalServer, free_port

__all__ = ["LENGTH", "MODEL", "FakeServer", "encode", "free_port", "outputs_files", "prompt_set", "staged_plan"]

MODEL = "fake-model"
LENGTH = 2048
NIAH, TRACKING, WORDS = "niah", "tracking", "words"
_COMMON = 5


def encode(text: str) -> list[int]:
    """A tokenizer with one token per word, ids spread over 30000 values."""
    return [zlib.crc32(word.encode()) % 30000 + 10 for word in text.split()]


def _solve(prompt: str) -> str:
    """The answer of a model that retrieves perfectly."""
    if "magic number" in prompt:
        key = prompt.rsplit("magic number for ", 1)[1].split(" ", 1)[0]
        return re.search(rf"special magic numbers for {re.escape(key)} is: (\d+)", prompt).group(1)  # type: ignore[union-attr]
    if "variable assignment" in prompt:
        value = re.search(r"assigned the value (\d+) in the text", prompt).group(1)  # type: ignore[union-attr]
        found, grew = [], True
        names = dict(re.findall(r"VAR (\w+) = (?:VAR )?(\w+)\.", prompt))
        holding = {value}
        while grew:
            grew = False
            for name, source in names.items():
                if name not in found and source in holding:
                    found.append(name)
                    holding.add(name)
                    grew = True
        return " ".join(found)
    words = Counter(re.findall(r"^\d+\. (\w+)$", prompt, re.M))
    return " ".join(word for word, _ in words.most_common(_COMMON))


def _family(prompt: str) -> str:
    return NIAH if "magic number" in prompt else TRACKING if "variable assignment" in prompt else WORDS


class FakeServer(LocalServer):
    """A completions server on a free local port. ``broken`` names the task families (``niah``, ``tracking``, ``words``)
    it answers wrongly; ``fail`` says which prompt texts get an HTTP 500; ``delay`` slows every answer;
    ``served`` is the prompt token count it reports."""

    def __init__(
        self,
        *,
        broken: Collection[str] = (),
        fail: Callable[[str], bool] = lambda text: False,
        delay: float = 0.0,
        port: int = 0,
    ) -> None:
        self.requests: list[dict] = []
        self.in_flight = self.most_in_flight = 0
        lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                return

            def _reply(self, document: dict, status: int = 200) -> None:
                raw = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                self._reply({"data": [{"id": MODEL}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    outer.requests.append(body)
                    outer.in_flight += 1
                    outer.most_in_flight = max(outer.most_in_flight, outer.in_flight)
                try:
                    time.sleep(delay)
                    prompt = body["prompt"]
                    if fail(prompt):
                        return self._reply({"error": "boom"}, 500)
                    text = "unknown" if _family(prompt) in broken else _solve(prompt)
                    usage = {"prompt_tokens": len(encode(prompt)) + 1}
                    self._reply({"choices": [{"text": text, "finish_reason": "stop"}], "usage": usage})
                finally:
                    with lock:
                        outer.in_flight -= 1

        super().__init__(Handler, port)


def prompt_set(length: int = LENGTH):
    return build_prompts(encode, length=length, seed=1)


def staged_plan(cache: Path, **fields: object) -> LongContextPlan:
    """Stage the test tokenizer's prompt set below ``cache`` and the plan that names it."""
    entry = stage_entry(cache, prompt_set(), {"seed": 1, "length": LENGTH, "tokenizer": {"fake": "0" * 64}})
    return LongContextPlan(entry.key, entry.prompts_sha256, LENGTH, **fields)  # type: ignore[arg-type]


class _NoServer:
    """A host that must never be asked for a server: the launch serves too little for the check to start one."""

    async def start(self, index: int, deadline: float):
        raise AssertionError("a launch that serves too little starts no server")

    async def stop(self) -> None:
        return None

    def should_stop(self) -> bool:
        return False


def outputs_files(
    plan: LongContextPlan, broken: Collection[str] = (), *, served: int | None = None
) -> dict[str, bytes]:
    """The ``outputs.json`` a runner leaves for a check whose server loses the families in ``broken`` (``niah``, ``tracking``,
    ``words``) and serves ``served`` tokens of context, keyed by its path in the bundle. A launch that serves too little is
    the real pass refusing it, not a file made to look so."""
    entries = [
        PromptEntry(
            p.id, p.task, p.references, p.tokens, PROMPT_OK,
            text="unknown" if _family(p.text) in broken else _solve(p.text),
            finish_reason="stop", prompt_tokens=p.tokens + 1, seconds=1.5,
        )
        for p in prompt_set(plan.length)
    ]  # fmt: skip
    with tempfile.TemporaryDirectory() as scratch:
        if served is not None and served < plan.needed_context:
            asyncio.run(
                run_long_context(plan, prompt_set(plan.length), _NoServer(), Path(scratch), served_context=served)
            )
            path = Path(scratch) / LONG_CONTEXT_DIR / OUTPUTS_FILE
        else:
            path = write_outputs(Path(scratch), plan, entries, RunFacts(served, "", 80.0, 96.0))
        return {f"{LONG_CONTEXT_DIR}/{OUTPUTS_FILE}": path.read_bytes()}
