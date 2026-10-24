# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the tests of the short-prompt check stand in for: a tokenizer, a staged prompt set and a server that answers.

The server speaks the two endpoints the check uses over real HTTP on a free port. How it answers is a function of the
prompt, so a test makes a *defective* launch by giving it a function that is wrong for short prompts, as the defect this
check exists for is.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hyperloom.measure.short_prompts.compare import Output
from hyperloom.measure.short_prompts.outputs import FAILED, OK, ProcessEntry, write_outputs
from hyperloom.measure.short_prompts.plan import OUTPUTS_FILE, SHORT_PROMPTS_DIR, ShortPromptPlan
from hyperloom.measure.short_prompts.prompts import Prompt, build_prompts
from hyperloom.measure.short_prompts.staging import stage_entry

MODEL = "fake-model"
SHORT = 16
Answer = Callable[[Sequence[int], int], list[str]]


def encode(text: str) -> list[int]:
    """A tokenizer with one token per word, ids spread over 30000 values."""
    return [zlib.crc32(word.encode()) % 30000 + 10 for word in text.split()]


def good_answer(ids: Sequence[int], count: int) -> list[str]:
    """The tokens a correct server gives: a function of the prompt alone."""
    return [f"t{(sum(ids) * 7 + step * 13) % 997}" for step in range(count)]


def short_prompts_wrong(ids: Sequence[int], count: int) -> list[str]:
    """The defect: a prompt of at most ``SHORT`` tokens gets garbage from its second token on."""
    tokens = good_answer(ids, count)
    return tokens if len(ids) > SHORT else [tokens[0], *(f"x{step}" for step in range(1, count))]


class _Http(ThreadingHTTPServer):
    """A burst of 32 requests at once needs more than the default backlog of 5."""

    request_queue_size = 128
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # a client that gave up is not an error here
        return


class FakeServer:
    """A completions server on a free local port; ``answer`` decides what it says to a prompt."""

    def __init__(
        self, answer: Answer = good_answer, *, with_tokens: bool = True, port: int = 0, trickle_sec: float = 0.0
    ) -> None:
        self.requests: list[dict] = []
        outer = self
        trickle = trickle_sec

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                return

            def _reply(self, document: dict) -> None:
                raw = json.dumps(document).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                if not trickle:
                    self.wfile.write(raw)
                    return
                for byte in raw:  # a server that never goes quiet for long, yet never finishes
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(trickle)

            def do_GET(self) -> None:
                self._reply({"data": [{"id": MODEL}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                tokens = answer(body["prompt"], body["max_tokens"])
                choice: dict = {"text": "".join(tokens), "finish_reason": "length", "logprobs": None}
                if with_tokens:
                    choice["logprobs"] = {"tokens": tokens}
                self._reply({"choices": [choice]})

        self._http = _Http(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._http.server_address[1]}"

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.shutdown()
        self._http.server_close()
        self._thread.join(timeout=5)


def free_port() -> int:
    """A port nothing listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def prompt_set() -> tuple[Prompt, ...]:
    return build_prompts(encode, seed=1)


def outputs_files(plan: ShortPromptPlan, behaviours: Sequence[str]) -> dict[str, bytes]:
    """The ``outputs.json`` a runner leaves for a check whose processes behave as ``behaviours`` say: ``good``, ``bad`` (wrong
    for short prompts), ``drift`` (a late difference on one long prompt) or ``fail`` (did not come up)."""
    prompts = prompt_set()
    long_prompt = next(p for p in prompts if p.length > SHORT)
    entries = []
    for index, behaviour in enumerate(behaviours):
        answers = {}
        for prompt in prompts:
            wrong = behaviour == "bad" or (behaviour == "drift" and prompt.id == long_prompt.id)
            tokens = (short_prompts_wrong if wrong else good_answer)(prompt.token_ids, plan.max_new_tokens)
            if behaviour == "drift" and wrong:
                tokens = [*tokens[:-1], "drifted"]
            answers[prompt.id] = Output("".join(tokens), tuple(tokens), "length")
        failed = behaviour == "fail"
        entries.append(
            ProcessEntry(
                index,
                FAILED if failed else OK,
                "did not come up (test)" if failed else "",
                80.0,
                20.0,
                {} if failed else answers,
            )
        )
    with tempfile.TemporaryDirectory() as scratch:
        path = write_outputs(Path(scratch), plan, [p.info() for p in prompts], "", entries)
        return {f"{SHORT_PROMPTS_DIR}/{OUTPUTS_FILE}": path.read_bytes()}


def staged_plan(cache: Path, **fields: object) -> ShortPromptPlan:
    """Stage the test tokenizer's prompt set below ``cache`` and the plan that names it."""
    entry = stage_entry(cache, prompt_set(), {"seed": 1, "tokenizer": {"fake": "0" * 64}})
    return ShortPromptPlan(entry.key, entry.prompts_sha256, **fields)  # type: ignore[arg-type]
