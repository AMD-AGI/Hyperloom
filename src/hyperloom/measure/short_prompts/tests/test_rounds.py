# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The check's one process, the test's own server (``answer_held``), against servers that answer over HTTP.

The server is already up: the check starts none, and in these tests giving it takes no time.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import pytest

from hyperloom.measure.short_prompts.client import CompletionError, complete, discover_model, run_prompts
from hyperloom.measure.short_prompts.outputs import FAILED, OK, OutputsError, read_outputs
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.rounds import PassFailed, Serving, answer_held, write_answered
from hyperloom.measure.short_prompts.tests.support import MODEL, FakeServer, good_answer, prompt_set

PLAN = ShortPromptPlan("short-prompts-test", "a" * 64)


class _Host:
    """The test's own server, already up: a fake server that answers as ``answer`` says, or none (``missing``)."""

    def __init__(self, answer: Callable = good_answer, *, missing: str = "", trickle_sec: float = 0.0) -> None:
        self._answer, self._missing, self._trickle = answer, missing, trickle_sec
        self._server: FakeServer | None = None
        self.started: list[int] = []
        self.deadlines: list[float] = []
        self.stopped = 0

    async def start(self, index: int, deadline: float) -> Serving:
        self.started.append(index)
        self.deadlines.append(deadline)
        if self._missing:
            raise PassFailed(self._missing)
        self._server = FakeServer(self._answer, trickle_sec=self._trickle).__enter__()
        return Serving(self._server.base_url, MODEL if self._trickle else "")

    async def stop(self) -> None:
        self.stopped += 1
        if self._server is not None:
            self._server.__exit__(None, None, None)
            self._server = None

    def should_stop(self) -> bool:
        return False


def _run(host: _Host, tmp_path, plan: ShortPromptPlan = PLAN):
    """The test's own server answers on ``host``, and its answers are written below ``tmp_path``."""

    async def check():
        return write_answered(await answer_held(plan, prompt_set(), host), tmp_path)

    return asyncio.run(check())


def test_the_tests_own_server_answers_every_prompt_once_and_the_check_starts_no_other(tmp_path):
    host = _Host()

    entry = _run(host, tmp_path)

    assert (host.started, host.stopped) == ([0], 1), "one server, the test's own, given once"
    assert (entry.index, entry.status, entry.boot_seconds) == (0, OK, pytest.approx(0.0, abs=0.5))
    found = read_outputs(tmp_path)
    assert [e.index for e in found.processes] == [0] and len(found.completed[0]) == len(found.prompts) == 64
    assert host.deadlines[0] - time.monotonic() <= PLAN.phase_cap_sec, "it answers within the plan's cap"


def test_the_prompts_reach_the_server_as_token_ids_greedily_in_both_modes(tmp_path):
    with FakeServer() as server:
        answers = run_prompts(server.base_url, MODEL, prompt_set(), max_new_tokens=32, deadline=time.monotonic() + 30)

    assert len(answers) == 64 and len(server.requests) == 64
    request = server.requests[0]
    assert request["temperature"] == 0 and request["max_tokens"] == 32 and request["stream"] is False
    assert all(isinstance(i, int) for i in request["prompt"]) and request["logprobs"] == 1
    sent = [r["prompt"] for r in server.requests]
    assert sent[:32] == [list(p.token_ids) for p in prompt_set()[:32]], (
        "the sequential prompts go one at a time, in order"
    )
    assert sorted(sent[32:]) == sorted(list(p.token_ids) for p in prompt_set()[32:])


def test_a_server_that_breaks_on_a_prompt_is_recorded_failed_with_the_reason(tmp_path):
    def broken(ids, count):
        raise RuntimeError("the server dies on this prompt")

    entry = _run(_Host(broken), tmp_path)

    assert entry.status == FAILED and entry.reason.startswith("the request failed") and entry.outputs == {}
    assert read_outputs(tmp_path).processes[0].status == FAILED


def test_a_test_whose_server_cannot_be_given_records_the_check_failed_with_that_reason(tmp_path):
    host = _Host(missing="the test's server was not held up for the check")

    entry = _run(host, tmp_path)

    assert (entry.status, entry.reason) == (FAILED, "the test's server was not held up for the check")
    assert host.stopped == 1 and read_outputs(tmp_path).completed == {}


def test_the_phase_cap_ends_the_questions_of_a_server_that_answers_slowly_between_its_prompts(tmp_path):
    def slow(ids, count):
        time.sleep(0.4)
        return good_answer(ids, count)

    plan = ShortPromptPlan("short-prompts-test", "a" * 64, phase_cap_sec=1.5)
    began = time.monotonic()

    entry = _run(_Host(slow), tmp_path, plan)

    assert time.monotonic() - began < 10, "32 sequential prompts at 0.4 s would take 13 s"
    assert (entry.status, entry.reason) == (FAILED, "the phase cap ended this process's questions")


def test_a_server_that_trickles_its_answer_cannot_hold_a_request_past_its_deadline():
    with FakeServer(trickle_sec=0.25) as server:  # about 90 bytes, one every quarter second, so it never idles for long
        began = time.monotonic()

        with pytest.raises(CompletionError, match="did not finish within its time"):
            complete(server.base_url, MODEL, prompt_set()[0], max_new_tokens=2, timeout_sec=1.0)

        assert time.monotonic() - began < 5, "the socket timeout alone would wait the whole 20 seconds"


def test_a_check_whose_server_trickles_ends_at_the_phase_cap_with_that_as_the_reason(tmp_path):
    plan = ShortPromptPlan("short-prompts-test", "a" * 64, phase_cap_sec=2.0)
    began = time.monotonic()

    entry = _run(_Host(trickle_sec=0.25), tmp_path, plan)

    assert time.monotonic() - began < 10
    assert entry.status == FAILED and entry.reason == "the phase cap ended this process's questions"


def test_a_missing_or_damaged_outputs_file_is_named_not_read_as_empty(tmp_path):
    with pytest.raises(OutputsError, match="no readable short_prompts/outputs.json"):
        read_outputs(tmp_path)
    (tmp_path / "short_prompts").mkdir()
    (tmp_path / "short_prompts" / "outputs.json").write_text('{"schema": 1}')
    with pytest.raises(OutputsError, match="holds the keys"):
        read_outputs(tmp_path)


def test_the_client_reads_the_model_the_server_lists_and_refuses_what_is_not_a_completion():
    with FakeServer(with_tokens=False) as server:
        assert discover_model(server.base_url, 5) == MODEL
        answer = complete(server.base_url, MODEL, prompt_set()[0], max_new_tokens=4, timeout_sec=5)
        assert answer.tokens is None and answer.text and answer.finish_reason == "length"
    with pytest.raises(CompletionError, match="request failed"):
        complete("http://127.0.0.1:1", MODEL, prompt_set()[0], max_new_tokens=4, timeout_sec=1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: {**e, "status": "ok", "outputs": {}}, "ok exactly when it holds outputs"),
        (lambda e: {**e, "status": "failed"}, "ok exactly when it holds outputs"),
        (lambda e: {**e, "status": "maybe"}, "index and a status of ok or failed"),
        (lambda e: {**e, "index": -1}, "index and a status of ok or failed"),
        (lambda e: {**e, "boot_seconds": -3}, "non-negative numbers"),
        (lambda e: {**e, "outputs": {"s00": {"text": "x"}}}, "expected the keys text, tokens and finish_reason"),
        (lambda e: {**e, "outputs": {"s00": {"text": "x", "tokens": [1], "finish_reason": ""}}}, "list of strings"),
    ],
)
def test_a_process_entry_that_is_not_what_the_runner_writes_is_refused(tmp_path, change, message):
    _run(_Host(), tmp_path)
    path = tmp_path / "short_prompts" / "outputs.json"
    document = json.loads(path.read_text())
    document["processes"][0] = change(document["processes"][0])
    path.write_text(json.dumps(document))

    with pytest.raises(OutputsError, match=message):
        read_outputs(tmp_path)


def test_an_outputs_file_that_repeats_a_process_index_is_refused(tmp_path):
    _run(_Host(), tmp_path)
    path = tmp_path / "short_prompts" / "outputs.json"
    document = json.loads(path.read_text())
    path.write_text(json.dumps({**document, "processes": [document["processes"][0]] * 2}))

    with pytest.raises(OutputsError, match="repeats a process index"):
        read_outputs(tmp_path)
