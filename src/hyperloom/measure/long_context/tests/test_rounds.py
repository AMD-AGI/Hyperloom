# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pass of the long-context check against real HTTP servers: what it asks, how, and what it leaves when it cannot finish."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.contracts.long_context import PROMPT_ERROR, PROMPT_NOT_RUN, PROMPT_OK
from hyperloom.measure.long_context.outputs import read_outputs
from hyperloom.measure.long_context.plan import LONG_CONTEXT_DIR, OUTPUTS_FILE
from hyperloom.measure.long_context.rounds import run_long_context
from hyperloom.measure.long_context.tests.support import MODEL, FakeServer, prompt_set, staged_plan
from hyperloom.measure.short_prompts.rounds import PassFailed, Serving


class _Host:
    """The backend's side: starts (or refuses to start) the server a test gives it."""

    def __init__(self, server: FakeServer | None, *, refuse: str = "") -> None:
        self.server, self.refuse = server, refuse
        self.started = self.stopped = 0

    async def start(self, index: int, deadline: float) -> Serving:
        self.started += 1
        if self.refuse:
            raise PassFailed(self.refuse)
        assert self.server is not None
        return Serving(self.server.base_url)

    async def stop(self) -> None:
        self.stopped += 1

    def should_stop(self) -> bool:
        return False


def _run(tmp_path: Path, host: _Host, *, served: int | None = None, **plan: object):
    planned = staged_plan(tmp_path / "cache", **plan)
    return asyncio.run(run_long_context(planned, prompt_set(), host, tmp_path / "out", served_context=served))


def test_every_prompt_is_asked_greedily_as_text_and_its_answer_is_recorded(tmp_path: Path) -> None:
    with FakeServer() as server:
        host = _Host(server)
        found = _run(tmp_path, host, served=4096)

    assert (host.started, host.stopped) == (1, 1), "one server, stopped after the questions"
    assert len(server.requests) == 64 and {r["model"] for r in server.requests} == {MODEL}
    first = server.requests[0]
    assert (first["temperature"], first["max_tokens"], first["stream"]) == (0, 128, False)
    assert isinstance(first["prompt"], str), "the text is sent as it is, not as token ids"
    assert [e.status for e in found.prompts] == [PROMPT_OK] * 64 and found.stopped == ""
    assert found.prompts[0].text and found.served_context == 4096
    assert (tmp_path / "out" / LONG_CONTEXT_DIR / OUTPUTS_FILE).is_file()


def test_requests_are_in_flight_up_to_the_plans_concurrency_and_no_more(tmp_path: Path) -> None:
    with FakeServer(delay=0.05) as server:
        _run(tmp_path, _Host(server), concurrency=4)

    assert 2 <= server.most_in_flight <= 4


def test_a_launch_that_serves_less_than_the_prompt_and_its_answer_starts_no_server(tmp_path: Path) -> None:
    host = _Host(None)

    found = _run(tmp_path, host, served=2048 + 127)

    assert host.started == 0, "refused before a server starts"
    assert found.stopped.startswith("served context too short: the launch serves 2175 tokens, the check needs 2176")
    assert {e.status for e in found.prompts} == {PROMPT_NOT_RUN} and found.served_context == 2175


def test_a_launch_that_serves_exactly_what_is_needed_is_asked(tmp_path: Path) -> None:
    with FakeServer() as server:
        found = _run(tmp_path, _Host(server), served=2048 + 128)

    assert found.stopped == "" and len(server.requests) == 64


def test_a_server_that_does_not_come_up_leaves_every_prompt_unrun_with_the_reason(tmp_path: Path) -> None:
    host = _Host(None, refuse="the server did not come up (boot_timeout)")

    found = _run(tmp_path, host)

    assert found.stopped == "the server did not come up (boot_timeout)" and host.stopped == 1
    assert {e.status for e in found.prompts} == {PROMPT_NOT_RUN}


def test_a_request_the_server_answers_with_an_error_is_that_prompts_error_and_the_others_are_answered(
    tmp_path: Path,
) -> None:
    with FakeServer(fail=lambda text: "numbered list of words" in text) as server:
        found = _run(tmp_path, _Host(server))

    statuses = {e.status for e in found.prompts}
    assert statuses == {PROMPT_OK, PROMPT_ERROR}
    errors = [e for e in found.prompts if e.status == PROMPT_ERROR]
    assert {e.task for e in errors} == {"common_words"} and len(errors) == 16
    assert all(e.reason == "the server answered HTTP 500" and e.text == "" for e in errors)


def test_the_phase_cap_leaves_the_prompts_it_did_not_reach_unrun_and_says_so(tmp_path: Path) -> None:
    with FakeServer(delay=0.3) as server:
        found = _run(tmp_path, _Host(server), concurrency=1, phase_cap_sec=1.0)

    done = [e for e in found.prompts if e.status == PROMPT_OK]
    unrun = [e for e in found.prompts if e.status == PROMPT_NOT_RUN]
    assert done and unrun and len(done) + len(unrun) + sum(e.status == PROMPT_ERROR for e in found.prompts) == 64
    assert found.stopped == "the phase cap ended the check before every prompt was sent"
    assert read_outputs(tmp_path / "out").stopped == found.stopped
