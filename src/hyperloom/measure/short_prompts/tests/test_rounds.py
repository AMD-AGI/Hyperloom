# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The passes of the check against servers that answer over HTTP, one of them defective as the aiter prefill was."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import pytest

from hyperloom.measure.short_prompts.client import CompletionError, complete, discover_model, run_prompts
from hyperloom.measure.short_prompts.compare import compare_outputs
from hyperloom.measure.short_prompts.outputs import FAILED, OK, OutputsError, read_outputs
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.short_prompts.rounds import CAP_MARGIN_FRACTION, CAP_MARGIN_SEC, PassFailed, Serving, run_passes
from hyperloom.measure.short_prompts.tests.support import (
    MODEL,
    FakeServer,
    good_answer,
    prompt_set,
    short_prompts_wrong,
)

PLAN = ShortPromptPlan("short-prompts-test", "a" * 64, processes=6)


class _Host:
    """Starts a fake server for each process; ``answers`` says how the process with that index answers."""

    def __init__(
        self,
        answers: Callable[[int], object],
        *,
        failing: frozenset[int] = frozenset(),
        clock=None,
        boots: Callable[[int], float] = lambda index: 400.0,
    ) -> None:
        self._answers, self._failing, self._clock, self._boots = answers, failing, clock, boots
        self._server: FakeServer | None = None
        self.started: list[int] = []
        self.deadlines: list[float] = []
        self.stopped = 0
        self.stop_now = False

    async def start(self, index: int, deadline: float) -> Serving:
        self.started.append(index)
        self.deadlines.append(deadline)
        if self._clock is not None:
            self._clock.now += self._boots(index)
        if index in self._failing:
            raise PassFailed("the server did not come up (test)")
        self._server = FakeServer(self._answers(index)).__enter__()
        return Serving(self._server.base_url, "")

    async def stop(self) -> None:
        self.stopped += 1
        if self._server is not None:
            self._server.__exit__(None, None, None)
            self._server = None

    def should_stop(self) -> bool:
        return self.stop_now


def _run(host: _Host, tmp_path, plan: ShortPromptPlan = PLAN, **kwargs):
    return asyncio.run(run_passes(plan, prompt_set(), host, tmp_path, **kwargs))


def test_a_server_that_is_wrong_for_short_prompts_in_one_launch_in_three_is_caught_by_the_other_launches(tmp_path):
    defective = {1, 4}
    host = _Host(lambda index: short_prompts_wrong if index in defective else good_answer)

    entries = _run(host, tmp_path)

    assert [e.status for e in entries] == [OK] * 6 and host.stopped == 6, "every process is stopped"
    found = read_outputs(tmp_path)
    own = compare_outputs(found.completed, None, prompt_set()).self_consistency
    assert own is not None
    assert [count > 0 for count in own.mismatches] == [i in defective for i in range(6)]
    assert (
        own.first is not None and own.first.process == 1 and own.first.prompt_tokens <= 16 and own.first.position == 1
    )


def test_a_server_that_answers_the_same_every_launch_is_clean_in_every_comparison(tmp_path):
    host = _Host(lambda index: good_answer)

    _run(host, tmp_path)

    found = read_outputs(tmp_path)
    result = compare_outputs(found.completed, found.completed, prompt_set())
    assert result.self_consistency.mismatches == (0,) * 6 and result.self_consistency.first is None
    assert result.reference.mismatches == (0,) * 6 and result.reference.unstable == 0
    assert found.stopped == "" and len(found.prompts) == 64


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


def test_a_process_that_does_not_come_up_is_recorded_failed_with_its_reason_and_the_next_one_is_tried(tmp_path):
    host = _Host(lambda index: good_answer, failing=frozenset({2}))

    entries = _run(host, tmp_path)

    assert [e.status for e in entries] == [OK, OK, FAILED, OK, OK, OK]
    assert entries[2].reason == "the server did not come up (test)" and entries[2].outputs == {}
    assert read_outputs(tmp_path).processes[2].status == FAILED and host.stopped == 6


def test_a_server_that_breaks_on_a_prompt_fails_that_process_only(tmp_path):
    def answers(index: int):
        def broken(ids, count):
            raise RuntimeError("the server dies on this prompt")

        return broken if index == 0 else good_answer

    entries = _run(_Host(answers), tmp_path, ShortPromptPlan("short-prompts-test", "a" * 64, processes=2))

    assert entries[0].status == FAILED and entries[0].reason.startswith("the request failed")
    assert entries[1].status == OK


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def _booted(output_dir, seconds: float) -> None:
    """The ready stamp the launcher leaves beside the test's own server log: the test's server booted in ``seconds``."""
    stamp = output_dir / "files" / "variant_00_test" / "bench" / "server_ready_at"
    stamp.parent.mkdir(parents=True)
    stamp.write_text(f"1700000000.000 {seconds:.3f}\n", encoding="utf-8")


def _sized(boot: float, answers: float = 0.0, processes: int = 6) -> float:
    return processes * (boot + answers) * (1 + CAP_MARGIN_FRACTION) + CAP_MARGIN_SEC


def test_the_first_check_is_capped_from_the_tests_own_boot_not_the_plans_fixed_cap(tmp_path):
    clock = _Clock()
    host = _Host(lambda index: good_answer, clock=clock)
    _booted(tmp_path, 400.0)

    entries = _run(host, tmp_path, PLAN, clock=clock)

    assert PLAN.phase_cap_sec == 1200.0 and host.deadlines[0] == _sized(400.0) == 3120.0
    assert [e.status for e in entries] == [OK] * 6, "six 400 s boots do not fit the plan's 1200 s, and fit 3120 s"
    assert read_outputs(tmp_path).stopped == ""


def test_with_no_boot_measured_the_plans_cap_holds_until_the_first_process_has_measured_one(tmp_path):
    clock = _Clock()
    host = _Host(lambda index: good_answer, clock=clock)
    plan = ShortPromptPlan("short-prompts-test", "a" * 64, processes=6, phase_cap_sec=1000.0)

    entries = _run(host, tmp_path, plan, clock=clock)

    assert host.deadlines[:2] == [1000.0, _sized(400.0)], "the plan's cap, then the cap the first boot sizes"
    assert len(entries) == 6 and all(entry.boot_seconds == 400.0 for entry in entries)


def test_the_cap_ends_a_check_whose_process_hangs_with_the_reason_and_the_processes_it_got_to(tmp_path):
    clock = _Clock()
    hangs = 2
    host = _Host(
        lambda index: good_answer,
        clock=clock,
        failing=frozenset({hangs}),
        boots=lambda i: 900.0 if i == hangs else 100.0,
    )
    _booted(tmp_path, 100.0)

    entries = _run(host, tmp_path, PLAN, clock=clock)

    assert host.deadlines == [_sized(100.0)] * 3, "a failed process measures nothing, so the cap stays 870 s"
    assert [e.status for e in entries] == [OK, OK, FAILED], "1100 s have passed: nothing fits in what is left"
    assert "phase cap" in read_outputs(tmp_path).stopped


def test_the_phase_cap_also_ends_a_process_whose_server_answers_slowly_between_its_prompts(tmp_path):
    def slow(index: int):
        def answer(ids, count):
            time.sleep(0.4)
            return good_answer(ids, count)

        return answer

    plan = ShortPromptPlan("short-prompts-test", "a" * 64, processes=3, phase_cap_sec=1.5)
    began = time.monotonic()

    entries = _run(_Host(slow), tmp_path, plan)

    assert time.monotonic() - began < 10, "32 sequential prompts at 0.4 s would take 13 s"
    assert entries[0].status == FAILED and entries[0].reason == "the phase cap ended this process's questions"
    assert len(entries) == 1, "no further process is started once the cap has passed"
    assert "phase cap" in read_outputs(tmp_path).stopped


def test_a_server_that_trickles_its_answer_cannot_hold_a_request_past_its_deadline():
    with FakeServer(trickle_sec=0.25) as server:  # about 90 bytes, one every quarter second, so it never idles for long
        began = time.monotonic()

        with pytest.raises(CompletionError, match="did not finish within its time"):
            complete(server.base_url, MODEL, prompt_set()[0], max_new_tokens=2, timeout_sec=1.0)

        assert time.monotonic() - began < 5, "the socket timeout alone would wait the whole 20 seconds"


def test_a_check_whose_server_trickles_ends_at_the_phase_cap_with_that_as_the_reason(tmp_path):
    def trickling(index: int):
        return good_answer

    class _Trickling(_Host):
        async def start(self, index: int, deadline: float) -> Serving:
            self.started.append(index)
            self._server = FakeServer(good_answer, trickle_sec=0.25).__enter__()
            return Serving(self._server.base_url, MODEL)

    plan = ShortPromptPlan("short-prompts-test", "a" * 64, processes=2, phase_cap_sec=2.0)
    began = time.monotonic()

    entries = _run(_Trickling(trickling), tmp_path, plan)

    assert time.monotonic() - began < 10
    assert entries[0].status == FAILED and entries[0].reason == "the phase cap ended this process's questions"


def test_a_stopped_run_starts_no_further_process(tmp_path):
    host = _Host(lambda index: good_answer)
    host.stop_now = True

    assert _run(host, tmp_path) == ()
    assert read_outputs(tmp_path).stopped == "the run was stopped"


def test_the_completed_processes_are_written_after_each_one_so_a_crash_keeps_them(tmp_path):
    seen = []

    class _Crashing(_Host):
        async def start(self, index: int, deadline: float) -> Serving:
            if index == 2:
                seen.append(len(read_outputs(tmp_path).processes))
                raise KeyboardInterrupt
            return await super().start(index, deadline)

    with pytest.raises(KeyboardInterrupt):
        _run(_Crashing(lambda index: good_answer), tmp_path)

    assert seen == [2]


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
    _run(_Host(lambda index: good_answer), tmp_path, ShortPromptPlan("short-prompts-test", "a" * 64, processes=1))
    path = tmp_path / "short_prompts" / "outputs.json"
    document = json.loads(path.read_text())
    document["processes"][0] = change(document["processes"][0])
    path.write_text(json.dumps(document))

    with pytest.raises(OutputsError, match=message):
        read_outputs(tmp_path)


def test_an_outputs_file_that_repeats_a_process_index_or_cuts_its_counts_is_refused(tmp_path):
    _run(_Host(lambda index: good_answer), tmp_path, ShortPromptPlan("short-prompts-test", "a" * 64, processes=2))
    path = tmp_path / "short_prompts" / "outputs.json"
    document = json.loads(path.read_text())
    path.write_text(json.dumps({**document, "processes": [document["processes"][0]] * 2}))

    with pytest.raises(OutputsError, match="repeats a process index"):
        read_outputs(tmp_path)
