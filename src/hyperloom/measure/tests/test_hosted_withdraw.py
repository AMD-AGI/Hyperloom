# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A withdrawn test gives its sandbox, and with it its cards, back at once (board #80, session MB-20261019T141640Z).

Cancelling the task that awaits ``run_test`` cannot reach the worker thread that measures, so a test nobody wanted any
more kept its card to the end and the tests queued behind it waited for it. ``withdraw`` is the way in that does.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.measure.hosted import STOPPED_RETURNCODE
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SENT
from hyperloom.measure.socket import TestOutcome
from hyperloom.measure.tests.hosted_fixtures import Rig, make_call, make_rig, point_yaml


@pytest.fixture
def rig(tmp_path: Path):
    made: list[Rig] = []

    def build(**kwargs) -> Rig:
        built = make_rig(tmp_path, **kwargs)
        made.append(built)
        return built

    yield build
    for built in made:
        built.backend.close()
        built.provider.close()


def _until(condition: Callable[[], bool], what: str, seconds: float = 15.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def _start(rig_: Rig, tmp: Path, task: str, **fake) -> tuple[threading.Thread, list[TestOutcome]]:
    """Run ``task`` in a worker thread, as the toolbox does, and return it with the list its outcome lands in."""
    call = make_call(tmp, task, point=point_yaml(tmp, f"{task}.yaml", **fake))
    outcomes: list[TestOutcome] = []
    worker = threading.Thread(target=lambda: outcomes.append(rig_.backend.run_test(call)), name=task)
    worker.start()
    return worker, outcomes


def _running(rig_: Rig) -> int:
    return len(rig_.provider.find({}, role="measurement"))


def _joined(worker: threading.Thread) -> None:
    worker.join(20)
    assert not worker.is_alive(), f"{worker.name} did not end"


def test_a_withdrawn_running_test_closes_its_sandbox_and_frees_its_card(rig, tmp_path) -> None:
    r = rig(gpus=1)
    worker, outcomes = _start(r, tmp_path, "req-1", fake_mode="hang")
    _until(lambda: r.provider.pool.free_cards() == () and r.backend.journal.latest("req-1") is not None, "the runner")
    r.backend.withdraw("req-1")
    _joined(worker)
    assert outcomes[0].returncode == STOPPED_RETURNCODE and outcomes[0].failure_class is None
    assert r.provider.pool.free_cards() == (0,), "the card is back in the pool"
    assert _running(r) == 0
    assert r.backend.journal.latest("req-1").state == STATE_FINISHED, "a restart does not re-attach to it"


def test_a_test_that_starts_after_its_withdrawal_makes_no_sandbox(rig, tmp_path) -> None:
    r = rig(gpus=1)
    made: list[str] = []
    real_create = r.provider.create
    r.provider.create = lambda *a, **k: made.append("sandbox") or real_create(*a, **k)  # type: ignore[method-assign]
    r.backend.withdraw("req-1")  # the worker thread of a test can start after the toolbox withdrew it
    call = make_call(tmp_path, "req-1", point=point_yaml(tmp_path, fake_mode="hang"))
    outcome = r.backend.run_test(call)
    assert outcome.returncode == STOPPED_RETURNCODE and made == []
    assert r.backend.journal.records() == [] and r.provider.pool.free_cards() == (0,)


def test_a_withdrawal_ends_that_test_and_no_other(rig, tmp_path) -> None:
    r = rig(gpus=2)
    first, first_out = _start(r, tmp_path, "req-1", fake_mode="hang")
    second, second_out = _start(r, tmp_path, "req-2", fake_mode="hang")
    _until(lambda: len({rec.request_id for rec in r.backend.journal.records()}) == 2, "both runners")
    assert r.provider.pool.free_cards() == ()
    held = {rec.request_id: rec.sandbox_id for rec in r.backend.journal.records()}
    r.backend.withdraw("req-1")
    _joined(first)
    assert first_out[0].returncode == STOPPED_RETURNCODE
    time.sleep(0.3)  # several polls: the other test must not have been touched
    assert second.is_alive() and second_out == []
    assert [i.sandbox_id for i in r.provider.find({}, role="measurement")] == [held["req-2"]]
    assert len(r.provider.pool.free_cards()) == 1, "exactly the withdrawn test's card came back"
    r.backend.withdraw("req-2")
    _joined(second)
    assert r.provider.pool.free_cards() == (0, 1)


def test_a_test_waiting_for_a_slot_leaves_the_wait_when_it_is_withdrawn(rig, tmp_path) -> None:
    r = rig(gpus=1, max_parallel_tests=1)
    holder, _ = _start(r, tmp_path, "req-1", fake_mode="hang")
    _until(lambda: _running(r) == 1, "the first runner")
    waiting, waiting_out = _start(r, tmp_path, "req-2", fake_mode="hang")
    time.sleep(0.3)
    assert waiting.is_alive(), "the second test waits for the only slot"
    r.backend.withdraw("req-2")
    _joined(waiting)
    assert waiting_out[0].returncode == STOPPED_RETURNCODE
    assert holder.is_alive() and _running(r) == 1, "the test that holds the slot runs on"
    assert r.backend.journal.latest("req-2") is None, "nothing was ever sent for the withdrawn one"
    r.backend.withdraw("req-1")
    _joined(holder)


class _Crash(BaseException):
    """What a killed application process looks like to the code under test: it unwinds without cleaning up."""


def test_a_withdrawn_test_that_would_re_attach_closes_the_sandbox_an_earlier_process_left(rig, tmp_path) -> None:
    r = rig(gpus=1)
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="hang"))
    dying = r.another_backend()
    dying._pool.release = lambda runner: None  # a dead process closes nothing

    def die(attempt):
        raise _Crash

    dying._wait = die  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        dying.run_test(call)
    assert r.backend.journal.latest("req-1").state == STATE_SENT and r.provider.pool.free_cards() == ()
    restarted = r.another_backend()
    restarted.withdraw("req-1")  # the withdrawal reaches the test before it re-attached
    outcome = restarted.run_test(call)
    assert outcome.returncode == STOPPED_RETURNCODE
    assert _running(r) == 0 and r.provider.pool.free_cards() == (0,), "the left sandbox is closed, its card is back"
    assert restarted.journal.latest("req-1").state == STATE_FINISHED, "and a later start does not re-attach to it"


def test_closing_the_backend_still_ends_a_running_test(rig, tmp_path) -> None:
    r = rig(gpus=1)
    worker, outcomes = _start(r, tmp_path, "req-1", fake_mode="hang")
    _until(lambda: _running(r) == 1, "the runner")
    r.backend.close()
    _joined(worker)
    assert len(outcomes) == 1 and _running(r) == 0
