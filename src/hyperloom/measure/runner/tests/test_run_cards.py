# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner and a process that is not the test's: it refuses the result, whatever the test did, and the signed result
says what it could and could not check of its cards."""

from __future__ import annotations

import functools
import time
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure.failures import Disposition, FailureClass, classify_test_failure
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.runner import run as runner
from hyperloom.measure.runner.card_watch import CardWatch
from hyperloom.measure.runner.tests.test_run import _Backend, _go, _ok, _sheet, _status, env, key  # noqa: F401 - fixtures
from hyperloom.measure.socket import TestCall, TestOutcome
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import CARD_GUARD_ENV, CardReading, own_sandbox_processes
from hyperloom.sandbox.card_occupancy import read_holders as read_the_machine
from hyperloom.sandbox.tests.kernel_tree import KernelTree

RUNNER = 100
EXIT_FOREIGN_PROCESS = 6


@pytest.fixture
def tree(tmp_path: Path, env, monkeypatch: pytest.MonkeyPatch) -> KernelTree:
    """The machine the runner runs on: it is the process ``RUNNER``, in a sandbox of its own, and card 0 is its card."""
    env[CARD_GUARD_ENV] = "enforce"  # a test of the observing guard says so
    machine = KernelTree(tmp_path / "machine")
    machine.process(RUNNER, environ=("E2B_SANDBOX_ID=sbx-1",))
    monkeypatch.setattr(
        card_occupancy, "read_holders", lambda cards: read_the_machine(cards, kfd=machine.kfd, procfs=machine.procfs)
    )
    monkeypatch.setattr(
        runner,
        "own_sandbox_processes",
        lambda environ: own_sandbox_processes(environ, pid=RUNNER, procfs=machine.procfs),
    )
    monkeypatch.setattr(runner, "CardWatch", functools.partial(CardWatch, poll_sec=0.01))
    return machine


def _manifest(tmp_path: Path) -> ResultManifest:
    return ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes())


def _intruder(tree: KernelTree, pid: int = 900, card: int = 0) -> None:
    tree.process(pid, command=("python3", "-c", "import torch"))
    tree.holds(pid, card)


def _refused_with_nothing_run(tmp_path: Path, env, backend: _Backend) -> dict:
    assert _go(tmp_path, env, _sheet()) == EXIT_FOREIGN_PROCESS
    status = _status(tmp_path)
    assert status["state"] == "failed" and status["error_class"] == "foreign_process_on_card"
    assert backend.calls == [] and not (tmp_path / "out" / "manifest.json").exists()
    return status


def test_a_process_on_the_card_when_the_runner_starts_refuses_the_test_before_any_of_it_runs(
    tmp_path, env, tree, monkeypatch
) -> None:
    _intruder(tree)
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    status = _refused_with_nothing_run(tmp_path, env, backend)

    assert "pid 900 on card 0" in status["message"] and "before the test: python3 -c import torch" in status["message"]


def test_a_process_that_takes_the_card_after_the_start_check_is_caught_before_the_test_starts(
    tmp_path, env, tree, monkeypatch
) -> None:
    """The sheet arrives (it can take minutes), then the environment is rebuilt: the card is looked at again at launch."""
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    real_prepare = runner._prepare

    def prepare_then_the_intruder_arrives(*args, **kwargs):
        done = real_prepare(*args, **kwargs)
        _intruder(tree)
        return done

    monkeypatch.setattr(runner, "_prepare", prepare_then_the_intruder_arrives)

    status = _refused_with_nothing_run(tmp_path, env, backend)

    assert "before the test" in status["message"]


def test_a_process_that_holds_the_card_for_a_while_during_the_test_refuses_the_result_after_it_left(
    tmp_path, env, tree, monkeypatch
) -> None:
    def a_test_a_process_interrupts(call: TestCall) -> TestOutcome:
        _intruder(tree)
        time.sleep(0.3)  # the watch looks every 10 ms
        tree.exit(900)
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(a_test_a_process_interrupts))

    assert _go(tmp_path, env, _sheet()) == EXIT_FOREIGN_PROCESS

    status = _status(tmp_path)
    assert status["error_class"] == "foreign_process_on_card" and "during the test" in status["message"]
    assert not (tmp_path / "out" / "manifest.json").exists(), "a result measured next to an intruder is never signed"


def test_a_process_still_on_the_card_when_the_test_ends_refuses_the_result(tmp_path, env, tree, monkeypatch) -> None:
    def a_test_that_ends_with_an_intruder(call: TestCall) -> TestOutcome:
        _intruder(tree)
        return _ok(call)

    monkeypatch.setattr(runner, "CardWatch", functools.partial(CardWatch, poll_sec=3600))
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(a_test_that_ends_with_an_intruder))

    assert _go(tmp_path, env, _sheet()) == EXIT_FOREIGN_PROCESS

    assert "after the test" in _status(tmp_path)["message"]


def test_a_test_that_failed_beside_an_intruder_is_refused_as_infrastructure_not_signed_as_the_candidates_failure(
    tmp_path, env, tree, monkeypatch
) -> None:
    """Out of memory beside another process's memory is not a fact about the candidate."""

    def failing(call: TestCall) -> TestOutcome:
        _intruder(tree)
        return TestOutcome(1, "", "HIP out of memory", (), SandboxFacts(sandbox_id="x"))

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(failing))

    assert _go(tmp_path, env, _sheet()) == EXIT_FOREIGN_PROCESS

    assert _status(tmp_path)["error_class"] == "foreign_process_on_card"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_the_servers_own_processes_on_the_card_are_not_an_intruder_and_the_result_says_clean(
    tmp_path, env, tree, monkeypatch
) -> None:
    def a_test_with_a_server(call: TestCall) -> TestOutcome:
        tree.process(901, parent=RUNNER, command=("python3", "-m", "sglang.launch_server"))
        tree.holds(901, 0)
        tree.process(902, command=("sglang::scheduler",), environ=("PATH=/bin", "E2B_SANDBOX_ID=sbx-1"))  # detached
        tree.holds(902, 0)
        time.sleep(0.1)
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(a_test_with_a_server))

    assert _go(tmp_path, env, _sheet()) == 0

    occupancy = _manifest(tmp_path).runner.card_occupancy
    assert occupancy is not None and (occupancy.state, occupancy.reason) == ("clean", None)
    assert occupancy.checks >= 4, "before the start, before the launch, while it ran and after it"
    assert _status(tmp_path)["message"] == ""


# --- the card guard: observing is the default, enforcing is the refusal above -----------------------------------------------


def test_a_guard_that_observes_keeps_a_result_measured_beside_a_process_and_signs_who_was_seen(
    tmp_path, env, tree, monkeypatch
) -> None:
    env[CARD_GUARD_ENV] = "observe"

    def a_test_an_intruder_interrupts(call: TestCall) -> TestOutcome:
        _intruder(tree)
        time.sleep(0.3)  # the watch looks every 10 ms
        tree.exit(900)
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(a_test_an_intruder_interrupts))

    assert _go(tmp_path, env, _sheet()) == 0

    occupancy = _manifest(tmp_path).runner.card_occupancy
    assert occupancy is not None and occupancy.state == "foreign" and occupancy.checks >= 4
    assert (
        occupancy.reason is not None
        and "pid 900 on card 0" in occupancy.reason
        and "during the test" in occupancy.reason
    )
    status = _status(tmp_path)
    assert status["state"] == "succeeded" and status["error_class"] is None
    assert status["message"].startswith("CARD GUARD OBSERVES ONLY (HYPERLOOM_CARD_GUARD=observe): this result was kept")
    assert "pid 900 on card 0" in status["message"]


def test_a_guard_that_observes_runs_a_test_whose_card_a_process_held_when_the_runner_started(
    tmp_path, env, tree, monkeypatch
) -> None:
    env[CARD_GUARD_ENV] = "observe"
    _intruder(tree)
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 0

    assert len(backend.calls) == 1
    occupancy = _manifest(tmp_path).runner.card_occupancy
    assert occupancy is not None and occupancy.state == "foreign" and "before the test" in (occupancy.reason or "")


def test_a_guard_named_nothing_observes(tmp_path, env, tree, monkeypatch) -> None:
    del env[CARD_GUARD_ENV]
    _intruder(tree)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert _manifest(tmp_path).runner.card_occupancy.state == "foreign"  # type: ignore[union-attr]


def test_a_guard_that_is_not_one_stops_the_runner_before_anything_runs(tmp_path, env, monkeypatch) -> None:
    env[CARD_GUARD_ENV] = "enfroce"
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 2

    status = _status(tmp_path)
    assert status["error_class"] == "runner_start_failed" and "HYPERLOOM_CARD_GUARD is 'enfroce'" in status["message"]
    assert backend.calls == []


def test_a_list_nobody_could_read_leaves_the_result_unknown_with_the_reason_never_clean(
    tmp_path, env, monkeypatch
) -> None:
    monkeypatch.setattr(
        card_occupancy,
        "read_holders",
        lambda cards: CardReading(unavailable="this process runs in a pid namespace of its own"),
    )
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    occupancy = _manifest(tmp_path).runner.card_occupancy
    assert occupancy is not None
    assert (occupancy.state, occupancy.reason) == ("unknown", "this process runs in a pid namespace of its own")
    assert occupancy.checks == 0


def test_a_runner_with_no_card_named_states_nothing_of_its_cards_and_says_it_checked_none(
    tmp_path, env, monkeypatch
) -> None:
    del env["ROCR_VISIBLE_DEVICES"]
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert _manifest(tmp_path).runner.card_occupancy is None
    assert (
        "ROCR_VISIBLE_DEVICES names no card; the cards were not checked for other processes"
        in _status(tmp_path)["message"]
    )


def test_the_card_is_looked_at_before_its_memory_so_the_failure_names_the_process(
    tmp_path, env, tree, monkeypatch
) -> None:
    """A process that holds a card also holds memory on it: the process list answers first and says who."""
    _intruder(tree)
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: [0])

    assert _go(tmp_path, env, None) == EXIT_FOREIGN_PROCESS

    assert _status(tmp_path)["error_class"] == "foreign_process_on_card"


def test_memory_held_by_something_the_list_does_not_name_is_still_a_start_failure(
    tmp_path, env, tree, monkeypatch
) -> None:
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: [0])

    assert _go(tmp_path, env, None) == 2

    assert _status(tmp_path)["error_class"] == "runner_start_failed"


def test_the_failure_is_an_infrastructure_failure_the_retry_policy_retries() -> None:
    verdict = classify_test_failure(returncode=0, error_class="foreign_process_on_card")

    assert (verdict.disposition, verdict.failure_class, verdict.failure_reason) == (
        Disposition.FAILED,
        FailureClass.INFRA,
        "foreign_process_on_card",
    )
