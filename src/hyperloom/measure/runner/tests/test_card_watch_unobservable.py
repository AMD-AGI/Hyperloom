# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's card watch in a container shown only some GPUs (a one-GPU Agent Sandbox pod read ``permission denied`` on
the other GPUs' topology nodes). A view that still reads its own card's node is watched as any pod is: the server counts,
a foreign holder is caught. A view refused the list for its cards is ``unobservable``, never ``clean``; enforcing, it is
refused before the test starts unless the one switch, ``REFUSE_UNOBSERVABLE``, says to keep it with a note."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import CARD_CLEAN, CARD_FOREIGN, CARD_UNOBSERVABLE
from hyperloom.measure.failures import CARDS_UNOBSERVABLE, Disposition, classify_test_failure
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.runner import card_watch
from hyperloom.measure.runner import run as runner
from hyperloom.measure.runner.card_watch import AFTER_TEST, BEFORE_TEST, DURING_TEST, CardWatch
from hyperloom.measure.runner.service import RunnerFailure
from hyperloom.measure.runner.tests.test_run import _Backend, _go, _ok, _sheet, _status, env, key  # noqa: F401 - fixtures
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import CARD_GUARD_ENV, METHOD_CENSUS_FDINFO, CardGuard, OwnProcesses
from hyperloom.sandbox.card_occupancy import read_holders as read_the_machine
from hyperloom.sandbox.tests.kernel_tree import KernelTree

#: The machine's GPU the pod is given; the pod numbers it 0.
GIVEN = 3
SERVER_VRAM = 200 << 30
FOREIGN_VRAM = 3 << 30
EXIT_UNNAMED_FAILURE = 1
_COULD_NOT_LOOK = "CARD GUARD COULD NOT LOOK"


def _pod(root: Path, monkeypatch: pytest.MonkeyPatch, *shown: int) -> KernelTree:
    machine = KernelTree(root, pid_namespace="pid:[4026552279]")
    machine.opened_only(1678127)
    machine.show_only(*shown)
    machine.refuse_denied(monkeypatch)
    monkeypatch.setattr(
        card_occupancy, "read_holders", lambda cards: read_the_machine(cards, kfd=machine.kfd, procfs=machine.procfs)
    )
    return machine


@pytest.fixture
def pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KernelTree:
    """A one-GPU pod: only the machine's GPU ``GIVEN`` may be read."""
    return _pod(tmp_path / "pod", monkeypatch, GIVEN)


@pytest.fixture
def blind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KernelTree:
    """A pod refused every GPU node of the topology."""
    return _pod(tmp_path / "blind", monkeypatch)


def _watch(guard: CardGuard, procfs: Path) -> CardWatch:
    return CardWatch((0,), OwnProcesses(frozenset({100}), procfs=procfs), guard)


@pytest.fixture
def keep_unobservable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The switch turned off: an enforcing guard keeps an unobservable result."""
    monkeypatch.setattr(card_watch, "REFUSE_UNOBSERVABLE", False)


def _server(pod: KernelTree) -> None:
    pod.holds(1827025, GIVEN, vram=SERVER_VRAM, queues=6)
    pod.process(6495)
    pod.renders(6495, GIVEN, kib=SERVER_VRAM >> 10)


def test_a_pod_that_reads_only_its_cards_node_watches_its_server_as_clean(pod: KernelTree) -> None:
    watch = _watch(CardGuard.ENFORCE, pod.procfs)
    watch.check(BEFORE_TEST)
    _server(pod)
    watch.check(DURING_TEST)
    watch.check(AFTER_TEST)
    watch.enforce_guard()

    statement = watch.statement()

    assert statement is not None
    assert (statement.state, statement.reason, statement.checks, statement.method) == (
        CARD_CLEAN,
        None,
        3,
        METHOD_CENSUS_FDINFO,
    )


def test_a_pod_that_reads_only_its_cards_node_still_refuses_a_foreign_holder_of_it(pod: KernelTree) -> None:
    watch = _watch(CardGuard.ENFORCE, pod.procfs)
    watch.check(BEFORE_TEST)
    _server(pod)
    pod.holds(1845543, GIVEN, vram=FOREIGN_VRAM, queues=2)
    watch.check(DURING_TEST)

    with pytest.raises(RunnerFailure) as refused:
        watch.enforce_guard()

    assert refused.value.error_class == "foreign_process_on_card"
    assert "2 processes with memory on card 0 at once" in str(refused.value)
    statement = watch.statement()
    assert statement is not None and statement.state == CARD_FOREIGN


def test_a_view_refused_every_gpu_node_is_unobservable_never_clean_and_an_enforcing_guard_refuses_it(
    blind: KernelTree,
) -> None:
    watch = _watch(CardGuard.ENFORCE, blind.procfs)
    watch.check(BEFORE_TEST)

    statement = watch.statement()
    assert statement is not None
    assert (statement.state, statement.checks, statement.method) == (CARD_UNOBSERVABLE, 0, None)
    assert statement.reason is not None and "permission denied" in statement.reason
    with pytest.raises(RunnerFailure) as refused:
        watch.enforce_guard()
    assert refused.value.error_class == CARDS_UNOBSERVABLE
    assert str(refused.value).startswith("the test's cards cannot be looked at: every one of the 8 GPU nodes")


def test_the_switch_refuses_by_default_and_turned_off_keeps_the_result_unobservable_with_a_note(
    blind: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert card_watch.REFUSE_UNOBSERVABLE is True
    monkeypatch.setattr(card_watch, "REFUSE_UNOBSERVABLE", False)
    kept = _watch(CardGuard.ENFORCE, blind.procfs)
    kept.check(BEFORE_TEST)
    kept.check(AFTER_TEST)

    kept.enforce_guard()

    statement = kept.statement()
    assert statement is not None and statement.state == CARD_UNOBSERVABLE
    note = kept.observation()
    assert note is not None and note.startswith(_COULD_NOT_LOOK) and "permission denied" in note


def test_an_observing_guard_keeps_an_unobservable_result_and_says_so(blind: KernelTree) -> None:
    watch = _watch(CardGuard.OBSERVE, blind.procfs)
    watch.check(BEFORE_TEST)

    watch.enforce_guard()

    statement = watch.statement()
    assert statement is not None and statement.state == CARD_UNOBSERVABLE
    assert (watch.observation() or "").startswith(_COULD_NOT_LOOK)


def test_one_refused_look_among_read_ones_leaves_the_result_unobservable_not_clean(pod: KernelTree) -> None:
    watch = _watch(CardGuard.ENFORCE, pod.procfs)
    watch.check(BEFORE_TEST)
    pod.denied.add(pod.kfd / "proc")
    watch.check(AFTER_TEST)

    statement = watch.statement()

    assert statement is not None
    assert (statement.state, statement.checks) == (CARD_UNOBSERVABLE, 1)
    assert statement.reason is not None and "process list" in statement.reason


def test_an_unobservable_refusal_is_a_program_error_not_retried_in_a_sandbox_that_would_be_refused_alike() -> None:
    verdict = classify_test_failure(returncode=1, error_class=CARDS_UNOBSERVABLE)

    assert (verdict.disposition, verdict.failure_class, verdict.failure_reason) == (
        Disposition.PROGRAM_ERROR,
        None,
        CARDS_UNOBSERVABLE,
    )


@pytest.fixture
def blind_runner(env, blind: KernelTree, monkeypatch: pytest.MonkeyPatch) -> _Backend:
    env[CARD_GUARD_ENV] = "enforce"
    monkeypatch.setattr(runner, "CardWatch", functools.partial(CardWatch, poll_sec=0.01))
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    return backend


def test_a_runner_whose_view_is_refused_its_cards_refuses_the_test_before_any_of_it_runs(
    tmp_path, env, blind_runner: _Backend
) -> None:
    assert _go(tmp_path, env, _sheet()) == EXIT_UNNAMED_FAILURE

    status = _status(tmp_path)
    assert (status["state"], status["error_class"]) == ("failed", CARDS_UNOBSERVABLE)
    assert blind_runner.calls == [] and not (tmp_path / "out" / "manifest.json").exists()


def test_a_runner_with_the_refusal_switched_off_signs_unobservable_and_notes_it(
    tmp_path, env, blind_runner: _Backend, keep_unobservable: None
) -> None:
    assert _go(tmp_path, env, _sheet()) == 0

    occupancy = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).runner.card_occupancy
    assert occupancy is not None and occupancy.state == CARD_UNOBSERVABLE and occupancy.reason
    assert _COULD_NOT_LOOK in _status(tmp_path)["message"]
