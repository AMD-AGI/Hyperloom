# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make-before-break through the hosted backend: the sandbox of a test whose result landed waits for the next test's
sandbox to run; the next one is created only after that test is done with its cards, and nothing outlives the session.

The backend runs over the local sandbox provider and the fake runner, behind a provider that records every sandbox
that runs and every one that is closed, in order, and what was true of the previous test when the next create began.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure import hosted
from hyperloom.measure.hosted import STOPPED_RETURNCODE, HostedMeasurementBackend
from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom.measure.sandboxes import Handover
from hyperloom.measure.socket import TestOutcome
from hyperloom.measure.tests.hosted_fixtures import Rig, make_call, make_rig, point_yaml
from hyperloom.sandbox.socket import SandboxGone


class _Recording:
    """The local provider, recording ``("running", id)`` when a create returns and ``("close", id)`` when one is killed;
    ``before_create`` runs before each create is passed on."""

    def __init__(self, provider: Any, before_create: Callable[[], None]) -> None:
        self._provider = provider
        self._before_create = before_create
        self.events: list[tuple[str, str]] = []
        self.hold: threading.Event | None = None
        self.creating = threading.Event()
        #: Wraps each sandbox's ``startup_exit_code``: how the runner's exit is reported (``None``: as it happens).
        self.exit_code: Callable[[Callable[[], int | None]], int | None] | None = None

    def create(self, template: str, **kwargs: Any) -> Any:
        self._before_create()
        self.creating.set()
        if self.hold is not None:
            assert self.hold.wait(20), "the test never let the create through"
        sandbox = self._provider.create(template, **kwargs)
        self.events.append(("running", sandbox.sandbox_id))
        kill = sandbox.kill

        def recorded_kill() -> None:
            if not sandbox.ended:
                self.events.append(("close", sandbox.sandbox_id))
            kill()

        sandbox.kill = recorded_kill
        if self.exit_code is not None:
            real, report = sandbox.startup_exit_code, self.exit_code
            sandbox.startup_exit_code = lambda: report(real)
        return sandbox

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def _backend(tmp: Path, before_create: Callable[[], None] = lambda: None, **config: Any):
    rig = make_rig(tmp, gpus=2, handover=Handover(3, 2), **config)
    rig.backend.close()
    recording = _Recording(rig.provider, before_create)
    backend = HostedMeasurementBackend(recording, rig.config)
    backend.start()
    return rig, recording, backend


def _run(backend: HostedMeasurementBackend, tmp: Path, task: str, **fake: Any) -> TestOutcome:
    return backend.run_test(make_call(tmp, task, point=point_yaml(tmp, f"{task}.yaml", **fake)))


def _live(rig: Rig) -> list[str]:
    return [info.sandbox_id for info in rig.provider.find({}, role="measurement")]


def _until(condition: Callable[[], bool], what: str, seconds: float = 15.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def test_the_next_sandbox_is_created_after_the_result_landed_and_runs_before_the_old_one_closes(tmp_path) -> None:
    seen: list[dict[str, Any]] = []
    rig: Rig | None = None
    backend: HostedMeasurementBackend | None = None

    def before_create() -> None:
        assert rig is not None and backend is not None
        record = backend.journal.latest("req-1")
        if record is None:
            return  # req-1's own create
        sandbox = rig.provider.connect(str(record.sandbox_id), role="measurement")
        seen.append(
            {
                "state": record.state,
                "landed": sorted(path.name for path in (tmp_path / "out" / "req-1").iterdir()),
                "runner_exit": sandbox.startup_exit_code(),
            }
        )

    rig, recording, backend = _backend(tmp_path, before_create)
    try:
        backend.pool.note_upcoming({"req-1", "req-2"})
        first = _run(backend, tmp_path, "req-1")
        assert first.returncode == 0, first.stderr
        one = str(backend.journal.latest("req-1").sandbox_id)
        assert backend.pool.parked == one and _live(rig) == [one], "the idle sandbox is kept for req-2"
        assert backend.pool.active == 0
        second = _run(backend, tmp_path, "req-2")
        assert second.returncode == 0, second.stderr
        two = str(backend.journal.latest("req-2").sandbox_id)
        [before] = seen
        assert before["state"] == STATE_FINISHED and before["landed"], "req-1's result landed before req-2's create"
        assert before["runner_exit"] == 0, "req-1's runner, and with it its server, had exited"
        assert recording.events[:3] == [("running", one), ("running", two), ("close", one)]
    finally:
        backend.close()
        rig.provider.close()


def test_a_result_that_did_not_land_releases_its_sandbox_at_once(tmp_path) -> None:
    rig, recording, backend = _backend(tmp_path)
    try:
        backend.pool.note_upcoming({"req-1", "req-2"})
        outcome = _run(backend, tmp_path, "req-1", fake_mode="tamper")
        assert outcome.returncode != 0
        assert backend.pool.parked is None and _live(rig) == []
    finally:
        backend.close()
        rig.provider.close()


def test_without_a_next_test_the_sandbox_is_released(tmp_path) -> None:
    rig, recording, backend = _backend(tmp_path)
    try:
        backend.pool.note_upcoming({"req-1"})
        assert _run(backend, tmp_path, "req-1").returncode == 0
        assert backend.pool.parked is None and _live(rig) == []
    finally:
        backend.close()
        rig.provider.close()


def test_a_session_closed_during_the_overlap_leaves_no_sandbox(tmp_path) -> None:
    rig, recording, backend = _backend(tmp_path)
    try:
        backend.pool.note_upcoming({"req-1", "req-2"})
        assert _run(backend, tmp_path, "req-1").returncode == 0
        one = backend.pool.parked
        assert one is not None
        recording.hold = threading.Event()
        recording.creating.clear()
        outcomes: list[TestOutcome] = []
        worker = threading.Thread(target=lambda: outcomes.append(_run(backend, tmp_path, "req-2")))
        worker.start()
        _until(recording.creating.is_set, "req-2's create")
        backend.close()
        recording.hold.set()
        worker.join(30)
        assert not worker.is_alive()
        assert outcomes[0].returncode == STOPPED_RETURNCODE
        assert _live(rig) == []
        try:
            rig.provider.connect(one, role="measurement")
        except SandboxGone:
            pass
        else:
            raise AssertionError("the kept sandbox outlived the session")
    finally:
        backend.close()
        rig.provider.close()


def test_the_next_sandbox_waits_for_the_runner_to_exit_after_the_result_landed(tmp_path) -> None:
    """A runner whose exit is reported late (the sandbox still sees it) holds the next create back until it is."""
    created_at: list[float] = []
    reported: list[float] = []
    landed: list[float] = []
    rig, recording, backend = _backend(tmp_path, lambda: created_at.append(time.monotonic()))

    def late(real: Callable[[], int | None]) -> int | None:
        record = backend.journal.latest("req-1")
        if record is None or record.state != STATE_FINISHED:
            return None
        landed.append(landed[0] if landed else time.monotonic())
        if time.monotonic() - landed[0] < 0.5:
            return None
        code = real()
        if code is not None and not reported:
            reported.append(time.monotonic())
        return code

    recording.exit_code = late
    try:
        backend.pool.note_upcoming({"req-1", "req-2"})
        assert _run(backend, tmp_path, "req-1").returncode == 0
        assert backend.pool.parked is not None
        assert _run(backend, tmp_path, "req-2").returncode == 0
        assert reported and created_at[1] >= reported[0], "req-2's create began after req-1's runner was seen to exit"
    finally:
        backend.close()
        rig.provider.close()


def test_a_runner_that_does_not_exit_after_its_result_landed_releases_its_sandbox(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(hosted, "_RUNNER_EXIT_WAIT_SEC", 0.3)
    rig, recording, backend = _backend(tmp_path)
    recording.exit_code = lambda real: None  # the sandbox can tell, and the runner never exits
    try:
        backend.pool.note_upcoming({"req-1", "req-2"})
        outcomes: list[TestOutcome] = []
        with caplog.at_level(logging.INFO, logger="hyperloom.measure.hosted"):
            worker = threading.Thread(target=lambda: outcomes.append(_run(backend, tmp_path, "req-1")), daemon=True)
            worker.start()
            worker.join(30)
            assert not worker.is_alive(), "the test waited for its runner to exit past the bound"
        assert outcomes[0].returncode == 0
        assert backend.pool.parked is None and _live(rig) == []
        assert "its runner has not exited" in caplog.text
    finally:
        backend.close()
        rig.provider.close()


@pytest.mark.parametrize("how", ["the test", "the run"])
def test_a_test_stopped_after_its_result_landed_releases_its_sandbox(tmp_path, how: str) -> None:
    stopping = threading.Event()
    rig, recording, backend = _backend(tmp_path, should_stop=stopping.is_set)
    land = backend._land

    def land_then_stop(attempt: Any, verified: Any) -> None:
        land(attempt, verified)
        (attempt.stop if how == "the test" else stopping).set()

    backend._land = land_then_stop  # type: ignore[method-assign]
    try:
        backend.pool.note_upcoming({"req-1", "req-2"})
        assert _run(backend, tmp_path, "req-1").returncode == 0, "the result had landed"
        assert backend.pool.parked is None and _live(rig) == [], "a stopping test keeps no sandbox"
    finally:
        backend.close()
        rig.provider.close()
