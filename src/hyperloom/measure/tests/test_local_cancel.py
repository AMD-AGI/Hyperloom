# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test the caller ends through its call's cancellation stops then, its server reaped, and its boot limit still holds.

The runner's card watch ends a test this way when a process outside it takes one of its cards. The rounds run a real
child under the launcher's own watchdog, as in ``test_boot_watch``; only ``run_grid`` is a stand-in.
"""

from __future__ import annotations

import dataclasses
import threading
import time

from hyperloom.measure import local
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.returncodes import SERVER_BOOT_TIMEOUT_RC
from hyperloom.measure.tests.test_boot_watch import (  # noqa: F401 - the fixture
    READY_SERVER,
    TUNING_SERVER,
    _call,
    _point,
    _ServerGrid,
    magpie_backend,
)
from hyperloom.orchestrator.actions.cancel_channel import CancelScope
from hyperloom.orchestrator.actions.executors import ORCHESTRATOR_CANCELLED_RETURNCODE

#: A server that is ready at once and then serves for as long as it is let.
SERVING_SERVER = READY_SERVER.replace("time.sleep(2.5)", "time.sleep(300)")


def _run(monkeypatch, tmp_path, script: str, cancel_after: float | None, timeout_sec: int = 30):
    monkeypatch.setattr(local, "run_grid", _ServerGrid(script))
    scope = CancelScope()
    call = dataclasses.replace(_call(tmp_path, _point(tmp_path), timeout_sec=timeout_sec), cancel=scope)
    timer = None
    if cancel_after is not None:
        timer = threading.Timer(cancel_after, lambda: scope.cancel(reason="foreign_process_on_card"))
        timer.start()
    started = time.monotonic()
    try:
        outcome = LocalMeasurementBackend().run_test(call)
    finally:
        if timer is not None:
            timer.cancel()
    return outcome, time.monotonic() - started


def test_cancelling_the_calls_scope_ends_a_running_test_and_reaps_its_server(monkeypatch, tmp_path) -> None:
    outcome, elapsed = _run(monkeypatch, tmp_path, SERVING_SERVER, cancel_after=1.0)

    assert [r.returncode for r in outcome.rounds] == [ORCHESTRATOR_CANCELLED_RETURNCODE]
    assert elapsed < 15, "the test ended at the cancellation, not at its 30 s cap or its server's 300 s"


def test_a_call_that_carries_a_scope_keeps_its_boot_limit(monkeypatch, tmp_path) -> None:
    """The scope the call carries is the one the boot limit cancels; carrying one must not switch the limit off."""
    outcome, elapsed = _run(monkeypatch, tmp_path, TUNING_SERVER, cancel_after=None, timeout_sec=4)

    assert outcome.returncode == SERVER_BOOT_TIMEOUT_RC
    assert outcome.failure_reason is not None and outcome.failure_reason.startswith("server_boot_timeout")
    assert 2 <= elapsed < 20, "ended at the boot limit (2 s of the 4 s cap)"
