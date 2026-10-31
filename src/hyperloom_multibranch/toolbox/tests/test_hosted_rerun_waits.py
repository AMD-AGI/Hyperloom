# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rerun of an unclear failure never waits for its sandbox without a deadline or without saying so (board #156, V6).

V6's three root tests ended ``unclear``; each rerun was to start in a fresh sandbox, but no attempt 2 was ever
journalled and the session said nothing for two hours. Through the hosted toolbox over the E2B stand-in: the platform
does not end the first attempt's sandbox when the toolbox kills it, so it keeps the run's only slot, as a platform that
stopped acting on kills does. The rerun must log that it waits, end when the test's own timeout has passed, journal
why, and end the test with that reason instead of retrying it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Validity
from hyperloom.measure import sandboxes
from hyperloom.measure.failures import SANDBOX_ACQUIRE_TIMEOUT
from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedOptions
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_hosted_toolbox import (  # noqa: F401 - service is a fixture
    _FAST,
    _GRANT,
    _RETRY,
    Service,
    _toolbox,
    service,
)

_TIMEOUT_SEC = 8


async def test_a_rerun_whose_sandbox_cannot_start_ends_the_test_with_the_reason_it_waited_for(
    service: Service, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandboxes, "WAIT_LOG_EVERY_SEC", 1.0)
    caplog.set_level(logging.INFO, logger="hyperloom.measure.sandboxes")
    options = HostedOptions(
        policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
        poll_interval_sec=0.05,
        timeout_sec=_TIMEOUT_SEC,
        max_infra_retries=3,  # a retry would wait again: the test must end at the first wait that ran out
        retry=_RETRY,
        lease_tuning=_FAST,
        framework_configs="",
    )
    grant = {**_GRANT, "max_sandboxes": 1}  # one test at a time: the slot the first attempt's sandbox keeps
    toolbox = _toolbox(service, tmp_path, planned={"fake_mode": "fail_unknown"}, grant_limits=grant, options=options)
    service.standin.plane.fail_next("kill", 503, times=100_000)  # kills are refused; the sandbox runs on
    try:
        await toolbox.start()
        started = time.monotonic()
        request_id = await toolbox.submit_test(req())
        await asyncio.wait_for(toolbox.drain(), 3 * _TIMEOUT_SEC + 20)
        took = time.monotonic() - started
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
        result = entry.outcome["result"]
        assert result["validity"] == Validity.MEASUREMENT_FAILED.value, result
        assert result["failure_class"] == FailureClass.INFRA.value, result
        reason = result["failure_reason"]
        assert reason.startswith(f"{SANDBOX_ACQUIRE_TIMEOUT}: "), reason
        assert "slot" in reason and f"{_TIMEOUT_SEC}" in reason, reason
        attempts = [a["failure_class"] for a in result["attempts"]]
        assert attempts == [FailureClass.UNCLEAR.value, FailureClass.INFRA.value], result["attempts"]
        assert took < 2 * _TIMEOUT_SEC + 15, f"the rerun ended {took:.0f}s after the submit"
        ended = [
            r
            for r in toolbox.backend.journal.records()
            if r.request_id == request_id and r.attempt == 2 and r.state == STATE_FINISHED
        ]
        assert [r.extra.get("verdict") for r in ended] == [SANDBOX_ACQUIRE_TIMEOUT], ended
        lines = [r.getMessage() for r in caplog.records if request_id in r.getMessage()]
        assert any("waits for a sandbox slot" in line and "attempt 2" in line for line in lines), lines
        assert any("still waiting for a sandbox slot" in line for line in lines), lines
        measurement_creates = [c for c in service.creates if c.role == "measurement"]
        assert len(measurement_creates) == 1, "attempt 2 never got as far as a create"
    finally:
        await toolbox.close()
