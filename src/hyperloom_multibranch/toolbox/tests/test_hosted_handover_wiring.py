# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted toolbox gives its sandbox pool what make-before-break needs: the grant to count the room in, what the
branch leases hold of it, and, after every change of the queue, which tests wait to start or run."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

from hyperloom.measure.sandboxes import Handover
from hyperloom_multibranch.toolbox.hosted_leases import GrantLimits
from hyperloom_multibranch.toolbox.hosted_toolbox import handover_for
from hyperloom_multibranch.toolbox.queue import QueuedTest, State
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_hosted_toolbox import _GRANT, _toolbox


def test_the_backend_counts_the_grant_and_what_the_leases_hold(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3, "max_gpus": 2})
    handover = toolbox.backend._config.handover
    assert handover == Handover(3, 2), "the grant of a hosted submission"
    assert toolbox.leases is not None and handover.reserved == toolbox.leases.reserved
    assert toolbox.backend.pool._handover is handover
    toolbox.leases._promised["b1"] = 1  # a branch's dev sandbox holds a card of the grant
    assert handover.reserved() == (1, 1)


def test_without_grant_limits_no_sandbox_is_kept(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits=None)
    assert toolbox.backend._config.handover is None
    grant = GrantLimits(max_sandboxes=3, max_gpus=2, gpu_hours=1.0, max_runtime_sec=60)
    assert handover_for(grant, None) == Handover(3, 2) and handover_for(grant, None).reserved() == (0, 0)


async def test_the_pool_hears_which_tests_wait_or_run_after_every_change_of_the_queue(tmp_path: Path) -> None:
    toolbox = _toolbox(None, tmp_path, grant_limits={**_GRANT, "max_sandboxes": 3, "max_gpus": 1})  # one at a time
    heard: list[frozenset[str]] = []
    toolbox.backend.pool.note_upcoming = heard.append  # type: ignore[method-assign]
    blocked = asyncio.Event()

    async def run(item: QueuedTest) -> None:
        await blocked.wait()

    toolbox._run = run  # type: ignore[method-assign]
    for seq, request_id in enumerate(("r1", "r2", "r3")):
        state = State.PENDING_BUILD if request_id == "r3" else State.QUEUED
        toolbox._queue.push(QueuedTest(dataclasses.replace(req(), request_id=request_id), 2, seq, state=state))
    toolbox._pump()
    assert heard[-1] == {"r1", "r2"}, "r1 runs and r2 waits; r3 waits for its builds, not for a sandbox"
    toolbox._queue.remove("r1")
    toolbox._pump()
    assert heard[-1] == {"r2"}
    toolbox._queue.remove("r2")
    toolbox._pump()
    assert heard[-1] == frozenset()
    blocked.set()
    await asyncio.gather(*toolbox._tasks.values())
