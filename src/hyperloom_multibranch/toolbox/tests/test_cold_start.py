# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox records the cold start each test measured, so the queue reports the recent one."""

from __future__ import annotations

from pathlib import Path

from hyperloom_multibranch.toolbox.tests.helpers import ScriptedExecutor, build, ok, req


async def test_the_cold_start_a_test_measured_reaches_the_queue(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=ScriptedExecutor([lambda job: ok(job, cold_start_seconds=82.2)]))
    assert (await rig.toolbox.queue_state()).cold_start_seconds is None

    await rig.toolbox.submit_test(req())
    await rig.toolbox.drain()

    assert (await rig.toolbox.queue_state()).cold_start_seconds == 82.2


async def test_the_latest_cold_start_wins_and_a_test_that_measured_none_leaves_it(tmp_path: Path) -> None:
    measured = iter([82.2, 90.0, None])  # the third test booted no server
    rig = build(tmp_path, executor=ScriptedExecutor([lambda job: ok(job, cold_start_seconds=next(measured))]))
    seen = []
    for size in (100, 200, 300):
        await rig.toolbox.submit_test(req(server_args=f"--max-num-seqs {size}"))
        await rig.toolbox.drain()
        seen.append((await rig.toolbox.queue_state()).cold_start_seconds)

    assert seen == [82.2, 90.0, 90.0]
