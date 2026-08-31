# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for cancellation-safe subprocess lifecycle handling."""

from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest

from kernelforge.llm.process_reaping import install_child_subreaper
from kernelforge.mcp_server.tools._subprocess import communicate_process_group


def _cancel_mid_communicate() -> int:
    """Start a group of two, cancel the communicate, answer the group's pgid.

    The inner process is what makes this a group rather than a process: it
    outlives the ``communicate`` and is only reached through the ``killpg``
    that the cancellation path is supposed to send.
    """

    async def _run() -> int:
        script = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        task = asyncio.create_task(
            communicate_process_group(proc, timeout=60),
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            result = await task
            pytest.fail(f"cancelled task returned unexpectedly: {result!r}")
        assert proc.returncode is not None
        return proc.pid

    return asyncio.run(_run())


def _assert_group_gone(pgid: int) -> None:
    """Fail unless the whole process group has left the process table."""
    for _ in range(20):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        asyncio.run(asyncio.sleep(0.05))
    os.killpg(pgid, signal.SIGKILL)
    pytest.fail("cancelled subprocess group still exists")


def test_cancelled_communicate_kills_and_reaps_process_group():
    _assert_group_gone(_cancel_mid_communicate())


def test_the_group_is_gone_even_once_this_process_is_a_subreaper():
    """The same case, in a process that has taken on orphans.

    ``PR_SET_CHILD_SUBREAPER`` is per-process and permanent, so one call
    anywhere in a pytest worker changes this test for the rest of its session:
    the inner process reparents here instead of to init when ``killpg`` takes
    its parent. A zombie still occupies its process group, so a campaign that
    inherits an orphan and never waits on it leaves ``killpg`` answering "still
    there" for a group in which nothing is left to kill. That is what turned
    this module red on CI, and which worker ran which test first is a sharding
    accident rather than something to rely on -- so the ordering is asserted
    here instead.
    """
    if not install_child_subreaper():
        pytest.skip("this kernel does not support PR_SET_CHILD_SUBREAPER")

    _assert_group_gone(_cancel_mid_communicate())
