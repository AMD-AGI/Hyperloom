# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio

import pytest

from hyperloom.contracts.measurement import (
    NOT_REPRODUCED,
    REPRODUCED,
    AttachmentRef,
    Purpose,
    TestKind,
    Validity,
)
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta
from hyperloom_multibranch.rulegate.regression import RESUBMIT_SECONDS, ReproRun, ReproRunner
from hyperloom_multibranch.rulegate.service import Polling
from hyperloom_multibranch.tests.fakes import RecordingVerifier, ReproToolbox

OVERFLOW = (AttachmentRef("fp8_overflow", "dir", "b" * 64),)
FIX = delta("--kv-cache-dtype fp8 --clamp-scales")


class _Rig:
    """A toolbox that runs repro tests, writing to a local notebook, and a runner reading it as the gate."""

    def __init__(self, root, answer=lambda request: NOT_REPRODUCED, verifier=None):
        self.toolbox = ReproToolbox(LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox")), answer)
        self.notebook = LocalNotebook(root, Writer(Role.GATE, "gate"))
        self.verifier = verifier or RecordingVerifier()
        self.sleeps: list[float] = []
        self.runner = ReproRunner(self.toolbox, self.notebook, self.verifier, polling=Polling(0.5, self._sleep))

    async def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)


async def test_a_case_runs_as_a_repro_test_and_its_verified_verdict_comes_back(tmp_path):
    rig = _Rig(tmp_path, lambda request: REPRODUCED)

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert run == ReproRun("p1", verdict=REPRODUCED)
    (submitted,) = rig.toolbox.submitted
    assert (submitted.kind, submitted.purpose) == (TestKind.REPRO, Purpose.REPRO)
    assert (submitted.branch_id, submitted.parent_commit, submitted.delta) == ("s1", "c1", FIX)
    assert submitted.attachments == OVERFLOW
    assert rig.verifier.asked == ["p1"]


async def test_a_full_queue_is_waited_out(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.busy = 2

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert run.verdict == NOT_REPRODUCED
    assert len(rig.toolbox.submitted) == 1
    assert rig.sleeps == [RESUBMIT_SECONDS, RESUBMIT_SECONDS]


async def test_the_runner_waits_for_the_test_to_end(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.hold = True
    running = asyncio.create_task(rig.runner.run("c1", FIX, OVERFLOW, branch="s1"))
    while not rig.sleeps:
        await asyncio.sleep(0)

    await rig.toolbox.finish()

    assert (await running).verdict == NOT_REPRODUCED
    assert rig.sleeps[0] == 0.5


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        (None, "withdrawn (the branch was stopped)"),
        (Validity.CANDIDATE_FAILED, "ended candidate_failed (the server did not start)"),
        ("", "succeeded without a verdict"),
    ],
)
async def test_a_case_that_ran_to_no_verdict_says_why(tmp_path, answer, failure):
    rig = _Rig(tmp_path, lambda request: answer)

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert run == ReproRun("p1", failure=failure)
    assert run.outcome == failure


async def test_a_verdict_that_does_not_re_verify_is_no_verdict(tmp_path):
    rig = _Rig(tmp_path, verifier=RecordingVerifier({"p1"}))

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert (run.verdict, run.failure) == ("", "p1 did not re-verify: the manifest hash is not the recorded one")


async def test_a_refused_test_has_no_request(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.refuse = lambda request: True

    run = await rig.runner.run("c9", FIX, OVERFLOW, branch="s1")

    assert run == ReproRun("", failure="refused by the toolbox (unknown_parent: c9 is not in the repository)")


async def test_a_cancelled_run_takes_its_test_back(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.hold = True
    running = asyncio.create_task(rig.runner.run("c1", FIX, OVERFLOW, branch="s1"))
    while not rig.sleeps:
        await asyncio.sleep(0)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert rig.toolbox.cancelled == ["p1"]
    assert (await rig.notebook.record("p1")).withdrawal_reason == "cancelled by its caller"
