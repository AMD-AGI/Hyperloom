# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's last BASELINE test tops the root's short-prompt pool up to the pack's ``processes`` (#146 c3217).

The floor the rule gate holds every candidate to is the root's inconsistency, and a merge test or a final candidate is
read over the pack's six processes; the root's BASELINE tests add one server each, so the last of them boots the rest, once
per session. A campaign that submits no root test (AgentX, ``root_tests`` 0) changes nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestRequest
from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup, EarlierBaselines
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup, short_prompt_rules
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, ScriptedExecutor, build, req, settle

SETUP = ShortPromptSetup("short-prompts-test", "a" * 64)
ROOT_TESTS = 3
BASELINE = req(purpose=Purpose.BASELINE)
OTHER_ROOT = "c1"


class _Planner:
    blobs = None

    def __init__(self, tmp_path: Path) -> None:
        self.point = tmp_path / "point.yaml"

    def plan(self, request, parent) -> AcceptedMeasurement:  # type: ignore[no-untyped-def]
        return AcceptedMeasurement(None, (self.point,))  # type: ignore[arg-type]


def _processes(tmp_path: Path, request, *, root_tests: int, earlier: EarlierBaselines, pooled: int) -> int:
    acceptance = Acceptance(
        AcceptanceSetup(FakeRepo(), _Planner(tmp_path), IMAGE, short_prompts=SETUP, root_tests=root_tests),  # type: ignore[arg-type]
        is_accepted=lambda _id: True,
        pooled=lambda key: pooled,
        earlier_baselines=lambda _request, _digest: earlier,
    )
    plan = asyncio.run(acceptance.prepare(request)).environment.short_prompts
    assert plan is not None
    return plan.processes


def test_the_pack_asks_for_six_processes():
    assert short_prompt_rules().processes == 6, "the counts below are written for the pack's six"


@pytest.mark.parametrize(
    ("root_tests", "accepted", "unfinished", "pooled", "processes"),
    [
        (ROOT_TESTS, 2, 2, 0, 4),
        (ROOT_TESTS, 2, 0, 2, 4),
        (ROOT_TESTS, 2, 1, 1, 4),
        (ROOT_TESTS, 2, 0, 1, 5),
        (ROOT_TESTS, 2, 0, 5, 1),
        (ROOT_TESTS, 2, 0, 9, 1),
        (ROOT_TESTS, 0, 0, 0, 1),
        (ROOT_TESTS, 1, 1, 0, 1),
        (ROOT_TESTS, 3, 3, 0, 1),
        (0, 2, 2, 0, 1),
        (1, 0, 0, 0, 6),
    ],
    ids=[
        "last-two-running",
        "last-two-pooled",
        "last-one-of-each",
        "last-one-failed",
        "last-pool-one-short-of-own",
        "last-pool-full",
        "first",
        "second",
        "fourth",
        "no-root-tests-agentx",
        "one-root-test",
    ],
)
def test_only_the_last_root_test_tops_the_root_pool_up_to_the_packs_count(
    tmp_path, root_tests, accepted, unfinished, pooled, processes
):
    request = req(purpose=Purpose.BASELINE)

    found = _processes(
        tmp_path, request, root_tests=root_tests, earlier=EarlierBaselines(accepted, unfinished), pooled=pooled
    )

    assert found == processes


@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.HEALTH])
def test_a_test_of_another_purpose_never_tops_up_whatever_the_log_says(tmp_path, purpose):
    request = req(purpose=purpose, consistency_check=True)

    found = _processes(tmp_path, request, root_tests=ROOT_TESTS, earlier=EarlierBaselines(2, 2), pooled=0)

    assert found == 1, "only merge tests, final candidates and the root's last test boot fresh processes"


def _plans(executor: ScriptedExecutor) -> list[int]:
    """The processes each started test's short-prompt plan asks for, 0 for a test that runs no check."""
    plans = [job.environment.short_prompts for job in executor.jobs]  # type: ignore[union-attr]
    return [0 if plan is None else plan.processes for plan in plans]


def _repo() -> FakeRepo:
    """The root ``PARENT`` and another root, ``OTHER_ROOT``, whose recipe is another."""
    repo = FakeRepo()
    repo.recipes[OTHER_ROOT] = Recipe(base_image_digest=IMAGE, launch=LaunchConfig("--max-running-requests 64", {}))
    return repo


async def _submitted(
    tmp_path: Path, requests: list[TestRequest], *, root_tests: int, restart: bool = False
) -> list[int]:
    """Submit ``requests`` to a toolbox whose executor holds every test running; with ``restart``, the toolbox is closed
    and a new one recovers them. The processes each one's plan asks for, in the order they started."""
    executor = ScriptedExecutor(gate=asyncio.Event())
    options = {"planner": _Planner(tmp_path), "short_prompts": SETUP, "root_tests": root_tests, "repo": _repo()}
    rig = build(tmp_path, executor=executor, **options)  # type: ignore[arg-type]
    await rig.toolbox.start()
    for request in requests:
        await rig.toolbox.submit_test(request)
    await settle()
    if restart:
        await rig.toolbox.close()
        executor = ScriptedExecutor(gate=asyncio.Event())
        rig = build(tmp_path, executor=executor, **options)  # type: ignore[arg-type]
        await rig.toolbox.start()
        await settle()
    try:
        return _plans(executor)
    finally:
        executor.gate.set()  # type: ignore[union-attr]
        await rig.toolbox.drain()
        await rig.toolbox.close()


async def test_the_toolbox_counts_the_root_tests_it_accepted_and_the_last_boots_what_the_pool_lacks(tmp_path):
    other = replace(BASELINE, parent_commit=OTHER_ROOT)
    requests = [req(purpose=Purpose.HEALTH), other, other] + [BASELINE] * (ROOT_TESTS + 1)

    processes = await _submitted(tmp_path, requests, root_tests=ROOT_TESTS)

    assert processes == [0, 1, 1, 1, 1, 4, 1], (
        "a health test of the root and the tests of another recipe are not its root tests; two running servers and "
        "the last one's own and three fresh make six, once"
    )


async def test_a_restarted_toolbox_plans_the_recovered_root_tests_as_the_one_before_it_did(tmp_path):
    processes = await _submitted(tmp_path, [BASELINE] * ROOT_TESTS, root_tests=ROOT_TESTS, restart=True)

    assert processes == [1, 1, 4], "each recovered test counts the ones accepted before it, not the whole log"


async def test_a_toolbox_with_no_root_tests_boots_no_fresh_process_for_a_baseline(tmp_path):
    processes = await _submitted(tmp_path, [BASELINE] * ROOT_TESTS, root_tests=0)

    assert processes == [1, 1, 1]
