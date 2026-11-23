# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An explore of a specialist branch is accepted only on that branch's head; every other test names any commit."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected
from hyperloom_multibranch.toolbox.tests.helpers import (
    IMAGE,
    PARENT,
    Rig,
    ScriptedExecutor,
    build,
    req,
    settle,
)

#: The commit branch ``b1`` moved to; ``PARENT`` is the commit it left.
HEAD = "c1"
CODE = "parent_not_head"


def _move(rig: Rig, branch: str = "b1") -> Rig:
    """``branch`` is at ``HEAD``; every other branch is still at ``PARENT``."""
    rig.repo.recipes[HEAD] = Recipe(base_image_digest=IMAGE)
    rig.repo.heads[branch] = HEAD
    return rig


def _on(request: TestRequest, commit: str) -> TestRequest:
    return replace(request, parent_commit=commit)


async def _refusal(rig: Rig, request: TestRequest) -> RequestRejected:
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(request)
    assert rig.toolbox._log.entries() == [] and rig.blackboard._state().requests == {}
    return refused.value


async def test_an_explore_on_a_commit_its_branch_has_left_is_refused_naming_the_head(tmp_path: Path) -> None:
    rig = _move(build(tmp_path))

    refused = await _refusal(rig, req("b1"))

    assert refused.code == CODE and CODE in REJECTION_CODES
    assert HEAD in refused.reason and PARENT in refused.reason
    assert rig.planner.calls == [], "nothing is compiled for a test that cannot land"


async def test_an_explore_on_the_head_its_branch_moved_to_is_accepted(tmp_path: Path) -> None:
    rig = _move(build(tmp_path))

    request_id = await rig.toolbox.submit_test(_on(req("b1"), HEAD))

    assert rig.blackboard._state().requests[request_id].parent_commit == HEAD


async def test_an_explore_of_the_integration_branch_names_a_commit_that_is_not_its_head(tmp_path: Path) -> None:
    rig = _move(build(tmp_path), INTEGRATION_BRANCH)

    request_id = await rig.toolbox.submit_test(req(INTEGRATION_BRANCH))

    assert rig.blackboard._state().requests[request_id].parent_commit == PARENT


async def test_the_confirm_of_an_explore_is_accepted_after_its_branch_moved(tmp_path: Path) -> None:
    rig = build(tmp_path)
    first = await rig.toolbox.submit_test(req("b1"))
    _move(rig)  # a sibling's gain landed while the explore was measured

    retest = await rig.toolbox.submit_test(req("b1", purpose=Purpose.CONFIRM, retest_of=first))

    assert rig.blackboard._state().requests[retest].parent_commit == PARENT


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": TestKind.REPRO, "purpose": Purpose.REPRO, "attachments": (AttachmentRef("case-1", "dir", "a" * 64),)},
        {"purpose": Purpose.HEALTH},
        {"kind": TestKind.ACCURACY, "purpose": Purpose.ACCURACY, "accuracy": True},
    ],
    ids=["repro", "health", "accuracy"],
)
async def test_a_test_that_lands_nothing_names_a_commit_its_branch_has_left(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    rig = _move(build(tmp_path))

    assert await rig.toolbox.submit_test(req("b1", **fields))


async def test_a_restart_runs_the_explore_accepted_before_its_branch_moved(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=ScriptedExecutor(gate=asyncio.Event()))
    request_id = await rig.toolbox.submit_test(req("b1"))
    await settle()
    await rig.toolbox.close()  # the process dies with the explore running
    _move(rig)

    again = build(tmp_path, repo=rig.repo)
    await again.toolbox.start()
    result = await asyncio.wait_for(again.toolbox.outcome(request_id), timeout=5)
    await again.toolbox.close()

    assert result.validity is Validity.SUCCEEDED
