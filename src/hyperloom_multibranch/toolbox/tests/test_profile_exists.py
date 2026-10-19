# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``profile_exists``: a PROFILE of a commit and recipe that already has a live one is refused at acceptance.

Branches profiled the same commit more than once (a profile is not a claimed test, so nothing deduped it) and every
repeat held a card. The toolbox now refuses the repeat and names the request to wait for. All of it goes through the
entry points: ``RealToolbox.submit_test`` and the MCP tools a specialist calls.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, CancelOutcome, RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.mcp_tools import SpecialistToolOptions, register_specialist_tools
from hyperloom_multibranch.toolbox.tests.helpers import (
    IMAGE,
    PARENT,
    FakeRepo,
    Rig,
    ScriptedExecutor,
    build,
    delta,
    failed,
    req,
    settle,
)

OTHER_COMMIT = "c1"


def profile(
    branch: str = "b1", *, commit: str = PARENT, change: RecipeDelta | None = None, **fields: Any
) -> TestRequest:
    """A profile of ``commit`` as the recipe it holds (no delta unless ``change`` is given)."""

    request = req(branch, kind=TestKind.PROFILE, purpose=Purpose.PROFILE, change=change or RecipeDelta(), **fields)
    return replace(request, parent_commit=commit)


def gated() -> ScriptedExecutor:
    executor = ScriptedExecutor()
    executor.gate = asyncio.Event()
    return executor


def with_second_commit(rig: Rig) -> Rig:
    rig.repo.recipes[OTHER_COMMIT] = Recipe(base_image_digest=IMAGE)
    return rig


async def refused(rig: Rig, request: TestRequest) -> RequestRejected:
    with pytest.raises(RequestRejected) as error:
        await rig.toolbox.submit_test(request)
    assert error.value.code == "profile_exists"
    return error.value


def test_profile_exists_is_in_the_closed_refusal_vocabulary() -> None:
    assert "profile_exists" in REJECTION_CODES


async def test_a_second_profile_of_the_same_commit_is_refused_with_the_first_ones_id(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=gated())
    first = await rig.toolbox.submit_test(profile("b1"))
    await settle()  # the first one is running, the repeat comes from another branch in the same second

    rejection = await refused(rig, profile("b2"))

    assert rejection.existing_request_id == first and first in rejection.reason
    assert [entry.request_id for entry in rig.toolbox._log.entries()] == [first], "the repeat is not even logged"
    assert list(rig.notebook._state().requests) == [first], "and it writes nothing to the notebook"
    state = await rig.toolbox.queue_state()
    assert (state.queued, state.running) == (0, 1)
    assert len(rig.planner.calls) == 1, "nothing was compiled for the refused one"
    await rig.toolbox.close()


async def test_a_queued_profile_covers_the_commit_too(tmp_path: Path) -> None:
    from hyperloom_multibranch.toolbox.queue import QueuePolicy

    rig = build(tmp_path, executor=gated(), policy=QueuePolicy(per_branch_k=1, global_concurrency=1))
    await rig.toolbox.submit_test(req("b3"))  # holds the only slot
    first = await rig.toolbox.submit_test(profile("b1"))
    await settle()
    assert (await rig.toolbox.queue_state()).queued == 1

    assert (await refused(rig, profile("b1"))).existing_request_id == first
    await rig.toolbox.close()


async def test_a_profile_that_measured_successfully_still_covers_its_commit(tmp_path: Path) -> None:
    rig = build(tmp_path)
    first = await rig.toolbox.submit_test(profile("b1"))
    await rig.toolbox.drain()
    assert (await rig.toolbox.outcome(first)).request_id == first

    assert (await refused(rig, profile("b2"))).existing_request_id == first
    await rig.toolbox.close()


@pytest.mark.parametrize("failure", [FailureClass.CANDIDATE, FailureClass.INFRA], ids=["candidate", "infra"])
async def test_a_profile_that_failed_does_not_block_a_new_one(tmp_path: Path, failure: FailureClass) -> None:
    rig = build(tmp_path, executor=ScriptedExecutor(script=[lambda job: failed(job, failure)]))
    first = await rig.toolbox.submit_test(profile("b1"))
    await rig.toolbox.drain()
    assert (await rig.toolbox.outcome(first)).validity.value != "succeeded"

    second = await rig.toolbox.submit_test(profile("b2"))
    assert second != first
    await rig.toolbox.close()


async def test_a_withdrawn_profile_does_not_block_a_new_one(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=gated())
    first = await rig.toolbox.submit_test(profile("b1"))
    await settle()
    assert (await rig.toolbox.cancel(first)) is CancelOutcome.WITHDRAWN

    second = await rig.toolbox.submit_test(profile("b2"))
    assert second != first
    await refused(rig, profile("b3"))  # and the new one covers the commit again
    await rig.toolbox.close()


async def test_another_commit_another_recipe_and_other_kinds_are_accepted(tmp_path: Path) -> None:
    rig = with_second_commit(build(tmp_path, executor=gated()))
    first = await rig.toolbox.submit_test(profile("b1"))

    accepted = [
        await rig.toolbox.submit_test(profile("b2", commit=OTHER_COMMIT)),
        await rig.toolbox.submit_test(profile("b2", change=delta("--other"))),
        await rig.toolbox.submit_test(req("b2", change=RecipeDelta())),  # a test of the same commit is no profile
    ]

    assert len({first, *accepted}) == 4
    await rig.toolbox.close()


async def test_asking_for_another_concurrency_does_not_make_it_another_profile(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=gated())
    first = await rig.toolbox.submit_test(profile("b1"))

    assert (await refused(rig, profile("b2", concurrency=(32,)))).existing_request_id == first, (
        "a profile runs the one canonical point whatever concurrency it names"
    )
    await rig.toolbox.close()


class YieldingRepo(FakeRepo):
    """Reading a recipe hands the loop to the other submits, as a repository on disk does."""

    async def read_recipe(self, commit: str) -> Recipe:
        await asyncio.sleep(0)
        return await super().read_recipe(commit)


async def test_profiles_submitted_in_the_same_instant_are_accepted_exactly_once(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=gated(), repo=YieldingRepo())

    outcomes = await asyncio.gather(
        *(rig.toolbox.submit_test(profile(f"b{n}")) for n in range(1, 7)), return_exceptions=True
    )

    accepted = [outcome for outcome in outcomes if isinstance(outcome, str)]
    rejections = [outcome for outcome in outcomes if isinstance(outcome, RequestRejected)]
    assert len(accepted) == 1 and len(rejections) == 5, outcomes
    assert {rejection.code for rejection in rejections} == {"profile_exists"}
    assert {rejection.existing_request_id for rejection in rejections} == set(accepted)
    assert [entry.request_id for entry in rig.toolbox._log.entries()] == accepted
    await rig.toolbox.close()


async def test_a_queued_profile_still_covers_its_commit_after_a_restart(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=gated())
    first = await rig.toolbox.submit_test(profile("b1"))
    await settle()
    await rig.toolbox.close()  # the process dies with the profile running

    again = build(tmp_path, executor=gated())
    await again.toolbox.start()

    assert (await refused(again, profile("b2"))).existing_request_id == first
    await again.toolbox.close()


async def test_a_measured_profile_still_covers_its_commit_after_a_restart(tmp_path: Path) -> None:
    rig = build(tmp_path)
    first = await rig.toolbox.submit_test(profile("b1"))
    await rig.toolbox.drain()
    await rig.toolbox.close()

    again = build(tmp_path)
    await again.toolbox.start()

    assert (await refused(again, profile("b2"))).existing_request_id == first
    await again.toolbox.close()


@pytest.mark.parametrize("how", ["failed", "withdrawn"])
async def test_a_failed_or_withdrawn_profile_does_not_cover_its_commit_after_a_restart(
    tmp_path: Path, how: str
) -> None:
    if how == "failed":
        rig = build(tmp_path, executor=ScriptedExecutor(script=[lambda job: failed(job, FailureClass.CANDIDATE)]))
    else:
        rig = build(tmp_path, executor=gated())
    first = await rig.toolbox.submit_test(profile("b1"))
    await settle()
    if how == "withdrawn":
        await rig.toolbox.cancel(first)
    else:
        await rig.toolbox.drain()
    await rig.toolbox.close()

    again = build(tmp_path)
    await again.toolbox.start()

    assert await again.toolbox.submit_test(profile("b2")) != first
    await again.toolbox.close()


# --- the MCP tool a specialist calls ----------------------------------------------------------------------------------


@dataclass
class Stack:
    rig: Rig
    server: McpServer
    executor: ScriptedExecutor

    def call(self, branch: str, name: str, /, **arguments: Any) -> dict[str, Any]:
        token = self.server.issue_token(f"specialist-{branch}", branch)
        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        headers = {"Content-Type": "application/json", "X-Hyperloom-Token": token}
        request = urllib.request.Request(self.server.url, json.dumps(message).encode(), headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    async def tool(self, branch: str, name: str, /, **arguments: Any) -> tuple[bool, str]:
        """``(is_error, text)`` of a tool call; a protocol error is returned as its message."""

        reply = await asyncio.to_thread(self.call, branch, name, **arguments)
        if "error" in reply:
            return True, str(reply["error"]["message"])
        return reply["result"]["isError"], str(reply["result"]["content"][0]["text"])


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    executor = gated()
    rig = build(tmp_path, executor=executor)
    await rig.toolbox.start()
    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w62")

    def notebooks(branch: str) -> LocalNotebook:
        return LocalNotebook(tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, branch, branch))

    options = SpecialistToolOptions(toolbox_loop=asyncio.get_running_loop(), wait_poll_seconds=0.05)
    register_specialist_tools(server, toolbox=rig.toolbox, notebooks=notebooks, options=options)
    server.start()
    try:
        yield Stack(rig, server, executor)
    finally:
        server.stop()
        await rig.toolbox.close()


_PROFILE_ARGS = {"kind": "profile", "purpose": "profile", "parent_commit": PARENT}


async def test_submit_test_refuses_a_second_profile_and_returns_the_existing_id(stack: Stack) -> None:
    is_error, first = await stack.tool("b1", "submit_test", **_PROFILE_ARGS)
    assert not is_error
    first_id = json.loads(first)["request_id"]

    is_error, text = await stack.tool("b2", "submit_test", **_PROFILE_ARGS)

    assert is_error and text.startswith("rejected (profile_exists):")
    assert first_id in text and "wait_test" in text and "op shares" in text
    assert list(stack.rig.notebook._state().requests) == [first_id]


async def test_the_refused_branch_can_wait_for_the_existing_profile_but_not_cancel_it(stack: Stack) -> None:
    _, first = await stack.tool("b1", "submit_test", **_PROFILE_ARGS)
    first_id = json.loads(first)["request_id"]
    await settle()

    is_error, waiting = await stack.tool("b2", "wait_test", request_id=first_id, wait_seconds=1)
    assert not is_error and json.loads(waiting)["state"] == "running"

    is_error, forbidden = await stack.tool("b2", "cancel_test", request_id=first_id)
    assert is_error and "own branch only" in forbidden

    stack.executor.gate.set()  # type: ignore[union-attr]
    await stack.rig.toolbox.drain()
    is_error, over = await stack.tool("b2", "wait_test", request_id=first_id)
    assert not is_error and json.loads(over)["state"] == "measured"
    is_error, read = await stack.tool("b2", "test_result", request_id=first_id)
    assert not is_error and json.loads(read)["result"]["request_id"] == first_id


async def test_another_branchs_other_tests_stay_unreadable(stack: Stack) -> None:
    _, submitted = await stack.tool("b1", "submit_test", kind="test", purpose="explore", parent_commit=PARENT)
    test_id = json.loads(submitted)["request_id"]

    for tool in ("wait_test", "test_result"):
        is_error, text = await stack.tool("b2", tool, request_id=test_id)
        assert is_error and "own branch only" in text
