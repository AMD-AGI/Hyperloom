# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose
from hyperloom.contracts.recipe import ComponentChange, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.tests.helpers import BAD_DELTA, IMAGE, OTHER_IMAGE, FakeRepo, build, req


async def _rejected(rig, request, code: str) -> None:
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(request)
    assert refused.value.code == code
    state = await rig.toolbox.queue_state()
    assert state.queued == 0 and state.running == 0
    assert rig.notebook._state().requests == {}, "a refused request must write nothing to the notebook"
    assert rig.toolbox._log.entries() == []


async def test_a_compile_failure_is_refused_and_never_queues(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req(change=BAD_DELTA), "delta_not_applicable")
    assert rig.planner.calls == [BAD_DELTA]


async def test_a_delta_that_does_not_fit_the_parent_recipe_is_refused_before_compiling(tmp_path: Path) -> None:
    rig = build(tmp_path)
    missing_component = RecipeDelta(components=(ComponentChange("sglang"),))
    await _rejected(rig, req(change=missing_component), "delta_not_applicable")
    assert rig.planner.calls == []


async def test_a_surface_mismatch_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    rig.planner.mismatch = "isl 2048 != 1024"
    with pytest.raises(RequestRejected, match="isl 2048") as refused:
        await rig.toolbox.submit_test(req())
    assert refused.value.code == "surface_mismatch"
    assert rig.notebook._state().requests == {}


async def test_an_unknown_parent_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request = req()
    request = type(request)(**{**request.__dict__, "parent_commit": "nope"})
    await _rejected(rig, request, "unknown_parent")


@pytest.mark.parametrize("field", ["base", "snapshot"])
async def test_a_recipe_on_another_base_image_is_refused(tmp_path: Path, field: str) -> None:
    recipe = (
        Recipe(base_image_digest=OTHER_IMAGE)
        if field == "base"
        else Recipe(base_image_digest=IMAGE, runtime_snapshot=OTHER_IMAGE)
    )
    rig = build(tmp_path, repo=FakeRepo(recipe))
    await _rejected(rig, req(), "base_image_mismatch")


async def test_a_retest_of_an_unknown_request_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req(purpose=Purpose.CONFIRM, retest_of="not-a-request"), "unknown_retest")


async def test_a_request_that_already_has_an_id_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req().accepted("mine"), "invalid_request")


async def test_a_valid_request_is_accepted_and_written_to_the_notebook(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request_id = await rig.toolbox.submit_test(
        req(purpose=Purpose.CONFIRM, retest_of=await rig.toolbox.submit_test(req()))
    )
    row = rig.notebook._state().requests[request_id]
    assert (row.request_id, row.branch, row.purpose, row.priority) == (request_id, "b1", "confirm", 0)
    assert row.retest_of is not None
