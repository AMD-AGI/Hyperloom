# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind
from hyperloom.contracts.recipe import ComponentChange, ComponentPatches, PatchRef, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind
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


async def test_a_profile_test_that_asks_for_accuracy_is_refused_and_never_queues(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE, accuracy=True), "invalid_request")
    assert not list(rig.toolbox._log.entries())


async def test_a_repro_test_that_asks_for_accuracy_is_refused_and_never_queues(tmp_path: Path) -> None:
    rig = build(tmp_path)
    case = AttachmentRef("case-1", "dir", "a" * 64)
    request = req(kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(case,), accuracy=True)
    await _rejected(rig, request, "invalid_request")
    assert not list(rig.toolbox._log.entries())


async def test_a_repro_test_is_accepted_without_accuracy_whatever_the_workload(tmp_path: Path) -> None:
    rig = build(tmp_path)  # an InferenceX campaign: a repro is not a purpose the gate reads accuracy from
    case = AttachmentRef("case-1", "dir", "a" * 64)
    request_id = await rig.toolbox.submit_test(req(kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(case,)))
    await rig.toolbox.drain()
    ran = next(j for j in rig.executor.jobs if j.request.request_id == request_id)
    assert ran.request.kind is TestKind.REPRO and ran.request.accuracy is False and ran.request.attachments == (case,)


async def test_a_profile_test_is_accepted_without_accuracy_whatever_the_workload(tmp_path: Path) -> None:
    rig = build(tmp_path)  # an InferenceX campaign: the gate-read purposes get accuracy, a profile does not
    request_id = await rig.toolbox.submit_test(req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE))
    await rig.toolbox.drain()
    ran = next(j for j in rig.executor.jobs if j.request.request_id == request_id)
    assert ran.request.kind is TestKind.PROFILE and ran.request.accuracy is False


# --- an InferenceX campaign runs the purposes the rule gate reads with accuracy (board topic 14) -------------------------------


async def _accepted_accuracy(rig, request) -> tuple[bool, bool]:
    """(the accuracy flag the submit log recorded, the flag the executor received)."""

    request_id = await rig.toolbox.submit_test(request)
    await rig.toolbox.drain()
    logged = next(e for e in rig.toolbox._log.entries() if e.request_id == request_id)
    ran = next(j for j in rig.executor.jobs if j.request.request_id == request_id)
    return bool(logged.request["accuracy"]), ran.request.accuracy


async def test_an_inferencex_explore_is_accepted_with_accuracy(tmp_path: Path) -> None:
    rig = build(tmp_path)
    assert await _accepted_accuracy(rig, req(purpose=Purpose.EXPLORE)) == (True, True)


async def test_an_inferencex_confirm_is_accepted_with_accuracy(tmp_path: Path) -> None:
    rig = build(tmp_path)
    first = await rig.toolbox.submit_test(req())
    assert await _accepted_accuracy(rig, req(purpose=Purpose.CONFIRM, retest_of=first)) == (True, True)


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.INTEGRATION, Purpose.FINAL])
async def test_an_inferencex_gate_read_purpose_is_accepted_with_accuracy(tmp_path: Path, purpose: Purpose) -> None:
    rig = build(tmp_path)
    assert await _accepted_accuracy(rig, req(purpose=purpose)) == (True, True)


@pytest.mark.parametrize("purpose", [Purpose.HEALTH, Purpose.REPRO, Purpose.PROFILE])
async def test_an_inferencex_purpose_the_gate_does_not_read_keeps_the_flag_off(
    tmp_path: Path, purpose: Purpose
) -> None:
    rig = build(tmp_path)
    assert await _accepted_accuracy(rig, req(purpose=purpose)) == (False, False)


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.INTEGRATION, Purpose.FINAL])
async def test_an_agentx_gate_read_purpose_is_left_as_submitted(tmp_path: Path, purpose: Purpose) -> None:
    rig = build(tmp_path, workload_kind=WorkloadKind.AGENTX)
    assert await _accepted_accuracy(rig, req(purpose=purpose)) == (False, False)


async def test_an_agentx_explore_is_left_as_submitted(tmp_path: Path) -> None:
    rig = build(tmp_path, workload_kind=WorkloadKind.AGENTX)
    assert await _accepted_accuracy(rig, req(purpose=Purpose.EXPLORE)) == (False, False)
    assert await _accepted_accuracy(rig, req(purpose=Purpose.EXPLORE, accuracy=True)) == (True, True)


async def test_an_explore_with_an_empty_delta_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req(change=RecipeDelta()), "empty_delta")


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.CONFIRM])
async def test_a_baseline_or_a_confirm_may_carry_an_empty_delta(tmp_path: Path, purpose: Purpose) -> None:
    rig = build(tmp_path)
    first = await rig.toolbox.submit_test(req())
    request = req(purpose=purpose, change=RecipeDelta(), **({"retest_of": first} if purpose is Purpose.CONFIRM else {}))
    assert await rig.toolbox.submit_test(request)


SGLANG_REPO = "https://github.com/sgl-project/sglang"
SGLANG_REF = "9f5d8712" + "0" * 32
OTHER_REF = "e8b97b24" + "0" * 32
PATCH = PatchRef("patch/sglang/0001-fuse.patch", "d" * 64)


def _with_sglang() -> FakeRepo:
    return FakeRepo(Recipe(base_image_digest=IMAGE, components=(ComponentPatches("sglang", SGLANG_REPO, SGLANG_REF),)))


@pytest.mark.parametrize(
    "change",
    [
        ComponentChange("sglang", upstream_ref=OTHER_REF),
        ComponentChange("sglang", upstream_ref=OTHER_REF, upstream_repo=SGLANG_REPO, add_patches=(PATCH,)),
        ComponentChange("sglang", upstream_repo="https://github.com/someone/sglang"),
        ComponentChange("sglang", SGLANG_REF, "https://github.com/someone/sglang"),
    ],
    ids=["ref", "ref_and_patch", "repo", "repo_with_equal_ref"],
)
async def test_a_delta_that_moves_the_upstream_of_a_component_the_parent_has_is_refused(
    tmp_path: Path, change: ComponentChange
) -> None:
    rig = build(tmp_path, repo=_with_sglang())
    with pytest.raises(RequestRejected, match="installed package") as refused:
        await rig.toolbox.submit_test(req(change=RecipeDelta(components=(change,))))
    assert refused.value.code == "upstream_label_unsupported" and "sglang" in refused.value.reason
    state = await rig.toolbox.queue_state()
    assert state.queued == 0 and state.running == 0
    assert rig.notebook._state().requests == {} and rig.toolbox._log.entries() == []
    assert rig.planner.calls == []


@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.INTEGRATION])
async def test_a_delta_that_restates_the_parents_upstream_is_accepted_unchanged(
    tmp_path: Path, purpose: Purpose
) -> None:
    rig = build(tmp_path, repo=_with_sglang())
    change = RecipeDelta(components=(ComponentChange("sglang", SGLANG_REF, SGLANG_REPO, add_patches=(PATCH,)),))
    request_id = await rig.toolbox.submit_test(req(purpose=purpose, change=change))
    assert rig.notebook._state().requests[request_id].purpose == purpose.value
    assert rig.planner.calls == [change], "the delta runs as submitted, so its fingerprint matches the claim's"


async def test_a_component_the_parent_lacks_is_still_accepted_with_its_repository_and_commit(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=_with_sglang())
    triton = RecipeDelta(components=(ComponentChange("triton", OTHER_REF, "https://github.com/triton-lang/triton"),))
    assert await rig.toolbox.submit_test(req(change=triton))
    assert rig.planner.calls == [triton]


async def test_a_component_the_parent_lacks_still_needs_its_repository_and_commit(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=_with_sglang())
    bare = RecipeDelta(components=(ComponentChange("triton", add_patches=(PATCH,)),))
    await _rejected(rig, req(change=bare), "delta_not_applicable")
