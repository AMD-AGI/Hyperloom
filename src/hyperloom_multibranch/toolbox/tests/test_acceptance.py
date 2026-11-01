# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind
from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    LaunchConfig,
    PatchRef,
    Recipe,
    RecipeDelta,
)
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.tests.helpers import BAD_DELTA, IMAGE, OTHER_IMAGE, FakePlanner, FakeRepo, build, req


async def _rejected(rig, request, code: str) -> None:
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(request)
    assert refused.value.code == code
    state = await rig.toolbox.queue_state()
    assert state.queued == 0 and state.running == 0
    assert rig.blackboard._state().requests == {}, "a refused request must write nothing to the blackboard"
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
    assert rig.blackboard._state().requests == {}


async def test_an_unknown_parent_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request = req()
    request = type(request)(**{**request.__dict__, "parent_commit": "nope"})
    await _rejected(rig, request, "unknown_parent")


async def test_a_recipe_on_another_base_image_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(Recipe(base_image_digest=OTHER_IMAGE, runtime_snapshot=IMAGE)))
    with pytest.raises(RequestRejected, match=f"recipe base image {OTHER_IMAGE} is not the active framework image"):
        await rig.toolbox.submit_test(req())
    await _rejected(rig, req(), "base_image_mismatch")


async def test_a_runtime_snapshot_is_not_compared_with_the_active_image(tmp_path: Path) -> None:
    """The runtime is the base image plus signed native builds, which no image digest names: only the base image is
    compared, as the compile's ``check_base_image`` does."""
    rig = build(tmp_path, repo=FakeRepo(Recipe(base_image_digest=IMAGE, runtime_snapshot=OTHER_IMAGE)))
    request_id = await rig.toolbox.submit_test(req())
    assert request_id in rig.blackboard._state().requests


async def test_a_retest_of_an_unknown_request_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req(purpose=Purpose.CONFIRM, retest_of="not-a-request"), "unknown_retest")


async def test_a_request_that_already_has_an_id_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path)
    await _rejected(rig, req().accepted("mine"), "invalid_request")


async def test_a_valid_request_is_accepted_and_written_to_the_blackboard(tmp_path: Path) -> None:
    rig = build(tmp_path)
    request_id = await rig.toolbox.submit_test(
        req(purpose=Purpose.CONFIRM, retest_of=await rig.toolbox.submit_test(req()))
    )
    row = rig.blackboard._state().requests[request_id]
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
    assert rig.blackboard._state().requests == {} and rig.toolbox._log.entries() == []
    assert rig.planner.calls == []


@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.INTEGRATION])
async def test_a_delta_that_restates_the_parents_upstream_is_accepted_unchanged(
    tmp_path: Path, purpose: Purpose
) -> None:
    rig = build(tmp_path, repo=_with_sglang())
    change = RecipeDelta(components=(ComponentChange("sglang", SGLANG_REF, SGLANG_REPO, add_patches=(PATCH,)),))
    request_id = await rig.toolbox.submit_test(req(purpose=purpose, change=change))
    assert rig.blackboard._state().requests[request_id].purpose == purpose.value
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


def _parent_with_settings() -> FakeRepo:
    launch = LaunchConfig("--max-num-seqs 256", {"SGLANG_FUSE": "1"})
    sglang = ComponentPatches("sglang", SGLANG_REPO, SGLANG_REF, patches=(PATCH,))
    return FakeRepo(Recipe(base_image_digest=IMAGE, launch=launch, components=(sglang,)))


RESTATED = RecipeDelta(
    config=ConfigDelta(server_args="--max-num-seqs 256", envs={"SGLANG_FUSE": "1"}),
    components=(ComponentChange("sglang", add_patches=(PATCH,), remove_patches=(PATCH.path,)),),
)


async def test_an_explore_whose_candidate_equals_its_parent_is_refused_naming_the_settings(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=_parent_with_settings())
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=RESTATED))
    assert refused.value.code == "no_effective_change" and "no_effective_change" in REJECTION_CODES
    for setting in ("--max-num-seqs 256", "SGLANG_FUSE=1", PATCH.path):
        assert setting in refused.value.reason
    await _rejected(rig, req(change=RESTATED), "no_effective_change")
    assert rig.planner.calls == []


async def test_an_explore_that_changes_one_setting_of_its_parent_is_accepted(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=_parent_with_settings())
    change = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256", envs={"SGLANG_FUSE": "0"}))
    assert await rig.toolbox.submit_test(req(change=change))


async def test_an_explore_that_only_removes_a_flag_its_parents_args_lack_is_accepted_for_measurement(
    tmp_path: Path,
) -> None:
    """The flag is one of the launch the framework resolves; the candidate's launch keeps it removed."""
    rig = build(tmp_path, repo=_parent_with_settings())
    change = RecipeDelta(config=ConfigDelta(remove_args=("--enable-linear-replayssm-spec",)))
    assert await rig.toolbox.submit_test(req(change=change))
    assert rig.planner.calls == [change]


@pytest.mark.parametrize("purpose", [Purpose.CONFIRM, Purpose.INTEGRATION, Purpose.FINAL])
async def test_a_confirm_integration_or_final_may_re_run_the_parent_recipe(tmp_path: Path, purpose: Purpose) -> None:
    rig = build(tmp_path, repo=_parent_with_settings())
    first = await rig.toolbox.submit_test(req(change=RecipeDelta(config=ConfigDelta(envs={"SGLANG_FUSE": "0"}))))
    retest = {"retest_of": first} if purpose is Purpose.CONFIRM else {}
    assert await rig.toolbox.submit_test(req(purpose=purpose, change=RESTATED, **retest))


# --- GSM8K on confirm, lead adoption and FINAL only, once the rule gate takes an explore without it (board topic 128) --


def _skipping_explore() -> Acceptance:
    setup = AcceptanceSetup(FakeRepo(), FakePlanner(), IMAGE, explore_accuracy=False)
    return Acceptance(setup, is_accepted=lambda request_id: True)


@pytest.mark.parametrize("asked", [False, True])
def test_a_plain_explore_skips_accuracy_and_its_request_records_the_skip_when_explore_accuracy_is_off(
    asked: bool,
) -> None:
    accepted = _skipping_explore().with_required_accuracy(req(purpose=Purpose.EXPLORE, accuracy=asked))

    assert accepted.accuracy is False


def test_an_explore_that_adopts_a_lead_with_the_consistency_check_still_runs_accuracy() -> None:
    request = req(purpose=Purpose.EXPLORE, consistency_check=True)

    assert _skipping_explore().with_required_accuracy(request).accuracy is True


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.INTEGRATION, Purpose.FINAL])
def test_with_explore_accuracy_off_every_other_purpose_the_gate_reads_still_runs_accuracy(purpose: Purpose) -> None:
    assert _skipping_explore().with_required_accuracy(req(purpose=purpose)).accuracy is True


def test_with_explore_accuracy_off_a_confirm_still_runs_accuracy() -> None:
    request = req(purpose=Purpose.CONFIRM, retest_of="first")

    assert _skipping_explore().with_required_accuracy(request).accuracy is True


async def test_by_default_a_plain_explore_still_runs_accuracy_because_the_gate_reads_its_score(tmp_path: Path) -> None:
    assert AcceptanceSetup(FakeRepo(), FakePlanner(), IMAGE).explore_accuracy is True
    assert await _accepted_accuracy(build(tmp_path), req(purpose=Purpose.EXPLORE)) == (True, True)
