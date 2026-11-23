# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance of a session that keeps launch configuration only (``run --no-source-changes``).

Every test of a session passes acceptance, and every commit is the recipe of a request it accepted, so a session whose
acceptance refuses a recipe that holds a patch or a kernel artifact hands on server arguments and environment variables
alone.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    PatchRef,
    Recipe,
    RecipeDelta,
)
from hyperloom_multibranch.contracts.toolbox import LAUNCH_CONFIG_ONLY_CODE, REJECTION_CODES, RequestRejected
from hyperloom_multibranch.toolbox.acceptance import Acceptance
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakePlanner, FakeRepo, build, req
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

FRAMEWORK_REPO = "https://github.com/sgl-project/sglang"
FRAMEWORK_REF = "9f5d8712" + "0" * 32
PATCH = PatchRef("patch/sglang/0001-fuse.patch", "d" * 64)
TABLE = KernelArtifact(
    KernelKind.GEMM,
    "qwen3-8b-bf16",
    "kernel/gemm/qwen3-8b-bf16/" + "e" * 64 + ".csv",
    "e" * 64,
    {"load": {"via": "env_file", "env": "AITER_CONFIG_GEMM_BF16"}},
)
SETTINGS = RecipeDelta(config=ConfigDelta(server_args="--max-running-requests 64", envs={"SGLANG_USE_AITER": "1"}))
WITH_PATCH = RecipeDelta(components=(ComponentChange("sglang", add_patches=(PATCH,)),))
WITH_TABLE = RecipeDelta(add_kernels=(TABLE,))
#: A component the root lacks, pinned without a patch: it carries no file, and is still no launch configuration.
WITH_COMPONENT = RecipeDelta(
    components=(ComponentChange("triton", "e8b97b24" + "0" * 32, "https://github.com/triton-lang/triton"),)
)


def _root() -> Recipe:
    """A session's root recipe as PRELUDE writes it: the framework pinned, without a patch."""
    return Recipe(base_image_digest=IMAGE, components=(ComponentPatches("sglang", FRAMEWORK_REPO, FRAMEWORK_REF),))


async def _code_of(
    tmp_path: Path, change: RecipeDelta, *, source_changes: bool = False, **fields: object
) -> str | None:
    """The code a submitted request is refused with, ``None`` when it is accepted."""
    rig = build(tmp_path, repo=FakeRepo(_root()), source_changes=source_changes)
    try:
        await rig.toolbox.submit_test(req(change=change, **fields))
    except RequestRejected as refused:
        assert rig.planner.calls == [], "a refused request must not reach the compiler"
        assert rig.blackboard._state().requests == {}, "a refused request must write nothing to the blackboard"
        return refused.code
    return None


def test_the_code_is_a_documented_rejection_code() -> None:
    assert LAUNCH_CONFIG_ONLY_CODE in REJECTION_CODES


@pytest.mark.parametrize("change", [WITH_PATCH, WITH_TABLE, WITH_COMPONENT], ids=["patch", "kernel_table", "component"])
@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.INTEGRATION, Purpose.FINAL])
async def test_a_candidate_that_holds_a_patch_a_kernel_table_or_a_new_component_is_refused_whatever_submits_it(
    tmp_path: Path, change: RecipeDelta, purpose: Purpose
) -> None:
    assert await _code_of(tmp_path, change, purpose=purpose) == LAUNCH_CONFIG_ONLY_CODE


async def test_a_change_to_settings_carried_with_a_patch_is_refused_whole(tmp_path: Path) -> None:
    both = dataclasses.replace(WITH_PATCH, config=SETTINGS.config)
    assert await _code_of(tmp_path, both) == LAUNCH_CONFIG_ONLY_CODE


async def test_the_refusal_names_what_the_recipe_holds_and_the_option(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(_root()), source_changes=False)
    both = dataclasses.replace(WITH_PATCH, add_kernels=(TABLE,))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=both))
    for named in ("--no-source-changes", "1 patch(es)", "1 kernel artifact(s)", "server arguments"):
        assert named in refused.value.reason, named


async def test_a_change_to_server_arguments_and_environment_variables_is_accepted(tmp_path: Path) -> None:
    assert await _code_of(tmp_path, SETTINGS) is None


@pytest.mark.parametrize(
    ("kind", "purpose"),
    [(TestKind.TEST, Purpose.BASELINE), (TestKind.PROFILE, Purpose.PROFILE)],
    ids=["baseline", "profile"],
)
async def test_a_test_of_the_root_as_it_is_is_accepted(tmp_path: Path, kind: TestKind, purpose: Purpose) -> None:
    assert await _code_of(tmp_path, RecipeDelta(), kind=kind, purpose=purpose) is None


@pytest.mark.parametrize(
    "change",
    [WITH_PATCH, WITH_TABLE, WITH_COMPONENT, SETTINGS],
    ids=["patch", "kernel_table", "component", "settings"],
)
async def test_a_session_that_keeps_source_changes_accepts_all_of_them(tmp_path: Path, change: RecipeDelta) -> None:
    assert await _code_of(tmp_path, change, source_changes=True) is None


async def test_the_check_that_queues_nothing_refuses_as_a_submit_does(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(_root()), source_changes=False)
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.check_test(req(purpose=Purpose.INTEGRATION, change=WITH_PATCH))
    assert refused.value.code == LAUNCH_CONFIG_ONLY_CODE
    await rig.toolbox.check_test(req(purpose=Purpose.INTEGRATION, change=SETTINGS))


async def test_the_recipe_is_judged_not_the_delta(tmp_path: Path) -> None:
    """A parent that holds a patch is refused under any delta but the one that takes the patch out."""
    patched = dataclasses.replace(
        _root(), components=(ComponentPatches("sglang", FRAMEWORK_REPO, FRAMEWORK_REF, (PATCH,)),)
    )
    setup = AcceptanceSetup(FakeRepo(patched), FakePlanner(), IMAGE, source_changes=False)
    acceptance = Acceptance(setup, is_accepted=lambda _: True)
    with pytest.raises(RequestRejected) as refused:
        await acceptance.prepare(req(change=SETTINGS))
    assert refused.value.code == LAUNCH_CONFIG_ONLY_CODE
    removed = RecipeDelta(components=(ComponentChange("sglang", remove_patches=(PATCH.path,)),))
    await acceptance.prepare(req(change=removed))
