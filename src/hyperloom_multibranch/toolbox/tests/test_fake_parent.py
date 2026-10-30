# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fake toolbox reads a parent's recipe from the experiment repository it is wired to, as the real one does.

The repository here is the real git one over a real ``vllm`` layout, and the merges are its own: nothing in these
tests says what a recipe holds except the repository that holds it.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ComponentChange, ComponentPatches, LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring
from hyperloom_multibranch.toolbox.fake_model import FakeToolboxConfig

IMAGE = "sha256:" + "c" * 64
VLLM = "https://github.com/vllm-project/vllm"
PIN = "1" * 40
UNKNOWN_COMMIT = "f" * 40
BASE = Recipe(
    IMAGE,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", VLLM, PIN),),
)
INSTALLED_CONFIG = "".join(f"{name} = {number}\n" for number, name in enumerate("abcdefgh", 1))


def _edit(**settings: str) -> str:
    """A patch of the installed ``config.py`` that sets each named line, with three lines of context as git writes it."""

    edited = INSTALLED_CONFIG
    for name, value in settings.items():
        edited = re.sub(rf"^{name} = .*$", f"{name} = {value}", edited, flags=re.MULTILINE)
    unified = difflib.unified_diff(
        INSTALLED_CONFIG.splitlines(keepends=True),
        edited.splitlines(keepends=True),
        "a/vllm/config.py",
        "b/vllm/config.py",
    )
    return "diff --git a/vllm/config.py b/vllm/config.py\n" + "".join(unified)


def _repository(tmp_path: Path) -> GitExperimentRepo:
    package = tmp_path / "site-packages" / "vllm"
    package.mkdir(parents=True)
    (package / "config.py").write_text(INSTALLED_CONFIG)
    installed = PackageBase("vllm", package, None, None)
    return GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _: installed)


async def _candidate(repo: GitExperimentRepo, branch: str, root: str, text: str) -> str:
    """A commit on a branch of its own off ``root`` that adds one patch of ``vllm``."""

    await repo.create_branch(branch, root)
    stored = await repo.store_patch("vllm", text)
    delta = RecipeDelta(components=(ComponentChange("vllm", add_patches=(stored,)),))
    return await repo.commit(branch, root, apply(await repo.read_recipe(root), delta), branch)


def _test_of(parent: str, delta: RecipeDelta, branch: str = INTEGRATION_BRANCH) -> TestRequest:
    return TestRequest(TestKind.TEST, Purpose.INTEGRATION, branch, parent, delta)


async def _rig(tmp_path: Path, **wiring: object) -> tuple[GitExperimentRepo, FakeToolbox, str]:
    """A repository holding ``BASE`` as its root, and a fake toolbox wired to it."""

    repo = _repository(tmp_path)
    root = await repo.init_root(BASE)
    box = FakeToolbox(FakeToolboxConfig(image_digest=IMAGE), wiring=FakeWiring(repo=repo, **wiring))  # type: ignore[arg-type]
    return repo, box, root


async def _patches_only(repo: GitExperimentRepo, component: str = "vllm", **lines: str) -> RecipeDelta:
    """A delta that adds one patch of ``component`` setting ``lines`` (default ``c = 30``) and names no upstream."""

    text = _edit(**(lines or {"c": "30"}))
    return RecipeDelta(components=(ComponentChange(component, add_patches=(await repo.store_patch(component, text),)),))


async def test_a_patches_only_delta_onto_a_commit_that_pins_the_component_is_accepted(tmp_path):
    repo, box, root = await _rig(tmp_path)
    delta = await _patches_only(repo)
    assert delta.components[0].upstream_ref is None and delta.components[0].upstream_repo is None

    request_id = await box.submit_test(_test_of(root, delta))

    assert box.requests[request_id].delta == delta
    await box.drain()


async def test_a_delta_naming_a_new_component_without_an_upstream_is_still_refused(tmp_path):
    repo, box, root = await _rig(tmp_path)

    with pytest.raises(RequestRejected, match="sglang is a new component") as refused:
        await box.submit_test(_test_of(root, await _patches_only(repo, "sglang")))

    assert refused.value.code == "delta_not_applicable"
    assert box.requests == {}


async def test_a_new_component_that_gives_its_upstream_is_accepted(tmp_path):
    _, box, root = await _rig(tmp_path)
    delta = RecipeDelta(components=(ComponentChange("sglang", upstream_ref="2" * 40, upstream_repo=VLLM),))

    request_id = await box.submit_test(_test_of(root, delta))

    assert box.requests[request_id].delta == delta
    await box.drain()


async def test_a_toolbox_with_no_repository_reads_every_parent_as_pinning_nothing(tmp_path):
    repo = _repository(tmp_path)
    await repo.init_root(BASE)
    box = FakeToolbox(FakeToolboxConfig(image_digest=IMAGE))

    with pytest.raises(RequestRejected, match="vllm is a new component") as refused:
        await box.submit_test(_test_of("c0", await _patches_only(repo)))

    assert refused.value.code == "delta_not_applicable"


async def test_a_commit_the_repository_does_not_hold_is_an_unknown_parent(tmp_path):
    repo, box, _ = await _rig(tmp_path)

    with pytest.raises(RequestRejected) as refused:
        await box.submit_test(_test_of(UNKNOWN_COMMIT, await _patches_only(repo)))

    assert refused.value.code == "unknown_parent"


async def test_parent_exists_still_narrows_a_commit_the_repository_holds(tmp_path):
    repo, box, root = await _rig(tmp_path, parent_exists=lambda commit: False)

    with pytest.raises(RequestRejected) as refused:
        await box.submit_test(_test_of(root, await _patches_only(repo)))

    assert refused.value.code == "unknown_parent"


async def test_a_recipe_on_another_image_than_the_toolbox_runs_is_refused(tmp_path):
    repo = _repository(tmp_path)
    root = await repo.init_root(BASE)
    box = FakeToolbox(FakeToolboxConfig(image_digest="sha256:" + "d" * 64), wiring=FakeWiring(repo=repo))

    with pytest.raises(RequestRejected) as refused:
        await box.submit_test(_test_of(root, await _patches_only(repo)))

    assert refused.value.code == "base_image_mismatch"


async def _head_with_first(repo: GitExperimentRepo, root: str, text: str) -> tuple[str, str]:
    """The integration head after the candidate that patches ``text`` merged onto the root, and that candidate."""

    first = await _candidate(repo, "s1", root, text)
    head = await repo.commit_integration(root, (await repo.merge(first, root)).delta, (first,))
    return head, first


async def test_a_second_patch_carrying_candidate_merges_onto_a_head_holding_the_first(tmp_path):
    """Nearby lines of one installed file: the second patch does not apply where the head put the first, so the
    merge rebases it, and its delta is patches only."""

    repo, box, root = await _rig(tmp_path)
    head, _ = await _head_with_first(repo, root, _edit(c="30"))
    merge = await repo.merge(await _candidate(repo, "s2", root, _edit(e="50")), head)
    (rebased,) = merge.rebased_patches
    (change,) = merge.delta.components
    assert merge.clean and change.add_patches == (rebased.patch,)
    assert change.upstream_ref is None and change.upstream_repo is None

    request_id = await box.submit_test(_test_of(head, merge.delta))

    assert box.requests[request_id].parent_commit == head and box.requests[request_id].delta == merge.delta
    merged = await repo.commit_integration(head, merge.delta, ())
    third = await box.submit_test(_test_of(merged, await _patches_only(repo, g="70")))
    assert box.requests[third].parent_commit == merged
    await box.drain()


async def test_a_merge_that_left_its_patch_out_as_a_conflict_is_still_accepted_on_a_pinned_head(tmp_path):
    """A line both sides changed: the patch is a conflict, the delta carries the change without it."""

    repo, box, root = await _rig(tmp_path)
    head, _ = await _head_with_first(repo, root, _edit(c="30", h="80"))
    merge = await repo.merge(await _candidate(repo, "s2", root, _edit(c="31", e="50")), head)
    (conflict,) = merge.patch_conflicts
    assert conflict.component == "vllm" and merge.rebased_patches == ()

    request_id = await box.submit_test(_test_of(head, merge.delta))

    assert box.requests[request_id].delta == merge.delta
    await box.drain()
