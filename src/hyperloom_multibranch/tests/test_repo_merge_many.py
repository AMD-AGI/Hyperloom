# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Several committed candidates carried onto one integration head together, for one merge test of them all (#138)."""

import difflib

from hyperloom.contracts.recipe import (
    ComponentChange,
    ComponentPatches,
    ConfigDelta,
    LaunchConfig,
    Recipe,
    RecipeDelta,
    apply,
)
from hyperloom.measure.compile import LocalBlobStore, PackageBase
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo

IMAGE = "rocm/sglang@sha256:" + "c" * 64
AITER = "https://github.com/ROCm/aiter"
BASE = Recipe(IMAGE, LaunchConfig(server_args="--tp 4"), components=(ComponentPatches("aiter", AITER, "1" * 40),))
OPS = "aiter/ops.py"
LINES = [f"{name} = {value}\n" for value, name in enumerate("abcdefgh", 1)]


def _patch(line: int, text: str) -> str:
    after = [*LINES[:line], f"{text}\n", *LINES[line + 1 :]]
    return f"diff --git a/{OPS} b/{OPS}\n" + "".join(difflib.unified_diff(LINES, after, f"a/{OPS}", f"b/{OPS}"))


async def _rig(tmp_path):
    (tmp_path / "aiter").mkdir()
    (tmp_path / "aiter" / "ops.py").write_text("".join(LINES))
    installed = PackageBase("aiter", tmp_path / "aiter", None, None)
    repo = GitExperimentRepo(tmp_path / "experiments.git", LocalBlobStore(tmp_path / "blobs"), lambda _: installed)
    return repo, await repo.init_root(BASE)


async def _candidate(repo, root, branch, *, args: str = "", patch: str = "") -> str:
    await repo.create_branch(branch, root)
    stored = (await repo.store_patch("aiter", patch),) if patch else ()
    delta = RecipeDelta(
        config=ConfigDelta(server_args=args),
        components=(ComponentChange("aiter", add_patches=stored),) if stored else (),
    )
    return await repo.commit(branch, root, apply(BASE, delta), branch)


async def test_a_bundle_carries_every_candidate_onto_the_head_in_offer_order(tmp_path):
    repo, root = await _rig(tmp_path)
    kv = await _candidate(repo, root, "b1", args="--kv-cache-dtype fp8_e4m3")
    mtp = await _candidate(repo, root, "b2", args="--speculative-algorithm EAGLE", patch=_patch(0, "a = 10"))

    bundle = await repo.merge_many((kv, mtp), root)

    merged = apply(BASE, bundle.merge.delta)
    assert (bundle.members, bundle.left_out) == ((kv, mtp), ())
    assert "--kv-cache-dtype fp8_e4m3" in merged.launch.server_args
    assert "--speculative-algorithm EAGLE" in merged.launch.server_args
    assert len(merged.component("aiter").patches) == 1
    assert bundle.merge.carried == (kv, mtp) and bundle.merge.clean


async def test_a_later_member_s_value_wins_and_is_recorded_as_a_conflict(tmp_path):
    repo, root = await _rig(tmp_path)
    first = await _candidate(repo, root, "b1", args="--chunked-prefill-size 4096")
    second = await _candidate(repo, root, "b2", args="--chunked-prefill-size 8192")

    bundle = await repo.merge_many((first, second), root)

    assert "--chunked-prefill-size 8192" in apply(BASE, bundle.merge.delta).launch.server_args
    (conflict,) = bundle.merge.config_conflicts
    assert conflict.name == "--chunked-prefill-size"


async def test_a_member_that_does_not_combine_with_the_bundle_is_left_out_and_the_rest_carried(tmp_path):
    repo, root = await _rig(tmp_path)
    first = await _candidate(repo, root, "b1", patch=_patch(0, "a = 10"))
    clash = await _candidate(repo, root, "b2", patch=_patch(0, "a = 20"))
    third = await _candidate(repo, root, "b3", args="--cuda-graph-max-bs 64")

    bundle = await repo.merge_many((first, clash, third), root)

    assert (bundle.members, bundle.left_out) == ((first, third), (clash,))
    assert bundle.merge.clean and bundle.merge.carried == (first, third)


async def test_a_candidate_already_on_the_head_is_left_out(tmp_path):
    repo, root = await _rig(tmp_path)
    kv = await _candidate(repo, root, "b1", args="--kv-cache-dtype fp8_e4m3")
    head = await repo.commit_integration(root, (await repo.merge(kv, root)).delta, (kv,))
    mtp = await _candidate(repo, root, "b2", args="--speculative-algorithm EAGLE")

    bundle = await repo.merge_many((kv, mtp), head)

    assert (bundle.members, bundle.left_out) == ((mtp,), (kv,))
