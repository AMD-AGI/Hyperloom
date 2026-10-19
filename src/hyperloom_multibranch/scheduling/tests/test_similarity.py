# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which running branches made much the same change: each branch's own change since its start, compared pair by pair."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, PatchRef, Recipe
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.roster import BranchOrigin, BranchStop, Roster, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.similarity import (
    MB_SIMILARITY_THRESHOLD,
    BranchSimilarity,
    SimilarBranch,
    own_change,
)
from hyperloom_multibranch.scheduling.tests.peer_fakes import BASE_IMAGE_DIGEST

ROOT = Recipe(BASE_IMAGE_DIGEST)
UPSTREAM = "0123456789abcdef0123456789abcdef01234567"
INHERITED = {f"INHERITED_{index}": "1" for index in range(8)}


def _config(args: str = "", **envs: str) -> Recipe:
    return Recipe(BASE_IMAGE_DIGEST, LaunchConfig(server_args=args, envs=envs))


def _patched(*patches: PatchRef) -> Recipe:
    component = ComponentPatches("vllm", "https://github.com/vllm-project/vllm", UPSTREAM, patches)
    return Recipe(BASE_IMAGE_DIGEST, components=(component,))


class _Repo:
    """The experiment repository's read side: each commit's recipe and each branch's head."""

    def __init__(self) -> None:
        self.recipes = {"c-root": ROOT}
        self.heads: dict[str, str] = {}

    async def read_recipe(self, commit: str) -> Recipe:
        return self.recipes[commit]

    async def head(self, branch: str) -> str:
        return self.heads[branch]


class _Session:
    def __init__(self, tmp_path: Path) -> None:
        caps = RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=8)
        self.roster = Roster(caps, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))
        self.roster.add_integration("c-root", 1)
        self.repo = _Repo()
        self.repo.heads[INTEGRATION_BRANCH] = "c-root"

    def branch(self, start: str, head: Recipe | None = None) -> str:
        """A running branch from ``start``, at a commit of its own holding ``head``, or still at ``start``."""

        branch = self.roster.next_branch_id()
        self.roster.open(branch, origin=BranchOrigin.FIRST_BATCH, start_commit=start, k=1)
        self.repo.heads[branch] = start
        if head is not None:
            self.repo.recipes[f"c-{branch}"] = head
            self.repo.heads[branch] = f"c-{branch}"
        return branch

    def similar_to(self) -> dict[str, tuple[SimilarBranch, ...]]:
        return dict(asyncio.run(BranchSimilarity(self.repo, self.roster).similar_to()))


def test_two_branches_that_set_the_same_config_are_identical_and_a_different_one_is_in_no_pair(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    first = session.branch("c-root", _config("--max-num-seqs 128", VLLM_USE_AITER="1"))
    second = session.branch("c-root", _config("--max-num-seqs 128", VLLM_USE_AITER="1"))
    session.branch("c-root", _config("--max-num-seqs 256"))

    assert session.similar_to() == {
        first: (SimilarBranch(second, 1.0, identical=True),),
        second: (SimilarBranch(first, 1.0, identical=True),),
    }


def test_config_changes_that_mostly_overlap_are_similar_but_not_identical(tmp_path: Path) -> None:
    shared = {f"KNOB_{index}": "1" for index in range(4)}
    session = _Session(tmp_path)
    first = session.branch("c-root", _config(**shared))
    second = session.branch("c-root", _config(**shared, EXTRA="1"))

    assert session.similar_to()[first] == (SimilarBranch(second, 0.8, identical=False),)
    assert MB_SIMILARITY_THRESHOLD == 0.8


def test_patches_compare_by_content_so_the_same_patch_under_another_name_is_similar_not_identical(
    tmp_path: Path,
) -> None:
    session = _Session(tmp_path)
    first = session.branch("c-root", _patched(PatchRef("patch/vllm/0001-moe.patch", "a" * 64)))
    second = session.branch("c-root", _patched(PatchRef("patch/vllm/0001-moe-align.patch", "a" * 64)))
    session.branch("c-root", _patched(PatchRef("patch/vllm/0001-attn.patch", "b" * 64)))

    assert session.similar_to() == {
        first: (SimilarBranch(second, 1.0, identical=False),),
        second: (SimilarBranch(first, 1.0, identical=False),),
    }


def test_branches_refilled_from_one_integration_head_without_a_commit_of_their_own_are_in_no_pair(
    tmp_path: Path,
) -> None:
    session = _Session(tmp_path)
    session.repo.recipes["c-integration"] = _config(**INHERITED)
    session.branch("c-integration")
    session.branch("c-integration")

    assert session.similar_to() == {}


def test_only_the_change_since_a_branchs_own_start_counts_not_what_it_inherited(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    session.repo.recipes["c-integration"] = _config(**INHERITED)
    session.branch("c-integration", _config(**INHERITED, ATTENTION="triton"))
    session.branch("c-integration", _config(**INHERITED, MOE="ck"))

    assert session.similar_to() == {}
    assert len(own_change(ROOT, _config(**INHERITED, ATTENTION="triton"))) == len(INHERITED) + 1


def test_stopped_branches_and_the_integration_branch_are_not_compared(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    same = _config(VLLM_USE_AITER="1")
    session.branch("c-root", same)
    stopped = session.branch("c-root", same)
    session.roster.stop(stopped, BranchStop(StopCause.JUDGE, "flat", 1.0))
    session.repo.recipes["c-merged"] = same
    session.repo.heads[INTEGRATION_BRANCH] = "c-merged"

    assert session.similar_to() == {}


def test_a_dropped_setting_is_a_change_of_its_own() -> None:
    assert own_change(_config("--enforce-eager", KNOB="1"), _config()) == frozenset(
        {"arg --enforce-eager unset", "env KNOB unset"}
    )


def test_reading_the_pairs_stops_no_branch(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    same = _config(VLLM_USE_AITER="1")
    session.branch("c-root", same)
    session.branch("c-root", same)

    assert session.similar_to()
    assert all(entry.stop is None for entry in session.roster.entries())
