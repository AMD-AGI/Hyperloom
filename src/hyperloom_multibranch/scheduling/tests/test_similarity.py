# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which running branches made much the same change: each branch's own change since the commit it last followed, compared
pair by pair."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, PatchRef, Recipe
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchNode, CommitNode, LineageView
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
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
    """The experiment repository's read side: each commit's recipe."""

    def __init__(self) -> None:
        self.recipes = {"c-root": ROOT}

    async def read_recipe(self, commit: str) -> Recipe:
        return self.recipes[commit]


class _Session:
    """The roster, the repository's recipes, and the notebook's lineage of the commits and branch heads."""

    def __init__(self, tmp_path: Path) -> None:
        caps = RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=8)
        self.roster = Roster(caps, tmp_path / "MB-roster.json", Clock(now=lambda: 0.0))
        self.roster.add_integration("c-root", 1)
        self.repo = _Repo()
        self.commits = [CommitNode("c-root", "", INTEGRATION_BRANCH, CommitOrigin.ROOT)]
        self.heads = {INTEGRATION_BRANCH: "c-root"}

    def integration(self, commit: str, recipe: Recipe) -> None:
        """Move the integration head onto a merge ``commit`` holding ``recipe``."""

        self.repo.recipes[commit] = recipe
        self.commits.append(CommitNode(commit, self.heads[INTEGRATION_BRANCH], INTEGRATION_BRANCH, CommitOrigin.MERGE))
        self.heads[INTEGRATION_BRANCH] = commit

    def branch(self, start: str, head: Recipe | None = None) -> str:
        """A running branch from ``start``, at a commit of its own holding ``head``, or still at ``start``."""

        branch = self.roster.next_branch_id()
        self.roster.open(branch, origin=BranchOrigin.FIRST_BATCH, start_commit=start, k=1)
        self.heads[branch] = start
        if head is not None:
            self.commit(branch, head)
        return branch

    def commit(self, branch: str, recipe: Recipe) -> None:
        commit = f"c-{branch}-{len(self.commits)}"
        self.repo.recipes[commit] = recipe
        self.commits.append(CommitNode(commit, self.heads[branch], branch, CommitOrigin.COMMIT))
        self.heads[branch] = commit

    def reroot(self, branch: str, onto: str) -> None:
        """The branch's runtime re-roots it onto the integration commit ``onto``, as its ``REROOT`` lineage reads."""

        self.heads[branch] = onto

    def lineage(self) -> LineageView:
        nodes = tuple(
            BranchNode(branch, LineageKind.OPEN, self.roster.entry(branch).start_commit, head)
            for branch, head in self.heads.items()
        )
        return LineageView(tuple(self.commits), nodes)

    def similar_to(self) -> dict[str, tuple[SimilarBranch, ...]]:
        return dict(asyncio.run(BranchSimilarity(self.repo, self.roster).similar_to(self.lineage())))


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
    session.integration("c-integration", _config(**INHERITED))
    session.branch("c-integration")
    session.branch("c-integration")

    assert session.similar_to() == {}


def test_only_the_change_since_a_branchs_own_start_counts_not_what_it_inherited(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    session.integration("c-integration", _config(**INHERITED))
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
    session.integration("c-merged", same)

    assert session.similar_to() == {}


def test_branches_re_rooted_onto_one_integration_head_without_a_commit_since_are_in_no_pair(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    first = session.branch("c-root", _config(ATTENTION="triton"))
    second = session.branch("c-root", _config(MOE="ck"))
    session.integration("c-integration", _config(ATTENTION="triton", MOE="ck", **INHERITED))
    session.reroot(first, "c-integration")
    session.reroot(second, "c-integration")

    assert session.similar_to() == {}


def test_a_re_rooted_branch_s_change_is_read_from_the_integration_commit_it_moved_onto(tmp_path: Path) -> None:
    session = _Session(tmp_path)
    first = session.branch("c-root", _config(ATTENTION="triton"))
    second = session.branch("c-root", _config(ATTENTION="triton"))
    third = session.branch("c-root", _config(MOE="ck"))
    session.integration("c-integration", _config(ATTENTION="triton", **INHERITED))
    for branch in (first, second, third):
        session.reroot(branch, "c-integration")
    session.commit(first, _config(ATTENTION="triton", **INHERITED, MOE_BLOCK="64"))
    session.commit(second, _config(ATTENTION="triton", **INHERITED, MOE_BLOCK="64"))
    session.commit(third, _config(ATTENTION="triton", **INHERITED, KV_CACHE="fp8"))

    assert session.similar_to() == {
        first: (SimilarBranch(second, 1.0, identical=True),),
        second: (SimilarBranch(first, 1.0, identical=True),),
    }


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
