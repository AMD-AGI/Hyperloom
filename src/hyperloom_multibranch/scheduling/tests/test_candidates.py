# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a change between two recipes names, which is what decides whether a not-stackable candidate comes back; which
commits a retraction keeps out of the offer; the order commits are offered in; and the cap on not-stackable answers."""

from __future__ import annotations

from dataclasses import replace

from hyperloom.contracts.recipe import (
    ComponentPatches,
    KernelArtifact,
    KernelKind,
    LaunchConfig,
    PatchRef,
    Recipe,
)
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, request
from hyperloom_multibranch.contracts.blackboard.records import CommittedCommit, MergeState, RetractionRef, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, MergeOutcome
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.rulegate.rules import RETRACTION_ID
from hyperloom_multibranch.scheduling.candidates import NOT_STACKABLE_ANSWERS, Candidates, touched

IMAGE_DIGEST = "sha256:" + "b" * 64
UPSTREAM_REF = "a" * 40
UPSTREAM_REPO = "https://example.com/vllm.git"
PATCH = PatchRef("patch/vllm/first.diff", "1" * 64)
OTHER_PATCH = PatchRef("patch/vllm/second.diff", "2" * 64)
#: What each patch touches, as :class:`~hyperloom_multibranch.scheduling.candidates.Candidates` reads it from a repo.
PATCHED = {PATCH.sha256: frozenset({"vllm/attention.py"}), OTHER_PATCH.sha256: frozenset({"vllm/moe.py"})}
VLLM = ComponentPatches("vllm", UPSTREAM_REPO, UPSTREAM_REF, (PATCH,))
KERNEL = KernelArtifact(KernelKind.GEMM, "qkv_proj", "kernel/qkv.hsaco", "3" * 64)
BEFORE = Recipe(IMAGE_DIGEST, LaunchConfig("--max-num-seqs 64", {"HL_MODE": "fast"}), (VLLM,), (KERNEL,))


def test_a_changed_server_argument_is_named_by_its_flag_whatever_the_value() -> None:
    raised = replace(BEFORE, launch=LaunchConfig("--max-num-seqs 256", BEFORE.launch.envs))
    lowered = replace(BEFORE, launch=LaunchConfig("--max-num-seqs 32", BEFORE.launch.envs))

    assert touched(BEFORE, raised, PATCHED) == frozenset({"arg --max-num-seqs"})
    assert touched(BEFORE, lowered, PATCHED) == touched(BEFORE, raised, PATCHED)


def test_an_added_server_argument_is_named_by_its_flag() -> None:
    added = replace(BEFORE, launch=LaunchConfig("--max-num-seqs 64 --enable-chunked-prefill", BEFORE.launch.envs))

    assert touched(BEFORE, added, PATCHED) == frozenset({"arg --enable-chunked-prefill"})


def test_a_changed_environment_variable_is_named() -> None:
    changed = replace(BEFORE, launch=LaunchConfig(BEFORE.launch.server_args, {"HL_MODE": "slow"}))
    unset = replace(BEFORE, launch=LaunchConfig(BEFORE.launch.server_args, {}))

    assert touched(BEFORE, changed, PATCHED) == frozenset({"env HL_MODE"})
    assert touched(BEFORE, unset, PATCHED) == frozenset({"env HL_MODE"})


def test_a_component_whose_upstream_ref_moved_is_named_whole() -> None:
    rebased = replace(BEFORE, components=(replace(VLLM, upstream_ref="c" * 40),))

    assert touched(BEFORE, rebased, PATCHED) == frozenset({"component vllm"})


def test_a_patch_a_component_gains_or_loses_is_named_by_the_files_it_touches() -> None:
    restacked = replace(BEFORE, components=(replace(VLLM, patches=(PATCH, OTHER_PATCH)),))
    dropped = replace(BEFORE, components=())

    assert touched(BEFORE, restacked, PATCHED) == frozenset({"component vllm file vllm/moe.py"})
    assert touched(BEFORE, dropped, PATCHED) == frozenset({"component vllm file vllm/attention.py"})


def test_two_changes_that_patch_one_component_in_different_files_name_nothing_in_common() -> None:
    bare = replace(BEFORE, components=())
    mine = replace(bare, components=(replace(VLLM, patches=(PATCH,)),))
    theirs = replace(bare, components=(replace(VLLM, patches=(OTHER_PATCH,)),))

    assert not touched(bare, mine, PATCHED) & touched(bare, theirs, PATCHED)


def test_a_kernel_artifact_is_named_by_its_kind_and_target() -> None:
    swapped = replace(BEFORE, kernels=(replace(KERNEL, ref="kernel/qkv2.hsaco", sha256="4" * 64),))
    added = replace(BEFORE, kernels=(KERNEL, KernelArtifact(KernelKind.FUSION, "rmsnorm", "kernel/n.hsaco", "5" * 64)))

    assert touched(BEFORE, swapped, PATCHED) == frozenset({"kernel gemm qkv_proj"})
    assert touched(BEFORE, added, PATCHED) == frozenset({"kernel fusion rmsnorm"})


def test_two_recipes_that_differ_in_nothing_name_nothing() -> None:
    assert touched(BEFORE, replace(BEFORE), PATCHED) == frozenset()


class _Committed:
    """The integrator's view of committed commits and the records behind them."""

    def __init__(self, commits: tuple[CommittedCommit, ...], records: dict[str, TestRecord]) -> None:
        self._commits, self._records = commits, records

    async def committed(self) -> tuple[CommittedCommit, ...]:
        return self._commits

    async def record(self, request_id: str) -> TestRecord:
        return self._records[request_id]


def _commit(commit_id: str, parent: str, branch: str = "b01") -> CommittedCommit:
    return CommittedCommit(commit_id, branch, parent, f"r-{commit_id}", 1.0, False, False)


def _landed(
    commit_id: str, verdict: Verdict, *attachments: AttachmentRef, retracts: RetractionRef | None = None
) -> TestRecord:
    asked = request(f"r-{commit_id}", intent("b01", "c0"), attachments=attachments)
    decided = judgement(-4.0 if verdict is Verdict.REGRESSED else 3.0, (asked.request_id or "",), verdict=verdict)
    return TestRecord(asked, judgement=decided, conclusion=Conclusion.COMMITTED, commit_id=commit_id, retracts=retracts)


async def test_the_commits_a_retraction_was_committed_on_are_not_offered_and_the_retraction_is() -> None:
    mark = AttachmentRef(RETRACTION_ID, "file", "b" * 64)
    commits = (_commit("c1", "c0"), _commit("c2", "c1"), _commit("c3", "c2"), _commit("d1", "c0", "b02"))
    records = {
        "r-c1": _landed("c1", Verdict.IMPROVED),
        "r-c2": _landed("c2", Verdict.IMPROVED),
        "r-c3": _landed("c3", Verdict.REGRESSED, mark, retracts=RetractionRef(7, 1, "f7", "c2")),
        "r-d1": _landed("d1", Verdict.IMPROVED),
    }

    offered = await Candidates(_Committed(commits, records), None).offered_on("i0")

    assert [commit.commit_id for commit in offered] == ["c3", "d1"]


async def test_a_commit_whose_record_names_no_retraction_hides_nothing_whatever_its_request_carries() -> None:
    mark = AttachmentRef(RETRACTION_ID, "file", "b" * 64)
    commits = (_commit("c1", "c0"), _commit("c2", "c1"))
    records = {"r-c1": _landed("c1", Verdict.IMPROVED), "r-c2": _landed("c2", Verdict.REGRESSED, mark)}

    offered = await Candidates(_Committed(commits, records), None).offered_on("i0")

    assert [commit.commit_id for commit in offered] == ["c1", "c2"]


def _conflicted(commit_id: str, gain: float, not_stackable: int = 0) -> CommittedCommit:
    """A commit whose latest merge, on head ``i0``, conflicted, after ``not_stackable`` answers on earlier heads."""

    last = MergeState(MergeOutcome.CONFLICT_SKIPPED, "i0")
    attempts = not_stackable + 1
    return replace(
        _commit(commit_id, "c0"), gain_pct=gain, last_merge=last, merge_attempts=attempts, not_stackable=not_stackable
    )


async def test_commits_never_tried_are_offered_before_those_coming_back_each_in_the_blackboard_s_order() -> None:
    commits = (
        _conflicted("back-large", 9.0),
        replace(_commit("new-large", "c0"), gain_pct=5.0),
        _conflicted("back-small", 2.0),
        replace(_commit("new-small", "c0"), gain_pct=1.0),
    )
    records = {f"r-{commit.commit_id}": _landed(commit.commit_id, Verdict.IMPROVED) for commit in commits}

    offered = await Candidates(_Committed(commits, records), None).offered_on("i1")

    assert [commit.commit_id for commit in offered] == ["new-large", "new-small", "back-large", "back-small"]


async def test_a_commit_not_stackable_on_as_many_heads_as_the_cap_is_not_offered_again_whatever_came_after() -> None:
    """A conflict on a later head replaces the commit's latest merge, and would bring it back on every head."""

    commits = (
        _conflicted("capped", 9.0, not_stackable=NOT_STACKABLE_ANSWERS),
        _conflicted("below", 5.0, not_stackable=NOT_STACKABLE_ANSWERS - 1),
    )
    records = {f"r-{commit.commit_id}": _landed(commit.commit_id, Verdict.IMPROVED) for commit in commits}

    offered = await Candidates(_Committed(commits, records), None).offered_on("i1")

    assert [commit.commit_id for commit in offered] == ["below"]
