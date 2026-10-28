# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from hyperloom.contracts.measurement import RoundValue
from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.peers import Purpose
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, BranchProgress, LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, LineageKind
from hyperloom_multibranch.contracts.repo import PatchConflict
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import intent, request, result
from hyperloom_multibranch.specialist.following import (
    SHORT,
    Followed,
    _write_conflicts,
    follow,
    integration_holdings,
    reroot_requested,
)
from hyperloom_multibranch.tests.branch_rig import BranchRig, branch_rig
from hyperloom_multibranch.tests.following_rig import OWN, diverged

BRANCH = "s1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)
#: The metric ``diverged``'s judgements record.
METRIC = "output_tps_pct"
REASON = "REROOT: the integration head took in chunked prefill"
ASKED = BranchLogEntry(BRANCH, BranchLogType.PLAN, REASON)


def _as(tmp_path: Path, role: Role) -> LocalBlackboard:
    return LocalBlackboard(tmp_path / "blackboard", Writer(role, role.value))


async def _follow(rig: BranchRig, tmp_path: Path) -> Followed:
    followed = await follow(rig.repo, rig.blackboard, await rig.blackboard.snapshot(), ASKED, tmp_path / "notes")
    assert followed is not None
    return followed


async def _row(rig: BranchRig, branch: str) -> BranchProgress:
    return next(row for row in (await rig.blackboard.snapshot()).progress if row.branch == branch)


async def test_the_holdings_name_the_integration_launch_its_merged_intents_and_what_the_head_lacks(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)

    text = await integration_holdings(rig.repo, await rig.blackboard.snapshot(), session.own)

    assert (
        f"The integration head is {session.onto}: server arguments --max-num-seqs 128 --enable-chunked-prefill" in text
    )
    merged = f"- {session.merged[:12]} of s0 (+3.00%), scheduler / scheduler change: add args --enable-chunked-prefill"
    assert f"{merged} [not on your head]" in text
    assert f"Your head {session.own} lacks of it: --enable-chunked-prefill." in text
    assert f"- {session.own[:12]} of s1 (+2.00%), attention / attention change" in text
    assert "the integrator has not tried it yet" in text


async def test_a_head_on_the_integration_head_lacks_nothing_of_it(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)

    text = await integration_holdings(rig.repo, await rig.blackboard.snapshot(), session.onto)

    assert f"Your head {session.onto} holds all of it." in text and "[not on your head]" not in text
    assert "Commits of your head" not in text


async def test_following_moves_the_branch_and_carries_its_own_change_onto_the_integration_head(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)

    followed = await _follow(rig, tmp_path)

    assert await rig.repo.head(BRANCH) == session.onto
    assert followed.notice == f"your branch was re-rooted from {session.own} onto the integration head {session.onto}"
    carried = followed.detail.split("as one delta: ", 1)[1].splitlines()[0]
    assert RecipeDelta.from_document(json.loads(carried)) == OWN
    assert "Call worktree_sync" in followed.detail and "the integrator has not tried it yet" in followed.detail


async def test_following_records_the_reroot_so_an_explore_on_the_new_head_waits_for_its_confirm(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    baseline = BaselineRecord(rig.root, {METRIC: 1.0}, {METRIC: 0.8}, {METRIC: 1.6}, ("b1", "b2"))
    await _as(tmp_path, Role.GATE).write_baseline(baseline)

    await _follow(rig, tmp_path)
    on_onto = intent(BRANCH, session.onto, "sampler", "fused top-k")
    toolbox = _as(tmp_path, Role.TOOLBOX)
    await toolbox.write_request(request("r5", on_onto))
    rounds = (RoundValue("warmup", 64, {METRIC: 1.0}), RoundValue("measure", 64, {METRIC: 3.5}))
    await toolbox.write_result(replace(result("r5"), rounds=rounds))
    await toolbox.write_request(replace(request("r5c", on_onto), purpose=Purpose.CONFIRM, retest_of="r5"))
    row = await _row(rig, BRANCH)

    assert (row.head_commit, [item.request_id for item in row.pending]) == (session.onto, ["r5"])
    assert [hit.summary for hit in await rig.blackboard.search("reroot")] == [
        f"reroot {BRANCH} at {session.onto}: {REASON}"
    ]


async def test_following_moves_onto_the_integration_head_the_blackboard_recorded_not_one_still_unjudged(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    unjudged = await rig.repo.commit_integration(session.onto, OWN, (session.own,))

    followed = await _follow(rig, tmp_path)

    assert await rig.repo.head(INTEGRATION_BRANCH) == unjudged
    assert await rig.repo.head(BRANCH) == session.onto and (await _row(rig, BRANCH)).head_commit == session.onto
    assert f"onto the integration head {session.onto}" in followed.notice


async def test_following_a_stopped_branch_moves_nothing(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    await _as(tmp_path, Role.SCHEDULER).record_lineage(LineageEvent(LineageKind.STOP, BRANCH, reason="no progress"))

    followed = await _follow(rig, tmp_path)

    assert await rig.repo.head(BRANCH) == session.own and followed.detail == ""
    assert followed.notice == "your branch was not re-rooted: it was stopped (no progress)"


async def test_a_branch_stopped_while_it_moves_is_taken_back_to_its_old_head(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    before = await rig.blackboard.snapshot()
    await _as(tmp_path, Role.SCHEDULER).record_lineage(LineageEvent(LineageKind.STOP, BRANCH, reason="no progress"))

    followed = await follow(rig.repo, rig.blackboard, before, ASKED, tmp_path / "notes")

    assert followed is not None and followed.detail == ""
    assert await rig.repo.head(BRANCH) == session.own and (await _row(rig, BRANCH)).head_commit == session.own
    assert "the blackboard refused the move" in followed.notice and "already stopped" in followed.notice


async def test_following_waits_while_the_blackboard_has_not_started_the_branch(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "blackboard", rig.root, ROOT, BRANCH)
    await rig.repo.create_branch("s9", rig.root)

    asked = replace(ASKED, branch="s9")
    waited = await follow(rig.repo, rig.blackboard, await rig.blackboard.snapshot(), asked, tmp_path / "notes")

    assert waited is None and await rig.repo.head("s9") == rig.root != session.onto


async def test_following_from_a_head_that_holds_the_integration_head_moves_nothing(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    head = await rig.repo.head(INTEGRATION_BRANCH)

    followed = await _follow(rig, tmp_path)

    assert await rig.repo.head(BRANCH) == head and followed.detail == ""
    assert "holds the integration head" in followed.notice


def test_only_an_entry_that_opens_with_reroot_asks_to_follow():
    assert reroot_requested("REROOT: the integration head took in fp8 KV, which my head lacks")
    assert not reroot_requested("I will not REROOT yet: my next fusion builds on c1")


async def test_a_conflicting_patch_is_written_readable_and_never_through_a_link_the_specialist_left(tmp_path):
    class _Repo:
        async def patch_text(self, head, path):
            return f"diff of {path} at {head}\n"

    head, target = "a" * 40, tmp_path / "apps-file"
    notes = tmp_path / "workspace" / "rerooted"
    (notes / head[:SHORT]).mkdir(parents=True)
    (notes / head[:SHORT] / "vllm-fix.patch").symlink_to(target)
    carried = SimpleNamespace(patch_conflicts=(PatchConflict("vllm", "patch/vllm/fix.patch", "does not apply"),))

    [written] = await _write_conflicts(_Repo(), head, carried, notes)

    assert not target.exists()
    assert written == notes / head[:SHORT] / "vllm-fix.patch" and not written.is_symlink()
    assert written.read_text() == f"diff of patch/vllm/fix.patch at {head}\n"
    assert written.stat().st_mode & 0o044 == 0o044
