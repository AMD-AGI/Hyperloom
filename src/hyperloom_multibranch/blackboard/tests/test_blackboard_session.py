# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's profile, manifest fingerprints and Arbor's map, and the snapshot every downstream view reads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, NotRecorded, Role, WriterNotAllowed
from hyperloom_multibranch.contracts.blackboard.peers import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    LineageEvent,
    ManifestFingerprint,
    MapDirection,
)
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus, Conclusion, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import (
    TTL,
    FakeClock,
    delta,
    intent,
    judgement,
    profile,
    request,
    result,
)
from hyperloom_multibranch.blackboard.tests.sessions import scripted_session

Open = Callable[..., LocalBlackboard]
GEMM = MapDirection("m-gemm", "Dense GEMMs: tuned kernels", "open", holder="b02", ops=("gemm_a8w8.kd",))
FP8 = MapDirection("m-fp8-weights", "FP8 weights", "killed", "pays only by changing the precision of record")


async def test_the_scheduler_writes_the_profile_and_the_latest_one_is_in_force(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)

    with pytest.raises(NotRecorded):
        await scheduler.session_profile()
    await scheduler.write_session_profile(profile())
    await scheduler.write_session_profile(profile(conc=128))

    assert (await open_as(Role.GATE).session_profile()).conc == 128
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.TOOLBOX).write_session_profile(profile())
    with pytest.raises(ValueError):
        await scheduler.write_session_profile(profile(session_id="has spaces"))


async def test_manifest_fingerprints_keep_every_activity_in_order(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)
    first = ManifestFingerprint("act-1", {"run_test": "hosted-1.2"}, {"specialist": "p7"}, {"scheduler": "s3"})
    second = ManifestFingerprint("act-2", {"run_test": "hosted-1.3"})

    await scheduler.write_manifest_fingerprint(first)
    await scheduler.write_manifest_fingerprint(second)

    assert await open_as(Role.OBSERVER).manifest_fingerprints() == (first, second)
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.GATE).write_manifest_fingerprint(first)


async def test_the_scheduler_writes_arbors_map_and_every_reader_sees_the_latest_one(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)
    fuse = MapDirection("m-fuse", "Small kernels folded into their producers", "open", holder="b03")

    await scheduler.write_agent_map([GEMM, FP8])
    first = (await open_as(Role.OBSERVER).snapshot()).agent_map
    await scheduler.write_agent_map((GEMM, fuse))

    assert first == (GEMM, FP8)
    assert (await open_as(Role.GATE).snapshot()).agent_map == (GEMM, fuse)


@pytest.mark.parametrize(
    ("role", "branch"),
    [(Role.TOOLBOX, ""), (Role.GATE, ""), (Role.INTEGRATOR, ""), (Role.OBSERVER, ""), (Role.SPECIALIST_RUNTIME, "b02")],
)
async def test_only_the_scheduler_writes_arbors_map(open_as: Open, role: Role, branch: str) -> None:
    with pytest.raises(WriterNotAllowed):
        await open_as(role, branch).write_agent_map([GEMM])


@pytest.mark.parametrize(
    "directions",
    [
        [GEMM, MapDirection("", "A direction without an id", "open")],
        [GEMM, FP8, MapDirection("m-gemm", "Again", "done")],
    ],
    ids=["no_id", "same_id"],
)
async def test_a_map_with_a_direction_without_an_id_or_with_anothers_is_refused_and_not_kept(
    open_as: Open, directions: list[MapDirection]
) -> None:
    scheduler = open_as(Role.SCHEDULER)

    with pytest.raises(ValueError, match="id"):
        await scheduler.write_agent_map(directions)
    assert (await scheduler.snapshot()).agent_map == ()


async def test_a_blackboard_written_without_arbors_map_reads_an_empty_one_held_by_nobody(tmp_path: Path) -> None:
    snap = await (await scripted_session(tmp_path / "blackboard", FakeClock())).snapshot()

    assert (snap.agent_map, snap.map_holders()) == ((), {})


async def test_an_open_direction_is_held_while_its_holder_runs_and_free_once_it_stops(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    for branch in ("b02", "b03"):
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="c0"))
    await scheduler.write_agent_map(
        [
            GEMM,
            MapDirection("m-fuse", "Small kernels folded into their producers", "open", holder="b03"),
            MapDirection("m-attn", "Decode attention", "open", holder="b09"),
            MapDirection("m-moe", "MoE routing", "open"),
            replace(FP8, holder="b02"),
        ]
    )

    running = (await scheduler.snapshot()).map_holders()
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "b03", reason="its agent finished"))
    stopped = (await scheduler.snapshot()).map_holders()

    assert running == {"m-gemm": "b02", "m-fuse": "b03"}
    assert stopped == {"m-gemm": "b02"}


async def test_a_snapshot_holds_every_view_with_the_times_the_facts_arrived(open_as: Open, clock: FakeClock) -> None:
    scheduler, toolbox, gate = open_as(Role.SCHEDULER), open_as(Role.TOOLBOX), open_as(Role.GATE)
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")
    await scheduler.write_session_profile(profile())
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await gate.write_baseline(
        BaselineRecord("c0", {"output_tps": 1000.0}, {"output_tps": 0.8}, {"output_tps": 1.6}, ())
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    target = intent("s1", "c0")
    claim = await s1.acquire(target, delta().fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request("r1", target))
    await toolbox.attach_request(claim.key, "r1", ttl_seconds=TTL)
    await toolbox.write_result(result("r1"))
    await gate.write_judgement(
        "r1", judgement(4.0, ("r1",), recorded={"output_tps": 1040.0}), Conclusion.COMMITTED, "c1"
    )
    await s1.acquire(
        intent("s1", "c1", "attention", "split kv"), delta("--block-size 32").fingerprint(), ttl_seconds=TTL
    )
    await s1.append_branch_log(BranchLogEntry("s1", BranchLogType.PLAN, "try split kv next"))
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s1", reason="budget"))
    clock.advance(2 * TTL)

    snap = await open_as(Role.OBSERVER).snapshot()

    (record,) = snap.records
    assert record.requested_at < record.measured_at < record.judged_at <= snap.taken_at
    commits = {node.commit_id: node for node in snap.lineage.commits}
    assert commits["c0"].at < commits["c1"].at == record.judged_at
    branch = {node.branch: node for node in snap.lineage.branches}["s1"]
    assert branch.opened_at < branch.stopped_at
    assert sorted(claim.status for claim in snap.claims) == [ClaimStatus.COMPLETED, ClaimStatus.EXPIRED]
    assert snap.profile == profile()
    assert dict(snap.value_of("c0")) == {"output_tps": 1000.0}
    assert dict(snap.value_of("c1")) == {"output_tps": 1040.0}
    assert snap.value_of("c-unknown") == {}
    assert snap.baseline is not None and snap.baseline.root_commit == "c0"
    assert [commit.commit_id for commit in snap.committed] == ["c1"]
    assert [row.branch for row in snap.progress] == [INTEGRATION_BRANCH, "s1"]
    assert [entry.entry.content for entry in snap.branch_logs["s1"]] == ["try split kv next"]
    assert set(snap.intents) == {target.fingerprint, intent("s1", "c1", "attention", "split kv").fingerprint}


async def test_an_empty_blackboard_has_an_empty_snapshot(open_as: Open) -> None:
    snap = await open_as(Role.OBSERVER).snapshot()

    assert (snap.profile, snap.records, snap.claims, snap.baseline) == (None, (), (), None)
    assert snap.lineage.commits == () and snap.branch_logs == {}
