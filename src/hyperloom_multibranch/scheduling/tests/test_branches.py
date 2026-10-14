# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch control: opening records ``OPEN`` and starts specialists, stopping withdraws everything once."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import BranchOrigin, BranchStop, Roster, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.tests.peer_fakes import ROOT_COMMIT, MbFakeLifecycle, MbFakeToolbox

NOW = 1_700_000_000.0
CLOCK = Clock(now=lambda: NOW)


class QuotaCheckingLifecycle(MbFakeLifecycle):
    """The lifecycle stand-in, noting the K the toolbox's quota held for a branch when it started."""

    def __init__(self, toolbox: MbFakeToolbox, refusing: frozenset[str]) -> None:
        super().__init__(refusing=refusing)
        self.toolbox = toolbox
        self.quota_at_start: dict[str, int | None] = {}

    async def start(self, branch: str, start_commit: str) -> None:
        self.quota_at_start[branch] = self.toolbox.quotas[-1].k_of(branch) if self.toolbox.quotas else None
        await super().start(branch, start_commit)


class Bench:
    def __init__(self, tmp_path: Path, refusing: frozenset[str] = frozenset()) -> None:
        self.notebook = LocalNotebook(tmp_path / "nb", Writer(Role.SCHEDULER, "scheduler"), clock=CLOCK.now)
        self.roster = Roster(RosterCaps(3, 2, 8), tmp_path / "MB-roster.json", CLOCK)
        self.toolbox = MbFakeToolbox(self.notebook)
        self.lifecycle = QuotaCheckingLifecycle(self.toolbox, refusing)
        self.control = BranchControl(
            self.roster,
            self.notebook,
            self.lifecycle,
            self.toolbox,
            ProcessRegistry(tmp_path / "MB-pids.jsonl", CLOCK),
            DecisionLog(tmp_path / "MB-decisions.jsonl", CLOCK),
        )

    async def rooted(self) -> Bench:
        await self.notebook.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=ROOT_COMMIT))
        self.roster.add_integration(ROOT_COMMIT, k=1)
        return self


async def test_the_opening_batch_is_recorded_started_and_given_quota(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()

    opened = await bench.control.open_branches(2, ROOT_COMMIT, 5, BranchOrigin.FIRST_BATCH)

    assert [entry.branch for entry in opened] == ["b01", "b02"]
    assert bench.lifecycle.started == {"b01": ROOT_COMMIT, "b02": ROOT_COMMIT}
    assert bench.lifecycle.k == {"b01": 2, "b02": 2}
    view = await bench.notebook.lineage()
    assert {branch.branch: branch.start_commit for branch in view.branches}["b02"] == ROOT_COMMIT
    quota = bench.toolbox.quotas[-1]
    assert (quota.k_of(INTEGRATION_BRANCH), quota.k_of("b01"), quota.k_of("b02")) == (1, 2, 2)
    assert bench.lifecycle.quota_at_start == {"b01": 2, "b02": 2}


async def test_a_fork_has_its_quota_before_its_specialist_starts(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, ROOT_COMMIT, 1, BranchOrigin.FIRST_BATCH)
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=2, from_branch="b01")

    await bench.control.start_fork("b02", "f1")

    assert bench.lifecycle.quota_at_start["b02"] == 2


async def test_a_failing_specialist_stop_still_withdraws_and_records_the_branch(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, refusing=frozenset({"b01"})).rooted()
    await bench.control.open_branches(1, ROOT_COMMIT, 1, BranchOrigin.FIRST_BATCH)

    with pytest.raises(RuntimeError, match="refused to stop b01"):
        await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert bench.toolbox.withdrawn == ["b01"]
    assert bench.toolbox.quotas[-1].k_of("b01") is None
    stops = {branch.branch: branch.stop_reason for branch in (await bench.notebook.lineage()).branches}
    assert stops["b01"] == "judge: flat"
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert logged["detail"]["failures"] == ["RuntimeError: the runtime refused to stop b01"]


async def test_stopping_a_branch_withdraws_it_once_and_its_fork_never_starts(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, ROOT_COMMIT, 1, BranchOrigin.FIRST_BATCH)
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")
    stop = BranchStop(StopCause.JUDGE, "flat", NOW)

    await bench.control.stop_branch("b01", stop)
    await bench.control.stop_branch("b01", BranchStop(StopCause.SESSION_END, "time_exhausted", NOW))
    await bench.control.stop_branch("b02", stop)

    assert bench.lifecycle.stopped == ["b01"]
    assert bench.toolbox.withdrawn == ["b01", "b02"]
    assert bench.roster.entry("b01").stop == stop
    assert await bench.control.start_fork("b02", "f1") is None
    assert "b02" not in bench.lifecycle.started
    stops = {branch.branch: branch.stop_reason for branch in (await bench.notebook.lineage()).branches}
    assert stops["b01"] == "judge: flat"
    assert bench.toolbox.quotas[-1].k_of("b01") is None


async def test_k_orders_skip_stopped_branches_and_clamp(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(2, ROOT_COMMIT, 1, BranchOrigin.FIRST_BATCH)
    await bench.control.stop_branch("b02", BranchStop(StopCause.JUDGE, "flat", NOW))

    await bench.control.apply_k({"b01": 9, "b02": 2, INTEGRATION_BRANCH: 2})

    assert bench.lifecycle.k == {"b01": 2, "b02": 1}
    assert bench.toolbox.quotas[-1].k_of(INTEGRATION_BRANCH) == 2
