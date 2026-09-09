# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the periodic Coordinator maintenance tick.

Maintenance runs on a long-horizon loop and touches the things that wedge a
run: expired leases, tasks whose worker died holding a lane, unbounded DB
growth, and a session partition filling up. So the property that matters is
that no single step can end the run -- each is independently guarded, and a
failure has to leave the others' results in the summary.

The disk trim is the destructive one, so its trigger and its retention are
pinned separately: it must do nothing while the partition is healthy, and when
it does fire it must keep the most recent work per action.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.loop import maintenance as maint
from hyperloom.orchestrator.loop.maintenance import (
    MaintenanceCollaborator,
    run_lease_and_db_reclaim,
)


class _Reconciler:
    """The pass that owns the serving-lease sweep; maintenance only reports it.

    An open bring-up round holds one of those leases and can only be settled in
    the reconciler, so a second sweep here would reap a lease a live round still
    holds. ``raises`` stands in for a report this pass could not produce.
    """

    def __init__(self, reaped=0, raises=False):
        self.last_report = None if raises else SimpleNamespace(leases_reaped=reaped)


class _Pool:
    def __init__(self, count=0, raises=False):
        self._count, self._raises = count, raises

    async def reap_expired(self):
        if self._raises:
            raise RuntimeError("pool gone")
        return self._count


class _Tasks:
    def __init__(self, reclaimed=(), raises=False):
        self._reclaimed, self._raises = reclaimed, raises
        self.reason = None

    async def reclaim_expired_running(self, *, reason):
        self.reason = reason
        if self._raises:
            raise RuntimeError("task table locked")
        return list(self._reclaimed)


def _host(**kw):
    return SimpleNamespace(
        reconciler=kw.get("reconciler", _Reconciler(reaped=2)),
        gpu_specialist_pool=kw.get("pool", _Pool(count=3)),
        tasks=kw.get("tasks", _Tasks(reclaimed=["t1"])),
        db=kw.get("db", object()),
    )


def _patch_retention(monkeypatch: pytest.MonkeyPatch, *, events=5, tasks=2, raises=False):
    from hyperloom.orchestrator.bus import db_maintenance as db_maint

    async def _run(_db):
        if raises:
            raise RuntimeError("vacuum failed")
        return SimpleNamespace(events_deleted=events, tasks_deleted=tasks)

    monkeypatch.setattr(db_maint, "run_db_retention", _run)


class TestReclaimReportsWhatEachStepDid:
    @pytest.mark.asyncio
    async def test_a_clean_pass_records_every_count(self, monkeypatch: pytest.MonkeyPatch):
        _patch_retention(monkeypatch)
        summary: dict = {}

        await run_lease_and_db_reclaim(_host(), summary, reason="maintenance_watchdog")

        assert summary == {
            "leases_reaped": 2,
            "gpu_leases_reaped": 3,
            "running_tasks_reclaimed": 1,
            "events_pruned": 5,
            "tasks_pruned": 2,
        }

    @pytest.mark.asyncio
    async def test_the_reason_reaches_the_task_reclaim(self, monkeypatch: pytest.MonkeyPatch):
        """The R6 watchdog records why a running task was failed."""
        _patch_retention(monkeypatch)
        tasks = _Tasks()

        await run_lease_and_db_reclaim(_host(tasks=tasks), {}, reason="cycle_soft_restart")

        assert tasks.reason == "cycle_soft_restart"


class TestNoSingleStepCanEndTheRun:
    @pytest.mark.asyncio
    async def test_an_unreadable_lease_sweep_leaves_the_other_steps_intact(self, monkeypatch: pytest.MonkeyPatch):
        _patch_retention(monkeypatch)
        summary: dict = {}

        await run_lease_and_db_reclaim(_host(reconciler=_Reconciler(raises=True)), summary, reason="r")

        assert "leases_reaped" not in summary
        assert summary["gpu_leases_reaped"] == 3
        assert summary["events_pruned"] == 5

    @pytest.mark.asyncio
    async def test_a_failed_gpu_lease_reap_is_survived(self, monkeypatch: pytest.MonkeyPatch):
        _patch_retention(monkeypatch)
        summary: dict = {}

        await run_lease_and_db_reclaim(_host(pool=_Pool(raises=True)), summary, reason="r")

        assert "gpu_leases_reaped" not in summary
        assert summary["leases_reaped"] == 2

    @pytest.mark.asyncio
    async def test_a_failed_task_reclaim_is_survived(self, monkeypatch: pytest.MonkeyPatch):
        _patch_retention(monkeypatch)
        summary: dict = {}

        await run_lease_and_db_reclaim(_host(tasks=_Tasks(raises=True)), summary, reason="r")

        assert "running_tasks_reclaimed" not in summary
        assert summary["tasks_pruned"] == 2

    @pytest.mark.asyncio
    async def test_a_failed_db_retention_is_survived(self, monkeypatch: pytest.MonkeyPatch):
        _patch_retention(monkeypatch, raises=True)
        summary: dict = {}

        await run_lease_and_db_reclaim(_host(), summary, reason="r")

        assert "events_pruned" not in summary
        assert summary["running_tasks_reclaimed"] == 1

    @pytest.mark.asyncio
    async def test_a_pass_that_swept_nothing_counts_as_zero(self, monkeypatch: pytest.MonkeyPatch):
        """A reconciler that ran and found nothing reports 0, not an absent key."""
        _patch_retention(monkeypatch)
        summary: dict = {}

        await run_lease_and_db_reclaim(_host(reconciler=_Reconciler(reaped=0)), summary, reason="r")

        assert summary["leases_reaped"] == 0


def _coordinator(session_dir: Path, **kw):
    """A stand-in exposing exactly what the collaborator reads off its host."""
    return SimpleNamespace(
        session_dir=session_dir,
        reconciler=_Reconciler(),
        gpu_specialist_pool=_Pool(),
        tasks=_Tasks(),
        db=object(),
        _STATE_JSON_WARN_BYTES=kw.get("warn_bytes", 50 * 1024 * 1024),
        _DISK_FREE_MIN_GB=kw.get("free_min_gb", 20.0),
        _DISK_USED_MAX_FRAC=kw.get("used_max_frac", 0.85),
        _DISK_RUNS_KEEP_PER_ACTION=kw.get("keep", 2),
    )


def _fake_usage(monkeypatch: pytest.MonkeyPatch, *, free_gb: float, used_frac: float, raises=False):
    total = 1000 * 1024**3

    def _usage(_path):
        if raises:
            raise OSError("no such partition")
        return SimpleNamespace(free=int(free_gb * 1024**3), used=int(total * used_frac), total=total)

    monkeypatch.setattr(shutil, "disk_usage", _usage)


class TestTheDiskTrimOnlyFiresWhenItHasTo:
    def test_an_unreadable_partition_is_not_an_error(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _fake_usage(monkeypatch, free_gb=0, used_frac=0, raises=True)
        c = MaintenanceCollaborator(_coordinator(tmp_path))

        assert c._maybe_prune_runs_for_disk() is None

    def test_a_healthy_partition_reports_but_deletes_nothing(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _fake_usage(monkeypatch, free_gb=500.0, used_frac=0.10)
        runs = tmp_path / "runs" / "explore"
        for i in range(5):
            (runs / f"task{i}").mkdir(parents=True)
        c = MaintenanceCollaborator(_coordinator(tmp_path))

        got = c._maybe_prune_runs_for_disk()

        assert got is not None
        assert "runs_pruned" not in got
        assert len(list(runs.iterdir())) == 5

    def test_low_free_space_keeps_only_the_most_recent_per_action(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _fake_usage(monkeypatch, free_gb=1.0, used_frac=0.10)
        runs = tmp_path / "runs" / "explore"
        for i in range(5):
            d = runs / f"task{i}"
            d.mkdir(parents=True)
            import os

            os.utime(d, (1_700_000_000 + i * 100, 1_700_000_000 + i * 100))
        (tmp_path / "runs" / "loose_file.txt").write_text("not an action dir", encoding="utf-8")
        c = MaintenanceCollaborator(_coordinator(tmp_path, keep=2))

        got = c._maybe_prune_runs_for_disk()

        assert got["runs_pruned"] == 3
        assert sorted(p.name for p in runs.iterdir()) == ["task3", "task4"]

    def test_a_full_partition_triggers_the_trim_even_with_free_space(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _fake_usage(monkeypatch, free_gb=500.0, used_frac=0.99)
        runs = tmp_path / "runs" / "explore"
        for i in range(4):
            (runs / f"task{i}").mkdir(parents=True)
        c = MaintenanceCollaborator(_coordinator(tmp_path, keep=1))

        assert c._maybe_prune_runs_for_disk()["runs_pruned"] == 3

    def test_an_action_within_the_keep_budget_is_untouched(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _fake_usage(monkeypatch, free_gb=1.0, used_frac=0.99)
        runs = tmp_path / "runs" / "explore"
        (runs / "only_task").mkdir(parents=True)
        c = MaintenanceCollaborator(_coordinator(tmp_path, keep=2))

        assert c._maybe_prune_runs_for_disk()["runs_pruned"] == 0
        assert (runs / "only_task").is_dir()

    def test_a_session_with_no_runs_tree_yet_reports_the_usage_only(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _fake_usage(monkeypatch, free_gb=1.0, used_frac=0.99)
        c = MaintenanceCollaborator(_coordinator(tmp_path))

        got = c._maybe_prune_runs_for_disk()

        assert "runs_pruned" not in got
        assert got["free_gb"] == 1.0

    def test_an_oversized_state_json_is_warned_about_not_deleted(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        _fake_usage(monkeypatch, free_gb=500.0, used_frac=0.10)
        from hyperloom.orchestrator.state.shared_state import SharedState

        state_path = SharedState.state_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("x" * 4096, encoding="utf-8")
        c = MaintenanceCollaborator(_coordinator(tmp_path, warn_bytes=1024))

        with caplog.at_level("WARNING"):
            c._maybe_prune_runs_for_disk()

        assert "state.json is" in caplog.text
        assert state_path.is_file(), "the warning must not delete state"

    def test_a_state_file_that_cannot_be_stat_ed_does_not_block_the_trim(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """The size warning races a concurrent prune; it must never be fatal."""
        _fake_usage(monkeypatch, free_gb=1.0, used_frac=0.99)
        from hyperloom.orchestrator.state.shared_state import SharedState

        class _Unstattable:
            def is_file(self):
                raise OSError("vanished mid-check")

        monkeypatch.setattr(SharedState, "state_path", staticmethod(lambda _d: _Unstattable()))
        runs = tmp_path / "runs" / "explore"
        for i in range(3):
            (runs / f"task{i}").mkdir(parents=True)
        c = MaintenanceCollaborator(_coordinator(tmp_path, keep=1))

        assert c._maybe_prune_runs_for_disk()["runs_pruned"] == 2

    def test_a_workspace_that_refuses_to_delete_is_logged_not_raised(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        _fake_usage(monkeypatch, free_gb=1.0, used_frac=0.99)
        runs = tmp_path / "runs" / "explore"
        for i in range(3):
            (runs / f"task{i}").mkdir(parents=True)

        def _refuse(*_a, **_k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(shutil, "rmtree", _refuse)
        c = MaintenanceCollaborator(_coordinator(tmp_path, keep=1))

        with caplog.at_level("WARNING"):
            got = c._maybe_prune_runs_for_disk()

        assert got["runs_pruned"] == 0
        assert "failed to prune" in caplog.text


class TestTheTickItself:
    @pytest.mark.asyncio
    async def test_the_summary_carries_the_tick_and_the_disk_status(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _patch_retention(monkeypatch)
        _fake_usage(monkeypatch, free_gb=500.0, used_frac=0.10)
        c = MaintenanceCollaborator(_coordinator(tmp_path))

        got = await c._run_maintenance(tick=11)

        assert got["tick"] == 11
        assert got["disk"]["free_gb"] == 500.0
        assert got["events_pruned"] == 5

    @pytest.mark.asyncio
    async def test_a_failing_disk_monitor_does_not_lose_the_rest_of_the_tick(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        _patch_retention(monkeypatch)
        c = MaintenanceCollaborator(_coordinator(tmp_path))
        monkeypatch.setattr(
            maint.MaintenanceCollaborator,
            "_maybe_prune_runs_for_disk",
            lambda _self: (_ for _ in ()).throw(RuntimeError("statvfs exploded")),
        )

        got = await c._run_maintenance(tick=3)

        assert "disk" not in got
        assert got["tick"] == 3
        assert got["events_pruned"] == 5

    def test_unknown_attributes_fall_through_to_the_coordinator(self, tmp_path):
        coord = _coordinator(tmp_path)
        coord.some_coordinator_only_thing = "reachable"

        assert MaintenanceCollaborator(coord).some_coordinator_only_thing == "reachable"
