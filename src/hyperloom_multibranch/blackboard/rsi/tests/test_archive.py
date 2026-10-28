# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A finished blackboard becomes a search tree, and the shared archive keeps each session's tree once."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.judgement import AccuracyVerdict
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import REPLAY_TARGET, BlackboardSnapshot, Probation, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import BranchDriver, CommitOrigin, LineageKind
from hyperloom_multibranch.blackboard.rsi.archive import (
    ARCHIVE_ENV,
    ARCHIVED,
    SINGLE_HOST,
    SKIPPED,
    WRITE_ONLY_ENV,
    ArchiveResult,
    TreeArchive,
    TreeDirection,
    TreeProbation,
    archive_session,
    session_start,
    similar_trees,
    tree_document,
    tree_from_document,
    tree_from_snapshot,
)
from hyperloom_multibranch.blackboard.rsi.similarity import TaskConditions
from hyperloom_multibranch.blackboard.rsi.tests.trees import HOUR, LATE_BLOOMER, TASK, branch, tree
from hyperloom_multibranch.blackboard.tests.factories import profile
from hyperloom_multibranch.blackboard.tests.sessions import METRIC


def test_the_tree_holds_each_branch_with_its_commits_tests_cost_and_times(snapshot: BlackboardSnapshot) -> None:
    searched = tree_from_snapshot(snapshot, METRIC)

    branches = {branch.name: branch for branch in searched.branches}
    s1, s2, integration = branches["s1"], branches["s2"], branches[INTEGRATION_BRANCH]
    assert [(commit.gain_pct, commit.origin) for commit in s1.commits] == [(pytest.approx(4.0), CommitOrigin.COMMIT)]
    assert [(commit.gain_pct, commit.origin) for commit in integration.commits] == [
        (pytest.approx(4.1), CommitOrigin.MERGE)
    ]
    assert (len(s1.test_times), s1.gpu_seconds, len(s2.test_times), s2.gpu_seconds) == (2, 1200.0, 3, 1800.0)
    assert integration.origin is LineageKind.ROOT and integration.parent_branch == ""
    assert s1.parent_branch == INTEGRATION_BRANCH and s1.closed_s is None
    assert s2.closed_s is not None and s2.closed_s > s2.opened_s
    assert s1.final_gain_pct == pytest.approx(4.0) and s1.tests_by(s1.test_times[0]) == 1
    assert searched.final_gain_pct == pytest.approx(4.1)
    assert (searched.noise.dispersion_pct, searched.noise.samples, searched.noise.environment) == (0.8, 2, "local")
    assert searched.task.labels["gpu"] == "MI355X" and searched.task.sizes["conc"] == 64.0
    assert dict(searched.policy_versions) == {"scheduler": "s1"}
    assert (
        searched.budget_seconds > 0 and tree_from_snapshot(snapshot, METRIC, budget_seconds=60.0).budget_seconds == 60.0
    )


def test_the_tree_keeps_the_roofline_bound_over_the_baseline(snapshot: BlackboardSnapshot) -> None:
    searched = tree_from_snapshot(snapshot, METRIC)
    unbound = dataclasses.replace(snapshot, profile=profile(roofline_ceiling={}))

    assert searched.roofline_ceiling_pct == pytest.approx(50.0)
    assert tree_from_snapshot(unbound, METRIC).roofline_ceiling_pct is None
    assert tree_from_document(tree_document(searched)) == searched


def test_the_tree_keeps_when_each_test_was_measured_and_judged_and_the_longest_root_test(
    snapshot: BlackboardSnapshot,
) -> None:
    searched = tree_from_snapshot(snapshot, METRIC)
    branches = {branch.name: branch for branch in searched.branches}
    s1, s2 = branches["s1"], branches["s2"]
    then = s1.as_of(s1.judged_times[0] if s1.judged_times else 0.0)

    assert (len(s1.measured_times or ()), len(s1.judged_times or ())) == (2, 2)
    assert (len(s2.measured_times or ()), len(s2.judged_times or ())) == (3, 2)
    assert (len(then.measured_times or ()), len(then.judged_times or ())) == (1, 1)
    assert searched.longest_root_test_seconds is None
    assert tree_from_document(tree_document(searched)) == searched


def test_the_tree_keeps_a_commit_whose_merge_failed_and_the_stalls_a_branch_was_held_through(
    snapshot: BlackboardSnapshot,
) -> None:
    start = session_start(snapshot)
    failed = tuple(
        dataclasses.replace(item, merged_into=None, first_failed_at=start + 900.0) if item.commit_id == "c1" else item
        for item in snapshot.committed
    )
    held = tuple(
        dataclasses.replace(node, probations=(Probation(start + 1500.0, 4, "c1"),)) if node.branch == "s1" else node
        for node in snapshot.lineage.branches
    )
    lineage = dataclasses.replace(snapshot.lineage, branches=held)
    searched = tree_from_snapshot(dataclasses.replace(snapshot, committed=failed, lineage=lineage), METRIC)
    s1 = next(branch for branch in searched.branches if branch.name == "s1")
    merged = next(branch for branch in tree_from_snapshot(snapshot, METRIC).branches if branch.name == "s1")

    assert [commit.merge_failed for commit in s1.commits] == [True]
    assert [commit.merge_failed for commit in merged.commits] == [False]
    assert s1.probations == (TreeProbation(1500.0, 4),)
    assert s1.as_of(1499.0).probations == () and s1.as_of(1500.0).probations == s1.probations
    assert tree_from_document(tree_document(searched)) == searched


def test_a_branch_as_of_a_time_has_only_what_it_had_by_then() -> None:
    times = (0.0, 1800.0, 3600.0, 7200.0)
    grown = branch("s1", LATE_BLOOMER, test_times=times, closed_s=7200.0)
    then = grown.as_of(3600.0)

    assert [commit.t_s for commit in then.commits] == [1800.0, 3600.0] and then.final_gain_pct == pytest.approx(2.3)
    assert then.test_times == (0.0, 1800.0, 3600.0)
    assert then.closed_s is None and then.gpu_seconds == pytest.approx(3 * HOUR)
    assert grown.as_of(7200.0).closed_s == 7200.0


def test_a_tree_as_of_a_time_holds_the_branches_opened_by_then_and_the_best_gain_reached_by_then() -> None:
    grown = tree("a", branch("s1", LATE_BLOOMER, test_times=(0.0, 3600.0)), branch("s2", [], opened_s=5400.0))

    then = grown.as_of(3600.0)

    assert [item.name for item in then.branches] == [INTEGRATION_BRANCH, "s1"]
    assert then.final_gain_pct == pytest.approx(2.3) and grown.final_gain_pct == pytest.approx(12.0)
    assert then.branches[1] == grown.branches[1].as_of(3600.0)


def test_the_tree_keeps_when_its_session_started_and_a_tree_archived_without_it_is_the_same_tree(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    searched = tree_from_snapshot(snapshot, METRIC)
    archive = TreeArchive(tmp_path / "archive")
    archive.put(dataclasses.replace(searched, started_at=None))

    archive.put(searched)

    assert searched.started_at == session_start(snapshot)
    assert archive.load()[0].started_at is None


def test_each_branch_keeps_why_it_stopped_and_the_directions_it_tried(snapshot: BlackboardSnapshot) -> None:
    branches = {branch.name: branch for branch in tree_from_snapshot(snapshot, METRIC).branches}
    s1, s2 = branches["s1"], branches["s2"]

    def words(direction: TreeDirection) -> TreeDirection:
        return dataclasses.replace(direction, family_target="", family_mechanism="")

    assert (s1.stop_reason, s2.stop_reason) == ("", "no progress")
    assert tuple(map(words, s1.directions)) == (
        TreeDirection("moe gemm", "tuned kernel", IntentLayer.KERNEL, 2, 1, pytest.approx(4.0)),
    )
    assert tuple(map(words, s2.directions)) == (
        TreeDirection("attention", "split kv", IntentLayer.CONFIG, 1, 0, pytest.approx(-2.0)),
        TreeDirection("scheduler", "chunked prefill", IntentLayer.CONFIG, 1, 0, None),
        TreeDirection("kv cache", "fp8 kv cache", IntentLayer.CONFIG, 1, 0, None),
    )
    assert [(item.family_target, item.family_mechanism) for item in s2.directions] == [
        ("attention", "kernel tuning"),
        ("scheduling and batching", "other"),
        ("KV cache", "quantization precision"),
    ]
    assert branches[INTEGRATION_BRANCH].directions == ()


def test_a_replay_of_a_champion_built_on_another_image_is_marked_in_the_tree(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    fingerprint, intent = next(
        (key, item) for key, item in snapshot.intents.items() if item.mechanism == "tuned kernel"
    )
    replayed = dataclasses.replace(intent, target=REPLAY_TARGET, mechanism="inference:older")
    renamed = dataclasses.replace(snapshot, intents={**snapshot.intents, fingerprint: replayed})
    archive = TreeArchive(tmp_path)

    archive_session(renamed, METRIC, archive, cross_image={"inference:older"})
    (marked,) = (branch for branch in archive.load()[0].branches if branch.name == "s1")
    (plain,) = (branch for branch in tree_from_snapshot(renamed, METRIC).branches if branch.name == "s1")

    assert [(item.mechanism, item.cross_image) for item in marked.directions] == [("inference:older", True)]
    assert [item.cross_image for item in plain.directions] == [False]


def test_a_gain_judged_without_its_eval_is_its_directions_best_gain_only_below_its_bar(
    snapshot: BlackboardSnapshot,
) -> None:
    def unmeasured(record: TestRecord) -> TestRecord:
        if record.judgement is None:
            return record
        return dataclasses.replace(
            record, judgement=dataclasses.replace(record.judgement, accuracy=AccuracyVerdict.NOT_MEASURED)
        )

    unjudged = dataclasses.replace(snapshot, records=tuple(map(unmeasured, snapshot.records)))
    branches = tree_from_snapshot(unjudged, METRIC).branches
    best = {(branch.name, item.target): item.best_gain_pct for branch in branches for item in branch.directions}

    assert best == {
        ("s1", "moe gemm"): 0.3,
        ("s2", "attention"): -2.0,
        ("s2", "scheduler"): None,
        ("s2", "kv cache"): None,
    }


def test_a_tree_written_before_stop_reasons_and_directions_still_reads(snapshot: BlackboardSnapshot) -> None:
    searched = tree_from_snapshot(snapshot, METRIC)
    document = tree_document(searched)
    for stored in document["tree"]["branches"]:  # type: ignore[index]
        del stored["stop_reason"], stored["directions"]

    older = tree_from_document(document)

    assert all(branch.stop_reason == "" and branch.directions == () for branch in older.branches)
    assert [branch.commits for branch in older.branches] == [branch.commits for branch in searched.branches]


def test_each_branch_keeps_who_drove_it_and_a_tree_written_before_reads_as_all_specialist(
    snapshot: BlackboardSnapshot,
) -> None:
    lineage = snapshot.lineage
    nodes = tuple(
        dataclasses.replace(node, driver=BranchDriver.AUTONOMOUS) if node.branch == "s2" else node
        for node in lineage.branches
    )
    driven = dataclasses.replace(snapshot, lineage=dataclasses.replace(lineage, branches=nodes))
    searched = tree_from_snapshot(driven, METRIC)
    document = tree_document(searched)
    for stored in document["tree"]["branches"]:  # type: ignore[index]
        del stored["driver"]

    drivers = {branch.name: (branch.driver, branch.steered) for branch in searched.branches}
    assert drivers == {
        INTEGRATION_BRANCH: (BranchDriver.SPECIALIST, False),
        "s1": (BranchDriver.SPECIALIST, True),
        "s2": (BranchDriver.AUTONOMOUS, False),
    }
    assert tree_from_document(tree_document(searched)) == searched
    assert all(branch.driver is BranchDriver.SPECIALIST for branch in tree_from_document(document).branches)


def test_a_baseline_read_from_the_prelude_childs_run_counts_as_one_sample_and_paces_by_it(
    snapshot: BlackboardSnapshot,
) -> None:
    assert snapshot.baseline is not None
    child_run = dataclasses.replace(snapshot.baseline, request_ids=(), source="prelude-child:r1", wall_seconds=5288.0)

    searched = tree_from_snapshot(dataclasses.replace(snapshot, baselines=(child_run,)), METRIC)

    assert (searched.noise.samples, searched.longest_root_test_seconds) == (1, 5288.0)


@pytest.mark.parametrize(("merge_node", "environment"), [("node-1", SINGLE_HOST), ("node-2", "hosted")])
def test_hosted_measurement_on_one_node_is_a_single_host(
    snapshot: BlackboardSnapshot, merge_node: str, environment: str
) -> None:
    def hosted(record: TestRecord) -> TestRecord:
        if record.result is None or record.result.sandbox is None:
            return record
        node = merge_node if record.request_id == "m1" else "node-1"
        moved = dataclasses.replace(record.result.sandbox, node_name=node, backend="hosted")
        return dataclasses.replace(record, result=dataclasses.replace(record.result, sandbox=moved))

    measured = dataclasses.replace(snapshot, records=tuple(map(hosted, snapshot.records)))

    assert tree_from_snapshot(measured, METRIC).noise.environment == environment


def test_a_tree_needs_the_profile_the_baseline_and_a_baselined_metric(snapshot: BlackboardSnapshot) -> None:
    with pytest.raises(NotRecorded):
        tree_from_snapshot(dataclasses.replace(snapshot, profile=None), METRIC)
    with pytest.raises(ValueError, match="no ttft"):
        tree_from_snapshot(snapshot, "ttft")


def test_the_archive_keeps_each_tree_once_and_skips_what_it_cannot_read(
    snapshot: BlackboardSnapshot, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    searched = tree_from_snapshot(snapshot, METRIC)
    archive = TreeArchive(tmp_path / "archive")

    archive.put(searched)
    archive.put(searched)
    with pytest.raises(FileExistsError):
        archive.put(dataclasses.replace(searched, final_gain_pct=1.0))
    with pytest.raises(ValueError, match="cannot name"):
        archive.put(dataclasses.replace(searched, session_id="../escape"))
    (archive.root / "broken.json").write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.blackboard.rsi.archive"):
        assert archive.load() == [searched]
    assert "broken.json" in caplog.text
    assert tree_from_document(tree_document(searched)) == searched


def test_close_archives_the_session_once_with_its_budget(snapshot: BlackboardSnapshot, tmp_path: Path) -> None:
    archive = TreeArchive(tmp_path / "archive")

    first = archive_session(snapshot, METRIC, archive, budget_seconds=7200.0)
    again = archive_session(snapshot, METRIC, archive, budget_seconds=7200.0)

    assert first == again == ArchiveResult(ARCHIVED, session_id="sess-1")
    assert archive.load() == [tree_from_snapshot(snapshot, METRIC, budget_seconds=7200.0)]


def test_close_archives_in_the_archive_the_environment_names(
    snapshot: BlackboardSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path))

    assert archive_session(snapshot, METRIC).status == ARCHIVED
    assert (tmp_path / "trees" / "sess-1.json").is_file()


def test_close_says_why_it_archived_nothing(
    snapshot: BlackboardSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)
    archive = TreeArchive(tmp_path / "archive")
    archive_session(snapshot, METRIC, archive, budget_seconds=7200.0)
    blocked = TreeArchive(tmp_path / "a-file")
    (tmp_path / "a-file").write_text("not a folder", encoding="utf-8")

    assert archive_session(snapshot, METRIC) == ArchiveResult(SKIPPED, "no_archive_configured")
    assert archive_session(dataclasses.replace(snapshot, profile=None), METRIC, archive).reason == (
        "no_profile_or_baseline"
    )
    assert archive_session(snapshot, "ttft", archive) == ArchiveResult(SKIPPED, "baseline_lacks_metric", "sess-1")
    assert archive_session(snapshot, METRIC, archive, budget_seconds=60.0) == ArchiveResult(
        SKIPPED, "another_tree_archived", "sess-1"
    )
    assert archive_session(snapshot, METRIC, blocked).reason == "archive_unwritable"
    renamed = dataclasses.replace(snapshot, profile=dataclasses.replace(snapshot.profile, session_id="../escape"))
    assert archive_session(renamed, METRIC, archive).reason == "session_id_not_a_file_name"


def test_the_archive_lives_where_the_deployment_says(tmp_path: Path) -> None:
    archive = TreeArchive.from_env({ARCHIVE_ENV: str(tmp_path)})

    assert archive is not None and archive.root == tmp_path / "trees"
    assert TreeArchive.from_env({}) is None


def test_a_write_only_session_reads_no_archive_and_still_archives_into_it(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    env = {ARCHIVE_ENV: str(tmp_path), WRITE_ONLY_ENV: "1"}
    written = TreeArchive.from_env(env)

    assert TreeArchive.read_from_env(env) is None and written is not None
    assert TreeArchive.read_from_env({**env, WRITE_ONLY_ENV: "0"}) is not None
    assert archive_session(snapshot, METRIC, written).status == ARCHIVED
    assert [item.session_id for item in written.load()] == ["sess-1"]


def test_similar_trees_leave_out_the_unrelated_and_put_the_closest_first() -> None:
    longer = tree("longer", task=dataclasses.replace(TASK, sizes={**TASK.sizes, "isl": 4096.0}))
    unrelated = tree("unrelated", task=TaskConditions(labels={"region": "x"}))

    ranked = similar_trees([longer, unrelated, tree("same")], TASK)

    assert [item.tree.session_id for item in ranked] == ["same", "longer"]
    assert ranked[0].weight == pytest.approx(1.0)
