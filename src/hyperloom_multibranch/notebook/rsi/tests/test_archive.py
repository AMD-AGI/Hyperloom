# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A finished notebook becomes a search tree, and the shared archive keeps each session's tree once."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.rsi.archive import (
    ARCHIVE_ENV,
    TreeArchive,
    similar_trees,
    tree_document,
    tree_from_document,
    tree_from_snapshot,
)
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import TASK, tree
from hyperloom_multibranch.notebook.tests.sessions import METRIC


def test_the_tree_holds_each_branch_with_its_commits_tests_cost_and_times(snapshot: NotebookSnapshot) -> None:
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


def test_a_tree_needs_the_profile_the_baseline_and_a_baselined_metric(snapshot: NotebookSnapshot) -> None:
    with pytest.raises(NotRecorded):
        tree_from_snapshot(dataclasses.replace(snapshot, profile=None), METRIC)
    with pytest.raises(ValueError, match="no ttft"):
        tree_from_snapshot(snapshot, "ttft")


def test_the_archive_keeps_each_tree_once_and_skips_what_it_cannot_read(
    snapshot: NotebookSnapshot, tmp_path: Path, caplog: pytest.LogCaptureFixture
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
    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.notebook.rsi.archive"):
        assert archive.load() == [searched]
    assert "broken.json" in caplog.text
    assert tree_from_document(tree_document(searched)) == searched


def test_the_archive_lives_where_the_deployment_says(tmp_path: Path) -> None:
    archive = TreeArchive.from_env({ARCHIVE_ENV: str(tmp_path)})

    assert archive is not None and archive.root == tmp_path / "trees"
    assert TreeArchive.from_env({}) is None


def test_similar_trees_leave_out_the_unrelated_and_put_the_closest_first() -> None:
    longer = tree("longer", task=dataclasses.replace(TASK, sizes={**TASK.sizes, "isl": 4096.0}))
    unrelated = tree("unrelated", task=TaskConditions(labels={"region": "x"}))

    ranked = similar_trees([longer, unrelated, tree("same")], TASK)

    assert [item.tree.session_id for item in ranked] == ["same", "longer"]
    assert ranked[0].weight == pytest.approx(1.0)
