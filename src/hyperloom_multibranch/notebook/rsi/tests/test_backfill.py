# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A finished single-lineage session becomes a one-branch search tree in the archive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.backfill import (
    JOURNAL_PATH,
    WORK_BRANCH,
    backfill,
    noise_from_breakdown,
    task_from_manifest,
    tree_from_hyperloom_session,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_similarity
from hyperloom_multibranch.notebook.tests.sessions import METRIC

TASK_CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1",
    "gpu_type": "MI355X",
    "framework_name": "vllm",
    "architecture": {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]},
    "framework_version": "0.11.0",
    "precision": "fp8",
    "tp": 8,
    "conc": 64,
    "isl": 1024,
    "osl": 1024,
}
JOURNAL = {
    "session_id": "old-1",
    "baseline_throughput": 1000.0,
    "entries": [
        {"ts": "2026-01-01T00:00:00Z", "kind": "baseline", "outcome": "KEEP", "throughput_after": 1000.0},
        {"ts": "2026-01-01T00:30:00Z", "kind": "config", "outcome": "KEEP", "throughput_after": 1050.0},
        {"ts": "2026-01-01T01:00:00Z", "kind": "config", "outcome": "REVERT", "throughput_after": 990.0},
        {"ts": "2026-01-01T02:00:00Z", "kind": "kernel", "outcome": "keep", "throughput_after": 1100.0},
    ],
}
BREAKDOWN = {
    "metadata": {"grading": {"tput_guard": {"noise_pct": 2.0}}},
    "timeline": [
        {
            "type": "baseline",
            "ext": {
                "actions": [
                    {"measurement": {"throughput_tok_s_per_gpu": 990.0}},
                    {"measurement": {"throughput_tok_s_per_gpu": 1010.0}},
                    {"measurement": {"throughput_tok_s_per_gpu": None}},
                ]
            },
        },
        {"type": "framework_agent"},
    ],
}


def _session(root: Path, journal: object = JOURNAL, *, manifest: bool = True, breakdown: bool = True) -> Path:
    (root / "reports").mkdir(parents=True)
    (root / JOURNAL_PATH).write_text(json.dumps(journal), encoding="utf-8")
    if manifest:
        (root / "manifest.json").write_text(json.dumps({"task_config": TASK_CONFIG}), encoding="utf-8")
    if breakdown:
        (root / "session_breakdown.json").write_text(json.dumps(BREAKDOWN), encoding="utf-8")
    return root


def test_an_old_session_is_one_work_branch_whose_keeps_are_commits(tmp_path: Path) -> None:
    searched = tree_from_hyperloom_session(_session(tmp_path / "old-1"))

    (work,) = [branch for branch in searched.branches if branch.name == WORK_BRANCH]
    assert [(commit.t_s, round(commit.gain_pct, 6)) for commit in work.commits] == [(1800.0, 5.0), (7200.0, 10.0)]
    assert (work.closed_s, len(work.test_times), searched.budget_seconds) == (7200.0, 4, 7200.0)
    assert searched.final_gain_pct == pytest.approx(10.0)
    assert searched.noise.dispersion_pct == pytest.approx(1.41421356, rel=1e-6)
    assert (searched.noise.noise_band_pct, searched.noise.samples, searched.noise.environment) == (2.0, 2, "local")


def test_an_old_session_matches_a_new_one_on_the_same_task(tmp_path: Path, snapshot: NotebookSnapshot) -> None:
    old = tree_from_hyperloom_session(_session(tmp_path / "old-1"))

    assert task_similarity(old.task, tree_from_snapshot(snapshot, METRIC).task) == pytest.approx(1.0)


def test_the_manifest_and_breakdown_may_be_missing_or_nested() -> None:
    nested = task_from_manifest(
        {"metadata": {"task_config": {"gpu_type": "MI300X", "architecture": {"model_class": "X"}}}}
    )

    assert dict(nested.labels) == {"gpu": "MI300X", "architecture": "X"}
    assert task_from_manifest({}).labels == {}
    assert noise_from_breakdown({}).dispersion_pct is None and noise_from_breakdown({}).noise_band_pct is None


def test_backfill_archives_what_it_can_and_reports_the_rest(tmp_path: Path) -> None:
    good = _session(tmp_path / "old-1", manifest=False, breakdown=False)
    unbaselined = _session(tmp_path / "old-2", {**JOURNAL, "baseline_throughput": 0})
    archive = TreeArchive(tmp_path / "archive")

    results = backfill([good, unbaselined, tmp_path / "missing"], archive)

    assert [(result.session_id, result.error.split(":")[0]) for result in results] == [
        ("old-1", ""),
        ("", "ValueError"),
        ("", "FileNotFoundError"),
    ]
    assert [tree.session_id for tree in archive.load()] == ["old-1"]
