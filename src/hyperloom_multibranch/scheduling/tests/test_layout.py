# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the MB- session directory: its name, its entries and its single-use creation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session

_NOW = datetime(2026, 10, 14, 8, 30, 5, tzinfo=timezone.utc).timestamp()
_CLOCK = Clock(now=lambda: _NOW)

_DIRS = (
    "MB-scheduling",
    "MB-prelude",
    "MB-notebook",
    "MB-reference",
    "MB-blobs",
    "MB-branches",
    "MB-toolbox",
    "MB-integration",
    "MB-final",
    "MB-dashboard",
    "MB-reports",
    "MB-logs",
)


def test_session_dir_is_named_by_model_utc_time_and_random_suffix(tmp_path: Path) -> None:
    layout = create_session(tmp_path, "Llama-3.1 70B/Instruct", _CLOCK)

    assert layout.root.parent == tmp_path / "Llama-3.1-70B-Instruct"
    assert re.fullmatch(r"MB-20261014T083005Z-[0-9a-f]{8}", layout.session_id)
    assert layout.session_id == layout.root.name


def test_create_session_makes_every_directory_and_no_file(tmp_path: Path) -> None:
    layout = create_session(tmp_path, "m", _CLOCK)

    assert sorted(p.name for p in layout.root.iterdir()) == sorted(_DIRS)
    assert all((layout.root / name).is_dir() for name in _DIRS)


def test_two_sessions_in_the_same_second_get_distinct_directories(tmp_path: Path) -> None:
    first = create_session(tmp_path, "m", _CLOCK)
    second = create_session(tmp_path, "m", _CLOCK)

    assert first.root != second.root


def test_an_existing_session_directory_is_never_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("secrets.token_hex", lambda _n: "deadbeef")
    create_session(tmp_path, "m", _CLOCK)

    with pytest.raises(FileExistsError):
        create_session(tmp_path, "m", _CLOCK)


def test_layout_paths() -> None:
    layout = SessionLayout(Path("/s/MB-x"))
    expected = {
        layout.manifest: "MB-manifest.json",
        layout.pids: "MB-pids.jsonl",
        layout.roster: "MB-scheduling/MB-roster.json",
        layout.decisions: "MB-scheduling/MB-decisions.jsonl",
        layout.judge: "MB-scheduling/MB-judge.jsonl",
        layout.prelude_launch: "MB-prelude/MB-prelude-launch.json",
        layout.prelude_child: "MB-prelude/MB-prelude-child.json",
        layout.root_recipe: "MB-prelude/MB-root-recipe.json",
        layout.prelude_log: "MB-prelude/MB-prelude.log",
        layout.merges: "MB-integration/MB-merges.jsonl",
        layout.report: "MB-reports/MB-report.json",
        layout.report_markdown: "MB-reports/MB-report.md",
        layout.scheduler_log: "MB-logs/MB-scheduler.log",
        layout.notebook: "MB-notebook",
        layout.repo: "MB-repo",
        layout.blobs: "MB-blobs",
        layout.toolbox: "MB-toolbox",
        layout.integration: "MB-integration",
        layout.dashboard: "MB-dashboard",
    }

    assert {path: str(path.relative_to(layout.root)) for path in expected} == expected


def test_branch_dir_is_one_safe_segment_under_branches() -> None:
    layout = SessionLayout(Path("/s/MB-x"))

    assert layout.branch_dir("b01") == layout.branches / "b01"
    assert layout.branch_dir("../evil/b") == layout.branches / "evil-b"
