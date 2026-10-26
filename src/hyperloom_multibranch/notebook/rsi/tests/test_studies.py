# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command that runs every study on what past sessions left."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeArchive
from hyperloom_multibranch.notebook.rsi.replay import ARCHIVE, OLD_RULE
from hyperloom_multibranch.notebook.rsi.studies import main
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import scripted_session


def _wide(session_id: str) -> SearchTree:
    """Four branches opened at once, two of which ever commit."""

    tested = (900.0, 1800.0, 2700.0)
    paying = (
        branch("s1", [(1800.0, 5.0)], judged_times=tested),
        branch("s2", [(1800.0, 3.0)], judged_times=tested),
    )
    return tree(session_id, *paying, branch("s3", [], judged_times=tested), branch("s4", [], judged_times=tested))


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    for session_id in ("a", "b"):
        TreeArchive(root).put(_wide(session_id))
    return root


def test_the_command_runs_the_calibration_and_the_scheduling_replay_on_an_archive(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "frontier.jsonl"
    log.write_text("", encoding="utf-8")

    assert main(["frontier", "--archive", str(archive), "--log", f"a={log}"]) == 0
    calibrated = capsys.readouterr().out
    assert main(["--json", "scheduling", "--archive", str(archive), "--runs", "2", "--beyond", "pessimistic"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert main(["scheduling", "--archive", str(archive), "--runs", "1"]) == 0
    table = capsys.readouterr().out

    assert calibrated.startswith("Frontier calibration") and "Too little evidence" in calibrated
    assert "| bandit-rv |" in calibrated and "| bandit UCB, C=1.4 |" in calibrated
    assert [row["label"] for row in replayed["rows"]][:4] == [ARCHIVE, "h0", OLD_RULE, "stall-4"]
    assert replayed["baseline"] == "h0"
    assert (replayed["runs"], replayed["beyond"], replayed["sessions"]) == (2, "pessimistic", ["a", "b"])
    assert table.startswith("Scheduling replay over 2 sessions, 1 runs each") and "| cap-6 |" in table
    assert main(["replay-check", "--archive", str(archive), "--runs", "1"]) == 0
    assert capsys.readouterr().out.startswith("Following the archive reproduces 2 of 2 sessions")


def test_the_command_reads_the_stall_odds_and_every_sessions_stops_off_one_archive_or_more(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    more = tmp_path / "more"
    TreeArchive(more).put(_wide("c"))

    assert main(["stalls", "--archive", str(archive), "--archive", str(more), "--archive", str(archive)]) == 0
    text = capsys.readouterr().out
    assert main(["--json", "stalls", "--archive", str(archive)]) == 0
    study = json.loads(capsys.readouterr().out)

    rows = [line.split(" | ")[0] for line in text.splitlines() if line.startswith(("| a ", "| b ", "| c "))]
    assert text.startswith("## any model type, inferencex") and "Over the archive the judge stopped 0 branches" in text
    assert rows == ["| a", "| b", "| c"]
    assert [audit["session_id"] for audit in study["audits"]] == ["a", "b"]


def test_a_study_that_cannot_run_says_why(archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scheduling", "--archive", str(tmp_path / "empty")]) == 2
    assert "the archives hold no session to replay" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["frontier", "--archive", str(archive), "--log", "no-separator"])
    with pytest.raises(SystemExit):
        main(["scheduling", "--archive", str(archive), "--beyond", "anywhere"])


def test_the_command_evaluates_a_sessions_learnings_and_the_dispersion_prior(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asyncio.run(scripted_session(tmp_path / "notebook", FakeClock()))
    log, shadow, labels = tmp_path / "learnings.jsonl", tmp_path / "consistency.jsonl", tmp_path / "labels.json"
    log.write_text(json.dumps({"at": 0.0, "branch": "s1", "source": "tool", "here": [], "past": []}) + "\n", "utf-8")
    item = {"type": "item", "id": "t1", "compared": True, "throughput_diff_signed_pct": -1.5, "hosted": {"cv_pct": 0.9}}
    shadow.write_text(json.dumps(item) + "\n", encoding="utf-8")
    labels.write_text(json.dumps({"t1": {"model": "m", "tp": 8, "gpu": "MI355X"}}), encoding="utf-8")

    assert main(["learnings", "--notebook", str(tmp_path / "notebook"), "--log", str(log)]) == 0
    learnt = capsys.readouterr().out
    prior_args = ["--shadow", str(shadow), "--labels", str(labels), "--model", "m", "--tp", "8", "--gpu", "MI355X"]
    assert main(["--json", "dispersion", *prior_args]) == 0
    prior = json.loads(capsys.readouterr().out)
    assert main(["dispersion", *prior_args]) == 0

    assert learnt.startswith("Learnings retrieval: 1 retrievals (tool 1)")
    assert (prior["prior"]["source"], prior["prior"]["dispersion_pct"], prior["prior"]["offset_pct"]) == (
        "shadow",
        0.9,
        -1.5,
    )
    assert "| prior | 0.90% from shadow (hosted) |" in capsys.readouterr().out
