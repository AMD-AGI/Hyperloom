# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MB2 studies: the branch cap replay (Q7), and the command that runs every study on what MB1 left."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeArchive
from hyperloom_multibranch.notebook.rsi.studies import CAPS, branch_cap_study, main
from hyperloom_multibranch.notebook.rsi.tests.trees import TASK, branch, tree
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import scripted_session


def _wide(session_id: str) -> SearchTree:
    """Four branches opened at once, two of which ever commit."""

    paying = (branch("s1", [(1800.0, 5.0)]), branch("s2", [(1800.0, 3.0)]))
    return tree(session_id, *paying, branch("s3", []), branch("s4", []))


def test_the_cap_that_replays_best_is_recommended() -> None:
    study = branch_cap_study([_wide("a"), _wide("b")], TASK, caps=(2, 3, 4))
    utilities = {row.policy.branch_cap: row.utility for row in study.rows}

    # A cap of 2 keeps both paying branches and saves the GPU time of the other two.
    assert utilities[2] == pytest.approx(5.0 + 0.5 * 5.0 / 8.0 * 8.0)
    assert utilities[2] > utilities[3] > utilities[4] == pytest.approx(5.0)
    assert study.recommended == 2 and study.note.startswith("Consider a branch cap of 2")


def test_of_caps_that_replay_alike_the_smaller_is_recommended() -> None:
    narrow = [tree("a", branch("s1", [(1800.0, 5.0)])), tree("b", branch("s1", [(1800.0, 4.0)]))]

    assert branch_cap_study(narrow, TASK, caps=(4, 2)).recommended == 2


def test_too_few_similar_sessions_give_no_advice() -> None:
    study = branch_cap_study([_wide("a")], TASK)

    assert study.recommended is None and study.note.startswith("Too little evidence")
    assert [row.policy.branch_cap for row in study.rows] == list(CAPS)


def test_the_study_renders_a_table_and_the_advice() -> None:
    text = branch_cap_study([_wide("a"), _wide("b")], TASK, caps=(2,)).render()

    assert "| branch cap | utility | similar sessions' worth |" in text
    assert "| 2 | 7.50 | 2.0 |" in text and text.endswith("Advice only; the cap is the scheduler's (Q7).")


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    for session_id in ("a", "b"):
        TreeArchive(root).put(_wide(session_id))
    return root


def test_the_command_runs_the_calibration_and_the_branch_cap_study_on_an_archive(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "frontier.jsonl"
    log.write_text("", encoding="utf-8")

    assert main(["frontier", "--archive", str(archive), "--log", f"a={log}"]) == 0
    calibrated = capsys.readouterr().out
    assert main(["--json", "branch-cap", "--archive", str(archive), "--like", "a", "--caps", "2,4"]) == 0
    study = json.loads(capsys.readouterr().out)

    assert calibrated.startswith("Frontier calibration") and "Too little evidence" in calibrated
    assert "| bandit R/V, decay 0.95 |" in calibrated and "| bandit UCB, C=1.4 |" in calibrated
    assert study["recommended"] == 2 and [row["policy"]["branch_cap"] for row in study["rows"]] == [2, 4]
    assert study["rows"][0]["policy"]["stop_pct"] is None


def test_a_study_that_cannot_run_says_why(archive: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["branch-cap", "--archive", str(archive), "--like", "nowhere"]) == 2
    assert "holds no session 'nowhere'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["frontier", "--archive", str(archive), "--log", "no-separator"])
    with pytest.raises(SystemExit):
        main(["branch-cap", "--archive", str(archive), "--like", "a", "--caps", "two"])


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
