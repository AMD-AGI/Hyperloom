# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command that runs every study on what past sessions left."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom.common.llm_config import AnthropicCompletionRequest, AnthropicMessageResult
from hyperloom_multibranch.blackboard.rsi import studies
from hyperloom_multibranch.blackboard.rsi.archive import SearchTree, TreeArchive
from hyperloom_multibranch.blackboard.rsi.decisions import archive_decisions
from hyperloom_multibranch.blackboard.rsi.ladder import JUDGE_SYSTEM_PROMPT
from hyperloom_multibranch.blackboard.rsi.replay import ARCHIVE, OLD_RULE
from hyperloom_multibranch.blackboard.rsi.studies import main
from hyperloom_multibranch.blackboard.rsi.tests.trees import branch, tree
from hyperloom_multibranch.blackboard.tests.factories import FakeClock
from hyperloom_multibranch.blackboard.tests.sessions import scripted_session


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


def test_the_command_runs_the_scheduling_replay_on_an_archive(
    archive: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "scheduling", "--archive", str(archive), "--runs", "2", "--beyond", "pessimistic"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert main(["scheduling", "--archive", str(archive), "--runs", "1"]) == 0
    table = capsys.readouterr().out

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
    assert main(["families", "--archive", str(archive)]) == 0
    assert capsys.readouterr().out.startswith("Over 0 archived tests, 0 of them committing")


def test_the_command_evolves_the_playbook_asking_the_judge_at_the_decisions_the_archives_kept(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    more = tmp_path / "more"
    for session_id in ("c", "d"):
        TreeArchive(more).put(_wide(session_id))
    transcript = tmp_path / "MB-judge.jsonl"
    stop = json.dumps({"stop": [{"branch": "s1", "reason": "lowest"}]})
    transcript.write_text(json.dumps({"at": 2700.0, "prompt": "view", "reply": stop}) + "\n", encoding="utf-8")
    for session_id in ("a", "b", "c", "d"):
        archive_decisions(transcript, TreeArchive(more), session_id)
    rule = {"add_rule": {"governs": "stop", "text": "Stop the later of two alike branches."}}

    async def complete(system: str, prompt: str) -> str:
        if system == JUDGE_SYSTEM_PROMPT:
            return '{"stop": []}'
        return json.dumps({"label": "alike", "change": rule, "hypothesis": "h", "measure": "early_stops"})

    monkeypatch.setattr(studies, "_completion", lambda model: complete)
    argv = ["--archive", str(archive), "--archive", str(more), "--model", "m", "--count", "1", "--runs", "2"]

    assert main(["playbook-generations", *argv, "--pool", str(tmp_path / "pool")]) == 0

    # Each evolution session's stop of s1, one test past its commit, should go; a judge that stops nothing gets it
    # right under either playbook.
    assert "R2 right on 2 of 2 points, h0 on 2" in capsys.readouterr().out


@pytest.mark.parametrize("refusals", [1, studies.REFUSAL_RETRIES + 1])
async def test_a_refused_reply_is_asked_again_a_few_times(monkeypatch: pytest.MonkeyPatch, refusals: int) -> None:
    replies = [AnthropicMessageResult("", studies.REFUSAL, None)] * refusals
    replies.append(AnthropicMessageResult('{"none": "nothing"}', "end_turn", None))
    asked: list[str] = []

    async def answer(*, request: AnthropicCompletionRequest, component: str, operation: str) -> AnthropicMessageResult:
        asked.append(request.system or "")
        return replies.pop(0)

    monkeypatch.setattr(studies, "aanthropic_completion", answer)
    text = await studies._completion("m")("system", "prompt")

    assert len(asked) == min(refusals + 1, studies.REFUSAL_RETRIES + 1)
    assert text == ('{"none": "nothing"}' if refusals <= studies.REFUSAL_RETRIES else "")


def test_a_study_that_cannot_run_says_why(archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scheduling", "--archive", str(tmp_path / "empty")]) == 2
    assert "the archives hold no session to replay" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["scheduling", "--archive", str(archive), "--beyond", "anywhere"])


def test_the_command_evaluates_a_sessions_learnings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    asyncio.run(scripted_session(tmp_path / "blackboard", FakeClock()))
    log = tmp_path / "learnings.jsonl"
    log.write_text(json.dumps({"at": 0.0, "branch": "s1", "source": "tool", "here": [], "past": []}) + "\n", "utf-8")

    assert main(["learnings", "--blackboard", str(tmp_path / "blackboard"), "--log", str(log)]) == 0

    assert capsys.readouterr().out.startswith("Learnings retrieval: 1 retrievals (tool 1)")
