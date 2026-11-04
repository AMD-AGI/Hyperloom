# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The submit log after a crash: a torn last line is cut off when the log is read back, so what is written next is kept."""

from __future__ import annotations

from pathlib import Path

from hyperloom_multibranch.toolbox.submission_log import LogEntry, SubmissionLog

_TORN = '{"type": "accepted", "request_id": "'


def _accept(log: SubmissionLog, branch: str) -> LogEntry:
    sequence = log.next_sequence(branch)
    entry = LogEntry(log.request_id(branch, sequence), branch, sequence, {"what": branch}, 2, float(sequence))
    log.accept(entry)
    return entry


def _tear(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_TORN)


def test_what_is_logged_after_a_torn_last_line_reads_back_and_its_number_is_not_handed_out_again(
    tmp_path: Path,
) -> None:
    path = tmp_path / "submissions.jsonl"
    _accept(SubmissionLog(path, "campaign"), "a")
    _tear(path)

    restarted = SubmissionLog(path, "campaign")
    x = _accept(restarted, "x")
    restarted.record_outcome(x.request_id, {"result": "measured"})
    reread = SubmissionLog(path, "campaign")

    found = reread.get(x.request_id)
    assert found is not None and found.outcome == {"result": "measured"}
    assert [entry.branch for entry in reread.entries()] == ["a", "x"]
    assert reread.next_sequence("x") == x.sequence + 1, "x's number is used, so a restart does not hand it out again"
    assert _TORN not in path.read_text(encoding="utf-8"), "the torn line is cut off the file"


def test_a_log_whose_last_line_is_complete_is_left_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "submissions.jsonl"
    _accept(SubmissionLog(path, "campaign"), "a")
    before = path.read_bytes()

    assert [entry.branch for entry in SubmissionLog(path, "campaign").entries()] == ["a"]
    assert path.read_bytes() == before


def test_a_log_that_holds_only_a_torn_line_reads_back_empty_and_takes_new_entries(tmp_path: Path) -> None:
    path = tmp_path / "submissions.jsonl"
    _tear(path)

    restarted = SubmissionLog(path, "campaign")
    entry = _accept(restarted, "a")

    assert SubmissionLog(path, "campaign").get(entry.request_id) is not None
