# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted journal: what survives a crash, and what is refused instead of guessed."""

from __future__ import annotations

import pytest

from hyperloom.measure.hosted import STOPPED_RETURNCODE
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SUBMITTED, JournalCorrupt, MeasurementJournal, Record


def _record(request_id: str = "r1", attempt: int = 1, state: str = STATE_SUBMITTED, **extra) -> Record:
    return Record(request_id, attempt, state, "sbx-1", "/out", "task", extra)


def test_the_newest_line_of_a_request_wins_and_attempts_count_up(tmp_path) -> None:
    journal = MeasurementJournal(tmp_path)
    journal.append(_record("a", 1))
    journal.append(_record("a", 1, STATE_FINISHED, verdict="x"))
    journal.append(_record("b", 1))
    assert journal.latest("a").state == STATE_FINISHED and journal.latest("a").extra == {"verdict": "x"}
    assert journal.next_attempt("a") == 2 and journal.next_attempt("b") == 2 and journal.next_attempt("new") == 1
    assert journal.latest("none") is None


def test_a_crash_that_cut_the_last_line_loses_only_that_line(tmp_path) -> None:
    journal = MeasurementJournal(tmp_path)
    journal.append(_record("a", 1))
    with open(journal.path, "ab") as handle:
        handle.write(b'{"request_id":"a","attem')
    assert [r.request_id for r in journal.records()] == ["a"]
    assert journal.latest("a").state == STATE_SUBMITTED


def test_an_unreadable_line_in_the_middle_is_refused_not_skipped(tmp_path) -> None:
    journal = MeasurementJournal(tmp_path)
    journal.append(_record("a", 1))
    with open(journal.path, "ab") as handle:
        handle.write(b"not json\n")
    journal.append(_record("b", 1))
    with pytest.raises(JournalCorrupt, match="line 2"):
        journal.records()


def test_a_complete_but_invalid_last_line_is_refused_too(tmp_path) -> None:
    journal = MeasurementJournal(tmp_path)
    journal.append(_record("a", 1))
    with open(journal.path, "ab") as handle:
        handle.write(b'{"state":"weird"}\n')
    with pytest.raises(JournalCorrupt):
        journal.records()


def test_sheets_are_kept_per_attempt_and_the_request_id_cannot_climb_out(tmp_path) -> None:
    journal = MeasurementJournal(tmp_path)
    journal.save_spec("req-1", 2, b"sheet")
    assert journal.load_spec("req-1", 2) == b"sheet"
    with pytest.raises(ValueError, match="not usable as a file name"):
        journal.save_spec("../escape", 1, b"x")


def test_the_stop_code_is_the_one_the_local_flow_uses() -> None:
    from hyperloom.orchestrator.actions.executors._subprocess_kill import ORCHESTRATOR_CANCELLED_RETURNCODE

    assert STOPPED_RETURNCODE == ORCHESTRATOR_CANCELLED_RETURNCODE
