# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One command says where a session's test time went: the phases of every measured test, by kind of test, from the
session directory's submit log."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import PhaseTiming, TestKind, TestResult, Validity
from hyperloom_multibranch.toolbox.phase_report import SUBMISSIONS, aggregate, main, read_session
from hyperloom_multibranch.toolbox.submission_log import LogEntry, SubmissionLog
from hyperloom_multibranch.toolbox.tests.helpers import req

#: The ``src`` directory of the checkout this file is in. The command under test runs from it, so the result does not
#: depend on which checkout the environment's editable install points at.
_SRC = Path(__file__).resolve().parents[3]


def _result(request_id: str, phases: list[tuple[str, float]], unaccounted: float = 0.0, gpus: int = 2) -> TestResult:
    timed = tuple(PhaseTiming(name, 100.0, seconds, gpu_seconds=seconds * gpus) for name, seconds in phases)
    return (
        TestResult(request_id, Validity.SUCCEEDED, phases=timed, unaccounted_seconds=unaccounted)
        if timed
        else (TestResult(request_id, Validity.SUCCEEDED))
    )


def _log(session: Path, entries: list[tuple[TestKind, dict | None]], torn: bool = False) -> None:
    """The submit log of a session: each entry a request of a kind and the outcome the toolbox recorded for it."""
    log = SubmissionLog(session / SUBMISSIONS, "campaign")
    for number, (kind, outcome) in enumerate(entries, start=1):
        request_id = f"req-{number}"
        document = req(kind=kind).accepted(request_id).to_document()
        log.accept(LogEntry(request_id, "b1", number, document, 2, float(number)))
        if outcome is not None:
            log.record_outcome(request_id, outcome)
    if torn:
        with (session / SUBMISSIONS).open("a", encoding="utf-8") as handle:
            handle.write('{"type": "outcome", "request_id": "req-')


def _outcome(*args, **kwargs) -> dict:
    return {"result": _result(*args, **kwargs).to_document()}


BOOT = [("sandbox_start", 10.0), ("server_boot", 80.0), ("measured", 30.0)]
SPLIT_BOOT = [
    ("sandbox_start", 12.0),
    ("server_boot.other", 20.0),
    ("server_boot.weights_load", 50.0),
    ("measured", 40.0),
]


@pytest.fixture
def session(tmp_path: Path) -> Path:
    _log(
        tmp_path,
        [
            (TestKind.TEST, _outcome("req-1", BOOT, 2.0)),
            (TestKind.TEST, _outcome("req-2", SPLIT_BOOT, 4.0)),
            (TestKind.TEST, _outcome("req-3", [("sandbox_start", 8.0), ("server_boot", 60.0)], 0.0)),
            (TestKind.SWEEP, _outcome("req-4", [("sandbox_start", 9.0), ("measured", 100.0), ("measured", 120.0)])),
            (TestKind.PROFILE, _outcome("req-5", [])),  # a result that has no phases
            (TestKind.TEST, {"result": {"request_id": "req-6", "validity": "bogus"}}),  # one that cannot be read
            (TestKind.TEST, {"withdrawal": "nobody wants it"}),
            (TestKind.TEST, None),  # not measured yet
        ],
        torn=True,
    )
    return tmp_path


def _row(rows, kind: str, phase: str):
    (found,) = [r for r in rows if (r.kind, r.phase) == (kind, phase)]
    return found


def test_the_phases_are_aggregated_by_kind_with_their_total_median_and_share_of_the_wall_time(session: Path) -> None:
    rows = aggregate(read_session(session).results)

    boot = _row(rows, "test", "server_boot")
    # 80 and 60 as one phase, and the 20 + 50 of the one whose engine log split it: three tests ran it
    assert (boot.tests, boot.total_seconds, boot.median_seconds) == (3, 210.0, 70.0)
    assert boot.gpu_seconds == 2 * 210.0
    measured = _row(rows, "test", "measured")
    assert (measured.tests, measured.total_seconds, measured.median_seconds) == (2, 70.0, 35.0)
    sweep = _row(rows, "sweep", "measured")
    assert (sweep.tests, sweep.total_seconds, sweep.median_seconds) == (1, 220.0, 220.0), (
        "a test's repeats are one entry"
    )


def test_the_rows_of_a_kind_add_up_to_its_wall_time_with_the_unaccounted_seconds_a_row_of_their_own(
    session: Path,
) -> None:
    rows = aggregate(read_session(session).results)

    unaccounted = _row(rows, "test", "unaccounted")
    assert unaccounted.total_seconds == 6.0 and unaccounted.tests == 3
    wall = (10 + 80 + 30 + 2) + (12 + 20 + 50 + 40 + 4) + (8 + 60)
    whole = [r for r in rows if r.kind == "test" and "." not in r.phase]
    assert sum(r.total_seconds for r in whole) == wall
    assert sum(r.share for r in whole) == pytest.approx(1.0)
    assert _row(rows, "test", "server_boot.weights_load").share == pytest.approx(50.0 / wall)


def test_a_split_phase_is_listed_after_the_phase_it_splits_and_phases_come_in_the_order_a_test_goes_through_them(
    session: Path,
) -> None:
    rows = aggregate(read_session(session).results)

    test_phases = [r.phase for r in rows if r.kind == "test"]
    assert test_phases == [
        "sandbox_start",
        "server_boot",
        "server_boot.other",
        "server_boot.weights_load",
        "measured",
        "unaccounted",
    ]
    assert [r.kind for r in rows] == sorted(r.kind for r in rows), "kinds in name order"


def test_what_the_log_holds_no_usable_phases_of_is_counted_and_not_dropped_in_silence(session: Path) -> None:
    read = read_session(session)

    assert (len(read.results), read.without_phases, read.unreadable) == (4, 1, 1)


def test_the_command_prints_one_table_for_each_kind_of_test(session: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(session)]) == 0

    text = capsys.readouterr().out
    assert text.splitlines()[0] == "4 measured tests with phases"
    assert "(1 without phases, 1 unreadable: not counted)" in text
    assert "test (3 tests)" in text and "sweep (1 tests)" in text
    boot_line = next(line for line in text.splitlines() if line.strip().startswith("server_boot "))
    assert boot_line.split() == ["server_boot", "3", "210.0", "70.0", f"{210.0 / 316.0:.1%}"]


def test_the_command_can_print_the_rows_as_a_document(session: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(session), "--json"]) == 0

    document = json.loads(capsys.readouterr().out)
    assert (document["tests"], document["without_phases"], document["unreadable"]) == (4, 1, 1)
    rows = {(r["kind"], r["phase"]): r for r in document["rows"]}
    assert rows[("test", "server_boot")]["total_seconds"] == 210.0


def test_the_command_runs_from_the_shell_on_a_session_directory(session: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "hyperloom_multibranch.toolbox.phase_report", str(session)],
        capture_output=True,
        text=True,
        check=False,
        cwd=session,
        env={**os.environ, "PYTHONPATH": str(_SRC)},
    )

    assert done.returncode == 0, done.stderr
    assert "server_boot" in done.stdout


def test_a_directory_that_is_not_a_session_is_refused_in_a_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path)]) == 2

    assert "cannot read" in capsys.readouterr().err
