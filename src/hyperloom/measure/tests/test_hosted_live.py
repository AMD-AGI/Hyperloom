# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the hosted backend keeps of the live output its runners report, and the log reading both sides share."""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import pytest

from hyperloom.measure import live_log
from hyperloom.measure.hosted_live import LiveOutputs, finite_number
from hyperloom.measure.live_log import LINE_CHARS, _tail, clean_line, last_output, last_output_below


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _document(line: str = "decode batch 8", age: float = 3.0, updated: float = 500.0) -> dict[str, object]:
    return {"phase": "running", "updated_unix": updated, "last_log_age_sec": age, "last_line": line}


def test_a_report_is_the_runners_age_plus_the_time_since_it_was_read() -> None:
    clock = _Clock()
    live = LiveOutputs(clock)
    assert live.get("r1") is None, "nothing was reported yet"
    live.note("r1", _document(age=3.0))
    assert live.get("r1") == (3.0, "decode batch 8")
    clock.now += 7.5
    live.note("r1", _document(age=3.0))  # the same file read again: the runner has not refreshed it
    assert live.get("r1") == (10.5, "decode batch 8"), "the time that passed since the first read still counts"
    live.note("r1", _document(line="decode batch 9", age=1.0, updated=520.0))
    assert live.get("r1") == (1.0, "decode batch 9"), "a refreshed file starts again from its own age"
    clock.now += 2.0
    assert live.get("r1") == (3.0, "decode batch 9")
    assert live.get("r2") is None, "each test has its own"


def test_the_line_is_cleaned_again_on_this_side_because_it_is_whatever_the_test_printed() -> None:
    live = LiveOutputs(_Clock())
    live.note("r1", _document(line="\x1b[31mERROR\x1b[0m\tbad\r\x07\n\nnext line\x00\n" + "y" * 1000 + "END"))
    seconds, line = live.get("r1") or (None, "")
    assert seconds == 3.0 and len(line) == LINE_CHARS and line.endswith("yyyEND")
    assert "\x1b" not in line and "\x00" not in line


@pytest.mark.parametrize(
    "document",
    [
        None,
        {},
        {"phase": "running"},
        {"last_line": "a line, but no age"},
        {"last_log_age_sec": 3.0},
        {"last_log_age_sec": "3", "last_line": "an age that is a string"},
        {"last_log_age_sec": True, "last_line": "an age that is a bool"},
        {"last_log_age_sec": -1.0, "last_line": "an age before the write"},
        {"last_log_age_sec": math.nan, "last_line": "an age that is not a number"},
        {"last_log_age_sec": math.inf, "last_line": "an age that never ends"},
        {"last_log_age_sec": 10**400, "last_line": "an age no float can hold"},
        {"last_log_age_sec": 3.0, "last_line": ["not", "text"]},
    ],
)
def test_a_document_without_a_usable_live_line_changes_nothing(document: dict[str, object] | None) -> None:
    live = LiveOutputs(_Clock())
    live.note("r1", document)
    assert live.get("r1") is None
    live.note("r1", _document())
    live.note("r1", document)
    assert live.get("r1") == (3.0, "decode batch 8"), "and it does not take back what was reported"


def test_a_test_that_is_over_reports_nothing_more() -> None:
    live = LiveOutputs(_Clock())
    live.note("r1", _document())
    live.forget("r1")
    live.forget("r1")
    assert live.get("r1") is None


@pytest.mark.parametrize(("value", "expected"), [(3, 3.0), (2.5, 2.5), (0, 0.0)])
def test_finite_numbers_are_numbers(value: object, expected: float) -> None:
    assert finite_number(value) == expected


@pytest.mark.parametrize("value", [None, True, False, "1.5", [1], math.nan, math.inf, -math.inf, 10**400, -(10**400)])
def test_nothing_else_is_a_number(value: object) -> None:
    assert finite_number(value) is None


def test_the_last_line_of_a_text_is_its_last_non_empty_line_cleaned_and_shortened() -> None:
    assert clean_line("") == clean_line("\n \r\n\t") == ""
    assert clean_line("first\n  second \x1b[1mbold\x1b[0m  \n\n") == "second bold"
    assert clean_line("a\tb\x07c") == "a b c"
    assert clean_line("x" * 1000 + "END") == "x" * (LINE_CHARS - 3) + "END"


def test_only_the_end_of_a_large_log_is_read(tmp_path: Path) -> None:
    log = tmp_path / "round" / "server.log"
    log.parent.mkdir()
    log.write_bytes(b"old line\n" * 100_000 + b"the newest line\n")
    os.utime(log, (1000.0, 1000.0))
    assert last_output(tmp_path, 1010.0) == (10.0, "the newest line")
    log.write_bytes(b"far back\n" + b"\n" * 200_000)
    os.utime(log, (1000.0, 1000.0))
    assert last_output(tmp_path, 1010.0) == (10.0, ""), "a line further from the end than the tail is not looked for"


def test_a_log_that_is_a_link_or_not_a_file_is_never_read(tmp_path: Path) -> None:
    secret = tmp_path / "elsewhere" / "token"
    secret.parent.mkdir()
    secret.write_text("a line the measured program must not be able to show\n")
    files = tmp_path / "files"
    (files / "linked").mkdir(parents=True)
    (files / "linked" / "server.log").symlink_to(secret)
    (files / "pipe").mkdir()
    os.mkfifo(files / "pipe" / "benchmark_stdout.log", 0o600)
    (files / "dir" / "server.log").mkdir(parents=True)
    assert last_output(files, os.path.getmtime(secret) + 5) is None
    real = files / "real" / "benchmark_stderr.log"
    real.parent.mkdir()
    real.write_text("the real line\n")
    os.utime(real, (500.0, 500.0))  # older than every link, so only skipping them can pick it
    assert last_output(files, 510.0) == (10.0, "the real line")


def test_a_log_swapped_for_a_link_after_it_was_listed_is_not_opened(tmp_path: Path) -> None:
    secret = tmp_path / "token"
    secret.write_text("a line the measured program must not be able to show\n")
    link = tmp_path / "server.log"
    link.symlink_to(secret)
    assert _tail(link) is None and _tail(tmp_path) is None and _tail(tmp_path / "missing.log") is None
    assert _tail(secret) == "a line the measured program must not be able to show\n"


def _log(path: Path, text: str, written: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (written, written))


@pytest.mark.parametrize("swapped_for", ["pipe", "link", "file"])
def test_a_directory_swapped_after_it_was_listed_is_skipped_never_waited_on_nor_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swapped_for: str
) -> None:
    """The measured program may put a pipe, a link or a file where the walk listed a directory, before it opens it:
    the walk skips it (a pipe opened to read waits for a writer; a link leads to a log anywhere) and still reads the
    log beside it."""
    files, elsewhere = tmp_path / "files", tmp_path / "elsewhere"
    (files / "sub").mkdir(parents=True)
    _log(files / "server.log", "the line\n", 500.0)
    _log(elsewhere / "server.log", "a line the measured program must not be able to show\n", 505.0)
    listed = live_log._listed
    swapped: list[str] = []

    def swapping(directory: int) -> tuple[list[str], list[str]]:
        found = listed(directory)
        if "sub" in found[1] and not swapped:
            swapped.append(swapped_for)
            (files / "sub").rmdir()
            if swapped_for == "pipe":
                os.mkfifo(files / "sub")
            elif swapped_for == "link":
                (files / "sub").symlink_to(elsewhere)
            else:
                (files / "sub").write_text("a file\n")
        return found

    monkeypatch.setattr(live_log, "_listed", swapping)
    read: dict[str, object] = {}
    reader = threading.Thread(target=lambda: read.update(output=last_output(files, 510.0)), daemon=True)
    reader.start()
    reader.join(10)
    if reader.is_alive():
        os.close(os.open(files / "sub", os.O_WRONLY | os.O_NONBLOCK))  # lets the reader go
        reader.join(10)
        pytest.fail("the walk waited on a pipe put where a directory was")
    assert swapped and read == {"output": (10.0, "the line")}


def test_the_walk_reads_no_deeper_than_its_limit(tmp_path: Path) -> None:
    """A tree the measured program made deep costs the reader no descriptor past the limit: a log there is not read."""
    files = tmp_path / "files"
    deepest = files.joinpath(*["d"] * live_log._MAX_DEPTH)
    _log(deepest / "server.log", "at the limit\n", 500.0)
    _log(deepest / "d" / "server.log", "below the limit\n", 505.0)
    assert last_output(files, 510.0) == (10.0, "at the limit")


def test_a_directory_that_cannot_be_listed_is_no_live_line_where_its_path_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """:func:`last_output` (the runner's, which reports its progress from it) says nothing rather than raise; the walk
    by descriptor raises, for its caller to say so."""
    _log(tmp_path / "sub" / "server.log", "the line\n", 500.0)
    listed = live_log._listed

    def failing_below_the_top(directory: int) -> tuple[list[str], list[str]]:
        if os.path.samestat(os.fstat(directory), os.stat(tmp_path / "sub")):
            raise PermissionError(13, "Permission denied")
        return listed(directory)

    monkeypatch.setattr(live_log, "_listed", failing_below_the_top)
    assert last_output(tmp_path, 510.0) is None
    top = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(PermissionError):
            last_output_below(top, 510.0)
    finally:
        os.close(top)
