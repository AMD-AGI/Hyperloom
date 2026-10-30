# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notes of a test's use of the cache: problems before facts before tallies, and a hundred files are one line (4.21)."""

from __future__ import annotations

from hyperloom.measure.jit.notes import LISTED, SessionNotes

CHANGED = "changed by the test; the stored bytes stay"


def test_a_tally_is_one_line_with_the_count_and_the_first_names_however_many_files_it_covers() -> None:
    notes = SessionNotes()
    for number in range(72):
        notes.tally("triton", CHANGED, f"hash/kernel_{number}.json")

    assert notes.lines() == (
        f"jit cache: triton: 72 files {CHANGED} (hash/kernel_0.json, hash/kernel_1.json, hash/kernel_2.json, and 69 more)",
    )


def test_a_tally_of_few_files_names_them_all_and_one_file_is_not_plural() -> None:
    notes = SessionNotes()
    notes.tally("triton", CHANGED, "a/x.json")
    notes.tally("aiter_jit", CHANGED, "m1.so")
    notes.tally("aiter_jit", CHANGED, "m2.so")

    assert notes.lines() == (
        f"jit cache: aiter_jit: 2 files {CHANGED} (m1.so, m2.so)",
        f"jit cache: triton: 1 file {CHANGED} (a/x.json)",
    )
    assert LISTED == 3


def test_problems_come_first_then_facts_then_tallies_whatever_order_they_were_noted_in() -> None:
    notes = SessionNotes()
    notes.tally("triton", CHANGED, "a/x.json")
    notes.fact("jit cache: triton is not used: why")
    notes.problem(
        "jit cache: aiter_jit snapshot 0123456789ab: the bytes are not the 3 bytes whose hash is listed; ignored"
    )
    notes.problem("jit cache: aiter_cpp could not be read (OSError); it builds cold")

    lines = notes.lines()

    assert [line.split(":")[1].strip().split(" ")[0] for line in lines] == [
        "aiter_jit",
        "aiter_cpp",
        "triton",
        "triton",
    ]
    assert "ignored" in lines[0] and "builds cold" in lines[1] and "not used" in lines[2] and "1 file" in lines[3]


def test_one_long_note_is_clipped_and_ends_in_dots() -> None:
    notes = SessionNotes()
    notes.problem("jit cache: x " + "y" * 1000)

    (line,) = notes.lines()

    assert len(line) == 300 and line.endswith("...")


def test_notes_beyond_the_budget_are_counted_not_dropped_silently_and_the_first_ones_are_kept() -> None:
    notes = SessionNotes()
    for number in range(30):
        notes.problem(f"jit cache: problem {number:02d} " + "z" * 150)
    notes.tally("triton", CHANGED, "a/x.json")

    lines = notes.lines()

    assert lines[0].startswith("jit cache: problem 00 ") and lines[-1].endswith("more notes left out")
    assert sum(len(line) + 2 for line in lines[:-1]) <= 1200 + 300
    left_out = int(lines[-1].split()[2])
    assert len(lines) - 1 + left_out == 31, "every note is either kept or counted"


def test_no_notes_is_no_lines() -> None:
    assert SessionNotes().lines() == ()
