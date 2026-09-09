# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a multi-rank driver is told, now that the harness holds most of it.

Almost every requirement this note used to carry is owned by ``dist_harness``
and cannot be written wrong. What is left is the handover -- which functions the
driver supplies and where they are handed over -- and the one requirement the
harness genuinely cannot hold: that the correctness reference is itself
distributed. That one is the difference between a wasted validation and a wrong
answer that passes, so the note has to say so rather than list it among the
rest.
"""

from __future__ import annotations

import pytest

from kernelforge.loop.task_preparer import _distributed_contract_note


@pytest.fixture(scope="module")
def note() -> str:
    """The note as one line, because it is wrapped and the claims are not.

    Asserting against the wrapped text pins where a sentence happens to break,
    which is not what any of these tests is about.
    """
    return " ".join(_distributed_contract_note(8).split())


@pytest.mark.parametrize("ranks", [0, 1])
def test_a_single_rank_task_is_told_nothing_about_ranks(ranks: int) -> None:
    """The note is the multi-rank contract; an ordinary task has none."""
    assert _distributed_contract_note(ranks) == ""


def test_the_driver_is_told_not_to_own_the_launch(note: str) -> None:
    """A driver that launches its own ranks is refused, so saying so is not style."""
    assert "do NOT launch ranks yourself" in note
    assert "Do NOT rewrite it" in note


def test_the_handover_is_shown_rather_than_described(note: str) -> None:
    """The agent writes three functions; an approximate import is a failed attempt."""
    assert "from dist_harness import Case, run" in note
    for name in ("build_inputs", "call_candidate", "reference"):
        assert name in note
    assert "world_size=8" in note


def test_the_rank_count_reaches_the_note(note: str) -> None:
    assert "world_size=2" in " ".join(_distributed_contract_note(2).split())


def test_the_note_says_which_guarantees_the_author_no_longer_owns(note: str) -> None:
    """An author told to hold a property the harness already holds writes it twice.

    Worse, an author who cannot tell which properties are held stops treating
    any of them as load-bearing.
    """
    assert "cannot be identical across ranks or across the two calls" in note
    assert "issues both candidate calls before comparing either" in note
    assert "SLOWEST rank's time" in note
    assert "None of that is yours to remember" in note


def test_the_distributed_reference_is_named_as_the_one_thing_left(note: str) -> None:
    """The harness cannot tell a single-GPU reference from a distributed one.

    Both are functions returning a tensor, and a candidate that quietly drops a
    rank's contribution matches the wrong one.
    """
    _, _, remaining = note.partition("What is still yours:")

    assert remaining, "the note must say what the harness does not hold"
    assert "the one requirement the harness cannot hold for you" in remaining
    assert "`torch.distributed` collective" in remaining


def test_a_correctness_slip_is_not_described_as_a_wasted_validation(note: str) -> None:
    """The end-to-end gate can adopt on throughput without scoring accuracy.

    Telling the author that the worst case is a wasted validation would be
    wrong for exactly the requirement that is still theirs.
    """
    assert "without ever scoring accuracy" in note
    assert "Nothing after you is guaranteed" in note


def test_the_note_states_what_is_actually_checked(note: str) -> None:
    """One fact, checkable, rather than a list the author has to trust."""
    assert "measured inside the harness" in note
    assert "rejected for that" in note
