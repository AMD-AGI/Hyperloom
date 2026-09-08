# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a multi-rank driver is told, for the parts nothing measures.

The deterministic check watches the run and can settle rank count, device
ownership, the reduction op and process-group teardown. The rest of the
measurement contract is not observable from a call-count probe, so this text is
the only thing standing behind it -- and some of what it carries is the
difference between a wasted validation and a wrong answer that passes.
"""

from __future__ import annotations

import pytest

from kernelforge.loop.task_preparer import _distributed_contract_note


@pytest.fixture(scope="module")
def note() -> str:
    return _distributed_contract_note(8)


@pytest.mark.parametrize("ranks", [0, 1])
def test_a_single_rank_task_is_told_nothing_about_ranks(ranks: int) -> None:
    """The note is the multi-rank contract; an ordinary task has none."""
    assert _distributed_contract_note(ranks) == ""


def test_parity_is_validated_over_two_calls_not_one(note: str) -> None:
    """A collective that reuses scratch races only against its own next call.

    The output usually lives in a registered buffer, so the second call can
    overwrite the first result before anything reads it. Validating a call the
    moment it is issued never sees that, and dropping a synchronisation -- the
    change this lane's optimizer reaches for first -- is what opens the race.
    """
    assert "two calls back to back, never one at a time" in note
    assert "compare both results only afterwards" in note
    assert "registered scratch buffer" in note


def test_the_correctness_reference_is_itself_distributed(note: str) -> None:
    assert "matching `torch.distributed` collective" in note


def test_inputs_are_seeded_per_rank(note: str) -> None:
    """Identical inputs let a collective that drops a rank still pass parity."""
    assert "Seed inputs per rank" in note
    assert "silently drops a rank" in note


def test_the_timed_region_is_kept_free_of_synchronisation(note: str) -> None:
    assert "Keep the timed region free of synchronization" in note
    assert "BEFORE `start.record()`" in note


def test_the_note_says_which_requirements_are_actually_enforced(note: str) -> None:
    """An author who cannot tell the two apart treats neither as load-bearing."""
    enforced, _, unenforced = note.partition("Every other requirement above is yours to hold")

    assert "The deterministic check observes the run itself" in enforced
    assert unenforced, "the note must say what nothing measures"


def test_a_correctness_slip_is_not_described_as_a_wasted_validation(note: str) -> None:
    """The end-to-end gate can adopt on throughput without scoring accuracy.

    Telling the author that the worst case is a wasted validation would be
    wrong for exactly the requirements that matter most.
    """
    assert "without ever scoring accuracy" in note
    assert "Nothing after you is guaranteed" in note
