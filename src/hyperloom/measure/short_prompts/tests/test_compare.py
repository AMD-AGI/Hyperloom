# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Comparing the outputs of several server processes with each other."""

from __future__ import annotations

from hyperloom.measure.short_prompts.compare import BY_TEXT, BY_TOKENS, Output, PromptInfo, compare_outputs

PROMPTS = [PromptInfo("s00", "sequential", 1), PromptInfo("s01", "sequential", 9), PromptInfo("b00", "batched", 40)]


def _outputs(**tokens: list[str] | None) -> dict[str, Output]:
    return {
        pid: Output("".join(toks if toks is not None else ["?"]), None if toks is None else tuple(toks), "length")
        for pid, toks in tokens.items()
    }


GOOD = _outputs(s00=["a", "b", "c"], s01=["d", "e", "f"], b00=["g", "h", "i"])


def _wrong(prompt: str, tokens: list[str]) -> dict[str, Output]:
    return {**GOOD, **_outputs(**{prompt: tokens})}


def test_identical_processes_differ_nowhere():
    result = compare_outputs({0: GOOD, 1: GOOD, 2: GOOD}, PROMPTS)

    assert result.compared_by == BY_TOKENS
    assert result.self_consistency is not None
    assert result.self_consistency.mismatches == (0, 0, 0) and result.self_consistency.first is None


def test_one_process_in_six_that_answers_a_short_prompt_wrongly_is_the_only_one_counted_and_is_located():
    bad = _wrong("s01", ["d", "X", "Y"])
    processes = dict.fromkeys(range(6), GOOD) | {3: bad}

    result = compare_outputs(processes, PROMPTS)

    own = result.self_consistency
    assert own is not None and own.mismatches == (0, 0, 0, 1, 0, 0)
    first = own.first
    assert (first.process, first.prompt_id, first.mode, first.prompt_tokens, first.position) == (
        3,
        "s01",
        "sequential",
        9,
        1,
    )
    assert first.expected == "def" and first.actual == "dXY"


def test_a_single_process_has_no_self_consistency_to_state():
    assert compare_outputs({0: GOOD}, PROMPTS).self_consistency is None


def test_the_consensus_is_the_most_common_output_and_a_tie_goes_to_the_lowest_process():
    bad = _wrong("s00", ["z", "b", "c"])

    assert compare_outputs({0: GOOD, 1: bad}, PROMPTS).self_consistency.mismatches == (0, 1)
    assert compare_outputs({0: bad, 1: GOOD}, PROMPTS).self_consistency.mismatches == (0, 1)
    assert compare_outputs({0: bad, 1: GOOD, 2: GOOD}, PROMPTS).self_consistency.mismatches == (1, 0, 0)


def test_a_late_drift_in_one_process_is_counted_for_that_process_at_its_position():
    drifted = _wrong("b00", ["g", "h", "Z"])

    own = compare_outputs({0: GOOD, 1: GOOD, 2: drifted}, PROMPTS).self_consistency

    assert own is not None and own.mismatches == (0, 0, 1)
    assert (own.first.process, own.first.position) == (2, 2), "a late drift is at its position"


def test_an_output_cut_short_differs_at_the_length_of_the_shorter():
    short = _wrong("s01", ["d"])

    first = compare_outputs({0: GOOD, 1: short, 2: GOOD}, PROMPTS).self_consistency.first

    assert (first.process, first.prompt_id, first.position) == (1, "s01", 1)


def test_outputs_without_tokens_are_compared_by_text_and_the_position_is_a_character():
    plain = {pid: Output(out.text, None) for pid, out in GOOD.items()}
    off = {**plain, "s01": Output("dXf", None)}

    result = compare_outputs({0: plain, 1: off}, PROMPTS)

    assert result.compared_by == BY_TEXT
    assert result.self_consistency.mismatches == (0, 1)
    assert result.self_consistency.first.position == 1


def test_one_output_without_tokens_makes_the_whole_comparison_by_text():
    mixed = {**GOOD, "s00": Output("abc", None)}

    assert compare_outputs({0: GOOD, 1: mixed}, PROMPTS).compared_by == BY_TEXT
