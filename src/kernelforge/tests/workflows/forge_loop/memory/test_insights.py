# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign's insight list: what a record turn is asked for, what is taken from its reply, and what the next session reads."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.memory.insights import (
    INSIGHT_PREFIX,
    MAX_INSIGHT_CHARS,
    MAX_INSIGHTS,
    InsightStore,
    insight_request,
    split_insights,
)
from kernelforge.workflows.forge_loop.memory.lessons import build_summary_prompt


def test_the_record_turn_is_shown_the_list_as_it_stands_and_asked_for_the_list_as_it_should_stand():
    request = insight_request(["a 1 GiB read flush is what makes a quick bench agree with the driver"])
    flowed = " ".join(request.split())

    assert "maintain the campaign's insight list" in flowed
    assert f"  {INSIGHT_PREFIX} a 1 GiB read flush is what makes a quick bench agree with the driver" in request
    assert f"each beginning `{INSIGHT_PREFIX}`" in flowed
    assert f"At most {MAX_INSIGHTS} entries" in flowed
    # An entry is a measured fact, so the record turn's own rules still bind it.
    assert "It is not a direction and not advice" in flowed


def test_an_empty_list_is_shown_as_empty():
    assert "The list as it stands:\n  (empty)" in insight_request([])


def test_the_request_closes_the_record_prompt_only_when_a_list_is_kept():
    request = insight_request([])

    assert build_summary_prompt(iteration=3, end_reason="converged", insight_request=request).endswith(request)
    assert "insight list" not in build_summary_prompt(iteration=3, end_reason="converged")


def test_a_reply_is_split_into_the_record_and_the_entries_it_carried():
    reply = (
        "Tried two tile shapes; both measured slower.\n"
        "NEGATIVES: 128x128 tiles (m4096 +7%)\n"
        "\n"
        f"{INSIGHT_PREFIX} the driver times m1024 15% slower than a warm bench; only the driver's number decides\n"
        f"- **{INSIGHT_PREFIX}** a raw read probe gives 4.1 TB/s, so the weights need at least 56 us\n"
    )

    record, entries = split_insights(reply)

    assert record == "Tried two tile shapes; both measured slower.\nNEGATIVES: 128x128 tiles (m4096 +7%)"
    assert entries == [
        "the driver times m1024 15% slower than a warm bench; only the driver's number decides",
        "a raw read probe gives 4.1 TB/s, so the weights need at least 56 us",
    ]


def test_entries_are_bounded_in_number_and_length_and_never_repeated():
    lines = [f"{INSIGHT_PREFIX} fact {index}" for index in range(MAX_INSIGHTS + 5)]
    lines.insert(1, f"{INSIGHT_PREFIX} fact 0")
    lines.append(f"{INSIGHT_PREFIX} " + "x" * (MAX_INSIGHT_CHARS + 50))

    _record, entries = split_insights("\n".join(lines))

    assert entries == [f"fact {index}" for index in range(MAX_INSIGHTS)]
    _record, long_entry = split_insights(lines[-1])
    assert long_entry == ["x" * MAX_INSIGHT_CHARS]


def test_a_reply_with_no_insight_lines_carries_none_and_keeps_its_record():
    assert split_insights("Nothing measured better.\nNEGATIVES: none") == (
        "Nothing measured better.\nNEGATIVES: none",
        [],
    )


def test_the_store_replaces_the_list_whole_and_reads_it_back(tmp_path):
    store = InsightStore(str(tmp_path))

    assert store.read() == []
    assert store.render_for_prompt() == ""
    assert store.replace(2, ["first fact", "second fact"]) is True
    assert store.replace(5, ["second fact, corrected", "third fact"]) is True

    assert store.read() == ["second fact, corrected", "third fact"]
    assert store.path == tmp_path / "forge_experiments" / "insights.md"


def test_a_reply_without_entries_leaves_the_list_as_it_was(tmp_path):
    store = InsightStore(str(tmp_path))
    store.replace(2, ["first fact"])

    assert store.replace(3, []) is False
    assert store.read() == ["first fact"]


def test_the_next_session_reads_the_whole_list_as_measured_facts(tmp_path):
    store = InsightStore(str(tmp_path))
    store.replace(2, ["first fact", "second fact"])

    block = store.render_for_prompt()

    assert block.startswith("## Campaign insights (cumulative)")
    assert "They describe what was measured, not what to do" in block
    assert block.endswith("- first fact\n- second fact")
