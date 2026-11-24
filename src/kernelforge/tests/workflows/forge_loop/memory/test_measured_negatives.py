# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The list of measured negatives a session is shown: one line per past iteration, with whether each still stands."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.memory.lessons import (
    LessonScope,
    LessonStore,
    format_scope_line,
    listed_negatives,
)
from kernelforge.workflows.forge_loop.memory.measured_negatives import (
    IN_SCOPE,
    MAX_LIST_CHARS,
    MIN_ENTRY_CHARS,
    RE_OPENABLE,
    UNSCOPED,
    render_measured_negatives,
)

_KERNEL = "BLOCK_N = 16\nnum_warps = 8\n"


def _document(negatives: str, scope: LessonScope | None) -> str:
    lines = ["Tried a few things.", f"NEGATIVES: {negatives}"]
    if scope is not None:
        lines += ["", format_scope_line(scope)]
    return "\n".join(lines)


def _scope(**overrides) -> LessonScope:
    fields = dict(cases=("m64", "m4096"), held_fixed=(("BLOCK_N", "16"),), carries_negative=True)
    fields.update(overrides)
    return LessonScope(**fields)


def test_a_document_lists_what_measured_worse_across_its_marker_lines():
    document = "NEGATIVES: split-K=4 (m64 0.105 vs 0.091 ms)\n- **NEGATIVES:** BLOCK_N=128 (m4096 +7%)."

    assert listed_negatives(document) == "split-K=4 (m64 0.105 vs 0.091 ms); BLOCK_N=128 (m4096 +7%)"
    assert listed_negatives("NEGATIVES: none") == ""
    assert listed_negatives("nothing recorded") == ""


def test_every_iteration_that_recorded_a_negative_gets_one_line_however_old(tmp_path):
    store = LessonStore(str(tmp_path))
    store.write(1, _document("split-K=4 (m64 0.105 vs 0.091 ms)", _scope()))
    store.write(2, _document("none", _scope(carries_negative=False)))
    store.write(9, _document("fused reduce in the attention launch (b1 +12%)", _scope()))

    rendered = render_measured_negatives(store, current_cases=("m64", "m4096"), kernel_source=_KERNEL)

    assert rendered.startswith("## Measured negatives from every past iteration")
    assert "Check a direction against this list before you choose it" in rendered
    lines = [line for line in rendered.splitlines() if line.startswith("- iter ")]
    assert lines == [
        f"- iter 1 [{IN_SCOPE}] split-K=4 (m64 0.105 vs 0.091 ms) | held fixed BLOCK_N=16 | measured on m64, m4096",
        f"- iter 9 [{IN_SCOPE}] fused reduce in the attention launch (b1 +12%) | held fixed BLOCK_N=16 "
        "| measured on m64, m4096",
    ]


def test_a_negative_measured_under_a_constant_that_has_since_moved_is_marked_re_openable(tmp_path):
    store = LessonStore(str(tmp_path))
    store.write(1, _document("split-K=4 (m64 +15%)", _scope(held_fixed=(("BLOCK_N", "32"),))))

    rendered = render_measured_negatives(store, current_cases=("m64", "m4096"), kernel_source=_KERNEL)

    assert f"- iter 1 [{RE_OPENABLE}: BLOCK_N is now 16 (pinned at 32 when this was measured)]" in rendered


def test_a_negative_not_measured_on_every_scored_case_is_marked_re_openable(tmp_path):
    store = LessonStore(str(tmp_path))
    store.write(1, _document("split-K=4 (m64 +15%)", _scope(cases=("m64",))))

    rendered = render_measured_negatives(store, current_cases=("m64", "m4096"), kernel_source=_KERNEL)

    assert f"- iter 1 [{RE_OPENABLE}: not measured on m4096]" in rendered


def test_a_negative_with_no_recorded_conditions_is_marked_unscoped(tmp_path):
    store = LessonStore(str(tmp_path))
    store.write(1, _document("split-K=4 (m64 +15%)", None))

    rendered = render_measured_negatives(store, current_cases=("m64",), kernel_source=_KERNEL)

    assert f"- iter 1 [{UNSCOPED}: the conditions it was measured under were not recorded] split-K=4" in rendered


def test_no_recorded_negative_means_no_list(tmp_path):
    store = LessonStore(str(tmp_path))
    store.write(1, _document("none", _scope(carries_negative=False)))

    assert render_measured_negatives(store, current_cases=("m64",), kernel_source=_KERNEL) == ""


def test_many_iterations_share_the_list_budget_without_any_being_dropped(tmp_path):
    store = LessonStore(str(tmp_path))
    count = 24
    for iteration in range(1, count + 1):
        store.write(iteration, _document("a long sweep " + "x" * 1000, _scope()))

    rendered = render_measured_negatives(store, current_cases=("m64", "m4096"), kernel_source=_KERNEL)

    lines = [line for line in rendered.splitlines() if line.startswith("- iter ")]
    room = max(MIN_ENTRY_CHARS, MAX_LIST_CHARS // count)
    assert len(lines) == count
    assert all(line.endswith(" ...") for line in lines)
    assert max(len(line) for line in lines) <= room + len(f"- iter {count} [{IN_SCOPE}] ")
