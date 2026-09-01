"""Unit tests for the per-iteration lesson documents.

GPU-free: the summarizer session is replaced by a plain async callable, so the
store, the prompt rendering, the character budget, and the two-author write
path are all exercised without an agent backend.
"""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.resources import resource_path
from kernelforge.loop.lessons import (
    CLAIM_DISPROVED,
    DEFAULT_RECENT_LESSONS,
    NO_FEASIBILITY_CLAIM,
    UNDISPROVEN_CLAIM,
    LessonScope,
    LessonStore,
    build_fallback_document,
    build_summary_prompt,
    cases_named_in,
    format_outcome_line,
    format_scope_line,
    is_claim_disproved,
    is_cutoff,
    parse_disproof_marker,
    parse_held_fixed,
    parse_negatives_marker,
    parse_scope_line,
    scan_constant_values,
    scan_sources_for_constants,
    scan_sources_with_coverage,
    scope_conflicts,
    summarize_iteration,
)


def _store(tmp_path, **kwargs) -> LessonStore:
    return LessonStore(str(tmp_path), **kwargs)


# ── store round-trip ──────────────────────────────────────────────────────────


def test_write_and_read_round_trip(tmp_path):
    store = _store(tmp_path)
    store.write(7, "swizzled the LDS tile\n\n- [better] xor swizzle | 1.11x")

    assert store.path(7).name == "iter_007.md"
    assert "xor swizzle" in store.read(7)
    assert store.existing_iterations() == [7]


def test_write_ignores_empty_text(tmp_path):
    store = _store(tmp_path)
    written = store.write(1, "   \n  ")
    assert written is None
    assert store.existing_iterations() == []


def test_read_missing_iteration_is_empty(tmp_path):
    assert _store(tmp_path).read(42) == ""


# ── the loop's machine-written half ───────────────────────────────────────────


def test_append_outcome_preserves_the_narrative(tmp_path):
    store = _store(tmp_path)
    store.write(4, "tried three things")
    appended = store.append_outcome(4, "OUTCOME: KEEP | wall 1.0000 ms")
    assert appended

    text = store.read(4)
    assert "tried three things" in text
    assert text.strip().endswith("OUTCOME: KEEP | wall 1.0000 ms")


def test_append_outcome_without_a_narrative_still_records(tmp_path):
    """A failed summarizer must not lose the objective verdict."""
    store = _store(tmp_path)
    appended = store.append_outcome(9, "OUTCOME: CRASH | session ended: turn_cap")
    assert appended
    assert store.read(9).strip() == "OUTCOME: CRASH | session ended: turn_cap"


def test_append_outcome_ignores_empty_line(tmp_path):
    store = _store(tmp_path)
    appended = store.append_outcome(1, "  ")
    assert appended is False


def test_format_outcome_line_includes_available_measurements():
    line = format_outcome_line(
        decision="REVERT_PERF",
        wall_ms=1.2340,
        best_wall_ms=1.1980,
        snr_db=42.25,
        end_reason="turn_cap",
    )
    assert line == ("OUTCOME: REVERT_PERF | wall 1.2340 ms vs best 1.1980 ms | snr 42.2 dB | session ended: turn_cap")


def test_format_outcome_line_tolerates_missing_measurements():
    line = format_outcome_line(
        decision="BUILD_FAILED",
        wall_ms=None,
        best_wall_ms=None,
        snr_db=None,
        end_reason="",
    )
    assert line == "OUTCOME: BUILD_FAILED"


# ── prompt rendering ──────────────────────────────────────────────────────────


def test_render_is_empty_before_any_lesson(tmp_path):
    assert _store(tmp_path).render_for_prompt() == ""


def test_render_inlines_only_the_recent_window(tmp_path):
    store = _store(tmp_path, recent=2)
    for iteration in range(1, 6):
        store.write(iteration, f"headline {iteration}\nbody {iteration}")

    rendered = store.render_for_prompt()

    assert "body 4" in rendered and "body 5" in rendered
    assert "body 1" not in rendered
    assert "2 of 5 shown" in rendered


def test_render_always_points_at_the_absolute_directory(tmp_path):
    """The implementer's cwd is not guaranteed to be the loop workspace."""
    store = _store(tmp_path)
    store.write(1, "headline")

    rendered = store.render_for_prompt()
    directory = str(store.root.resolve())

    assert directory in rendered
    assert directory.startswith("/")


def test_render_drops_oldest_documents_over_the_char_budget(tmp_path):
    store = _store(tmp_path, recent=5, max_prompt_chars=400)
    for iteration in range(1, 6):
        store.write(iteration, f"headline {iteration}\n" + "x" * 300)

    rendered = store.render_for_prompt()

    # Newest survives, oldest is evicted, and the pointer is never dropped.
    assert "headline 5" in rendered
    assert "headline 1" not in rendered
    assert str(store.root.resolve()) in rendered


def test_render_keeps_one_document_even_when_over_budget(tmp_path):
    """Never return a window with no lesson in it at all."""
    store = _store(tmp_path, recent=3, max_prompt_chars=10)
    for iteration in (1, 2):
        store.write(iteration, f"headline {iteration}\n" + "y" * 500)

    rendered = store.render_for_prompt()
    assert "headline 2" in rendered


def test_default_window_is_five(tmp_path):
    assert DEFAULT_RECENT_LESSONS == 5
    store = _store(tmp_path)
    for iteration in range(1, 8):
        store.write(iteration, f"headline {iteration}")

    rendered = store.render_for_prompt()
    assert "headline 3" in rendered
    assert "headline 2" not in rendered


# ── summarizer prompt ─────────────────────────────────────────────────────────


def test_prompt_demands_every_direction_and_its_result():
    prompt = build_summary_prompt(iteration=5, end_reason="candidate_submitted")

    assert "EVERY direction you tried" in prompt
    assert "actually measured" in prompt
    assert "not measured" in prompt
    assert "incomplete attempt" in prompt


def test_prompt_is_free_form_and_rejects_subjective_direction_judgments():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")

    assert "There is no required output format" in prompt
    assert "hard floor" in prompt
    assert "this direction is exhausted" in prompt
    assert "the next\niteration should" in prompt
    assert "JSON" not in prompt


def test_prompt_flags_a_cut_off_session(tmp_path):
    prompt = build_summary_prompt(iteration=5, end_reason="turn_cap")
    assert "cut off" in prompt
    assert "turn_cap" in prompt


def test_prompt_stays_quiet_for_a_normal_session():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "cut off" not in prompt


@pytest.mark.parametrize(
    "end_reason,expected",
    [
        ("turn_cap", True),
        ("block_budget_exhausted", True),
        ("converged", False),
        ("candidate_submitted", False),
        ("", False),
    ],
)
def test_cutoff_classification(end_reason, expected):
    assert is_cutoff(end_reason) is expected


# ── summarize_iteration ───────────────────────────────────────────────────────


def test_summarize_writes_the_returned_text(tmp_path):
    store = _store(tmp_path)
    seen: list[str] = []

    async def fake_summarizer(prompt: str) -> str:
        seen.append(prompt)
        return "headline\n- [worse] bigger tile | 0.94x"

    outcome = asyncio.run(
        summarize_iteration(
            store=store,
            iteration=2,
            end_reason="turn_cap",
            summarizer=fake_summarizer,
        )
    )

    assert outcome
    assert "bigger tile" in outcome.text
    assert "bigger tile" in store.read(2)
    assert "turn_cap" in seen[0]


def test_summarize_without_a_resumable_provider(tmp_path):
    store = _store(tmp_path)
    outcome = asyncio.run(
        summarize_iteration(
            store=store,
            iteration=2,
            end_reason="",
            summarizer=None,
        )
    )
    assert not outcome
    assert "cannot resume" in outcome.reason
    assert store.existing_iterations() == []


def test_summarize_reports_why_a_failing_session_produced_nothing(tmp_path):
    """The reason reaches the caller: a live run needs it to diagnose."""
    store = _store(tmp_path)

    async def broken(prompt: str) -> str:
        raise RuntimeError("backend exploded")

    outcome = asyncio.run(
        summarize_iteration(
            store=store,
            iteration=2,
            end_reason="",
            summarizer=broken,
        )
    )
    assert not outcome
    assert "RuntimeError" in outcome.reason
    assert "backend exploded" in outcome.reason
    assert store.existing_iterations() == []


def test_summarize_ignores_an_empty_reply(tmp_path):
    store = _store(tmp_path)

    async def empty(prompt: str) -> str:
        return "   \n "

    outcome = asyncio.run(
        summarize_iteration(
            store=store,
            iteration=2,
            end_reason="",
            summarizer=empty,
        )
    )
    assert not outcome
    assert "no text" in outcome.reason
    assert store.existing_iterations() == []


def test_summarize_reports_a_lesson_store_write_failure(tmp_path, monkeypatch):
    store = _store(tmp_path)

    async def summary(_prompt: str) -> str:
        return "use wider vector loads\n- [better] vectorized loads | 1.04x"

    monkeypatch.setattr(store, "write", lambda _iteration, _text: None)
    outcome = asyncio.run(
        summarize_iteration(
            store=store,
            iteration=2,
            end_reason="",
            summarizer=summary,
        )
    )

    assert not outcome
    assert outcome.reason == "failed to persist lesson document"
    assert store.existing_iterations() == []


# ── machine-written fallback ──────────────────────────────────────────────────


def test_fallback_document_carries_the_gate_rejections():
    doc = build_fallback_document(
        diff_summary="softmax_kernel.py | 12 +++---",
        findings=(
            "Your change is NOT finished: the kernel fails correctness.\ndetail\n"
            "---\n"
            "The kernel is CORRECT but NOT faster than the current best.\nmore"
        ),
        end_reason="block_budget_exhausted",
    )

    assert doc.splitlines()[0].startswith("(no agent summary)")
    assert "2 gate rejection(s)" in doc.splitlines()[0]
    assert "softmax_kernel.py | 12 +++---" in doc
    assert "fails correctness" in doc
    assert "NOT faster" in doc
    assert "block_budget_exhausted" in doc
    # It must not pass itself off as the agent's own account.
    assert "no account of" in doc


def test_fallback_document_without_findings_still_records_the_diff():
    doc = build_fallback_document(
        diff_summary="a.py | 1 +",
        findings="",
        end_reason="",
    )
    assert doc.splitlines()[0].startswith("(no agent summary)")
    assert "a.py | 1 +" in doc


def test_fallback_document_is_empty_when_nothing_was_observed():
    assert (
        build_fallback_document(
            diff_summary="",
            findings="",
            end_reason="turn_cap",
        )
        == ""
    )


def test_fallback_document_caps_the_rejection_list():
    findings = "\n---\n".join(f"rejection {i}" for i in range(20))
    doc = build_fallback_document(
        diff_summary="",
        findings=findings,
        end_reason="",
        max_findings=3,
    )
    listed = [line for line in doc.splitlines() if line.startswith("- rejection")]
    assert len(listed) == 3
    assert "- rejection 19" in doc  # newest kept
    assert "- rejection 0" not in doc  # oldest dropped


# ── the scope a result was measured under ─────────────────────────────────────

_SPLIT_K = LessonScope(
    cases=("decode-t1",),
    held_fixed=(("BLOCK_N", "16"), ("num_warps", "8")),
    lane_restricted=True,
)


def test_scope_line_round_trips():
    parsed = parse_scope_line(f"prose\n{format_scope_line(_SPLIT_K)}\nmore")
    assert parsed == _SPLIT_K


def test_scope_line_round_trips_when_nothing_was_recorded():
    empty = LessonScope()
    parsed = parse_scope_line(format_scope_line(empty))
    assert parsed == empty
    assert "(not recorded)" in format_scope_line(empty)


def test_a_document_without_a_scope_line_parses_to_none():
    assert parse_scope_line("swept split-K; all slower") is None


def test_held_fixed_is_read_only_from_its_own_marker_lines():
    """A pair in the prose is as likely a result as a premise."""
    document = (
        "swept split-K, the best point was SPLIT_K=4 at 13.19 us\n"
        "HELD-FIXED: BLOCK_N=16, num_warps=8\n"
        "HELD-FIXED: BLOCK_N=64\n"
    )
    assert parse_held_fixed(document) == (("BLOCK_N", "16"), ("num_warps", "8"))
    assert parse_held_fixed("SPLIT_K=4 was best") == ()


def test_scan_constant_values_finds_every_assignment():
    """A call site passing a LITERAL is a real pin of that literal.

    In Triton that is where tile sizes and warp counts live, so both the
    module-level binding and the launch keyword are values the name is pinned
    to right now. A name the source never mentions is a different fact.
    """
    source = "BLOCK_N = 64\nnum_warps=8\nfoo(BLOCK_N=128)\n"
    found = scan_constant_values(source, ["BLOCK_N", "num_warps", "SPLIT_K"])
    assert found["BLOCK_N"] == ("64", "128")
    assert found["num_warps"] == ("8",)
    assert "SPLIT_K" not in found


def test_scope_holds_when_the_case_and_the_constants_still_match():
    assert (
        scope_conflicts(
            _SPLIT_K,
            current_cases=("decode-t1",),
            kernel_source="BLOCK_N = 16\nnum_warps = 8\n",
        )
        == ()
    )


def test_a_case_outside_the_measured_scope_reopens_the_negative():
    reasons = scope_conflicts(
        _SPLIT_K,
        current_cases=("decode-t1", "decode-t64", "prefill-t16384"),
        kernel_source="BLOCK_N = 16\nnum_warps = 8\n",
    )
    assert reasons == ("not measured on decode-t64, prefill-t16384",)


def test_a_moved_held_fixed_value_reopens_the_negative():
    reasons = scope_conflicts(
        _SPLIT_K,
        current_cases=("decode-t1",),
        kernel_source="BLOCK_N = 64\nnum_warps = 8\n",
    )
    assert reasons == ("BLOCK_N is now 64 (pinned at 16 when this was measured)",)


def test_a_held_fixed_constant_that_no_longer_exists_reopens_the_negative():
    reasons = scope_conflicts(
        _SPLIT_K,
        current_cases=("decode-t1",),
        kernel_source="num_warps = 8\n",
    )
    assert reasons == ("BLOCK_N is not assigned in the kernel source checked (pinned at 16 when this was measured)",)


def test_an_unchecked_kernel_source_is_reported_rather_than_assumed_clean():
    reasons = scope_conflicts(_SPLIT_K, current_cases=("decode-t1",))
    assert reasons == ("held-fixed values were not checked against the current kernel",)


def test_a_scope_with_nothing_recorded_cannot_close_anything():
    reasons = scope_conflicts(
        LessonScope(),
        current_cases=("decode-t1",),
        kernel_source="",
    )
    assert reasons == (
        "the cases it was measured on were not recorded",
        "the constants it was measured under were not recorded",
    )


def test_an_unrecorded_premise_reopens_even_inside_the_measured_cases():
    """Not knowing what was pinned is not the same as nothing being pinned."""
    reasons = scope_conflicts(
        LessonScope(cases=("decode-t1",)),
        current_cases=("decode-t1",),
        kernel_source="BLOCK_N = 16\n",
    )
    assert reasons == ("the constants it was measured under were not recorded",)


def test_no_scope_yields_no_conflicts_of_its_own():
    """An unscoped document is handled by the renderer, not by this check."""
    assert scope_conflicts(None, current_cases=("decode-t1",)) == ()


def test_cases_named_in_recovers_a_lane_restriction():
    assert cases_named_in(
        "Tune the decode-t1 dispatch only; leave the others alone.",
        ("decode-t1", "decode-t64", "prefill-t16384"),
    ) == ("decode-t1",)


# ── the scope in the store and in the prompt ──────────────────────────────────


def test_append_scope_preserves_the_narrative(tmp_path):
    store = _store(tmp_path)
    store.write(3, "swept split-K")
    assert store.append_scope(3, _SPLIT_K)

    assert "swept split-K" in store.read(3)
    assert store.scope_of(3) == _SPLIT_K


def test_append_scope_without_a_narrative_still_records(tmp_path):
    """An outcome-only document must still say what it was measured under."""
    store = _store(tmp_path)
    assert store.append_scope(5, _SPLIT_K)
    assert store.scope_of(5) == _SPLIT_K


def test_append_scope_reports_a_store_write_failure(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.write(3, "swept split-K")
    monkeypatch.setattr(store, "write", lambda _iteration, _text: None)
    assert store.append_scope(3, _SPLIT_K) is False


def test_scope_of_a_document_without_one_is_none(tmp_path):
    store = _store(tmp_path)
    store.write(3, "swept split-K")
    assert store.scope_of(3) is None


def test_render_marks_a_still_valid_negative_as_in_scope(tmp_path):
    store = _store(tmp_path)
    store.write(1, "swept split-K on decode-t1; all slower")
    store.append_scope(1, _SPLIT_K)

    rendered = store.render_for_prompt(
        current_cases=("decode-t1",),
        kernel_source="BLOCK_N = 16\nnum_warps = 8\n",
    )
    assert "VALIDITY: IN SCOPE" in rendered
    assert "do not re-derive it" in rendered


def test_render_marks_a_negative_cited_outside_its_scope_as_reopenable(tmp_path):
    store = _store(tmp_path)
    store.write(1, "swept split-K on decode-t1; all slower")
    store.append_scope(1, _SPLIT_K)

    rendered = store.render_for_prompt(
        current_cases=("decode-t1", "decode-t64"),
        kernel_source="BLOCK_N = 64\n",
    )
    assert "VALIDITY: RE-OPENABLE" in rendered
    assert "not measured on decode-t64" in rendered
    assert "BLOCK_N is now 64" in rendered
    # The record itself is still there — re-openable, not deleted.
    assert "all slower" in rendered


def test_render_keeps_an_unscoped_legacy_document_and_says_it_is_unscoped(tmp_path):
    store = _store(tmp_path)
    store.write(1, "SPLIT-K IS A MEASURED DEAD END -- do not re-try")

    rendered = store.render_for_prompt(current_cases=("decode-t1",))
    assert "MEASURED DEAD END" in rendered
    assert "VALIDITY: UNSCOPED" in rendered
    assert "closes nothing on its own" in rendered


def test_render_without_scope_inputs_still_states_the_citation_rule(tmp_path):
    store = _store(tmp_path)
    store.write(1, "headline")
    assert "closes nothing on its own" in store.render_for_prompt()


def test_prompt_asks_for_the_premise_behind_a_negative():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "HELD-FIXED:" in prompt
    assert "Name the cases in the same sentence" in prompt


# ── reading a constant out of a real kernel ───────────────────────────────────

_EXAMPLES = resource_path("examples")
_MXFP8_KERNEL = _EXAMPLES / "triton2flydsl-mxfp8-grouped-gemm" / "mxfp8_grouped_gemm.py"
_SOFTMAX_KERNEL = _EXAMPLES / "triton-softmax-forge-loop" / "softmax_kernel.py"


def test_a_constexpr_parameter_is_not_an_assignment_of_the_constant():
    """`BLOCK_N: tl.constexpr` and `BLOCK_N=block_n` say nothing about a pin.

    Reading either as the kernel's current value renders "BLOCK_N is now
    tl.constexpr/block_n" into every implementer prompt — a false statement
    about the source, which closes or re-opens an axis on a fiction. Reporting
    the name as absent is the opposite false statement: the kernel plainly runs
    at some BLOCK_N. Neither: the name is there, no literal was read for it.
    """
    source = _MXFP8_KERNEL.read_text()
    assert "BLOCK_N: tl.constexpr" in source  # the annotation is there
    assert "BLOCK_N=block_n" in source  # so is the call-site keyword

    assert scan_constant_values(source, ["BLOCK_N"]) == {"BLOCK_N": ()}


def test_a_call_site_keyword_is_not_an_assignment_of_the_constant():
    source = _SOFTMAX_KERNEL.read_text()
    assert "num_warps=num_warps" in source  # the call-site keyword

    assert scan_constant_values(source, ["num_warps"]) == {"num_warps": ("1",)}


def test_a_tuning_table_entry_counts_as_an_assignment():
    """A dict literal is exactly the shape a pinned tile constant lives in."""
    source = 'CONFIG = {"BLOCK_N": 16, "num_warps": 8}\n'
    assert scan_constant_values(source, ["BLOCK_N", "num_warps"]) == {
        "BLOCK_N": ("16",),
        "num_warps": ("8",),
    }


def test_an_annotated_assignment_records_the_value_not_the_annotation():
    """A bare declaration binds no value, but it is not an absent name either."""
    found = scan_constant_values("BLOCK_N: int = 16\nSPLIT_K: int\n", ["BLOCK_N", "SPLIT_K"])
    assert found == {"BLOCK_N": ("16",), "SPLIT_K": ()}


def test_a_parameter_default_is_not_an_assignment_of_the_constant():
    """A parameter is a name the caller supplies, so no value was checked.

    Reporting {} would say the source dropped BLOCK_N, and it plainly has not:
    the name is right there. "Not checked" is the fact, and it is rendered as
    such rather than as a constant that is gone.
    """
    source = "def launch(BLOCK_N=32):\n    return BLOCK_N\n"
    assert scan_constant_values(source, ["BLOCK_N"]) == {"BLOCK_N": ()}


def test_a_source_that_cannot_be_parsed_is_not_a_source_that_dropped_it():
    """ "Not assigned" and "could not be checked" are different facts."""
    assert scan_constant_values("def broken(:\n", ["BLOCK_N"]) is None


def test_a_constant_is_looked_for_in_every_declared_source_file():
    """Tile and dispatch constants move to a sibling file; that is not gone."""
    found = scan_sources_for_constants(["import config\n", "BLOCK_N = 16\n"], ["BLOCK_N"])
    assert found == {"BLOCK_N": ("16",)}


def test_a_constant_absent_from_every_parsed_file_is_absent():
    found = scan_sources_for_constants(["import config\n", "num_warps = 8\n"], ["BLOCK_N"])
    assert found == {}


def test_nothing_is_known_when_no_source_file_could_be_parsed():
    assert scan_sources_for_constants(["def broken(:\n"], ["BLOCK_N"]) is None
    assert scan_sources_for_constants([], ["BLOCK_N"]) is None


def test_an_unparsable_source_is_reported_as_unchecked_not_as_moved():
    reasons = scope_conflicts(
        _SPLIT_K,
        current_cases=("decode-t1",),
        kernel_source="def broken(:\n",
    )
    assert reasons == ("held-fixed values were not checked: the declared source could not be parsed",)


def test_a_pin_still_held_in_a_sibling_file_keeps_the_negative_in_scope():
    assert (
        scope_conflicts(
            _SPLIT_K,
            current_cases=("decode-t1",),
            kernel_source=["BLOCK_N = 16\n", "num_warps = 8\n"],
        )
        == ()
    )


# ── a document with nothing negative in it has nothing to re-open ─────────────


def test_a_document_with_no_negative_is_not_reopened_by_an_unrecorded_premise():
    """An all-positive iteration closed nothing, so there is nothing to re-open."""
    assert (
        scope_conflicts(
            LessonScope(cases=("decode-t1",), carries_negative=False),
            current_cases=("decode-t1",),
            kernel_source="BLOCK_N = 16\n",
        )
        == ()
    )


def test_a_recorded_negative_without_a_premise_still_reopens():
    assert scope_conflicts(
        LessonScope(cases=("decode-t1",), carries_negative=True),
        current_cases=("decode-t1",),
        kernel_source="BLOCK_N = 16\n",
    ) == ("the constants it was measured under were not recorded",)


def test_a_document_with_no_negative_still_reopens_outside_its_cases():
    """The case scope is about where a number was taken, negative or not."""
    assert scope_conflicts(
        LessonScope(cases=("decode-t1",), carries_negative=False),
        current_cases=("decode-t1", "decode-t64"),
    ) == ("not measured on decode-t64",)


def test_the_negative_flag_round_trips_in_both_states():
    for flag in (True, False, None):
        scope = LessonScope(cases=("decode-t1",), carries_negative=flag)
        assert parse_scope_line(format_scope_line(scope)) == scope


def test_a_positive_iteration_renders_in_scope(tmp_path):
    store = _store(tmp_path)
    store.write(1, "widened the tile; 1.14x on decode-t1")
    store.append_scope(1, LessonScope(cases=("decode-t1",), carries_negative=False))

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: IN SCOPE" in rendered
    assert "VALIDITY: RE-OPENABLE" not in rendered


# ── one case id must not swallow another ──────────────────────────────────────


def test_a_case_id_inside_a_longer_id_is_not_named():
    """The dangerous direction: a scope wider than what was measured."""
    assert cases_named_in("focus on decode-t16 only", ("decode-t1", "decode-t16", "prefill-t1")) == ("decode-t16",)


def test_a_case_id_is_named_next_to_ordinary_punctuation():
    assert cases_named_in(
        "tune (decode-t1, prefill-t1) and nothing else.",
        ("decode-t1", "decode-t16", "prefill-t1"),
    ) == ("decode-t1", "prefill-t1")


# ── what the document itself says about its negatives ─────────────────────────


def test_the_negatives_marker_reads_three_states():
    """ "None recorded" is not "none happened", and must not become one."""
    assert parse_negatives_marker("tried three tiles\nNEGATIVES: BLOCK_N=128 measured 0.94x") is True
    assert parse_negatives_marker("widened the tile\nNEGATIVES: none") is False
    assert parse_negatives_marker("widened the tile; 1.2x") is None
    assert parse_negatives_marker("") is None
    assert parse_negatives_marker("NEGATIVES:") is None


def test_a_named_negative_outweighs_a_none_line():
    """The marker is a presence check: one named negative means the document has one."""
    assert parse_negatives_marker("NEGATIVES: none\nNEGATIVES: split-K=4 at 0.91x") is True


def test_the_negatives_marker_survives_a_markdown_list_item():
    assert parse_negatives_marker("- **NEGATIVES:** none.") is False


def test_the_prompt_requires_the_negatives_marker_in_both_directions():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "NEGATIVES: none" in prompt
    assert "including\n   ones you reverted inside the session" in prompt


def test_a_full_scope_line_round_trips_in_all_three_negative_states():
    """Every field at once: a partial parse silently widens a recorded scope."""
    for flag in (True, False, None):
        scope = LessonScope(
            cases=("decode-t1", "prefill-t16384"),
            held_fixed=(("BLOCK_N", "16"), ("num_warps", "8")),
            lane_restricted=True,
            carries_negative=flag,
        )
        line = format_scope_line(scope)
        assert parse_scope_line(f"prose\n{line}\nmore prose") == scope
    unknown = format_scope_line(LessonScope(cases=("decode-t1",)))
    # An unknown answer must never render as the answer "no".
    assert "no measured negative" not in unknown
    assert "not recorded" in unknown


# ── a premise that could only be checked in part ──────────────────────────────


def test_a_pin_living_in_the_unparsable_file_is_not_reported_as_gone():
    """One broken file among several must not indict the constant it holds.

    Skipping it and reading the survivors as the whole declared set turns "the
    file I could not read" into "the task deleted this constant".
    """
    mapping, complete = scan_sources_with_coverage(["def broken(:\n", "num_warps = 8\n"], ["BLOCK_N", "num_warps"])
    assert mapping == {"num_warps": ("8",)}
    assert complete is False

    reasons = scope_conflicts(
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
        current_cases=("decode-t1",),
        kernel_source=["def broken(:\n", "num_warps = 8\n"],
    )
    assert reasons == (
        "BLOCK_N was not checked: part of the declared source could not be "
        "read or parsed (pinned at 16 when this was measured)",
    )
    assert "is not assigned" not in " ".join(reasons)


def test_a_declared_file_that_could_not_be_read_travels_as_unchecked():
    """None INSIDE the list is one unreadable file, not a shorter source set."""
    mapping, complete = scan_sources_with_coverage([None, "num_warps = 8\n"], ["BLOCK_N"])
    assert mapping == {} and complete is False
    reasons = scope_conflicts(
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
        current_cases=("decode-t1",),
        kernel_source=[None, "num_warps = 8\n"],
    )
    assert reasons == (
        "BLOCK_N was not checked: part of the declared source could not be "
        "read or parsed (pinned at 16 when this was measured)",
    )


def test_a_moved_value_is_still_reported_when_part_of_the_source_is_unchecked():
    """ "is now X" is an observation about a file that WAS read, not an inference."""
    reasons = scope_conflicts(
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
        current_cases=("decode-t1",),
        kernel_source=["def broken(:\n", "BLOCK_N = 64\n"],
    )
    assert reasons == ("BLOCK_N is now 64 (pinned at 16 when this was measured)",)


# ── a launch keyword is where a Triton constant is actually pinned ────────────


def test_a_launch_keyword_holding_a_literal_keeps_the_pin_in_scope():
    """`num_warps=8` at the launch IS the pin; reporting it gone defeats the check."""
    source = "def launch(x):\n    kernel[(1,)](x, BLOCK_N=128, num_warps=8)\n"
    assert scan_constant_values(source, ["BLOCK_N", "num_warps"]) == {
        "BLOCK_N": ("128",),
        "num_warps": ("8",),
    }
    assert (
        scope_conflicts(
            LessonScope(
                cases=("decode-t1",),
                held_fixed=(("BLOCK_N", "128"), ("num_warps", "8")),
                carries_negative=True,
            ),
            current_cases=("decode-t1",),
            kernel_source=source,
        )
        == ()
    )


def test_an_autotune_config_entry_is_a_pin():
    source = "CONFIGS = [triton.Config({'BLOCK_N': 64}, num_warps=8)]\n"
    assert scan_constant_values(source, ["BLOCK_N", "num_warps"]) == {
        "BLOCK_N": ("64",),
        "num_warps": ("8",),
    }


def test_a_keyword_forwarding_a_local_is_a_value_that_was_not_checked():
    """`BLOCK_N=block_n` says the name is live and says nothing about its value."""
    assert scan_constant_values("kernel(BLOCK_N=block_n)\n", ["BLOCK_N"]) == {"BLOCK_N": ()}
    reasons = scope_conflicts(
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "128"),),
            carries_negative=True,
        ),
        current_cases=("decode-t1",),
        kernel_source="kernel(BLOCK_N=block_n)\n",
    )
    assert reasons == (
        "BLOCK_N was not checked: the source names it but binds no literal to "
        "it (pinned at 128 when this was measured)",
    )


def test_a_walrus_binding_is_a_binding():
    assert scan_constant_values("if (BLOCK_N := 16):\n    pass\n", ["BLOCK_N"]) == {"BLOCK_N": ("16",)}


# ── one right-hand side is not every name's value ─────────────────────────────


def test_a_tuple_unpacking_pairs_each_name_with_its_own_element():
    assert scan_constant_values("BLOCK_M, BLOCK_N = 64, 32\n", ["BLOCK_M", "BLOCK_N"]) == {
        "BLOCK_M": ("64",),
        "BLOCK_N": ("32",),
    }
    assert (
        scope_conflicts(
            LessonScope(
                cases=("decode-t1",),
                held_fixed=(("BLOCK_N", "32"),),
                carries_negative=True,
            ),
            current_cases=("decode-t1",),
            kernel_source="BLOCK_M, BLOCK_N = 64, 32\n",
        )
        == ()
    )


def test_an_unpairable_tuple_target_records_no_value_rather_than_a_wrong_one():
    """ "is now (64, 32)" is a false statement; "not checked" is a true one."""
    assert scan_constant_values("BLOCK_M, BLOCK_N = shape()\n", ["BLOCK_N"]) == {"BLOCK_N": ()}
    assert scan_constant_values("BLOCK_M, *rest = 64, 32, 16\n", ["BLOCK_M"]) == {"BLOCK_M": ()}


# ── the same number written two ways is the same pin ──────────────────────────


def test_a_pin_recorded_as_an_int_matches_a_float_of_the_same_value():
    assert (
        scope_conflicts(
            LessonScope(
                cases=("decode-t1",),
                held_fixed=(("BLOCK_N", "16"),),
                carries_negative=True,
            ),
            current_cases=("decode-t1",),
            kernel_source="BLOCK_N = 16.0\n",
        )
        == ()
    )


def test_a_pin_recorded_in_hex_matches_the_same_decimal_value():
    assert (
        scope_conflicts(
            LessonScope(
                cases=("decode-t1",),
                held_fixed=(("BLOCK_N", "16"),),
                carries_negative=True,
            ),
            current_cases=("decode-t1",),
            kernel_source="BLOCK_N = 0x10\n",
        )
        == ()
    )


def test_a_genuinely_different_number_still_reopens_the_negative():
    assert scope_conflicts(
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
        current_cases=("decode-t1",),
        kernel_source="BLOCK_N = 16.5\n",
    ) == ("BLOCK_N is now 16.5 (pinned at 16 when this was measured)",)


def test_a_long_value_is_marked_where_it_was_cut():
    """A truncated expression must not reach a prompt looking complete."""
    long_value = " + ".join(str(n) for n in range(40))
    found = scan_constant_values(f"BLOCK_N = {long_value}\n", ["BLOCK_N"])
    assert found["BLOCK_N"][0].endswith(" ...")


def test_scanning_for_no_names_still_reports_an_unparsable_source():
    """ "Nothing was asked" and "nothing could be read" are different answers."""
    assert scan_constant_values("def broken(:\n", []) is None
    assert scan_constant_values("BLOCK_N = 16\n", []) == {}


# ── a "cannot" must carry the experiment that would have falsified it ─────────

_UNREACHABLE = LessonScope(
    cases=("decode-t1",),
    held_fixed=(("BLOCK_N", "16"),),
    carries_negative=True,
    disproof=UNDISPROVEN_CLAIM,
)


def test_a_named_disproof_round_trips_on_the_scope_line():
    scope = LessonScope(
        cases=("decode-t1",),
        held_fixed=(("BLOCK_N", "16"),),
        lane_restricted=True,
        carries_negative=True,
        disproof="build-only ISA screen of ds_read_b64_tr_b16; it assembled",
    )
    line = format_scope_line(scope)
    assert parse_scope_line(f"prose\n{line}\nmore prose") == scope


def test_every_disproof_state_round_trips():
    """Five different facts, and none of them may become another in transit."""
    for answer in (
        None,
        NO_FEASIBILITY_CLAIM,
        UNDISPROVEN_CLAIM,
        "one probe",
        CLAIM_DISPROVED + "the ISA screen assembled it",
    ):
        scope = LessonScope(cases=("decode-t1",), disproof=answer)
        assert parse_scope_line(format_scope_line(scope)) == scope


def test_a_disproved_claim_is_never_read_as_a_surviving_one():
    """The two verdicts are opposite, so neither may match inside the other."""
    line = format_scope_line(LessonScope(disproof=CLAIM_DISPROVED + "dir() lists the emitter method"))
    assert parse_scope_line(line).disproof != UNDISPROVEN_CLAIM
    assert UNDISPROVEN_CLAIM not in line
    assert parse_scope_line(format_scope_line(LessonScope(disproof=UNDISPROVEN_CLAIM))).disproof == UNDISPROVEN_CLAIM


def test_a_disproof_with_no_evidence_behind_it_reopens_instead():
    """Nothing to repeat is nothing to stand on, whichever way it came out."""
    scope = LessonScope(cases=("decode-t1",), disproof=CLAIM_DISPROVED)
    assert parse_scope_line(format_scope_line(scope)).disproof == UNDISPROVEN_CLAIM


def test_a_document_from_before_the_disproof_field_reads_as_unknown():
    """An absent field is "nobody asked", never "the premise was tested"."""
    legacy = (
        "SPLIT-K CANNOT WORK ON THIS BUILD\n"
        "SCOPE: measured on decode-t1 | held fixed BLOCK_N=16 | "
        "carries a measured negative"
    )
    scope = parse_scope_line(legacy)
    assert scope.disproof is None
    assert scope.carries_negative is True
    rendered = format_scope_line(scope)
    assert "feasibility claim tested by" not in rendered
    assert "no feasibility claim" not in rendered
    assert "whether a feasibility claim was disproved was not recorded" in rendered


def test_an_unrecorded_disproof_does_not_convict_a_document_by_itself(tmp_path):
    """The transition case: the state this field deliberately does not fire on.

    A document whose summarizer never answered the question can still contain a
    "cannot" sentence. Firing on that would convict every record written before
    the marker existed of a claim most of them never made, and a verdict every
    document receives ranks none of them. So the note stays IN SCOPE, and what
    separates this document from a tested one is the SCOPE line plus the
    citation rule — until summarizers emit the marker, that prose is the whole
    protection, which is why it is asserted here and not only in the rule's own
    test.
    """
    store = _store(tmp_path)
    store.write(1, "the only real fix is a transposing LDS read; THIS BUILD CANNOT")
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
    )

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: IN SCOPE" in rendered
    assert "whether a feasibility claim was disproved was not recorded" in rendered
    assert "feasibility claim tested by" not in rendered
    assert "does not record the question at all" in rendered


def test_an_undisproven_cannot_claim_renders_reopenable(tmp_path):
    store = _store(tmp_path)
    store.write(
        1,
        "the only real fix is a transposing LDS read; THIS BUILD CANNOT REACH IT",
    )
    store.append_scope(1, _UNREACHABLE)

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: RE-OPENABLE (undisproven feasibility claim)" in rendered
    assert "names no experiment" in rendered
    # Re-openable, not deleted: the record of what was tried is still there.
    assert "transposing LDS read" in rendered


def test_a_cannot_claim_whose_experiment_was_run_stays_in_scope(tmp_path):
    store = _store(tmp_path)
    store.write(
        1,
        "the only real fix is a transposing LDS read; THIS BUILD CANNOT REACH IT",
    )
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
            disproof="build-only ISA screen: ds_read_b64_tr_b16 fails to assemble",
        ),
    )

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: IN SCOPE" in rendered
    assert "VALIDITY: RE-OPENABLE" not in rendered
    assert "feasibility claim tested by build-only ISA screen" in rendered


def test_a_claim_its_own_experiment_refuted_does_not_keep_suppressing(tmp_path):
    """The inversion: a summarizer reporting its own premise FALSE.

    "falsified — gfx950 accepts the instruction" says the axis is reachable.
    Scoring that as an obligation discharged would leave the document IN SCOPE
    and the closure still suppressing the direction the same line proved open,
    which is the one outcome this marker must never produce.
    """
    store = _store(tmp_path)
    store.write(
        1,
        "the only real fix is a transposing LDS read; THIS BUILD CANNOT REACH IT\n"
        "DISPROOF: falsified — an ISA screen shows gfx950 assembles "
        "ds_read_b64_tr_b16",
    )
    document = store.read(1)
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
            disproof=parse_disproof_marker(document),
        ),
    )

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: IN SCOPE" not in rendered
    assert "VALIDITY: RE-OPEN (feasibility claim disproved)" in rendered
    assert "feasibility claim disproved by an ISA screen" in rendered
    assert "feasibility claim tested by" not in rendered
    assert "re-enter it" in rendered


def test_a_disproved_claim_outranks_the_scope_checks_it_passes(tmp_path):
    """Every pin in place and every case current still does not close it."""
    store = _store(tmp_path)
    store.write(1, "split-K CANNOT be reached from this template")
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1", "decode-t64"),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
            disproof=CLAIM_DISPROVED + "one probe call reached the split-K path",
        ),
    )

    rendered = store.render_for_prompt(current_cases=("decode-t1", "decode-t64"), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: RE-OPEN (feasibility claim disproved)" in rendered
    assert "certifies no" in rendered


def test_a_measured_closure_without_a_disproof_is_still_reopenable(tmp_path):
    """The case the naive rule gets wrong, and the reason for this field.

    Everything the older reading asked for is present: real numbers, the pins
    they were taken under, and the case they were taken on, all still current.
    What is missing is any test of the premise beside them — and a premise is
    what closed the axis, not the numbers.
    """
    store = _store(tmp_path)
    store.write(
        1,
        "peeling the GQA head loop cost 0.206 -> 0.237 ms on decode-t1, and "
        "0.244 ms with the second variant; the general form needs a "
        "data-dependent branch, which this kernel cannot express\n"
        "HELD-FIXED: BLOCK_N=16",
    )
    store.append_scope(1, _UNREACHABLE)

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: RE-OPENABLE (undisproven feasibility claim)" in rendered
    assert "0.237 ms" in rendered
    # The measurement is intact and in scope; only the premise is re-opened.
    assert "whatever else the document measured" in rendered


def test_an_undisproven_claim_out_of_scope_reports_both_reasons(tmp_path):
    store = _store(tmp_path)
    store.write(1, "split-K CANNOT be reached from this template")
    store.append_scope(1, _UNREACHABLE)

    rendered = store.render_for_prompt(current_cases=("decode-t1", "decode-t64"), kernel_source="BLOCK_N = 64\n")
    assert "VALIDITY: RE-OPENABLE (undisproven feasibility claim)" in rendered
    assert "not measured on decode-t64" in rendered
    assert "BLOCK_N is now 64" in rendered


def test_a_document_claiming_nothing_unreachable_is_not_reopened_by_this(tmp_path):
    """The obligation is on feasibility claims, not on every measured negative."""
    store = _store(tmp_path)
    store.write(1, "swept split-K on decode-t1; all slower")
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1",),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
            disproof=NO_FEASIBILITY_CLAIM,
        ),
    )

    rendered = store.render_for_prompt(current_cases=("decode-t1",), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: IN SCOPE" in rendered
    assert "do not re-derive it" in rendered


def test_a_pipe_in_a_named_experiment_does_not_split_the_scope_line():
    """The line is pipe-separated; a named experiment must not add a field."""
    scope = LessonScope(disproof="objdump | grep ds_read_b64_tr_b16")
    parsed = parse_scope_line(format_scope_line(scope))
    assert parsed.disproof == "objdump / grep ds_read_b64_tr_b16"
    assert parsed.carries_negative is None


def test_a_long_named_experiment_is_marked_where_it_was_cut():
    line = format_scope_line(LessonScope(disproof="probe " * 60))
    assert line.endswith(" ...")


# ── what the document itself says it ran against its own "cannot" ─────────────


def test_the_disproof_marker_reads_four_states():
    assert (
        parse_disproof_marker(
            "the template CANNOT emit it\nDISPROOF: tested — a build-only screen rejected the instruction"
        )
        == "a build-only screen rejected the instruction"
    )
    assert (
        parse_disproof_marker("DISPROOF: untested — one dir() over the installed emitter would settle it")
        == UNDISPROVEN_CLAIM
    )
    assert parse_disproof_marker("widened the tile\nDISPROOF: none") == NO_FEASIBILITY_CLAIM
    assert parse_disproof_marker("widened the tile; 1.2x") is None
    assert parse_disproof_marker("DISPROOF:") is None


def test_an_experiment_that_was_run_but_not_named_is_not_a_disproof():
    """A later iteration has to be able to repeat it, or it settles nothing."""
    assert parse_disproof_marker("DISPROOF: tested") == UNDISPROVEN_CLAIM


def test_an_unrecognized_disproof_answer_is_read_as_an_open_obligation():
    """Be wrong in the direction that re-opens an axis, never the one that closes it."""
    assert parse_disproof_marker("DISPROOF: this would need a redesign of the whole template") == UNDISPROVEN_CLAIM


def test_an_outstanding_obligation_outweighs_a_discharged_one():
    assert (
        parse_disproof_marker(
            "DISPROOF: tested — the ISA screen rejected it\nDISPROOF: untested — nothing was run on the second claim"
        )
        == UNDISPROVEN_CLAIM
    )
    assert (
        parse_disproof_marker("DISPROOF: none\nDISPROOF: tested — the ISA screen rejected it")
        == "the ISA screen rejected it"
    )


def test_a_falsifying_result_is_not_an_obligation_discharged():
    """ "The experiment ran" and "the experiment won" are opposite answers."""
    verdict = parse_disproof_marker(
        "the template CANNOT emit it\nDISPROOF: falsified — the ISA screen shows gfx950 accepts ds_read_b64_tr_b16"
    )
    assert is_claim_disproved(verdict)
    assert verdict == (CLAIM_DISPROVED + "the ISA screen shows gfx950 accepts ds_read_b64_tr_b16")
    assert is_claim_disproved(parse_disproof_marker("DISPROOF: disproved — dir() lists the method"))
    assert not is_claim_disproved(parse_disproof_marker("DISPROOF: tested — the ISA screen rejected it"))
    assert not is_claim_disproved(UNDISPROVEN_CLAIM)
    assert not is_claim_disproved(None)


def test_a_falsification_nobody_can_repeat_is_not_one():
    """Same rule as an unnamed experiment: it re-opens, it does not settle."""
    assert parse_disproof_marker("DISPROOF: falsified") == UNDISPROVEN_CLAIM


def test_a_disproved_claim_outranks_every_other_answer():
    """It is the only answer that is a fact about the route, not the variant."""
    for document in (
        "DISPROOF: untested — nothing was run on the first claim\n"
        "DISPROOF: disproved — the installed module binds the symbol",
        "DISPROOF: disproved — the installed module binds the symbol\n"
        "DISPROOF: untested — nothing was run on the second claim",
        "DISPROOF: tested — the ISA screen rejected it\nDISPROOF: disproved — the installed module binds the symbol",
        "DISPROOF: none\nDISPROOF: disproved — the installed module binds the symbol",
    ):
        assert parse_disproof_marker(document) == (CLAIM_DISPROVED + "the installed module binds the symbol"), document


def test_an_obligation_after_a_discharged_line_still_wins():
    """The scan reads every marker now; ranking must not become line order."""
    assert (
        parse_disproof_marker(
            "DISPROOF: untested — one dir() would settle it\nDISPROOF: tested — the ISA screen rejected it"
        )
        == UNDISPROVEN_CLAIM
    )


def test_the_disproof_marker_survives_a_markdown_list_item():
    assert parse_disproof_marker("- **DISPROOF:** none.") == NO_FEASIBILITY_CLAIM


# ── what the summarizer is asked for ──────────────────────────────────────────


def test_the_prompt_demands_the_cheapest_falsifying_experiment():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "DISPROOF:" in prompt
    assert "CHEAPEST experiment that would show that claim\n   to be FALSE" in prompt
    assert '"Further investigation"' in prompt
    assert "DISPROOF: none" in prompt


def test_the_prompt_asks_for_the_disproof_line_either_way():
    """An absent line must mean an unanswered question, not an answered one."""
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "read as never having\n   answered the question" in prompt


def test_the_prompt_names_the_four_reach_classes():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "REBIND AN INSTALLED SYMBOL" in prompt
    assert "INJECT DEVICE-SIDE SOURCE THROUGH THE FRAMEWORK'S OWN HOOK" in prompt
    assert "CHANGE A MODULE-LEVEL CONSTANT ANOTHER MODULE'S DISPATCH READS" in prompt
    assert "APPEND A ROW TO A PERMITTED DATA OR CONFIG FILE" in prompt
    assert "os.environ" in prompt


def test_the_reach_classes_are_asked_about_rather_than_offered_as_routes():
    """A checklist read as a list of routes only makes the closed list longer."""
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "why each one does not apply" in prompt
    assert "not routes to try and tick off" in prompt


def test_the_summary_prompt_says_a_number_does_not_discharge_a_cannot():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "a number says what the variant you ran did" in prompt


def test_the_citation_rule_covers_a_cannot_nobody_asked_about(tmp_path):
    """The rule is what protects a document written before the marker existed."""
    store = _store(tmp_path)
    store.write(1, "headline")
    rule = store.render_for_prompt()
    assert "does not record the question at all" in rule
    assert "closes nothing however many numbers surround it" in rule


def test_the_citation_rule_explains_a_disproved_claim(tmp_path):
    """A reader meeting the strongest verdict must know it points at a route."""
    store = _store(tmp_path)
    store.write(1, "headline")
    rule = store.render_for_prompt()
    assert "the claim was DISPROVED" in rule
    assert "known reachable" in rule


def test_the_summary_prompt_asks_which_way_the_experiment_came_out():
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert "DISPROOF: disproved — <what you ran and what it showed>" in prompt
    assert "if you ran it say which way\n   it came out" in prompt


def test_one_disproof_verdict_per_document_is_stated_where_it_is_met(tmp_path):
    """The known limit of Finding 2, on the three surfaces a reader meets.

    One ``disproof`` value answers for one claim. Until the field can hold a
    verdict per claim, the only thing standing between a record making three
    "cannot" claims and a reader who thinks all three were checked is that the
    limit is written down: in the prompt that asks for the markers, in the rule
    printed beside every document, and in the field's own docstring.
    """
    prompt = build_summary_prompt(iteration=5, end_reason="converged")
    assert 'ONE such line per "cannot" claim' in prompt
    assert "A claim you write no line\n   for is recorded as unanswered" in prompt

    store = _store(tmp_path)
    store.write(1, "headline")
    assert "A scope answers for one claim" in store.render_for_prompt()

    assert "answering for one of them" in (LessonScope.__doc__ or "")
