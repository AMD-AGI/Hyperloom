# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX prompt blocks read the corpus shape, they do not hardcode it.

Both prompt builders render these lines, so a literal here would be a second
source of truth for numbers that move when the corpus is re-measured. The tests
below pin that the rendering follows ``SharedState.agentx_corpus_shape``.
"""

from __future__ import annotations

import pytest

from hyperloom.common.perf_metric import AGENTX_KEEP_THRESHOLD_FLOOR_PCT
from hyperloom.orchestrator.prompts.agentx_context import corpus_lines, grading_lines

_SHAPE = {
    "corpus_loader": "semianalysis_cc_traces_weka_062126",
    "corpus_entries": 393,
    "duration_s": 3600.0,
    "isl": {"avg": 113814, "p50": 94821, "p90": 163328, "p99": 506158},
    "osl": {"avg": 806, "p50": 333, "p90": 1874, "p99": 6386},
    "prefix_cache_hit": 0.975,
}


def test_a_session_with_no_shape_renders_nothing():
    """A synthetic session must not get an AgentX block at all."""
    assert corpus_lines(None) == []
    assert corpus_lines({}) == []


def test_the_corpus_line_names_loader_entries_and_window():
    body = "\n".join(corpus_lines(_SHAPE))
    assert "- corpus: semianalysis_cc_traces_weka_062126, 393 traces, 3600s window" in body


def test_both_axes_render_the_slow_end_of_the_distribution():
    """p50/p90/p99, because a single scalar cannot describe this corpus."""
    body = "\n".join(corpus_lines(_SHAPE))
    assert "- input/req : p50 94,821   p90 163,328   p99 506,158" in body
    assert "- output/req: p50 333   p90 1,874   p99 6,386" in body


def test_the_prefix_cache_hit_is_rendered_as_a_percentage():
    assert "prefix cache hit ~97.5%" in "\n".join(corpus_lines(_SHAPE))


def test_the_numbers_follow_the_shape_rather_than_a_literal():
    """The defect this guards: numbers frozen into the prompt text.

    A re-measured corpus must move the rendered figures, or the prompt is
    describing a workload the session is not running.
    """
    measured = {
        **_SHAPE,
        "isl": {"p50": 120000, "p90": 180000, "p99": 520000},
        "osl": {"p50": 400, "p90": 2000, "p99": 6500},
        "prefix_cache_hit": 0.9758,
    }
    body = "\n".join(corpus_lines(measured))
    assert "p50 120,000   p90 180,000   p99 520,000" in body
    assert "p50 400   p90 2,000   p99 6,500" in body
    assert "~97.6%" in body
    assert "94,821" not in body


def test_an_axis_with_no_percentiles_says_so_instead_of_reading_zero():
    body = "\n".join(corpus_lines({"corpus_loader": "x", "isl": {}, "osl": None}))
    assert "- input/req : (not measured)" in body
    assert "- output/req: (not measured)" in body


def test_a_shape_missing_its_optional_fields_still_renders():
    """Only the loader is required; the rest are omitted, not defaulted."""
    body = "\n".join(corpus_lines({"corpus_loader": "only-a-loader"}))
    assert "- corpus: only-a-loader" in body
    assert "traces" not in body
    assert "window" not in body
    assert "prefix cache hit" not in body


def test_an_unnamed_corpus_falls_back_to_a_generic_label():
    assert "- corpus: agentic traces" in "\n".join(corpus_lines({"isl": {"p50": 1}}))


@pytest.mark.parametrize(
    ("field", "absent"),
    [
        ("corpus_entries", " traces"),
        ("duration_s", " window"),
        ("prefix_cache_hit", "prefix cache hit"),
    ],
)
def test_a_zero_is_treated_as_absent_not_as_a_measurement(field, absent):
    """Zero traces / a zero window / a zero hit rate are unset, not measured."""
    assert absent not in "\n".join(corpus_lines({**_SHAPE, field: 0}))


def test_the_grading_block_names_the_axis_and_the_three_verdicts():
    body = "\n".join(grading_lines())
    assert "E2E normalised interactivity P90" in body
    assert "SLOW tail" in body
    assert "REVERT" in body and "RECORDED" in body
    assert "not the objective" in body


def test_the_grading_block_takes_the_threshold_from_the_one_constant():
    """A literal here would drift from the floor the resolver actually applies."""
    assert f">={AGENTX_KEEP_THRESHOLD_FLOOR_PCT:.0f}%" in "\n".join(grading_lines())
