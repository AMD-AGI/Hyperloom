# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Analysis Agent divides a round's work among its lanes; the loop reads only how many focuses there are."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.workflows.forge_loop.roles.analysis import AnalysisAgentService, AnalysisScope
from kernelforge.workflows.forge_loop.roles.analysis_bundle import read_lane_focuses
from kernelforge.workflows.forge_loop.roles.analysis_prompts import LANE_PARTITION_FILE, LANE_PARTITION_PROMPT
from kernelforge.workflows.forge_loop.roles.contracts import EvidenceRef

_REQUEST = {"analysis_trigger": "canonical_baseline"}


def _service(lanes: int, *, profiling_enabled: bool = True) -> AnalysisAgentService:
    return AnalysisAgentService(
        backend=None,
        config=SimpleNamespace(),
        timeout_sec=10,
        max_turns=10,
        scope=AnalysisScope(profiling_enabled=profiling_enabled, lanes=lanes),
    )


@pytest.mark.parametrize("profiling_enabled", [True, False], ids=["profiled", "static"])
def test_a_campaign_of_several_lanes_asks_the_analysis_to_divide_the_work(tmp_path, profiling_enabled):
    prompt = _service(3, profiling_enabled=profiling_enabled)._system_prompt(_REQUEST, tmp_path / "rocpc_profile.py")

    assert "Lane partition:" in prompt
    assert "up to 3 Implementer sessions side by side" in prompt
    assert f"write\n{LANE_PARTITION_FILE}" in prompt
    assert "`## Lane 1`, `## Lane 2`" in prompt
    # It is guided toward the two ways of dividing, and held to neither.
    assert "By optimization direction" in prompt and "By scored case" in prompt
    assert "A focus is where a lane starts, not a fence." in prompt


def test_a_campaign_of_one_lane_is_asked_for_no_division(tmp_path):
    """One lane has nothing to divide, and its Analysis prompt is the one a campaign without lanes always had."""
    one = _service(1)._system_prompt(_REQUEST, tmp_path / "rocpc_profile.py")
    three = _service(3)._system_prompt(_REQUEST, tmp_path / "rocpc_profile.py")

    assert "Lane partition" not in one
    assert LANE_PARTITION_FILE not in one
    assert three.replace(LANE_PARTITION_PROMPT.format(lanes=3, partition_file=LANE_PARTITION_FILE), "") == one


def test_the_focuses_are_the_sections_of_the_partition_in_lane_order(tmp_path):
    (tmp_path / LANE_PARTITION_FILE).write_text(
        "# How I would divide the work\n\n"
        "Preamble the loop does not read.\n\n"
        "## Lane 1\nFuse the two passes; cases m64 and m128.\n\n"
        "### lane 2 - large shapes\n\nSpecialize the dispatch for m4096.\nEvidence: report.md section 2.\n\n"
        "## Lane 3\n\n"
    )

    assert read_lane_focuses(tmp_path) == [
        "Fuse the two passes; cases m64 and m128.",
        "large shapes\nSpecialize the dispatch for m4096.\nEvidence: report.md section 2.",
    ]


def test_a_title_on_the_lanes_heading_line_is_part_of_its_focus(tmp_path):
    """An Analysis names a lane, and the cases it scores, on the heading line: the session it is for has to see that."""
    (tmp_path / LANE_PARTITION_FILE).write_text(
        "## Lane 1: decode regime, M <= 64 (scored: m1, m16, m64)\nOne pass over the weight.\n\n"
        "## Lane 2 \u2014 large M (scored: m4096)\n\n"
        "## Lane 3:\n"
    )

    assert read_lane_focuses(tmp_path) == [
        "decode regime, M <= 64 (scored: m1, m16, m64)\nOne pass over the weight.",
        "large M (scored: m4096)",
    ]


@pytest.mark.parametrize("content", [None, "", "No division is worth a session here.\n"])
def test_an_analysis_that_divided_nothing_gives_no_focuses(tmp_path, content):
    if content is not None:
        (tmp_path / LANE_PARTITION_FILE).write_text(content)

    assert read_lane_focuses(tmp_path) == []


def _loop_reading(context) -> IterationLoop:
    loop = object.__new__(IterationLoop)
    loop._active_analysis_context = context
    return loop


def test_the_loop_reads_the_focuses_of_the_analysis_evidence_in_force(tmp_path):
    """A KEEP leaves the evidence in force until the next Analysis, so the lanes keep their focuses in between."""
    bundle = tmp_path / "analysis" / "abc123" / "generation-001"
    bundle.mkdir(parents=True)
    (bundle / LANE_PARTITION_FILE).write_text("## Lane 1\nsmall cases\n## Lane 2\nlarge cases\n")
    refs = (
        EvidenceRef(kind="source_map", path=str(tmp_path / "source_map.md"), summary="map"),
        EvidenceRef(kind="analysis_bundle", path=str(bundle), summary="bundle"),
    )

    assert _loop_reading(SimpleNamespace(evidence_refs=refs))._lane_focuses() == ["small cases", "large cases"]
    assert _loop_reading(SimpleNamespace(evidence_refs=refs[:1]))._lane_focuses() == []
    assert _loop_reading(None)._lane_focuses() == []


def test_the_partition_is_published_with_the_bundles_other_artifacts():
    from kernelforge.workflows.forge_loop.roles.analysis_catalog import _GLOBAL_ARTIFACTS

    kind, _description, _information = _GLOBAL_ARTIFACTS[LANE_PARTITION_FILE]
    assert kind == "lane_partition"
    assert Path(LANE_PARTITION_FILE).suffix == ".md"
