# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Reading the published file, and the attainment block a campaign injects."""

from __future__ import annotations

import json

from kernelforge.roofline_ceiling.contract import load_report
from kernelforge.roofline_ceiling.report import read_report, render_for_prompt


def _report(cases=(("c0", 12.8),)):
    return load_report({"cases": {case_id: ideal for case_id, ideal in cases}})


def test_a_file_the_analyst_wrote_is_read_back(tmp_path):
    path = tmp_path / "performance_ceiling.json"
    path.write_text(json.dumps({"cases": {"c0": 12.8}, "mean_ideal_ms": 12.8}), encoding="utf-8")

    restored = read_report(path)

    assert restored.ideal_ms() == {"c0": 12.8}


def test_the_roofline_block_says_the_ceiling_is_an_estimate_and_gates_no_keep():
    rendered = render_for_prompt(_report())

    assert "estimate, not a measurement" in rendered
    assert "decides no KEEP" in rendered


def test_the_roofline_block_states_attainment_against_the_measured_latency():
    rendered = render_for_prompt(_report(), {"c0": 12.8 * 4.0})

    # A kernel at four times its ceiling's latency is delivering a quarter of it.
    assert "25.0%" in rendered
    assert "4.00x" in rendered


def test_the_roofline_block_names_the_target_and_how_many_cases_are_short():
    rendered = render_for_prompt(
        _report((("c0", 9.0), ("c1", 5.0))),
        {"c0": 10.0, "c1": 10.0},
        target=0.86,
    )

    assert "Target: **86%**" in rendered
    assert "1 case(s) below it" in rendered


def test_the_worst_case_is_listed_first_because_that_is_where_the_effort_goes():
    rendered = render_for_prompt(_report((("c0", 9.0), ("c1", 4.0))), {"c0": 10.0, "c1": 10.0})

    assert rendered.index("`c1`") < rendered.index("`c0`")


def test_a_case_the_campaign_never_measured_is_named_with_its_reason():
    rendered = render_for_prompt(_report((("c0", 9.0), ("c1", 4.0))), {"c0": 20.0})

    assert "no measured latency for this case" in rendered
    assert "`c1`" in rendered


def test_a_ceiling_below_the_measured_latency_is_reported_not_scored():
    """Attainment above one is the estimate contradicting itself."""
    rendered = render_for_prompt(_report(), {"c0": 6.4}, target=0.86)

    assert "understates the minimum legal work" in rendered
    assert "Campaign attainment" not in rendered


def test_without_measured_latencies_the_block_still_states_the_ceilings():
    rendered = render_for_prompt(_report((("c0", 9.0), ("c1", 4.0))))

    assert "`c0`" in rendered and "`c1`" in rendered
    assert "Campaign attainment" not in rendered


def test_an_empty_report_renders_nothing_to_inject():
    assert render_for_prompt(load_report({"cases": {"a": 1.0}}).__class__(cases=())) == ""
