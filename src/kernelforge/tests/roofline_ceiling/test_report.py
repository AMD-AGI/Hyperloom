# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Publication, the cache identity, and the advisory block consumers inject."""

from __future__ import annotations

import json

from kernelforge.roofline_ceiling.contract import build_report
from kernelforge.roofline_ceiling.report import (
    DOCUMENT_FILENAME,
    REPORT_FILENAME,
    publish,
    read_report,
    render_document,
    render_for_prompt,
)
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_DATASHEET, PEAK_SOURCE_MEASURED

_HARDWARE = {
    "peak_source": PEAK_SOURCE_MEASURED,
    "peak_flops": {"bf16_mfma": 1.23e15, "fp16_mfma": 1.23e15},
    "bandwidth": {"hbm": 6.24e12, "mall": 8.49e12},
    "dispatch_floor_s": 3.0e-6,
    "method": "rocprof-compute --roof-only",
}

_ANALYSIS = """# Performance ceiling analysis

## Conclusion

- `c0`: 12.8 ms, memory-bound.

## Proof approach

Bytes `M*K*2 + K*N*2` = 8e10 against the measured 6.24 TB/s HBM roof gives
12.8 ms; the same case needs only 1.19 ms at the bf16 MFMA roof. Expert weights
are counted once, for the experts this case actually activates.
"""


def _report(peak_source: str = PEAK_SOURCE_MEASURED, case_ids=("c0",)):
    payload = {
        "hardware": {**_HARDWARE, "peak_source": peak_source},
        "cases": [{"case_id": case_id, "t_ideal_ms": 12.8} for case_id in case_ids],
        "analysis_md": _ANALYSIS + "\n" + "\n".join(f"Case `{case_id}` covered." for case_id in case_ids),
    }
    return build_report(
        payload,
        canonical_id="roofline-ceiling:op:gfx950",
        arch="gfx950",
        expected_case_ids=list(case_ids),
    )


def test_publish_writes_both_the_contract_and_the_document(tmp_path):
    path = publish(_report(), tmp_path)

    assert path == tmp_path / REPORT_FILENAME
    assert (tmp_path / DOCUMENT_FILENAME).is_file()
    assert json.loads(path.read_text())["cases"] == {"c0": 12.8}


def test_a_published_report_is_readable_by_a_consumer(tmp_path):
    original = _report()
    path = publish(original, tmp_path)

    restored = read_report(path)
    assert restored.ideal_ms() == original.ideal_ms()
    assert restored.mean_ideal_ms() == original.mean_ideal_ms()
    # The derivation lives in the document beside it, not in the file.
    assert restored.analysis_md == ""


def test_the_document_leads_with_the_answer_and_carries_the_derivation_verbatim():
    rendered = render_document(_report())

    assert "## Conclusion" in rendered
    assert "`c0`" in rendered
    # The analyst's own document, reproduced rather than summarized.
    assert "M*K*2 + K*N*2" in rendered
    assert "Expert weights" in rendered


def test_the_document_states_the_mean_and_where_the_roofs_came_from():
    """The published file has the numbers; this is where a reader weighs them."""
    rendered = render_document(_report())

    assert "mean (equal weight)" in rendered
    assert "measured_on_this_box" in rendered


def test_the_document_says_when_its_peaks_were_only_recalled():
    rendered = render_document(_report(PEAK_SOURCE_DATASHEET))

    assert "measured on no card" in rendered


def test_validator_findings_reach_the_document_rather_than_only_the_json():
    report = build_report(
        {
            "hardware": _HARDWARE,
            "cases": [{"case_id": "c0", "t_ideal_ms": 99.0}],
            "analysis_md": "case `c0` is estimated at 99 ms",
        },
        canonical_id="roofline-ceiling:op:gfx950",
        arch="gfx950",
        expected_case_ids=["c0"],
        observed_ms={"c0": 10.0},
    )

    rendered = render_document(report)

    assert "## Findings" in rendered
    assert "exceeds the observed" in rendered


def test_the_roofline_block_says_the_ceiling_is_an_estimate_and_gates_no_keep():
    rendered = render_for_prompt(_report())

    assert "estimate, not a measurement" in rendered
    assert "decides no KEEP" in rendered


def test_the_roofline_block_states_attainment_against_the_measured_latency():
    report = _report()
    ideal = report.cases[0].t_ideal_ms

    rendered = render_for_prompt(report, {"c0": ideal * 4.0})

    # A kernel at four times its ceiling's latency is delivering a quarter of it.
    assert "25.0%" in rendered
    assert "4.00x" in rendered


def test_the_roofline_block_names_the_target_and_how_many_cases_are_short():
    report = _report(case_ids=("c0", "c1"))
    ideal = report.cases[0].t_ideal_ms

    rendered = render_for_prompt(
        report,
        {"c0": ideal / 0.9, "c1": ideal / 0.5},
        target=0.86,
    )

    assert "Target: **86%**" in rendered
    assert "1 case(s) below it" in rendered


def test_the_worst_case_is_listed_first_because_that_is_where_the_effort_goes():
    report = _report(case_ids=("c0", "c1"))
    ideal = report.cases[0].t_ideal_ms

    rendered = render_for_prompt(report, {"c0": ideal / 0.9, "c1": ideal / 0.4})

    assert rendered.index("`c1`") < rendered.index("`c0`")


def test_a_case_the_campaign_never_measured_is_named_with_its_reason():
    rendered = render_for_prompt(_report(case_ids=("c0", "c1")), {"c0": 20.0})

    assert "no measured latency for this case" in rendered
    assert "`c1`" in rendered


def test_a_ceiling_below_the_measured_latency_is_reported_not_scored():
    """Attainment above one is the estimate contradicting itself."""
    report = _report()
    ideal = report.cases[0].t_ideal_ms

    rendered = render_for_prompt(report, {"c0": ideal / 2.0}, target=0.86)

    assert "understates the minimum legal work" in rendered
    # No attainment figure was produced, so no campaign standing is claimed.
    assert "Campaign attainment" not in rendered


def test_an_empty_report_renders_nothing_to_inject():
    report = _report()
    empty = type(report)(
        canonical_id=report.canonical_id,
        peak_source=report.peak_source,
        cases=(),
    )

    assert render_for_prompt(empty) == ""
