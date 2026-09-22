# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Publication, the cache identity, and the advisory block consumers inject."""

from __future__ import annotations

import json

from kernelforge.roofline_ceiling.contract import Hardware, build_report
from kernelforge.roofline_ceiling.report import (
    DOCUMENT_FILENAME,
    REPORT_FILENAME,
    cache_key,
    cache_path,
    case_set_hash,
    publish,
    read_cached,
    read_report,
    render_document,
    render_for_prompt,
    store_in_cache,
)
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_DATASHEET, PEAK_SOURCE_EMPIRICAL

_ANALYSIS = """# Performance ceiling analysis

## Conclusion

- `c0`: 12.8 ms, memory-bound.

## Proof approach

Bytes `M*K*2 + K*N*2` = 8e10 against the measured 6.24 TB/s HBM roof gives
12.8 ms; the same case needs only 1.19 ms at the bf16 MFMA roof. Expert weights
are counted once, for the experts this case actually activates.
"""


def _hardware(peak_source: str = PEAK_SOURCE_EMPIRICAL) -> Hardware:
    return Hardware(
        arch="gfx950",
        peak_flops={"bf16_mfma": 1.686e15},
        bandwidth={"hbm": 6.24e12, "mall": 8.49e12},
        peak_source=peak_source,
        dispatch_floor_s=3.0e-6,
    )


def _report(peak_source: str = PEAK_SOURCE_EMPIRICAL, case_ids=("c0",)):
    payload = {
        "cases": [{"case_id": case_id, "t_ideal_ms": 12.8, "bound": "memory"} for case_id in case_ids],
        "confidence": "medium",
        "analysis_md": _ANALYSIS + "\n" + "\n".join(f"Case `{case_id}` covered." for case_id in case_ids),
        "caveats": ["trace captured on two of three shapes"],
    }
    return build_report(
        payload,
        canonical_id="roofline-ceiling:op:gfx950",
        hardware=_hardware(peak_source),
        expected_case_ids=list(case_ids),
    )


def test_publish_writes_both_the_contract_and_the_document(tmp_path):
    path = publish(_report(), tmp_path)

    assert path == tmp_path / REPORT_FILENAME
    assert (tmp_path / DOCUMENT_FILENAME).is_file()
    assert json.loads(path.read_text())["cases"][0]["case_id"] == "c0"


def test_a_published_report_is_readable_by_a_consumer(tmp_path):
    original = _report()
    path = publish(original, tmp_path)

    restored = read_report(path)
    assert restored.ideal_ms() == original.ideal_ms()
    assert restored.analysis_md == original.analysis_md


def test_the_cache_key_separates_a_measured_ceiling_from_a_datasheet_one():
    """Serving a datasheet answer to a caller who asked for a measured one is the
    silent degrade the whole module is built to prevent."""
    common = {"canonical_id": "roofline-ceiling:op:gfx950", "case_ids": ["c0"], "arch": "gfx950"}

    assert cache_key(**common, peak_source=PEAK_SOURCE_EMPIRICAL) != cache_key(
        **common, peak_source=PEAK_SOURCE_DATASHEET
    )


def test_the_cache_key_moves_when_the_scored_case_set_does():
    common = {"canonical_id": "roofline-ceiling:op:gfx950", "arch": "gfx950", "peak_source": PEAK_SOURCE_EMPIRICAL}

    assert cache_key(**common, case_ids=["c0"]) != cache_key(**common, case_ids=["c0", "c1"])


def test_the_case_set_hash_ignores_the_order_the_driver_happened_to_print():
    assert case_set_hash(["b", "a"]) == case_set_hash(["a", "b"])


def test_a_cached_report_round_trips(tmp_path):
    store_in_cache(_report(), "key0", tmp_path)

    restored = read_cached("key0", tmp_path)

    assert restored is not None
    assert restored.ideal_ms() == _report().ideal_ms()


def test_a_corrupt_cache_entry_is_ignored_rather_than_raised(tmp_path):
    store_in_cache(_report(), "key0", tmp_path)
    cache_path("key0", tmp_path).write_text("{not json", encoding="utf-8")

    assert read_cached("key0", tmp_path) is None


def test_an_absent_cache_entry_is_simply_absent(tmp_path):
    assert read_cached("never-written", tmp_path) is None


def test_the_document_leads_with_the_answer_and_carries_the_derivation_verbatim():
    rendered = render_document(_report())

    assert "## Conclusion" in rendered
    assert "`c0`" in rendered
    # The analyst's own document, reproduced rather than summarized.
    assert "M*K*2 + K*N*2" in rendered
    assert "Expert weights" in rendered


def test_the_document_states_every_figure_the_estimate_was_taken_against():
    """A reader recomputing the numbers needs the roofs, not just the answer."""
    rendered = render_document(_report())

    assert "Bandwidth (hbm)" in rendered
    assert "Bandwidth (mall)" in rendered
    assert "Peak (bf16_mfma)" in rendered
    assert "Dispatch floor" in rendered


def test_the_document_says_when_its_peaks_are_only_a_datasheet():
    rendered = render_document(_report(PEAK_SOURCE_DATASHEET))

    assert "not an achievable target" in rendered


def test_validator_findings_reach_the_document_rather_than_only_the_json():
    report = build_report(
        {
            "cases": [{"case_id": "c0", "t_ideal_ms": 99.0, "bound": "memory"}],
            "confidence": "low",
            "analysis_md": "case `c0` is estimated at 99 ms",
        },
        canonical_id="roofline-ceiling:op:gfx950",
        hardware=_hardware(),
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


def test_a_datasheet_ceiling_warns_that_attainment_reads_too_low():
    rendered = render_for_prompt(_report(PEAK_SOURCE_DATASHEET), {"c0": 20.0})

    assert "absolute lower bound that no implementation reaches" in rendered


def test_an_empty_report_renders_nothing_to_inject():
    report = _report()
    empty = type(report)(
        schema_version=report.schema_version,
        canonical_id=report.canonical_id,
        hardware=report.hardware,
        cases=(),
        confidence=report.confidence,
    )

    assert render_for_prompt(empty) == ""
