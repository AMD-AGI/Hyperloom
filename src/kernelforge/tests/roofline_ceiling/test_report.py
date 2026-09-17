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


def _hardware(peak_source: str = PEAK_SOURCE_EMPIRICAL) -> Hardware:
    return Hardware(
        arch="gfx950",
        hbm_bw_bytes_per_s=8.0e12,
        peak_flops={"bf16_mfma": 2.0e15},
        peak_source=peak_source,
        dispatch_floor_s=2.0e-6,
    )


def _report(peak_source: str = PEAK_SOURCE_EMPIRICAL, case_ids=("c0",)):
    payload = {
        "cases": [
            {
                "case_id": case_id,
                "bound_note": "dominated by weight traffic",
                "stages": [
                    {
                        "name": "gemm",
                        "flops": 2.0e12,
                        "bytes": 8.0e10,
                        "instruction_path": "bf16_mfma",
                        "dispatch_count": 1,
                        "formula_flops": "2*M*N*K",
                        "formula_bytes": "M*K*2 + K*N*2",
                        "assumptions": ["weights counted once"],
                    }
                ],
            }
            for case_id in case_ids
        ],
        "confidence": "medium",
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

    assert read_report(path).ideal_ms() == original.ideal_ms()


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


def test_the_document_states_the_ideal_latency_and_the_formulas_behind_it():
    rendered = render_document(_report())

    assert "# Performance ceiling analysis" in rendered
    assert "`c0`" in rendered
    assert "2*M*N*K" in rendered
    assert "weights counted once" in rendered
    assert "dominated by weight traffic" in rendered


def test_the_document_says_when_its_peaks_are_only_a_datasheet():
    rendered = render_document(_report(PEAK_SOURCE_DATASHEET))

    assert "not an achievable target" in rendered


def test_the_advisory_block_frames_itself_as_advisory_and_not_a_gate():
    rendered = render_for_prompt(_report())

    assert "advisory" in rendered.lower()
    assert "KEEP is decided by measurement" in rendered
    assert "can be wrong" in rendered


def test_the_advisory_block_reports_headroom_against_a_caller_supplied_baseline():
    report = _report()
    ideal = report.cases[0].t_ideal_ms

    rendered = render_for_prompt(report, {"c0": ideal * 4.0})

    assert "Headroom" in rendered
    assert "4.00x" in rendered


def test_the_advisory_block_omits_headroom_when_the_caller_has_no_baseline():
    rendered = render_for_prompt(_report())

    assert "Headroom" not in rendered


def test_a_case_without_a_baseline_reads_unknown_rather_than_one_times():
    rendered = render_for_prompt(_report(case_ids=("c0", "c1")), {"c0": 10.0})

    assert "n/a" in rendered


def test_a_datasheet_advisory_warns_against_reading_the_ratio_as_efficiency():
    rendered = render_for_prompt(_report(PEAK_SOURCE_DATASHEET), {"c0": 10.0})

    assert "absolute lower bounds" in rendered


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
