# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""What the contract still enforces once the composition belongs to the analyst."""

from __future__ import annotations

import pytest

from kernelforge.roofline_ceiling.contract import (
    CeilingContractError,
    build_report,
    load_report,
    response_schema,
)
from kernelforge.roofline_ceiling.specs import (
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_MEASURED,
)

_ANALYSIS = """# Performance ceiling analysis

## Conclusion

- `c0`: 12.8 ms, memory-bound.

## Proof approach

Bytes 8e10 against the measured 6.24 TB/s HBM roof gives 12.8 ms, which exceeds
the 1.19 ms the same case needs at the bf16 MFMA roof.
"""


def _hardware(**overrides) -> dict:
    """The block the analyst reports after measuring the machine."""
    base = {
        "peak_source": PEAK_SOURCE_MEASURED,
        # Below the gfx950 datasheet on every path, as a measurement must be.
        "peak_flops": {"bf16_mfma": 1.23e15, "fp16_mfma": 1.23e15},
        "bandwidth": {"hbm": 6.24e12, "mall": 8.49e12, "l2": 34.5e12},
        "dispatch_floor_s": 3.0e-6,
        "method": "rocprof-compute 3.4.0 --roof-only, column MFMAF16Flops",
    }
    base.update(overrides)
    return base


def _payload(**overrides) -> dict:
    payload = {
        "hardware": _hardware(),
        "cases": [{"case_id": "c0", "t_ideal_ms": 12.8}],
        "analysis_md": _ANALYSIS,
    }
    payload.update(overrides)
    return payload


def _build(payload, **kwargs):
    defaults = {
        "canonical_id": "roofline-ceiling:op:gfx950",
        "arch": "gfx950",
        "expected_case_ids": ["c0"],
    }
    defaults.update(kwargs)
    return build_report(payload, **defaults)


def test_a_well_formed_answer_is_published_as_given():
    report = _build(_payload())

    assert report.ideal_ms() == {"c0": 12.8}
    assert report.mean_ideal_ms() == pytest.approx(12.8)
    assert report.peak_source == PEAK_SOURCE_MEASURED  # kept in memory, for the document
    assert report.analysis_md.startswith("# Performance ceiling analysis")


def test_the_derivation_is_required_because_nothing_else_records_the_reasoning():
    with pytest.raises(CeilingContractError, match="analysis_md"):
        _build(_payload(analysis_md="   "))


def test_a_case_the_derivation_never_mentions_is_flagged_as_unaudited():
    """A number with no reasoning behind it is exactly the one nobody can check."""
    payload = _payload(
        cases=[
            {"case_id": "c0", "t_ideal_ms": 12.8},
            {"case_id": "c1", "t_ideal_ms": 1.0},
        ]
    )

    report = _build(payload, expected_case_ids=["c0", "c1"])

    assert report.case("c0").issues == ()
    assert any("unaudited" in issue for issue in report.case("c1").issues)


@pytest.mark.parametrize("value", [None, "fast", float("inf"), float("nan"), 0, -1.0])
def test_a_latency_that_is_not_a_positive_number_is_refused(value):
    with pytest.raises(CeilingContractError, match="t_ideal_ms"):
        _build(_payload(cases=[{"case_id": "c0", "t_ideal_ms": value}]))


def test_a_ceiling_above_the_observed_latency_is_flagged_not_clamped():
    """The one check that survives dropping the work model, and the sharpest one."""
    report = _build(_payload(), observed_ms={"c0": 5.0})

    case = report.cases[0]
    assert case.t_ideal_ms == 12.8
    assert any("exceeds the observed" in issue for issue in case.issues)


def test_a_ceiling_below_the_observed_latency_is_unremarkable():
    report = _build(_payload(), observed_ms={"c0": 40.0})

    assert report.cases[0].issues == ()


def test_a_missing_scored_case_is_refused():
    with pytest.raises(CeilingContractError, match="no ceiling for scored case"):
        _build(_payload(), expected_case_ids=["c0", "c1"])


def test_a_case_the_driver_never_scored_is_refused():
    payload = _payload(
        cases=[
            {"case_id": "c0", "t_ideal_ms": 1.0},
            {"case_id": "ghost", "t_ideal_ms": 1.0},
        ]
    )

    with pytest.raises(CeilingContractError, match="never scored"):
        _build(payload)


def test_a_duplicated_case_is_refused():
    payload = _payload(
        cases=[
            {"case_id": "c0", "t_ideal_ms": 1.0},
            {"case_id": "c0", "t_ideal_ms": 2.0},
        ]
    )

    with pytest.raises(CeilingContractError, match="appears twice"):
        _build(payload)


def test_an_empty_cases_array_is_refused():
    with pytest.raises(CeilingContractError, match="non-empty 'cases'"):
        _build(_payload(cases=[]))


def test_where_the_roofs_came_from_reaches_the_document_not_the_file():
    recalled = _build(_payload(hardware=_hardware(peak_source=PEAK_SOURCE_DATASHEET)))

    assert recalled.peak_source == PEAK_SOURCE_DATASHEET
    assert "peak_source" not in recalled.to_dict()


# --- the roofs the analyst reports ---------------------------------------------


def test_a_roof_above_the_vendor_peak_cannot_be_a_measurement():
    """A column read by the wrong name, or a unit left unscaled."""
    with pytest.raises(CeilingContractError, match="above the published gfx950 peak"):
        _build(_payload(hardware=_hardware(peak_flops={"bf16_mfma": 9.9e15})))


def test_a_bandwidth_above_the_vendor_peak_is_refused_too():
    with pytest.raises(CeilingContractError, match="above the published gfx950 peak"):
        _build(_payload(hardware=_hardware(bandwidth={"hbm": 9.0e12})))


def test_the_bf16_halving_artifact_is_refused_rather_than_shipped():
    """Left in, every bf16 ceiling is twice as loose as it should be."""
    with pytest.raises(CeilingContractError, match="runs both at one rate"):
        _build(_payload(hardware=_hardware(peak_flops={"bf16_mfma": 0.6e15, "fp16_mfma": 1.2e15})))


def test_an_instruction_path_nobody_can_check_is_refused():
    with pytest.raises(CeilingContractError, match="not an instruction path this build knows"):
        _build(_payload(hardware=_hardware(peak_flops={"tf32_mfma": 1.0e14})))


def test_a_memory_level_outside_the_vocabulary_is_refused():
    with pytest.raises(CeilingContractError, match="memory level"):
        _build(_payload(hardware=_hardware(bandwidth={"hbm": 6.0e12, "scalar_cache": 1.0e12})))


def test_a_report_with_no_hbm_roof_is_refused():
    with pytest.raises(CeilingContractError, match="needs an 'hbm' figure"):
        _build(_payload(hardware=_hardware(bandwidth={"l2": 34.0e12})))


def test_a_response_without_a_hardware_block_is_refused():
    payload = _payload()
    del payload["hardware"]

    with pytest.raises(CeilingContractError, match="hardware"):
        _build(payload)


def test_an_invented_peak_source_is_refused():
    with pytest.raises(CeilingContractError, match="peak_source"):
        _build(_payload(hardware=_hardware(peak_source="vibes")))


def test_how_the_roofs_were_obtained_is_kept_on_the_record():
    report = _build(_payload())

    assert "MFMAF16Flops" in report.hardware.provenance["method"]


def test_cases_come_back_in_the_order_the_driver_scored_them():
    payload = _payload(
        cases=[
            {"case_id": "b", "t_ideal_ms": 1.0},
            {"case_id": "a", "t_ideal_ms": 2.0},
        ],
        analysis_md="covers a and b",
    )

    report = _build(payload, expected_case_ids=["a", "b"])

    assert [case.case_id for case in report.cases] == ["a", "b"]


def test_every_measured_memory_level_is_carried_on_the_record():
    """The analyst may bound a case against Infinity Cache, so the report says what it had."""
    report = _build(_payload())

    assert set(report.hardware.bandwidth) == {"hbm", "mall", "l2"}
    assert report.hardware.hbm_bw_bytes_per_s == pytest.approx(6.24e12)
    # Kept for the document, never published.
    assert "hardware" not in report.to_dict()


def test_a_published_report_round_trips():
    original = _build(_payload(), observed_ms={"c0": 40.0})

    restored = load_report(original.to_dict())

    assert restored.ideal_ms() == original.ideal_ms()
    assert restored.mean_ideal_ms() == pytest.approx(original.mean_ideal_ms())


def test_the_published_file_carries_only_the_latencies_their_mean_and_their_origin():
    """Everything a consumer needs to compute attainment, and nothing to weigh it by."""
    published = _build(_payload()).to_dict()

    assert set(published) == {"mean_ideal_ms", "cases"}
    assert published["cases"] == {"c0": 12.8}


def test_the_mean_is_equal_weight_so_it_matches_how_the_suite_is_scored():
    payload = _payload(
        cases=[{"case_id": "a", "t_ideal_ms": 1.0}, {"case_id": "b", "t_ideal_ms": 9.0}],
        analysis_md="covers a and b",
    )

    report = _build(payload, expected_case_ids=["a", "b"])

    assert report.mean_ideal_ms() == pytest.approx(5.0)
