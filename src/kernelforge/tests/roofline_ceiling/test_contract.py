# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""The derivation, and the refusals that keep a derived number honest."""

from __future__ import annotations

import pytest

from kernelforge.roofline_ceiling.contract import (
    BOUND_COMPUTE,
    BOUND_LATENCY,
    BOUND_MEMORY,
    BOUND_MIXED,
    CeilingContractError,
    Hardware,
    build_report,
    load_report,
)
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_DATASHEET, PEAK_SOURCE_EMPIRICAL


def _hardware(**overrides) -> Hardware:
    base = {
        "arch": "gfx950",
        "hbm_bw_bytes_per_s": 8.0e12,
        "peak_flops": {"bf16_mfma": 2.0e15, "fp32_valu": 1.0e14},
        "peak_source": PEAK_SOURCE_EMPIRICAL,
        "dispatch_floor_s": 2.0e-6,
    }
    base.update(overrides)
    return Hardware(**base)


def _stage(**overrides) -> dict:
    stage = {
        "name": "gemm",
        "flops": 2.0e12,
        "bytes": 8.0e9,
        "instruction_path": "bf16_mfma",
        "dispatch_count": 1,
        "formula_flops": "2*M*N*K",
        "formula_bytes": "M*K*2 + K*N*2 + M*N*2",
    }
    stage.update(overrides)
    return stage


def _payload(*stages, case_id: str = "c0", **case_overrides) -> dict:
    case = {"case_id": case_id, "stages": list(stages) or [_stage()]}
    case.update(case_overrides)
    return {"cases": [case], "confidence": "high"}


def _build(payload, **kwargs):
    defaults = {
        "canonical_id": "roofline-ceiling:op:gfx950",
        "hardware": _hardware(),
        "expected_case_ids": ["c0"],
    }
    defaults.update(kwargs)
    return build_report(payload, **defaults)


def test_ideal_latency_is_dispatch_floor_plus_the_larger_service_term():
    # compute 2e12/2e15 = 1 ms; memory 8e9/8e12 = 1 ms; latency 1 * 2 us.
    report = _build(_payload(_stage()))

    case = report.cases[0]
    assert case.t_ideal_ms == pytest.approx(1.0 + 0.002)
    assert case.timings[0].t_compute_s == pytest.approx(1.0e-3)
    assert case.timings[0].t_memory_s == pytest.approx(1.0e-3)


def test_serial_stages_are_summed_not_collapsed_into_one_max():
    """Merging first would claim one stage's compute hides the other's traffic."""
    compute_only = _stage(name="a", flops=2.0e12, bytes=1.0, dispatch_count=1)
    memory_only = _stage(name="b", flops=1.0, bytes=8.0e9, dispatch_count=1)

    report = _build(_payload(compute_only, memory_only))

    # Summed: 1 ms + 1 ms + two dispatch floors. A single max would have said 1 ms.
    assert report.cases[0].t_ideal_ms == pytest.approx(2.0 + 0.004)


def test_a_stage_needing_several_dispatches_pays_the_floor_for_each():
    report = _build(_payload(_stage(dispatch_count=4)))

    assert report.cases[0].timings[0].t_latency_s == pytest.approx(4 * 2.0e-6)


def test_extra_latency_is_added_on_top_of_the_dispatch_floor():
    report = _build(_payload(_stage(extra_latency_s=5.0e-6)))

    assert report.cases[0].timings[0].t_latency_s == pytest.approx(2.0e-6 + 5.0e-6)


@pytest.mark.parametrize(
    ("flops", "bytes_moved", "dispatches", "expected"),
    [
        (2.0e13, 8.0e9, 1, BOUND_COMPUTE),
        (2.0e12, 8.0e10, 1, BOUND_MEMORY),
        (2.0e9, 8.0e6, 5000, BOUND_LATENCY),
        (2.0e12, 8.0e9, 1, BOUND_MIXED),
    ],
)
def test_bound_names_the_term_that_dominates(flops, bytes_moved, dispatches, expected):
    report = _build(_payload(_stage(flops=flops, bytes=bytes_moved, dispatch_count=dispatches)))

    assert report.cases[0].bound == expected


def test_a_real_path_this_box_has_no_peak_for_is_reported_not_priced_at_zero():
    """Charging nothing for unmodellable arithmetic would report a free kernel."""
    report = _build(_payload(_stage(instruction_path="mxfp4_scaled_mfma")))

    case = report.cases[0]
    assert case.timings[0].t_compute_s == 0.0
    assert any("mxfp4_scaled_mfma" in issue for issue in case.issues)
    assert any("has no peak for it" in issue for issue in case.issues)
    assert any("compute term is missing" in issue for issue in case.issues)


def test_a_path_name_that_does_not_exist_is_named_as_the_mistake_it_is():
    """A typo and a genuine measurement gap need different fixes, so they read differently."""
    report = _build(_payload(_stage(instruction_path="bf16_mma")))

    assert any("not a canonical instruction path name" in issue for issue in report.cases[0].issues)


def test_a_ceiling_above_the_observed_latency_is_flagged_not_clamped():
    report = _build(_payload(_stage()), observed_ms={"c0": 0.5})

    case = report.cases[0]
    assert case.t_ideal_ms > 0.5
    assert any("exceeds the observed" in issue for issue in case.issues)


def test_a_ceiling_below_the_observed_latency_is_unremarkable():
    report = _build(_payload(_stage()), observed_ms={"c0": 40.0})

    assert report.cases[0].issues == ()
    assert report.cases[0].profiler_observed_ms == 40.0


def test_a_missing_scored_case_is_refused():
    with pytest.raises(CeilingContractError, match="no ceiling for scored case"):
        _build(_payload(_stage()), expected_case_ids=["c0", "c1"])


def test_a_case_the_driver_never_scored_is_refused():
    payload = _payload(_stage())
    payload["cases"].append({"case_id": "ghost", "stages": [_stage()]})

    with pytest.raises(CeilingContractError, match="never scored"):
        _build(payload)


def test_a_missing_case_is_named_before_an_extra_one():
    """Both are true at once; the one the caller has to act on is the missing case."""
    with pytest.raises(CeilingContractError, match="no ceiling for scored case"):
        _build(_payload(_stage(), case_id="ghost"))


def test_a_duplicated_case_is_refused():
    payload = _payload(_stage())
    payload["cases"].append({"case_id": "c0", "stages": [_stage()]})

    with pytest.raises(CeilingContractError, match="appears twice"):
        _build(payload)


def test_a_case_without_stages_is_refused():
    with pytest.raises(CeilingContractError, match="at least one stage"):
        _build({"cases": [{"case_id": "c0", "stages": []}], "confidence": "high"})


def test_a_stage_declaring_neither_work_nor_traffic_is_refused():
    with pytest.raises(CeilingContractError, match="neither work nor traffic"):
        _build(_payload(_stage(flops=0, bytes=0)))


def test_a_stage_without_a_dispatch_is_refused():
    with pytest.raises(CeilingContractError, match="at least one dispatch"):
        _build(_payload(_stage(dispatch_count=0)))


def test_a_missing_confidence_is_refused():
    payload = _payload(_stage())
    del payload["confidence"]

    with pytest.raises(CeilingContractError, match="confidence"):
        _build(payload)


def test_datasheet_peaks_add_the_caveat_that_says_so():
    report = _build(_payload(_stage()), hardware=_hardware(peak_source=PEAK_SOURCE_DATASHEET))

    assert any("not measured on this box" in caveat for caveat in report.caveats)
    assert any("absolute lower bound" in caveat for caveat in report.caveats)


def test_an_unmeasured_dispatch_floor_is_declared_rather_than_absorbed():
    report = _build(_payload(_stage()), hardware=_hardware(dispatch_floor_s=0.0))

    assert any("Dispatch floor was not measured" in caveat for caveat in report.caveats)


def test_cases_come_back_in_the_order_the_driver_scored_them():
    payload = {
        "cases": [
            {"case_id": "b", "stages": [_stage()]},
            {"case_id": "a", "stages": [_stage()]},
        ],
        "confidence": "medium",
    }

    report = _build(payload, expected_case_ids=["a", "b"])

    assert [case.case_id for case in report.cases] == ["a", "b"]


def test_a_published_report_round_trips():
    original = _build(_payload(_stage()), observed_ms={"c0": 40.0})

    restored = load_report(original.to_dict())

    assert restored.ideal_ms() == original.ideal_ms()
    assert restored.cases[0].bound == original.cases[0].bound
    assert restored.cases[0].stages[0].formula_flops == "2*M*N*K"
    assert restored.hardware.peak_source == PEAK_SOURCE_EMPIRICAL


def test_a_report_from_a_future_schema_is_refused_rather_than_guessed_at():
    payload = _build(_payload(_stage())).to_dict()
    payload["schema_version"] = 99

    with pytest.raises(CeilingContractError, match="unsupported ceiling schema"):
        load_report(payload)
