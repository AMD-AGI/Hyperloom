# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Evidence collection: unit handling, the degrade ladder, and what it records."""

from __future__ import annotations

import pytest

from kernelforge.roofline_ceiling import evidence as evidence_module
from kernelforge.roofline_ceiling.evidence import (
    _read_roofline_csv,
    collect_empirical_peaks,
    correct_known_tool_artifacts,
    discover_scored_cases,
    measure_dispatch_floor,
    resolve_hardware,
)
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_DATASHEET, PEAK_SOURCE_EMPIRICAL

# One device row, in the shape rocprofiler-compute writes: a leading device-id
# column its own reader drops before taking the header. This spelling is the one
# that reports the low-precision matrix roofs as a single merged F6F4 column.
_ROOFLINE_CSV = (
    "devID,MFMABF16Flops,MFMAF8Flops,MFMA_FLOPs_F6F4,FP32Flops,HBMBw,L2Bw\n"
    "0,1686000,3567000,5663000,137000,5300,11900\n"
)

# A real header and row from an MI355X ``--roof-only`` run, trimmed to the
# columns this module reads. Two things differ from the merged spelling above
# and both have bitten: the id column is named ``device``, and the low-precision
# matrix roofs arrive as separate ``MFMAF4Flops`` / ``MFMAF6Flops`` columns with
# no ``MFMA_FLOPs_F6F4`` anywhere.
_REAL_MI355X_CSV = (
    "device,HBMBw,HBMBwLow,hbmBwHigh,MALLBw,L2Bw,L1Bw,LDSBw,FP8Flops,FP16Flops,"
    "BF16Flops,FP32Flops,FP64Flops,I8Ops,MFMAF4Flops,MFMAF6Flops,MFMAF8Flops,"
    "MFMAF16Flops,MFMABF16Flops,MFMAF32Flops,MFMAF64Flops,MFMAI8Ops\n"
    "0,6237.3418,6233.0127,6241.6709,8496.8027,34513.656,38008.023,54161.102,"
    "81253.758,38558.672,5764.7715,150419.81,76324.766,81175.656,9769632,8810305,"
    "2456336.2,1229004.5,616227.75,154634.66,77521.898,1228687.2\n"
)


def _collect(tmp_path, monkeypatch, csv_text: str):
    workload = tmp_path / "roofline"
    workload.mkdir(exist_ok=True)
    (workload / "roofline.csv").write_text(csv_text, encoding="utf-8")
    monkeypatch.setattr(evidence_module.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (0, ""))
    return collect_empirical_peaks(command=["true"], workdir=tmp_path, artifacts_dir=tmp_path)


def _write_roofline(tmp_path, text: str = _ROOFLINE_CSV):
    path = tmp_path / "roofline.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_roofline_columns_are_read_past_the_device_id_column(tmp_path):
    columns = _read_roofline_csv(_write_roofline(tmp_path))

    assert columns["MFMABF16Flops"] == pytest.approx(1686000.0)
    assert columns["HBMBw"] == pytest.approx(5300.0)
    assert "devID" not in columns


def test_empirical_peaks_are_scaled_out_of_the_giga_units_the_csv_uses(tmp_path, monkeypatch):
    """roofline.csv is GFLOP/s and GB/s; a missing 1e9 understates every roof."""
    peaks, bandwidth, provenance = _collect(tmp_path, monkeypatch, _ROOFLINE_CSV)

    assert peaks["bf16_mfma"] == pytest.approx(1.686e15)
    assert bandwidth["hbm"] == pytest.approx(5.3e12)
    assert provenance["measured"] is True


def test_every_memory_level_the_tool_measured_is_carried_not_only_hbm(tmp_path, monkeypatch):
    """A cache-resident working set rides a roof well above HBM; the analyst needs it."""
    _peaks, bandwidth, _prov = _collect(tmp_path, monkeypatch, _REAL_MI355X_CSV)

    assert bandwidth["hbm"] == pytest.approx(6.2373418e12)
    assert bandwidth["mall"] == pytest.approx(8.4968027e12)
    assert bandwidth["l2"] == pytest.approx(34.513656e12)
    assert bandwidth["lds"] == pytest.approx(54.161102e12)


def test_the_scaled_mxfp_paths_share_their_unscaled_twins_pipeline(tmp_path, monkeypatch):
    peaks, _bw, _prov = _collect(tmp_path, monkeypatch, _ROOFLINE_CSV)

    assert peaks["mxfp8_scaled_mfma"] == peaks["fp8_mfma"]
    assert peaks["mxfp4_scaled_mfma"] == peaks["fp4_mfma"]
    assert peaks["mxfp6_scaled_mfma"] == peaks["fp6_mfma"]


def test_a_real_mi355x_roofline_csv_yields_every_low_precision_roof(tmp_path, monkeypatch):
    """The split F4/F6 spelling: pinning the merged one alone left MXFP kernels roofless."""
    peaks, bandwidth, provenance = _collect(tmp_path, monkeypatch, _REAL_MI355X_CSV)

    assert peaks["fp4_mfma"] == pytest.approx(9.769632e15)
    assert peaks["fp6_mfma"] == pytest.approx(8.810305e15)
    assert peaks["mxfp4_scaled_mfma"] == pytest.approx(9.769632e15)
    assert peaks["mxfp8_scaled_mfma"] == pytest.approx(2.4563362e15)
    assert bandwidth["hbm"] == pytest.approx(6.2373418e12)
    assert provenance["measured"] is True


def test_the_column_each_roof_came_from_is_recorded(tmp_path, monkeypatch):
    merged = _collect(tmp_path, monkeypatch, _ROOFLINE_CSV)[2]
    split = _collect(tmp_path, monkeypatch, _REAL_MI355X_CSV)[2]

    assert merged["column_by_instruction_path"]["fp4_mfma"] == "MFMA_FLOPs_F6F4"
    assert split["column_by_instruction_path"]["fp4_mfma"] == "MFMAF4Flops"


def test_measured_roofs_sit_below_the_datasheet_peaks_they_stand_in_for(tmp_path, monkeypatch):
    """The whole reason --roof-only is the default: a datasheet roof is unreachable."""
    from kernelforge.roofline_ceiling.specs import arch_spec

    peaks, bandwidth, _prov = _collect(tmp_path, monkeypatch, _REAL_MI355X_CSV)
    datasheet = arch_spec("gfx950")

    assert bandwidth["hbm"] < datasheet.hbm_bw_bytes_per_s
    for path in ("bf16_mfma", "fp8_mfma", "fp4_mfma"):
        assert peaks[path] < datasheet.peak_flops[path]


def test_a_device_id_column_under_either_name_is_dropped(tmp_path):
    """Its header moved from devID to device; a parser keying on either breaks."""
    named_device = _read_roofline_csv(_write_roofline(tmp_path, _REAL_MI355X_CSV))

    assert "device" not in named_device
    assert named_device["HBMBw"] == pytest.approx(6237.3418)


def test_a_missing_profiler_degrades_with_a_reason_rather_than_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_module.shutil, "which", lambda _name: None)

    peaks, bandwidth, provenance = collect_empirical_peaks(command=["true"], workdir=tmp_path, artifacts_dir=tmp_path)

    assert peaks == {} and bandwidth == {}
    assert provenance["measured"] is False
    assert "no roofline-capable profiler" in provenance["detail"]


def test_a_profiler_that_writes_no_csv_degrades_with_its_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_module.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (2, "boom"))

    _peaks, _bw, provenance = collect_empirical_peaks(command=["true"], workdir=tmp_path, artifacts_dir=tmp_path)

    assert provenance["measured"] is False
    assert provenance["exit_code"] == 2
    assert "no roofline.csv" in provenance["detail"]


def test_measured_roofs_win_and_are_labelled_as_measured():
    hardware = resolve_hardware(
        arch="gfx950",
        empirical_peaks={"bf16_mfma": 1.686e15},
        empirical_bandwidth={"hbm": 5.3e12, "mall": 11.9e12},
        dispatch_floor_s=2.0e-6,
    )

    assert hardware.peak_source == PEAK_SOURCE_EMPIRICAL
    assert hardware.is_empirical
    assert hardware.peak_flops == {"bf16_mfma": 1.686e15}
    assert hardware.bandwidth["mall"] == pytest.approx(11.9e12)


def test_peaks_without_an_hbm_measurement_fall_all_the_way_back_rather_than_mixing():
    """Half measured and half datasheet makes two terms incomparable, silently."""
    hardware = resolve_hardware(
        arch="gfx950",
        empirical_peaks={"bf16_mfma": 1.686e15},
        empirical_bandwidth={"l2": 34.5e12},
    )

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET
    assert hardware.peak_flops["bf16_mfma"] == pytest.approx(2.5e15)
    assert hardware.bandwidth == {"hbm": pytest.approx(8.0e12)}
    # The measured L2 figure is dropped with the rest: keeping it would leave one
    # level measured and another datasheet inside a record labelled datasheet.
    assert "l2" not in hardware.bandwidth


def test_datasheet_fallback_records_where_its_numbers_came_from():
    hardware = resolve_hardware(arch="gfx942")

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET
    assert hardware.provenance["datasheet_source"]
    assert hardware.hbm_bw_bytes_per_s == pytest.approx(5.325e12)


def test_the_datasheet_path_offers_no_cache_roof_rather_than_an_invented_one():
    """The knowledge base gives Infinity Cache a latency and no bandwidth."""
    assert set(resolve_hardware(arch="gfx950").bandwidth) == {"hbm"}


def test_an_arch_with_neither_measurement_nor_datasheet_refuses_to_guess(monkeypatch):
    monkeypatch.setattr(evidence_module, "detect_arch", lambda: "")

    with pytest.raises(ValueError, match="no measured roofs and no datasheet peaks"):
        resolve_hardware(arch="gfx1100")


def test_marketing_names_resolve_to_the_same_arch_a_probe_reports():
    assert resolve_hardware(arch="MI355X").arch == "gfx950"


def test_scored_cases_come_from_the_driver_not_from_a_configuration(tmp_path, monkeypatch):
    output = "case_ms: decode-t1 0.012\ncase_ms: prefill-t16384 1.4\n"
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (0, output))

    scored, observed, notes = discover_scored_cases(command=["true"], workdir=tmp_path, artifacts_dir=tmp_path)

    assert scored == ["decode-t1", "prefill-t16384"]
    assert observed == {"decode-t1": 0.012, "prefill-t16384": 1.4}
    assert notes == []


def test_correctness_only_cases_get_no_ceiling(tmp_path, monkeypatch):
    """An unscored case is outside the objective, so a ceiling for it means nothing."""
    output = "case_ms: decode-t1 0.012\ncase_ms: shape-check 9.0 unscored\n"
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (0, output))

    scored, observed, notes = discover_scored_cases(command=["true"], workdir=tmp_path, artifacts_dir=tmp_path)

    assert scored == ["decode-t1"]
    assert "shape-check" not in observed
    assert any("unscored" in note for note in notes)


def test_a_failed_driver_run_says_its_case_list_may_be_short(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (1, "case_ms: decode-t1 0.012\n"))

    _scored, _observed, notes = discover_scored_cases(command=["false"], workdir=tmp_path, artifacts_dir=tmp_path)

    assert any("exited 1" in note for note in notes)


def test_a_driver_emitting_nothing_says_so_rather_than_returning_an_empty_success(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (0, "no timings here"))

    scored, _observed, notes = discover_scored_cases(command=["true"], workdir=tmp_path, artifacts_dir=tmp_path)

    assert scored == []
    assert any("no scored case_ms lines" in note for note in notes)


def test_a_dispatch_floor_probe_without_a_device_returns_zero_and_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evidence_module,
        "_run",
        lambda *a, **k: (0, evidence_module._PROBE_SENTINEL + '{"ok": false, "detail": "no device"}'),
    )

    seconds, provenance = measure_dispatch_floor(workdir=tmp_path)

    assert seconds == 0.0
    assert provenance["measured"] is False
    assert provenance["detail"] == "no device"


def test_a_dispatch_floor_probe_result_is_read_off_the_sentinel(tmp_path, monkeypatch):
    payload = '{"ok": true, "dispatch_floor_s": 3.1e-06, "rounds": 25}'
    monkeypatch.setattr(
        evidence_module,
        "_run",
        lambda *a, **k: (0, "chatter\n" + evidence_module._PROBE_SENTINEL + payload + "\ntrailing"),
    )

    seconds, provenance = measure_dispatch_floor(workdir=tmp_path)

    assert seconds == pytest.approx(3.1e-6)
    assert provenance["measured"] is True


def test_a_halved_bf16_matrix_roof_is_repaired_from_the_fp16_measurement():
    """The profiler reports bf16 at half rate on an arch that runs both at one rate.

    Left alone the bf16 ceiling comes out twice as loose, so the kernel reads as
    twice as close to its limit as it is.
    """
    peaks, corrections = correct_known_tool_artifacts(
        {"bf16_mfma": 616227.75e9, "fp16_mfma": 1229004.5e9, "fp8_mfma": 2456336.2e9},
        "gfx950",
    )

    assert peaks["bf16_mfma"] == pytest.approx(1229004.5e9)
    assert "bf16_mfma" in corrections
    assert "artifact" in corrections["bf16_mfma"]
    # Nothing else is touched.
    assert peaks["fp8_mfma"] == pytest.approx(2456336.2e9)


def test_a_bf16_roof_that_agrees_with_fp16_is_left_alone():
    peaks, corrections = correct_known_tool_artifacts(
        {"bf16_mfma": 1229004.5e9, "fp16_mfma": 1229004.5e9},
        "gfx950",
    )

    assert corrections == {}
    assert peaks["bf16_mfma"] == pytest.approx(1229004.5e9)


def test_a_bf16_roof_far_from_half_is_a_measurement_not_an_artifact():
    """A ratio outside the window is never guessed at, in either direction."""
    for bf16 in (1229004.5e9 / 4, 1229004.5e9 / 1.5):
        _peaks, corrections = correct_known_tool_artifacts(
            {"bf16_mfma": bf16, "fp16_mfma": 1229004.5e9},
            "gfx950",
        )
        assert corrections == {}


def test_no_artifact_is_corrected_on_an_arch_with_no_datasheet_to_judge_by():
    peaks, corrections = correct_known_tool_artifacts(
        {"bf16_mfma": 5e14, "fp16_mfma": 1e15},
        "gfx1151",
    )

    assert corrections == {}
    assert peaks["bf16_mfma"] == pytest.approx(5e14)
