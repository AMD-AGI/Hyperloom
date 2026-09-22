# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Committed device profiles: what they claim, and the self-check on their figures."""

from __future__ import annotations

import json

import pytest

from kernelforge.roofline_ceiling import device_profile as profile_module
from kernelforge.roofline_ceiling.device_profile import (
    DeviceIdentity,
    DeviceProfile,
    load_reference,
    validate_profile,
)
from kernelforge.roofline_ceiling.evidence import resolve_hardware
from kernelforge.roofline_ceiling.specs import (
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_REFERENCE,
    arch_spec,
)

_MI355X = DeviceIdentity(
    arch="gfx950",
    device_name="AMD Instinct MI355X",
    compute_partition="SPX",
    memory_partition="NPS1",
)


def _profile(**overrides) -> DeviceProfile:
    base = {
        "peak_flops": {"bf16_mfma": 1.2e15, "fp16_mfma": 1.2e15},
        "bandwidth": {"hbm": 6.2e12, "mall": 8.7e12},
        "dispatch_floor_s": 1.55e-6,
    }
    base.update(overrides)
    return DeviceProfile(**base)


def test_partition_mode_is_part_of_the_identity_not_metadata():
    """Slicing a card into CPX changes what one slice can reach, so it is a different machine."""
    cpx = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X", compute_partition="CPX")

    assert _MI355X.slug() != cpx.slug()
    assert not cpx.matches({"arch": "gfx950", "compute_partition": "SPX"})


def test_a_profile_may_claim_a_whole_architecture():
    assert _MI355X.matches({"arch": "gfx950"})


def test_an_undetermined_field_does_not_match_a_profile_that_states_one():
    """Guessing an undetected partition is the profile's is the silent substitution to avoid."""
    unknown = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X")

    assert not unknown.matches({"arch": "gfx950", "compute_partition": "SPX"})


def test_no_shipped_profile_claims_a_card_this_box_is_not():
    assert load_reference(DeviceIdentity(arch="gfx942", device_name="AMD Instinct MI300X")) is None


# --- the shipped MI355X entry --------------------------------------------------


def test_the_shipped_mi355x_profile_is_matched_and_carries_measured_roofs():
    reference = load_reference(_MI355X)

    assert reference is not None
    assert reference.origin.startswith("shipped:")
    datasheet = arch_spec("gfx950")
    assert reference.bandwidth["hbm"] < datasheet.hbm_bw_bytes_per_s
    assert reference.peak_flops["mxfp4_scaled_mfma"] < datasheet.peak_flops["mxfp4_scaled_mfma"]


def test_the_shipped_profile_offers_the_cache_levels_the_datasheet_cannot():
    reference = load_reference(_MI355X)

    assert {"hbm", "mall", "l2", "lds"} <= set(reference.bandwidth)
    assert set(arch_spec("gfx950").bandwidth()) == {"hbm"}


def test_the_shipped_profile_records_where_every_figure_came_from():
    reference = load_reference(_MI355X)

    assert "rocprof-compute" in reference.source_by_figure["mxfp4_scaled_mfma"]
    assert "graph_replay" in reference.source_by_figure["dispatch_floor_s"]
    assert reference.measurement["rocm_version"]
    assert reference.measurement["measured_at"]


def test_the_shipped_profile_carries_a_bf16_roof_the_chip_can_reach():
    """The profiler halves bf16; gfx950 runs both MFMA paths at one rate."""
    reference = load_reference(_MI355X)

    assert reference.peak_flops["bf16_mfma"] == pytest.approx(reference.peak_flops["fp16_mfma"])
    assert "MFMABF16Flops" in reference.source_by_figure["bf16_mfma"]
    assert any("bf16_mfma" in note and "artifact" in note for note in reference.notes)


def test_every_shipped_profile_passes_its_own_self_check():
    """The guard is worthless if the entry it ships with would not survive it."""
    reference = load_reference(_MI355X)

    assert validate_profile(reference, "gfx950") == []


# --- the self-check ------------------------------------------------------------


def test_a_figure_above_the_vendor_peak_cannot_be_a_measurement():
    """A column read off by the wrong name, or a unit left unscaled."""
    problems = validate_profile(_profile(peak_flops={"bf16_mfma": 9.9e15}), "gfx950")

    assert any("above the gfx950 datasheet peak" in problem for problem in problems)


def test_a_bandwidth_above_the_vendor_peak_is_caught_too():
    problems = validate_profile(_profile(bandwidth={"hbm": 9.0e12}), "gfx950")

    assert any("hbm bandwidth" in problem and "above the" in problem for problem in problems)


def test_paths_the_datasheet_rates_together_must_stay_together():
    """The halving artifact, transcribed without its correction."""
    problems = validate_profile(
        _profile(peak_flops={"bf16_mfma": 0.6e15, "fp16_mfma": 1.2e15}),
        "gfx950",
    )

    assert any("runs both at one rate" in problem for problem in problems)


def test_a_sound_profile_raises_nothing():
    assert validate_profile(_profile(), "gfx950") == []


def test_rounding_does_not_trip_the_check():
    datasheet = arch_spec("gfx950").hbm_bw_bytes_per_s

    assert validate_profile(_profile(bandwidth={"hbm": datasheet * 1.01}), "gfx950") == []


def test_an_arch_with_no_datasheet_has_nothing_to_judge_against():
    assert validate_profile(_profile(), "gfx1151") == []


def test_a_profile_that_fails_the_check_is_refused_not_used(tmp_path, monkeypatch, caplog):
    """Falling to the datasheet reads attainment too low, which is the safe direction."""
    directory = tmp_path / "device_profiles"
    directory.mkdir()
    payload = {
        "schema_version": 1,
        "match": {"arch": "gfx950"},
        # Half of fp16 on a chip that runs both at one rate.
        "peak_flops": {"bf16_mfma": 0.6e15, "fp16_mfma": 1.2e15},
        "bandwidth": {"hbm": 6.2e12},
        "dispatch_floor_s": 1.5e-6,
    }
    (directory / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(profile_module, "resource_path", lambda _rel: directory)

    assert load_reference(DeviceIdentity(arch="gfx950")) is None
    assert "not self-consistent" in caplog.text


def test_a_profile_from_a_future_schema_is_ignored(tmp_path, monkeypatch):
    directory = tmp_path / "device_profiles"
    directory.mkdir()
    (directory / "future.json").write_text(
        json.dumps({"schema_version": 99, "match": {"arch": "gfx950"}, "peak_flops": {"bf16_mfma": 1.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_module, "resource_path", lambda _rel: directory)

    assert load_reference(DeviceIdentity(arch="gfx950")) is None


def test_a_profile_without_an_hbm_bandwidth_cannot_stand_in_for_a_measurement(tmp_path, monkeypatch):
    directory = tmp_path / "device_profiles"
    directory.mkdir()
    (directory / "partial.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "match": {"arch": "gfx950"},
                "peak_flops": {"bf16_mfma": 1.0e15},
                "bandwidth": {"l2": 34.0e12},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_module, "resource_path", lambda _rel: directory)

    assert load_reference(DeviceIdentity(arch="gfx950")) is None


# --- how a profile reaches the hardware record ---------------------------------


def test_a_profile_outranks_the_datasheet():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X)

    assert hardware.peak_source == PEAK_SOURCE_REFERENCE
    assert hardware.is_measured
    assert hardware.hbm_bw_bytes_per_s < arch_spec("gfx950").hbm_bw_bytes_per_s


def test_the_record_says_which_card_the_profile_was_matched_against():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X)

    matched = hardware.provenance["device_profile"]["matched_device"]
    assert matched["device_name"] == "AMD Instinct MI355X"
    assert matched["compute_partition"] == "SPX"


def test_a_machine_no_profile_describes_falls_to_the_datasheet():
    hardware = resolve_hardware(arch="gfx942", identity=DeviceIdentity(arch="gfx942"))

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET
