# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofs measured once and read thereafter, and the shipped reference card."""

from __future__ import annotations

import json

import pytest

from kernelforge.roofline_ceiling.device_profile import (
    PROFILE_SCHEMA_VERSION,
    DeviceIdentity,
    DeviceProfile,
    load_local,
    load_reference,
    local_path,
    store_local,
)
from kernelforge.roofline_ceiling.evidence import resolve_hardware
from kernelforge.roofline_ceiling.specs import (
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_EMPIRICAL,
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
        "peak_flops": {"bf16_mfma": 1.6e15},
        "bandwidth": {"hbm": 6.2e12, "mall": 8.7e12},
        "dispatch_floor_s": 1.55e-6,
    }
    base.update(overrides)
    return DeviceProfile(**base)


def test_partition_mode_is_part_of_the_identity_not_metadata():
    """Slicing a card into CPX changes what one slice can reach, so it is a different machine."""
    spx = _MI355X
    cpx = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X", compute_partition="CPX")

    assert spx.slug() != cpx.slug()
    assert not cpx.matches({"arch": "gfx950", "compute_partition": "SPX"})


def test_a_profile_may_claim_a_whole_architecture():
    assert _MI355X.matches({"arch": "gfx950"})


def test_an_undetermined_field_does_not_match_a_profile_that_states_one():
    """Guessing an undetected partition is the profile's is the silent substitution to avoid."""
    unknown = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X")

    assert not unknown.matches({"arch": "gfx950", "compute_partition": "SPX"})


def test_a_measurement_round_trips_through_the_local_cache(tmp_path):
    store_local(_profile(), _MI355X, tmp_path)

    restored = load_local(_MI355X, tmp_path)

    assert restored is not None
    assert restored.bandwidth["hbm"] == pytest.approx(6.2e12)
    assert restored.dispatch_floor_s == pytest.approx(1.55e-6)


def test_a_cache_written_for_another_configuration_is_not_read(tmp_path):
    store_local(_profile(), _MI355X, tmp_path)
    other = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X", compute_partition="CPX")

    assert load_local(other, tmp_path) is None


def test_an_unreadable_cache_entry_is_ignored_rather_than_raised(tmp_path):
    store_local(_profile(), _MI355X, tmp_path)
    local_path(_MI355X, tmp_path).write_text("{not json", encoding="utf-8")

    assert load_local(_MI355X, tmp_path) is None


def test_a_profile_from_a_future_schema_is_ignored(tmp_path):
    store_local(_profile(), _MI355X, tmp_path)
    path = local_path(_MI355X, tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = PROFILE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_local(_MI355X, tmp_path) is None


def test_a_profile_without_a_bandwidth_cannot_stand_in_for_a_measurement(tmp_path):
    store_local(_profile(bandwidth={"l2": 34.0e12}), _MI355X, tmp_path)

    assert load_local(_MI355X, tmp_path) is None


def test_the_shipped_mi355x_reference_is_matched_and_carries_measured_roofs():
    reference = load_reference(_MI355X)

    assert reference is not None
    assert reference.origin.startswith("shipped:")
    # Measured, so below the datasheet figures they stand in for.
    datasheet = arch_spec("gfx950")
    assert reference.bandwidth["hbm"] < datasheet.hbm_bw_bytes_per_s
    assert reference.peak_flops["mxfp4_scaled_mfma"] < datasheet.peak_flops["mxfp4_scaled_mfma"]


def test_the_shipped_reference_offers_the_cache_levels_the_datasheet_cannot():
    reference = load_reference(_MI355X)

    assert {"hbm", "mall", "l2", "lds"} <= set(reference.bandwidth)
    assert set(arch_spec("gfx950").bandwidth()) == {"hbm"}


def test_the_shipped_reference_records_where_every_figure_came_from():
    reference = load_reference(_MI355X)

    assert "rocprof-compute" in reference.source_by_figure["mxfp4_scaled_mfma"]
    assert "graph_replay" in reference.source_by_figure["dispatch_floor_s"]
    assert reference.measurement["rocm_version"]
    assert reference.measurement["measured_at"]


def test_the_shipped_reference_flags_the_bf16_figure_it_does_not_trust():
    """Measured bf16 is exactly half of fp16 though gfx950 runs both at one rate."""
    reference = load_reference(_MI355X)

    ratio = reference.peak_flops["fp16_mfma"] / reference.peak_flops["bf16_mfma"]
    assert ratio == pytest.approx(2.0, abs=0.02)
    assert any("bf16_mfma" in note and "tool artifact" in note for note in reference.notes)


def test_no_shipped_profile_claims_a_card_this_box_is_not():
    assert load_reference(DeviceIdentity(arch="gfx942", device_name="AMD Instinct MI300X")) is None


def test_a_local_measurement_outranks_the_shipped_reference():
    hardware = resolve_hardware(
        arch="gfx950",
        empirical_peaks={"bf16_mfma": 1.7e15},
        empirical_bandwidth={"hbm": 6.3e12},
        identity=_MI355X,
    )

    assert hardware.peak_source == PEAK_SOURCE_EMPIRICAL
    assert hardware.peak_flops == {"bf16_mfma": 1.7e15}


def test_the_shipped_reference_outranks_the_datasheet():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X)

    assert hardware.peak_source == PEAK_SOURCE_REFERENCE
    assert hardware.is_measured
    assert not hardware.is_empirical
    assert hardware.hbm_bw_bytes_per_s < arch_spec("gfx950").hbm_bw_bytes_per_s


def test_the_reference_records_which_card_it_was_matched_against():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X)

    matched = hardware.provenance["reference_profile"]["matched_device"]
    assert matched["device_name"] == "AMD Instinct MI355X"
    assert matched["compute_partition"] == "SPX"


def test_a_box_this_reference_does_not_describe_falls_to_the_datasheet():
    hardware = resolve_hardware(arch="gfx942", identity=DeviceIdentity(arch="gfx942"))

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET


def test_the_reference_tier_can_be_declined():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X, allow_reference=False)

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET


def test_a_locally_measured_dispatch_floor_wins_over_the_references():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X, dispatch_floor_s=2.0e-6)

    assert hardware.dispatch_floor_s == pytest.approx(2.0e-6)


def test_the_references_dispatch_floor_is_used_when_this_box_measured_none():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X, dispatch_floor_s=0.0)

    assert hardware.dispatch_floor_s > 0


def test_a_successful_measurement_is_cached_and_the_next_run_reads_it(tmp_path, monkeypatch):
    """The whole point of moving the roofs out of the per-operator cache."""
    from kernelforge.roofline_ceiling import evidence as evidence_module

    artifacts = tmp_path / "evidence"
    measured = {"peaks": {"bf16_mfma": 1.7e15}, "bw": {"hbm": 6.3e12}}
    calls = {"roof": 0}

    def fake_roofs(**_kwargs):
        calls["roof"] += 1
        return (
            dict(measured["peaks"]),
            dict(measured["bw"]),
            {"measured": True, "tool": "rocprof-compute", "column_by_instruction_path": {"bf16_mfma": "MFMABF16Flops"}},
        )

    monkeypatch.setattr(evidence_module, "collect_empirical_peaks", fake_roofs)
    monkeypatch.setattr(evidence_module, "describe_device", lambda _arch="": _MI355X)
    monkeypatch.setattr(evidence_module, "capture_kernel_trace", lambda **_k: {"captured": False, "detail": "x"})
    monkeypatch.setattr(
        evidence_module, "measure_dispatch_floor", lambda **_k: (1.6e-6, {"measured": True, "mode": "graph_replay"})
    )
    monkeypatch.setattr(
        evidence_module,
        "discover_scored_cases",
        lambda **_k: (["c0"], {"c0": 10.0}, []),
    )

    common = {
        "performance_command": ["true"],
        "workdir": tmp_path,
        "artifacts_dir": artifacts,
        "arch": "gfx950",
        "project_root": tmp_path / "state",
    }

    first, _ = evidence_module.collect_evidence(**common)
    second, _ = evidence_module.collect_evidence(**common)

    assert calls["roof"] == 1, "the second run must read the cached profile, not remeasure"
    assert first.hardware.peak_source == second.hardware.peak_source == PEAK_SOURCE_EMPIRICAL
    assert second.hardware.peak_flops == first.hardware.peak_flops
    assert any("reused this box's cached roofs" in note for note in second.notes)

    third, _ = evidence_module.collect_evidence(**common, remeasure_device=True)
    assert calls["roof"] == 2, "--remeasure-device must bypass the cache"
    assert third.hardware.peak_source == PEAK_SOURCE_EMPIRICAL
