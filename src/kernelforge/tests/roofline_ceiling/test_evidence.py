# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Evidence collection: the roof lookup, the case set, and what it records."""

from __future__ import annotations

import pytest

from kernelforge.roofline_ceiling import evidence as evidence_module
from kernelforge.roofline_ceiling.device_profile import DeviceIdentity
from kernelforge.roofline_ceiling.evidence import (
    discover_scored_cases,
    resolve_hardware,
)
from kernelforge.roofline_ceiling.specs import PEAK_SOURCE_DATASHEET, PEAK_SOURCE_REFERENCE

_MI355X = DeviceIdentity(
    arch="gfx950",
    device_name="AMD Instinct MI355X",
    compute_partition="SPX",
    memory_partition="NPS1",
)


def test_a_committed_profile_answers_for_the_machine_it_claims():
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X)

    assert hardware.peak_source == PEAK_SOURCE_REFERENCE
    assert hardware.is_measured
    # Measured, so below the vendor peak it stands in for.
    assert hardware.hbm_bw_bytes_per_s < 8.0e12
    # And carrying the cache levels the datasheet has no figure for.
    assert {"mall", "l2", "l1", "lds"} <= set(hardware.bandwidth)


def test_a_profile_brings_its_own_dispatch_floor():
    """Nothing probes the box any more, so the launch cost rides on the profile."""
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X)

    assert hardware.dispatch_floor_s > 0


def test_a_machine_no_profile_covers_falls_to_the_datasheet_whole():
    unknown = DeviceIdentity(arch="gfx942", device_name="AMD Instinct MI300X")

    hardware = resolve_hardware(arch="gfx942", identity=unknown)

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET
    assert hardware.provenance["datasheet_source"]
    assert hardware.hbm_bw_bytes_per_s == pytest.approx(5.325e12)
    assert hardware.provenance["no_device_profile_for"]["device_name"] == "AMD Instinct MI300X"


def test_sources_are_never_blended():
    """Half measured and half datasheet leaves no field able to say which is which."""
    hardware = resolve_hardware(arch="gfx950", identity=_MI355X, allow_reference=False)

    assert hardware.peak_source == PEAK_SOURCE_DATASHEET
    # Not one measured cache level smuggled into a record labelled datasheet.
    assert set(hardware.bandwidth) == {"hbm"}


def test_the_datasheet_path_offers_no_cache_roof_rather_than_an_invented_one():
    """The knowledge base gives Infinity Cache a latency and no bandwidth."""
    assert set(resolve_hardware(arch="gfx950", allow_reference=False).bandwidth) == {"hbm"}


def test_the_datasheet_path_declares_it_has_no_launch_cost():
    """Zero is reported as a caveat downstream, never absorbed as "free"."""
    assert resolve_hardware(arch="gfx942", allow_reference=False).dispatch_floor_s == 0.0


def test_an_arch_with_neither_profile_nor_datasheet_refuses_to_guess(monkeypatch):
    monkeypatch.setattr(evidence_module, "detect_arch", lambda: "")

    with pytest.raises(ValueError, match="no device profile and no datasheet peaks"):
        resolve_hardware(arch="gfx1100", identity=DeviceIdentity(arch="gfx1100"))


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


def test_a_caller_that_already_benched_pays_for_no_discovery_run(tmp_path, monkeypatch):
    """A campaign has better numbers than one profiled pass, and already has them."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the driver was run to discover cases the caller already knew")

    monkeypatch.setattr(evidence_module, "discover_scored_cases", refuse)
    monkeypatch.setattr(evidence_module, "capture_kernel_trace", lambda **_k: {"captured": False, "detail": "stub"})

    bundle, scored = evidence_module.collect_evidence(
        performance_command=["true"],
        workdir=tmp_path,
        artifacts_dir=tmp_path / "ev",
        arch="gfx950",
        known_case_ids=["a", "b"],
        known_case_ms={"a": 1.0, "b": 2.0, "stale": 9.0},
    )

    assert scored == ["a", "b"]
    assert bundle.observed_ms == {"a": 1.0, "b": 2.0}
    assert bundle.observed_origin == evidence_module.OBSERVED_CAMPAIGN


def test_a_datasheet_run_says_the_target_will_be_out_of_reach(tmp_path, monkeypatch):
    """Attainment against vendor peaks reads far too low, so say so loudly."""
    monkeypatch.setattr(evidence_module, "capture_kernel_trace", lambda **_k: {"captured": True, "tool": "stub"})
    monkeypatch.setattr(evidence_module, "load_reference", lambda _identity: None)

    bundle, _scored = evidence_module.collect_evidence(
        performance_command=["true"],
        workdir=tmp_path,
        artifacts_dir=tmp_path / "ev",
        arch="gfx942",
        known_case_ids=["a"],
        known_case_ms={"a": 1.0},
    )

    assert bundle.hardware.peak_source == PEAK_SOURCE_DATASHEET
    assert any("no device profile covers this machine" in note for note in bundle.notes)
    assert any("attainment reads far below" in note for note in bundle.notes)
