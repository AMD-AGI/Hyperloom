# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Evidence collection: the machine's identity, the case set, and what it records."""

from __future__ import annotations

from kernelforge.roofline_ceiling import evidence as evidence_module
from kernelforge.roofline_ceiling.device_profile import DeviceIdentity
from kernelforge.roofline_ceiling.evidence import discover_scored_cases, resolve_identity


def test_the_machine_is_named_so_the_analyst_measures_the_right_one(monkeypatch):
    described = DeviceIdentity(
        arch="gfx950",
        device_name="AMD Instinct MI355X",
        compute_partition="SPX",
        memory_partition="NPS1",
    )
    monkeypatch.setattr(evidence_module, "describe_device", lambda _arch="": described)

    identity = resolve_identity("gfx950")

    assert identity.slug() == "gfx950-mi355x-spx-nps1"


def test_marketing_names_resolve_to_the_arch_a_probe_reports(monkeypatch):
    monkeypatch.setattr(
        evidence_module,
        "describe_device",
        lambda arch="": DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X"),
    )

    assert resolve_identity("MI355X").arch == "gfx950"


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
    """A campaign has better numbers than a single run, and already has them."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the driver was run to discover cases the caller already knew")

    monkeypatch.setattr(evidence_module, "discover_scored_cases", refuse)
    monkeypatch.setattr(evidence_module, "capture_kernel_trace", lambda **_k: {"captured": False, "detail": "stub"})
    monkeypatch.setattr(
        evidence_module,
        "resolve_identity",
        lambda _arch="": DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X"),
    )

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


def test_a_discovered_latency_is_labelled_as_the_single_unprofiled_run_it_is(tmp_path, monkeypatch):
    """The trace run is the profiled one, and its timings are discarded; discovery runs bare."""
    monkeypatch.setattr(evidence_module, "_run", lambda *a, **k: (0, "case_ms: a 1.0\n"))
    monkeypatch.setattr(evidence_module, "capture_kernel_trace", lambda **_k: {"captured": False, "detail": "stub"})
    monkeypatch.setattr(
        evidence_module,
        "resolve_identity",
        lambda _arch="": DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X"),
    )

    bundle, scored = evidence_module.collect_evidence(
        performance_command=["true"],
        workdir=tmp_path,
        artifacts_dir=tmp_path / "ev",
        arch="gfx950",
    )

    assert scored == ["a"]
    assert bundle.observed_origin == evidence_module.OBSERVED_SINGLE_RUN


def test_the_bundle_carries_no_roofs_because_the_analyst_measures_them(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_module, "capture_kernel_trace", lambda **_k: {"captured": True, "tool": "stub"})
    monkeypatch.setattr(
        evidence_module,
        "resolve_identity",
        lambda _arch="": DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X"),
    )

    bundle, _scored = evidence_module.collect_evidence(
        performance_command=["true"],
        workdir=tmp_path,
        artifacts_dir=tmp_path / "ev",
        arch="gfx950",
        known_case_ids=["a"],
        known_case_ms={"a": 1.0},
    )

    assert not hasattr(bundle, "hardware")
    assert bundle.identity.arch == "gfx950"
    assert any("measures its roofs itself" in note for note in bundle.notes)
