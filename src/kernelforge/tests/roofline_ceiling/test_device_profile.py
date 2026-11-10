# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Machine identity: what distinguishes one measurable configuration from another."""

from __future__ import annotations

from kernelforge.roofline_ceiling import device_profile as profile_module
from kernelforge.roofline_ceiling.device_profile import DeviceIdentity, describe_device

_MI355X = DeviceIdentity(
    arch="gfx950",
    device_name="AMD Instinct MI355X",
    compute_partition="SPX",
    memory_partition="NPS1",
)


def test_partition_mode_is_part_of_the_identity_not_metadata():
    """Slicing a card into CPX changes what one slice can reach, so its roofs differ."""
    cpx = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X", compute_partition="CPX")

    assert _MI355X.slug() != cpx.slug()


def test_the_slug_names_the_whole_configuration():
    assert _MI355X.slug() == "gfx950-mi355x-spx-nps1"


def test_an_undetermined_field_drops_out_of_the_slug_rather_than_being_guessed():
    partial = DeviceIdentity(arch="gfx950", device_name="AMD Instinct MI355X")

    assert partial.slug() == "gfx950-mi355x"


def test_a_machine_nothing_could_be_read_from_still_has_a_name():
    assert DeviceIdentity(arch="").slug() == "unknown"


def test_the_identity_is_handed_to_the_analyst_whole():
    described = _MI355X.describe()

    assert described == {
        "arch": "gfx950",
        "device_name": "AMD Instinct MI355X",
        "compute_partition": "SPX",
        "memory_partition": "NPS1",
    }


def test_the_probes_are_read_from_rocminfo_and_rocm_smi(monkeypatch):
    outputs = {
        "rocminfo": "  Marketing Name:    AMD Instinct MI355X\n",
        "rocm-smi": "Compute Partition: SPX\nMemory Partition: NPS1\n",
    }
    monkeypatch.setattr(profile_module, "_run_text", lambda argv, **_k: outputs.get(argv[0], ""))
    monkeypatch.setattr(profile_module, "canon_arch", lambda value: "gfx950" if value else "")

    identity = describe_device("gfx950")

    assert identity == _MI355X


def test_a_probe_that_is_not_installed_leaves_its_field_empty(monkeypatch):
    """Empty says 'not determined', which is what lowers the analyst's confidence."""
    monkeypatch.setattr(profile_module, "_run_text", lambda _argv, **_k: "")
    monkeypatch.setattr(profile_module, "canon_arch", lambda value: "gfx950" if value else "")

    identity = describe_device("gfx950")

    assert identity.arch == "gfx950"
    assert (identity.device_name, identity.compute_partition, identity.memory_partition) == ("", "", "")


def test_nothing_here_stores_a_measured_roof():
    """Roofs are measured per session and never written down; only identity lives here."""
    exported = set(profile_module.__all__)

    assert exported == {"DeviceIdentity", "describe_device"}
