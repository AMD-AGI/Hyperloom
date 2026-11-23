# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU naming and the local probe every kernelforge entry point resolves its target through."""

from __future__ import annotations

import subprocess

import pytest

from kernelforge.kernel_backends.constants import KERNEL_BACKENDS
from kernelforge.utils.gpu_arch import (
    CDNA_OCCUPANCY,
    Profiler,
    arch_profile,
    canon_arch,
    canonical_gpu_type,
    detect_arch,
)

# The labels stored KB records are filed under, by the gfx target each stands for.
STORED_LABELS = [("gfx942", "mi300x"), ("gfx950", "mi355x"), ("gfx1100", "w7900"), ("gfx1151", "radeon8060s")]


@pytest.mark.parametrize(("gfx", "label"), STORED_LABELS)
def test_a_stored_label_and_its_gfx_target_name_each_other(gfx, label):
    assert canon_arch(label) == gfx
    assert canonical_gpu_type(gfx) == label
    assert canonical_gpu_type(label) == label


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("GFX950", "gfx950"),
        ("gfx1201", "gfx1201"),
        ("gfx950:sramecc+:xnack-", "gfx950"),
        ("MI300X", "gfx942"),
        ("radeon8060s", "gfx1151"),
    ],
)
def test_canon_arch_reads_a_gfx_target_or_a_stored_label(value, expected):
    assert canon_arch(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "gfx12-generic",
        "mi325x",
        "r9700",
        "AMD Instinct MI355X",
        "mi300x_vf",
    ],
)
def test_canon_arch_names_no_target_for_a_product_outside_the_stored_labels(value):
    assert canon_arch(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("MI300X", "mi300x"),
        ("GFX950", "mi355x"),
        ("Radeon8060S", "radeon8060s"),
        ("gfx1030", "gfx1030"),
        ("MI250-X", "mi250_x"),
    ],
)
def test_canonical_gpu_type_records_one_chip_under_one_name(value, expected):
    assert canonical_gpu_type(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gfx1201", "gfx1201"),
        ("r9700", "r9700"),
        ("r9600d", "r9600d"),
        ("MI325X", "mi325x"),
        ("amd instinct mi355x", "amd_instinct_mi355x"),
        ("mi300x_vf", "mi300x_vf"),
        ("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack_"),
        ("my-mi355x-box", "my_mi355x_box"),
    ],
)
def test_canonical_gpu_type_keeps_the_spelling_of_any_other_value(value, expected):
    """Boards sharing a target (``r9700``/``r9600d`` on gfx1201) keep their own records, and so does gfx1201."""
    assert canonical_gpu_type(value) == expected


def _rocminfo(monkeypatch, *, returncode=0, stdout=""):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["rocminfo"], returncode=returncode, stdout=stdout, stderr=""
        ),
    )


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "  Name:  gfx942\n  Name: gfx942\n", "gfx942"),
        (0, "  Name:  gfx1201\n  Name:  amdgcn-amd-amdhsa--gfx12-generic\n", "gfx1201"),
        (0, "Name: gfx942\nName: gfx90a\n", ""),
        (0, "no amd device here\n", ""),
        (1, "Name: gfx942\n", ""),
    ],
)
def test_detect_arch_reports_only_an_unambiguous_target(monkeypatch, returncode, stdout, expected):
    _rocminfo(monkeypatch, returncode=returncode, stdout=stdout)

    assert detect_arch() == expected


def test_detect_arch_is_empty_without_rocminfo(monkeypatch):
    def _missing(*args, **kwargs):
        raise FileNotFoundError("rocminfo")

    monkeypatch.setattr(subprocess, "run", _missing)

    assert detect_arch() == ""


@pytest.mark.parametrize("gpu_target", ["gfx942", "gfx950", "gfx950:sramecc+:xnack-", "MI355X", "gfx1151", "r9700", ""])
def test_instinct_and_unprofiled_targets_keep_the_instinct_defaults(gpu_target):
    profile = arch_profile(gpu_target)

    assert profile.hardware_knowledge == "hardware"
    assert profile.profiler is Profiler.ROCPROF_COMPUTE
    assert all(profile.supports_kernel_backend(backend) for backend in KERNEL_BACKENDS)
    assert profile.occupancy is CDNA_OCCUPANCY
    assert profile.prompt_note == ""


@pytest.mark.parametrize("gpu_target", ["gfx1100", "GFX1100", "gfx1100:xnack-", "w7900"])
def test_gfx1100_has_the_rdna3_profile(gpu_target):
    profile = arch_profile(gpu_target)

    assert profile.family == "RDNA3"
    assert profile.hardware_knowledge == "hardware/rdna3"
    assert profile.profiler is Profiler.ROCPROFV3
    assert {backend for backend in KERNEL_BACKENDS if profile.supports_kernel_backend(backend)} == {
        "triton",
        "hip",
        "hipblaslt",
        "ck",
    }
    assert profile.occupancy is not CDNA_OCCUPANCY
    assert "WMMA" in profile.rendered_prompt_note()


def test_the_rdna3_note_names_exactly_the_backends_its_profile_admits():
    """The note's backend sentence is derived from ``kernel_backends``, so the two cannot drift."""
    note = arch_profile("gfx1100").rendered_prompt_note()

    assert "The kernel backends that build here are ck, hip, hipblaslt, triton." in note
    assert "{" not in note


def test_every_profiled_hardware_pillar_ships_with_its_map():
    from kernelforge.config.resources import resource_path

    root = resource_path("local_knowledge")
    for gpu_target in ("gfx942", "gfx950", "gfx1100", "gfx1201"):
        assert (root / arch_profile(gpu_target).hardware_knowledge / "INDEX.md").is_file(), gpu_target


@pytest.mark.parametrize("gpu_target", ["gfx1201", "GFX1201", "gfx1201:sramecc-:xnack-"])
def test_gfx1201_has_the_rdna4_profile(gpu_target):
    profile = arch_profile(gpu_target)

    assert profile.family == "RDNA4"
    assert profile.hardware_knowledge == "hardware/rdna4"
    assert profile.profiler is Profiler.ROCPROFV3
    assert profile.occupancy.vgpr_limit == 96
    assert profile.occupancy.lds_limit_kib == 64
    assert "Use wave32 and WMMA" in profile.prompt_note
    assert "rocprofv3 kernel tracing" in profile.prompt_note
    assert "Analysis Agent stays static-only" in profile.prompt_note
