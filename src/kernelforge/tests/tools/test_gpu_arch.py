# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU naming and the local probe every kernelforge entry point resolves its target through."""

from __future__ import annotations

import subprocess

import pytest

from kernelforge.utils.gpu_arch import canon_arch, canonical_gpu_type, detect_arch, product_arch


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gfx942", "gfx942"),
        ("GFX950", "gfx950"),
        ("gfx1201", "gfx1201"),
        ("MI300X", "gfx942"),
        ("mi308x", "gfx942"),
        ("mi355x", "gfx950"),
        ("AMD Instinct MI355X", "gfx950"),
        ("amd_instinct_mi300x", "gfx942"),
    ],
)
def test_canon_arch_folds_product_names_onto_their_gfx_target(value, expected):
    assert canon_arch(value) == expected


@pytest.mark.parametrize("value", ["", "some-new-gpu", "gfx12-generic", "gfx"])
def test_canon_arch_names_no_target_rather_than_a_wrong_one(value):
    assert canon_arch(value) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gfx942", "mi300x"),
        ("MI325X", "mi300x"),
        ("amd instinct mi355x", "mi355x"),
        ("gfx1100", "gfx1100"),
        ("MI250-X", "mi250_x"),
    ],
)
def test_canonical_gpu_type_records_one_chip_under_one_name(value, expected):
    assert canonical_gpu_type(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("mi300x_vf", "mi300x_vf"),
        ("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack_"),
        ("my-mi355x-box", "my_mi355x_box"),
    ],
)
def test_canonical_gpu_type_keeps_a_spelling_that_only_contains_a_known_name(value, expected):
    assert canonical_gpu_type(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("mi300x", "gfx942"),
        (" MI355X ", "gfx950"),
        ("amd_instinct_mi300x", "gfx942"),
        ("mi300x_vf", ""),
        ("amd instinct mi300x", ""),
        ("gfx942", ""),
    ],
)
def test_product_arch_maps_only_an_exact_product_name(value, expected):
    assert product_arch(value) == expected


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
