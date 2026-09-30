# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Experimental Radeon dispatch must never select an Instinct runner."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hyperloom.common.provenance import detect_gfx_arch
from hyperloom.inference_optimizer import gpu_types
from kernelforge.fusion.gpu_arch import canon_arch


def test_radeon_identity_and_runner():
    assert gpu_types.amd_gpu_dispatch_identity("radeon8065s") == ("gfx1151", 40)
    assert gpu_types._gpu_runner_type("radeon8065s") == "radeon8065s"
    assert detect_gfx_arch({}, gpu_type="radeon8065s") == "gfx1151"
    assert canon_arch("radeon8065s") == "gfx1151"


def test_spaced_product_name(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="AMD Radeon 8065S"))
    assert gpu_types._autodetect_gpu_type() == "radeon8065s"


def test_torch_product_name(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    props = SimpleNamespace(name="AMD Radeon 8065S Graphics", gcnArchName="gfx1151:sramecc-:xnack-")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: props))
    )
    assert gpu_types._autodetect_gpu_type() == "radeon8065s"


def test_generic_gfx1151_does_not_guess_board(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    props = SimpleNamespace(name="AMD Radeon Graphics", gcnArchName="gfx1151:sramecc-:xnack-")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: props))
    )
    assert gpu_types._autodetect_gpu_type() is None
    assert gpu_types._resolve_gpu_type("radeon8065s", "")[0] == "radeon8065s"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("AMD Instinct MI300X", "mi300x"), ("AMD Instinct MI325X", "mi325x"), ("AMD Instinct MI355X", "mi355x")],
)
def test_instinct_product_detection_preserved(monkeypatch, name, expected):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=name))
    assert gpu_types._autodetect_gpu_type() == expected


def test_explicit_radeon_identity_overrides_environment(monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    assert gpu_types._resolve_amd_gpu_type("radeon8065s") == "radeon8065s"
    assert gpu_types.amd_gpu_dispatch_identity("radeon8065s") == ("gfx1151", 40)
