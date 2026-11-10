# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Strix Halo host resolves to ``radeon8060s`` instead of falling back to ``unknown_gpu``."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import gpu_types

# `rocm-smi --showproductname` from a Ryzen AI Max+ 395.
STRIX_HALO_ROCM_SMI = """\
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]\t\t: Card Series: \t\tRadeon 8060S Graphics
GPU[0]\t\t: Card Model: \t\t0x1586
GPU[0]\t\t: Card Vendor: \t\tAdvanced Micro Devices, Inc. [AMD/ATI]
GPU[0]\t\t: Card SKU: \t\tSTRXLGEN
GPU[0]\t\t: GFX Version: \t\tgfx1151
==========================================================================================
================================== End of ROCm SMI Log ===================================
"""


@pytest.fixture
def probe(monkeypatch):
    """Run the autodetect against the given Card Series, with torch reporting gfx1151."""
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda _idx: SimpleNamespace(gcnArchName="gfx1151"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def _run(card_series: str) -> str | None:
        stdout = STRIX_HALO_ROCM_SMI.replace("Radeon 8060S Graphics", card_series)
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=stdout))
        return gpu_types._autodetect_gpu_type()

    return _run


@pytest.mark.parametrize(
    "card_series",
    ["Radeon 8060S Graphics", "AMD Radeon 8060S Graphics", "AMD Radeon8060S Graphics"],
)
def test_probe_reads_radeon8060s_from_the_product_name(probe, card_series):
    assert probe(card_series) == "radeon8060s"


@pytest.mark.parametrize(
    ("card_series", "expected"),
    [
        ("AMD Instinct MI300X OAM", "mi300x"),
        ("AMD Instinct MI300X VF", "mi300x"),
        ("AMD Instinct MI308X", "mi308x"),
        ("AMD Instinct MI325X OAM", "mi325x"),
        ("AMD Instinct MI355X", "mi355x"),
        ("AMD Radeon AI PRO R9700", "r9700"),
        ("Radeon 8050S Graphics", None),
        ("Radeon 8065S Graphics", None),
        ("AMD Radeon Graphics", None),
        ("N/A", None),
    ],
)
def test_other_product_names_resolve_as_before(probe, card_series, expected):
    """gfx1151 alone names no board: it spans the 8050S and the 8060S."""
    assert probe(card_series) == expected


def test_radeon8060s_is_an_accepted_gpu_type(monkeypatch):
    monkeypatch.delenv("TARGET_GPU_TYPE", raising=False)
    monkeypatch.setenv("GPU_TYPE", "radeon8060s")
    assert gpu_types._resolve_amd_gpu_type() == "radeon8060s"
    assert gpu_types._resolve_amd_gpu_type("radeon8060s") == "radeon8060s"


def test_radeon8060s_dispatches_as_gfx1151_with_40_cus():
    board = gpu_types.amd_gpu_board("radeon8060s")
    assert (board.gfx, board.compute_units) == ("gfx1151", 40)


def test_radeon8060s_names_its_own_magpie_script():
    assert gpu_types.gpu_runner_type("radeon8060s") == "radeon8060s"


def test_radeon8060s_carries_only_the_figures_amd_publishes():
    from hyperloom.inference_optimizer.roofline_ceiling import HW_SPECS, resolve_compute_peak_provenance

    spec = HW_SPECS["radeon8060s"]
    assert (spec["hbm_gb"], spec["hbm_bw_gbps"]) == (None, 256.0)
    assert spec["peak_tflops"]["bf16"] == spec["peak_tflops"]["fp16"] == 60.0
    assert resolve_compute_peak_provenance("radeon8060s", "bf16")["compute_peak_convention"] == "vendor"
    for precision in ("fp8", "fp32", "mxfp4"):
        assert resolve_compute_peak_provenance("radeon8060s", precision)["compute_peak_convention"] == "unknown"
