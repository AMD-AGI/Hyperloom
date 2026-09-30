# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for roofline_provider (Milestone 1: protocol + NativeRooflineProvider + factory).

M1 is additive: NativeRooflineProvider must be a faithful delegate of the stock native math, and
the factory must return native for every mode (nothing is wired to the provider yet, so a run is
byte-identical by construction). These tests pin the delegation + factory contract.
"""

from __future__ import annotations

import hyperloom.inference_optimizer.roofline_ceiling as rcl
import hyperloom.inference_optimizer.roofline_provider as rp


def test_native_satisfies_protocol() -> None:
    assert isinstance(rp.NativeRooflineProvider(), rp.RooflineProvider)


def test_factory_native_without_csv_dir() -> None:
    class _S:
        roofline_csv_dir = ""
        roofline_csv_strict = False

    # default (no dir) and --no-roofline-csv both stay pure native
    assert isinstance(rp.make_roofline_provider(_S()), rp.NativeRooflineProvider)


def test_factory_external_with_csv_dir(tmp_path) -> None:
    class _S:
        roofline_csv_dir = str(tmp_path)
        roofline_csv_strict = False

    prov = rp.make_roofline_provider(_S())
    assert isinstance(prov, rp.CsvRooflineProvider)
    # an empty external dir -> every value misses -> lenient fallback to native (no raise)
    assert prov.arch_peak("mi355x", "bf16") == rp.NativeRooflineProvider().arch_peak("mi355x", "bf16")


def test_factory_external_strict_raises_on_missing(tmp_path) -> None:
    class _S:
        roofline_csv_dir = str(tmp_path)
        roofline_csv_strict = True

    prov = rp.make_roofline_provider(_S())
    assert isinstance(prov, rp.CsvRooflineProvider)
    import pytest

    with pytest.raises(FileNotFoundError):
        prov.arch_peak("mi355x", "bf16")  # empty dir + strict -> raise, no fallback


def test_arch_peak_delegates_to_native() -> None:
    prov = rp.NativeRooflineProvider()
    for device, dtype in [("mi355x", "bf16"), ("mi300x", "fp8"), ("nonexistent-gpu", "bf16")]:
        ach = rcl._resolve_achievable_tflops(device, dtype)
        vendor = rcl._resolve_peak_tflops(device, dtype)
        expected = float(ach) if ach and ach > 0 else (float(vendor) if vendor and vendor > 0 else None)
        assert prov.arch_peak(device, dtype) == expected


def test_mem_bw_delegates_to_hw_specs() -> None:
    prov = rp.NativeRooflineProvider()
    for device in list(rcl.HW_SPECS) + ["nonexistent-gpu"]:
        spec = rcl.HW_SPECS.get((device or "").strip().lower())
        bw = spec.get("hbm_bw_gbps") if spec else None
        expected = float(bw) if isinstance(bw, (int, float)) and bw > 0 else None
        assert prov.mem_bw(device) == expected


def test_kernel_is_none_for_native() -> None:
    # Native has no per-kernel CSV row; consumers keep their inline computation on None.
    assert rp.NativeRooflineProvider().kernel("triton_softmax_kernel") is None
