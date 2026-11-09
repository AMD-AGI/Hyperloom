# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every per-board decision, pinned to its value, so a change to the board tables shows exactly what it moves."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.roofline_ceiling import resolve_compute_peak_provenance

_BOARDS = [
    # product, targeted build validated, bypass peak convention
    ("mi300x", True, "achievable"),
    ("mi308x", True, "vendor"),
    ("mi325x", True, "achievable"),
    ("mi350x", True, "vendor"),
    ("mi355x", True, "vendor"),
    ("r9700", False, "vendor"),
]


@pytest.mark.parametrize(("product", "build_validated", "convention"), _BOARDS)
def test_board_decisions(product, build_validated, convention):
    assert gpu_types.targeted_build_validated(product) is build_validated
    assert resolve_compute_peak_provenance(product, "bf16")["compute_peak_convention"] == convention


def test_radeon8060s_has_no_framework_restriction():
    assert gpu_types.targeted_build_validated("radeon8060s") is True
    assert gpu_types._board_framework_error("radeon8060s", "sglang") is None


@pytest.mark.parametrize("product", ["mi300x", "mi308x", "mi325x", "mi350x", "mi355x"])
@pytest.mark.parametrize("framework", ["vllm", "sglang", "atom", "xdit"])
def test_instinct_boards_leave_the_framework_unconstrained(product, framework):
    assert gpu_types._board_framework_error(product, framework) is None


@pytest.mark.parametrize("framework", ["sglang", "atom", "xdit"])
def test_r9700_is_validated_for_vllm_only(framework):
    assert gpu_types._board_framework_error("R9700", "vllm") is None
    assert gpu_types._board_framework_error("r9700", framework) == (
        f"r9700 is currently validated only for vllm; framework {framework!r} has not been validated on it"
    )


def test_only_r9700_pins_its_geak_identity():
    assert gpu_types.geak_identity_pin("r9700") == {"expected_target": "r9700", "expected_gfx": "gfx1201"}
    for product in ("mi300x", "mi308x", "mi325x", "mi350x", "mi355x", "radeon8060s", "unknown_gpu", ""):
        assert gpu_types.geak_identity_pin(product) == {}, product


@pytest.mark.parametrize(
    ("text", "product"),
    [
        ("Card Series: AMD Instinct MI300X OAM", "mi300x"),
        ("Card Series: AMD Instinct MI300X VF", "mi300x"),
        ("Card Series: AMD Instinct MI308X OAM", "mi308x"),
        ("Card Series: AMD Instinct MI325X OAM", "mi325x"),
        ("Card Series: AMD Instinct MI350X OAM", "mi350x"),
        ("Card Series: AMD Instinct MI355X OAM", "mi355x"),
        ("AMD_INSTINCT_MI300X", "mi300x"),
        ("Card Series: AMD Radeon AI PRO R9700", "r9700"),
        ("card series: amd radeon ai pro r9700", "r9700"),
        ("Card Series: AMD Instinct MI300XL", None),
        ("Card Series: AMD Radeon AI PRO R9700S", None),
        ("Card Series: AMD Radeon AI PRO XR9700", None),
        ("Card Series: AMD Radeon 8060S Graphics", "radeon8060s"),
        ("gfx1151", None),
        ("gfx942:sramecc+:xnack-", "mi300x"),
        ("gfx950:sramecc+:xnack-", "mi355x"),
        ("gfx1201", None),
        ("gfx1100", None),
        ("nothing here", None),
        ("", None),
        (None, None),
    ],
)
def test_probe_output_resolves_to_the_same_board(text, product):
    """A longer product name that contains a tag (``MI300XL``, ``R9700S``) is a different product."""
    assert gpu_types.parse_gpu_type(text) == product
