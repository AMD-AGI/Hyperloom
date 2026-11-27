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
    ("r9600d", False, "vendor"),
    ("w7900", False, "vendor"),
    ("w7900d", False, "vendor"),
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


@pytest.mark.parametrize(("board", "allowed"), [("r9700", "llamacpp, sglang, vllm"), ("r9600d", "sglang, vllm")])
def test_gfx1201_board_frameworks(board, allowed):
    for framework in allowed.split(", "):
        gpu_types.require_board_framework(board.upper(), framework)
    for framework in ("atom", "xdit"):
        assert gpu_types._board_framework_error(board, framework) == (
            f"{board} is currently validated only for {allowed}; framework {framework!r} has not been validated on it"
        )


@pytest.mark.parametrize("board", ["w7900", "w7900d"])
def test_gfx1100_boards_still_refuse_sglang(board):
    assert gpu_types._board_framework_error(board, "sglang") is not None


def test_llamacpp_is_validated_on_r9700_and_w7900d_only():
    radeon = ("r9700", "r9600d", "w7900", "w7900d")
    assert [board for board in radeon if gpu_types._board_framework_error(board, "llamacpp") is None] == [
        "r9700",
        "w7900d",
    ]


def test_only_the_gfx1201_radeon_boards_pin_their_geak_identity():
    assert gpu_types.geak_identity_pin("r9700") == {"expected_target": "r9700", "expected_gfx": "gfx1201"}
    assert gpu_types.geak_identity_pin("r9600d") == {"expected_target": "r9600d", "expected_gfx": "gfx1201"}
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
        ("Card Series: AMD Radeon AI PRO R9600D", "r9600d"),
        ("Card Series: AMD Radeon PRO W7900", "w7900"),
        ("Card Series: AMD Radeon PRO W7900D", "w7900d"),
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
