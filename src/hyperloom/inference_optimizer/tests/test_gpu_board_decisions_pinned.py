# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every per-board decision, pinned to its value, so a change to the board tables shows exactly what it moves."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.multi_node._internal.gpu_probe import _parse_gpu_type
from hyperloom.inference_optimizer.roofline_ceiling import HW_SPECS
from hyperloom.orchestrator.actions.executors._server_lifecycle import MAGPIE_BUILTIN_SCRIPTS
from hyperloom.orchestrator.trace_analysis._bypass_roofline import _resolve_hw_spec

_BOARDS = [
    # product, runner, targeted build validated, (hbm GB, hbm GB/s, bf16, fp8, mxfp4, fp32), bypass peak convention
    ("mi300x", "mi300x", True, (192.0, 5300.0, 1307.4, 2614.9, None, 163.4), "achievable"),
    ("mi308x", "mi300x", True, (192.0, 5300.0, 1307.4, 2614.9, None, 163.4), "achievable"),
    ("mi325x", "mi300x", True, (256.0, 6000.0, 1307.4, 2614.9, None, 163.4), "achievable"),
    ("mi355x", "mi355x", True, (288.0, 8000.0, 2516.6, 5033.2, 10066.4, 157.3), "vendor"),
    ("r9700", "gfx12", False, (32.0, 640.0, 191.0, 383.0, None, 47.8), "vendor"),
]


@pytest.mark.parametrize(("product", "runner", "build_validated", "vendor", "convention"), _BOARDS)
def test_board_decisions(product, runner, build_validated, vendor, convention):
    assert gpu_types.gpu_runner_type(product) == runner
    assert gpu_types.targeted_build_validated(product) is build_validated
    spec = HW_SPECS[product]
    peaks = spec["peak_tflops"]
    assert (
        spec["hbm_gb"],
        spec["hbm_bw_gbps"],
        peaks["bf16"],
        peaks["fp8"],
        peaks.get("mxfp4"),
        peaks["fp32"],
    ) == vendor
    assert _resolve_hw_spec(product)[1] == convention


def test_server_reuse_scripts():
    assert MAGPIE_BUILTIN_SCRIPTS == {
        "vllm_mi300x.sh",
        "vllm_mi355x.sh",
        "vllm_gfx12.sh",
        "sglang_mi300x.sh",
        "sglang_mi355x.sh",
        "atom_mi300x.sh",
        "atom_mi355x.sh",
    }


@pytest.mark.parametrize("product", ["mi300x", "mi308x", "mi325x", "mi355x"])
@pytest.mark.parametrize("framework", ["vllm", "sglang", "atom", "xdit"])
def test_instinct_boards_leave_the_framework_unconstrained(product, framework):
    assert gpu_types._runner_framework_error(product, framework) is None


@pytest.mark.parametrize("framework", ["sglang", "atom", "xdit"])
def test_the_gfx12_runner_is_validated_for_vllm_only(framework):
    assert gpu_types._runner_framework_error("r9700", "vllm") is None
    assert gpu_types._runner_framework_error("r9700", framework) == (
        "Magpie runner 'gfx12' is currently validated only for vllm; "
        f"framework {framework!r} has no validated gfx12 runner"
    )


def test_only_r9700_pins_its_geak_identity():
    assert gpu_types.geak_identity_pin("r9700") == {"expected_target": "r9700", "expected_gfx": "gfx1201"}
    for product in ("mi300x", "mi308x", "mi325x", "mi355x", "unknown_gpu", ""):
        assert gpu_types.geak_identity_pin(product) == {}, product


@pytest.mark.parametrize(
    ("text", "product"),
    [
        ("Card Series: AMD Instinct MI300X OAM", "mi300x"),
        ("Card Series: AMD Instinct MI300X VF", "mi300x"),
        ("Card Series: AMD Instinct MI308X OAM", "mi308x"),
        ("Card Series: AMD Instinct MI325X OAM", "mi325x"),
        ("Card Series: AMD Instinct MI355X OAM", "mi355x"),
        ("AMD_INSTINCT_MI300X", "mi300x"),
        ("Card Series: AMD Radeon AI PRO R9700", "r9700"),
        ("Card Series: AMD Radeon AI PRO R9700S", None),
        ("gfx942:sramecc+:xnack-", "mi300x"),
        ("gfx950:sramecc+:xnack-", "mi355x"),
        ("gfx1201", None),
        ("gfx1100", None),
        ("nothing here", None),
    ],
)
def test_probe_output_resolves_to_the_same_board(text, product):
    assert _parse_gpu_type(text) == product
