# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The supported-board list must have exactly one definition, and every board table derives from it."""

from __future__ import annotations

import pytest

from hyperloom_common.gpu_identity import (
    AMD_GPU_BOARDS,
    gfx_arch_for_gpu_type,
    is_gfx_arch,
)
from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.gpu_types import amd_gpu_board


def _gpu_type_choices(parser) -> list | None:
    """Find ``--gpu-type``'s choices, which are defined on a subcommand."""
    for action in parser._actions:
        if "--gpu-type" in (action.option_strings or []):
            return list(action.choices or [])
        # Only a subparsers action carries a dict of parsers here; an ordinary option's ``choices`` is a plain
        # sequence of values.
        if isinstance(getattr(action, "choices", None), dict):
            for sub in action.choices.values():
                found = _gpu_type_choices(sub)
                if found is not None:
                    return found
    return None


def test_every_board_has_exactly_one_policy_row():
    """Each hardware row pairs with one policy row, so every board launches through its own runner."""
    assert set(gpu_types._BOARD_POLICIES) == set(AMD_GPU_BOARDS)


def test_cli_accepts_exactly_the_boards_that_resolve():
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    choices = _gpu_type_choices(_build_parser())
    assert choices is not None, "--gpu-type is no longer a CLI option"
    assert sorted(choices) == sorted(AMD_GPU_BOARDS)


def test_every_listed_board_actually_resolves():
    """The failure the copies produced: a known board answering ``None``."""
    for product, board in AMD_GPU_BOARDS.items():
        assert amd_gpu_board(product) is board, product


def test_r9700_has_explicit_gfx1201_dispatch_identity():
    board = AMD_GPU_BOARDS["r9700"]
    assert (board.gfx, board.compute_units) == ("gfx1201", 64)


def test_the_preflight_warning_names_the_boards_the_cli_accepts(capsys, monkeypatch):
    """The warning tells the operator what to pass, so it has to stay true."""
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setattr(preflight, "detect_gfx_arch", lambda *a, **k: None)
    preflight._check_gfx_arch_resolvable(None)

    out = capsys.readouterr().out
    assert f"--gpu-type ({'/'.join(sorted(AMD_GPU_BOARDS))})" in out


def test_the_gfx_arch_of_a_board_comes_from_the_board_table():
    for product, board in AMD_GPU_BOARDS.items():
        assert gfx_arch_for_gpu_type(product) == board.gfx
        assert gfx_arch_for_gpu_type(product.upper()) == board.gfx


@pytest.mark.parametrize("gpu_type", [None, "", "   ", "unknown_gpu", "mi250x"])
def test_a_board_without_an_identity_has_no_arch(gpu_type):
    """A board the CLI does not accept must not acquire an arch from a side table."""
    assert gfx_arch_for_gpu_type(gpu_type) is None


def test_the_build_path_reads_the_table_rather_than_its_own_copy():
    """The copy deleted here had drifted: it lacked mi325x and still named gfx90a boards."""
    from hyperloom.orchestrator.enablement import build

    assert build.gfx_arch_for_gpu_type is gfx_arch_for_gpu_type


def test_an_arch_names_itself_and_every_board_that_dispatches_to_it():
    for product, board in AMD_GPU_BOARDS.items():
        assert is_gfx_arch(product, board.gfx)
        assert is_gfx_arch(product.upper(), board.gfx)
        assert is_gfx_arch(board.gfx, board.gfx)


@pytest.mark.parametrize("gpu_type", [None, "", "   ", "auto", "unknown_gpu", "mi250x"])
def test_an_unresolvable_gpu_type_names_no_arch(gpu_type):
    assert not is_gfx_arch(gpu_type, "gfx950")
    assert not is_gfx_arch(gpu_type, "gfx942")


def test_a_board_never_answers_to_another_boards_arch():
    for product, board in AMD_GPU_BOARDS.items():
        for other in {b.gfx for b in AMD_GPU_BOARDS.values()} - {board.gfx}:
            assert not is_gfx_arch(product, other), f"{product} answered to {other}"


def test_the_t0_recipe_isa_map_is_the_table():
    """T0 warm-start offers a recipe across same-ISA boards, so its map has to be the table's."""
    from hyperloom.orchestrator.knowledge import recipe_kb_t0

    assert recipe_kb_t0._GPU_ISA_BY_SKU == {product: board.gfx for product, board in AMD_GPU_BOARDS.items()}


def test_every_board_falls_back_to_exactly_its_same_isa_siblings():
    from hyperloom.orchestrator.knowledge import recipe_kb_t0

    for product, board in AMD_GPU_BOARDS.items():
        expected = [p for p, b in AMD_GPU_BOARDS.items() if b.gfx == board.gfx]
        assert recipe_kb_t0._hardware_fallback_values(product) == expected
        assert recipe_kb_t0._hardware_fallback_values(f"{product}_ws2_tp8") == [f"{p}_ws2_tp8" for p in expected]


def test_the_aiter_per_token_gate_is_every_gfx942_board():
    """The kernel ships per arch, so a new gfx942 board has to be gated in with its siblings."""
    from hyperloom.orchestrator.actions.executors._workload_envs import _GFX942_GPU_TYPES

    assert _GFX942_GPU_TYPES == frozenset(p for p, b in AMD_GPU_BOARDS.items() if b.gfx == "gfx942")


def test_the_fp8_dense_quant_type_reads_gfx950_from_the_table():
    from hyperloom.orchestrator.kernel._untuned_gemm_csv import _is_gfx950

    for product, board in AMD_GPU_BOARDS.items():
        assert _is_gfx950(product) is (board.gfx == "gfx950"), product
    assert _is_gfx950("gfx950") is True


def test_the_roofline_peaks_carry_every_precision_spelling():
    """Vendor and achievable peaks are keyed by precision once and read under each of its spellings."""
    from hyperloom.inference_optimizer.roofline_ceiling import HW_SPECS, HW_SPECS_ACHIEVABLE

    vendor, mi355x = AMD_GPU_BOARDS["mi300x"].vendor_peak_tflops, HW_SPECS["mi355x"]["peak_tflops"]
    assert {
        tag: HW_SPECS["mi300x"]["peak_tflops"][tag] for tag in ("bfloat16", "float16", "float8_e5m2", "float32")
    } == {
        "bfloat16": vendor["bf16"],
        "float16": vendor["fp16"],
        "float8_e5m2": vendor["fp8"],
        "float32": vendor["fp32"],
    }
    assert mi355x["mxfp4"] == mi355x["float4"] == AMD_GPU_BOARDS["mi355x"].vendor_peak_tflops["fp4"]
    achievable = HW_SPECS_ACHIEVABLE["mi325x"]["peak_tflops"]
    assert achievable["bfloat16"] == achievable["bf16"]
    assert achievable["float8_e4m3fn"] == achievable["float8_e5m2"] == achievable["fp8"]
    assert set(HW_SPECS) == {product for product, board in AMD_GPU_BOARDS.items() if board.hbm_bw_gbps is not None}
