# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The supported-board list must have exactly one definition."""

from __future__ import annotations

import pytest

from hyperloom.common.gpu_identity import (
    AMD_GPU_DISPATCH_IDENTITIES,
    gfx_arch_for_gpu_type,
    is_gfx_arch,
)
from hyperloom.inference_optimizer.gpu_types import (
    _AMD_GPU_TYPES,
    _PRODUCT_TAGS,
    amd_gpu_dispatch_identity,
)


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


def test_accepted_boards_are_the_boards_with_identities():
    assert _AMD_GPU_TYPES == frozenset(AMD_GPU_DISPATCH_IDENTITIES)


def test_cli_accepts_exactly_the_boards_that_resolve():
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    choices = _gpu_type_choices(_build_parser())
    assert choices is not None, "--gpu-type is no longer a CLI option"
    assert sorted(choices) == sorted(AMD_GPU_DISPATCH_IDENTITIES)


def test_every_listed_board_actually_resolves():
    """The failure the copies produced: a known board answering ``None``."""
    for board in AMD_GPU_DISPATCH_IDENTITIES:
        assert amd_gpu_dispatch_identity(board) is not None, board


def test_product_tags_cover_the_same_boards():
    assert set(_PRODUCT_TAGS) == {b.upper() for b in AMD_GPU_DISPATCH_IDENTITIES}


def test_the_preflight_warning_names_the_boards_the_cli_accepts(capsys, monkeypatch):
    """The warning tells the operator what to pass, so it has to stay true."""
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setattr(preflight, "detect_gfx_arch", lambda *a, **k: None)
    preflight._check_gfx_arch_resolvable(None)

    out = capsys.readouterr().out
    assert f"--gpu-type ({'/'.join(sorted(AMD_GPU_DISPATCH_IDENTITIES))})" in out


def test_a_tag_never_precedes_one_it_is_a_prefix_of():
    """Tags are substring-matched against rocm-smi output, so order decides."""
    for i, tag in enumerate(_PRODUCT_TAGS):
        for later in _PRODUCT_TAGS[i + 1 :]:
            assert not later.startswith(tag), f"{tag} shadows {later}"


def test_the_gfx_arch_of_a_board_comes_from_the_identities_table():
    for board, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items():
        assert gfx_arch_for_gpu_type(board) == arch
        assert gfx_arch_for_gpu_type(board.upper()) == arch


@pytest.mark.parametrize("gpu_type", [None, "", "   ", "unknown_gpu", "mi250x"])
def test_a_board_without_an_identity_has_no_arch(gpu_type):
    """A board the CLI does not accept must not acquire an arch from a side table."""
    assert gfx_arch_for_gpu_type(gpu_type) is None


def test_the_build_path_reads_the_table_rather_than_its_own_copy():
    """The copy deleted here had drifted: it lacked mi325x and still named gfx90a boards."""
    from hyperloom.orchestrator.enablement import build

    assert build.gfx_arch_for_gpu_type is gfx_arch_for_gpu_type


def test_an_arch_names_itself_and_every_board_that_dispatches_to_it():
    for board, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items():
        assert is_gfx_arch(board, arch)
        assert is_gfx_arch(board.upper(), arch)
        assert is_gfx_arch(arch, arch)


@pytest.mark.parametrize("gpu_type", [None, "", "   ", "auto", "unknown_gpu", "mi250x"])
def test_an_unresolvable_gpu_type_names_no_arch(gpu_type):
    assert not is_gfx_arch(gpu_type, "gfx950")
    assert not is_gfx_arch(gpu_type, "gfx942")


def test_a_board_never_answers_to_another_boards_arch():
    for board, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items():
        for other in {a for a, _c in AMD_GPU_DISPATCH_IDENTITIES.values()} - {arch}:
            assert not is_gfx_arch(board, other), f"{board} answered to {other}"


def test_the_t0_recipe_isa_map_is_the_table():
    """T0 warm-start offers a recipe across same-ISA boards, so its map has to be the table's."""
    from hyperloom.orchestrator.knowledge import recipe_kb_t0

    assert recipe_kb_t0._GPU_ISA_BY_SKU == {b: a for b, (a, _c) in AMD_GPU_DISPATCH_IDENTITIES.items()}


def test_every_board_falls_back_to_exactly_its_same_isa_siblings():
    from hyperloom.orchestrator.knowledge import recipe_kb_t0

    for board, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items():
        expected = [b for b, (a, _c) in AMD_GPU_DISPATCH_IDENTITIES.items() if a == arch]
        assert recipe_kb_t0._hardware_fallback_values(board) == expected
        assert recipe_kb_t0._hardware_fallback_values(f"{board}_ws2_tp8") == [f"{b}_ws2_tp8" for b in expected]


def test_the_aiter_per_token_gate_is_every_gfx942_board():
    """The kernel ships per arch, so a new gfx942 board has to be gated in with its siblings."""
    from hyperloom.orchestrator.actions.executors._workload_envs import _GFX942_GPU_TYPES

    assert _GFX942_GPU_TYPES == frozenset(b for b, (a, _c) in AMD_GPU_DISPATCH_IDENTITIES.items() if a == "gfx942")


def test_the_fp8_dense_quant_type_reads_gfx950_from_the_table():
    from hyperloom.orchestrator.kernel.request_handlers import _is_gfx950

    for board, (arch, _cus) in AMD_GPU_DISPATCH_IDENTITIES.items():
        assert _is_gfx950(board) is (arch == "gfx950"), board
    assert _is_gfx950("gfx950") is True
