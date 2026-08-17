# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The supported-board list must have exactly one definition.

``AMD_GPU_DISPATCH_IDENTITIES`` is that definition. On ``main`` the list is
retyped in five other places, which means adding a board there gives it a
dispatch identity while the resolver, the CLI and the preflight warning still
name the old set -- the copies do not fail loudly, they just disagree.
"""

from __future__ import annotations

from hyperloom.common.gpu_identity import AMD_GPU_DISPATCH_IDENTITIES
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
        # Only a subparsers action carries a dict of parsers here; an ordinary
        # option's ``choices`` is a plain sequence of values.
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
    """The warning tells the operator what to pass, so it has to stay true.

    A hand-typed list here fails the quiet way the others did: it keeps naming
    the old boards while ``--gpu-type`` has already moved on, and it is only
    printed on hosts where nothing resolved, which is where it is least likely
    to be noticed.
    """
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setattr(preflight, "detect_gfx_arch", lambda *a, **k: None)
    preflight._check_gfx_arch_resolvable(None)

    out = capsys.readouterr().out
    assert f"--gpu-type ({'/'.join(sorted(AMD_GPU_DISPATCH_IDENTITIES))})" in out


def test_a_tag_never_precedes_one_it_is_a_prefix_of():
    """Tags are substring-matched against rocm-smi output, so order decides.

    A shorter tag tested first would claim a longer board's name -- "MI300X"
    would answer for an "MI300XL" -- so the derived order has to keep the
    longer tag ahead of any tag that prefixes it.
    """
    for i, tag in enumerate(_PRODUCT_TAGS):
        for later in _PRODUCT_TAGS[i + 1 :]:
            assert not later.startswith(tag), f"{tag} shadows {later}"
