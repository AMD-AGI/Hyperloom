# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--framework atom`` defaults the kernel phase to the forge backend.

GEAK's extraction declines to guess a rewrite seam for a quantized, non-vLLM
backend, so on atom it returns no candidates -- and the default phase split
hands that phase half the session. forge is the backend that works there, so it
is the default rather than an opt-in the operator has to know about. A value the
operator named is kept: running GEAK on atom on purpose stays possible.
"""

from __future__ import annotations

import argparse
import os

import pytest

from hyperloom.inference_optimizer.cli import _apply_atom_auto_tighten


def _args(**overrides: object) -> argparse.Namespace:
    base = {"nodes": 1, "no_kernel": False}
    base.update(overrides)
    return argparse.Namespace(**base)


_KEY = "KERNEL_OPT_BACKEND_ORDER"
_WARNING = "so the kernel phase runs GEAK"


def test_unset_backend_defaults_to_forge(monkeypatch, capsys):
    monkeypatch.delenv(_KEY, raising=False)

    _apply_atom_auto_tighten(_args())

    assert os.environ[_KEY] == "forge"
    out = capsys.readouterr()
    assert "defaulted to 'forge'" in out.out
    assert _WARNING not in out.err, "the default path must not emit a warning"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_value_counts_as_unset(monkeypatch, capsys, blank):
    """``.env`` files routinely carry an empty assignment; it is not a choice."""
    monkeypatch.setenv(_KEY, blank)

    _apply_atom_auto_tighten(_args())

    assert os.environ[_KEY] == "forge"
    assert _WARNING not in capsys.readouterr().err


def test_an_explicit_forge_is_left_alone_and_not_warned_about(monkeypatch, capsys):
    monkeypatch.setenv(_KEY, "forge")

    _apply_atom_auto_tighten(_args())

    assert os.environ[_KEY] == "forge"
    assert _WARNING not in capsys.readouterr().err


def test_forge_is_matched_case_insensitively(monkeypatch, capsys):
    """``forge_explicitly_enabled`` lowercases, so a shouted value is still an opt-in."""
    monkeypatch.setenv(_KEY, "FORGE")

    _apply_atom_auto_tighten(_args())

    assert os.environ[_KEY] == "FORGE"
    assert _WARNING not in capsys.readouterr().err


@pytest.mark.parametrize("value", ["geak", "GEAK", "forge,geak"])
def test_an_operator_named_backend_is_kept_and_warned_about(monkeypatch, capsys, value):
    """The opt-in is an exact match on ``forge``; everything else means GEAK."""
    monkeypatch.setenv(_KEY, value)

    _apply_atom_auto_tighten(_args())

    assert os.environ[_KEY] == value, "a named backend must not be rewritten"
    assert _WARNING in capsys.readouterr().err


def test_no_kernel_leaves_the_backend_alone(monkeypatch, capsys):
    """Nothing runs the kernel phase, so the backend it would have used is moot."""
    monkeypatch.delenv(_KEY, raising=False)

    _apply_atom_auto_tighten(_args(no_kernel=True))

    assert _KEY not in os.environ
    out = capsys.readouterr()
    assert _WARNING not in out.err
    assert "defaulted to 'forge'" not in out.out


def test_the_default_lands_before_the_session_records_it(monkeypatch):
    """``_seed_shared_state`` reads the env to record ``kernel_optimizer``.

    The call order in ``main`` puts this function first; assert the observable
    half of that contract, so a session cannot record ``geak`` while running forge.
    """
    from hyperloom.common.env import forge_explicitly_enabled

    monkeypatch.delenv(_KEY, raising=False)

    _apply_atom_auto_tighten(_args())

    assert forge_explicitly_enabled() is True


def test_multi_node_still_fails_fast(monkeypatch):
    """The pre-existing IR-8 guard is untouched, and fails before any defaulting."""
    monkeypatch.delenv(_KEY, raising=False)

    with pytest.raises(SystemExit) as exc:
        _apply_atom_auto_tighten(_args(nodes=2))

    assert exc.value.code == 2
    assert _KEY not in os.environ
