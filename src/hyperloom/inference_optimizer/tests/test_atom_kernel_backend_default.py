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


def test_the_call_site_stays_behind_the_atom_guard():
    """SGLang and vLLM must keep GEAK, and only the call site enforces that.

    Every test above calls this function directly, so none of them would notice
    the guard being widened or dropped. Read it out of the source instead: the
    call has to sit under a comparison of ``framework`` against ``"atom"``.
    """
    import ast
    import pathlib

    # Read the file rather than import the package: this assertion is about source
    # shape, and the import chain needs a POSIX-only module.
    cli_init = pathlib.Path(__file__).resolve().parents[1] / "cli" / "__init__.py"
    tree = ast.parse(cli_init.read_text(encoding="utf-8"))

    def calls_it(node: ast.AST) -> bool:
        return any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "_apply_atom_auto_tighten"
            for inner in ast.walk(node)
        )

    def guards_on_atom(test: ast.expr) -> bool:
        return any(
            isinstance(cmp, ast.Compare)
            and isinstance(cmp.left, ast.Name)
            and cmp.left.id == "framework"
            and any(isinstance(c, ast.Constant) and c.value == "atom" for c in cmp.comparators)
            for cmp in ast.walk(test)
        )

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and calls_it(node) and guards_on_atom(node.test)
    ]
    assert guarded, "the _apply_atom_auto_tighten call is no longer behind a framework == 'atom' test"

    # And nowhere else: an unguarded second call would reach every framework.
    total = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_apply_atom_auto_tighten"
    )
    assert total == 1, f"expected exactly one call site, found {total}"
