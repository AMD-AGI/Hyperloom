# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--framework atom`` defaults the kernel phase to the forge backend.

GEAK's extractor may not guess a rewrite seam on a quantized, non-vLLM backend --
it has to resolve one from the live server, a path unproven on atom -- and the
default phase split hands that phase half the session. forge needs no seam
discovery at all, so it is the default rather than an opt-in the operator has to
know about. A value the operator named is kept: running GEAK on atom on purpose
stays possible.
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


def _cli_source_tree():
    """Parse ``cli/__init__.py`` as source.

    Read the file rather than import the package: these assertions are about
    source shape, and the import chain needs a POSIX-only module.
    """
    import ast
    import pathlib

    cli_init = pathlib.Path(__file__).resolve().parents[1] / "cli" / "__init__.py"
    return ast, ast.parse(cli_init.read_text(encoding="utf-8"))


def _calls_auto_tighten(ast, node) -> bool:
    return any(
        isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "_apply_atom_auto_tighten"
        for inner in ast.walk(node)
    )


def test_every_call_site_stays_behind_an_atom_guard():
    """SGLang and vLLM must keep GEAK, and only the call sites enforce that.

    Every behavioural test above calls this function directly, so none of them
    would notice a guard being widened or dropped. Read it out of the source
    instead: each call has to sit under a test that the framework is atom.
    """
    ast, tree = _cli_source_tree()

    def guards_on_atom(test: ast.expr) -> bool:
        """True for ``framework == "atom"`` and for ``state.framework == "atom"``."""
        for cmp in ast.walk(test):
            if not isinstance(cmp, ast.Compare):
                continue
            left = cmp.left
            name = left.id if isinstance(left, ast.Name) else (left.attr if isinstance(left, ast.Attribute) else "")
            if name != "framework":
                continue
            if any(isinstance(c, ast.Constant) and c.value == "atom" for c in cmp.comparators):
                return True
        return False

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_apply_atom_auto_tighten"
    ]
    guarded_ifs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _calls_auto_tighten(ast, node) and guards_on_atom(node.test)
    ]
    assert len(calls) == len(guarded_ifs), (
        f"{len(calls)} call site(s) but only {len(guarded_ifs)} behind a framework == 'atom' test; "
        "an unguarded call would reach every framework"
    )


def test_resume_applies_the_default_too():
    """A resumed atom session must not silently fall back to GEAK.

    ``KERNEL_OPT_BACKEND_ORDER`` lives in the process environment, not in the
    session, so it is gone in the new process. The example documents
    ``--resume-from`` as the crash-recovery path and tells the operator to leave
    the variable unset, so a resume that skips the default hands them GEAK --
    without even the warning, which lives in the same skipped function.
    """
    ast, tree = _cli_source_tree()

    resume_ifs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(sub, ast.Attribute) and sub.attr == "resume_from" for sub in ast.walk(node.test))
        and (node.body or node.orelse)
    ]
    assert resume_ifs, "no `if args.resume_from:` branch found; this test needs updating"

    reached = [
        node
        for node in resume_ifs
        if any(_calls_auto_tighten(ast, stmt) for stmt in node.body)
        and any(_calls_auto_tighten(ast, stmt) for stmt in node.orelse)
    ]
    assert reached, (
        "_apply_atom_auto_tighten is applied on only one side of `if args.resume_from:`; "
        "a resumed atom session would use a different kernel backend than the launch did"
    )
