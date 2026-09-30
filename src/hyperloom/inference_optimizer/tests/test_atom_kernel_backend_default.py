# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--framework atom`` takes the same kernel-backend default as every other framework.

atom used to be filled in with ``forge`` when the operator named no backend, on
the grounds that GEAK could not resolve a rewrite seam there. GEAK reaches atom
now, so the framework carries no backend rule of its own: an unset
``KERNEL_OPT_BACKEND_ORDER`` resolves through
``_DEFAULT_KERNEL_PHASE_BACKEND_ORDER`` like sglang and vllm, and forge is the
exact-match opt-in it is everywhere else.

What survives from the retired default is the atom-only shape around it: the
``--nodes>=2`` fail-fast guard (IR-8), and the requirement that every call site
stays behind a framework test so no other framework reaches it.
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


@pytest.fixture(autouse=True)
def _restore_kernel_backend_order():
    """Restore direct production writes that monkeypatch does not track."""
    original = os.environ.get(_KEY)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(_KEY, None)
        else:
            os.environ[_KEY] = original


def test_an_unset_backend_is_left_for_the_shared_default(monkeypatch):
    """Writing anything here would make atom resolve differently from every other framework."""
    monkeypatch.delenv(_KEY, raising=False)

    _apply_atom_auto_tighten(_args())

    assert _KEY not in os.environ


@pytest.mark.parametrize("value", ["", "   ", "geak", "GEAK", "forge", "FORGE", "forge,geak"])
def test_a_named_backend_survives_verbatim(monkeypatch, value):
    """Every spelling reaches the shared resolver unedited, including the blank ones."""
    monkeypatch.setenv(_KEY, value)

    _apply_atom_auto_tighten(_args())

    assert os.environ[_KEY] == value


def test_the_shared_default_puts_atom_on_geak(monkeypatch):
    """The observable end of removing the special case, read off the shared resolver."""
    from hyperloom.orchestrator.kernel.request_handlers import _raw_kernel_backend_order

    monkeypatch.delenv(_KEY, raising=False)

    _apply_atom_auto_tighten(_args())

    assert _raw_kernel_backend_order() == ["geak"]


def test_multi_node_still_fails_fast(monkeypatch):
    """IR-8: atom multi-node TP wiring is deferred, so ``--nodes>=2`` exits rather than launching."""
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
    """The IR-8 exit is atom-only, and only the call sites enforce that.

    The behavioural tests above call this function directly, so none of them
    would notice a guard being widened or dropped. Read it out of the source
    instead: each call has to sit under a test that the framework is atom, or a
    multi-node sglang/vllm launch would exit 2.
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


def test_resume_reaches_the_guard_too():
    """A resumed atom session gets the same IR-8 check the launch did.

    ``--resume-from`` is the documented crash-recovery path, and it re-reads the
    CLI arguments in a fresh process. A resume that skips this function would let
    a ``--nodes>=2`` atom session start the very configuration the launch refused.
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
        "a resumed atom session would skip the IR-8 multi-node guard"
    )
