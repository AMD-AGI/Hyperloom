# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A name used but never bound is invisible until the line runs.

Import-time checks and the test suite both pass on it, because the call sites
that reach it are the ones that need a GPU, a server, or a failure to have
happened. Two of these shipped into a run and were catalogued as bad kernels:
the validator raised NameError, the loop wrote "VALIDATE FAILED" against the
recipe, and the attempt was spent.
"""

from __future__ import annotations

import ast
import builtins
import inspect
from pathlib import Path

import kernelforge.fusion.author
import kernelforge.fusion.command
import kernelforge.fusion.discover
import kernelforge.fusion.emit
import kernelforge.fusion.locate
import kernelforge.fusion.validate

MODULES = (
    kernelforge.fusion.author,
    kernelforge.fusion.command,
    kernelforge.fusion.discover,
    kernelforge.fusion.emit,
    kernelforge.fusion.locate,
    kernelforge.fusion.validate,
)


def _bound_at_module_level(tree: ast.Module) -> set[str]:
    """Every name the module itself binds, at any nesting depth.

    Deliberately flat: this is looking for names bound nowhere at all, not for
    names bound in the wrong scope, so collecting them all keeps it from
    reporting a local that a sibling function happens to share a name with.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                bound.update(a.arg for a in args.posonlyargs + args.args + args.kwonlyargs)
                for extra in (args.vararg, args.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
    return bound


def _unbound_reads(source: str) -> set[str]:
    tree = ast.parse(source)
    bound = _bound_at_module_level(tree) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return used - bound


def test_no_module_uses_a_name_it_never_binds() -> None:
    offenders = {}
    for module in MODULES:
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
        missing = _unbound_reads(source)
        if missing:
            offenders[module.__name__] = sorted(missing)

    assert offenders == {}


def test_the_check_catches_a_gap_it_is_meant_to_catch() -> None:
    # Exactly the shape that shipped: called in a branch, imported nowhere.
    source = "def validate():\n    return unreached_fusion_symbols('x', [])\n"

    assert _unbound_reads(source) == {"unreached_fusion_symbols"}
