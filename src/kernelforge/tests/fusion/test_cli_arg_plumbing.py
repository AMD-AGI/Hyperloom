# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A CLI option is only wired once every frame between it and its use can see it.

``--server-extra`` reaches the serving smoke through several frames, and a gap in
any one of them is a ``NameError`` raised from inside the gate -- which the loop
records as a failed authoring attempt and spends its whole budget retrying, so
the miswiring reads as a bad kernel rather than a bad call.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import kernelforge.fusion.command as cli_module

CLI_SOURCE = Path(inspect.getfile(cli_module))

# Options that travel from the command down into a nested helper.
THREADED_OPTIONS = ("server_extra", "pristine_dir", "tp", "block_size", "max_model_len")


def _functions_missing_binding(tree: ast.Module, name: str) -> list[str]:
    """Names of functions that read ``name`` without it being bound in any scope."""
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    missing = []
    for fn in functions:
        reads = any(
            isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load) for node in ast.walk(fn)
        )
        if not reads:
            continue
        enclosing = [fn] + [
            outer
            for outer in functions
            if outer is not fn and outer.lineno < fn.lineno and (outer.end_lineno or 0) >= (fn.end_lineno or 0)
        ]
        bound = False
        for scope in enclosing:
            args = scope.args
            params = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
            if name in params:
                bound = True
                break
            if any(
                isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
                for node in ast.walk(scope)
            ):
                bound = True
                break
        if not bound:
            missing.append(fn.name)
    return missing


def test_threaded_options_are_bound_in_every_frame_that_reads_them() -> None:
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))

    for option in THREADED_OPTIONS:
        assert _functions_missing_binding(tree, option) == []


def test_the_check_catches_a_gap_it_is_meant_to_catch() -> None:
    # Same shape as the real bug: the inner frame reads what only the command defines.
    tree = ast.parse(
        "def command(server_extra=''):\n"
        "    helper()\n"
        "def helper():\n"
        "    return serving_smoke(server_extra=server_extra)\n"
    )

    assert _functions_missing_binding(tree, "server_extra") == ["helper"]


def test_the_serving_gate_accepts_the_serving_args() -> None:
    params = inspect.signature(cli_module.apply_serving_gate).parameters

    assert "server_extra" in params
    assert "pristine_dir" in params
    assert "tp" in params
    assert "block_size" in params
    assert "max_model_len" in params


def test_pristine_snapshot_is_threaded_through_the_autoloop() -> None:
    """The snapshot must cross both calls between ``run`` and the serving gate."""
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))

    def _call_keywords(callee: str) -> list[set[str]]:
        return [
            {keyword.arg for keyword in node.keywords if keyword.arg}
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee
        ]

    assert any("pristine_dir" in names for names in _call_keywords("_run_fusion_autoloop"))
    assert any("pristine_dir" in names for names in _call_keywords("apply_serving_gate"))
