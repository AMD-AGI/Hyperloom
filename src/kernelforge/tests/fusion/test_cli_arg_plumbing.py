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
THREADED_OPTIONS = ("server_extra", "pristine_dir", "tp", "block_size", "max_model_len", "max_recipes")


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


def test_recipe_ceiling_caps_at_the_supplied_budget() -> None:
    """A ceiling below the discovered count is what limits the run."""
    assert cli_module._recipe_ceiling(7, 3) == 3


def test_recipe_ceiling_never_exceeds_what_was_discovered() -> None:
    """A ceiling above the discovered count cannot invent recipes."""
    assert cli_module._recipe_ceiling(2, 5) == 2


def test_recipe_ceiling_treats_no_budget_as_uncapped() -> None:
    """An absent ceiling leaves every discovered recipe eligible.

    Zero reaches here when the session is unbounded and no lane share could be
    derived. Reading it as a real cap would silence the lane for a whole run.
    """
    assert cli_module._recipe_ceiling(4, 0) == 4
    assert cli_module._recipe_ceiling(4, -1) == 4


def test_the_recipe_ceiling_reaches_the_loop_config() -> None:
    """The autoloop must build LoopConfig from the ceiling, not the raw count."""
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))
    loop_config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LoopConfig"
    ]
    assert loop_config_calls, "LoopConfig is constructed somewhere in the CLI module"
    ceilings = [
        keyword.value for call in loop_config_calls for keyword in call.keywords if keyword.arg == "max_recipes"
    ]
    assert ceilings, "LoopConfig is given a max_recipes"
    assert all(
        isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "_recipe_ceiling"
        for value in ceilings
    )
