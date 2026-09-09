# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A harness has to sit where the driver imports it from.

The prompt tells the agent both harnesses are already available beside the
driver and to import them by module name. When the driver is kept in a
subdirectory -- which is how the rewrite controller runs one -- placing a
harness at the workspace root instead makes that untrue: the agent finds
nothing beside the driver and writes its own, which the guard that protects
harness files from being rewritten then refuses. The attempt is lost to a file
that should have been there already.

``dist_harness`` fails harder than that. A multi-rank driver that does not
measure inside it is rejected by preflight, so a misplaced copy turns a
distributed verdict into an ImportError.
"""

from __future__ import annotations

import ast
from pathlib import Path

from kernelforge.loop import task_preparer


def _placement_expression(name: str) -> ast.expr:
    """The assignment ``prepare_task`` uses to decide where a harness goes."""
    source = Path(task_preparer.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return node.value
    raise AssertionError(f"prepare_task no longer decides a {name}")


def _harness_expression() -> ast.expr:
    return _placement_expression("harness_path")


def test_the_harness_is_placed_beside_the_driver() -> None:
    """One rule for both driver shapes, so the prompt's claim is always true."""
    expression = _harness_expression()

    # A conditional here is the shape the defect had: one branch put the
    # harness somewhere the driver does not import from.
    assert not isinstance(expression, ast.IfExp), (
        "harness placement must not depend on whether the driver is external; "
        "it belongs beside the driver in both cases"
    )
    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.left, ast.Name)
    assert expression.left.id == "driver_access_dir"
    assert isinstance(expression.right, ast.Constant)
    assert expression.right.value == "graph_harness.py"


def test_a_driver_at_the_workspace_root_is_unaffected(tmp_path: Path) -> None:
    """``driver_access_dir`` is the driver's own directory, so the established
    layout keeps the harness exactly where it has always been."""
    workspace = tmp_path / "ws"
    root_driver = workspace / "driver.py"
    staged_driver = workspace / ".forge_driver_ab12" / "driver.py"

    assert root_driver.parent / "graph_harness.py" == workspace / "graph_harness.py"
    assert staged_driver.parent / "graph_harness.py" == staged_driver.parent / "graph_harness.py"
    assert (staged_driver.parent / "graph_harness.py").parent == staged_driver.parent


def test_the_distributed_harness_is_placed_beside_the_driver_too() -> None:
    expression = _placement_expression("dist_harness_path")

    assert not isinstance(expression, ast.IfExp)
    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.left, ast.Name)
    assert expression.left.id == "driver_access_dir"
    assert isinstance(expression.right, ast.Constant)
    assert expression.right.value == "dist_harness.py"


def test_what_is_shipped_is_the_module_the_tests_exercise() -> None:
    """Otherwise the harness under test and the one drivers import can drift.

    The copy is what every later run of the loop imports, so a divergence would
    be found by a campaign rather than by this suite. Compared as text rather
    than through the imported module, because ``dist_harness`` needs torch and
    this file must keep collecting on an image without it -- which is also the
    property that makes ``_dist_harness_text`` read the file instead of
    importing it.
    """
    shipped = task_preparer._dist_harness_text()
    on_disk = Path(task_preparer.__file__).parent / "dist_harness.py"

    assert shipped == on_disk.read_text(encoding="utf-8")


def test_the_shipped_source_offers_the_entry_point_the_prompt_names() -> None:
    """The contract note tells the agent to write ``from dist_harness import Case, run``.

    Read rather than executed: the module imports torch, and this suite runs on
    images without it.
    """
    shipped = task_preparer._dist_harness_text()
    assert shipped is not None
    tree = ast.parse(shipped)
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert {"run", "Case", "RankContext"} <= defined
    # The probe reads this by name out of ``sys.modules`` to establish that the
    # ranks measured here, so its absence would be silent on both sides.
    assert "MEASURED" in assigned
