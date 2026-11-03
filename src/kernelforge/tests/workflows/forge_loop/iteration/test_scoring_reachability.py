# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scoring helpers the loop once shipped without a call site: a guard nobody calls protects nothing."""

from __future__ import annotations

import ast

import pytest

from kernelforge.tests.conftest import PACKAGE_ROOT

LOOP = PACKAGE_ROOT / "workflows" / "forge_loop"


def _calls_in_runner(name: str) -> int:
    """How many times ``name`` is called across the loop package, ignoring its own def."""
    trees = [ast.parse(path.read_text()) for path in sorted(LOOP.rglob("*.py"))]
    return sum(
        1
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            or (isinstance(node.func, ast.Name) and node.func.id == name)
        )
    )


@pytest.mark.parametrize(
    "helper",
    ["_promote_best"],
)
def test_scoring_helpers_are_reachable(helper):
    """A guard nobody calls protects nothing."""
    assert _calls_in_runner(helper) > 0, f"{helper} has no call site"
