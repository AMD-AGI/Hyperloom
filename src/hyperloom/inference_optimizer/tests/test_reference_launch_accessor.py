# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every reader of the session's reference launch goes through ``resolve_reference_launch``."""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
_FIELDS = {"reference_server_args", "reference_envs", "reference_launch_controls", "reference_model"}
# The owner of the fields, and the accessor that reads them for everyone else.
_ALLOWED = {
    Path("hyperloom/orchestrator/state/shared_state.py"),
    Path("hyperloom/orchestrator/actions/executors/_workload_envs.py"),
}


def _field_reads(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FIELDS:
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _FIELDS
        ):
            lines.append(node.lineno)
    return lines


def test_only_the_accessor_reads_the_reference_fields():
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        relative = path.relative_to(_SRC)
        if "tests" in relative.parts or relative in _ALLOWED:
            continue
        offenders += [f"{relative}:{line}" for line in _field_reads(ast.parse(path.read_text(encoding="utf-8")))]
    assert offenders == []
