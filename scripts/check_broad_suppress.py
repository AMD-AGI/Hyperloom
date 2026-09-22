# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reject unjustified contextlib.suppress(Exception) in production code.

``suppress(Exception)`` is ``except Exception: pass`` in another spelling, but ruff's
BLE001 does not see it and no other rule covers it. Without this check, enabling BLE001
only pushes broad swallowing into the form the gate is blind to.

A site that genuinely cannot enumerate its failures marks itself on the ``with`` line::

    with contextlib.suppress(Exception):  # broad-suppress: caller-supplied callback

This mirrors ``# noqa: BLE001`` on the ``except`` form. ``# noqa`` itself cannot be used
here: ruff reports no diagnostic on the line, so RUF100 would strip the marker as unused.

Usage:
    python scripts/check_broad_suppress.py src scripts

Exit code 1 when an unjustified broad suppress is found. Test files are skipped.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from typing import Iterator


_BROAD = {"Exception", "BaseException"}

#: Marks a broad suppress whose failure modes genuinely cannot be enumerated.
_WAIVER = re.compile(r"#\s*broad-suppress:\s*\S")


def _scan_file(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """Yield (lineno, text) for each unjustified broad suppress call in path."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError:
        return
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            if not ast.unparse(call.func).endswith("suppress"):
                continue
            caught = {ast.unparse(a) for a in call.args}
            if not caught & _BROAD:
                continue
            if _WAIVER.search(lines[node.lineno - 1]):
                continue
            yield node.lineno, f"suppress({', '.join(sorted(caught))})"


def _is_test_path(path: pathlib.Path) -> bool:
    """Return True for test files (test_*.py) or files inside a tests/ directory."""
    return path.name.startswith("test_") or "tests" in path.parts


def main(argv: list[str] | None = None) -> int:
    roots = [pathlib.Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not roots:
        roots = [pathlib.Path("src"), pathlib.Path("scripts")]

    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))

    errors: list[str] = []
    for path in files:
        if _is_test_path(path):
            continue
        for lineno, text in _scan_file(path):
            errors.append(f"{path}:{lineno}: broad suppress: {text}")

    for msg in sorted(errors):
        print(msg)

    if errors:
        print(
            f"\nFound {len(errors)} unjustified broad suppress call(s). Narrow to the "
            "exception type(s) the body can raise, let them surface, or -- when the "
            "failures genuinely cannot be enumerated -- append "
            "'# broad-suppress: <reason>' to the with line.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
