# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture guard: no module re-aliases the canonical timestamp helper."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PRUNED_DIR_NAMES = frozenset({"__pycache__", "build", "dist", "tests"})


def _find_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()

pytestmark = pytest.mark.skipif(
    _REPO_ROOT is None,
    reason="guard needs the source checkout (pyproject.toml + src/)",
)


def _bare_now_iso_aliases(text: str, path: str) -> list[str]:
    """Module-level ``X = now_iso`` bindings, which shadow the canonical name.

    A ``functools.partial(now_iso, ...)`` binding is a distinct precision and is
    not reported.
    """
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return []
    return [
        f"{path}:{node.lineno}: {target.id} = {node.value.id}"
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "now_iso"
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


def test_no_module_aliases_now_iso() -> None:
    assert _REPO_ROOT is not None
    found: list[str] = []
    for path in sorted((_REPO_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(_REPO_ROOT)
        if any(part in _PRUNED_DIR_NAMES for part in relative.parts) or relative.name.startswith("test_"):
            continue
        found.extend(_bare_now_iso_aliases(path.read_text(encoding="utf-8"), relative.as_posix()))
    if found:
        pytest.fail(
            "import now_iso from hyperloom.common.timeutil instead of aliasing it:\n  " + "\n  ".join(found),
            pytrace=False,
        )
