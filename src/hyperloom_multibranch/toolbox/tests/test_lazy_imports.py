# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every ``from hyperloom... import name`` that importing a module does not run still names something that exists.

Such an import is inside a function or under ``if TYPE_CHECKING``. A test that replaces the module it reads from with a
stand-in never sees the name go missing: a sync with main moved two names out of ``tracelens_analysis`` and the
``tracelens`` tool failed on every job while every test stayed green. This reads the names out of the source, so an import
that is only run when a tool runs is checked as well as the ones at the top of a module.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
OWN_PACKAGES = ("hyperloom.", "hyperloom_multibranch.")


def _type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _deferred_imports(path: Path) -> list[tuple[int, str, str]]:
    """``(line, module, name)`` of each absolute ``from <our package> import name`` of ``path`` that does not run when
    the module is imported."""
    found: set[tuple[int, str, str]] = set()
    for scope in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        deferred = isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) or (
            isinstance(scope, ast.If) and _type_checking(scope.test)
        )
        for node in ast.walk(scope) if deferred else ():
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith(OWN_PACKAGES):
                found.update((node.lineno, node.module or "", alias.name) for alias in node.names)
    return sorted(found)


def _unresolved(path: Path) -> list[str]:
    missing = []
    for line, module, name in _deferred_imports(path):
        try:
            target = importlib.import_module(module)
        except ImportError as exc:
            missing.append(f"{path.name}:{line} imports {module}: {exc}")
            continue
        if not hasattr(target, name) and not _is_submodule(module, name):
            missing.append(f"{path.name}:{line} imports {name} from {module}, which has no such name")
    return missing


def _is_submodule(module: str, name: str) -> bool:
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return False
    return True


def test_every_deferred_import_of_the_package_names_something_that_exists() -> None:
    files = [path for path in sorted(PACKAGE.rglob("*.py")) if "tests" not in path.relative_to(PACKAGE).parts]
    assert any(_deferred_imports(path) for path in files), "the scan found no deferred import to check"
    assert [problem for path in files for problem in _unresolved(path)] == []


def test_the_scan_names_an_import_whose_name_is_gone(tmp_path: Path) -> None:
    stale = tmp_path / "stale.py"
    stale.write_text(
        "def run():\n    from hyperloom.orchestrator.trace_analysis.tracelens_analysis import no_such_name\n",
        encoding="utf-8",
    )
    assert _unresolved(stale) == [
        "stale.py:2 imports no_such_name from hyperloom.orchestrator.trace_analysis.tracelens_analysis, "
        "which has no such name"
    ]
