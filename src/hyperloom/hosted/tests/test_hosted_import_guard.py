# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may import hosted code: nothing on main reaches a hosted implementation or e2b except their own owners."""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]

# Hosted implementations, and dev-branch-only packages: nothing on main may reach them except their own owners.
_HOSTED = (
    "hyperloom.hosted",
    "hyperloom.sandbox",
    "hyperloom.measure.hosted",
    "hyperloom.measure.hosted_copy",
    "hyperloom.measure.hosted_log",
    "hyperloom.measure.hosted_logs",
    "hyperloom.measure.jit.publish",
    "hyperloom.measure.jit.serve",
    "hyperloom.measure.sandboxes",
    "hyperloom.measure.local",
    "hyperloom.measure.local_outcome",
    "hyperloom.measure.local_templates",
    "hyperloom.measure.runner",
    "hyperloom_multibranch",
)
_E2B_OWNER = "hyperloom/sandbox/e2b.py"


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
            found += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return found


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _hosted_import_violations(rel: str, source: str) -> list[str]:
    bad = []
    for line, module in _imports(ast.parse(source)):
        if _under(module, _HOSTED) and not _under(rel.removesuffix(".py").replace("/", "."), _HOSTED):
            bad.append(f"{rel}:{line}: imports hosted implementation {module}")
        if module.split(".")[0] == "e2b" and rel != _E2B_OWNER:
            bad.append(f"{rel}:{line}: imports e2b outside {_E2B_OWNER}")
    return bad


def _main_modules() -> list[Path]:
    return [p for p in _SRC.rglob("*.py") if "tests" not in p.relative_to(_SRC).parts and ".venv" not in p.parts]


def test_no_module_imports_a_hosted_implementation_outside_its_owners() -> None:
    problems = [m for p in _main_modules() for m in _hosted_import_violations(str(p.relative_to(_SRC)), p.read_text())]
    assert problems == []


def test_the_guard_names_the_hosted_module_imported_even_inside_a_function() -> None:
    bad = _hosted_import_violations("hyperloom/orchestrator/x.py", "def f():\n    import hyperloom.measure.hosted\n")
    assert bad == ["hyperloom/orchestrator/x.py:2: imports hosted implementation hyperloom.measure.hosted"]


def test_the_guard_lets_the_owners_import_hosted_code() -> None:
    assert _hosted_import_violations("hyperloom/hosted/entry.py", "import hyperloom.sandbox.e2b") == []
    assert (
        _hosted_import_violations(
            "hyperloom/measure/runner/native/session_engine.py", "import hyperloom.measure.runner"
        )
        == []
    )


def test_only_the_e2b_module_may_import_e2b() -> None:
    assert _hosted_import_violations("hyperloom/sandbox/e2b.py", "import e2b") == []
    assert _hosted_import_violations("hyperloom/sandbox/local.py", "import e2b") == [
        "hyperloom/sandbox/local.py:1: imports e2b outside hyperloom/sandbox/e2b.py"
    ]
