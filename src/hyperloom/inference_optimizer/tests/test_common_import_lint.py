# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Import-lint guard for ``hyperloom.common`` ("防环规则").

``hyperloom.common`` is the zero-dependency shared library: it may import only
the stdlib (plus ``httpx`` for the future ``llm`` submodule) and must NEVER
import a first-party package (``inference_optimizer`` / ``orchestrator`` /
``agents`` / ``ci`` / …). This keeps the dependency graph acyclic so any package
can safely depend on ``common``. This test statically parses every module under
``hyperloom.common`` and fails if a forbidden import creeps in.
"""

from __future__ import annotations

import ast
from pathlib import Path

import hyperloom.common

# First-party top-level module names that ``common`` must never import.
_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "inference_optimizer",
        "orchestrator",
        "robustness_agent",
        "framework_agent",
        "critic",
        "critic_agent",
        "kernel_agent",
        "quantization_agent",
        "ci",
    }
)

# Third-party packages ``common`` is explicitly allowed to depend on.
_ALLOWED_THIRD_PARTY = frozenset({"httpx"})


def _common_py_files() -> list[Path]:
    root = Path(hyperloom.common.__file__).resolve().parent
    return sorted(root.rglob("*.py"))


def _imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) stays inside hyperloom.common.
            if node.level and node.level > 0:
                names.append("hyperloom.common")
            elif node.module:
                names.append(node.module)
    return names


def test_common_modules_exist():
    files = _common_py_files()
    assert files, "expected at least hyperloom/common/__init__.py to exist"


def test_common_has_no_first_party_imports():
    offenders: list[str] = []
    for path in _common_py_files():
        for module in _imported_module_names(path):
            top = module.split(".")[0]
            if top in _FORBIDDEN_TOP_LEVEL:
                offenders.append(f"{path.name}: import {module}")
            elif top == "hyperloom":
                parts = module.split(".")
                # Only hyperloom.common(.*) is allowed; any other hyperloom
                # subpackage would be a first-party dependency / cycle risk.
                if len(parts) >= 2 and parts[1] != "common":
                    offenders.append(f"{path.name}: import {module}")

    assert not offenders, (
        "hyperloom.common must not import first-party packages "
        f"(tree-reform.MD §7): {offenders}"
    )
