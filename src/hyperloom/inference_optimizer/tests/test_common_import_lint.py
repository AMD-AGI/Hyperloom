# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Import-lint guard for the two shared layers, ``hyperloom_common`` and ``hyperloom.common``."""

from __future__ import annotations

import ast
from pathlib import Path

import hyperloom.common
import hyperloom_common

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


def _py_files(package: object) -> list[Path]:
    root = Path(package.__file__).resolve().parent  # type: ignore[attr-defined]
    return sorted(p for p in root.rglob("*.py") if "tests" not in p.parts)


def _imported_module_names(path: Path, package_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import stays inside the package it is written in.
            if node.level and node.level > 0:
                names.append(package_name)
            elif node.module:
                names.append(node.module)
    return names


def test_common_modules_exist():
    assert _py_files(hyperloom_common), "expected at least hyperloom_common/__init__.py to exist"
    assert _py_files(hyperloom.common), "expected at least hyperloom/common/__init__.py to exist"


def test_common_imports_nothing_first_party():
    """``hyperloom_common`` is the floor: it may not reach into either shipped package."""
    offenders = [
        f"{path.name}: import {module}"
        for path in _py_files(hyperloom_common)
        for module in _imported_module_names(path, "hyperloom_common")
        if module.split(".")[0] in {"hyperloom", "kernelforge"} | _FORBIDDEN_TOP_LEVEL
    ]
    assert not offenders, f"hyperloom_common must not import first-party packages: {offenders}"


def test_hyperloom_common_imports_only_common():
    offenders: list[str] = []
    for path in _py_files(hyperloom.common):
        for module in _imported_module_names(path, "hyperloom.common"):
            top = module.split(".")[0]
            if top in _FORBIDDEN_TOP_LEVEL or top == "kernelforge":
                offenders.append(f"{path.name}: import {module}")
            elif top == "hyperloom":
                parts = module.split(".")
                # Only hyperloom.common(.*) is allowed; `hyperloom_common` itself is the layer below.
                if len(parts) >= 2 and parts[1] != "common":
                    offenders.append(f"{path.name}: import {module}")

    assert not offenders, f"hyperloom.common must not import first-party packages (tree-reform.MD §7): {offenders}"
