# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structural contracts for the KernelForge knowledge domains."""

from __future__ import annotations

import ast
from pathlib import Path

import kernelforge

_PACKAGE_ROOT = Path(kernelforge.__file__).resolve().parent
_KNOWLEDGE_ROOT = _PACKAGE_ROOT / "knowledge"


def _imports(root: Path) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((path, node.module))
    return imports


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_PACKAGE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("kernelforge", *parts))


def _module_import_graph(root: Path) -> dict[str, set[str]]:
    paths = sorted(root.rglob("*.py"))
    modules = {_module_name(path) for path in paths}
    graph = {module: set() for module in modules}
    for path in paths:
        source = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                candidates = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            else:
                continue
            graph[source].update(candidate for candidate in candidates if candidate in modules)
    return graph


def _find_import_cycle(graph: dict[str, set[str]]) -> list[str]:
    visited: set[str] = set()
    active: list[str] = []

    def visit(module: str) -> list[str]:
        if module in active:
            start = active.index(module)
            return [*active[start:], module]
        if module in visited:
            return []
        active.append(module)
        for imported in sorted(graph[module]):
            cycle = visit(imported)
            if cycle:
                return cycle
        active.pop()
        visited.add(module)
        return []

    for module in sorted(graph):
        cycle = visit(module)
        if cycle:
            return cycle
    return []


def test_data_directory_is_retired() -> None:
    assert not (_PACKAGE_ROOT / "data").exists()


def test_knowledge_root_contains_no_flat_implementation_modules() -> None:
    assert {path.name for path in _KNOWLEDGE_ROOT.glob("*.py")} == {"__init__.py"}


def test_kb_store_does_not_depend_on_campaigns() -> None:
    forbidden = ("kernelforge.loop", "kernelforge.rewrite_by_flydsl")
    violations = [
        f"{path.relative_to(_PACKAGE_ROOT)} -> {module}"
        for path, module in _imports(_KNOWLEDGE_ROOT / "kb_store")
        if module.startswith(forbidden)
    ]
    assert not violations, violations


def test_kb_store_import_graph_is_acyclic() -> None:
    cycle = _find_import_cycle(_module_import_graph(_KNOWLEDGE_ROOT / "kb_store"))
    assert not cycle, " -> ".join(cycle)


def test_local_wiki_and_pr_knowledge_do_not_depend_on_kb_store() -> None:
    violations = [
        f"{path.relative_to(_PACKAGE_ROOT)} -> {module}"
        for domain in ("local_wiki", "pr_knowledge")
        for path, module in _imports(_KNOWLEDGE_ROOT / domain)
        if module.startswith("kernelforge.knowledge.kb_store")
    ]
    assert not violations, violations
