"""Scan a repository for Python modules, classes, and functions that are
missing docstrings or have docstrings that look incomplete.

Usage:
    python docstring_inventory.py <root> [--include-tests]
"""
from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass, field, asdict


@dataclass
class FileReport:
    path: str
    module_missing: bool = False
    missing: list = field(default_factory=list)  # list of (kind, qualname, lineno, reason)


def _is_incomplete(doc: str | None, node) -> tuple[bool, str]:
    """Return (is_problem, reason)."""
    if doc is None:
        return True, "missing"
    stripped = doc.strip()
    if not stripped:
        return True, "empty"
    if len(stripped) < 8:
        return True, "too-short"
    # For functions with params (excluding self/cls), flag if no Args section
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        names = [n for n in names if n not in ("self", "cls")]
        has_params = bool(names) or args.vararg or args.kwarg
        low = stripped.lower()
        mentions_args = any(k in low for k in ("args:", "arguments:", "parameters:", "param ", ":param"))
        if has_params and not mentions_args:
            return True, "no-args-section"
        # Returns: only flag if function clearly returns a value
        returns_value = any(
            isinstance(n, ast.Return) and n.value is not None
            for n in ast.walk(node)
        )
        mentions_return = any(k in low for k in ("returns:", "return:", ":return", "yields:"))
        if returns_value and not mentions_return:
            return True, "no-returns-section"
    return False, ""


def scan_file(path: str) -> FileReport:
    rep = FileReport(path=path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError) as exc:
        rep.missing.append(("error", "<parse>", 0, str(exc)))
        return rep

    if ast.get_docstring(tree) is None:
        rep.module_missing = True
        rep.missing.append(("module", os.path.basename(path), 1, "missing"))

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def _qual(self, name):
            return ".".join(self.stack + [name])

        def _check(self, node, kind):
            # skip private dunder-less helpers? No, include all.
            doc = ast.get_docstring(node)
            problem, reason = _is_incomplete(doc, node)
            if problem:
                rep.missing.append((kind, self._qual(node.name), node.lineno, reason))

        def visit_ClassDef(self, node):
            self._check(node, "class")
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            # skip property setters / dunder __repr__ etc.? keep but allow short
            self._check(node, "function")
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return rep


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    include_tests = "--include-tests" in sys.argv

    reports = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (
            ".git", ".venv", "venv", "__pycache__", "node_modules",
            ".clone_cache", "_audit_artifacts", "build", "dist", ".mypy_cache",
            ".pytest_cache", ".ruff_cache",
        )]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            is_test = ("test" in fn.lower()) or (os.sep + "tests" + os.sep in full)
            if is_test and not include_tests:
                continue
            rep = scan_file(full)
            rep.path = rel
            if rep.missing:
                reports.append(rep)

    # Summary
    total_missing = sum(len(r.missing) for r in reports)
    by_kind = {}
    for r in reports:
        for kind, *_ in r.missing:
            by_kind[kind] = by_kind.get(kind, 0) + 1

    out = {
        "root": os.path.abspath(root),
        "files_with_gaps": len(reports),
        "total_gaps": total_missing,
        "by_kind": by_kind,
        "reports": [asdict(r) for r in reports],
    }
    with open(os.path.join(os.path.dirname(__file__), "docstring_inventory.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"Files with gaps: {len(reports)}")
    print(f"Total gaps: {total_missing}")
    print(f"By kind: {by_kind}")
    # Per-file counts sorted desc
    print("\nTop files by gap count:")
    for r in sorted(reports, key=lambda r: len(r.missing), reverse=True)[:40]:
        print(f"  {len(r.missing):4d}  {r.path}")


if __name__ == "__main__":
    main()
