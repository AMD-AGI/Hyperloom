"""Report missing/incomplete docstrings for the given Python files.

Usage:
    python check_files.py <file1> [<file2> ...]

Mirrors the heuristics used by docstring_inventory.py so a worker can confirm
its assigned files reach zero gaps.
"""
from __future__ import annotations

import ast
import sys


def _is_incomplete(doc, node):
    if doc is None:
        return True, "missing"
    s = doc.strip()
    if not s:
        return True, "empty"
    if len(s) < 8:
        return True, "too-short"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs
                 if x.arg not in ("self", "cls")]
        has_params = bool(names) or a.vararg or a.kwarg
        low = s.lower()
        if has_params and not any(k in low for k in
                                  ("args:", "arguments:", "parameters:", ":param")):
            return True, "no-args-section"
        returns_value = any(isinstance(n, ast.Return) and n.value is not None
                            for n in ast.walk(node))
        if returns_value and not any(k in low for k in
                                     ("returns:", "return:", ":return", "yields:")):
            return True, "no-returns-section"
    return False, ""


def scan(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [("error", "<parse>", 0, str(exc))]
    out = []
    if ast.get_docstring(tree) is None:
        out.append(("module", "<module>", 1, "missing"))

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def _check(self, node, kind):
            prob, reason = _is_incomplete(ast.get_docstring(node), node)
            if prob:
                out.append((kind, ".".join(self.stack + [node.name]),
                            node.lineno, reason))

        def visit_ClassDef(self, node):
            self._check(node, "class")
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self._check(node, "function")
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    V().visit(tree)
    return out


def main():
    total = 0
    for path in sys.argv[1:]:
        gaps = scan(path)
        total += len(gaps)
        print(f"{len(gaps):4d}  {path}")
        for kind, name, line, reason in gaps:
            print(f"        L{line} {kind} {name} :: {reason}")
    print(f"TOTAL remaining gaps: {total}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
