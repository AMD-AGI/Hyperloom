# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The mechanical half of a delivery review: what a change adds to its Python files' comments and docstrings.

Only what the change owns is judged: a comment block that touches a line it added, a docstring that is new or differs
from the base, and the lines it added. Debt the base already carried is left to whoever owns it.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_BLOCK_LINES = 3
MAX_COMMENT_DENSITY = 0.04
_NUMBERS = re.compile(r"\d+\.\d+|\d+x\d+|\bdB\b|TFLOP|\b\d{3,}\b")
_DEBUG = re.compile(r"\bdebug\b|\btmp\b|\bprint\(|\bbreakpoint\(|\bTODO\b|\bFIXME\b|\bXXX\b", re.IGNORECASE)
_CJK = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


@dataclass(frozen=True)
class StyleFinding:
    """One rule a change breaks, and where."""

    rule: str
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(workspace), *args], capture_output=True, text=True)


def _added_lines(workspace: Path, base: str, path: str) -> set[int]:
    lines: set[int] = set()
    for match in _HUNK.finditer(_git(workspace, "diff", "-U0", base, "--", path).stdout):
        start, count = int(match.group(1)), int(match.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def _license_header_end(lines: list[str]) -> int:
    """How many leading comment lines are the license header, which no rule counts."""
    end = 0
    while end < len(lines) and lines[end].lstrip().startswith("#"):
        end += 1
    header = "\n".join(lines[:end])
    return end if "SPDX" in header or "Copyright" in header else 0


def _docstrings(source: str) -> dict[str, tuple[str, int]]:
    """Every docstring in ``source`` by the dotted name of what it documents, with its line."""
    tree = ast.parse(source)
    found = {}
    doc = ast.get_docstring(tree, clean=False)
    if doc is not None:
        found["<module>"] = (doc, tree.body[0].lineno)
    pending = [(tree, "")]
    while pending:
        node, prefix = pending.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                doc = ast.get_docstring(child, clean=False)
                if doc is not None:
                    found[name] = (doc, child.body[0].lineno)
                pending.append((child, f"{name}."))
    return found


def _docstring_findings(path: str, source: str, base_source: str | None) -> list[StyleFinding]:
    base_docs = _docstrings(base_source) if base_source is not None else {}
    findings = []
    for name, (doc, line) in _docstrings(source).items():
        if base_docs.get(name, (None,))[0] == doc:
            continue
        length = doc.strip("\n").count("\n") + 1
        if length > MAX_BLOCK_LINES:
            findings.append(StyleFinding("block", path, line, f"docstring of {name} has {length} lines"))
        findings += [
            StyleFinding("number", path, line, f"docstring of {name} quotes {hit!r}")
            for hit in sorted(set(_NUMBERS.findall(doc)))
        ]
    return findings


def _comment_findings(path: str, lines: list[str], added: set[int], header: int) -> list[StyleFinding]:
    findings = []
    is_comment = [line.lstrip().startswith("#") for line in lines]
    start = header
    while start < len(lines):
        if not is_comment[start]:
            start += 1
            continue
        end = start
        while end < len(lines) and is_comment[end]:
            end += 1
        if set(range(start + 1, end + 1)) & added:
            if end - start > MAX_BLOCK_LINES:
                findings.append(StyleFinding("block", path, start + 1, f"comment block of {end - start} lines"))
            findings += [
                StyleFinding("number", path, index + 1, f"comment quotes {hit!r}")
                for index in range(start, end)
                for hit in _NUMBERS.findall(lines[index])
            ]
        start = end
    return findings


def _line_findings(path: str, lines: list[str], scope: list[int]) -> list[StyleFinding]:
    findings = []
    comments = sum(lines[number - 1].lstrip().startswith("#") for number in scope)
    if scope and comments / len(scope) > MAX_COMMENT_DENSITY:
        findings.append(
            StyleFinding(
                "density",
                path,
                scope[0],
                f"{comments} of the {len(scope)} added lines are comments, over {MAX_COMMENT_DENSITY:.0%}",
            )
        )
    for number in scope:
        text = lines[number - 1].strip()
        if _DEBUG.search(text):
            findings.append(StyleFinding("debug", path, number, text[:100]))
        if _CJK.search(text):
            findings.append(StyleFinding("language", path, number, text[:100]))
    return findings


def style_findings(workspace: str | Path, base: str, paths: list[str]) -> list[StyleFinding]:
    """What the working tree adds since ``base`` that breaks the rules, over the Python files among ``paths``."""
    workspace = Path(workspace)
    findings: list[StyleFinding] = []
    for path in paths:
        file = workspace / path
        if not path.endswith(".py") or not file.is_file():
            continue
        source = file.read_text(encoding="utf-8")
        shown = _git(workspace, "show", f"{base}:{path}")
        base_source = shown.stdout if shown.returncode == 0 else None
        lines = source.splitlines()
        header = _license_header_end(lines)
        added = set(range(1, len(lines) + 1)) if base_source is None else _added_lines(workspace, base, path)
        added -= set(range(1, header + 1))
        scope = sorted(number for number in added if number <= len(lines))
        findings += _docstring_findings(path, source, base_source)
        findings += _comment_findings(path, lines, added, header)
        findings += _line_findings(path, lines, scope)
    return findings
