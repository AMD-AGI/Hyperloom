# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure the white-box metrics that ``scripts/code_metrics.py`` ratchets.

Each collector runs one pinned tool over an explicit file list and turns its output
into :class:`Finding` rows keyed by ``(metric, path, unit)``. A unit is a qualified
function or class name (``Class.method``) for function- and class-level metrics and
``<module>`` for file-level ones, so moving code inside a file does not change a key.

Only values over the threshold come back: the gate never needs the value of a unit
that is within its limit.

Tools run from an empty temporary directory with absolute paths, so no in-repo tool
config (``[tool.complexipy]``, ``[tool.vulture]``, ``.jscpd.json``, ruff per-file
ignores) can narrow what they measure. Ruff also runs with ``--isolated`` and
``--ignore-noqa`` and complexipy with ``--no-ignore``; jscpd, which has no such
switch, reads copies with its ``jscpd:ignore-start`` markers defused line for line.
Vulture alone honours ``# noqa``: Ruff owns that marker (``# noqa: F401`` is the
sanctioned form for a side-effect import or a re-export, and RUF100 flags one that
suppresses nothing), so it is not a way around this gate.

Scope entries are passed to git as literal paths (``--literal-pathspecs``), so a
pathspec such as ``:(exclude)x.py`` in ``roots`` names a path, it does not exclude one.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

MODULE = "<module>"
#: Metric names used by more than one collector.
COGNITIVE, FUNCTION_LINES, MODULE_LINES = "cognitive-complexity", "function-lines", "module-lines"
_JSON, _CONFIG = "json", "--config"


@dataclasses.dataclass(frozen=True)
class Metric:
    """One gated dimension; a higher value is always worse."""

    name: str
    label: str


METRICS = {
    m.name: m
    for m in (
        Metric("cyclomatic-complexity", "Cyclomatic complexity"),
        Metric(COGNITIVE, "Cognitive complexity"),
        Metric(FUNCTION_LINES, "Function length (lines)"),
        Metric("nested-blocks", "Nested block depth"),
        Metric(MODULE_LINES, "Module length (lines)"),
        Metric("duplicated-lines", "Duplicated lines per file"),
        Metric("dead-code", "Dead code (vulture)"),
    )
}

#: Ruff rule -> (metric, the ruff setting that carries its threshold).
RUFF_RULES = {
    "C901": ("cyclomatic-complexity", "lint.mccabe.max-complexity"),
    "PLR1702": ("nested-blocks", "lint.pylint.max-nested-blocks"),
}

_RUFF_VALUE = re.compile(r"\((\d+) > \d+\)")
_VULTURE_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): (?P<message>.+) \(\d+% confidence")
_QUOTED = re.compile(r"'([^']+)'")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFS = (*_FUNCTIONS, ast.ClassDef)
_THREADS = "4"
#: Suppression markers each tool honours, and what they are rewritten to (same line count).
_JSCPD_MARKER = (re.compile(rb"jscpd:ignore", re.IGNORECASE), b"jscpd-defused")


class ToolError(RuntimeError):
    """A tool is missing, at the wrong version, or produced output the gate cannot read."""


@dataclasses.dataclass(frozen=True)
class Finding:
    metric: str
    path: str
    unit: str
    value: int
    line: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.metric, self.path, self.unit)


def worse(metric: str, value: float, reference: float) -> bool:
    """True when ``value`` is worse than ``reference`` for ``metric`` (every metric: higher)."""
    if metric not in METRICS:
        raise ToolError(f"unknown metric {metric}")
    return value > reference


def violates(metric: str, value: float, threshold: float) -> bool:
    return worse(metric, value, threshold)


@dataclasses.dataclass(frozen=True)
class Def:
    """One class or function: its ``def``/``class`` line, last line and qualified name."""

    start: int
    end: int
    name: str
    is_function: bool


class Units:
    """Parsed files and the qualified names of their classes and functions, on demand."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._trees: dict[str, ast.Module] = {}
        self._defs: dict[str, list[Def]] = {}

    def tree(self, path: str) -> ast.Module:
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse((self._root / path).read_bytes(), filename=path)
            except (SyntaxError, ValueError) as exc:
                raise ToolError(f"cannot parse {path}: {exc}") from exc
        return self._trees[path]

    def defs(self, path: str) -> list[Def]:
        if path not in self._defs:
            self._defs[path] = list(_walk_defs(self.tree(path), ""))
        return self._defs[path]

    def owner(self, path: str, line: int) -> str:
        """The def starting on ``line``, else the innermost def enclosing it."""
        defs = self.defs(path)
        exact = [d.name for d in defs if d.start == line]
        if exact:
            return exact[-1]
        enclosing = [(d.start, d.name) for d in defs if d.start <= line <= d.end]
        return max(enclosing)[1] if enclosing else MODULE

    def line_of(self, path: str, unit: str) -> int:
        """First line of the def named ``unit`` (or ending in ``.unit``), else 1."""
        for d in self.defs(path):
            if d.name == unit or d.name.endswith("." + unit):
                return d.start
        return 1


def _walk_defs(node: ast.AST, prefix: str) -> Iterator[Def]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFS):
            name = prefix + child.name
            yield Def(child.lineno, child.end_lineno or child.lineno, name, isinstance(child, _FUNCTIONS))
            yield from _walk_defs(child, name + ".")
        else:
            yield from _walk_defs(child, prefix)


def list_files(root: Path, roots: Iterable[str], exclude: Iterable[str]) -> tuple[list[str], list[str]]:
    """(production, tests): the tracked and untracked-but-not-ignored ``.py`` files under ``roots``."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "--literal-pathspecs",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *roots,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    excluded = tuple(exclude)
    production: list[str] = []
    tests: list[str] = []
    for path in sorted(set(out.split("\0"))):
        if not path.endswith(".py") or is_excluded(path, excluded) or not (root / path).is_file():
            continue
        (tests if "tests" in Path(path).parts[:-1] else production).append(path)
    return production, tests


def is_excluded(path: str, exclude: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in exclude)


def defused_copy(root: Path, files: list[str], marker: tuple[re.Pattern[bytes], bytes], work: str) -> Path:
    """Copies of ``files`` under ``work`` with every suppression ``marker`` rewritten, line for line.

    The copy keeps each file's relative path and line numbers, so findings map back to
    the real file by path alone.
    """
    pattern, replacement = marker
    tree = Path(work) / "tree"
    for path in files:
        target = tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pattern.sub(replacement, (root / path).read_bytes()))
    return tree


def _run(argv: list[str], cwd: str, ok: tuple[int, ...] = (0,)) -> str:
    env = {**os.environ, "RAYON_NUM_THREADS": _THREADS, "NO_COLOR": "1"}
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ToolError(f"{argv[0]} is not installed; see the install line in scripts/code_metrics.py") from exc
    if proc.returncode not in ok:
        raise ToolError(f"{argv[0]} exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    return proc.stdout


def _relative(root: Path, name: str) -> str:
    return Path(name).resolve().relative_to(root.resolve()).as_posix()


def tool_version(argv: list[str]) -> str:
    """The last whitespace-separated token of ``<tool> --version``."""
    with tempfile.TemporaryDirectory() as work:
        words = _run([*argv, "--version"], work).split()
    if not words:
        raise ToolError(f"{argv[0]} --version printed nothing")
    return words[-1]


def ruff_findings(root: Path, files: list[str], thresholds: dict[str, int], units: Units) -> list[Finding]:
    argv = ["ruff", "check", "--isolated", "--no-cache", "--ignore-noqa", "--exit-zero", "--output-format", _JSON]
    argv += ["--preview", _CONFIG, "lint.explicit-preview-rules=true", "--select", ",".join(RUFF_RULES)]
    for metric, setting in RUFF_RULES.values():
        argv += [_CONFIG, f"{setting}={thresholds[metric]}"]
    with tempfile.TemporaryDirectory() as work:
        rows = json.loads(_run([*argv, *(str(root / f) for f in files)], work))
    return parse_ruff(rows, root, units)


def parse_ruff(rows: list[dict], root: Path, units: Units) -> list[Finding]:
    findings = []
    for row in rows:
        match = _RUFF_VALUE.search(row["message"])
        if row["code"] not in RUFF_RULES or match is None:
            raise ToolError(f"unexpected ruff diagnostic: {row['code']} {row['message']}")
        path = _relative(root, row["filename"])
        line = row["location"]["row"]
        metric = RUFF_RULES[row["code"]][0]
        findings.append(Finding(metric, path, units.owner(path, line), int(match.group(1)), line))
    return findings


def complexipy_findings(root: Path, files: list[str], threshold: int, units: Units) -> list[Finding]:
    with tempfile.TemporaryDirectory() as work:
        report = Path(work) / "complexipy.json"
        # No --quiet: complexipy 8.0.1 then exits 1 despite --ignore-complexity.
        argv = ["complexipy", *(str(root / f) for f in files), "--ignore-complexity", "--no-ignore"]
        argv += ["--output-format", _JSON, "--output", str(report), "--cache-dir", str(Path(work) / "cache")]
        _run(argv, work)
        rows = json.loads(report.read_text(encoding="utf-8"))
    return parse_complexipy(rows, root, threshold, units)


def parse_complexipy(rows: list[dict], root: Path, threshold: int, units: Units) -> list[Finding]:
    findings = []
    for row in rows:
        value = int(row["complexity"])
        if not violates(COGNITIVE, value, threshold):
            continue
        path = _relative(root, row["path"])
        unit = row["function_name"].replace("::", ".")
        findings.append(Finding(COGNITIVE, path, unit, value, units.line_of(path, unit)))
    return findings


def module_lines(root: Path, path: str) -> int:
    """Physical lines of ``path``."""
    return len((root / path).read_bytes().splitlines())


def module_line_findings(root: Path, files: list[str], threshold: int) -> list[Finding]:
    findings = []
    for path in files:
        lines = module_lines(root, path)
        if violates(MODULE_LINES, lines, threshold):
            findings.append(Finding(MODULE_LINES, path, MODULE, lines, 1))
    return findings


def function_line_findings(files: list[str], threshold: int, units: Units) -> list[Finding]:
    """Functions longer than ``threshold`` physical lines.

    A function counts from its ``def`` line through its last line: decorators are not
    counted, blank, comment and docstring lines are. A nested function is a unit of its
    own, measured the same way; its lines are also part of the function around it, since
    they are lines a reader of that function scrolls past.
    """
    findings = []
    for path in files:
        for d in units.defs(path):
            length = d.end - d.start + 1
            if d.is_function and violates(FUNCTION_LINES, length, threshold):
                findings.append(Finding(FUNCTION_LINES, path, d.name, length, d.start))
    return findings


def vulture_findings(root: Path, files: list[str], min_confidence: int, units: Units) -> list[Finding]:
    argv = ["vulture", _CONFIG, os.devnull, "--min-confidence", str(min_confidence)]
    # The real files, not a copy: vulture honours Ruff's noqa markers (see the module doc).
    with tempfile.TemporaryDirectory() as work:
        # vulture exits 3 when it found dead code; 1 and 2 are input/usage errors.
        text = _run([*argv, *(str(root / f) for f in files)], work, ok=(0, 3))
    return parse_vulture(text, root, units)


def parse_vulture(text: str, root: Path, units: Units) -> list[Finding]:
    findings = []
    for raw in text.splitlines():
        match = _VULTURE_LINE.match(raw)
        if match is None:
            raise ToolError(f"unexpected vulture output: {raw}")
        path = _relative(root, match["path"])
        line = int(match["line"])
        quoted = _QUOTED.search(match["message"])
        name = quoted.group(1) if quoted else match["message"]
        findings.append(Finding("dead-code", path, f"{units.owner(path, line)}:{name}", 1, line))
    return findings


def jscpd_command() -> list[str]:
    """``jscpd`` on PATH, or the command in ``CODE_METRICS_JSCPD`` (e.g. ``npx --yes jscpd@5.4.1``)."""
    return shlex.split(os.environ.get("CODE_METRICS_JSCPD", "jscpd"))


def jscpd_findings(root: Path, files: list[str], min_tokens: int, min_lines: int) -> list[Finding]:
    argv = [*jscpd_command(), "--min-tokens", str(min_tokens), "--min-lines", str(min_lines), "--format", "python"]
    argv += ["--reporters", _JSON, "--absolute", "--silent", "--no-tips", "--workers", _THREADS]
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as copy:
        tree = defused_copy(root, files, _JSCPD_MARKER, copy)
        _run([*argv, "--output", work, *(str(tree / f) for f in files)], work)
        data = json.loads((Path(work) / "jscpd-report.json").read_text(encoding="utf-8"))
        return parse_jscpd(data, tree)


def parse_jscpd(data: dict, root: Path) -> list[Finding]:
    """One finding per file: the number of its lines that belong to any clone."""
    lines: dict[str, set[int]] = {}
    for clone in data["duplicates"]:
        for side in (clone["firstFile"], clone["secondFile"]):
            lines.setdefault(_relative(root, side["name"]), set()).update(range(side["start"], side["end"] + 1))
    return [
        Finding("duplicated-lines", path, MODULE, len(numbers), min(numbers)) for path, numbers in sorted(lines.items())
    ]


def worst_per_unit(findings: Iterable[Finding]) -> dict[tuple[str, str, str], Finding]:
    """Collapse findings that share a key (e.g. a property getter and setter) to the worst one."""
    out: dict[tuple[str, str, str], Finding] = {}
    for finding in findings:
        kept = out.get(finding.key)
        if kept is None or worse(finding.metric, finding.value, kept.value):
            out[finding.key] = finding
    return out
