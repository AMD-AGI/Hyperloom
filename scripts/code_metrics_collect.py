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
``--ignore-noqa`` and complexipy with ``--no-ignore``; vulture and jscpd, which have
no such switch, read copies with their markers (``# noqa``, ``jscpd:ignore-start``)
defused line for line. A suppression comment cannot hide a unit either.

Scope entries are passed to git as literal paths (``--literal-pathspecs``), so a
pathspec such as ``:(exclude)x.py`` in ``roots`` names a path, it does not exclude one.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

MODULE = "<module>"


@dataclasses.dataclass(frozen=True)
class Metric:
    """One gated dimension. ``higher_is_better`` flips every comparison (MI)."""

    name: str
    label: str
    higher_is_better: bool = False


METRICS = {
    m.name: m
    for m in (
        Metric("cyclomatic-complexity", "Cyclomatic complexity"),
        Metric("cognitive-complexity", "Cognitive complexity"),
        Metric("statements", "Statements per function"),
        Metric("branches", "Branches per function"),
        Metric("returns", "Returns per function"),
        Metric("arguments", "Arguments"),
        Metric("positional-arguments", "Positional arguments"),
        Metric("locals", "Local variables"),
        Metric("nested-blocks", "Nested block depth"),
        Metric("public-methods", "Public methods per class"),
        Metric("module-lines", "Module length (lines)"),
        Metric("maintainability-index", "Maintainability index", higher_is_better=True),
        Metric("duplicated-lines", "Duplicated lines per file"),
        Metric("dead-code", "Dead code (vulture)"),
    )
}

#: Ruff rule -> (metric, the ruff setting that carries its threshold).
RUFF_RULES = {
    "C901": ("cyclomatic-complexity", "lint.mccabe.max-complexity"),
    "PLR0915": ("statements", "lint.pylint.max-statements"),
    "PLR0912": ("branches", "lint.pylint.max-branches"),
    "PLR0911": ("returns", "lint.pylint.max-returns"),
    "PLR0913": ("arguments", "lint.pylint.max-args"),
    "PLR0917": ("positional-arguments", "lint.pylint.max-positional-args"),
    "PLR0914": ("locals", "lint.pylint.max-locals"),
    "PLR1702": ("nested-blocks", "lint.pylint.max-nested-blocks"),
    "PLR0904": ("public-methods", "lint.pylint.max-public-methods"),
}

_RUFF_VALUE = re.compile(r"\((\d+) > \d+\)")
_VULTURE_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): (?P<message>.+) \(\d+% confidence")
_QUOTED = re.compile(r"'([^']+)'")
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_THREADS = "4"
#: Suppression markers each tool honours, and what they are rewritten to (same line count).
_VULTURE_MARKER = (re.compile(rb"#\s*noqa", re.IGNORECASE), b"#")
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
    """True when ``value`` is worse than ``reference`` for ``metric``."""
    if METRICS[metric].higher_is_better:
        return value < reference
    return value > reference


def violates(metric: str, value: float, threshold: float) -> bool:
    return worse(metric, value, threshold)


class Units:
    """Qualified names of the classes and functions in each file, parsed on demand."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._defs: dict[str, list[tuple[int, int, str]]] = {}

    def _load(self, path: str) -> list[tuple[int, int, str]]:
        if path not in self._defs:
            tree = ast.parse((self._root / path).read_bytes(), filename=path)
            self._defs[path] = list(_walk_defs(tree, ""))
        return self._defs[path]

    def owner(self, path: str, line: int) -> str:
        """The def starting on ``line``, else the innermost def enclosing it."""
        defs = self._load(path)
        exact = [name for start, _end, name in defs if start == line]
        if exact:
            return exact[-1]
        enclosing = [(start, name) for start, end, name in defs if start <= line <= end]
        return max(enclosing)[1] if enclosing else MODULE

    def line_of(self, path: str, unit: str) -> int:
        """First line of the def named ``unit`` (or ending in ``.unit``), else 1."""
        for start, _end, name in self._load(path):
            if name == unit or name.endswith("." + unit):
                return start
        return 1


def _walk_defs(node: ast.AST, prefix: str) -> Iterator[tuple[int, int, str]]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEFS):
            name = prefix + child.name
            yield child.lineno, child.end_lineno or child.lineno, name
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
        if not path.endswith(".py") or _is_excluded(path, excluded) or not (root / path).is_file():
            continue
        (tests if "tests" in Path(path).parts[:-1] else production).append(path)
    return production, tests


def _is_excluded(path: str, exclude: tuple[str, ...]) -> bool:
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
    argv = ["ruff", "check", "--isolated", "--no-cache", "--ignore-noqa", "--exit-zero", "--output-format", "json"]
    argv += ["--preview", "--config", "lint.explicit-preview-rules=true", "--select", ",".join(RUFF_RULES)]
    for metric, setting in RUFF_RULES.values():
        argv += ["--config", f"{setting}={thresholds[metric]}"]
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
        argv += ["--output-format", "json", "--output", str(report), "--cache-dir", str(Path(work) / "cache")]
        _run(argv, work)
        rows = json.loads(report.read_text(encoding="utf-8"))
    return parse_complexipy(rows, root, threshold, units)


def parse_complexipy(rows: list[dict], root: Path, threshold: int, units: Units) -> list[Finding]:
    findings = []
    for row in rows:
        value = int(row["complexity"])
        if not violates("cognitive-complexity", value, threshold):
            continue
        path = _relative(root, row["path"])
        unit = row["function_name"].replace("::", ".")
        findings.append(Finding("cognitive-complexity", path, unit, value, units.line_of(path, unit)))
    return findings


def radon_findings(root: Path, files: list[str], threshold: int) -> list[Finding]:
    with tempfile.TemporaryDirectory() as work:
        data = json.loads(_run(["radon", "mi", "--json", *(str(root / f) for f in files)], work))
    return parse_radon(data, root, threshold)


def parse_radon(data: dict[str, dict], root: Path, threshold: int) -> list[Finding]:
    """MI is recorded rounded up to a whole point.

    radon ranks a file C ("extremely low") at MI <= 9, so with the limit at 10 the
    rounded-up value is under the limit exactly when radon would rank the file C.
    """
    findings = []
    for name, result in data.items():
        if "mi" not in result:
            raise ToolError(f"radon could not measure {name}: {result.get('error')}")
        value = math.ceil(result["mi"])
        if violates("maintainability-index", value, threshold):
            findings.append(Finding("maintainability-index", _relative(root, name), MODULE, value, 1))
    return findings


def module_line_findings(root: Path, files: list[str], threshold: int) -> list[Finding]:
    findings = []
    for path in files:
        lines = len((root / path).read_bytes().splitlines())
        if violates("module-lines", lines, threshold):
            findings.append(Finding("module-lines", path, MODULE, lines, 1))
    return findings


def vulture_findings(root: Path, files: list[str], min_confidence: int, units: Units) -> list[Finding]:
    argv = ["vulture", "--config", os.devnull, "--min-confidence", str(min_confidence)]
    # The copy lives outside the tool's working directory: vulture prints paths under
    # its cwd relative to it.
    with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as copy:
        tree = defused_copy(root, files, _VULTURE_MARKER, copy)
        # vulture exits 3 when it found dead code; 1 and 2 are input/usage errors.
        text = _run([*argv, *(str(tree / f) for f in files)], work, ok=(0, 3))
        return parse_vulture(text, tree, units)


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
    argv += ["--reporters", "json", "--absolute", "--silent", "--no-tips", "--workers", _THREADS]
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
