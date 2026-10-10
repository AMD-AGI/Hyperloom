# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Absolute and diff-only checks that ``scripts/code_metrics.py`` runs next to the ratchet.

None of these has a baseline:

* **Comments** (diff-only, added lines of ``.py`` files): an added run of more than
  :data:`MAX_COMMENT_BLOCK` consecutive full-line ``#`` comments, or an added comment
  that points at a pull request or issue (``#123`` or longer, ``PR 1234``, ``issue #12``,
  a ``/pull/`` link) or narrates an incident (a dated incident or outage, a postmortem,
  "after the outage"). The history belongs in the commit message and the PR; a comment
  says what the code does and why. A comment starting with ``TODO`` and the line after it
  may carry an issue link, which is what Ruff's ``TD003`` asks for. A comment line whose
  text the change removed elsewhere (moved, re-indented, a renamed file) is not added.
* **English only** (whole tree): no git-tracked text file carries a CJK character
  (:data:`CJK`); on a pull request the title, the body and every commit message are
  held to the same rule.
* **Production code does not import test code** (whole tree): no module under the
  gate's roots that is not itself a test imports a ``tests`` package or a ``test_*``
  module.
* **Repeated literals** (diff-only, production modules the change touches): the change
  may not add an occurrence of a string or bytes value (3+ characters) or a number (other
  than 0, 1, -1 and 2) that takes it from fewer than :data:`MAX_LITERAL_REPEATS` to that
  many or more in the module. Keys (dict-literal keys, ``x["k"]``, the key of ``.get`` /
  ``.pop`` / ``.setdefault`` and of an ``in`` test), docstrings, f-string text,
  annotations and ``__all__`` do not count. A renamed module is compared with the file it
  was renamed from, and a literal whose count over all the touched modules did not grow
  only moved (a module split in two).
"""

from __future__ import annotations

import ast
import codecs
import dataclasses
import io
import json
import re
import subprocess
import tokenize
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

from code_metrics_collect import ToolError, Units, is_excluded, list_files

#: A run of added full-line comments longer than this fails.
MAX_COMMENT_BLOCK = 8
#: A literal written this many times in one module, by an added occurrence, fails.
MAX_LITERAL_REPEATS = 3
#: Numbers that are idioms rather than magic values.
_PLAIN_NUMBERS = frozenset({0, 1, -1, 2})
_MIN_STRING = 3
#: Methods whose first argument is a key, as ``x["k"]``'s is.
_KEY_CALLS = frozenset({"get", "pop", "setdefault"})
#: The PR label that waives baseline growth and new violations; read from the API.
OVERRIDE_LABEL = "baseline-raise"
ENCODING = "utf-8"
#: CJK Unified Ideographs (with extension A, the compatibility block and the
#: supplementary-plane extensions), CJK Symbols and Punctuation, Halfwidth and
#: Fullwidth Forms.
CJK = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0003134f]")
# A bare number needs 3-5 digits with no leading zero, so a colour (#000000) is not one.
_REFERENCE = re.compile(
    r"(?<![\w&/])#[1-9]\d{2,4}\b"
    r"|\b(?:PR|GH|pull request)[\s-]*#?\s*\d+\b"
    r"|\bissue\s*(?:#\s*\d+|\d{3,})\b"
    r"|github\.com/\S+/(?:pull|issues)/\d+",
    re.IGNORECASE,
)
_EVENT = r"(?:incident|outage)s?"
_DATE = r"\d{4}-\d{2}-\d{2}"
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_NARRATION = re.compile(
    rf"\b{_DATE}\b.{{0,40}}\b{_EVENT}\b"
    rf"|\b{_EVENT}\b.{{0,20}}\b{_DATE}\b"
    r"|\bpost-?mortems?\b"
    rf"|\b(?:during|after|before|since|from|in|against|caused)\s+(?:the|this|that|last|an?|our|yesterday's|today's)"
    rf"\s+(?:[\w-]+\s+)?{_EVENT}\b"
    rf"|\b{_EVENT}\s+(?:on|in|of|last)\s+(?:{_MONTH}|\d|yesterday|week|month)",
    re.IGNORECASE,
)
_TODO = re.compile(r"#\s*TODO\b")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_SNIFF_BYTES = 8192
#: Check names, as ``Problem.check`` and the report's sections carry them.
COMMENTS, ENGLISH, TEST_IMPORTS, LITERALS = "comments", "english", "test-imports", "literals"
#: What makes a module test code: a ``tests`` package on its path, or a ``test_`` file name.
_TESTS, _TEST_PREFIX, _CONFTEST = "tests", "test_", "conftest"
#: Calls that import the module their first (string) argument names.
_IMPORT_CALLS = frozenset({"import_module", "__import__"})


@dataclasses.dataclass(frozen=True)
class Problem:
    """One refusal: where it is and what it says."""

    check: str
    where: str
    line: int
    message: str


@dataclasses.dataclass
class PullRequest:
    """What the checks read from a pull request, fetched from the API at run time."""

    number: int
    labels: list[str]
    title: str
    body: str
    #: (sha, message) per commit.
    commits: list[tuple[str, str]]

    @property
    def raises_baseline(self) -> bool:
        return OVERRIDE_LABEL in self.labels


def git(root: Path, *args: str, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "--literal-pathspecs", *args], capture_output=True, text=True, check=False
    )
    if proc.returncode not in ok:
        raise ToolError(f"git {' '.join(args[:3])} exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return proc


def touched_files(root: Path, since: str) -> set[str]:
    """Paths the change touches: the working tree against ``since``, plus untracked files."""
    changed = git(root, "diff", "--name-only", "--no-renames", "-z", since, "--").stdout
    return {path for path in (changed + untracked_files(root)).split("\0") if path}


def untracked_files(root: Path) -> str:
    """NUL-separated untracked, not ignored paths."""
    return git(root, "ls-files", "-z", "--others", "--exclude-standard").stdout


def show(root: Path, ref: str, path: str) -> str | None:
    """``path`` at ``ref`` as text, or None when it did not exist there."""
    if git(root, "cat-file", "-e", f"{ref}:{path}", ok=(0, 1, 128)).returncode != 0:
        return None
    return git(root, "show", f"{ref}:{path}").stdout


@dataclasses.dataclass
class Diff:
    """What the change adds, per new path, and the text of every line it removes."""

    #: New-side line numbers each path gains (whitespace-only changes are not gains).
    added: dict[str, set[int]]
    #: Stripped text of the removed lines, over every file: an "added" line with the same
    #: text was moved or re-indented, not written.
    removed: set[str]
    #: New path -> old path of each rename the diff detected.
    renames: dict[str, str]


#: Read every file as text whatever ``.gitattributes`` says (``-diff``/``binary`` would hide
#: the hunks), with no external diff or textconv driver, renames paired, whitespace ignored.
_DIFF_ARGS = ("diff", "--text", "--no-textconv", "--no-ext-diff", "--no-color", "-M", "-w", "-U0")


def read_diff(root: Path, since: str) -> Diff:
    """The change from ``since`` to the working tree (untracked files are wholly added)."""
    diff = parse_diff(git(root, "-c", "core.quotePath=false", *_DIFF_ARGS, since, "--").stdout)
    for path in untracked_files(root).split("\0"):
        if path and not (root / path).is_symlink() and (root / path).is_file():
            diff.added[path] = set(range(1, _line_count(root / path) + 1))
    return diff


def parse_diff(text: str) -> Diff:
    """A :class:`Diff` from ``git diff -M -U0`` output."""
    diff = Diff(added={}, removed=set(), renames={})
    path = old = None
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            path = old = None
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
            if path is not None:
                diff.added.setdefault(path, set()).update(parse_added_lines(line))
        elif in_hunk:
            if line.startswith("-"):
                diff.removed.add(line[1:].strip())
        elif line.startswith("rename from "):
            old = line[len("rename from ") :]
        elif line.startswith("rename to ") and old is not None:
            diff.renames[line[len("rename to ") :]] = old
        elif line.startswith("+++ "):
            path = line[len("+++ b/") :] if line.startswith("+++ b/") else None
    return diff


def _line_count(path: Path) -> int:
    return len(path.read_bytes().splitlines())


def parse_added_lines(diff: str) -> set[int]:
    """New-side line numbers of every ``+`` hunk in a ``-U0`` diff."""
    added: set[int] = set()
    for line in diff.splitlines():
        match = _HUNK.match(line)
        if match:
            start, count = int(match.group(1)), int(match.group(2) or 1)
            added.update(range(start, start + count))
    return added


# --- comments ------------------------------------------------------------------


def comment_lines(text: str) -> tuple[set[int], dict[int, str]]:
    """(lines holding only a comment, comment text per line) of Python source ``text``."""
    full: set[int] = set()
    comments: dict[int, str] = {}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return _comment_lines_by_text(text)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            row, col = token.start
            comments[row] = token.string
            if not token.line[:col].strip():
                full.add(row)
    return full, comments


def _comment_lines_by_text(text: str) -> tuple[set[int], dict[int, str]]:
    comments = {n: line.strip() for n, line in enumerate(text.splitlines(), 1) if line.lstrip().startswith("#")}
    return set(comments), comments


def comment_problems(path: str, text: str, added: set[int], removed: Iterable[str] = ()) -> list[Problem]:
    """Refusals for the comments ``added`` puts into ``text``.

    A line whose stripped text is in ``removed`` (the change took it out somewhere) was
    moved or re-indented, not written, and does not count.
    """
    full, comments = comment_lines(text)
    moved = set(removed)
    if moved:
        lines = text.splitlines()
        added = {row for row in added if row > len(lines) or lines[row - 1].strip() not in moved}
    problems = [
        Problem(COMMENTS, path, start, f"adds a {length}-line comment block (limit {MAX_COMMENT_BLOCK})")
        for start, length in _runs(sorted(added & full))
        if length > MAX_COMMENT_BLOCK
    ]
    todo_rows = {row for row, comment in comments.items() if _TODO.match(comment)}
    exempt = todo_rows | {row + 1 for row in todo_rows if row + 1 in full}
    for row in sorted(added & set(comments) - exempt):
        problems += _comment_text_problems(path, row, comments[row])
    return problems


def _comment_text_problems(path: str, row: int, comment: str) -> list[Problem]:
    found = []
    reference = _REFERENCE.search(comment)
    if reference:
        found.append(Problem(COMMENTS, path, row, f"comment points at a PR or issue (`{reference.group(0)}`)"))
    narration = _NARRATION.search(comment)
    if narration:
        found.append(Problem(COMMENTS, path, row, f"comment narrates an incident (`{narration.group(0)}`)"))
    return found


def _runs(rows: list[int]) -> Iterator[tuple[int, int]]:
    """(first row, length) of each run of consecutive numbers in sorted ``rows``."""
    start = previous = None
    for row in rows:
        if previous is not None and row == previous + 1:
            previous = row
            continue
        if start is not None:
            yield start, previous - start + 1
        start = previous = row
    if start is not None:
        yield start, previous - start + 1


# --- English only ----------------------------------------------------------------


def cjk_in_text(where: str, text: str) -> list[Problem]:
    problems = []
    for row, line in enumerate(text.splitlines(), 1):
        match = CJK.search(line)
        if match:
            problems.append(Problem(ENGLISH, where, row, f"CJK character U+{ord(match.group(0)):04X}"))
    return problems


def tracked_text_files(root: Path) -> list[str]:
    listed = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout
    return sorted({path for path in listed.split("\0") if path})


def cjk_problems(root: Path, paths: Iterable[str]) -> list[Problem]:
    """CJK characters in every text file of ``paths`` (binary and non-UTF-8 files are skipped)."""
    problems: list[Problem] = []
    for path in paths:
        full = root / path
        if full.is_symlink() or not full.is_file():
            continue
        text = _decode(full.read_bytes())
        if text is not None:
            problems += cjk_in_text(path, text)
    return problems


def _decode(data: bytes) -> str | None:
    """``data`` as text: UTF-16 when it opens with a byte-order mark, else UTF-8; None for binary."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if b"\0" in data[:_SNIFF_BYTES]:
        return None
    try:
        return data.decode(ENCODING)
    except UnicodeDecodeError:
        return None


def pull_request_cjk(pr: PullRequest) -> list[Problem]:
    problems = cjk_in_text("PR title", pr.title) + cjk_in_text("PR body", pr.body)
    for sha, message in pr.commits:
        problems += cjk_in_text(f"commit {sha[:12]} message", message)
    return problems


# --- production imports of test code --------------------------------------------


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = parts[-1]
    return _TESTS in parts[:-1] or name.startswith(_TEST_PREFIX) or name == "conftest.py"


def _is_test_module_name(dotted: str) -> bool:
    return any(part in (_TESTS, _CONFTEST) or part.startswith(_TEST_PREFIX) for part in dotted.split("."))


def test_import_problems(root: Path, files: Iterable[str], units: Units, roots: Iterable[str]) -> list[Problem]:
    """Imports of a ``tests`` package or a ``test_*`` module from a non-test module."""
    search = [root / r for r in roots]
    problems = []
    for path in files:
        if is_test_path(path):
            continue
        for node in ast.walk(units.tree(path)):
            # One refusal per import statement: ``from pkg.tests import test_x`` names one test module.
            target = next((t for t in _imported(node, root / path, search) if _is_test_module_name(t)), None)
            if target is not None:
                problems.append(Problem(TEST_IMPORTS, path, node.lineno, f"production code imports `{target}`"))
    return problems


def _imported(node: ast.AST, file: Path, search: list[Path]) -> list[str]:
    """Dotted module paths ``node`` imports; a relative one keeps its leading dots."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.Call):
        return _dynamic_import(node)
    if not isinstance(node, ast.ImportFrom):
        return []
    base = "." * node.level + (node.module or "")
    targets = [base] if node.module else []
    folders = _source_folders(node, file, search)
    targets += [
        f"{base}.{alias.name}" if node.module else base + alias.name
        for alias in node.names
        if _names_module(alias.name, folders)
    ]
    return targets


def _dynamic_import(call: ast.Call) -> list[str]:
    """The module ``importlib.import_module("...")`` or ``__import__("...")`` names."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if name not in _IMPORT_CALLS or not call.args:
        return []
    first = call.args[0]
    return [first.value] if isinstance(first, ast.Constant) and isinstance(first.value, str) else []


def _source_folders(node: ast.ImportFrom, file: Path, search: list[Path]) -> list[Path]:
    """Where the package ``node`` imports from would live."""
    parts = (node.module or "").split(".") if node.module else []
    if node.level:
        package = file.parent
        for _ in range(node.level - 1):
            package = package.parent
        return [package.joinpath(*parts)]
    return [folder.joinpath(*parts) for folder in search]


def _names_module(name: str, folders: list[Path]) -> bool:
    """Whether ``from package import name`` names a test module (``tests``/``conftest`` always do)."""
    if name in (_TESTS, _CONFTEST):
        return True
    return name.startswith(_TEST_PREFIX) and any(
        (folder / f"{name}.py").is_file() or (folder / name).is_dir() for folder in folders
    )


# --- repeated literals ----------------------------------------------------------


Literal = tuple[str, object]


def literal_counts(tree: ast.AST) -> dict[Literal, list[int]]:
    """Line numbers of each countable literal in ``tree``, value positions only."""
    skip: set[int] = set()
    _mark_non_values(tree, skip)
    found: dict[Literal, list[int]] = {}
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        literal = _literal(node, skip)
        if literal is not None:
            found.setdefault(literal, []).append(node.lineno)
    return found


def _literal(node: ast.AST, skip: set[int]) -> Literal | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and _is_number(node.operand):
        skip.add(id(node.operand))
        return _number(-node.operand.value)
    if not isinstance(node, ast.Constant):
        return None
    if isinstance(node.value, (str, bytes)):
        return (type(node.value).__name__, node.value) if len(node.value) >= _MIN_STRING else None
    return _number(node.value) if _is_number(node) else None


def _is_number(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and not isinstance(node.value, bool)
    )


def _number(value: object) -> Literal | None:
    return None if value in _PLAIN_NUMBERS else (type(value).__name__, value)


def _mark_non_values(tree: ast.AST, skip: set[int]) -> None:
    """Add every node in a key, docstring, f-string text, annotation or ``__all__`` position."""
    for node in ast.walk(tree):
        for excluded in _non_value_children(node):
            skip.update(id(sub) for sub in ast.walk(excluded))


def _non_value_children(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Dict):
        return [key for key in node.keys if key is not None]
    if isinstance(node, ast.Subscript):
        return [node.slice]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _KEY_CALLS:
        return node.args[:1]
    if isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.In, ast.NotIn)):
        return [node.left]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [value for value in node.values if isinstance(value, ast.Constant)]
    if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
        return [node.format_spec]
    if isinstance(node, (ast.AnnAssign, ast.arg)):
        return [node.annotation] if node.annotation is not None else []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [node.returns] if node.returns is not None else []
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and _assigns_all(node):
        return [node.value] if node.value is not None else []
    return []


def _assigns_all(node: ast.Assign | ast.AugAssign | ast.AnnAssign) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets)


def repeated_literal_problems(
    path: str, before: str | None, after: str, moved: Iterable[Literal] = ()
) -> list[Problem]:
    """Literals the change takes from under 3 to 3+ occurrences in ``path``.

    A literal in ``moved`` only moved between modules (its count over every touched
    module did not grow) and is not refused.
    """
    old = {lit: len(rows) for lit, rows in _parse_literals(path, before).items()}
    skip = set(moved)
    problems = []
    for literal, rows in sorted(_parse_literals(path, after).items(), key=lambda item: item[1][0]):
        if len(rows) >= MAX_LITERAL_REPEATS > old.get(literal, 0) and literal not in skip:
            problems.append(
                Problem(
                    LITERALS,
                    path,
                    rows[0],
                    f"{_show(literal)} is written {len(rows)} times (was {old.get(literal, 0)}); "
                    "name it as a module-level constant",
                )
            )
    return problems


def moved_literals(befores: Iterable[tuple[str, str | None]], afters: Iterable[tuple[str, str]]) -> set[Literal]:
    """Literals whose total count over the touched modules did not grow: they only moved."""
    total: dict[Literal, int] = {}
    for sign, texts in ((-1, befores), (1, afters)):
        for path, text in texts:
            for literal, rows in _parse_literals(path, text).items():
                total[literal] = total.get(literal, 0) + sign * len(rows)
    return {literal for literal, grew in total.items() if grew <= 0}


def _parse_literals(path: str, text: str | None) -> dict[Literal, list[int]]:
    if text is None:
        return {}
    try:
        return literal_counts(ast.parse(text, filename=path))
    except (SyntaxError, ValueError) as exc:
        raise ToolError(f"cannot parse {path}: {exc}") from exc


def _show(literal: Literal) -> str:
    shown = repr(literal[1])
    return f"`{shown[:60]}...`" if len(shown) > 60 else f"`{shown}`"


# --- all of the above ------------------------------------------------------------


def run_all(root: Path, roots: Iterable[str], exclude: Iterable[str], units: Units, since: str | None) -> list[Problem]:
    """Every check that reads the tree; the diff-only ones only when ``since`` names the base.

    Comments are checked in every touched ``.py`` file outside ``exclude``, tests included;
    literals and imports in the production modules under ``roots``.
    """
    production = list_files(root, roots, exclude)[0]
    problems = cjk_problems(root, tracked_text_files(root))
    problems += test_import_problems(root, production, units, roots)
    if since is None:
        return problems
    diff = read_diff(root, since)
    excluded, modules = tuple(exclude), set(production)
    touched = sorted(touched_files(root, since))
    literal_paths = []
    for path in touched:
        full = root / path
        if not path.endswith(".py") or is_excluded(path, excluded) or full.is_symlink() or not full.is_file():
            continue
        text = full.read_text(encoding=ENCODING)
        problems += comment_problems(path, text, diff.added.get(path, set()), diff.removed)
        if path in modules and not is_test_path(path):
            literal_paths.append((path, text))
    befores = [(p, show(root, since, p)) for p in touched if _counts_literals(p, roots, excluded)]
    moved = moved_literals(befores, literal_paths)
    for path, text in literal_paths:
        problems += repeated_literal_problems(path, show(root, since, diff.renames.get(path, path)), text, moved)
    return problems


def _counts_literals(path: str, roots: Iterable[str], excluded: tuple[str, ...]) -> bool:
    """Whether ``path`` is a production module of the scope (at the base or now)."""
    under = any(path == r.rstrip("/") or path.startswith(r.rstrip("/") + "/") for r in roots)
    return under and path.endswith(".py") and not is_excluded(path, excluded) and not is_test_path(path)


# --- pull request metadata -----------------------------------------------------


def fetch_pull_request(api: str, repo: str, number: int, token: str) -> PullRequest:
    """Labels, title, body and commit messages of PR ``number``, read from the GitHub API now."""
    pr = _get_json(f"{api}/repos/{repo}/pulls/{number}", token)
    commits: list[tuple[str, str]] = []
    page = 1
    while True:
        rows = _get_json(f"{api}/repos/{repo}/pulls/{number}/commits?per_page=100&page={page}", token)
        commits += [(row["sha"], row["commit"]["message"]) for row in rows]
        if len(rows) < 100:
            break
        page += 1
    return PullRequest(
        number=number,
        labels=[label["name"] for label in pr.get("labels", [])],
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        commits=commits,
    )


def _get_json(url: str, token: str) -> object:
    if not url.startswith("https://"):
        raise ToolError(f"refusing to read {url}: the API URL must be https")
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310 -- https checked above
            return json.loads(response.read().decode(ENCODING))
    except OSError as exc:
        raise ToolError(f"could not read {url}: {exc}") from exc


def load_pull_request(path: Path) -> PullRequest:
    """A :class:`PullRequest` from a JSON file, for a local run that simulates one."""
    data = json.loads(path.read_text(encoding=ENCODING))
    return PullRequest(
        number=int(data.get("number", 0)),
        labels=list(data.get("labels", [])),
        title=data.get("title", ""),
        body=data.get("body", ""),
        commits=[tuple(c) for c in data.get("commits", [])],
    )
