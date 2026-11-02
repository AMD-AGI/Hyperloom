# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The points' trees merged into one vLLM patch, each behind its operating point.

Every point a points directory ran ends with the recipe its FINAL chose, its best tree. The merge writes one patch that
holds every point's tree; the variable :data:`POINT_VARIABLE` names the point whose tree runs, and unset, the root's
runs. A point's performance is its own tree's: the merge lets one image serve the whole curve, it does not mix the
points. What a point changed reaches the tree as code under that variable:

- **server arguments** become the defaults of vLLM's parser (``EngineArgs.add_cli_args``), so an argument given on the
  command line still wins;
- **environment** is set when vLLM is imported (``vllm/env_override.py``), where a variable is not set already;
- **kernel tables** ship with the patch, laid over the image's default tables as the point's tests read them, and the
  variable that loads them names them;
- **code** runs under ``if`` its point is the one named. Each change is widened to the whole statements that hold it;
  where two points changed the same statements the branches are ``if``/``elif``, and points that changed them alike
  share one.

A change the merge cannot put behind a point (outside Python source, a new or deleted file, or one that leaves no
whole statements) ends the merge with why. Which of one point's changes another could use is not decided here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json, atomic_write_text

POINT_VARIABLE = "HYPERLOOM_OPERATING_POINT"
#: Where in a points directory the merge writes: the patch, the recipe that carries it, and what each point brought.
MERGE_DIR = "merged"
PATCH_FILE = "merged.patch"
SUMMARY_FILE = "merged.json"
RECIPE_FILE = "recipe.json"
PAGE_FILE = "MERGED.md"
#: The module the patch adds to vLLM, and where the tables it names are kept, below the vllm package.
MODULE = "vllm/hyperloom_points.py"
DATA_DIR = "vllm/hyperloom_points_data"
#: The files the patch hooks the module into, and how they import it.
ENVS_FILE = "vllm/envs.py"
ENV_OVERRIDE_FILE = "vllm/env_override.py"
ARGS_FILE = "vllm/engine/arg_utils.py"
IMPORT_NAME = "_hyperloom_points"
IMPORT_LINE = f"from vllm import hyperloom_points as {IMPORT_NAME}\n"
#: Where in a session the merge reads: FINAL's choice, the recipes, their blobs and the installed vLLM they patch.
REPORT_FILE = "MB-reports/MB-report.json"
REPO_DIR = "MB-repo"
BLOBS_DIR = "MB-blobs"
SOURCE_DIR = "MB-source"
COMPONENT = "vllm"
#: The installed tree's commit, which every point's patches apply to and the merged patch is taken against.
HEAD = "HEAD"
ENCODING = "utf-8"
INDENT = "    "


class MergeRefused(Exception):
    """The points' trees cannot be merged as asked; the message says why."""


@dataclass(frozen=True)
class PointTree:
    """One point's best tree: its ``conc``, the ``session`` it ran in, FINAL's ``label`` and ``commit``, the recipe
    there and the root's."""

    conc: int
    session: Path
    label: str
    commit: str
    recipe: Mapping[str, Any]
    root: Mapping[str, Any]

    @property
    def name(self) -> str:
        return str(self.conc)

    def blob(self, sha256: str) -> Path:
        path = self.session / BLOBS_DIR / sha256
        if not path.is_file() or _sha256(path.read_bytes()) != sha256:
            raise MergeRefused(f"C{self.conc}: the blob {sha256} is missing from {path.parent} or holds other bytes")
        return path


@dataclass
class _Delta:
    """What one point changed from the root: its server argument words, environment, tables by variable (the paths
    the variable lists, relative ones below :data:`DATA_DIR`) and its patches of vLLM, in order."""

    server_args: list[str] = field(default_factory=list)
    envs: dict[str, str] = field(default_factory=dict)
    tables: dict[str, list[str]] = field(default_factory=dict)
    patches: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class _Edit:
    """One point's change of a file's lines ``start``..``end`` (0-based, end exclusive) to ``lines``."""

    point: str
    start: int
    end: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Merged:
    """What the merge wrote: the points it merged, those it left out and why, and its files."""

    points: tuple[int, ...]
    left_out: Mapping[int, str]
    patch: Path
    summary: Path


def read_trees(root: Path) -> tuple[list[PointTree], dict[int, str]]:
    """Every point in ``root`` whose last run stage ended with FINAL's choice, by conc; and why each other point that
    ran is left out."""

    trees: list[PointTree] = []
    left_out: dict[int, str] = {}
    for path in sorted(root.glob("c*/point.json")):
        document = _read_json(path) or {}
        run, conc = document.get("run"), document.get("conc")
        if not isinstance(run, dict) or not isinstance(conc, int):
            continue
        session = Path(str(run.get("session") or ""))
        chosen = ((_read_json(session / REPORT_FILE) or {}).get("final") or {}).get("chosen") or {}
        if run.get("exit_code") != 0 or not chosen.get("commit"):
            left_out[conc] = f"its run stage exited {run.get('exit_code')} with no FINAL choice in {session}"
            continue
        repo = session / REPO_DIR
        roots = _git(repo, "rev-list", "--max-parents=0", chosen["commit"]).split()
        if len(roots) != 1:
            raise MergeRefused(f"C{conc}: {chosen['commit']} in {repo} descends from {len(roots)} roots, not one")
        recipe = json.loads(_git(repo, "show", f"{chosen['commit']}:recipe.json"))
        base = json.loads(_git(repo, "show", f"{roots[0]}:recipe.json"))
        label = str(chosen.get("label") or chosen.get("branch") or "")
        trees.append(PointTree(conc, session, label, chosen["commit"], recipe, base))
    return sorted(trees, key=lambda tree: tree.conc), left_out


def merge(root: Path, aiter: Path | None = None) -> Merged:
    """Merge the best trees of the points in ``root`` into one patch behind :data:`POINT_VARIABLE`, written below
    ``root/merged``. ``aiter`` is the image's aiter package, which a rows-only kernel table is laid over (found in this
    environment when not given). Raise :class:`MergeRefused` with why when the trees cannot be merged."""

    trees, left_out = read_trees(root)
    if not trees:
        raise MergeRefused(f"no point in {root} has a run stage that ended with FINAL's choice")
    base = trees[0].root
    for tree in trees[1:]:
        if tree.root != base:
            raise MergeRefused(f"C{tree.conc} started from another root than C{trees[0].conc}; only one root merges")
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "installed"
        _git(_installed_source(trees), "clone", "--quiet", "--no-hardlinks", ".", str(source))
        deltas = {tree.name: _delta(tree) for tree in trees}
        changes = _point_changes(source, deltas, Path(scratch))
        _ship_tables(trees, deltas, source, aiter)
        _write_tree(source, changes, deltas)
        _git(source, "add", "-A")
        patch = _git(source, "diff", "--cached", "--binary", HEAD)
    out = root / MERGE_DIR
    if out.exists():
        shutil.rmtree(out)
    (out / "patch" / COMPONENT).mkdir(parents=True)
    digest = _sha256(patch.encode(ENCODING))
    (out / PATCH_FILE).write_text(patch, encoding=ENCODING)
    shutil.copyfile(out / PATCH_FILE, out / "patch" / COMPONENT / f"{digest}.patch")
    summary = _summary(trees, left_out, deltas, changes, digest)
    atomic_write_json(out / SUMMARY_FILE, summary)
    atomic_write_json(out / RECIPE_FILE, _recipe(base, digest))
    atomic_write_text(out / PAGE_FILE, _page(root, summary))
    return Merged(tuple(tree.conc for tree in trees), left_out, out / PATCH_FILE, out / SUMMARY_FILE)


# --- one point's changes ------------------------------------------------------------------------------------------


def _delta(tree: PointTree) -> _Delta:
    """What ``tree`` changed from its root; raise :class:`MergeRefused` for a change the merge cannot carry."""

    delta = _launch_delta(tree)
    delta.patches = _added_patches(tree)
    return delta


def _launch_delta(tree: PointTree) -> _Delta:
    """The server arguments and environment ``tree`` added to its root's launch."""

    delta = _Delta()
    launch, base = tree.recipe.get("launch") or {}, tree.root.get("launch") or {}
    words, root_words = (
        shlex.split(launch.get("extra_server_args") or ""),
        shlex.split(base.get("extra_server_args") or ""),
    )
    if words[: len(root_words)] != root_words:
        raise MergeRefused(f"C{tree.conc} changed the root's own server arguments {root_words}; only additions merge")
    delta.server_args = words[len(root_words) :]
    root_envs = base.get("extra_envs") or {}
    for name, value in (launch.get("extra_envs") or {}).items():
        if name in root_envs and root_envs[name] != value:
            raise MergeRefused(f"C{tree.conc} changed the root's {name}; only additions merge")
        if name not in root_envs:
            delta.envs[name] = str(value)
    return delta


def _added_patches(tree: PointTree) -> list[Path]:
    """The blobs of the patches ``tree`` added to its root's vLLM, in order."""

    root_components = {item["component"]: item for item in tree.root.get("components") or []}
    patches: list[Path] = []
    for item in tree.recipe.get("components") or []:
        before = {patch["sha256"] for patch in root_components.get(item["component"], {}).get("patches") or []}
        added = [patch for patch in item.get("patches") or [] if patch["sha256"] not in before]
        if added and item["component"] != COMPONENT:
            raise MergeRefused(f"C{tree.conc} patches {item['component']}; only {COMPONENT} patches merge")
        patches += [tree.blob(patch["sha256"]) for patch in added]
    return patches


def _point_changes(source: Path, deltas: Mapping[str, _Delta], scratch: Path) -> dict[str, list[_Edit]]:
    """Each point's patches applied in order to the installed tree, as edits of its files, by file."""

    edits: dict[str, list[_Edit]] = {}
    for name, delta in deltas.items():
        if not delta.patches:
            continue
        work = scratch / f"c{name}"
        _git(source, "worktree", "add", "--quiet", "--detach", str(work), HEAD)
        for patch in delta.patches:
            _git(work, "apply", "--whitespace=nowarn", str(patch))
        added = _git(work, "ls-files", "--others", "--exclude-standard").split()
        if added:
            raise MergeRefused(f"C{name} adds files ({', '.join(added)}); a new file cannot be put behind a point")
        for path in _git(work, "diff", "--name-only", HEAD).split():
            if not (work / path).is_file():
                raise MergeRefused(f"C{name} deletes {path}; a deleted file cannot be put behind a point")
            before = (source / path).read_text(encoding=ENCODING).splitlines(keepends=True)
            after = (work / path).read_text(encoding=ENCODING).splitlines(keepends=True)
            matcher = SequenceMatcher(a=before, b=after, autojunk=False)
            edits.setdefault(path, []).extend(
                _Edit(name, i1, i2, tuple(after[j1:j2]))
                for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                if tag != "equal"
            )
        _git(source, "worktree", "remove", "--force", str(work))
    return edits


# --- kernel tables ------------------------------------------------------------------------------------------------


def _ship_tables(trees: Sequence[PointTree], deltas: Mapping[str, _Delta], work: Path, aiter: Path | None) -> None:
    """Ship each point's kernel tables below :data:`DATA_DIR` in ``work`` and name them in its delta's tables."""

    for tree in trees:
        root_shas = {kernel["sha256"] for kernel in tree.root.get("kernels") or []}
        for kernel in tree.recipe.get("kernels") or []:
            if kernel["sha256"] in root_shas:
                continue
            load = (kernel.get("metadata") or {}).get("load") or {}
            variable = load.get("env")
            if load.get("via") != "env_file" or not variable:
                raise MergeRefused(f"C{tree.conc}: {kernel['ref']} is not loaded through a variable; it cannot merge")
            if variable in deltas[tree.name].envs or variable in deltas[tree.name].tables:
                raise MergeRefused(f"C{tree.conc} sets {variable} twice")
            folder = f"c{tree.conc}/{variable.lower()}"
            rows = _place(work, f"{folder}/{Path(kernel['ref']).name}", tree.blob(kernel["sha256"]).read_bytes())
            parts = _laid(work, folder, variable, rows, aiter) if load.get("rows_only") else [rows]
            deltas[tree.name].tables[variable] = parts


def _laid(work: Path, folder: str, variable: str, rows: str, aiter: Path | None) -> list[str]:
    """What ``variable`` lists for the rows file ``rows``: the image's default tables, or a shipped copy of each the
    rows take shapes out of, then the rows, as a test reads them (``hyperloom.measure.framework.copies``)."""

    from hyperloom.common.aiter_jit_cache import resolve_package_root
    from hyperloom.measure.framework.rows import lay_rows_over
    from hyperloom.measure.framework.table_modules import default_tables
    from hyperloom.measure.framework.tables import read_variables

    package = aiter or resolve_package_root()
    if package is None or not (package / "jit" / "core.py").is_file():
        raise MergeRefused(f"{variable}: a rows-only table is laid over the image's aiter, which is not here")
    known = read_variables(package / "jit" / "core.py").get(variable)
    if known is None:
        raise MergeRefused(f"{variable}: the image's aiter declares no such table variable")
    defaults = [Path(path) for path in default_tables(package, known) if Path(path).is_file()]
    if not defaults:
        raise MergeRefused(f"{variable}: the image ships no default table to lay the rows over")
    over = lay_rows_over(defaults, [work / DATA_DIR / rows], package / "configs", known.tuned)
    listed = [
        _place(work, f"{folder}/kept/{path.name}", over.kept[path]) if path in over.kept else str(path)
        for path in defaults
    ]
    return [*listed, rows]


def _place(work: Path, relative: str, data: bytes) -> str:
    target = work / DATA_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return relative


# --- code ---------------------------------------------------------------------------------------------------------


def _write_tree(work: Path, changes: Mapping[str, list[_Edit]], deltas: Mapping[str, _Delta]) -> None:
    """Write every point's code behind its point, then the module and its hooks, into ``work``."""

    for path, edits in sorted(changes.items()):
        if not path.endswith(".py"):
            raise MergeRefused(f"{path}: only Python source branches by operating point, and {_named(edits)} change it")
        text = (work / path).read_text(encoding=ENCODING)
        (work / path).write_text(_with_import(_branched(path, text, edits)), encoding=ENCODING)
    _hook(work)
    (work / MODULE).write_text(_module(deltas), encoding=ENCODING)


def _branched(path: str, text: str, edits: Sequence[_Edit]) -> str:
    """``text`` with each group of overlapping edits put behind its points, widened to the statements that hold it."""

    lines = text.splitlines(keepends=True)
    spans = _spans(ast.parse(text))
    widened = sorted(((*_statements(spans, edit.start, edit.end), edit) for edit in edits), key=lambda item: item[:2])
    groups: list[tuple[int, int, list[_Edit]]] = []
    for start, end, edit in widened:
        last = groups[-1] if groups else None
        if last is not None and (start < last[1] or (start, end) == (last[0], last[1])):
            groups[-1] = (last[0], max(end, last[1]), [*last[2], edit])
        else:
            groups.append((start, end, [edit]))
    for start, end, group in reversed(groups):
        lines[start:end] = _branch(path, lines, start, end, group)
    result = "".join(lines)
    try:
        ast.parse(result)
    except SyntaxError as error:
        raise MergeRefused(f"{path}: its points' changes, branched, are no longer Python ({error})") from error
    return result


def _spans(tree: ast.Module) -> list[list[tuple[int, int]]]:
    """The lines (0-based, end exclusive) of the statements of each body in ``tree``, decorators included."""

    bodies = []
    for node in ast.walk(tree):
        for body in [getattr(node, name, None) for name in ("body", "orelse", "finalbody")] + [
            handler.body for handler in getattr(node, "handlers", None) or []
        ]:
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                bodies.append([(_first_line(stmt) - 1, stmt.end_lineno or stmt.lineno) for stmt in body])
    return bodies


def _statements(spans: list[list[tuple[int, int]]], start: int, end: int) -> tuple[int, int]:
    """The lines of the fewest whole statements, side by side in one body, that hold lines ``start``..``end``. An
    insertion between two statements stays where it is; one inside a statement takes that statement."""

    if start == end:
        if any(first == start or last == start for body in spans for first, last in body):
            return start, end
        holding = [(first, last) for body in spans for first, last in body if first < start < last]
        return min(holding, key=lambda span: span[1] - span[0], default=(start, end))
    best: tuple[int, int] | None = None
    for body in spans:
        held = [(first, last) for first, last in body if first < end and last > start]
        if held and held[0][0] <= start and held[-1][1] >= end:
            candidate = (held[0][0], held[-1][1])
            if best is None or candidate[1] - candidate[0] < best[1] - best[0]:
                best = candidate
    return best or (start, end)


def _first_line(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", None) or []
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def _branch(path: str, lines: list[str], start: int, end: int, group: Sequence[_Edit]) -> list[str]:
    """Lines ``start``..``end`` as one branch per text the points of ``group`` gave them, then the original's."""

    variants: dict[tuple[str, ...], list[str]] = {}
    for point in dict.fromkeys(edit.point for edit in group):
        text: list[str] = []
        at = start
        for edit in sorted((edit for edit in group if edit.point == point), key=lambda edit: edit.start):
            text += [*lines[at : edit.start], *edit.lines]
            at = edit.end
        variants.setdefault((*text, *lines[at:end]), []).append(point)
    original = tuple(lines[start:end])
    blocks = [*variants, original]
    indent = min((_indent(line) for block in blocks for line in block if line.strip()), key=len, default="")
    for block in blocks:
        _check_whole(path, start, block, indent)
    out: list[str] = []
    for number, (text, points) in enumerate(variants.items()):
        names = ", ".join(json.dumps(point) for point in points)
        out += [f"{indent}{'elif' if number else 'if'} {IMPORT_NAME}.active({names}):\n", *_nested(text, indent)]
    if any(line.strip() for line in original):
        out += [f"{indent}else:\n", *_nested(original, indent)]
    return out


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _check_whole(path: str, start: int, block: Sequence[str], indent: str) -> None:
    text = "".join(line[len(indent) :] if line.strip() else line for line in block)
    try:
        ast.parse(text)
    except SyntaxError as error:
        raise MergeRefused(
            f"{path}: the change at line {start + 1} leaves no whole statements to put behind a point ({error.msg})"
        ) from error


def _nested(block: Sequence[str], indent: str) -> list[str]:
    if not any(line.strip() for line in block):
        return [f"{indent}{INDENT}pass\n"]
    return [INDENT + line if line.strip() else line for line in block]


def _named(edits: Sequence[_Edit]) -> str:
    return ", ".join(f"C{point}" for point in dict.fromkeys(edit.point for edit in edits))


def _with_import(text: str) -> str:
    """``text`` importing the module before its first statement after the docstring and ``__future__`` imports."""

    if IMPORT_LINE in text:
        return text
    body = list(ast.parse(text).body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    while body and isinstance(body[0], ast.ImportFrom) and body[0].module == "__future__":
        body = body[1:]
    lines = text.splitlines(keepends=True)
    at = _first_line(body[0]) - 1 if body else len(lines)
    return "".join([*lines[:at], IMPORT_LINE, *lines[at:]])


# --- the module and its hooks -------------------------------------------------------------------------------------


def _hook(work: Path) -> None:
    """Hook the module into vLLM: the variable declared, the environment set at import, the server arguments made the
    parser's defaults."""

    envs = work / ENVS_FILE
    text = envs.read_text(encoding=ENCODING)
    declared, table = "if TYPE_CHECKING:\n", "environment_variables: dict[str, Callable[[], Any]] = {\n"
    _require(ENVS_FILE, text, declared, table)
    text = text.replace(declared, f"{declared}{INDENT}{POINT_VARIABLE}: str | None = None\n", 1)
    entry = f'{INDENT}"{POINT_VARIABLE}": lambda: os.getenv("{POINT_VARIABLE}"),\n'
    envs.write_text(text.replace(table, f"{table}{entry}", 1), encoding=ENCODING)

    override = work / ENV_OVERRIDE_FILE
    text = override.read_text(encoding=ENCODING)
    _require(ENV_OVERRIDE_FILE, text, "import os\n")
    call = f"import os\n\n{'' if IMPORT_LINE in text else IMPORT_LINE}\n{IMPORT_NAME}.apply_environment()\n"
    override.write_text(text.replace("import os\n", call, 1), encoding=ENCODING)

    args = work / ARGS_FILE
    text = args.read_text(encoding=ENCODING)
    returned = _add_cli_args_return(ast.parse(text))
    if returned is None:
        raise MergeRefused(f"{ARGS_FILE}: EngineArgs.add_cli_args does not end with a return to hook the defaults into")
    lines = text.splitlines(keepends=True)
    at = returned.lineno - 1
    lines.insert(at, f"{_indent(lines[at])}{IMPORT_NAME}.apply_server_args(parser)\n")
    args.write_text(_with_import("".join(lines)), encoding=ENCODING)


def _require(path: str, text: str, *anchors: str) -> None:
    missing = [anchor.strip() for anchor in anchors if anchor not in text]
    if missing:
        raise MergeRefused(f"{path}: this vLLM has no {', '.join(missing)} to hook the operating points into")


def _add_cli_args_return(tree: ast.Module) -> ast.Return | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EngineArgs":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "add_cli_args":
                    last = method.body[-1]
                    return last if isinstance(last, ast.Return) else None
    return None


def _module(deltas: Mapping[str, _Delta]) -> str:
    points = {
        name: {"server_args": delta.server_args, "envs": delta.envs, "tables": delta.tables}
        for name, delta in sorted(deltas.items(), key=lambda item: int(item[0]))
    }
    data = json.dumps(points, indent=4)
    if '"""' in data or data.endswith("\\"):
        raise MergeRefused('a point\'s launch holds """, which the module cannot quote')
    return MODULE_TEMPLATE.replace("__POINTS__", f'r"""\n{data}\n"""')


#: The module the patch adds as ``vllm/hyperloom_points.py``, its points' data in place of ``__POINTS__``.
MODULE_TEMPLATE = '''\
# SPDX-License-Identifier: Apache-2.0
"""Hyperloom's operating points, merged into one tree.

Each operating point Hyperloom optimized has its own tree: server arguments, environment, kernel tables and code.
HYPERLOOM_OPERATING_POINT names the point whose tree runs; unset, the tree they all started from runs. Written by
``hyperloom-multibranch points merge``: merge again rather than edit it.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

VARIABLE = "HYPERLOOM_OPERATING_POINT"
DATA = Path(__file__).with_name("hyperloom_points_data")
POINTS: dict[str, dict[str, Any]] = json.loads(__POINTS__)
_set: dict[str, str] = {}


def point() -> str | None:
    """The operating point whose tree runs, None for the tree they all started from."""
    value = os.environ.get(VARIABLE, "").strip()
    if not value:
        return None
    if value not in POINTS:
        raise ValueError(f"{VARIABLE}={value} names no operating point of this tree, which has {', '.join(POINTS)}")
    return value


def active(*points: str) -> bool:
    """Whether the tree of one of points runs."""
    return point() in points


def apply_environment() -> None:
    """Set the running point's environment and kernel tables, where a variable is not set already."""
    name = point()
    if name is None:
        return
    tree = POINTS[name]
    tables = {
        key: os.pathsep.join(part if os.path.isabs(part) else str(DATA / part) for part in parts)
        for key, parts in tree["tables"].items()
    }
    for key, value in {**tree["envs"], **tables}.items():
        if key not in os.environ:
            os.environ[key] = _set[key] = value


def apply_server_args(parser: Any) -> None:
    """Make the running point's server arguments the defaults of parser, so an argument given still wins."""
    name = point()
    if name is None:
        return
    words = POINTS[name]["server_args"]
    parsed, unknown = parser.parse_known_args(words)
    options = parser._option_string_actions
    dests = {options[word.split("=", 1)[0]].dest for word in words if word.split("=", 1)[0] in options}
    parser.set_defaults(**{dest: getattr(parsed, dest) for dest in dests})
    print(
        f"[hyperloom] operating point {name}: server defaults {sorted(dests)}, environment set {sorted(_set)}"
        + (f"; left out, not this parser's: {unknown}" if unknown else ""),
        file=sys.stderr,
    )
'''


# --- what the merge wrote -----------------------------------------------------------------------------------------


def _summary(
    trees: Sequence[PointTree],
    left_out: Mapping[int, str],
    deltas: Mapping[str, _Delta],
    changes: Mapping[str, list[_Edit]],
    digest: str,
) -> dict[str, Any]:
    def code(name: str) -> list[str]:
        return sorted(path for path, edits in changes.items() if any(edit.point == name for edit in edits))

    return {
        "variable": POINT_VARIABLE,
        "patch_sha256": digest,
        "base_image_digest": trees[0].root.get("base_image_digest"),
        "points": [
            {
                "conc": tree.conc,
                "session": str(tree.session),
                "final": {"label": tree.label, "commit": tree.commit},
                "server_args": deltas[tree.name].server_args,
                "envs": deltas[tree.name].envs,
                "tables": deltas[tree.name].tables,
                "code": code(tree.name),
            }
            for tree in trees
        ],
        "shared_code": sorted(path for path, edits in changes.items() if len({edit.point for edit in edits}) > 1),
        "left_out": {str(conc): why for conc, why in sorted(left_out.items())},
    }


def _recipe(base: Mapping[str, Any], digest: str) -> dict[str, Any]:
    """The root's recipe carrying the merged patch: a point's tree is this recipe with the variable set to it."""

    recipe = json.loads(json.dumps(base))
    for item in recipe.get("components") or []:
        if item["component"] == COMPONENT:
            item["patches"] = [
                *item.get("patches", []),
                {"path": f"patch/{COMPONENT}/{digest}.patch", "sha256": digest},
            ]
    return recipe


def _page(root: Path, summary: Mapping[str, Any]) -> str:
    lines = [f"# The points of {root}, merged", "", f"One patch, `{PATCH_FILE}` (sha256 `{summary['patch_sha256']}`)."]
    lines += [f"`{POINT_VARIABLE}` names the point whose tree runs; unset, the root's runs.", ""]
    lines += [
        "| Point | FINAL's choice | Server arguments | Environment | Tables | Code |",
        "|---|---|---|---|---|---|",
    ]
    for point in summary["points"]:
        args = " ".join(point["server_args"]) or "-"
        lines.append(
            f"| C{point['conc']} | {point['final']['label']} `{point['final']['commit'][:12]}` | `{args}` "
            f"| {', '.join(point['envs']) or '-'} | {', '.join(point['tables']) or '-'} | {', '.join(point['code']) or '-'} |"
        )
    lines += [
        "",
        f"Run one point's tree: `{POINT_VARIABLE}=<conc> vllm serve ...`, the patch applied to the image's vLLM.",
    ]
    lines += [f"- Left out: C{conc}, {why}" for conc, why in summary["left_out"].items()]
    return "\n".join(lines) + "\n"


# --- helpers ------------------------------------------------------------------------------------------------------


def _installed_source(trees: Sequence[PointTree]) -> Path:
    """The installed vLLM the points' patches apply to: one tree, the same in every point's session."""

    sources = {tree.conc: tree.session / SOURCE_DIR for tree in trees}
    contents = {conc: _git(path, "rev-parse", "HEAD^{tree}").strip() for conc, path in sources.items()}
    if len(set(contents.values())) != 1:
        raise MergeRefused(f"the points' sessions installed different vLLM trees: {contents}")
    return sources[trees[0].conc]


def _git(cwd: Path, *args: str) -> str:
    # An inherited GIT_* (GIT_DIR, GIT_WORK_TREE) would aim the command at another repository.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False, env=env)
    if done.returncode != 0:
        raise MergeRefused(f"git {' '.join(args)} in {cwd}: {done.stderr.strip() or done.returncode}")
    return done.stdout


def _read_json(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["MODULE_TEMPLATE", "POINT_VARIABLE", "MergeRefused", "Merged", "PointTree", "merge", "read_trees"]
