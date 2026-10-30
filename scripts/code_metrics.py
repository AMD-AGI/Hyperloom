# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""White-box code-metrics gate: every unit over a threshold compared with the base, plus absolute checks.

The dimensions, their thresholds (with the reason for each number) and the pinned tool
versions live in ``pyproject.toml`` under ``[tool.hyperloom.code_metrics]``. Nothing else
is recorded: the base is the reference, so a unit a merged change improved is the next
change's limit.

With ``--base-ref``, the tree at the merge base with that ref is measured too, with this
tree's configuration, and only files the change touches (the diff of the working tree
against the merge base) are judged; the rest of the tree is reported for information and
never fails an unrelated change. In a touched file:

* a unit over a threshold that the base did not have over it fails as *new*;
* a unit worse than at the base fails as *worse*;
* a unit the base already had at that value or worse passes (the backlog is the base's,
  as in the style guide's complexity ceiling); one the change improved is listed.

The thresholds, scope and tool parameters may not loosen relative to ``--base-ref`` (the
scope is compared by the files each config measures in this tree, not by its text).
Without ``--base-ref`` there is nothing to compare: every unit over a threshold is listed
and none fails. A push run (``--pushed-sha``) lists them too, so the job summary of each
merge shows the whole backlog.

The pull-request label ``baseline-raise``, read from the GitHub API at run time
(``--pr-number``; on a push, ``--pushed-sha`` finds the PR whose merge made the commit;
``--pull-request-json`` stands in for the API in a local run), turns new or worse units
and the failing design checks into *waived* findings: still listed in full in the log
and the report, no longer failing. It waives nothing else.

Next to the comparison, ``code_metrics_checks`` runs checks that have no baseline: added
comment blocks and history in comments, CJK characters (in the tree and, on a pull
request, in its title, body and commit messages), production imports of test code, and
added repeated literals; ``code_metrics_design`` runs the diff-only design checks (A1
private-name reach, A8 flag arguments, A9 tuple returns, A10 raw vocabulary, A11 global
writes, A12 forwarders and single implementations). A9, A10, A11 and A12 single
implementations are advisory, as the A rules rate them: listed, never failing. The
diff-only ones need ``--base-ref``.

CI runs the base branch's copy of these scripts against the change, so editing the gate
does not change the verdict on the same change; the report lists any edit to the gate
implementation or its tool pins for a reviewer.

A unit that only moved to another file (same metric, same qualified name, or the same
file name for a module; value no worse) is compared with the unit it left at the base;
see :func:`match_moves` for the limit.

Usage::

    pip install ruff==0.16.2 complexipy==8.0.1 radon==6.0.1 vulture==2.16
    npm install -g jscpd@5.4.1      # or CODE_METRICS_JSCPD="npx --yes jscpd@5.4.1"

    python scripts/code_metrics.py                         # list every unit over a threshold
    python scripts/code_metrics.py --base-ref origin/main  # judge the files changed since the merge base

Exit code 0 when the gate passes, 1 when it fails, 2 when it could not run (a tool is
missing or at another version, a ref does not resolve, the API could not be read).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import code_metrics_checks as checks
import code_metrics_design as design
from code_metrics_collect import (
    DIRECTORY_ENTRIES,
    METRICS,
    MODULE,
    PENDING,
    Finding,
    Key,
    ToolError,
    Units,
    complexipy_findings,
    directory_findings,
    function_arg_findings,
    function_line_findings,
    jscpd_command,
    jscpd_findings,
    list_files,
    module_line_findings,
    radon_findings,
    module_lines,
    ruff_findings,
    tool_version,
    vulture_findings,
    worse,
    worst_per_unit,
)
from code_metrics_report import Outcome, render

_PYPROJECT = "pyproject.toml"
_ENCODING = "utf-8"
_TOOL_COMMANDS = {
    "ruff": ["ruff"],
    "complexipy": ["complexipy"],
    "radon": ["radon"],
    "vulture": ["vulture"],
}
#: Scalar parameters where a larger value makes the gate see less.
_NO_RAISE = (
    "duplication-min-tokens",
    "duplication-min-lines",
    "dead-code-min-confidence",
    "module-lines-warning",
    "tuple-return-max-elements",
)
#: A scope entry is a plain repository path: no pathspec magic, glob, ``.`` or ``..`` (a dotfile is fine).
_SEGMENT = r"(?!\.\.?(?:/|$))[A-Za-z0-9_.-]+"
_PLAIN_PATH = re.compile(rf"{_SEGMENT}(?:/{_SEGMENT})*/?")
#: The gate's own implementation; an edit to any of these is listed for review.
GATE_FILES = (
    "scripts/code_metrics.py",
    "scripts/code_metrics_checks.py",
    "scripts/code_metrics_collect.py",
    "scripts/code_metrics_design.py",
    "scripts/code_metrics_report.py",
    ".github/workflows/code-metrics.yml",
    ".github/workflows/code-metrics-comment.yml",
    ".github/scripts/code_metrics_comment.js",
)
_MAX_LISTED = 20
#: Exit codes ``checks.git`` accepts when the caller reads the code itself.
_ANY = (0, 1, 128)


@dataclasses.dataclass(frozen=True)
class Config:
    roots: tuple[str, ...]
    exclude: tuple[str, ...]
    thresholds: dict[str, int]
    tools: dict[str, str]
    params: dict[str, int]
    #: The import-linter contracts that define the package units of the A1 check.
    contracts: str
    #: Where the A10 and A12 checks collect enums and implementations.
    vocabulary_roots: tuple[str, ...]


def parse_config(text: str) -> Config | None:
    """The ``[tool.hyperloom.code_metrics]`` table, or None when the file has none."""
    table = tomllib.loads(text).get("tool", {}).get("hyperloom", {}).get("code_metrics")
    if table is None:
        return None
    thresholds = dict(table["thresholds"])
    required = set(METRICS) - set(PENDING)
    if not required <= set(thresholds) <= set(METRICS):
        raise ToolError(
            f"thresholds must name {sorted(required)} and may add {sorted(PENDING)}, got {sorted(thresholds)}"
        )
    return Config(
        roots=_scope(table, "roots"),
        exclude=_scope(table, "exclude"),
        thresholds=thresholds,
        tools=dict(table["tools"]),
        params={name: int(table[name]) for name in _NO_RAISE},
        contracts=_file_path(table, "import-contracts"),
        vocabulary_roots=_scope(table, "vocabulary-roots"),
    )


def _file_path(table: dict, name: str) -> str:
    path = str(table[name])
    if not _PLAIN_PATH.fullmatch(path) or path.endswith("/"):
        raise ToolError(f"`{name}` takes a plain repository file path")
    return path


def _scope(table: dict, name: str) -> tuple[str, ...]:
    paths = tuple(table[name])
    bad = [path for path in paths if not _PLAIN_PATH.fullmatch(path)]
    if bad:
        raise ToolError(f"`{name}` takes plain repository paths, not pathspecs or globs: {bad}")
    return paths


def _is_under(path: str, roots: Iterable[str]) -> bool:
    return any(path.rstrip("/") == r.rstrip("/") or path.startswith(r.rstrip("/") + "/") for r in roots)


def scope_loss(root: Path, head: Config, base: Config) -> list[str]:
    """Files of this tree that ``base``'s scope measures and ``head``'s does not."""
    measured = {
        name: set(sum(list_files(root, c.roots, c.exclude), [])) for name, c in (("head", head), ("base", base))
    }
    lost = sorted(measured["base"] - measured["head"])
    problems = [f"scope no longer measures `{path}`" for path in lost[:_MAX_LISTED]]
    if len(lost) > _MAX_LISTED:
        problems.append(f"...and {len(lost) - _MAX_LISTED} more files dropped from the scope")
    return problems


def gate_changes(root: Path, ref: str, head: Config, base: Config) -> list[str]:
    """Edits to the gate implementation or its tool pins relative to ``ref``."""
    diff = checks.git(root, "diff", "--name-only", ref, "--", *GATE_FILES).stdout
    changes = [f"`{path}` changed" for path in diff.split()]
    changes += [
        f"tool pin `{name}` changed from {base.tools.get(name)} to {head.tools.get(name)}"
        for name in sorted(set(head.tools) | set(base.tools))
        if head.tools.get(name) != base.tools.get(name)
    ]
    return changes


def config_loosening(head: Config, base: Config) -> list[str]:
    """Every way ``head`` lets the gate see less than ``base`` did."""
    problems = [
        f"threshold `{metric}` loosened from {limit} to {head.thresholds.get(metric)}"
        for metric, limit in base.thresholds.items()
        if metric not in head.thresholds or worse(metric, head.thresholds[metric], limit)
    ]
    problems += [f"`exclude` gained `{path}`" for path in sorted(set(head.exclude) - set(base.exclude))]
    problems += [f"`roots` lost `{path}`" for path in sorted(set(base.roots)) if not _is_under(path, head.roots)]
    if head.contracts != base.contracts:
        problems.append(f"`import-contracts` moved from `{base.contracts}` to `{head.contracts}`")
    problems += [
        f"`vocabulary-roots` lost `{path}`"
        for path in sorted(set(base.vocabulary_roots))
        if not _is_under(path, head.vocabulary_roots)
    ]
    problems += [
        f"`{name}` raised from {base.params[name]} to {head.params[name]}"
        for name in _NO_RAISE
        if head.params[name] > base.params[name]
    ]
    return problems


def move_identity(key: Key) -> tuple[str, str]:
    """What a move keeps: the metric and the qualified name, or the file name for a module."""
    metric, path, unit = key
    return metric, (Path(path).name if unit == MODULE else unit)


def match_moves(removed: dict[Key, int], added: dict[Key, int]) -> dict[Key, Key]:
    """Pair each added key with the removed key it moved from: added -> removed.

    A pair needs a :func:`move_identity` that exactly one removed and exactly one added key
    share, and an added value no worse than the removed one. Uniqueness on both sides
    makes the pairing deterministic and stops a worse unit from borrowing the allowance of
    a same-named one (two ``main`` functions, say). The limit: a renamed unit, a method
    moved to another class, or a name shared by several moved units is not matched and has
    to meet the limits as new code.
    """
    olds, news = _by_identity(removed), _by_identity(added)
    return {
        news[identity][0]: keys[0]
        for identity, keys in olds.items()
        if len(keys) == 1
        and len(news.get(identity, ())) == 1
        and not worse(identity[0], added[news[identity][0]], removed[keys[0]])
    }


def _by_identity(keys: Iterable[Key]) -> dict[tuple[str, str], list[Key]]:
    groups: dict[tuple[str, str], list[Key]] = {}
    for key in keys:
        groups.setdefault(move_identity(key), []).append(key)
    return groups


def compare(findings: dict[Key, Finding], base: dict[Key, Finding], outcome: Outcome) -> None:
    """Fill ``outcome.new``, ``.worsened`` and ``.improved`` against ``base``; outside ``outcome.touched``, ``.untouched``."""
    was = {key: f.value for key, f in base.items()}
    added = {key: f.value for key, f in findings.items() if key not in was}
    moves = match_moves({key: value for key, value in was.items() if key not in findings}, added)
    moved = {old: findings[new] for new, old in moves.items()}
    for key, finding in sorted(findings.items()):
        before = was.get(key)
        if key in moves or (before is not None and not worse(finding.metric, finding.value, before)):
            continue
        if not outcome.judged(finding.path):
            outcome.untouched.append((key, before, finding))
        elif before is None:
            outcome.new.append(finding)
        else:
            outcome.worsened.append((finding, before))
    for key, before in sorted(was.items()):
        now = findings.get(key, moved.get(key))
        if outcome.judged(key.path) and (now is None or worse(key.metric, before, now.value)):
            outcome.improved.append((key, before, now))


def check_tools(config: Config) -> dict[str, str]:
    """Refuse to measure with a tool version other than the pinned one."""
    commands = {**_TOOL_COMMANDS, "jscpd": jscpd_command()}
    versions = {name: tool_version(argv) for name, argv in commands.items()}
    wrong = [
        f"{name} {versions[name]} (pinned {pin})" for name, pin in config.tools.items() if versions.get(name) != pin
    ]
    if wrong:
        raise ToolError("tool version mismatch: " + ", ".join(wrong))
    return versions


def per_file_findings(root: Path, files: list[str], limits: dict[str, int], units: Units) -> list[Finding]:
    """The metrics of ``files`` whose values depend on each file alone."""
    if not files:
        return []
    findings = ruff_findings(root, files, limits, units)
    findings += radon_findings(root, files, limits["cyclomatic-complexity"], units)
    findings += complexipy_findings(root, files, limits["cognitive-complexity"], units)
    findings += function_line_findings(files, limits["function-lines"], units)
    findings += function_arg_findings(files, limits["function-args"], units)
    findings += module_line_findings(root, files, limits["module-lines"])
    return findings


def measure(root: Path, config: Config, units: Units) -> dict[Key, Finding]:
    production, tests = list_files(root, config.roots, config.exclude)
    findings = per_file_findings(root, production, config.thresholds, units)
    findings += vulture_findings(root, production, config.params["dead-code-min-confidence"], units)
    tokens, lines = config.params["duplication-min-tokens"], config.params["duplication-min-lines"]
    findings += jscpd_findings(root, production + tests, tokens, lines)
    if DIRECTORY_ENTRIES in config.thresholds:
        findings += directory_findings(production, config.thresholds[DIRECTORY_ENTRIES])
    return worst_per_unit(findings)


def measure_base(root: Path, since: str, config: Config) -> dict[Key, Finding]:
    """The tree at ``since``, measured with this tree's ``config``: what every judged unit is compared with."""
    listed = checks.git(root, "ls-tree", "-z", "--name-only", since, "--", *config.roots).stdout
    roots = [path for path in listed.split("\0") if path]
    with tempfile.TemporaryDirectory() as work:
        if roots:
            tar = _checked(
                ["git", "-C", str(root), "--literal-pathspecs", "archive", "--format=tar", since, "--", *roots]
            )
            _checked(["tar", "-x", "-C", work], tar)
        # ``list_files`` reads the scope from git: track every extracted file, so no ignore rule drops one.
        in_work = ["git", "-C", work]
        _checked([*in_work, "init", "-q"])
        _checked([*in_work, "add", "-A", "-f"])
        return measure(Path(work), config, Units(Path(work)))


def _checked(argv: list[str], data: bytes | None = None) -> bytes:
    proc = subprocess.run(argv, input=data, capture_output=True, check=False)
    if proc.returncode != 0:
        error = proc.stderr.decode(_ENCODING, errors="replace").strip()[-500:]
        raise ToolError(f"{' '.join(argv[:5])} exited {proc.returncode}: {error}")
    return proc.stdout


def merge_base(root: Path, ref: str) -> str:
    """The merge base of ``ref`` and HEAD, or ``ref`` itself when history is too shallow to tell."""
    found = checks.git(root, "merge-base", ref, "HEAD", ok=_ANY)
    return found.stdout.strip() if found.returncode == 0 and found.stdout.strip() else ref


def module_warnings(root: Path, files: Iterable[str], config: Config) -> list[tuple[str, int]]:
    """(path, lines) of modules over the warning length and within the failing one."""
    limit, warn = config.thresholds["module-lines"], config.params["module-lines-warning"]
    sized = ((path, module_lines(root, path)) for path in sorted(files))
    return [(path, lines) for path, lines in sized if warn < lines <= limit]


def git_show(root: Path, ref: str, path: str) -> str | None:
    """``path`` at ``ref``, or None when the ref has no such file; a bad ref is an error."""
    if checks.git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", ok=_ANY).returncode != 0:
        raise ToolError(f"--base-ref {ref} does not resolve to a commit (fetch it first)")
    shown = checks.git(root, "show", f"{ref}:{path}", ok=_ANY)
    return shown.stdout if shown.returncode == 0 else None


def compare_with_base(root: Path, ref: str, config: Config, outcome: Outcome) -> None:
    """Fill ``outcome.loosened`` and ``.gate_changes`` from the config on ``ref``."""
    base_config = parse_config(git_show(root, ref, _PYPROJECT) or "")
    if base_config is None:
        outcome.notes.append(f"`{ref}` has no code-metrics config yet; the loosening check starts once it does.")
        return
    outcome.loosened = config_loosening(config, base_config) + scope_loss(root, config, base_config)
    outcome.gate_changes = gate_changes(root, ref, config, base_config)


def read_pull_request(args: argparse.Namespace, outcome: Outcome) -> checks.PullRequest | None:
    """The PR under test: from the GitHub API now (CI), or a JSON stand-in (local).

    On a push (``--pushed-sha``) it is the PR whose merge made the pushed commit, so a
    reviewed override that passed on the PR still applies after the merge; a push no PR
    made has no label to read and waives nothing.
    """
    if args.pull_request_json:
        return checks.load_pull_request(args.pull_request_json)
    if not (args.pr_number or args.pushed_sha):
        return None
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        raise ToolError("reading the pull request from the API needs GITHUB_REPOSITORY (owner/name)")
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    token = os.environ.get("GITHUB_TOKEN", "")
    number = args.pr_number
    if args.pushed_sha:
        number = checks.merged_pull_request(api, repo, args.pushed_sha, token)
        if number is None:
            outcome.notes.append(f"No merged pull request produced `{args.pushed_sha}`, so no label applies.")
            return None
        outcome.notes.append(f"`{args.pushed_sha}` is the merge of #{number}; its labels apply.")
    return checks.fetch_pull_request(api, repo, number, token)


def run(args: argparse.Namespace, outcome: Outcome) -> int:
    root = args.root.resolve()
    config = parse_config(_read_regular(root, _PYPROJECT) or "")
    if config is None:
        raise ToolError(f"{_PYPROJECT} has no [tool.hyperloom.code_metrics] table")
    outcome.thresholds = config.thresholds
    outcome.module_warning = config.params["module-lines-warning"]
    outcome.versions = check_tools(config)
    if Path(__file__).resolve().parent != (root / "scripts").resolve():
        outcome.notes.append(
            f"Verdict computed by the gate scripts in `{Path(__file__).resolve().parent}`, not by this tree's copy."
        )
    units = Units(root)
    outcome.findings = measure(root, config, units)
    since = None
    if args.base_ref:
        git_show(root, args.base_ref, _PYPROJECT)
        since = merge_base(root, args.base_ref)
        outcome.touched = checks.touched_files(root, since)
        outcome.base = measure_base(root, since, config)
        compare(outcome.findings, outcome.base, outcome)
        compare_with_base(root, args.base_ref, config, outcome)
    outcome.listed = since is None or bool(args.pushed_sha)
    production, _tests = list_files(root, config.roots, config.exclude)
    outcome.module_warnings = module_warnings(root, filter(outcome.judged, production), config)
    outcome.problems = checks.run_all(root, config.roots, config.exclude, units, since)
    if since:
        for problem in design.design_problems(root, design_scope(root, config, since), units, since):
            if problem.check in design.ADVISORY:
                outcome.advisories.append(problem)
            else:
                outcome.design.append(problem)
    pull_request = read_pull_request(args, outcome)
    if pull_request is not None:
        outcome.problems += checks.pull_request_cjk(pull_request)
        outcome.waived = pull_request.raises_baseline
    return 1 if outcome.failed else 0


def design_scope(root: Path, config: Config, since: str) -> design.Scope:
    """The design checks' scope; A1's package units come from the base's contracts when it has them.

    The change cannot redraw the units it is judged by (a contracts file with only the
    root package would make every private name same-unit); a unit it adds counts from
    the next change on.
    """
    contracts = checks.show(root, since, config.contracts) or _read_regular(root, config.contracts)
    if contracts is None:
        raise ToolError(f"`{config.contracts}` (import-contracts) does not exist; the A1 check reads its packages")
    return design.Scope(
        roots=config.roots,
        exclude=config.exclude,
        contracts=contracts,
        vocabulary_roots=config.vocabulary_roots,
        max_tuple=config.params["tuple-return-max-elements"],
    )


def _read_regular(root: Path, path: str) -> str | None:
    """``path`` under ``root`` as text, or None when missing; a symlink or a path outside is refused.

    The PR controls these paths, and the job holds a token: a link to ``/proc/self/environ``
    must not become a parse error quoted in a PR comment.
    """
    full = root / path
    if full.is_symlink() or not full.resolve().is_relative_to(root):
        raise ToolError(f"`{path}` must be a regular file in the repository, not a symlink")
    return full.read_text(encoding=_ENCODING) if full.is_file() else None


def parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base-ref", help="judge the files changed since the merge base with this ref against the tree there"
    )
    parser.add_argument("--report", type=Path, help="also write the Markdown report here")
    parser.add_argument("--link-base", default="", help="prefix for file links, e.g. https://github.com/o/r/blob/<sha>")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pr-number", type=int, help="read this PR's labels, title, body and commits from the API")
    source.add_argument("--pull-request-json", type=Path, help="a local stand-in for --pr-number (no API call)")
    source.add_argument("--pushed-sha", help="on a push: read the PR whose merge made this commit from the API")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    outcome = Outcome()
    try:
        code = run(args, outcome)
    except ToolError as exc:
        outcome.error = str(exc)
        code = 2
    if args.report:
        args.report.write_text(render(outcome, args.link_base), encoding=_ENCODING)
    # The log gets every row (the report truncates long sections), waived ones included.
    print(render(outcome, args.link_base, max_rows=None))
    return code


if __name__ == "__main__":
    sys.exit(main())
