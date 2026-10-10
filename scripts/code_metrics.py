# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""White-box code-metrics gate: thresholds over a ratcheting baseline, plus absolute checks.

The dimensions, their thresholds (with the reason for each number) and the pinned tool
versions live in ``pyproject.toml`` under ``[tool.hyperloom.code_metrics]``. Units that
were already over a threshold when the gate landed are recorded with their current
value in the baseline file named there, one line per unit. The baseline only goes down.

With ``--base-ref``, only files the change touches (the diff of the working tree against
the merge base with that ref) are judged; the rest of the tree is reported for
information and never fails an unrelated change. In a touched file:

* a unit that is not in the baseline and is over a threshold fails, unless the same
  file at the base already had it at this value or worse (the backlog is the base's, as
  in the style guide's complexity ceiling);
* a baselined unit whose value got worse than its recorded value and than its value at
  the base fails;
* a baselined unit that improved, dropped under the threshold or disappeared fails as
  *out of date* until the baseline is tightened in the same change, with
  ``--update-baseline`` (which can only lower or remove entries).

Whole-file, not scoped to the touched files: the baseline may only lose entries or lower
values relative to the baseline on ``--base-ref``, and the thresholds, scope and tool
parameters may not loosen (the scope is compared by the files each config measures in
this tree, not by its text). Editing the baseline is not a way to make the gate pass.

The pull-request label ``baseline-raise``, read from the GitHub API at run time
(``--pr-number``; ``--pull-request-json`` stands in for the API in a local run), turns
baseline growth and new or worse units into *waived* findings: still listed in full in
the log and the report, no longer failing. It waives nothing else.

Next to the ratchet, ``code_metrics_checks`` runs checks that have no baseline: added
comment blocks and history in comments, CJK characters (in the tree and, on a pull
request, in its title, body and commit messages), production imports of test code, and
added repeated literals. The diff-only ones need ``--base-ref``.

CI runs the base branch's copy of these scripts against the change, so editing the gate
does not change the verdict on the same change; the report lists any edit to the gate
implementation or its tool pins for a reviewer.

A unit that only moved to another file (same metric, same qualified name, or the same
file name for a module; value no worse) keeps the baseline entry it left; see
:func:`match_moves` for the limit.

Usage::

    pip install ruff==0.16.2 complexipy==8.0.1 vulture==2.16
    npm install -g jscpd@5.4.1      # or CODE_METRICS_JSCPD="npx --yes jscpd@5.4.1"

    python scripts/code_metrics.py                         # judge every file
    python scripts/code_metrics.py --base-ref origin/main  # judge the files changed since the merge base
    python scripts/code_metrics.py --update-baseline       # tighten after an improvement
    python scripts/code_metrics.py --seed-baseline         # write a missing baseline from the tree

Exit code 0 when the gate passes, 1 when it fails, 2 when it could not run (a tool is
missing or at another version, a ref does not resolve, the API could not be read).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import code_metrics_checks as checks
from code_metrics_collect import (
    METRICS,
    MODULE,
    Finding,
    ToolError,
    Units,
    complexipy_findings,
    function_line_findings,
    jscpd_command,
    jscpd_findings,
    list_files,
    module_line_findings,
    module_lines,
    ruff_findings,
    tool_version,
    vulture_findings,
    worse,
    worst_per_unit,
)
from code_metrics_report import UPDATE_COMMAND, Outcome, render

Key = tuple[str, str, str]
_PYPROJECT = "pyproject.toml"
_ENCODING = "utf-8"
#: Metrics a single file determines, so the same file at the base can be measured alone.
PER_FILE = ("cyclomatic-complexity", "cognitive-complexity", "function-lines", "nested-blocks", "module-lines")
#: A unit's place in a baseline line: ``<path>::<unit>``; a module-level entry is the path alone.
_UNIT_SEPARATOR = "::"
_BASELINE_HEADER = f"""\
# Code-metrics baseline: units already over a threshold in [tool.hyperloom.code_metrics],
# one per line as "<metric> <path>::<unit> <value>" ("<metric> <path> <value>" for a module).
# Entries may only be lowered or removed, never added or raised: {UPDATE_COMMAND}
# A file is judged only when a change touches it; the label baseline-raise waives growth.
"""
_TOOL_COMMANDS = {
    "ruff": ["ruff"],
    "complexipy": ["complexipy"],
    "vulture": ["vulture"],
}
#: Scalar parameters where a larger value makes the gate see less.
_NO_RAISE = (
    "duplication-min-tokens",
    "duplication-min-lines",
    "dead-code-min-confidence",
    "module-lines-warning",
)
#: A scope entry is a plain repository path: no pathspec magic, glob, ``.`` or ``..``.
_PLAIN_PATH = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*/?")
#: The gate's own implementation; an edit to any of these is listed for review.
GATE_FILES = (
    "scripts/code_metrics.py",
    "scripts/code_metrics_checks.py",
    "scripts/code_metrics_collect.py",
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
    baseline: str
    roots: tuple[str, ...]
    exclude: tuple[str, ...]
    thresholds: dict[str, int]
    tools: dict[str, str]
    params: dict[str, int]


def parse_config(text: str) -> Config | None:
    """The ``[tool.hyperloom.code_metrics]`` table, or None when the file has none."""
    table = tomllib.loads(text).get("tool", {}).get("hyperloom", {}).get("code_metrics")
    if table is None:
        return None
    thresholds = dict(table["thresholds"])
    if set(thresholds) != set(METRICS):
        raise ToolError(f"thresholds must name exactly {sorted(METRICS)}, got {sorted(thresholds)}")
    return Config(
        baseline=table["baseline"],
        roots=_scope(table, "roots"),
        exclude=_scope(table, "exclude"),
        thresholds=thresholds,
        tools=dict(table["tools"]),
        params={name: int(table[name]) for name in _NO_RAISE},
    )


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
    problems += [
        f"`{name}` raised from {base.params[name]} to {head.params[name]}"
        for name in _NO_RAISE
        if head.params[name] > base.params[name]
    ]
    return problems


def load_baseline(text: str) -> dict[Key, int]:
    """Parse the one-line-per-entry baseline; a malformed line is an error, not a skip."""
    entries: dict[Key, int] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = _parse_entry(line, number)
        if key in entries:
            raise ToolError(f"baseline line {number} repeats `{line}`")
        entries[key] = value
    return entries


def _parse_entry(line: str, number: int) -> tuple[Key, int]:
    metric, _, rest = line.partition(" ")
    where, _, value = rest.rpartition(" ")
    if metric not in METRICS or not where or not value.isdigit():
        raise ToolError(f"baseline line {number} is not `<metric> <path>[::<unit>] <value>`: {line}")
    path, separator, unit = where.partition(_UNIT_SEPARATOR)
    return (metric, path, unit if separator else MODULE), int(value)


def baseline_line(key: Key, value: int) -> str:
    metric, path, unit = key
    where = path if unit == MODULE else f"{path}{_UNIT_SEPARATOR}{unit}"
    return f"{metric} {where} {value}"


def dump_baseline(entries: dict[Key, int]) -> str:
    return _BASELINE_HEADER + "".join(baseline_line(key, value) + "\n" for key, value in sorted(entries.items()))


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


def compare(findings: dict[Key, Finding], baseline: dict[Key, int], outcome: Outcome) -> None:
    """Fill ``outcome.new``, ``.worsened`` and ``.stale``; outside ``outcome.touched``, ``.untouched``."""
    new = {key: f for key, f in findings.items() if key not in baseline}
    gone = {key: value for key, value in baseline.items() if key not in findings}
    moves = match_moves(gone, {key: f.value for key, f in new.items()})
    moved_from = {old: findings[key] for key, old in moves.items()}
    fresh = [f for key, f in sorted(new.items()) if key not in moves]
    worsened = [
        (f, baseline[key])
        for key, f in sorted(findings.items())
        if key in baseline and worse(f.metric, f.value, baseline[key])
    ]
    stale = [(key, value, moved_from.get(key)) for key, value in sorted(gone.items())]
    stale += [
        (key, value, findings[key])
        for key, value in sorted(baseline.items())
        if key in findings and worse(key[0], value, findings[key].value)
    ]
    judged = outcome.judged
    outcome.new = [f for f in fresh if judged(f.path)]
    outcome.worsened = [(f, was) for f, was in worsened if judged(f.path)]
    outcome.stale = [entry for entry in stale if judged(entry[0][1])]
    outcome.untouched = [(f.key, None, f) for f in fresh if not judged(f.path)]
    outcome.untouched += [(f.key, was, f) for f, was in worsened if not judged(f.path)]
    outcome.untouched += [entry for entry in stale if not judged(entry[0][1])]


def growth(head: dict[Key, int], base: dict[Key, int]) -> list[tuple[Key, int | None, int]]:
    """Entries ``head`` adds or raises over ``base``: (key, base value or None, head value)."""
    added = {key: value for key, value in head.items() if key not in base}
    moves = match_moves({key: value for key, value in base.items() if key not in head}, added)
    grown: list[tuple[Key, int | None, int]] = [(key, None, value) for key, value in added.items() if key not in moves]
    grown += [(key, base[key], value) for key, value in head.items() if key in base and worse(key[0], value, base[key])]
    return sorted(grown)


def tightened(findings: dict[Key, Finding], baseline: dict[Key, int]) -> dict[Key, int]:
    """The baseline with every entry lowered to the tree's value or dropped; never raised, never grown."""
    out = {
        key: (findings[key].value if worse(key[0], value, findings[key].value) else value)
        for key, value in baseline.items()
        if key in findings
    }
    new = {key: f.value for key, f in findings.items() if key not in baseline}
    gone = {key: value for key, value in baseline.items() if key not in findings}
    out.update((key, new[key]) for key in match_moves(gone, new))
    return out


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
    """The :data:`PER_FILE` metrics of ``files``: each file's values depend on that file alone."""
    if not files:
        return []
    findings = ruff_findings(root, files, limits, units)
    findings += complexipy_findings(root, files, limits["cognitive-complexity"], units)
    findings += function_line_findings(files, limits["function-lines"], units)
    findings += module_line_findings(root, files, limits["module-lines"])
    return findings


def measure(root: Path, config: Config, units: Units) -> dict[Key, Finding]:
    production, tests = list_files(root, config.roots, config.exclude)
    findings = per_file_findings(root, production, config.thresholds, units)
    findings += vulture_findings(root, production, config.params["dead-code-min-confidence"], units)
    tokens, lines = config.params["duplication-min-tokens"], config.params["duplication-min-lines"]
    findings += jscpd_findings(root, production + tests, tokens, lines)
    return worst_per_unit(findings)


def base_values(root: Path, since: str, paths: Iterable[str], config: Config) -> dict[Key, int]:
    """:data:`PER_FILE` values over the thresholds of ``paths`` as they were at ``since``."""
    with tempfile.TemporaryDirectory() as work:
        tree = Path(work)
        present = []
        for path in sorted(set(paths)):
            text = checks.show(root, since, path)
            if text is not None:
                (tree / path).parent.mkdir(parents=True, exist_ok=True)
                (tree / path).write_text(text, encoding=_ENCODING)
                present.append(path)
        found = worst_per_unit(per_file_findings(tree, present, config.thresholds, Units(tree)))
    return {key: f.value for key, f in found.items()}


def excuse_base_backlog(outcome: Outcome, base: dict[Key, int]) -> None:
    """Move new or worse units the base already had at that value or worse to ``outcome.base_backlog``.

    This is the complexity ceiling's comparison: the verdict is head against the merge
    base, so a unit the change did not make worse is backlog even when the baseline
    missed it (main pushed past the gate) or recorded it lower.
    """

    def carried(finding: Finding) -> bool:
        return finding.key in base and not worse(finding.metric, finding.value, base[finding.key])

    outcome.base_backlog = [(f, base[f.key]) for f in outcome.new if carried(f)]
    outcome.base_backlog += [(f, base[f.key]) for f, _was in outcome.worsened if carried(f)]
    outcome.new = [f for f in outcome.new if not carried(f)]
    outcome.worsened = [(f, was) for f, was in outcome.worsened if not carried(f)]


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
    """Fill ``outcome.growth`` / ``.loosened`` from the baseline and config on ``ref``."""
    base_config = parse_config(git_show(root, ref, _PYPROJECT) or "")
    if base_config is None:
        outcome.notes.append(f"`{ref}` has no code-metrics config yet; growth and loosening checks start once it does.")
        return
    base_text = git_show(root, ref, base_config.baseline)
    if base_text is None:
        raise ToolError(f"{ref} configures the baseline {base_config.baseline} but has no such file")
    outcome.base_baseline = load_baseline(base_text)
    outcome.growth = growth(outcome.baseline, outcome.base_baseline)
    outcome.loosened = config_loosening(config, base_config) + scope_loss(root, config, base_config)
    outcome.gate_changes = gate_changes(root, ref, config, base_config)


def read_pull_request(args: argparse.Namespace) -> checks.PullRequest | None:
    """The PR under test: from the GitHub API now (CI), or a JSON stand-in (local)."""
    if args.pull_request_json:
        return checks.load_pull_request(args.pull_request_json)
    if not args.pr_number:
        return None
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        raise ToolError("--pr-number needs GITHUB_REPOSITORY (owner/name)")
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    return checks.fetch_pull_request(api, repo, args.pr_number, os.environ.get("GITHUB_TOKEN", ""))


def run(args: argparse.Namespace, outcome: Outcome) -> int:
    root = args.root.resolve()
    config = parse_config((root / _PYPROJECT).read_text(encoding=_ENCODING))
    if config is None:
        raise ToolError(f"{_PYPROJECT} has no [tool.hyperloom.code_metrics] table")
    outcome.thresholds = config.thresholds
    outcome.baseline_path = config.baseline
    outcome.versions = check_tools(config)
    if Path(__file__).resolve().parent != (root / "scripts").resolve():
        outcome.notes.append(
            f"Verdict computed by the gate scripts in `{Path(__file__).resolve().parent}`, not by this tree's copy."
        )
    baseline_path = root / config.baseline
    units = Units(root)
    findings = measure(root, config, units)
    if args.seed_baseline:
        return seed(baseline_path, findings)
    if baseline_path.is_file():
        outcome.baseline = load_baseline(baseline_path.read_text(encoding=_ENCODING))
    else:
        outcome.notes.append(f"`{config.baseline}` does not exist, so every unit over a threshold counts as new.")
    if args.update_baseline:
        return update(baseline_path, findings, outcome.baseline)
    since = None
    if args.base_ref:
        git_show(root, args.base_ref, _PYPROJECT)
        since = merge_base(root, args.base_ref)
        outcome.touched = checks.touched_files(root, since)
    compare(findings, outcome.baseline, outcome)
    if since:
        touched_units = {f.path for f in outcome.new} | {f.path for f, _ in outcome.worsened}
        excuse_base_backlog(outcome, base_values(root, since, touched_units, config))
        compare_with_base(root, args.base_ref, config, outcome)
    production, _tests = list_files(root, config.roots, config.exclude)
    outcome.module_warnings = module_warnings(root, filter(outcome.judged, production), config)
    outcome.problems = checks.run_all(root, config.roots, config.exclude, units, since)
    pull_request = read_pull_request(args)
    if pull_request is not None:
        outcome.problems += checks.pull_request_cjk(pull_request)
        outcome.waived = pull_request.raises_baseline
    return 1 if outcome.failed else 0


def seed(path: Path, findings: dict[Key, Finding]) -> int:
    if path.exists():
        raise ToolError(f"{path} exists; --seed-baseline only creates a missing baseline")
    path.write_text(dump_baseline({key: f.value for key, f in findings.items()}), encoding=_ENCODING)
    print(f"wrote {len(findings)} entries to {path}")
    return 0


def update(path: Path, findings: dict[Key, Finding], baseline: dict[Key, int]) -> int:
    entries = tightened(findings, baseline)
    path.write_text(dump_baseline(entries), encoding=_ENCODING)
    lowered = sum(1 for key, value in entries.items() if key in baseline and value != baseline[key])
    print(f"baseline: {len(baseline)} -> {len(entries)} entries, {lowered} lowered; written to {path}")
    return 0


def parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base-ref", help="judge only the files changed since the merge base with this ref; refuse baseline growth"
    )
    parser.add_argument("--report", type=Path, help="also write the Markdown report here")
    parser.add_argument("--link-base", default="", help="prefix for file links, e.g. https://github.com/o/r/blob/<sha>")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pr-number", type=int, help="read this PR's labels, title, body and commits from the API")
    source.add_argument("--pull-request-json", type=Path, help="a local stand-in for --pr-number (no API call)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--update-baseline", action="store_true", help="lower or remove baseline entries to match the tree"
    )
    mode.add_argument("--seed-baseline", action="store_true", help="create a missing baseline from the current tree")
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    outcome = Outcome()
    try:
        code = run(args, outcome)
    except ToolError as exc:
        outcome.error = str(exc)
        code = 2
    if args.update_baseline or args.seed_baseline:
        if outcome.error:
            print(f"code-metrics: {outcome.error}", file=sys.stderr)
        return code
    if args.report:
        args.report.write_text(render(outcome, args.link_base), encoding=_ENCODING)
    # The log gets every row (the report truncates long sections), waived ones included.
    print(render(outcome, args.link_base, max_rows=None))
    return code


if __name__ == "__main__":
    sys.exit(main())
