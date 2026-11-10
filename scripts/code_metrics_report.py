# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Markdown report of one code-metrics gate run (PR comment, job summary and job log).

The first line is a hidden marker the CI comment step uses to find and update its one
sticky comment instead of adding a new one per push.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from code_metrics_collect import METRICS, MODULE, MODULE_LINES, PENDING, Finding, Key

MARKER = "<!-- code-metrics-report -->"
OVERRIDE_LABEL = "baseline-raise"
GATE_CHANGED = "Gate implementation changed (needs review)"
WAIVED = f" (waived by the `{OVERRIDE_LABEL}` label)"
ADVISORY = " (advisory, never fails)"
_MAX_ROWS = 25
_SEP = " | "
_UNIT, _DIMENSION, _BASE, _NOW = "Unit", "Dimension", "Base", "Now"
_UNIT_CHANGE = [_UNIT, _DIMENSION, _BASE, _NOW]
_UNIT_LIMIT = [_UNIT, _DIMENSION, "Value", "Limit"]
#: Section title and how-to-fix line of each check in ``code_metrics_checks``.
CHECKS = {
    "comments": (
        "Comments",
        "a comment block of at most 8 lines that says what the code does and why; the history (PR, issue, "
        "incident) goes in the commit message and the PR description",
    ),
    "broad-noqa": (
        "Broad catch without a reason",
        "say on the `# noqa: BLE001` line why the handler may catch every exception (an outermost boundary "
        "that logs it); otherwise catch the exceptions the block can raise",
    ),
    "broad-unrecorded": (
        "Broad catch that records nothing",
        "re-raise, or log the error with its traceback at the outermost boundary or the one-item isolation "
        "boundary that owns it; otherwise catch the exceptions the block can raise or let them propagate",
    ),
    "english": ("English only", "English text only (no CJK characters) in files, PR title, body and commits"),
    "test-imports": (
        "Production code imports test code",
        "move what both need into a non-test module and import that from the test and the production code",
    ),
    "literals": (
        "Repeated literals",
        "name the value once as a module-level constant and use the name",
    ),
    "private-reach": (
        "A1 Private name of another package",
        "make the name public in the package that owns it (or move the caller into that package); "
        "the units are the root packages, layers and independent modules of `.importlinter`",
    ),
    "flag-argument": (
        "A8 Flag argument",
        "split the function into one function per value of the flag and let the caller pick the one it means",
    ),
    "tuple-return": (
        "A9 Tuple return",
        "return a dataclass or NamedTuple whose fields name the parts, so callers read them by name",
    ),
    "raw-vocabulary": (
        "A10 Raw vocabulary comparison",
        "compare with the enum member instead of its string value; convert outside input to the enum once, "
        "where it enters",
    ),
    "global-write": (
        "A11 Hidden global write",
        "pass the value in and return the result, or keep the state on an object the caller owns",
    ),
    "forwarder": (
        "A12 Pure forwarder",
        "call the target directly, or give the layer a job of its own (validation, conversion, a narrower type)",
    ),
    "single-implementation": (
        "A12 Single-implementation abstraction",
        "use the one implementation directly until a second one exists; a lower package declaring an interface "
        "for a higher one to implement is exempt",
    ),
}


@dataclasses.dataclass
class Outcome:
    """Everything one run decided; ``failed`` is the gate verdict."""

    error: str | None = None
    thresholds: dict[str, int] = dataclasses.field(default_factory=dict)
    versions: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Every unit over a threshold in this tree, and in the tree at the base (None without a base).
    findings: dict[Key, Finding] = dataclasses.field(default_factory=dict)
    base: dict[Key, Finding] | None = None
    #: Files the change touches; None without a base to diff against.
    touched: set[str] | None = None
    #: Units over a threshold the base did not have over it; (finding, base value) for units worse than there.
    new: list[Finding] = dataclasses.field(default_factory=list)
    worsened: list[tuple[Finding, int]] = dataclasses.field(default_factory=list)
    #: (base key, base value, the tree's finding: None when within the limit or gone, another key when moved)
    improved: list[tuple[Key, int, Finding | None]] = dataclasses.field(default_factory=list)
    #: New or worse units in files the change does not touch: (key, base value or None, finding).
    untouched: list[tuple[Key, int | None, Finding]] = dataclasses.field(default_factory=list)
    #: List every unit over a threshold in full (no base, or a push run's job summary).
    listed: bool = False
    loosened: list[str] = dataclasses.field(default_factory=list)
    #: Edits to the gate's own implementation or tool pins; reported, not a failure.
    gate_changes: list[str] = dataclasses.field(default_factory=list)
    #: (path, lines) of judged modules between the warning and the failing length.
    module_warnings: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    #: ``module-lines-warning``: a longer module is listed as a warning, never failed.
    module_warning: int | None = None
    #: Refusals of the checks without a baseline (``code_metrics_checks.Problem``).
    problems: list = dataclasses.field(default_factory=list)
    #: Failing design findings (A1, A8, A12 forwarder); the override label waives them.
    design: list = dataclasses.field(default_factory=list)
    #: Design findings the A rules rate advisory (A9, A10, A11, A12 single implementation).
    advisories: list = dataclasses.field(default_factory=list)
    #: The PR carries the override label: new, worse and design are listed but do not fail.
    waived: bool = False
    notes: list[str] = dataclasses.field(default_factory=list)

    def judged(self, path: str) -> bool:
        """A file is judged when the change touches it, a directory when it touches a file under it."""
        if self.touched is None or path in self.touched:
            return True
        return any(touched.startswith(path + "/") for touched in self.touched)

    @property
    def waivable_failed(self) -> bool:
        """Findings the override label waives: new, worse and failing design checks."""
        return bool(self.new or self.worsened or self.design)

    @property
    def failed(self) -> bool:
        waivable = self.waivable_failed and not self.waived
        return bool(self.error or waivable or self.loosened or self.problems)


def render(outcome: Outcome, link_base: str = "", max_rows: int | None = _MAX_ROWS) -> str:
    if outcome.error:
        return f"{MARKER}\n## Code metrics gate: could not run\n\n```\n{outcome.error}\n```\n"
    verdict = "FAILED" if outcome.failed else "PASSED"
    if outcome.waived and outcome.waivable_failed:
        verdict += f" with waived findings (`{OVERRIDE_LABEL}`)"
    parts = [f"{MARKER}\n## Code metrics gate: {verdict}\n", _scope(outcome), _summary(outcome)]
    parts += _sections(outcome, link_base, max_rows)
    if outcome.failed or outcome.waived:
        parts.append(_how_to_fix(outcome))
    parts += _informational(outcome, link_base, max_rows)
    parts += [f"> {note}\n" for note in outcome.notes]
    tools = ", ".join(f"{name} {version}" for name, version in sorted(outcome.versions.items()))
    parts.append(
        f"<sub>Thresholds and sources: `[tool.hyperloom.code_metrics]` in pyproject.toml. Tools: {tools}.</sub>\n"
    )
    return "\n".join(part for part in parts if part)


def _scope(outcome: Outcome) -> str:
    if outcome.touched is None:
        return "No base to compare with: every unit over a threshold is listed below, none fails.\n"
    return (
        f"Judged: the {len(outcome.touched)} files this change touches. Files it does not touch never fail it; "
        "anything found there is listed under *Outside this change* for information.\n"
    )


def _summary(outcome: Outcome) -> str:
    counts = {
        "new": Counter(f.metric for f in outcome.new),
        "worse": Counter(f.metric for f, _ in outcome.worsened),
        "improved": Counter(key.metric for key, _, _ in outcome.improved),
        "head": Counter(key.metric for key in outcome.findings),
        "base": Counter(key.metric for key in outcome.base or {}),
    }
    rows = ["| Dimension | Limit | New | Worse | Improved | Over the limit | vs base |"]
    rows.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, metric in METRICS.items():
        if name in PENDING and name not in outcome.thresholds:
            continue
        delta = "n/a" if outcome.base is None else f"{counts['head'][name] - counts['base'][name]:+d}"
        cells = [counts[c][name] for c in ("new", "worse", "improved", "head")]
        rows.append(
            f"| {metric.label} | {_dimension_limit(name, outcome)} | " + _SEP.join(map(str, cells)) + f" | {delta} |"
        )
    found = Counter(problem.check for problem in _all_problems(outcome))
    effects = {p.check: "advisory, never fails" for p in outcome.advisories}
    effects.update({p.check: f"fails; `{OVERRIDE_LABEL}` waives" for p in outcome.design})
    rows += ["", "| Check | Findings | Effect |", "|---|---:|---|"]
    rows += [
        f"| {title} | {found[name]} | {effects.get(name, 'fails') if found[name] else ''} |"
        for name, (title, _fix) in CHECKS.items()
    ]
    return "\n".join(rows) + "\n"


def _limit(metric: str, thresholds: dict[str, int]) -> str:
    return f"fails > {thresholds.get(metric, '?')}"


def _dimension_limit(metric: str, outcome: Outcome) -> str:
    """Every level of the dimension: module length also warns, below its failing length."""
    warning, failing = outcome.module_warning, outcome.thresholds.get(MODULE_LINES)
    if metric == MODULE_LINES and warning is not None and failing is not None and warning < failing:
        return f"{_warn_limit(outcome)}, {_limit(metric, outcome.thresholds)}"
    return _limit(metric, outcome.thresholds)


def _warn_limit(outcome: Outcome) -> str:
    return f"warns > {_or_unknown(outcome.module_warning)}"


def _or_unknown(value: int | None) -> object:
    return "?" if value is None else value


def _sections(outcome: Outcome, link_base: str, max_rows: int | None) -> list[str]:
    limits = outcome.thresholds
    waived = WAIVED if outcome.waived else ""
    sections = [
        _table(
            f"New violations (not at the base){waived}",
            _UNIT_LIMIT,
            [[_unit(f, link_base), _label(f.metric), f.value, _limit(f.metric, limits)] for f in outcome.new],
            max_rows,
        ),
        _table(
            f"Worse than at the base{waived}",
            _UNIT_CHANGE,
            [[_unit(f, link_base), _label(f.metric), was, f.value] for f, was in outcome.worsened],
            max_rows,
        ),
    ]
    for name, (title, _fix) in CHECKS.items():
        for problems, suffix in ((outcome.problems, ""), (outcome.design, waived), (outcome.advisories, ADVISORY)):
            rows = [[_where(p, link_base), p.message] for p in problems if p.check == name]
            sections.append(_table(title + suffix, ["Where", "Problem"], rows, max_rows))
    if outcome.loosened:
        sections.append("### Gate configuration loosened\n\n" + "".join(f"- {p}\n" for p in outcome.loosened))
    if outcome.gate_changes:
        sections.append(
            f"### {GATE_CHANGED}\n\n> [!WARNING]\n> This change edits the gate itself. CI judged it with the base "
            "branch's copy of the gate scripts; a reviewer has to approve the gate change on its own merits.\n\n"
            + "".join(f"- {c}\n" for c in outcome.gate_changes)
        )
    return sections


def _informational(outcome: Outcome, link_base: str, max_rows: int | None) -> list[str]:
    """Sections that never fail the gate."""
    warning, failing = _or_unknown(outcome.module_warning), _or_unknown(outcome.thresholds.get(MODULE_LINES))
    warn = _table(
        f"Module length warning: over {warning} lines, within the {failing}-line failure limit (not failing)",
        ["Module", "Lines", "Limit"],
        [[f"`{path}`", lines, _warn_limit(outcome)] for path, lines in outcome.module_warnings],
        max_rows,
    )
    improved = _table(
        "Improved by this change (informational)",
        _UNIT_CHANGE,
        [[_key(key), _label(key.metric), was, _now(key, now, link_base)] for key, was, now in outcome.improved],
        max_rows,
    )
    rows = [
        [_key(key), _label(key.metric), "absent" if was is None else was, _now(key, now, link_base)]
        for key, was, now in outcome.untouched
    ]
    outside = _folded(
        f"{len(rows)} findings in files this change does not touch",
        _table("Outside this change (informational)", _UNIT_CHANGE, rows, max_rows),
    )
    # Every row, whatever ``max_rows``: this list is what the backlog is.
    rows = [
        [_unit(f, link_base), _label(f.metric), f.value, _limit(f.metric, outcome.thresholds)]
        for _, f in sorted(outcome.findings.items())
    ]
    inventory = _folded(
        f"{len(rows)} units over a threshold in this tree",
        _table("Over the limit in this tree", _UNIT_LIMIT, rows if outcome.listed else [], None),
    )
    return [warn, improved, outside, inventory]


def _folded(summary: str, table: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{table}\n</details>\n" if table else ""


def _all_problems(outcome: Outcome) -> list:
    return outcome.problems + outcome.design + outcome.advisories


def _table(title: str, header: list[str], rows: list[list[object]], max_rows: int | None) -> str:
    if not rows:
        return ""
    shown = rows if max_rows is None else rows[:max_rows]
    lines = [f"### {title}: {len(rows)}\n", "| " + _SEP.join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + _SEP.join(_cell(c) for c in row) + " |" for row in shown]
    if len(rows) > len(shown):
        lines.append(f"\n...and {len(rows) - len(shown)} more; the job log lists every row.")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _label(metric: str) -> str:
    return METRICS[metric].label


def _key(key: Key) -> str:
    _metric, path, unit = key
    return f"`{path}`" if unit == MODULE else f"`{path}` `{unit}`"


def _link(path: str, line: int, link_base: str) -> str:
    where = f"{path}:{line}"
    return f"[`{where}`]({link_base}/{path}#L{line})" if link_base else f"`{where}`"


def _unit(finding: Finding, link_base: str) -> str:
    link = _link(finding.path, finding.line, link_base)
    return link if finding.unit == MODULE else f"{link} `{finding.unit}`"


def _where(problem: object, link_base: str) -> str:
    """A file line as a link; PR title, body or a commit message as plain text."""
    path, line = problem.where, problem.line
    if path.startswith(("PR ", "commit ")):
        return f"{path}, line {line}"
    return _link(path, line, link_base)


def _now(key: Key, finding: Finding | None, link_base: str) -> str:
    if finding is None:
        return "within the limit, or removed"
    if finding.key != key:
        return f"moved to {_unit(finding, link_base)} ({finding.value})"
    return str(finding.value)


def _how_to_fix(outcome: Outcome) -> str:
    pins = outcome.versions
    pip = " ".join(f"{name}=={pins[name]}" for name in ("ruff", "complexipy", "radon", "vulture") if name in pins)
    lines = [
        "### How to fix\n",
        (
            "- **New or worse**: split the unit (extract a function, a class, a module) until it is back under the "
            "limit, or no worse than the base branch has it. There is nothing to record: the base is the reference."
        ),
    ]
    lines += [
        f"- **{title}**: {fix}."
        for name, (title, fix) in CHECKS.items()
        if any(p.check == name for p in _all_problems(outcome))
    ]
    lines.append(
        f"- **Override**: the `{OVERRIDE_LABEL}` PR label waives new and worse findings and the "
        "failing design checks (they stay listed here). Use it only with the reason in the PR description; it "
        "waives nothing else."
    )
    lines.append(
        f"- Tools for a local run: `pip install {pip}` and `npm install -g jscpd@{pins.get('jscpd', '?')}` "
        "(or point `CODE_METRICS_JSCPD` at another jscpd command, such as `npx --yes jscpd@<version>`)."
    )
    return "\n".join(lines) + "\n"
