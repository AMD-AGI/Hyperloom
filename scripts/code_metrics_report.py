# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Markdown report of one code-metrics gate run (PR comment, job summary and job log).

The first line is a hidden marker the CI comment step uses to find and update its one
sticky comment instead of adding a new one per push.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from code_metrics_collect import METRICS, MODULE, MODULE_LINES, Finding

MARKER = "<!-- code-metrics-report -->"
UPDATE_COMMAND = "python scripts/code_metrics.py --update-baseline"
OVERRIDE_LABEL = "baseline-raise"
GATE_CHANGED = "Gate implementation changed (needs review)"
WAIVED = f" (waived by the `{OVERRIDE_LABEL}` label)"
_MAX_ROWS = 25
Key = tuple[str, str, str]
_SEP = " | "
_UNIT, _DIMENSION, _BASELINE, _BASE, _NOW = "Unit", "Dimension", "Baseline", "Base", "Now"
_UNIT_CHANGE = [_UNIT, _DIMENSION, _BASELINE, _NOW]
#: Section title and how-to-fix line of each check in ``code_metrics_checks``.
CHECKS = {
    "comments": (
        "Comments",
        "a comment block of at most 8 lines that says what the code does and why; the history (PR, issue, "
        "incident) goes in the commit message and the PR description",
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
}


@dataclasses.dataclass
class Outcome:
    """Everything one run decided; ``failed`` is the gate verdict."""

    error: str | None = None
    baseline_path: str = ""
    thresholds: dict[str, int] = dataclasses.field(default_factory=dict)
    versions: dict[str, str] = dataclasses.field(default_factory=dict)
    baseline: dict[Key, int] = dataclasses.field(default_factory=dict)
    base_baseline: dict[Key, int] | None = None
    #: Files the change touches; None judges every file (no base to diff against).
    touched: set[str] | None = None
    new: list[Finding] = dataclasses.field(default_factory=list)
    worsened: list[tuple[Finding, int]] = dataclasses.field(default_factory=list)
    #: (baseline key, recorded value, the tree's finding: None when gone, another key when moved)
    stale: list[tuple[Key, int, Finding | None]] = dataclasses.field(default_factory=list)
    #: New, worse or out-of-date units in files the change does not touch: (key, recorded or None, finding).
    untouched: list[tuple[Key, int | None, Finding | None]] = dataclasses.field(default_factory=list)
    #: New or worse units the base already had at that value or worse: (finding, base value).
    base_backlog: list[tuple[Finding, int]] = dataclasses.field(default_factory=list)
    #: (key, base value or None when added, head value)
    growth: list[tuple[Key, int | None, int]] = dataclasses.field(default_factory=list)
    loosened: list[str] = dataclasses.field(default_factory=list)
    #: Edits to the gate's own implementation or tool pins; reported, not a failure.
    gate_changes: list[str] = dataclasses.field(default_factory=list)
    #: (path, lines) of judged modules between the warning and the failing length.
    module_warnings: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    #: ``module-lines-warning``: a longer module is listed as a warning, never failed.
    module_warning: int | None = None
    #: Refusals of the checks without a baseline (``code_metrics_checks.Problem``).
    problems: list = dataclasses.field(default_factory=list)
    #: The PR carries the override label: new, worse and growth are listed but do not fail.
    waived: bool = False
    notes: list[str] = dataclasses.field(default_factory=list)

    def judged(self, path: str) -> bool:
        return self.touched is None or path in self.touched

    @property
    def ratchet_failed(self) -> bool:
        return bool(self.new or self.worsened or self.growth)

    @property
    def failed(self) -> bool:
        waivable = self.ratchet_failed and not self.waived
        return bool(self.error or waivable or self.stale or self.loosened or self.problems)


def render(outcome: Outcome, link_base: str = "", max_rows: int | None = _MAX_ROWS) -> str:
    if outcome.error:
        return f"{MARKER}\n## Code metrics gate: could not run\n\n```\n{outcome.error}\n```\n"
    verdict = "FAILED" if outcome.failed else "PASSED"
    if outcome.waived and outcome.ratchet_failed:
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
        return "Judged: every file (no base to compare with).\n"
    return (
        f"Judged: the {len(outcome.touched)} files this change touches. Files it does not touch never fail it; "
        "anything found there is listed under *Outside this change* for information.\n"
    )


def _summary(outcome: Outcome) -> str:
    counts = {
        "new": Counter(f.metric for f in outcome.new),
        "worse": Counter(f.metric for f, _ in outcome.worsened),
        "stale": Counter(key[0] for key, _, _ in outcome.stale),
        "head": Counter(key[0] for key in outcome.baseline),
        "base": Counter(key[0] for key in outcome.base_baseline or {}),
    }
    rows = ["| Dimension | Limit | New | Worse | Baseline out of date | Baseline entries | vs base |"]
    rows.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, metric in METRICS.items():
        delta = "n/a" if outcome.base_baseline is None else f"{counts['head'][name] - counts['base'][name]:+d}"
        cells = [counts[c][name] for c in ("new", "worse", "stale", "head")]
        rows.append(
            f"| {metric.label} | {_dimension_limit(name, outcome)} | " + _SEP.join(map(str, cells)) + f" | {delta} |"
        )
    checks = Counter(problem.check for problem in outcome.problems)
    rows += ["", "| Check | Refusals |", "|---|---:|"]
    rows += [f"| {title} | {checks[name]} |" for name, (title, _fix) in CHECKS.items()]
    return "\n".join(rows) + "\n"


def _limit(metric: str, thresholds: dict[str, int]) -> str:
    return f"fails > {thresholds.get(metric, '?')}"


def _dimension_limit(metric: str, outcome: Outcome) -> str:
    """Every level of the dimension: module length also warns, below its failing length."""
    if metric == MODULE_LINES and outcome.module_warning is not None:
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
            f"New violations (not in the baseline){waived}",
            [_UNIT, _DIMENSION, "Value", "Limit"],
            [[_unit(f, link_base), _label(f.metric), f.value, _limit(f.metric, limits)] for f in outcome.new],
            max_rows,
        ),
        _table(
            f"Worse than the baseline{waived}",
            _UNIT_CHANGE,
            [[_unit(f, link_base), _label(f.metric), was, f.value] for f, was in outcome.worsened],
            max_rows,
        ),
        _table(
            f"Baseline out of date (run `{UPDATE_COMMAND}`)",
            ["Baseline entry", _DIMENSION, _BASELINE, _NOW],
            [[_key(key), _label(key[0]), was, _now(key, now, link_base)] for key, was, now in outcome.stale],
            max_rows,
        ),
        _table(
            f"Baseline grew relative to the base branch{waived}",
            ["Baseline entry", _DIMENSION, _BASE, "This change"],
            [[_key(key), _label(key[0]), "absent" if was is None else was, now] for key, was, now in outcome.growth],
            max_rows,
        ),
    ]
    for name, (title, _fix) in CHECKS.items():
        rows = [[_where(p, link_base), p.message] for p in outcome.problems if p.check == name]
        sections.append(_table(title, ["Where", "Problem"], rows, max_rows))
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
    backlog = _table(
        "Already over at the base (backlog, not this change's)",
        [_UNIT, _DIMENSION, _BASE, _NOW],
        [[_unit(f, link_base), _label(f.metric), was, f.value] for f, was in outcome.base_backlog],
        max_rows,
    )
    rows = [
        [_key(key), _label(key[0]), "absent" if was is None else was, _now(key, now, link_base)]
        for key, was, now in outcome.untouched
    ]
    outside = _table("Outside this change (informational)", _UNIT_CHANGE, rows, max_rows)
    if outside:
        outside = f"<details><summary>{len(rows)} findings in files this change does not touch</summary>\n\n{outside}\n</details>\n"
    return [warn, backlog, outside]


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
    pip = " ".join(f"{name}=={pins[name]}" for name in ("ruff", "complexipy", "vulture") if name in pins)
    lines = [
        "### How to fix\n",
        "- **New or worse**: split the unit (extract a function, a class, a module) until it is back under the "
        "limit, or no worse than its recorded value. The baseline cannot absorb a new violation: CI refuses any "
        "baseline entry added or raised relative to the base branch.",
        f"- **Baseline out of date**: something improved. Run `{UPDATE_COMMAND}` and commit "
        f"`{outcome.baseline_path}`; the command only lowers or removes entries.",
    ]
    lines += [
        f"- **{title}**: {fix}."
        for name, (title, fix) in CHECKS.items()
        if any(p.check == name for p in outcome.problems)
    ]
    lines.append(
        f"- **Override**: the `{OVERRIDE_LABEL}` PR label waives new, worse and baseline-growth findings (they stay "
        "listed here). Use it only with the reason in the PR description; it waives nothing else."
    )
    lines.append(
        f"- Tools for a local run: `pip install {pip}` and `npm install -g jscpd@{pins.get('jscpd', '?')}` "
        "(or point `CODE_METRICS_JSCPD` at another jscpd command, such as `npx --yes jscpd@<version>`)."
    )
    return "\n".join(lines) + "\n"
