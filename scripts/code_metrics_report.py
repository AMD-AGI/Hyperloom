# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Markdown report of one code-metrics gate run (PR comment and job summary).

The first line is a hidden marker the CI comment step uses to find and update its one
sticky comment instead of adding a new one per push.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from code_metrics_collect import METRICS, MODULE, Finding

MARKER = "<!-- code-metrics-report -->"
UPDATE_COMMAND = "python scripts/code_metrics.py --update-baseline"
_MAX_ROWS = 25
Key = tuple[str, str, str]


@dataclasses.dataclass
class Outcome:
    """Everything one run decided; ``failed`` is the gate verdict."""

    error: str | None = None
    baseline_path: str = ""
    thresholds: dict[str, int] = dataclasses.field(default_factory=dict)
    versions: dict[str, str] = dataclasses.field(default_factory=dict)
    baseline: dict[Key, int] = dataclasses.field(default_factory=dict)
    base_baseline: dict[Key, int] | None = None
    new: list[Finding] = dataclasses.field(default_factory=list)
    worsened: list[tuple[Finding, int]] = dataclasses.field(default_factory=list)
    #: (baseline key, recorded value, the tree's finding: None when gone, another key when moved)
    stale: list[tuple[Key, int, Finding | None]] = dataclasses.field(default_factory=list)
    #: (key, base value or None when added, head value)
    growth: list[tuple[Key, int | None, int]] = dataclasses.field(default_factory=list)
    loosened: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.error or self.new or self.worsened or self.stale or self.growth or self.loosened)


def render(outcome: Outcome, link_base: str = "") -> str:
    if outcome.error:
        return f"{MARKER}\n## Code metrics gate: could not run\n\n```\n{outcome.error}\n```\n"
    verdict = "FAILED" if outcome.failed else "PASSED"
    parts = [f"{MARKER}\n## Code metrics gate: {verdict}\n", _summary(outcome), *_sections(outcome, link_base)]
    if outcome.failed:
        parts.append(_how_to_fix(outcome))
    parts += [f"> {note}\n" for note in outcome.notes]
    tools = ", ".join(f"{name} {version}" for name, version in sorted(outcome.versions.items()))
    parts.append(
        f"<sub>Thresholds and sources: `[tool.hyperloom.code_metrics]` in pyproject.toml. Tools: {tools}.</sub>\n"
    )
    return "\n".join(parts)


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
            f"| {metric.label} | {_limit(name, outcome.thresholds)} | " + " | ".join(map(str, cells)) + f" | {delta} |"
        )
    return "\n".join(rows) + "\n"


def _limit(metric: str, thresholds: dict[str, int]) -> str:
    sign = "<" if METRICS[metric].higher_is_better else ">"
    return f"fails {sign} {thresholds.get(metric, '?')}"


def _sections(outcome: Outcome, link_base: str) -> list[str]:
    limits = outcome.thresholds
    sections = [
        _table(
            "New violations (not in the baseline)",
            ["Unit", "Dimension", "Value", "Limit"],
            [[_unit(f, link_base), _label(f.metric), f.value, _limit(f.metric, limits)] for f in outcome.new],
        ),
        _table(
            "Worse than the baseline",
            ["Unit", "Dimension", "Baseline", "Now"],
            [[_unit(f, link_base), _label(f.metric), was, f.value] for f, was in outcome.worsened],
        ),
        _table(
            f"Baseline out of date (run `{UPDATE_COMMAND}`)",
            ["Baseline entry", "Dimension", "Baseline", "Now"],
            [[_key(key), _label(key[0]), was, _now(key, now, link_base)] for key, was, now in outcome.stale],
        ),
        _table(
            "Baseline grew relative to the base branch",
            ["Baseline entry", "Dimension", "Base", "This change"],
            [[_key(key), _label(key[0]), "absent" if was is None else was, now] for key, was, now in outcome.growth],
        ),
    ]
    if outcome.loosened:
        sections.append("### Gate configuration loosened\n\n" + "".join(f"- {p}\n" for p in outcome.loosened))
    return [s for s in sections if s]


def _table(title: str, header: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return ""
    lines = [f"### {title}: {len(rows)}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows[:_MAX_ROWS]]
    if len(rows) > _MAX_ROWS:
        lines.append(f"\n...and {len(rows) - _MAX_ROWS} more; run the script locally for the full list.")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _label(metric: str) -> str:
    return METRICS[metric].label


def _key(key: Key) -> str:
    _metric, path, unit = key
    return f"`{path}`" if unit == MODULE else f"`{path}` `{unit}`"


def _unit(finding: Finding, link_base: str) -> str:
    where = f"{finding.path}:{finding.line}"
    link = f"[`{where}`]({link_base}/{finding.path}#L{finding.line})" if link_base else f"`{where}`"
    return link if finding.unit == MODULE else f"{link} `{finding.unit}`"


def _now(key: Key, finding: Finding | None, link_base: str) -> str:
    if finding is None:
        return "within the limit, or removed"
    if finding.key != key:
        return f"moved to {_unit(finding, link_base)} ({finding.value})"
    return str(finding.value)


def _how_to_fix(outcome: Outcome) -> str:
    pins = outcome.versions
    pip = " ".join(f"{name}=={pins[name]}" for name in ("ruff", "complexipy", "radon", "vulture") if name in pins)
    return (
        "### How to fix\n\n"
        "- **New or worse**: split the unit (extract a function, a class, a module) until it is back under the "
        "limit, or no worse than its recorded value. The baseline cannot absorb a new violation: CI refuses any "
        "baseline entry added or raised relative to the base branch.\n"
        f"- **Baseline out of date**: something improved. Run `{UPDATE_COMMAND}` and commit "
        f"`{outcome.baseline_path}`; the command only lowers or removes entries.\n"
        f"- Tools for a local run: `pip install {pip}` and `npm install -g jscpd@{pins.get('jscpd', '?')}` "
        "(or point `CODE_METRICS_JSCPD` at another jscpd command, such as `npx --yes jscpd@<version>`).\n"
    )
