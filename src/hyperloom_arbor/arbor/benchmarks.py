"""A benchmark's study, as the ``benchmark`` plugin's ``study`` skill writes it under ``benchmarks/<name>/``.

The four files are ``README.md`` (the index), ``rules.yaml`` (the rules a change could break, in the form a gate can
check), ``reference.md`` (what a campaign is judged against) and ``metric.md`` (what to measure). This module reads the
rules and checks the study::

    python3 -m arbor.benchmarks check <dir>     # exit 0 when it passes, 1 with every reason otherwise

Gates that check the rules read them with ``rules``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

FILES = ("README.md", "rules.yaml", "reference.md", "metric.md")
KINDS = ("fixes", "forbids", "requires")
CHECKERS = ("host-gate", "arbor-gate", "validator", "nobody")
REQUIRED = ("id", "kind", "what", "source", "checked_by")
#: Fewer words than this is a file not written yet (a stub or a copied heading).
MIN_WORDS = 40


@dataclass(frozen=True)
class Rule:
    """One rule a change could break: what it fixes, forbids or requires, where it comes from, and who checks it."""

    id: str
    kind: str
    what: str
    source: str
    checked_by: str
    setting: str = ""
    value: str = ""


def rules(directory) -> tuple[list[Rule], list[str]]:
    """The study's rules, and why any could not be read."""

    path = Path(directory) / "rules.yaml"
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return [], [f"{path}: missing"]
    except yaml.YAMLError as exc:
        return [], [f"{path}: not YAML: {str(exc).splitlines()[0]}"]
    entries = document.get("rules") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        return [], [f"{path}: no rules list"]
    found, errors = [], []
    for at, entry in enumerate(entries, 1):
        why = _entry_errors(entry)
        if why:
            errors += [f"{path}: rule {at}: {reason}" for reason in why]
            continue
        found.append(Rule(**{key: str(entry.get(key) or "") for key in Rule.__dataclass_fields__}))
    ids = [rule.id for rule in found]
    errors += [f"{path}: rule id {name!r} is given twice" for name in sorted({i for i in ids if ids.count(i) > 1})]
    return found, errors


def _entry_errors(entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return ["not a mapping"]
    errors = [f"no {key}" for key in REQUIRED if not str(entry.get(key) or "").strip()]
    if entry.get("kind") and entry["kind"] not in KINDS:
        errors.append(f"kind {entry['kind']!r} is not one of {', '.join(KINDS)}")
    if entry.get("checked_by") and entry["checked_by"] not in CHECKERS:
        errors.append(f"checked_by {entry['checked_by']!r} is not one of {', '.join(CHECKERS)}")
    return errors


def check(directory) -> list[str]:
    """Why the study in ``directory`` is not done; ``[]`` when it passes."""

    directory = Path(directory)
    errors = []
    for name in FILES:
        path = directory / name
        if not path.is_file():
            errors.append(f"{path}: missing")
        elif name.endswith(".md") and len(path.read_text(encoding="utf-8").split()) < MIN_WORDS:
            errors.append(f"{path}: fewer than {MIN_WORDS} words")
    if (directory / "rules.yaml").is_file():
        errors += rules(directory)[1]
    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] != "check":
        print("usage: python3 -m arbor.benchmarks check <dir>", file=sys.stderr)
        return 2
    errors = check(args[1])
    for line in errors:
        print(f"error: {line}")
    found, _ = rules(args[1])
    nobody = sum(1 for rule in found if rule.checked_by == "nobody")
    print(f"{args[1]}: {len(found)} rules ({nobody} checked by nobody); {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
