#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Grade a pull request's risk from the paths it changes.

references/tiers.md owns the data: a Tier 1 backbone file is high and a Tier 2 one medium; any other path takes the
highest risk among the risk-table rows whose glob matches it (``*`` crosses ``/``), and medium when none does. A test
is low wherever it sits. Prints risk.txt: ``risk: <level>``, then ``<level> <path> -- <why>`` per path, highest first.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path, PurePosixPath

TIERS = Path(__file__).resolve().parent / "references" / "tiers.md"
_BACKBONE_ROW = re.compile(r"^\| ([12]) \| ([^|]+) \|")
_RISK_ROW = re.compile(r"^\| (low|medium|high) \| ([^|]+) \| ([^|]+) \|")
_CODE = re.compile(r"`([^`]+)`")


class Risk(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Rule:
    risk: Risk
    glob: str
    why: str


@dataclass(frozen=True)
class Graded:
    path: str
    risk: Risk
    why: str


def _rules_of(line: str) -> list[Rule]:
    """A backbone row's paths as globs anchored below ``src/``, or a risk-table row's globs; nothing for other lines."""
    if backbone := _BACKBONE_ROW.match(line):
        risk = Risk.HIGH if backbone[1] == "1" else Risk.MEDIUM
        why = f"Tier {backbone[1]} backbone file"
        return [
            Rule(risk, f"*/{path}*" if path.endswith("/") else f"*/{path}", why) for path in _CODE.findall(backbone[2])
        ]
    if row := _RISK_ROW.match(line):
        return [Rule(Risk[row[1].upper()], glob, row[3].strip()) for glob in _CODE.findall(row[2])]
    return []


def load_rules(tiers: Path) -> list[Rule]:
    return [rule for line in tiers.read_text(encoding="utf-8").splitlines() for rule in _rules_of(line)]


def is_test(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1]
    return "tests" in parts[:-1] or name == "conftest.py" or (name.startswith("test_") and name.endswith(".py"))


def grade(path: str, rules: list[Rule]) -> Graded:
    if is_test(path):
        return Graded(path, Risk.LOW, "test")
    matched = [rule for rule in rules if fnmatch.fnmatchcase(path, rule.glob)]
    if not matched:
        return Graded(path, Risk.MEDIUM, "no row in references/tiers.md matches it")
    top = max(matched, key=lambda rule: rule.risk)
    return Graded(path, top.risk, top.why)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a pull request's risk from the paths it changes.")
    parser.add_argument("files", type=Path, help="changed paths, one per line (fetch.sh's files.txt)")
    parser.add_argument("--tiers", type=Path, default=TIERS, help="the tier and risk tables (default: %(default)s)")
    args = parser.parse_args(argv)
    paths = [line for line in args.files.read_text(encoding="utf-8").splitlines() if line]
    if not paths:
        parser.error(f"{args.files} names no changed path")
    rules = load_rules(args.tiers)
    if not rules:
        parser.error(f"{args.tiers} holds no backbone or risk-table row")
    graded = sorted((grade(path, rules) for path in paths), key=lambda g: (-g.risk, g.path))
    print(f"risk: {graded[0].risk}")
    for g in graded:
        print(f"{g.risk} {g.path} -- {g.why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
