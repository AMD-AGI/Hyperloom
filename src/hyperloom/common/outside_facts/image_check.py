# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The image check: does the environment still say what the rule pack says about the projects outside this repository?

The rule pack states facts about aiter, SGLang, vLLM, PyTorch, Transformers and InferenceX that rules and log readers
depend on. A fact that stops being true (a flag renamed in a newer SGLang, an aiter table that moved, a log line that was
reworded) makes the rule that reads it stop firing, and nothing else says so. Every entry may declare ``image_check``
(see ``image_check_spec``); this module evaluates them against the environment it runs in and reports, per fact, one of
*pass*, *fail* or *unverifiable*. A fact with no check is unverifiable, and says so.

``python -m hyperloom.common.outside_facts.image_check`` runs it and exits 1 when a fact fails. The check looks at the
environment of the process that runs it, which need not be the environment the measurements run in: a hosted sandbox
has its own image. A failure is therefore a reason to look and not a proof.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom_common.io import WriteOptions, atomic_write_json

from hyperloom.common.outside_facts.image_check_eval import Status, evaluate
from hyperloom.common.outside_facts.image_check_spec import CheckSpec
from hyperloom.common.outside_facts.image_env import HostImage, Image, ImageReadError
from hyperloom.common.outside_facts.rule_pack import RulePack, RulePackIdentity, default_pack

__all__ = [
    "CheckResult",
    "FactResult",
    "ImageCheckReport",
    "build_report",
    "evaluate_pack",
    "main",
]

logger = logging.getLogger(__name__)

_WORKERS = 4
_NOT_DECLARED = "no image_check is declared for this fact, so nothing in the environment was compared with it"


@dataclass(frozen=True)
class CheckResult:
    """One declared check and what it found."""

    check: str
    what: str
    status: Status
    detail: str

    def to_json(self) -> dict[str, str]:
        return {"check": self.check, "what": self.what, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True)
class FactResult:
    """One entry of the pack. ``status`` is pass only when every check of the entry passed, fail when one failed, and
    unverifiable otherwise (including an entry that declares no check)."""

    fact: str
    project: str
    status: Status
    checks: tuple[CheckResult, ...]
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "fact": self.fact,
            "project": self.project,
            "status": self.status.value,
            "note": self.note,
            "checks": [check.to_json() for check in self.checks],
        }

    def reasons(self) -> list[str]:
        """The line that says why this fact is not a pass, one per check that was not."""
        found = [f"{check.what}: {check.detail}" for check in self.checks if check.status is not Status.PASS]
        return found or ([self.note] if self.note else [])


@dataclass(frozen=True)
class ImageCheckReport:
    """What the check found: per fact, what each declared check saw."""

    pack: RulePackIdentity
    checked_at: str
    environment: Mapping[str, Any]
    facts: tuple[FactResult, ...]

    def counts(self) -> dict[str, int]:
        held = Counter(fact.status for fact in self.facts)
        return {status.value: held[status] for status in Status}

    def failed(self) -> tuple[FactResult, ...]:
        return tuple(fact for fact in self.facts if fact.status is Status.FAIL)

    def summary(self) -> dict[str, Any]:
        """The few fields worth recording: against which pack, how many facts passed or could not be confirmed, and which
        failed."""
        counts = self.counts()
        return {
            "pack_digest": self.pack["digest"],
            "pass": counts[Status.PASS.value],
            "unverifiable": counts[Status.UNVERIFIABLE.value],
            "fail": [fact.fact for fact in self.failed()],
        }

    def to_json(self) -> dict[str, Any]:
        """The whole report: every fact with every check and what it found."""
        return {
            "pack": dict(self.pack),
            "checked_at": self.checked_at,
            "environment": dict(self.environment),
            "counts": self.counts(),
            "facts": [fact.to_json() for fact in self.facts],
        }

    def markdown_lines(self) -> list[str]:
        """The report for people: the counts, every failing fact with the reason, and the facts nothing could confirm."""
        counts = self.counts()
        lines = [
            f"Image check of rule pack {self.pack['version']}: "
            f"{counts[Status.PASS.value]} pass, {counts[Status.FAIL.value]} fail, "
            f"{counts[Status.UNVERIFIABLE.value]} unverifiable",
        ]
        for fact in self.failed():
            lines.append(f"- FAIL {fact.fact} ({fact.project})")
            lines.extend(f"    {reason}" for reason in fact.reasons())
        unverifiable = [fact.fact for fact in self.facts if fact.status is Status.UNVERIFIABLE]
        if unverifiable:
            lines.append(f"- Unverifiable here: {', '.join(unverifiable)}")
        return lines

    def write(self, path: Path) -> None:
        """Write the whole report to ``path`` as JSON, replacing what is there."""
        atomic_write_json(path, self.to_json(), options=WriteOptions(make_parents=True))


def _roll_up(results: Sequence[CheckResult]) -> Status:
    statuses = {result.status for result in results}
    if Status.FAIL in statuses:
        return Status.FAIL
    return Status.PASS if statuses == {Status.PASS} else Status.UNVERIFIABLE


def _packages(specs: Sequence[CheckSpec]) -> list[str]:
    return sorted({package for package in (spec.package() for spec in specs) if package is not None})


def evaluate_pack(pack: RulePack, image: Image) -> tuple[FactResult, ...]:
    """Every entry of ``pack`` against ``image``; the checks run side by side, because most of the time is a server's
    help text being printed."""
    declared = [(entry_id, spec) for entry_id, entry in pack.entries.items() for spec in entry.image_checks]
    specs = [spec for _, spec in declared]
    try:
        image.prepare(_packages(specs))
    except ImageReadError:
        logger.debug("the interpreter could not report its packages; each check that needs one says so")
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        outcomes = list(pool.map(evaluate, specs, itertools.repeat(image)))
    found: dict[str, list[CheckResult]] = {entry_id: [] for entry_id in pack.entries}
    for (entry_id, spec), outcome in zip(declared, outcomes):
        found[entry_id].append(CheckResult(spec.kind.value, spec.describe(), outcome.status, outcome.detail))
    return tuple(
        FactResult(
            entry_id,
            entry.project,
            _roll_up(found[entry_id]) if found[entry_id] else Status.UNVERIFIABLE,
            tuple(found[entry_id]),
            "" if found[entry_id] else _NOT_DECLARED,
        )
        for entry_id, entry in pack.entries.items()
    )


def build_report(pack: RulePack | None = None, image: Image | None = None) -> ImageCheckReport:
    """Evaluate ``pack`` (the one this process reads) against ``image`` (this machine) and report."""
    held = default_pack() if pack is None else pack
    identity: RulePackIdentity = {"version": held.version, "digest": held.digest}
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    where = HostImage() if image is None else image
    facts = evaluate_pack(held, where)
    return ImageCheckReport(identity, checked_at, where.describe(), facts)


def main(argv: Sequence[str] | None = None, *, pack: RulePack | None = None, image: Image | None = None) -> int:
    """``python -m hyperloom.common.outside_facts.image_check``: print the report; exit 1 when a fact fails."""
    parser = argparse.ArgumentParser(
        prog="python -m hyperloom.common.outside_facts.image_check", description=(__doc__ or "").split("\n\n")[0]
    )
    parser.add_argument("--python", help="the interpreter whose packages count (default: the one running this)")
    parser.add_argument("--json", action="store_true", help="print the whole report as JSON instead of the summary")
    parser.add_argument("--output", type=Path, help="also write the whole report as JSON to this file")
    args = parser.parse_args(argv)
    if image is None and args.python:
        image = HostImage(python=args.python)
    report = build_report(pack, image)
    if args.output is not None:
        report.write(args.output)
    if args.json:
        sys.stdout.write(json.dumps(report.to_json(), indent=2) + "\n")
    else:
        sys.stdout.write("\n".join(report.markdown_lines()) + "\n")
    return 1 if report.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
