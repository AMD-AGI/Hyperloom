# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a forge-loop run keeps its records, under its workspace's ``forge_experiments`` directory.

This module owns the layout end to end: the stores that write a run's records take their paths from it, and
``kernelforge export-runs`` reads an archived run through the same paths. It only names paths; nothing here reads,
writes or creates a file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kernelforge.config.edit_policy import FORGE_LOOP_OUTPUT_DIRNAME

CAMPAIGN_CONFIG, RUN_STATE, EVENTS = "campaign_config.json", "run_state.json", "events.jsonl"
CANDIDATE_META = "meta.json"
#: How the archive stamps its records: naive local time, to the second.
ARCHIVE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_TAG_PREFIX = "iter_"
_TAGGED = re.compile(rf"{re.escape(_TAG_PREFIX)}(\d+)")


def iteration_tag(iteration: int) -> str:
    """The name an iteration's records are filed under: ``iter_007`` for iteration 7."""
    return f"{_TAG_PREFIX}{iteration:03d}"


def tagged_iteration(name: str) -> int | None:
    """The iteration a name ``iteration_tag`` gave stands for, or None for any other name."""
    match = _TAGGED.fullmatch(name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ExperimentsLayout:
    """The paths of one run's records under ``root``, its workspace's ``forge_experiments`` directory."""

    root: Path

    @classmethod
    def of(cls, workspace: str | Path) -> ExperimentsLayout:
        """The layout of the run kept in ``workspace``, taken as given: a caller that wants it resolved resolves it."""
        return cls(Path(workspace) / FORGE_LOOP_OUTPUT_DIRNAME)

    @property
    def campaign_config(self) -> Path:
        return self.root / CAMPAIGN_CONFIG

    @property
    def run_state(self) -> Path:
        return self.root / RUN_STATE

    @property
    def events(self) -> Path:
        return self.root / EVENTS

    @property
    def candidates(self) -> Path:
        return self.root / "candidates"

    @property
    def handoffs(self) -> Path:
        return self.root / "handoffs"

    @property
    def lessons(self) -> Path:
        return self.root / "lessons"

    @property
    def insights(self) -> Path:
        """The campaign's cumulative insight list, replaced whole by each session's record turn."""
        return self.root / "insights.md"

    @property
    def analysis(self) -> Path:
        """The Analysis evidence, one directory per analyzed commit. It is the bulk of a long campaign's records."""
        return self.root / "analysis"

    @property
    def rounds(self) -> Path:
        """What each round's lanes left, one directory per iteration."""
        return self.root / "rounds"

    @property
    def device_usage(self) -> Path:
        """One line per hold of the device a round's lanes share: who waited, and how long it was held."""
        return self.root / "device_usage.jsonl"

    def candidate(self, iteration: int) -> Path:
        """The directory an iteration's candidate is archived in, its ``CANDIDATE_META`` among the files."""
        return self.candidates / iteration_tag(iteration)

    def handoff(self, iteration: int) -> Path:
        return self.handoffs / f"{iteration_tag(iteration)}.json"

    def lesson(self, iteration: int) -> Path:
        return self.lessons / f"{iteration_tag(iteration)}.md"

    def round(self, iteration: int) -> Path:
        """The directory of the round ``iteration`` ran: one ``lane_K`` directory per lane."""
        return self.rounds / iteration_tag(iteration)
