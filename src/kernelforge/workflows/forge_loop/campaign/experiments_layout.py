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
CANDIDATE_META, ROUND_CONTEXT = "meta.json", "context.json"
LANE_PLAN_GLOB = "lane_[0-9][0-9][0-9].md"
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


def lane_plan_name(lane: int) -> str:
    """The file a round writes one lane's plan to: ``optimization_plan.md`` for its first lane, ``lane_NNN.md`` after."""
    return "optimization_plan.md" if lane <= 1 else f"lane_{lane:03d}.md"


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
    def orchestration(self) -> Path:
        return self.root / "orchestration"

    @property
    def lane_queue(self) -> Path:
        """Where a fan-out round's unmeasured candidates wait for the iterations that measure them."""
        return self.orchestration / "lane_queue.json"

    def candidate(self, iteration: int) -> Path:
        """The directory an iteration's candidate is archived in, its ``CANDIDATE_META`` among the files."""
        return self.candidates / iteration_tag(iteration)

    def handoff(self, iteration: int) -> Path:
        return self.handoffs / f"{iteration_tag(iteration)}.json"

    def lesson(self, iteration: int) -> Path:
        return self.lessons / f"{iteration_tag(iteration)}.md"

    def round(self, iteration: int) -> Path:
        """The planning directory of the round ``iteration`` opened: its lanes' plans and its ``ROUND_CONTEXT``."""
        return self.orchestration / iteration_tag(iteration)
