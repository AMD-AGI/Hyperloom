# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lane candidates a round paid for that the canonical tree moved past before any iteration could measure them.

Such a candidate is kept apart from the candidate archive, whose every entry an iteration measured: merge stacking,
the history digest, export and the knowledge base all read that archive's entries as measured ones. This store is
evidence for later planners, who can take an idea up again from the session that already built it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from hyperloom_common.io import append_jsonl, fsync_directory, fsync_tree
from kernelforge.workflows.forge_loop.campaign.experiments_layout import CANDIDATE_META, ExperimentsLayout

log = logging.getLogger(__name__)

DIFF_FILE, PLAN_FILE, LESSON_FILE = "change.diff", "plan.md", "lesson.md"


@dataclass(frozen=True)
class UnmeasuredLane:
    """One lane candidate dropped unmeasured, and where it stood against the tree."""

    round_iteration: int
    lane_id: str
    # The commit the round's lanes were copied from, which ``diff`` applies to.
    base_commit: str
    # The iteration that tried to apply it, and the canonical commit it no longer applied to.
    dropped_iteration: int
    canonical_commit: str
    plan: str
    summary: str
    diff: str
    lesson: str = ""
    lesson_reason: str = ""


class UnmeasuredLaneArchive:
    """``unmeasured_lanes/``: one directory per dropped candidate, and an index of them."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self._layout = ExperimentsLayout.of(workspace_dir)

    def record(self, lane: UnmeasuredLane) -> Path | None:
        """File ``lane``; None when an earlier process already filed it or the store could not be written."""
        directory = self._layout.unmeasured_lane(lane.round_iteration, lane.lane_id)
        if (directory / CANDIDATE_META).is_file():
            # A process that filed it and stopped before rewriting its lane queue drops it again on resume.
            return None
        root = self._layout.unmeasured_lanes
        staging: Path | None = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{directory.name}.tmp-"))
            meta = {key: value for key, value in asdict(lane).items() if key not in ("plan", "diff", "lesson")}
            files = {DIFF_FILE: lane.diff, PLAN_FILE: lane.plan, CANDIDATE_META: json.dumps(meta, indent=2) + "\n"}
            if lane.lesson:
                files[LESSON_FILE] = lane.lesson
            for name, text in files.items():
                (staging / name).write_text(text, encoding="utf-8")
            fsync_tree(staging)
            os.rename(staging, directory)
            staging = None
            fsync_directory(root)
            append_jsonl(
                self._layout.unmeasured_lane_index,
                {**meta, "dir": directory.name, "has_lesson": bool(lane.lesson)},
                fsync=True,
            )
        except OSError as error:
            log.warning("unmeasured lane %s of round %d not filed: %s", lane.lane_id, lane.round_iteration, error)
            return None
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
        return directory
