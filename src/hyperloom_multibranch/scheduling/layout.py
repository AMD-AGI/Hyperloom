# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-branch session directory: the single source of every in-session path.

A session lives at ``<workspace>/<model_basename>/MB-<UTC>-<rand8>/``. Every entry the
scheduler creates carries the ``MB-`` prefix so it never collides with a single-lineage
session in the same workspace. Each run creates a fresh directory and nothing resumes, so
there is no session lock.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.inference_optimizer.session.session_paths import fs_safe_id
from hyperloom_multibranch.scheduling.clock import Clock

SESSION_PREFIX = "MB-"
SESSION_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
SESSION_SUFFIX_BYTES = 4

MANIFEST_FILE = "MB-manifest.json"
SCHEDULING_DIR = "MB-scheduling"
PRELUDE_DIR = "MB-prelude"
NOTEBOOK_DIR = "MB-notebook"
REPO_DIR = "MB-repo"
BLOBS_DIR = "MB-blobs"
BRANCHES_DIR = "MB-branches"
SOURCE_DIR = "MB-source"
TOOLBOX_DIR = "MB-toolbox"
INTEGRATION_DIR = "MB-integration"
FINAL_DIR = "MB-final"
DASHBOARD_DIR = "MB-dashboard"
REPORTS_DIR = "MB-reports"
LOGS_DIR = "MB-logs"
PIDS_FILE = "MB-pids.jsonl"

ROSTER_FILE = "MB-roster.json"
DECISIONS_FILE = "MB-decisions.jsonl"
JUDGE_FILE = "MB-judge.jsonl"
PRELUDE_LAUNCH_FILE = "MB-prelude-launch.json"
PRELUDE_CHILD_FILE = "MB-prelude-child.json"
ROOT_RECIPE_FILE = "MB-root-recipe.json"
PRELUDE_LOG_FILE = "MB-prelude.log"
PRELUDE_PATCHES_DIR = "MB-unexpressed-patches"
MERGES_FILE = "MB-merges.jsonl"
REPORT_FILE = "MB-report.json"
REPORT_MARKDOWN_FILE = "MB-report.md"
SCHEDULER_LOG_FILE = "MB-scheduler.log"

SESSION_SUBDIRS: tuple[str, ...] = (
    SCHEDULING_DIR,
    PRELUDE_DIR,
    NOTEBOOK_DIR,
    BLOBS_DIR,
    BRANCHES_DIR,
    TOOLBOX_DIR,
    INTEGRATION_DIR,
    FINAL_DIR,
    DASHBOARD_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)


@dataclass(frozen=True)
class SessionLayout:
    """Paths inside one session directory; reading a property creates nothing."""

    root: Path

    @property
    def session_id(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_FILE

    @property
    def scheduling(self) -> Path:
        return self.root / SCHEDULING_DIR

    @property
    def prelude(self) -> Path:
        return self.root / PRELUDE_DIR

    @property
    def notebook(self) -> Path:
        return self.root / NOTEBOOK_DIR

    @property
    def repo(self) -> Path:
        """The experiment repository; PRELUDE creates it with the root commit."""
        return self.root / REPO_DIR

    @property
    def blobs(self) -> Path:
        """The patch and kernel files the session's recipes name, by content."""
        return self.root / BLOBS_DIR

    @property
    def branches(self) -> Path:
        return self.root / BRANCHES_DIR

    @property
    def source(self) -> Path:
        """The source base every branch's worktree branches off; the specialist sessions create it."""
        return self.root / SOURCE_DIR

    @property
    def toolbox(self) -> Path:
        return self.root / TOOLBOX_DIR

    @property
    def integration(self) -> Path:
        return self.root / INTEGRATION_DIR

    @property
    def final(self) -> Path:
        return self.root / FINAL_DIR

    @property
    def dashboard(self) -> Path:
        return self.root / DASHBOARD_DIR

    @property
    def reports(self) -> Path:
        """Holds ``MB-report.json`` and the notebook's ``session_breakdown.json``."""
        return self.root / REPORTS_DIR

    @property
    def logs(self) -> Path:
        return self.root / LOGS_DIR

    @property
    def pids(self) -> Path:
        return self.root / PIDS_FILE

    @property
    def roster(self) -> Path:
        return self.scheduling / ROSTER_FILE

    @property
    def decisions(self) -> Path:
        return self.scheduling / DECISIONS_FILE

    @property
    def judge(self) -> Path:
        return self.scheduling / JUDGE_FILE

    @property
    def prelude_launch(self) -> Path:
        """The ``--launch-info-file`` handed to the baseline-prelude child."""
        return self.prelude / PRELUDE_LAUNCH_FILE

    @property
    def prelude_child(self) -> Path:
        return self.prelude / PRELUDE_CHILD_FILE

    @property
    def root_recipe(self) -> Path:
        return self.prelude / ROOT_RECIPE_FILE

    @property
    def prelude_log(self) -> Path:
        return self.prelude / PRELUDE_LOG_FILE

    @property
    def prelude_patches(self) -> Path:
        """The kept patches the root recipe names as unexpressed, copied out of the child session."""
        return self.prelude / PRELUDE_PATCHES_DIR

    @property
    def merges(self) -> Path:
        """One line per merge attempt: the candidate, the integration head, the outcome and overridden values."""
        return self.integration / MERGES_FILE

    @property
    def report(self) -> Path:
        return self.reports / REPORT_FILE

    @property
    def report_markdown(self) -> Path:
        return self.reports / REPORT_MARKDOWN_FILE

    @property
    def scheduler_log(self) -> Path:
        return self.logs / SCHEDULER_LOG_FILE

    def branch_dir(self, branch: str) -> Path:
        return self.branches / fs_safe_id(branch)


def session_dir_name(now: float) -> str:
    stamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime(SESSION_TIMESTAMP_FORMAT)
    return f"{SESSION_PREFIX}{stamp}-{secrets.token_hex(SESSION_SUFFIX_BYTES)}"


def create_session(workspace: Path, model_basename: str, clock: Clock) -> SessionLayout:
    """Create a fresh session directory with every subdirectory; an existing one is an error."""
    root = workspace / fs_safe_id(model_basename) / session_dir_name(clock.now())
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    for name in SESSION_SUBDIRS:
        (root / name).mkdir()
    return SessionLayout(root)


__all__ = [
    "SESSION_PREFIX",
    "SessionLayout",
    "create_session",
    "session_dir_name",
]
