from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

# ── Per-model record ────────────────────────────────────────────────────────────


@dataclass
class SubmissionRecord:
    """Per-model record tracking submission, completion, and CI delivery.

    Accumulates state across the submit → wait → collect pipeline and is
    serialised into the submission manifest.

    Attributes:
        model (str): HuggingFace repo id for this entry.
        status (str): Local submit stage (submitted/dry-run/skipped/failed).
        task_id (str | None): SaFE optimization task id once submitted.
        claw_session_id (str | None): Claw session UUID SaFE created.
        display_name (str | None): Task display name.
        model_path (str | None): Resolved model path reported by SaFE.
        safe_user_id (str | None): SaFE user id owning the task.
        safe_started_at (str | None): SaFE task start timestamp.
        safe_finished_at (str | None): SaFE task finish timestamp.
        detected (dict | None): Auto-detected config as a dict.
        overrides (dict): User/CLI overrides applied.
        pool (dict): Production-pool audit metadata.
        error (str | None): Error message when submit failed/skipped.
        category (str | None): Coarse model shape (moe/dense/"").
        sandbox_duration_seconds (float | None): SaFE wallclock duration.
        final_status (str | None): SaFE terminal status.
        final_phase (int | None): Current phase at the terminal moment.
        final_message (str | None): SaFE task message.
        ci_status (str | None): CI delivery status (separate from SaFE status).
        ci_success (bool): Whether usable artifacts were delivered.
        delivery_reason (str | None): Explanation for the CI delivery status.
        artifacts_dir (str | None): Local directory artifacts landed in.
        artifact_count (int): Number of collected artifacts.
        artifact_files (list[str]): Collected artifact file paths.
        artifact_sources (list[dict]): Provenance entries per artifact.
    """

    model: str
    status: str = "pending"  # local stage: submitted/dry-run/skipped/failed
    task_id: str | None = None
    # Claw session UUID SaFE creates at submit; used to correlate ci_metrics.json
    # under /wekafs/users/<uid>/<session>/ with the task (set in wait_and_collect_one).
    claw_session_id: str | None = None
    display_name: str | None = None
    model_path: str | None = None
    safe_user_id: str | None = None
    safe_started_at: str | None = None
    safe_finished_at: str | None = None
    detected: dict | None = None
    overrides: dict = field(default_factory=dict)
    pool: dict = field(default_factory=dict)
    error: str | None = None
    # Audit fields so each persisted artifact is self-describing.
    category: str | None = None  # moe / dense / "" — from detected.arch
    sandbox_duration_seconds: float | None = None  # SaFE startedAt -> finishedAt
    final_status: str | None = None  # SaFE: Succeeded/Failed/Interrupted/Timeout
    final_phase: int | None = None  # currentPhase at terminal moment
    final_message: str | None = None  # task.Message
    # CI delivery status is separate from SaFE final_status: a SaFE timeout may
    # still have written a useful session_breakdown worth publishing.
    ci_status: str | None = None  # Delivered / Missing artifacts / ...
    ci_success: bool = False
    delivery_reason: str | None = None
    artifacts_dir: str | None = None  # local dir where artifacts landed
    artifact_count: int = 0
    artifact_files: list[str] = field(default_factory=list)
    artifact_sources: list[dict] = field(default_factory=list)
