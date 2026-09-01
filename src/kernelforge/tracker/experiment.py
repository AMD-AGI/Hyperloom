# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Experiment tracker — persistent JSON-file storage for kernel development experiments."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from kernelforge.tracker.schema import (
    EXPERIMENT_COMPLETED,
    EXPERIMENT_INTERRUPTED,
    EXPERIMENT_RUNNING,
    Experiment,
    Iteration,
)
from kernelforge.durable_io import atomic_write_text


class ExperimentTracker:
    """Manages experiment lifecycle and persistence.

    Each experiment is stored as a single JSON file in the experiments directory.
    Files are named {experiment_id}.json.
    """

    def __init__(self, experiments_dir: str | Path):
        self.dir = Path(experiments_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._on_complete_callbacks: list[Callable[[Experiment], None]] = []

    def on_complete(self, callback: Callable[[Experiment], None]) -> None:
        """Register a callback to run when an experiment completes."""
        self._on_complete_callbacks.append(callback)

    def mark_complete(self, experiment_id: str) -> Experiment:
        """Mark an experiment complete once and fire callbacks once."""
        transitioned = False
        with self._experiment_lock(experiment_id):
            exp = self._load(experiment_id)
            if exp.status not in {EXPERIMENT_COMPLETED, EXPERIMENT_INTERRUPTED}:
                exp.status = EXPERIMENT_COMPLETED
                exp.ended_at = datetime.now().isoformat()
                self._save(exp)
                transitioned = True

        if not transitioned:
            return exp
        for cb in self._on_complete_callbacks:
            with contextlib.suppress(Exception):
                cb(exp)
        return exp

    def mark_interrupted(self, experiment_id: str) -> Experiment:
        """Mark an unfinished segment interrupted without rewriting terminal state."""
        with self._experiment_lock(experiment_id):
            exp = self._load(experiment_id)
            return self._mark_interrupted_locked(exp)

    def _mark_interrupted_locked(self, exp: Experiment) -> Experiment:
        """Transition a locked experiment to interrupted if it is still running."""
        if exp.status in {EXPERIMENT_COMPLETED, EXPERIMENT_INTERRUPTED}:
            return exp
        exp.status = EXPERIMENT_INTERRUPTED
        exp.ended_at = datetime.now().isoformat()
        self._save(exp)
        return exp

    def _path(self, experiment_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id):
            raise ValueError(f"invalid experiment_id path component: {experiment_id!r}")
        return self.dir / f"{experiment_id}.json"

    @contextlib.contextmanager
    def _experiment_lock(self, experiment_id: str):
        lock_path = self.dir / f".experiment.{experiment_id}.lock"
        with open(lock_path, "a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _campaign_lock(self, campaign_id: str):
        lock_id = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()
        lock_path = self.dir / f".campaign.{lock_id}.lock"
        with open(lock_path, "a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _find_segment(self, campaign_id: str, segment_index: int) -> Experiment | None:
        for path in self.dir.glob("*.json"):
            try:
                with open(path) as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("campaign_id") == campaign_id and payload.get("segment_index") == segment_index:
                return Experiment.from_dict(payload)
        return None

    def _save(self, experiment: Experiment) -> None:
        atomic_write_text(
            self._path(experiment.experiment_id),
            json.dumps(experiment.to_dict(), indent=2),
        )

    def _load(self, experiment_id: str) -> Experiment:
        path = self._path(experiment_id)
        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        with open(path) as f:
            return Experiment.from_dict(json.load(f))

    def create(
        self,
        task_id: str = "",
        backend: str = "",
        kernel_backend: str = "",
        description: str = "",
        target_wall_ms: float | None = None,
        baseline_wall_ms: float | None = None,
        campaign_id: str = "",
        segment_index: int = 0,
        parent_experiment_id: str = "",
        experiment_id: str | None = None,
    ) -> Experiment:
        """Create a new experiment and persist it."""
        started_at = datetime.now().isoformat()
        exp = Experiment(
            experiment_id=experiment_id or str(uuid.uuid4())[:8],
            task_id=task_id,
            backend=backend,
            kernel_backend=kernel_backend,
            description=description,
            target_wall_ms=target_wall_ms,
            baseline_wall_ms=baseline_wall_ms,
            campaign_id=campaign_id,
            segment_index=segment_index,
            parent_experiment_id=parent_experiment_id,
            status=EXPERIMENT_RUNNING,
            started_at=started_at,
        )
        self._save(exp)
        return exp

    def create_segment(
        self,
        *,
        campaign_id: str,
        segment_index: int,
        parent_experiment_id: str = "",
        task_id: str = "",
        backend: str = "",
        kernel_backend: str = "",
        description: str = "",
        target_wall_ms: float | None = None,
        baseline_wall_ms: float | None = None,
    ) -> Experiment:
        """Create a linked campaign segment and close an abandoned parent."""
        campaign_id = (campaign_id or "").strip()
        parent_experiment_id = (parent_experiment_id or "").strip()
        if not campaign_id:
            raise ValueError("campaign_id is required")
        if segment_index < 1:
            raise ValueError("segment index must be at least 1")

        with self._campaign_lock(campaign_id):
            existing = self._find_segment(campaign_id, segment_index)
            if existing is not None:
                if existing.parent_experiment_id != parent_experiment_id:
                    raise ValueError(
                        "parent_experiment_id mismatch: existing segment "
                        f"{existing.experiment_id} has parent "
                        f"{existing.parent_experiment_id or 'none'}, requested "
                        f"{parent_experiment_id or 'none'}"
                    )
                return existing

            if parent_experiment_id:
                with self._experiment_lock(parent_experiment_id):
                    parent = self._load(parent_experiment_id)
                    if parent.campaign_id != campaign_id:
                        raise ValueError(f"campaign mismatch: parent belongs to {parent.campaign_id or 'unknown'}")
                    expected_index = parent.segment_index + 1
                    if segment_index != expected_index:
                        raise ValueError(f"segment index must be {expected_index} after parent {parent_experiment_id}")
                    if parent.status == EXPERIMENT_RUNNING:
                        self._mark_interrupted_locked(parent)
            elif segment_index != 1:
                raise ValueError("segment index must be 1 when no parent is provided")

            return self.create(
                task_id=task_id,
                backend=backend,
                kernel_backend=kernel_backend,
                description=description,
                target_wall_ms=target_wall_ms,
                baseline_wall_ms=baseline_wall_ms,
                campaign_id=campaign_id,
                segment_index=segment_index,
                parent_experiment_id=parent_experiment_id,
            )

    def get(self, experiment_id: str) -> Experiment:
        """Load an experiment by ID."""
        return self._load(experiment_id)

    def log_iteration(self, experiment_id: str, **kwargs) -> Iteration:
        """Add a new iteration to an experiment and persist."""
        with self._experiment_lock(experiment_id):
            exp = self._load(experiment_id)
            iteration = exp.add_iteration(**kwargs)
            self._save(exp)
            return iteration

    def set_llm_usage(self, experiment_id: str, usage: dict) -> None:
        """Persist the run's total LLM token spend onto the experiment.

        ``usage`` is the canonical totals dict from
        :class:`~kernelforge.tracker.usage.UsageAccumulator`. No-op on an
        empty/falsy usage so a no-agent run leaves the field unset.
        """
        if not usage:
            return
        with self._experiment_lock(experiment_id):
            exp = self._load(experiment_id)
            exp.llm_usage = dict(usage)
            self._save(exp)

    def set_kb_experience(self, experiment_id: str, kb_experience: dict) -> None:
        """Persist remote experience KB read/write outcome onto the experiment."""
        if not kb_experience:
            return
        with self._experiment_lock(experiment_id):
            exp = self._load(experiment_id)
            exp.kb_experience = dict(kb_experience)
            self._save(exp)

    def set_checkpoint(self, experiment_id: str, checkpoint: dict) -> None:
        """Persist the last validated best commit for external recovery."""
        if not checkpoint:
            return
        exp = self._load(experiment_id)
        exp.checkpoint = dict(checkpoint)
        self._save(exp)

    def set_baseline(self, experiment_id: str, baseline_wall_ms: float) -> None:
        """Persist an auto-measured baseline onto an experiment.

        No-op if the experiment already has a baseline — task-supplied
        baselines take precedence over the measured anchor.
        """
        with self._experiment_lock(experiment_id):
            exp = self._load(experiment_id)
            if exp.baseline_wall_ms is None:
                exp.baseline_wall_ms = baseline_wall_ms
                self._save(exp)

    def list_experiments(self) -> list[Experiment]:
        """List all experiments, newest first."""
        experiments = []
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path) as f:
                    payload = json.load(f)
                if not isinstance(payload, dict) or not payload.get("experiment_id") or not payload.get("created_at"):
                    continue
                experiments.append(Experiment.from_dict(payload))
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                OSError,
                UnicodeDecodeError,
            ):
                continue
        return experiments

    def get_best(self, experiment_id: str) -> Iteration | None:
        """Get the best-performing iteration from an experiment."""
        return self._load(experiment_id).best_iteration()

    def summary(self, experiment_id: str) -> str:
        """Get a formatted summary table for an experiment."""
        return self._load(experiment_id).summary_table()
