# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data schemas for experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

EXPERIMENT_RUNNING = "running"
EXPERIMENT_COMPLETED = "completed"
EXPERIMENT_INTERRUPTED = "interrupted"


@dataclass
class Iteration:
    """A single build-test-bench-profile cycle."""

    iteration_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration that was tested
    config: dict = field(default_factory=dict)

    # Correctness
    snr_db: float | None = None
    allclose: bool | None = None
    max_diff: float | None = None

    # Raw aggregate diagnostic; not the optimization objective and not monotonic,
    # but it withdraws the published improvement badge when it contradicts the
    # score (see BestResultPublisher.publish).
    wall_ms: float | None = None
    # Equal-weight arithmetic mean of per-case speedups.
    mean_case_speedup: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None

    # PMC analysis
    pmc: dict = field(default_factory=dict)
    wait_mfma_ratio: float | None = None
    pmc_diagnosis: str = ""

    # Register info
    vgpr: int | None = None
    agpr: int | None = None
    spill_bytes: int = 0

    # Decision made after this iteration
    decision: str = ""  # "KEEP" / "REVERT" / ""
    notes: str = ""

    def to_dict(self) -> dict:
        # Keep existing semantics: skip falsy/empty fields for compactness, but
        # preserve the new ones explicitly when populated.
        return {k: v for k, v in self.__dict__.items() if v is not None and v != "" and v != {} and v != []}

    @classmethod
    def from_dict(cls, d: dict) -> Iteration:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Experiment:
    """A complete development experiment spanning multiple iterations."""

    experiment_id: str
    task_id: str = ""
    backend: str = ""  # ck, flydsl, triton, aiter
    kernel_backend: str = ""  # which kernel backend prompt drove this
    description: str = ""
    target_wall_ms: float | None = None
    baseline_wall_ms: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    campaign_id: str = ""
    segment_index: int = 0
    parent_experiment_id: str = ""
    status: str = ""
    started_at: str = ""
    ended_at: str = ""
    iterations: list[Iteration] = field(default_factory=list)
    changes_reverted: list[str] = field(default_factory=list)

    # NEW: total LLM token spend for the whole run, summed from terminal
    # provider usage records (see tracker/usage.py). Canonical
    # keys: input_tokens / output_tokens / cache_creation_input_tokens /
    # cache_read_input_tokens / total_cost_usd / cost_available / cost_source /
    # calls. Empty until the loop finishes (or when no agent ran), so an external
    # caller can distinguish unavailable provider pricing from a real zero cost.
    llm_usage: dict = field(default_factory=dict)

    # Remote experience KB observability for forge-loop: selected warm-start
    # solution, apply outcome, write-back reason, and written slugs.
    kb_experience: dict = field(default_factory=dict)
    # Last validated KEEP committed by forge-loop. This is persisted before
    # post-KEEP profiling so an external timeout owner can recover the best
    # source and measurements even when the loop never writes its final result.
    checkpoint: dict = field(default_factory=dict)

    def add_iteration(self, **kwargs) -> Iteration:
        """Add a new iteration with auto-incrementing ID."""
        iter_id = len(self.iterations) + 1
        iteration = Iteration(iteration_id=iter_id, **kwargs)
        self.iterations.append(iteration)
        return iteration

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "iterations"}
        d["iterations"] = [it.to_dict() for it in self.iterations]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Experiment:
        payload = dict(d)
        iterations = [Iteration.from_dict(it) for it in payload.pop("iterations", [])]
        exp = cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})
        exp.iterations = iterations
        return exp
