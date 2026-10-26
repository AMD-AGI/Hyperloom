# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measured outcome of one iteration's candidate."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IterationResult:
    """Result of a single iteration."""

    iteration: int
    duration_sec: float
    validation_passed: bool
    validation_summary: str
    validation_outcome: str = ""
    wall_ms: float | None = None
    mean_case_speedup: float | None = None
    snr_db: float | None = None
    pmc_diagnosis: str = ""
    vgpr: int | None = None
    kept: bool = False  # True if change was kept, False if reverted
    commit_hash: str = ""
    agent_rationale: str = ""
    # Real error tail from the first failing validation stage (for the ledger); populated on validation failure so
    # gate-off runs still record true errors.
    error_output: str = ""
    # True when the iteration raised an unexpected exception (build/validate/bench crash) rather than merely failing
    # validation.
    crashed: bool = False
    # Full measurement detail for the candidate archive (not just the scalars above).
    bench_detail: dict = field(default_factory=dict)
    pmc_full: str = ""
    # Structured profile metadata (backend, bottleneck, target kernels, roofline dtype/AI/HBM+compute pct, SoL
    # metrics) for the candidate archive's meta.json, so the supervisor / next agent can consume it without parsing
    # prose.
    profile_meta: dict = field(default_factory=dict)
    # Why the agent session ended this iteration (from the in-session gate / SDK): converged / block_budget_exhausted
    # / block_cap / turn_cap / gate_error / agent_stopped / sdk_*. "" when no agent ran.
    session_end_reason: str = ""
    turns: int | None = None
    # Independent of session termination: a candidate that changed protected measurement state is rejected before any
    # canonical driver is executed.
    integrity_violation: bool = False
    # Why the workspace could not be cleared of leftover processes.
    workspace_contention: str = ""
    # Exact model-harness profile that produced this candidate.
    agent_profile_id: str = ""
    agent_provider: str = ""
    agent_model: str = ""
    blockage_epoch: int = 0


__all__ = ["IterationResult"]
