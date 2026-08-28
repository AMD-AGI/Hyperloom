# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data schemas for experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

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

    def summary_row(self) -> str:
        """One-line summary for experiment log table."""
        snr = f"{self.snr_db:.1f}" if self.snr_db is not None else "?"
        wall = f"{self.wall_ms:.3f}" if self.wall_ms is not None else "?"
        ratio = f"{self.wait_mfma_ratio:.1f}" if self.wait_mfma_ratio is not None else "?"
        vgpr_s = str(self.vgpr) if self.vgpr is not None else "?"
        return f"| {self.iteration_id:4d} | {snr:>8s} | {wall:>9s} | {ratio:>8s} | {vgpr_s:>5s} | {self.decision} |"


@dataclass(frozen=True)
class KernelScoringView:
    """One precomputed scoring/display view over kernel iterations."""

    best: Iteration | None
    speedup: float | None
    speedup_label: str
    authoritative: bool


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

    def best_iteration(self) -> Iteration | None:
        """Return the non-reverted iteration with highest mean case speedup."""
        view = self.scoring_view()
        return view.best if view.authoritative else None

    def scoring_view(self) -> KernelScoringView:
        """Resolve authoritative or legacy display state with one list scan."""
        authoritative = [
            iteration
            for iteration in self.iterations
            if iteration.snr_db is not None
            and iteration.snr_db >= DEFAULT_SNR_THRESHOLD_DB
            and iteration.wall_ms is not None
            and iteration.mean_case_speedup is not None
            and iteration.decision != "REVERT"
        ]
        if authoritative:
            best = max(
                authoritative,
                key=lambda iteration: iteration.mean_case_speedup,
            )
            return KernelScoringView(
                best=best,
                speedup=best.mean_case_speedup,
                speedup_label="mean case speedup",
                authoritative=True,
            )

        legacy = [
            iteration
            for iteration in self.iterations
            if iteration.snr_db is not None
            and iteration.snr_db >= DEFAULT_SNR_THRESHOLD_DB
            and iteration.wall_ms is not None
            and iteration.decision != "REVERT"
        ]
        best = min(legacy, key=lambda iteration: iteration.wall_ms) if legacy else None
        baseline = self.effective_baseline_ms()
        speedup = (
            baseline / best.wall_ms
            if best is not None and baseline is not None and best.wall_ms is not None and best.wall_ms > 0
            else None
        )
        return KernelScoringView(
            best=best,
            speedup=speedup,
            speedup_label="legacy raw ratio",
            authoritative=False,
        )

    def uses_authoritative_scoring(self) -> bool:
        """Whether this experiment has a passing per-case-scored iteration."""
        return self.scoring_view().authoritative

    def legacy_best_iteration(self) -> Iteration | None:
        """Return the lowest raw wall-time iteration for legacy display only."""
        view = self.scoring_view()
        return None if view.authoritative else view.best

    def display_best_iteration(self) -> Iteration | None:
        """Return authoritative best, or a display-only legacy raw best."""
        return self.scoring_view().best

    def display_speedup(self) -> tuple[float | None, str]:
        """Return a display value and an explicit metric label."""
        view = self.scoring_view()
        return view.speedup, view.speedup_label

    def is_plateaued(self, n: int = 3, threshold: float = 0.02) -> bool:
        """Check if last n passing kernel iterations improved less than threshold."""
        authoritative = [
            iteration.mean_case_speedup
            for iteration in self.iterations
            if iteration.snr_db is not None
            and iteration.snr_db >= DEFAULT_SNR_THRESHOLD_DB
            and iteration.mean_case_speedup is not None
        ]
        if len(authoritative) < n:
            return False
        recent = authoritative[-n:]
        return (max(recent) - min(recent)) / min(recent) < threshold

    def is_gate_met(
        self,
        scoring: KernelScoringView | None = None,
    ) -> bool:
        """Check the wall target for the authoritative selected iteration."""
        view = scoring or self.scoring_view()
        if not view.authoritative or view.best is None or self.target_wall_ms is None:
            return False
        return view.best.wall_ms <= self.target_wall_ms

    def effective_baseline_ms(self) -> float | None:
        """Kernel-baseline anchor for speedup reporting (unchanged semantics)."""
        if self.baseline_wall_ms is not None:
            return self.baseline_wall_ms
        for it in self.iterations:
            if it.wall_ms is not None:
                return it.wall_ms
        return None

    def best_mean_case_speedup(self) -> float | None:
        """Mean case speedup of the best kernel iteration."""
        best = self.best_iteration()
        if best is None:
            return None
        return best.mean_case_speedup

    def consecutive_reverts(self) -> int:
        """How many of the most-recent iterations were REVERTs in a row.

        Used by the orchestrator to bail out of a session that's only
        producing reverts (cross-session signal).
        """
        n = 0
        for it in reversed(self.iterations):
            if it.decision == "REVERT":
                n += 1
            elif it.decision == "KEEP":
                break
            # ignore "" (incomplete) rows
        return n

    def summary_table(self) -> str:
        """Markdown table of all iterations."""
        header = "| Iter |   SNR dB |  wall_ms |  variance | vgpr | Decision |"
        sep = "|------|----------|----------|-----------|------|----------|"
        rows = [it.summary_row() for it in self.iterations]
        lines = [header, sep] + rows

        # Summaries
        scoring = self.scoring_view()
        best_k = scoring.best
        if best_k:
            prefix = "Best kernel iter" if scoring.authoritative else "Legacy raw best (display only)"
            lines.append(f"\n{prefix}: {best_k.iteration_id} @ {best_k.wall_ms:.3f} ms")
        if scoring.speedup is not None:
            lines.append(f"{scoring.speedup_label}: {scoring.speedup:.3f}x")
        if scoring.authoritative and self.target_wall_ms:
            gate_met = self.is_gate_met(scoring)
            lines.append(f"Gate ({self.target_wall_ms} ms): {'MET' if gate_met else 'NOT MET'}")
        if self.is_plateaued():
            lines.append("Status: PLATEAUED (last 3 kernel iters <2% improvement)")

        if self.changes_reverted:
            lines.append(f"Reverted: {', '.join(self.changes_reverted)}")
        return "\n".join(lines)

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
