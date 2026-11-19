# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KB warm-start: measure prior solutions on this machine and adopt the fastest as the loop's starting point."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from kernelforge.knowledge.kb_store import warmstart_policy
from kernelforge.knowledge.kb_store.identity.run import LoopKernel
from kernelforge.evaluation.scoring import (
    DEFAULT_SNR_THRESHOLD_DB,
    KEEP_MEASUREMENT_COUNT,
    aggregate_regression_detail,
    keep_score,
    passes_keep_threshold,
)
from kernelforge.evaluation.benchmark.runner import (
    CaseCoverageError,
    aggregate_benchmark_measurements,
    calculate_measurement_case_speedups,
)
from kernelforge.evaluation.jit_rebuild import JitRebuildUnavailable
from kernelforge.workflows.forge_loop.memory.knowledge_integration import (
    WarmStartRestoreError,
    _clear_kb_references,
    _editable_workspace_paths,
    _git_apply,
    _git_commit_all,
    _git_discard_worktree,
    _PatchTarget,
    _persist_kb_references,
    _tracked_workspace_clean,
    _untracked_files,
    kb_reference_program_md,
)

log = logging.getLogger(__name__)

# How much of the speedup a candidate was ranked on its own measurement has to reproduce for that ranking to count as
# honest. The regression this answers measured 32% below the claim that had won the ranking.
_WARMSTART_CLAIM_CONFIRMED_RATIO = 0.9


def _bench_once(driver: str, bench_repeat: int = 1) -> dict | None:
    """Run the driver's full benchmark suite once."""
    from kernelforge.evaluation.benchmark.runner import BenchRun, bench_wallclock

    try:
        res = asyncio.run(bench_wallclock(BenchRun(driver, driver_args=[], repeat=bench_repeat)))
        if not isinstance(res, dict) or not res.get("success") or not res.get("case_times"):
            return None
        return res
    except (OSError, ValueError) as e:  # a failed probe just disables warm-start
        print(f"  [kb] bench probe failed: {e}", flush=True)
        return None


def _correctness_once(driver: str, snr_threshold: float) -> bool:
    """Run the driver's complete SNR parity probe once."""
    from kernelforge.evaluation.correctness.driver import test_correctness

    try:
        res = asyncio.run(
            test_correctness(
                driver_script=driver,
                driver_args=[],
                snr_threshold=snr_threshold,
            )
        )
        return bool(res.get("passed")) if isinstance(res, dict) else False
    except (OSError, ValueError) as e:  # a failed probe just rejects warm-start
        print(f"  [kb] correctness probe failed: {e}", flush=True)
        return False


def _apply_candidate_patch(
    sol: dict,
    *,
    workspace_dir,
    allowed_paths,
    pre_untracked,
) -> str:
    """Put one candidate's diff in the working tree, or say why it did not land."""
    patch = sol.get("patch_content") or ""
    if not patch.strip():
        return "empty_patch"
    implementation_identity = (
        sol.get("implementation_identity") if isinstance(sol.get("implementation_identity"), dict) else {}
    )
    canonical_source_paths = {str(path) for path in implementation_identity.get("source_paths", []) if str(path)}
    consumer_source_map = sol.get("consumer_source_map") if isinstance(sol.get("consumer_source_map"), dict) else {}
    target = _PatchTarget(allowed_paths, canonical_source_paths, consumer_source_map)
    if not _git_apply(workspace_dir, patch, check_only=True, target=target):
        return "patch_touches_protected_path_or_not_applicable"
    if not _git_apply(workspace_dir, patch, target=target):
        _git_discard_worktree(
            workspace_dir,
            pre_untracked=pre_untracked,
        )
        return "apply_failed"
    return ""


def _force_jit_rebuild(workspace_dir, kernel, source_files) -> None:
    """Invalidate the artifacts of the sources the applied patch just changed."""
    from kernelforge.evaluation.jit_rebuild import force_jit_rebuild_for_changes

    force_jit_rebuild_for_changes(
        workspace_dir,
        [path for path in [kernel, *(source_files or [])] if path],
    )


def _adopt_measured_candidate(
    sol: dict,
    *,
    kernel,
    workspace_dir,
    source_files,
    allowed_paths,
) -> tuple[str, str]:
    """Re-apply one already-measured candidate and commit it as the start."""
    pre_untracked = _untracked_files(workspace_dir)
    reject_reason = _apply_candidate_patch(
        sol,
        workspace_dir=workspace_dir,
        allowed_paths=allowed_paths,
        pre_untracked=pre_untracked,
    )
    if reject_reason:
        print(
            f"  [kb] warm-start candidate rejected: re-apply failed ({reject_reason})",
            flush=True,
        )
        return "", reject_reason
    try:
        _force_jit_rebuild(workspace_dir, kernel, source_files)
    except (JitRebuildUnavailable, subprocess.TimeoutExpired) as error:
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: rebuild failed ({error})",
            flush=True,
        )
        return "", "rebuild_failed"
    # The driver already rejected an incorrect or slower candidate before this
    # one was chosen, and re-applying the same patch to the same base it was
    # measured on reproduces what it measured. Judging it again here asked the
    # driver the same question, through a task configuration the engine had to
    # assume was written in one particular shape.
    try:
        commit = _git_commit_all(
            workspace_dir,
            f"kb warm-start: apply {sol.get('solution_slug', '')}",
            allowed_paths=allowed_paths,
        )
    except WarmStartRestoreError:
        raise
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        _git_discard_worktree(workspace_dir, pre_untracked=pre_untracked)
        print(
            f"  [kb] warm-start candidate rejected: commit failed ({error})",
            flush=True,
        )
        return "", "commit_failed"
    return commit, ""


def _ranked_speedup(sol: dict) -> float | None:
    """The speedup a candidate was ranked on: its measurement, else its claim."""
    for value in (sol.get("measured_speedup"), sol.get("speedup")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if float(value) > 0.0:
            return float(value)
    return None


def _measurement_confirms_rank(sol: dict, measured_mean_case_speedup: float) -> bool:
    """Whether a measurement backs the speedup this candidate was ranked on."""
    ranked = _ranked_speedup(sol)
    if ranked is None:
        return False
    return measured_mean_case_speedup >= ranked * _WARMSTART_CLAIM_CONFIRMED_RATIO


def _outranks_remaining(
    measured_mean_case_speedup: float,
    remaining: list[dict],
) -> bool:
    """Whether no candidate left in the field is ranked above this measurement."""
    for sol in remaining:
        ranked = _ranked_speedup(sol)
        if ranked is not None and ranked > measured_mean_case_speedup:
            return False
    return True


def _record_measured_speedup(
    config,
    sol: dict,
    measured_mean_case_speedup: float,
    *,
    rank: int,
) -> dict:
    """Write one measured speedup back onto the KB record it was read from."""
    from kernelforge.knowledge.kb_store.errors import (
        kb_store_secrets,
        sanitize_read_error,
    )
    from kernelforge.knowledge.kb_store.recipe import KernelRecipeKB
    from kernelforge.knowledge.kb_store.record_store import KBStoreError, RewriteRecordError

    solution_slug = str(sol.get("solution_slug") or "")
    canonical_id = str(sol.get("kernel_slug") or "")
    session_id = str(sol.get("session_id") or "")
    if not canonical_id or not session_id:
        reason = "missing_record_address"
    else:
        try:
            outcome = KernelRecipeKB.open_canonical_id(
                canonical_id,
                config,
            ).record_measured_speedup(session_id, measured_mean_case_speedup)
        except (KBStoreError, RewriteRecordError, OSError, ValueError) as error:
            outcome = {
                "recorded": False,
                "reason": sanitize_read_error(
                    error,
                    secrets=kb_store_secrets(config),
                ),
            }
        reason = "" if outcome.get("recorded") else str(outcome.get("reason") or "write_failed")
    if reason:
        print(
            f"  [kb] warm-start measured write-back failed for {solution_slug}: {reason}",
            flush=True,
        )
    return {
        "rank": rank,
        "solution_slug": solution_slug,
        "measured_mean_case_speedup": measured_mean_case_speedup,
        "recorded": not reason,
        "reason": reason,
    }


def _rejected_reference_status(reason: str, writeback: dict | None) -> str:
    """The reference index status for a candidate this run did not adopt."""
    if writeback is None:
        return f"rejected:{reason}"
    measured = float(writeback["measured_mean_case_speedup"])
    outcome = "recorded" if writeback["recorded"] else "write-back refused"
    return f"rejected:{reason} (measured {measured:.6f}x {outcome})"


@dataclass(frozen=True)
class _CandidateTrial:
    """What trying one warm-start candidate established about it."""

    adoptable_ms: float | None
    adoptable_mean_case_speedup: float | None
    adoptable_bench: dict | None
    reject_reason: str
    measured_mean_case_speedup: float | None

    @classmethod
    def rejected(
        cls,
        reason: str,
        *,
        measured_mean_case_speedup: float | None,
    ) -> "_CandidateTrial":
        """A candidate that will not be adopted, and what it measured first."""
        return cls(None, None, None, reason, measured_mean_case_speedup)

    @classmethod
    def adoptable(
        cls,
        *,
        applied_ms: float,
        mean_case_speedup: float,
        bench: dict,
    ) -> "_CandidateTrial":
        """A candidate that cleared every measured gate at ``mean_case_speedup``."""
        return cls(applied_ms, mean_case_speedup, bench, "", mean_case_speedup)


@dataclass(frozen=True)
class WarmStartHarness:
    """How a warm-start candidate is judged on this machine: its driver, correctness threshold and bench repeats."""

    driver: str
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    bench_repeat: int = 1


@dataclass(frozen=True)
class _CandidateField:
    """What every candidate trial is judged against: the kernel, the harness, the pristine bench and editable paths."""

    kernel: LoopKernel
    harness: WarmStartHarness
    pristine_bench: dict
    allowed_paths: set[str]


def _try_apply_candidate(sol: dict, trial: _CandidateField, *, pre_untracked) -> _CandidateTrial:
    """Measure one candidate solution as a possible starting point."""
    workspace_dir, harness, pristine_bench = trial.kernel.workspace, trial.harness, trial.pristine_bench
    reject_reason = _apply_candidate_patch(
        sol,
        workspace_dir=workspace_dir,
        allowed_paths=trial.allowed_paths,
        pre_untracked=pre_untracked,
    )
    if reject_reason:
        return _CandidateTrial.rejected(
            reject_reason,
            measured_mean_case_speedup=None,
        )
    try:
        _force_jit_rebuild(workspace_dir, trial.kernel.kernel_path, trial.kernel.source_files)
        passed = _correctness_once(harness.driver, harness.snr_threshold)
        if not passed:
            return _discard_and_reject(
                workspace_dir, "correctness_failed", pre_untracked=pre_untracked, measured_mean_case_speedup=None
            )
        applied_runs = [_bench_once(harness.driver, harness.bench_repeat) for _ in range(KEEP_MEASUREMENT_COUNT)]
        applied_bench = aggregate_benchmark_measurements(applied_runs)
        try:
            measurement_scores = calculate_measurement_case_speedups(
                applied_bench,
                pristine_bench.get("case_times"),
                expected_measurements=KEEP_MEASUREMENT_COUNT,
            )
        except (AttributeError, CaseCoverageError):
            return _discard_and_reject(
                workspace_dir, "case_coverage_failed", pre_untracked=pre_untracked, measured_mean_case_speedup=None
            )
        mean_case_speedup = keep_score(measurement_scores)
        applied_bench["measurement_mean_case_speedups"] = measurement_scores
        applied_bench["mean_case_speedup"] = mean_case_speedup
    except WarmStartRestoreError:
        raise
    except Exception as exc:  # isolation boundary: one KB warm-start candidate; worktree is discarded
        log.warning("warm-start candidate probe failed: %s", exc, exc_info=True)
        return _discard_and_reject(
            workspace_dir, "probe_failed", pre_untracked=pre_untracked, measured_mean_case_speedup=None
        )

    return _judge_measured_candidate(
        trial, applied_bench, measurement_scores, mean_case_speedup, pre_untracked=pre_untracked
    )


def _discard_and_reject(
    workspace_dir, reason: str, *, pre_untracked, measured_mean_case_speedup: float | None, notice: str = ""
) -> _CandidateTrial:
    """Put the worktree back to pristine, print ``notice`` if any, and reject the candidate with what it measured."""
    _git_discard_worktree(
        workspace_dir,
        pre_untracked=pre_untracked,
    )
    if notice:
        print(notice, flush=True)
    return _CandidateTrial.rejected(
        reason,
        measured_mean_case_speedup=measured_mean_case_speedup,
    )


def _judge_measured_candidate(
    trial: _CandidateField, applied_bench: dict, measurement_scores: list[float], mean_case_speedup, *, pre_untracked
) -> _CandidateTrial:
    """Adopt a measured candidate only when it clears the keep, pristine-aggregate and aggregate-regression gates."""
    workspace_dir = trial.kernel.workspace
    applied_ms = applied_bench.get("median_ms") if isinstance(applied_bench, dict) else None
    pristine_ms = trial.pristine_bench.get("median_ms")
    # The suite ran, so this candidate measured something the KB record it came from can be amended with.
    measured_mean_case_speedup = float(mean_case_speedup)

    if (
        not isinstance(applied_ms, (int, float))
        or float(applied_ms) <= 0
        or not passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=1.0,
        )
    ):
        return _discard_and_reject(
            workspace_dir,
            "performance_failed",
            pre_untracked=pre_untracked,
            measured_mean_case_speedup=measured_mean_case_speedup,
        )

    # The gate below is only as closed as the baseline it is given: aggregate_regression_detail reports no
    # contradiction when either wall time is unknown -- correct for a run holding no best yet, wrong as an adoption
    # verdict -- so a pristine aggregate that is absent, non-numeric or not positive would pass a candidate on a
    # silent "" instead of on a comparison.
    if not isinstance(pristine_ms, (int, float)) or float(pristine_ms) <= 0:
        return _discard_and_reject(
            workspace_dir,
            "pristine_aggregate_missing",
            pre_untracked=pre_untracked,
            measured_mean_case_speedup=measured_mean_case_speedup,
            notice=(
                "  [kb] warm-start candidate rejected: the pristine bench reported "
                "no usable aggregate wall time to compare against"
            ),
        )

    # The keep gate above votes on the equal-weight mean of per-case speedups, which can clear the threshold while the
    # candidate is slower in aggregate wall time: a few cheap cases improving outvote one expensive case collapsing,
    # because that mean is unbounded above and bounded at 0 below.
    aggregate_regression = aggregate_regression_detail(
        baseline_ms=pristine_ms,
        best_ms=applied_ms,
        mean_case_speedup=mean_case_speedup,
    )
    if aggregate_regression:
        return _discard_and_reject(
            workspace_dir,
            "aggregate_regression",
            pre_untracked=pre_untracked,
            measured_mean_case_speedup=measured_mean_case_speedup,
            notice=f"  [kb] warm-start candidate rejected: {aggregate_regression}",
        )

    return _CandidateTrial.adoptable(
        applied_ms=float(applied_ms),
        mean_case_speedup=float(mean_case_speedup),
        bench=applied_bench,
    )


def _resume_warmstart_result(workspace_dir) -> dict:
    """The warm-start verdict for a resumed run, which re-reads nothing.

    The field was measured and adopted in the run being resumed; all that is
    owed is the pointer to the references it left behind.
    """
    result = {
        "candidate": False,
        "skipped": "resume",
        "read_reason": "resume",
        "read_error": "",
    }
    pointer = kb_reference_program_md(workspace_dir)
    if pointer:
        result["program_md_addition"] = pointer
        result["reference_program_md_addition"] = pointer
    return result


@dataclass(frozen=True)
class _MeasuredField:
    """What measuring the warm-start field established, in rank order."""

    measurements: list[dict]
    writebacks: list[dict]
    reject_reason: str


def _measure_warmstart_field(
    sols: list[dict],
    statuses: list[str],
    *,
    config,
    trial: _CandidateField,
) -> _MeasuredField:
    """Measure candidates in rank order until one confirms its rank or the budget is spent.

    ``statuses`` is the per-candidate ledger and is filled in place. Every trial
    starts from the pristine tree, so a measured candidate is put back before
    the next one is tried; the winner is re-applied from its own patch later.

    ``top_k()`` caps the field, but a count does not bound wall time: one
    candidate is a compile plus a correctness suite plus a benchmark, minutes on
    the heaviest kernels. On expiry the field closes and the best already
    measured is adopted by the caller.
    """
    measurements: list[dict] = []
    writebacks: list[dict] = []
    reject_reason = ""
    search_deadline = time.monotonic() + warmstart_policy.budget_sec()
    for idx, sol in enumerate(sols):
        if time.monotonic() >= search_deadline:
            for later_index in range(idx, len(statuses)):
                statuses[later_index] = "not_attempted_search_budget"
            print(
                f"  [kb] warm-start search budget spent after {len(measurements)} measured candidate(s); "
                "adopting the best of them",
                flush=True,
            )
            break
        pre_untracked = _untracked_files(trial.kernel.workspace)
        outcome = _try_apply_candidate(sol, trial, pre_untracked=pre_untracked)
        # One write-back per measured candidate, adopted or not.
        writeback = None
        if outcome.measured_mean_case_speedup is not None:
            writeback = _record_measured_speedup(config, sol, outcome.measured_mean_case_speedup, rank=idx + 1)
            writebacks.append(writeback)
        if outcome.reject_reason:
            statuses[idx] = _rejected_reference_status(outcome.reject_reason, writeback)
            reject_reason = outcome.reject_reason
            continue
        _git_discard_worktree(trial.kernel.workspace, pre_untracked=pre_untracked)
        measurements.append(
            {
                "index": idx,
                "ms": float(outcome.adoptable_ms),
                "mean_case_speedup": float(outcome.adoptable_mean_case_speedup),
                "bench": dict(outcome.adoptable_bench or {}),
            }
        )
        ranked = _ranked_speedup(sol)
        ranked_txt = f"{ranked:.6f}x" if ranked is not None else "unrecorded"
        print(
            f"  [kb] warm-start rank {idx + 1} {sol.get('solution_slug')} measured "
            f"{float(outcome.adoptable_mean_case_speedup):.6f}x against a ranked {ranked_txt}",
            flush=True,
        )
        measured_now = float(outcome.adoptable_mean_case_speedup)
        if _measurement_confirms_rank(sol, measured_now) and _outranks_remaining(measured_now, sols[idx + 1 :]):
            for later_index in range(idx + 1, len(statuses)):
                statuses[later_index] = "not_attempted_after_apply"
            break
    return _MeasuredField(measurements, writebacks, reject_reason)


@dataclass
class _WarmStartAttempt:
    """One warm-start search: the candidate field, statuses, and the measured winner if any."""

    config: Any
    kernel: LoopKernel
    harness: WarmStartHarness
    read_status: dict[str, str] = field(
        default_factory=lambda: {"read_reason": "solution_pages_missing", "read_error": ""}
    )
    sols: list[dict] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    pristine_bench: dict = field(default_factory=dict)
    pristine_ms: float | None = None
    keep_baseline_ms: float | None = None
    mean_case_speedup: float | None = None
    applied_bench: dict = field(default_factory=dict)
    reference_reason: str = ""
    applied_commit: str = ""
    applied_idx: int | None = None
    measured_writebacks: list[dict] = field(default_factory=list)

    def load_solutions(self) -> None:
        """Read and filter the ranked field; clear references when the reader raises."""
        from kernelforge.knowledge.kb_store.reader import read_top_solutions

        kernel_source = ""
        with contextlib.suppress(OSError):
            kernel_source = Path(self.kernel.kernel_path).read_text(errors="replace")
        try:
            sols = read_top_solutions(
                config=self.config,
                kernel=replace(self.kernel, kernel_source=kernel_source),
                top_k=warmstart_policy.top_k(),
                read_status=self.read_status,
            )
        except Exception:
            _clear_kb_references(self.kernel.workspace)
            raise
        # A trial's cost scales with how slow the candidate is -- the correctness suite and the benchmark both run the
        # kernel -- so a port claiming to be orders of magnitude off the pace can spend the whole search budget on
        # itself. The claim only has to be right about the magnitude for that to be the wrong trade.
        admissible = [sol for sol in sols if not warmstart_policy.below_floor(_ranked_speedup(sol))]
        if len(admissible) != len(sols):
            print(
                f"  [kb] warm-start ignoring {len(sols) - len(admissible)} candidate(s) "
                f"claiming under {warmstart_policy.min_claimed_speedup():.2f}x",
                flush=True,
            )
        self.sols = admissible

    def measure_and_adopt(self) -> None:
        """Bench the pristine tree, measure the field, and adopt the best surviving candidate."""
        pristine_runs = [
            _bench_once(self.harness.driver, self.harness.bench_repeat) for _ in range(KEEP_MEASUREMENT_COUNT)
        ]
        self.pristine_bench = aggregate_benchmark_measurements(pristine_runs)
        self.pristine_ms = self.pristine_bench.get("median_ms")
        self.keep_baseline_ms = self.pristine_ms
        if self.pristine_ms is None:
            self.reference_reason = "baseline_unavailable"
            self.statuses = [f"rejected:{self.reference_reason}" for _ in self.sols]
            _persist_kb_references(self.kernel.workspace, self.sols, self.statuses)
            print(
                "  [kb] warm-start reference-only: baseline_unavailable; injecting top solutions as reference",
                flush=True,
            )
            return
        self.mean_case_speedup = 1.0
        allowed_paths = _editable_workspace_paths(
            self.kernel.workspace,
            self.kernel.kernel_path,
            self.kernel.source_files,
            self.harness.driver,
        )
        field = _measure_warmstart_field(
            self.sols,
            self.statuses,
            config=self.config,
            trial=_CandidateField(self.kernel, self.harness, self.pristine_bench, allowed_paths),
        )
        self.measured_writebacks = field.writebacks
        self.reference_reason = field.reject_reason
        self._adopt_best_measured(field.measurements, allowed_paths)

    def _adopt_best_measured(self, measurements: list[dict], allowed_paths: set[str]) -> None:
        """Re-apply the fastest measured candidate that still applies cleanly, and settle the others' statuses."""
        # One entry per candidate that reached a measurement, in rank order.
        for measurement in sorted(
            measurements,
            key=lambda item: (-item["mean_case_speedup"], item["index"]),
        ):
            idx = measurement["index"]
            sol = self.sols[idx]
            self.applied_commit, reject_reason = _adopt_measured_candidate(
                sol,
                kernel=self.kernel.kernel_path,
                workspace_dir=self.kernel.workspace,
                source_files=self.kernel.source_files,
                allowed_paths=allowed_paths,
            )
            if reject_reason:
                self.statuses[idx] = f"rejected:{reject_reason}"
                self.reference_reason = reject_reason
                continue
            self.applied_idx = idx
            self.statuses[idx] = "applied"
            self.keep_baseline_ms = measurement["ms"]
            self.mean_case_speedup = measurement["mean_case_speedup"]
            self.applied_bench = dict(measurement["bench"])
            base_txt = f"{self.pristine_ms:.4f} ms" if self.pristine_ms is not None else "unmeasured"
            print(
                f"  [kb] warm-start applied: {sol.get('solution_slug')} "
                f"(rank {idx + 1}, prior speedup {sol.get('speedup')}, "
                f"measured mean case speedup "
                f"{measurement['mean_case_speedup']:.6f}x, "
                f"raw mean {measurement['ms']:.4f} ms vs baseline {base_txt})",
                flush=True,
            )
            break
        if self.applied_idx is not None:
            for measurement in measurements:
                other = measurement["index"]
                if other != self.applied_idx and self.statuses[other] == "not_attempted":
                    self.statuses[other] = f"rejected:outperformed_by_rank_{self.applied_idx + 1}"
        else:
            self.reference_reason = self.reference_reason or "no_candidate_applied"
            print(
                "  [kb] warm-start reference-only: no candidate applied "
                "cleanly + faster; injecting top solutions as reference",
                flush=True,
            )

    def final_result(self) -> dict:
        _persist_kb_references(self.kernel.workspace, self.sols, self.statuses)
        chosen = self.sols[self.applied_idx] if (self.applied_idx is not None) else self.sols[0]
        prompt_pointer = kb_reference_program_md(
            self.kernel.workspace,
            applied_rank=(None if self.applied_idx is None else self.applied_idx + 1),
            solution_slug=str(chosen.get("solution_slug") or "") if (self.applied_idx is not None) else "",
        )
        return {
            "candidate": True,
            **self.read_status,
            "applied": (self.applied_idx is not None),
            "match_mode": str(chosen.get("match_mode") or "reference"),
            "reference_reason": "" if (self.applied_idx is not None) else self.reference_reason,
            "pristine_ms": self.pristine_ms,
            "baseline_case_times": dict(self.pristine_bench.get("case_times") or {}),
            "baseline_unscored_cases": list(self.pristine_bench.get("unscored_cases") or []),
            "keep_baseline_ms": self.keep_baseline_ms,
            "mean_case_speedup": self.mean_case_speedup,
            "case_times": dict(self.applied_bench.get("case_times") or {}),
            "unscored_cases": list(self.applied_bench.get("unscored_cases") or []),
            "applied_commit": self.applied_commit,
            "program_md_addition": prompt_pointer,
            "reference_program_md_addition": kb_reference_program_md(
                self.kernel.workspace,
                detect_applied=False,
            ),
            "solution_slug": str(chosen.get("solution_slug") or ""),
            "speedup": chosen.get("speedup", 0.0),
            "num_references": len(self.sols),
            "applied_rank": (None if self.applied_idx is None else self.applied_idx + 1),
            "measured_writebacks": self.measured_writebacks,
        }


def kb_warmstart(
    *,
    config,
    kernel: LoopKernel,
    harness: WarmStartHarness,
    resume=False,
) -> dict:
    """Look up + apply the best prior solution as the loop's starting point.

    Candidates are measured on this machine and the fastest is adopted, because the speedup a record claims is not
    evidence this consumer can reproduce it. ``warmstart_policy`` bounds that search and is read at call time.
    """
    if resume:
        return _resume_warmstart_result(kernel.workspace)
    try:
        attempt = _WarmStartAttempt(config, kernel, harness)
        attempt.load_solutions()
        if not attempt.sols:
            _clear_kb_references(kernel.workspace)
            return {
                "candidate": False,
                "read_reason": attempt.read_status["read_reason"],
                "read_error": attempt.read_status["read_error"],
            }
        # Keep identity-match metadata populated by the reader (for example a
        # fuzzy donor's requested/selected canonical ids) while normalizing the
        # final read verdict.
        attempt.read_status.update(read_reason="hit", read_error="")
        attempt.statuses = ["not_attempted" for _ in attempt.sols]
        _persist_kb_references(kernel.workspace, attempt.sols, attempt.statuses)
        if not _tracked_workspace_clean(kernel.workspace):
            reason = "workspace_dirty"
            attempt.statuses = [f"rejected:{reason}" for _ in attempt.sols]
            _persist_kb_references(kernel.workspace, attempt.sols, attempt.statuses)
            reference = kb_reference_program_md(kernel.workspace)
            best = attempt.sols[0]
            return {
                "candidate": True,
                **attempt.read_status,
                "applied": False,
                "match_mode": str(best.get("match_mode") or "reference"),
                "reference_reason": reason,
                "pristine_ms": None,
                "keep_baseline_ms": None,
                "applied_commit": "",
                "program_md_addition": reference,
                "reference_program_md_addition": reference,
                "solution_slug": str(best.get("solution_slug") or ""),
                "speedup": best.get("speedup", 0.0),
                "num_references": len(attempt.sols),
                "applied_rank": None,
            }
        attempt.measure_and_adopt()
        return attempt.final_result()
    except WarmStartRestoreError:
        raise
    except Exception as e:  # noqa: BLE001 - isolation boundary: the run's KB warm-start; it cold-starts instead
        from kernelforge.knowledge.kb_store.errors import (
            kb_store_secrets,
            sanitize_read_error,
        )

        error = sanitize_read_error(e, secrets=kb_store_secrets(config))
        # Sanitized reason and stack frames only: the raw message may carry the KB store credentials.
        log.warning(
            "KB warm-start failed: %s (%s)\n%s", error, type(e).__name__, "".join(traceback.format_tb(e.__traceback__))
        )
        print(f"  [kb] warm-start skipped ({error})", flush=True)
        return {
            "candidate": False,
            "read_reason": "warm_start_error",
            "read_error": error,
        }
