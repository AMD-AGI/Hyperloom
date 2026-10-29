# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publishing the campaign's best result: the patch it ships, the history beside it, and the reconciliation a resume owes it."""

from __future__ import annotations

import json
import logging
import subprocess

from kernelforge.workflows.forge_loop.iteration.iteration_types import IterationResult
from kernelforge.workflows.forge_loop.campaign.reporting import KeepBundle, KeepIdentity
from kernelforge.workflows.forge_loop.campaign.run_state import BestRecord
from kernelforge.utils.git import git

log = logging.getLogger(__name__)


def _require_best_is_keep(best: BestRecord, result: IterationResult) -> None:
    """Refuse a run state whose best is not the KEEP it already applied."""
    if (
        best.iteration != result.iteration
        or best.commit_hash != result.commit_hash
        or best.wall_ms != result.wall_ms
        or best.mean_case_speedup != result.mean_case_speedup
    ):
        raise ValueError(f"run state conflicts with KEEP iteration {result.iteration}")


class BestPublicationMixin:
    """Publish one KEEP as the campaign's best, and bring an interrupted publication back in line with the state that recorded it."""

    def _publication_changed_files(self, commit_hash: str) -> list[str]:
        base = self.ic.campaign_base_commit
        if not base:
            return self._commit_changed_files(commit_hash)
        output = self._git(
            "diff",
            "--name-only",
            base,
            commit_hash,
            "--",
            ".",
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _publication_patch(self, commit_hash: str) -> str:
        base = self.ic.campaign_base_commit
        if not base:
            return self._full_diff(commit_hash)
        return git("diff", base, commit_hash, "--", ".", cwd=self.ic.workspace_dir).stdout

    def _publish_best_result(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        pending: dict | None = None,
    ) -> bool:
        """Publish one KEEP before another Agent session may start."""
        if not result.kept or not result.commit_hash:
            return False
        pending = pending or {}
        score = self._keep_score(result, pending, best_before)
        if score is None:
            return False
        benchmark = dict(pending.get("benchmark") or result.bench_detail or {})
        benchmark.setdefault("median_ms", result.wall_ms)
        benchmark.setdefault("mean_case_speedup", result.mean_case_speedup)
        try:
            self.best_publisher.publish(
                KeepIdentity(
                    campaign_id=self.run_state.campaign_id,
                    session_index=int(pending.get("session_index", self.run_state.session_index)),
                    experiment_id=(
                        str(pending.get("experiment_id") or "")
                        or (self.experiment.experiment_id if self.experiment else "")
                    ),
                    iteration=result.iteration,
                    commit_hash=result.commit_hash,
                    plan=plan,
                ),
                score,
                KeepBundle(
                    validation_text=pending.get("validation_text") or result.keep_validation_text(),
                    benchmark=benchmark,
                    changed_files=(
                        list(pending.get("publication_changed_files") or [])
                        or self._publication_changed_files(result.commit_hash)
                    ),
                    patch=(str(pending.get("publication_patch") or "") or self._publication_patch(result.commit_hash)),
                ),
                round_budget=self._round_budget_summary(),
            )
            return True
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            first_failure = not self.persistence_degraded
            self.persistence_degraded = True
            self.persistence_errors.append(f"publish best iteration {result.iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            # The first drop into degraded persistence is the one an operator can still act on; the 12-hour run buried
            # it at debug and the run looked healthy until the final report.
            if first_failure:
                log.warning("failed to publish best result", exc_info=True)
            else:
                log.debug("failed to publish best result", exc_info=True)
            return False

    def _finalize_keep_checkpoint(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        pending: dict,
    ) -> None:
        """Durably finalize the compact state and event for one KEEP commit."""
        self._record_iteration_outcome(
            result,
            plan=plan,
            require_durable=True,
            checkpoint_metadata=pending,
        )

    def _reconcile_best_publication(self) -> None:
        """Repair manifest and derived best views from the durable run state."""
        best = self.run_state.best
        if not best.commit_hash or best.wall_ms is None or best.mean_case_speedup is None:
            return
        # A resumed session recomputes session_index and experiment_id, which legitimately differ from what the stored
        # manifest was written with, so republishing an already-current best tripped the same-iteration conflict guard
        # and reported persistence_degraded over a bundle that was already correct.
        if self.best_publisher.describes_current_best(
            iteration=best.iteration,
            commit_hash=best.commit_hash,
        ):
            return
        metadata = self.archive.load_meta(best.iteration)
        published: dict = {}
        publication_paths = [
            self.best_publisher.manifest_path,
            (self.best_publisher.best_root / f"iter_{best.iteration:03d}" / "publication.json"),
        ]
        for path in publication_paths:
            try:
                candidate = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as error:
                raise ValueError(f"invalid best publication metadata: {path}") from error
            if (
                int(candidate.get("iteration", 0) or 0) == best.iteration
                and candidate.get("commit_hash") == best.commit_hash
            ):
                published = candidate
                break
        validation_text = (
            self.archive.read_candidate_file(
                best.iteration,
                "validation.txt",
            )
            or "canonical validation passed (recovered from run state)"
        )
        benchmark = dict(metadata.get("bench") or {})
        benchmark.setdefault("median_ms", best.wall_ms)
        benchmark.setdefault("mean_case_speedup", best.mean_case_speedup)
        pending = {
            "session_index": published.get("session_index", self.run_state.session_index),
            "experiment_id": published.get("experiment_id", self.run_state.last_experiment_id),
            "baseline_wall_ms": published.get("baseline_wall_ms", self.run_state.baseline_wall_ms),
            "validation_text": validation_text,
            "benchmark": benchmark,
        }
        result = IterationResult(
            iteration=best.iteration,
            duration_sec=0.0,
            validation_passed=True,
            validation_summary=validation_text,
            wall_ms=best.wall_ms,
            mean_case_speedup=best.mean_case_speedup,
            snr_db=published.get("snr_db", metadata.get("snr_db")),
            kept=True,
            commit_hash=best.commit_hash,
            bench_detail=benchmark,
        )
        if not self._publish_best_result(
            result,
            plan=str(published.get("plan") or metadata.get("plan") or best.plan),
            best_before=None,
            pending=pending,
        ):
            log.debug(
                "best publication derived views remain unavailable for iteration %s",
                best.iteration,
            )

    def _publish_optimization_history(self) -> None:
        """Regenerate history from durable events and candidate metadata."""
        events = self.state_store.read_events()
        metadata: dict[int, dict] = {}
        for event in events:
            if event.get("type") != "iteration_result":
                continue
            iteration = int(event.get("iter", 0) or 0)
            candidate = self.archive.load_meta(iteration)
            if not candidate:
                continue
            candidate["archive_path"] = f"candidates/iter_{iteration:03d}/"
            candidate["change_diff"] = self.archive.read_candidate_file(
                iteration,
                "change.diff",
            )
            metadata[iteration] = candidate
        try:
            self.best_publisher.publish_history(
                events=events,
                candidate_metadata=metadata,
            )
        except (OSError, ValueError) as error:
            self.persistence_degraded = True
            self.persistence_errors.append(f"publish optimization history: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug("failed to publish optimization history", exc_info=True)
