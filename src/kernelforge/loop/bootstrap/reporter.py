# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How one forge-loop run reports its result: the anchors it claims against, the file it writes, the KB it publishes to."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kernelforge.evaluation.scoring import warm_start_improvement_flags
from kernelforge.loop.bootstrap.publication import (
    _initial_remote_publication_state,
    _record_remote_publication_result,
    _remote_publication_view,
    _warm_start_publication_covers,
)
from kernelforge.loop.bootstrap.relay import agent_relay_summary
from kernelforge.loop.knowledge_integration import ExperiencePublication, kb_read_status, write_experience_to_kb
from kernelforge.loop.recovery import load_published_best
from kernelforge.tools.durable_io import atomic_write_json
from kernelforge.tools.git import git_head

if TYPE_CHECKING:
    from kernelforge.loop.bootstrap.run_context import _ForgeRun
    from kernelforge.loop.bootstrap.warm_start import _WarmStart

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SearchAnchors:
    """Where this run's search started, and the untouched kernel it is scored against.

    ``search_start_ms`` is the warm start when there was one, else the first
    baseline; ``pristine_ms`` is what the published speedup divides by, which a
    warm start does not move.
    """

    search_start_ms: float | None
    search_start_mean_case_speedup: float
    pristine_ms: float | None

    def claims(self, *, best_ms, mean_case_speedup) -> dict:
        """What a best measured at ``best_ms`` may report against these anchors, in the order a report carries it."""
        improved_during_search = bool(mean_case_speedup and mean_case_speedup > self.search_start_mean_case_speedup)
        return {
            "baseline_ms": self.pristine_ms,
            "pristine_baseline_ms": self.pristine_ms,
            "search_start_ms": self.search_start_ms,
            "best_ms": best_ms,
            "mean_case_speedup": mean_case_speedup,
            "search_start_mean_case_speedup": self.search_start_mean_case_speedup,
            # KEEP is decided on the mean of per-case speedups while these are aggregate wall times, so the two can
            # legitimately disagree by a hair.
            **warm_start_improvement_flags(
                pristine_ms=self.pristine_ms, best_ms=best_ms, mean_case_speedup=mean_case_speedup
            ),
            "incremental_improved": improved_during_search,
            "improved_during_search": improved_during_search,
        }


def _search_anchors(loop_runner, *, pristine_fallback=None) -> _SearchAnchors:
    """Read the run's search-start and pristine anchors off the live runner."""
    search_start_ms = getattr(loop_runner.ic, "warm_start_wall_ms", None) or getattr(
        loop_runner.ic, "baseline_wall_ms", None
    )
    search_start_mean_case_speedup = (
        getattr(loop_runner.ic, "warm_start_mean_case_speedup", None)
        or getattr(loop_runner, "search_start_mean_case_speedup", None)
        or 1.0
    )
    pristine_ms = (
        getattr(loop_runner.ic, "pristine_baseline_wall_ms", None)
        or getattr(loop_runner.ic, "publication_baseline_wall_ms", None)
        or pristine_fallback
        or search_start_ms
    )
    return _SearchAnchors(search_start_ms, search_start_mean_case_speedup, pristine_ms)


@dataclass
class _Implementer:
    """The backend and model the implementer sessions actually resolved to."""

    backend: str
    model: str


@dataclass
class _RunReporter:
    """How one forge-loop run reports its result: the file it writes, and the KB it publishes to.

    The runner calls three of these back mid-run, so they belong to one owner
    rather than to five closures over the same twenty values.
    """

    run: _ForgeRun
    loop_runner: object
    warm_start: _WarmStart
    assembly_preparation: dict | None
    implementer: _Implementer
    remote_publication: dict = field(init=False)

    def __post_init__(self) -> None:
        self.remote_publication = _initial_remote_publication_state(self.warm_start.warm)

    def build_result(self, kb_experience) -> dict:
        """Assemble the loop result dict from live runner state."""
        loop_runner = self.loop_runner
        anchors = _search_anchors(loop_runner, pristine_fallback=self.warm_start.kb_pristine_baseline_ms)
        best = getattr(loop_runner, "best_wall_ms", None)
        total_speedup = getattr(loop_runner, "best_mean_case_speedup", None)
        incremental_speedup = (
            float(total_speedup) / float(anchors.search_start_mean_case_speedup) if total_speedup else None
        )
        exp_id = getattr(loop_runner.experiment, "experiment_id", None)
        best_iteration = loop_runner.run_state.best.iteration
        best_commit = loop_runner.run_state.best.commit_hash
        # A validated warm-start is published before IterationLoop creates a run-state best.
        if not best_commit:
            published = load_published_best(self.run.iter_config.workspace_dir) or {}
            if int(published.get("iteration", -1)) == 0:
                best = published.get("best_wall_ms")
                total_speedup = published.get("mean_case_speedup")
                anchors = _SearchAnchors(
                    published.get("search_start_ms") or best or anchors.search_start_ms,
                    published.get("search_start_mean_case_speedup")
                    or total_speedup
                    or anchors.search_start_mean_case_speedup,
                    published.get("pristine_baseline_ms") or published.get("baseline_wall_ms") or anchors.pristine_ms,
                )
                incremental_speedup = 1.0
                best_iteration = 0
                best_commit = str(published.get("commit_hash") or "")
        result = {
            **anchors.claims(best_ms=best, mean_case_speedup=total_speedup),
            # Reported so a consumer can tell a faster transfer from a cheaper barrier; wall time alone cannot say
            # which one a kept kernel bought.
            "case_bandwidth": dict(getattr(loop_runner, "last_case_bandwidth", {}) or {}),
            "total_speedup": total_speedup,
            "incremental_speedup": incremental_speedup,
            "experiment_id": exp_id,
            "campaign_id": getattr(loop_runner.run_state, "campaign_id", ""),
            "session_index": getattr(loop_runner.run_state, "session_index", 0),
            "segment_index": getattr(loop_runner.experiment, "segment_index", 0),
            "next_iteration": getattr(loop_runner.run_state, "next_iteration", 1),
            "best_iteration": best_iteration,
            "best_commit": best_commit,
            "remote_publication": _remote_publication_view(self.remote_publication, best_commit),
            "best_manifest": str(self.run.campaign_store.root / "best" / "manifest.json"),
            "optimization_report": str(self.run.campaign_store.root / "optimization_report.md"),
            "optimization_history": str(self.run.campaign_store.root / "optimization_history.md"),
            "persistence_degraded": bool(getattr(loop_runner, "persistence_degraded", False)),
            "persistence_errors": list(getattr(loop_runner, "persistence_errors", [])),
            "iteration_count": 0,
            "kb_experience": kb_experience,
            "agent_backend": self.implementer.backend,
            "agent_model": self.implementer.model,
            "agent_relay": agent_relay_summary(loop_runner.run_state),
            "llm_usage": getattr(loop_runner, "llm_usage", {}) or {},
        }
        if self.assembly_preparation is not None:
            from kernelforge.assembly.prepare import select_result

            select_result(result, self.assembly_preparation)
        if exp_id:
            try:
                completed_experiment = self.run.tracker.get(exp_id)
            except FileNotFoundError:
                # An incomplete run has no tracker record; the result must still be emitted so callers can reject it.
                return result
            result["iteration_count"] = len(completed_experiment.iterations)
            result["checkpoint"] = completed_experiment.checkpoint
        return result

    def write_result_json(self, result: dict) -> None:
        """Persist the result dict to --result-json; a failed write is reported, not raised."""
        if not self.run.request.artifacts.result_json:
            return
        try:
            atomic_write_json(self.run.request.artifacts.result_json, result)
        except (OSError, TypeError, ValueError) as e:
            print(f"  [forge-loop] result-json snapshot skipped ({e})", flush=True)

    def attempt_remote_publication(
        self,
        *,
        commit: str,
        llm_summary: bool,
        incremental_summary: dict | None = None,
        snr_db_override: float | None = None,
    ) -> dict:
        """Publish the current durable best idempotently within this process."""
        if self.assembly_preparation is not None and not self.build_result(None)["improved"]:
            return {"written": False, "reason": "no_assembly_source_win"}
        if not self.run.request.knowledge.experience_kb:
            return {"written": False, "reason": "disabled"}
        if _warm_start_publication_covers(self.remote_publication, commit):
            return dict(self.remote_publication["last_result"])
        if (
            not llm_summary
            and commit
            and self.remote_publication["published_commit"] == commit
            and not self.remote_publication["pending_commit"]
        ):
            return self.remote_publication["last_result"] or {
                "written": True,
                "reason": "already_published",
            }
        self.remote_publication["pending_commit"] = commit
        self.remote_publication["best_commit"] = commit
        self.remote_publication["last_attempted_commit"] = commit
        self.remote_publication["state"] = "publishing"
        self.remote_publication["source"] = "campaign_publication"
        try:
            status = write_experience_to_kb(
                config=self.run.config,
                loop_runner=self.loop_runner,
                kernel=self.run.loop_kernel,
                publication=ExperiencePublication(
                    base_sha=self.run.campaign.base_commit,
                    gpu_target=self.run.config.gpu_target,
                    experience_id=self.run.request.artifacts.experience_id,
                    pristine_baseline_ms=self.warm_start.kb_pristine_baseline_ms,
                    reused_speedup=self.warm_start.kb_reused_speedup,
                    snr_db_override=snr_db_override,
                    llm_summary=llm_summary,
                    incremental_summary=incremental_summary,
                ),
                usage=self.run.usage,
            )
        except Exception as e:  # isolation boundary: one checkpoint's experience publication
            log.warning("experience KB publication failed for %s: %s", commit, e, exc_info=True)
            status = {"written": False, "reason": f"error:{e!r}"}
        _record_remote_publication_result(
            self.remote_publication,
            commit=commit,
            result=status,
        )
        return status

    def publish_final_best(self, *, llm_summary: bool) -> dict:
        """Upgrade the per-run solution page for the final best, and return the run's KB read/write report.

        A publication still pending after the summarised attempt is retried once without the LLM summary.
        """
        final_best_commit = str(getattr(self.loop_runner.run_state.best, "commit_hash", "") or "")
        kb_write = self.attempt_remote_publication(commit=final_best_commit, llm_summary=llm_summary)
        if self.remote_publication["pending_commit"]:
            kb_write = self.attempt_remote_publication(commit=final_best_commit, llm_summary=False)
        return {
            "read": kb_read_status(self.warm_start.warm),
            "write": kb_write,
            "publication": _remote_publication_view(self.remote_publication, final_best_commit),
        }

    def publish_remote_best(self, result) -> None:
        """Publish the KEEP the runner just made durable.

        The runner calls this after the KEEP commit, run state, event and local
        best artifact are durable, but before post-KEEP profiling.
        """
        if not getattr(result, "kept", False):
            return
        commit = str(getattr(result, "commit_hash", "") or git_head(self.run.iter_config.workspace_dir))
        self.attempt_remote_publication(
            commit=commit,
            llm_summary=False,
            incremental_summary={
                "category": "",
                "strategy": str(getattr(self.loop_runner.run_state.best, "plan", "") or ""),
                "recipe": "",
                "lessons": "",
            },
            snr_db_override=getattr(result, "snr_db", None),
        )
        self.write_result_json(self.build_result(kb_experience=None))

    def checkpoint_on_best_committed(self, result) -> None:
        """Persist the durable best (external recovery) before optional post-KEEP profiling."""
        if not getattr(result, "kept", False):
            return
        experiment = self.loop_runner.experiment
        best_commit = str(getattr(result, "commit_hash", "") or git_head(self.run.iter_config.workspace_dir))
        anchors = _search_anchors(self.loop_runner)
        mean_case_speedup = getattr(result, "mean_case_speedup", None)
        checkpoint = {
            "state": "best_committed",
            "decision": "KEEP",
            "experiment_id": (
                self.run.caller_experiment_id or (experiment.experiment_id if experiment is not None else "")
            ),
            "base_commit": self.run.campaign.base_commit,
            "best_commit": best_commit,
            "best_iteration": int(getattr(result, "iteration", 0) or 0),
            **anchors.claims(best_ms=getattr(result, "wall_ms", None), mean_case_speedup=mean_case_speedup),
            "total_speedup": mean_case_speedup,
            "incremental_speedup": (
                float(mean_case_speedup) / float(anchors.search_start_mean_case_speedup) if mean_case_speedup else None
            ),
            "validation_passed": bool(getattr(result, "validation_passed", False)),
            "validation_summary": str(getattr(result, "validation_summary", "") or ""),
            "snr_db": getattr(result, "snr_db", None),
        }
        if experiment is not None or self.run.caller_experiment_id:
            try:
                if experiment is not None:
                    self.run.tracker.set_checkpoint(experiment.experiment_id, checkpoint)
                # Mirror onto the caller-owned record so an external hard kill can still recover this KEEP; refreshed
                # on every resumed segment.
                if self.run.caller_experiment_id:
                    self.run.tracker.set_checkpoint(self.run.caller_experiment_id, checkpoint)
            except (OSError, ValueError, TypeError) as e:
                print(f"  [checkpoint] skipped ({e})", flush=True)
        self.write_result_json(self.build_result(kb_experience=None))
