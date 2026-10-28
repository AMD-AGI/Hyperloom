# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-loop`` command: one Forge IterationLoop run as a standalone subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Sequence

import click

from kernelforge.cli.lanes import LaneBinding
from kernelforge.cli.options import (
    AgentRuntimeOptions,
    CeilingSwitch,
    GpuOptions,
    _agent_runtime_options,
    _agent_runtime_overrides,
    _gpu_options,
    _roofline_ceiling_option,
    _validate_max_hours,
    _validate_optional_agent_provider,
    _validate_roofline_target,
)
from kernelforge.option_groups import option_group
from kernelforge.cli.pr_references import (
    PR_KB_BUDGET_SEC,
    _collect_pr_references,
    _git_remote_url,
    _pr_kb_enabled,
    _pr_refs_event_fields,
    _write_pr_provenance,
)
from kernelforge.cli.publication import (
    _initial_remote_publication_state,
    _record_remote_publication_result,
    _remote_publication_view,
    _warm_start_publication_covers,
)
from kernelforge.cli.relay import agent_relay_summary
from kernelforge.cli.session_budget import _forge_session_timeout_sec, _is_long_horizon
from kernelforge.loop.api import format_forge_loop_result
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.config import Config, KnowledgeConfig
from kernelforge.knowledge.kb_store.identity.run import LoopKernel
from kernelforge.knowledge.pr_knowledge.context import PrReferenceQuery
from kernelforge.loop.knowledge_integration import (
    ExperiencePublication,
    WarmStartHarness,
    WarmStartRollbackError,
    kb_reference_program_md,
    kb_read_status,
    kb_warmstart,
    mark_kb_reference_rejected,
    write_experience_to_kb,
)
from kernelforge.loop.recovery import (
    CallerRecovery,
    load_published_best,
    publish_warm_start_recovery,
    rollback_unpublished_warm_start,
)
from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB, warm_start_improvement_flags
from kernelforge.tools.durable_io import atomic_write_json
from kernelforge.tools.git import git_head
from kernelforge.tracker.schema import ExperimentSubject

if TYPE_CHECKING:
    from kernelforge.loop.campaign_config import CampaignConfig, CampaignConfigStore
    from kernelforge.loop.campaign_setup import CampaignResolution
    from kernelforge.loop.iteration.iteration_types import IterationConfig
    from kernelforge.loop.runner import IterationLoop
    from kernelforge.loop.task_preparer import PrepareResult
    from kernelforge.roofline_ceiling.command import AnalystAgent
    from kernelforge.tracker import ExperimentTracker, UsageAccumulator

log = logging.getLogger(__name__)

# A high runaway backstop, not a time bound.
FORGE_IMPLEMENTER_TURN_BACKSTOP = 2000
ANALYSIS_TIMEOUT_SEC = 7200


def _persist_declared_spec(invocation_spec_file: str, driver: str) -> None:
    """Place the declared invocation spec beside the driver that reads it."""
    if not invocation_spec_file:
        return
    from kernelforge.loop.task_preparer import _materialize_invocation_spec

    destination, _ = _materialize_invocation_spec(
        invocation_spec_file,
        Path(driver).resolve().parent,
    )
    if destination is None:
        print(
            f"  [prepare] could not place {invocation_spec_file} beside {driver}; "
            "the driver will only be able to read it from where it was passed"
        )


def _make_ceiling_estimator(
    *,
    workspace_dir: str,
    driver_script: str,
    source_files: Sequence[str],
    analyst: AnalystAgent,
):
    """Build the callable the loop uses to estimate a ceiling.

    Only ``--roofline-ceiling on`` asks for one, because an estimate costs a
    profiler pass and an analyst session. Whether it is called is the loop's
    decision: a fresh campaign estimates, and a resumed one reads back the
    ceiling it already estimated. The analyst runs in its own session under its
    own role, so the backend is resolved here, where the registry already
    lives, and the loop is handed a callable instead of a dependency on it.

    The benchmark the estimate is taken against is the one the campaign scores,
    built by ``bench_command`` rather than assembled again here: a ceiling
    derived against a differently timed region would not be comparable with the
    latencies it is divided by.
    """

    async def estimate(*, case_ids: Sequence[str], case_ms: dict[str, float]):
        from kernelforge.evaluation.benchmark.runner import BenchRun, bench_command
        from kernelforge.roofline_ceiling.analyst import AnalystSession, CeilingTask
        from kernelforge.roofline_ceiling.command import resolve_analyst_backend
        from kernelforge.roofline_ceiling.estimate import estimate_ceiling
        from kernelforge.roofline_ceiling.evidence import KnownCases, PerformanceRun

        return await estimate_ceiling(
            resolve_analyst_backend(analyst),
            CeilingTask(
                run=PerformanceRun(bench_command(BenchRun(driver_script)), workspace_dir),
                kernel_files=[str(path) for path in source_files],
                driver_script=driver_script,
                known=KnownCases(list(case_ids), dict(case_ms)),
            ),
            AnalystSession(model=analyst.model, timeout_sec=analyst.timeout_sec),
        )

    return estimate


def _load_external_baseline(path: str) -> tuple[float, dict[str, float]]:
    """Read the ``--baseline-json`` scoring anchor, rejecting one that cannot anchor a speedup.

    Every number the run publishes divides by these per-case times, so a malformed or non-positive entry is refused
    here rather than silently producing ratios no consumer can interpret.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise click.BadParameter(f"--baseline-json could not be read as JSON: {error}")
    if not isinstance(payload, dict):
        raise click.BadParameter("--baseline-json must hold a JSON object")
    try:
        wall_ms = float(payload.get("wall_ms"))
    except (TypeError, ValueError):
        raise click.BadParameter("--baseline-json needs a numeric 'wall_ms'")
    if not math.isfinite(wall_ms) or wall_ms <= 0.0:
        raise click.BadParameter("--baseline-json 'wall_ms' must be finite and positive")
    raw_cases = payload.get("case_times")
    if not isinstance(raw_cases, dict) or not raw_cases:
        raise click.BadParameter("--baseline-json needs a non-empty 'case_times' object")
    case_times: dict[str, float] = {}
    for case_id, value in raw_cases.items():
        try:
            ms = float(value)
        except (TypeError, ValueError):
            raise click.BadParameter(f"--baseline-json case {case_id!r} has a non-numeric time")
        if not math.isfinite(ms) or ms <= 0.0:
            raise click.BadParameter(f"--baseline-json case {case_id!r} must be finite and positive")
        case_times[str(case_id)] = ms
    return wall_ms, case_times


# ─── Autonomous Forge loop command ───


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


@dataclass(frozen=True)
class _ForgeRun:
    """The request, campaign and collaborators every stage of one forge-loop run works from."""

    request: ForgeLoopRequest
    campaign: CampaignConfig
    campaign_store: CampaignConfigStore
    config: Config
    iter_config: IterationConfig
    tracker: ExperimentTracker
    usage: UsageAccumulator
    deadline: _ForgeDeadline

    @property
    def caller_experiment_id(self) -> str:
        """The caller-owned experiment ID: an external recovery channel."""
        return (self.request.artifacts.experiment_id or "").strip()

    @property
    def loop_kernel(self) -> LoopKernel:
        """The kernel this run optimizes, as the knowledge base identifies it."""
        return LoopKernel(
            kernel_path=self.iter_config.kernel_file,
            kernel_backend=self.campaign.kernel_backend,
            target_functions=self.iter_config.target_functions,
            source_files=self.iter_config.source_files,
            framework=self.campaign.framework,
            operator_name=self.campaign.operator_name,
            workspace=self.iter_config.workspace_dir,
        )


def _print_prepared_driver(prep: PrepareResult) -> None:
    """Tell the operator what a successful preparation wrote, and what its final preflight cost."""
    # Cap the file list: an external driver bundle can legitimately publish dozens of files, and one run
    # scrolled ~700 cache paths through the operator's log for a 3-file change.
    shown = prep.wrote_files[:12]
    extra = len(prep.wrote_files) - len(shown)
    print(
        f"  [prepare] prepared measurement driver in {prep.attempts} attempt(s); "
        f"wrote {', '.join(shown) or '(none)'}" + (f" (+{extra} more)" if extra > 0 else "")
    )
    # Telemetry only: never let a missing field break a good prep.
    preflight_sec = getattr(prep.final_preflight, "duration_sec", 0.0) or 0.0
    if preflight_sec:
        stages = ", ".join(
            f"{name}={detail['seconds']:.0f}s"
            for name, detail in (getattr(prep.final_preflight, "details", {}) or {}).items()
            if isinstance(detail, dict) and "seconds" in detail
        )
        print(f"  [prepare] preflight: {preflight_sec:.0f}s" + (f" ({stages})" if stages else ""))
    if prep.audit_dir:
        print(f"  [prepare] audit: {prep.audit_dir}")


def _exit_failed_preparation(prep: PrepareResult, result_json: str | None) -> NoReturn:
    """End the run with the reason preparation failed: on the console, in --result-json and in the stdout result."""
    # prep.message carries the real failure reason (e.g. "driver conformed but commit didn't land"); the
    # preflight summary can read "ok" even when prep failed downstream, so lead with prep.message and
    # append the preflight summary as extra context.
    detail = prep.message or (prep.final_preflight.summary() if prep.final_preflight else "")
    if prep.message and prep.final_preflight:
        detail = f"{prep.message} | preflight: {prep.final_preflight.summary()}"
    disposition = "rolled back" if prep.rolled_back else "workspace preserved"
    print(f"  [prepare] FAILED after {prep.attempts} attempt(s); {disposition}. {detail}")
    err_result = {
        "error": "task_preparation_failed",
        "detail": detail,
        "attempts": prep.attempts,
        "rolled_back": prep.rolled_back,
        "experiment_id": None,
    }
    if prep.audit_dir:
        err_result["task_preparation_audit_dir"] = prep.audit_dir
    if result_json:
        atomic_write_json(result_json, err_result)
    click.echo(format_forge_loop_result(err_result))
    sys.exit(2)


def _prepare_measurement_driver(run: _ForgeRun, program_md: str) -> None:
    """Make the driver conform to the measurement contract, or leave with the reason it cannot.

    A driver that already conforms is left alone; one that does not is handed
    to the prep agent, and a preparation that fails ends the run rather than
    measuring against a driver nobody vouched for.
    """
    from kernelforge.loop import task_preparer
    from kernelforge.loop.task_preflight import DriverContract

    iter_config = run.iter_config
    invocation_spec_file = run.request.task_context.invocation_spec_file or ""
    prep_deadline = run.deadline.deadline_unix - run.deadline.finalize_reserve_sec
    # Preparation runs the driver, so it needs the rank count before the loop that normally exports it exists.
    if iter_config.nproc_per_node > 1:
        os.environ["FORGE_NPROC_PER_NODE"] = str(iter_config.nproc_per_node)
    else:
        os.environ.pop("FORGE_NPROC_PER_NODE", None)

    run.deadline.require("task preparation", 10.0)
    # The spec's declared suite gates this too: a driver that times a subset of the task's cases is not "already
    # conforming", it just measures less than the task asks for, and accepting it here skips the only step that
    # would have repaired it.
    try:
        expected_case_ids = task_preparer.declared_case_ids(invocation_spec_file)
    except ValueError as exc:
        # Continuing would switch the driver's case check off and spend the whole run measuring a suite the
        # operator never got to state.
        raise click.ClickException(str(exc)) from exc
    pf = task_preparer.preflight_task(
        driver=iter_config.driver_script,
        contract=DriverContract(
            snr_threshold=iter_config.snr_threshold,
            require_graph=True,
            require_profile=True,
            require_ranks=iter_config.nproc_per_node,
            expected_case_ids=expected_case_ids,
        ),
        deadline_unix=prep_deadline,
    )
    if pf.ok:
        print("  [prepare] task already conforms to the driver contract; skipping")
        _persist_declared_spec(invocation_spec_file, iter_config.driver_script)
        return
    print(f"  [prepare] task does not conform ({pf.summary()}); invoking prep agent...")
    # The budget that actually applies is min(wall, what the per-kernel deadline leaves), and it decides how
    # many attempts ever start.
    prep_wall = min(float(task_preparer.PREPARE_MAX_WALL_SEC), max(0.0, prep_deadline - time.time()))
    print(
        f"  [prepare] budget: wall={prep_wall:.0f}s "
        f"attempt_cap={task_preparer.PER_ATTEMPT_CAP_SEC}s "
        f"max_attempts={task_preparer.PREPARE_MAX_ATTEMPTS}"
    )
    prep = task_preparer.prepare_task_sync(
        config=run.config,
        task=task_preparer.PreparationTask(
            workspace_dir=iter_config.workspace_dir,
            kernel=iter_config.kernel_file,
            driver=iter_config.driver_script,
            program_md=program_md,
            target_functions=iter_config.target_functions,
            source_files=iter_config.source_files,
            snr_threshold=iter_config.snr_threshold,
            invocation_spec_file=invocation_spec_file,
            expected_case_ids=expected_case_ids,
            read_only_files=[path for path in (run.request.task_context.program_md_file, invocation_spec_file) if path],
            # A collective task needs a driver that launches its own ranks; the preparer cannot infer that from
            # the kernel source.
            nproc_per_node=iter_config.nproc_per_node,
        ),
        # Let the default PREPARE_MAX_WALL_SEC (3000s, sized for a cold-JIT preflight) apply; prepare_task
        # clamps it to the per-kernel deadline_unix below.
        budget=task_preparer.PreparationBudget(deadline_unix=prep_deadline),
        preflight=pf,
        usage=run.usage,
    )
    if not prep.ok:
        _exit_failed_preparation(prep, run.request.artifacts.result_json)
    _print_prepared_driver(prep)


def _collect_pr_kb_context(run: _ForgeRun) -> tuple[str, str]:
    """Inject PR references into the loop's prompt context, and say which repo they came from.

    Independent of the experience-KB lifecycle, and skipped outright when
    too little of the deadline is left to collect them.
    """
    iter_config = run.iter_config
    campaign = run.campaign
    deadline = run.deadline
    pr_task_context = ""
    pr_kb_repo = ""
    if not _pr_kb_enabled(run.request.knowledge.pr_kb):
        return pr_task_context, pr_kb_repo
    from kernelforge.knowledge.pr_knowledge.context import (
        REASON_LOCAL_FAILURE,
        REASON_SKIPPED_DEADLINE,
    )

    if deadline.remaining(reserve=deadline.finalize_reserve_sec) < 20.0:
        print("  [pr-kb] skipped (deadline)")
        iter_config.pr_kb_event = _pr_refs_event_fields(
            REASON_SKIPPED_DEADLINE,
            {},
        )
    else:
        pr_result = _collect_pr_references(
            iter_config.workspace_dir,
            PrReferenceQuery(
                kernel_backend=campaign.kernel_backend or "",
                git_remote=_git_remote_url(Path(iter_config.workspace_dir).resolve()),
                source_files=campaign.source_files,
                operator_name=campaign.operator_name or "",
                target_functions=iter_config.target_functions,
            ),
            min(PR_KB_BUDGET_SEC, deadline.remaining(reserve=deadline.finalize_reserve_sec)),
        )
        if pr_result is None:
            iter_config.pr_kb_event = _pr_refs_event_fields(
                REASON_LOCAL_FAILURE,
                {"degraded_reason": REASON_LOCAL_FAILURE},
            )
        else:
            pr_task_context = pr_result.prompt_context
            pr_kb_repo = pr_result.repo
            iter_config.pr_reference_context = pr_task_context
            iter_config.pr_reference_labels = tuple(
                f"{reference.repo}#{reference.number}" for reference in pr_result.references
            )
            iter_config.pr_kb_event = _pr_refs_event_fields(pr_result.reason, pr_result.stats)
            iter_config.pr_kb_snapshot = pr_result.pending_snapshot
            if pr_result.injected:
                print(
                    f"  [pr-kb] injected "
                    f"{pr_result.stats.get('injected_entries', 0)} "
                    f"references ({pr_result.stats.get('injected_bytes', 0)} B)"
                )
            else:
                print(f"  [pr-kb] no references ({pr_result.reason or 'empty'})")
    return pr_task_context, pr_kb_repo


@dataclass(frozen=True)
class _WarmStart:
    """What the Experience KB offered this run before iteration 1.

    ``warm`` is the read verdict the result dict reports; the rest is what
    the loop scores against once a candidate has been adopted.
    """

    warm: dict
    program_md: str
    kb_pristine_baseline_ms: float | None
    kb_reused_speedup: float | None


def _publish_warm_start(run: _ForgeRun, warm: dict) -> dict:
    """Publish an adopted warm start as the recoverable best before iteration 1, and return the verdict that stands.

    A warm start that cannot be published durably is rolled back to the pristine source, and the run keeps it as a
    reference only.
    """
    iter_config = run.iter_config
    workspace_dir = iter_config.workspace_dir
    result_json = run.request.artifacts.result_json
    iter_config.publication_baseline_wall_ms = warm.get("pristine_ms")
    try:
        published = publish_warm_start_recovery(
            workspace_dir,
            base_commit=run.campaign.base_commit,
            warm=warm,
            caller=CallerRecovery(run.caller_experiment_id, run.request.artifacts.experience_id, result_json),
            tracker=run.tracker,
        )
        iter_config.warm_start_publication = dict(published)
        if published["persistence_degraded"]:
            print(
                "  [warm-start] recovery publication degraded: " + "; ".join(published["persistence_errors"]),
                flush=True,
            )
        print("  [warm-start] published recoverable best before iteration 1", flush=True)
    except Exception as error:  # noqa: BLE001 - translated into a rollback below
        try:
            rollback_unpublished_warm_start(
                workspace_dir, base_commit=run.campaign.base_commit, result_json=result_json
            )
        except Exception as rollback_error:
            raise click.ClickException(
                f"failed to publish validated warm-start and could not restore pristine workspace: {rollback_error}"
            ) from rollback_error
        mark_kb_reference_rejected(workspace_dir, int(warm.get("applied_rank") or 0), "publication_failed")
        reference = kb_reference_program_md(workspace_dir, detect_applied=False) or warm.get(
            "reference_program_md_addition",
            "## Warm-start reference only\n"
            "The prior patch could not be published durably and was "
            "removed before optimization.",
        )
        iter_config.publication_baseline_wall_ms = None
        print(
            "  [warm-start] recovery publication failed; restored "
            f"pristine source and continuing reference-only ({error})",
            flush=True,
        )
        return {**warm, "applied": False, "keep_baseline_ms": warm.get("pristine_ms"), "program_md_addition": reference}
    return warm


def _read_kb_warm_start(run: _ForgeRun, program_md: str, kb_warmstart_enabled: bool) -> _WarmStart:
    """Adopt the best prior solution the KB offers, so iteration 1 starts from it rather than from scratch.

    Its experience goes into the prompt too, so the agent continues from the best-known state.
    """
    iter_config = run.iter_config
    resume = run.request.resume
    experience_kb = run.request.knowledge.experience_kb
    kb_reused_speedup = None
    warm = {
        "candidate": False,
        "read_reason": "resume" if resume else "deadline",
        "read_error": "",
    }
    if (
        experience_kb
        and kb_warmstart_enabled
        and not resume
        and run.deadline.remaining(reserve=run.deadline.finalize_reserve_sec) >= 600
    ):
        try:
            warm = kb_warmstart(
                config=run.config,
                kernel=run.loop_kernel,
                harness=WarmStartHarness(
                    iter_config.driver_script,
                    snr_threshold=iter_config.snr_threshold,
                    bench_repeat=iter_config.bench_repeat,
                ),
            )
        except WarmStartRollbackError as error:
            failure = click.ClickException(f"warm-start rollback failed; workspace may be inconsistent: {error}")
            failure.exit_code = 2
            raise failure from error
    elif not resume and not experience_kb:
        warm["read_reason"] = "disabled"
        print("  [kb] experience KB disabled by caller")
    elif not resume:
        print("  [kb] warm-start skipped: absolute deadline reserve")
    if warm.get("candidate"):
        if warm.get("applied"):
            # The floor this campaign starts from.
            kb_reused_speedup = warm.get("mean_case_speedup")
            warm = _publish_warm_start(run, warm)
        if warm.get("program_md_addition"):
            program_md = program_md + "\n\n" + warm["program_md_addition"]
        # Keep the pristine raw baseline immutable.
        if warm.get("pristine_ms"):
            iter_config.baseline_wall_ms = warm["pristine_ms"]
        if warm.get("baseline_case_times"):
            iter_config.baseline_case_times = dict(warm["baseline_case_times"])
        iter_config.preloop_baseline_unscored_cases = list(warm.get("baseline_unscored_cases") or [])
        if warm.get("applied"):
            iter_config.pristine_baseline_wall_ms = warm.get("pristine_ms")
            iter_config.warm_start_wall_ms = warm.get("keep_baseline_ms")
            iter_config.warm_start_mean_case_speedup = warm.get("mean_case_speedup")
            iter_config.warm_start_bench = {
                "case_times": dict(warm.get("case_times") or {}),
                "unscored_cases": list(warm.get("unscored_cases") or []),
            }
            iter_config.warm_start_commit = warm.get("applied_commit", "")
            iter_config.warm_start_solution_slug = warm.get("solution_slug", "")
    return _WarmStart(
        warm=warm,
        program_md=program_md,
        kb_pristine_baseline_ms=warm.get("pristine_ms") if warm.get("candidate") else None,
        kb_reused_speedup=kb_reused_speedup,
    )


def _profile_timeout_override(env_timeout, profile_timeout_sec: int) -> int:
    """The Analysis workflow deadline, overridden by the environment when it names a usable one.

    This is the deadline resume tooling sets; ``--profile-timeout-sec`` is the
    default when the environment is unset.
    """
    if env_timeout is None or env_timeout.strip() == "":
        return profile_timeout_sec
    try:
        override = int(env_timeout)
    except ValueError as error:
        raise click.ClickException("FORGE_PROFILE_TIMEOUT_SEC must be an integer") from error
    if override <= 0:
        raise click.ClickException("FORGE_PROFILE_TIMEOUT_SEC must be positive")
    return override


@dataclass(frozen=True)
class _AssemblyPreparation:
    """The verified assembly binding, and what the loop optimizes against instead of the kernel."""

    preparation: dict | None
    source_files: list[str]
    kb_warmstart_enabled: bool
    program_md: str


def _prepare_assembly_task(run: _ForgeRun, program_md: str) -> _AssemblyPreparation:
    """Export and bind the compiler output an assembly task is optimized against.

    A task on any other backend passes through untouched.
    """
    config = run.config
    iter_config = run.iter_config
    campaign = run.campaign
    kernel_backend = campaign.kernel_backend
    kernel = iter_config.kernel_file
    driver = iter_config.driver_script
    source_files_list = iter_config.source_files
    workspace = Path(iter_config.workspace_dir).resolve()
    snr_threshold = iter_config.snr_threshold
    deadline_unix = run.deadline.deadline_unix
    finalize_reserve_sec = run.deadline.finalize_reserve_sec
    resume = run.request.resume
    unchanged = _AssemblyPreparation(None, source_files_list, run.request.knowledge.kb_warmstart_enabled, program_md)
    if kernel_backend != "assembly":
        return unchanged
    from kernelforge.assembly.prepare import AssemblyTask, prepare_assembly, seed_source_baseline

    try:
        assembly_preparation = asyncio.run(
            prepare_assembly(
                config,
                AssemblyTask(kernel=kernel, driver=driver, sources=source_files_list, threshold=snr_threshold),
                base_commit=campaign.base_commit,
                deadline=deadline_unix - finalize_reserve_sec,
                resume=resume,
            )
        )
        if not resume:
            seed_source_baseline(iter_config, assembly_preparation)
    except (ValueError, OSError, subprocess.SubprocessError, asyncio.TimeoutError) as error:
        raise click.ClickException(f"assembly preparation failed; optimization was not started: {error}") from error
    source_files_list = [str(workspace / assembly_preparation["assembly"])]
    iter_config.source_files = source_files_list
    iter_config.commit_new_paths = []
    return _AssemblyPreparation(
        preparation=assembly_preparation,
        source_files=source_files_list,
        # A cached whole-implementation patch could replace the launcher preparation just verified.
        kb_warmstart_enabled=False,
        program_md=program_md
        + (
            "\n\nAssembly preparation is complete. Optimize only "
            + source_files_list[0]
            + ". The Python launcher, original reference, driver, ABI and specialization are frozen."
        ),
    )


@dataclass(frozen=True)
class _ForgeDeadline:
    """The run's absolute deadline, and the reserve finalization needs out of it.

    Every stage asks the same two questions -- how long is left, and is there
    enough left to start this -- so they are answered in one place rather than
    by two closures over the same pair of numbers.
    """

    deadline_unix: float
    finalize_reserve_sec: float

    @classmethod
    def for_budget(cls, budget: TimeBudget) -> _ForgeDeadline:
        """The caller's absolute deadline, else ``--max-hours`` from now, and the env's finalization reserve."""
        deadline_unix = budget.deadline_unix
        if deadline_unix <= 0:
            deadline_unix = time.time() + budget.max_hours * 3600.0
        finalize_reserve_sec = max(30.0, float(os.environ.get("FORGE_FINALIZE_RESERVE_SEC", "120") or 120))
        return cls(deadline_unix, finalize_reserve_sec)

    def remaining(self, *, reserve: float = 0.0) -> float:
        """Seconds left before the deadline, less whatever reserve the caller must leave."""
        return max(0.0, self.deadline_unix - time.time() - reserve)

    def require(self, phase: str, minimum: float = 1.0) -> None:
        """Refuse to start a phase the finalization reserve no longer leaves room for."""
        if self.remaining(reserve=self.finalize_reserve_sec) < minimum:
            raise click.ClickException(f"absolute Forge deadline exhausted before {phase}")


def _reject_incompatible_loop_options(*, producer: str) -> None:
    """Refuse an option combination the loop cannot honour, before anything is set up."""
    if producer:
        from kernelforge.knowledge.kb_store.identity.kernel_recipe import KERNEL_RECIPE_PRODUCERS

        if producer not in KERNEL_RECIPE_PRODUCERS:
            raise click.UsageError(f"--producer must be one of: {', '.join(sorted(KERNEL_RECIPE_PRODUCERS))}")


def _resumed_profile_id(state_path: Path) -> str:
    """The Agent profile a resumed campaign was running, or ``""`` for a fresh one."""
    from kernelforge.loop.run_state import read_run_state

    try:
        return read_run_state(state_path).agent_relay.active_profile_id
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def _check_workspace_matches_resume(*, resume: bool, campaign_store) -> None:
    """Refuse a workspace that does not match what ``--resume`` says about it.

    Resuming needs the records a prior campaign left; starting fresh needs a
    workspace that has none, so a second campaign cannot be read as a
    continuation of the first.
    """
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"
    has_run_artifacts = (
        state_path.exists()
        or (campaign_root / "events.jsonl").exists()
        or any((campaign_root / "candidates").glob("iter_*"))
    )
    if resume and not state_path.is_file():
        raise click.ClickException("--resume requires <workspace>/forge_experiments/run_state.json")
    if resume and not campaign_store.exists():
        raise click.ClickException("--resume requires <workspace>/forge_experiments/campaign_config.json")
    if not resume and has_run_artifacts:
        raise click.ClickException("workspace already contains a Forge campaign; pass --resume to continue it")


def _record_experiment_accounting(tracker, *, experiment_id, llm_usage: dict, kb_experience) -> None:
    """Record what this run spent and what the KB gave it, against the experiment record.

    Accounting is best-effort on both counts: neither may fail a run that has
    already produced its result.
    """
    if not experiment_id:
        return
    try:
        if llm_usage.get("calls"):
            tracker.set_llm_usage(experiment_id, llm_usage)
    except (OSError, ValueError, TypeError) as exc:
        click.echo(
            f"Warning: failed to record LLM usage for experiment {experiment_id}: {exc}",
            err=True,
        )
    try:
        tracker.set_kb_experience(experiment_id, kb_experience)
    except (OSError, ValueError, TypeError) as exc:
        click.echo(
            f"Warning: failed to record KB experience for experiment {experiment_id}: {exc}",
            err=True,
        )


def _ensure_recovery_experiment(tracker, caller_experiment_id: str, *, kernel: str) -> None:
    """Make sure the caller-owned recovery record exists, so a hard kill still has one to recover from."""
    if not caller_experiment_id:
        return
    try:
        tracker.get(caller_experiment_id)
    except FileNotFoundError:
        tracker.create(
            ExperimentSubject(Path(kernel).stem, description=f"External recovery channel for {caller_experiment_id}"),
            experiment_id=caller_experiment_id,
        )


def _program_context(
    resolution,
    campaign_store,
    *,
    resume: bool,
    experience_kb: bool,
    workspace_dir: str,
) -> str:
    """The program context the implementer is given, plus any KB reference a resume should carry."""
    if resolution.save_deferred:
        # The immutable config (and its program.md) is not on disk yet; use the in-memory program text captured from
        # --program-md-file (matching read_program_md's "" for a campaign without program context).
        program_md = resolution.program_text or ""
    else:
        try:
            program_md = campaign_store.read_program_md(resolution.campaign)
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
    if resume and experience_kb:
        reference_pointer = kb_reference_program_md(workspace_dir)
        if reference_pointer:
            program_md = program_md + "\n\n" + reference_pointer
    return program_md


def _resolve_campaign(request: ForgeLoopRequest, workspace: Path, agent_relay) -> CampaignResolution:
    """The stored campaign on resume, else one created from this invocation's options."""
    from kernelforge.loop.campaign_config import CampaignInputs
    from kernelforge.loop.campaign_setup import resolve_campaign

    try:
        return resolve_campaign(
            str(workspace),
            CampaignInputs(
                kernel=request.campaign.kernel,
                driver=request.campaign.driver,
                snr_threshold=request.campaign.snr_threshold,
                source_files=request.identity.source_files,
                program_md_file=request.task_context.program_md_file,
                target_functions=request.identity.target_functions,
                operator_name=request.identity.operator_name,
                producer=request.knowledge.producer,
                kernel_backend=request.task_context.kernel_backend or "",
                git_branch=request.git_branch or "",
                gpu_target=request.gpu.gpu_target or "",
                gpu_type=request.gpu.gpu_type,
                task_type=request.identity.task_type,
                framework=request.identity.framework,
                nproc_per_node=request.loop_shape.nproc_per_node,
                bench_repeat=request.loop_shape.bench_repeat,
                commit_new_paths=tuple(request.loop_shape.commit_new_paths),
                agent_relay=agent_relay,
            ),
            resume=request.resume,
            prepare_task=request.prepare_task,
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error


def _session_config(
    probe: SpecialistProbeOptions,
    campaign: CampaignConfig,
    workspace: Path,
    knowledge_config: KnowledgeConfig,
    agent_runtime_overrides: dict,
) -> Config:
    """The Config this session runs under: the campaign's GPU and producer, plus only the overrides asked for."""
    overrides = {"gpu_target": campaign.gpu_target, "gpu_type": campaign.gpu_type, "producer": campaign.producer}
    # Only what was actually asked for: an override present with a None value still wins over ``Config.from_env``'s
    # environment lookup, which is what made the FORGE_SPECIALIST_PROBE* variables dead on this path.
    for _name, _value in (
        ("specialist_probe", probe.specialist_probe),
        ("specialist_probe_max", probe.specialist_probe_max),
        ("specialist_probe_budget_sec", probe.specialist_probe_budget_sec),
        ("specialist_probe_scratch_root", probe.specialist_probe_scratch_root),
    ):
        if _value is not None:
            overrides[_name] = _value
    overrides.update(agent_runtime_overrides)
    if campaign.gpu_target:
        os.environ["GPU_TARGET"] = campaign.gpu_target
    return Config.from_env(workspace=str(workspace), knowledge_config=knowledge_config, **overrides)


def _isolate_aiter_cache(experiments_dir: Path, max_cache_gb: float, source_files: list[str]) -> None:
    """Give this attempt its own AITER runtime-build cache, so parallel forge processes never evict its kernels.

    The pristine BASELINE shard is seeded with the package's prebuilt .so so the task-preparation preflight imports
    warm modules instead of cold-compiling the CK instance-factory TU (>26 min on gfx950, past the preflight timeout).
    """
    from kernelforge.tools.aiter_cache import (
        activate_aiter_cache_for_sources,
        configure_aiter_cache_isolation,
        seed_prebuilt_modules,
    )

    aiter_cache = configure_aiter_cache_isolation(experiments_dir, max_cache_bytes=int(max_cache_gb * 1024**3))
    print(f"  [aiter-cache] isolated runtime builds under {aiter_cache.cache_root}")
    baseline_cache = activate_aiter_cache_for_sources(source_files)
    if baseline_cache is not None:
        seed_stats = seed_prebuilt_modules(baseline_cache.aiter_jit_dir)
        print(
            f"  [aiter-cache] seeded {seed_stats['seeded']} prebuilt module(s)"
            f" (skipped {seed_stats['skipped']}, errors {seed_stats['errors']})"
            f" from {seed_stats['src'] or 'n/a'}"
        )


def _persist_prepared_campaign(run: _ForgeRun, workspace: Path, program_text: str | None) -> _ForgeRun:
    """Fix a fresh campaign's driver digest and pristine base commit from the post-preparation tree, and save it."""
    import dataclasses
    import hashlib

    from kernelforge.loop.campaign_config import derive_campaign_implementation_contract

    campaign = run.campaign
    iter_config = run.iter_config
    prepared_driver_sha256 = hashlib.sha256(Path(iter_config.driver_script).read_bytes()).hexdigest()
    prepared_base_commit = git_head(str(workspace)) or campaign.base_commit
    prepared_signature, prepared_identity = derive_campaign_implementation_contract(
        workspace_dir=str(workspace),
        kernel_path=campaign.kernel_path,
        source_files=campaign.source_files,
        framework=campaign.framework,
        base_commit=prepared_base_commit,
    )
    campaign = dataclasses.replace(
        campaign,
        driver_sha256=prepared_driver_sha256,
        base_commit=prepared_base_commit,
        implementation_signature=prepared_signature,
        implementation_identity=prepared_identity,
    )
    iter_config.canonical_driver_sha256 = campaign.driver_sha256
    iter_config.campaign_base_commit = campaign.base_commit
    iter_config.implementation_signature = campaign.implementation_signature
    iter_config.implementation_identity = dict(campaign.implementation_identity)
    try:
        run.campaign_store.save(campaign, program_md=program_text)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    return dataclasses.replace(run, campaign=campaign)


def _build_loop_runner(run: _ForgeRun, source_files: list[str], session_timeout_sec: int) -> IterationLoop:
    """Construct the IterationLoop, refusing a resume whose stored state it cannot continue."""
    from kernelforge.loop.runner import IterationLoop
    from kernelforge.roofline_ceiling.command import AnalystAgent

    request = run.request
    loop_runner = IterationLoop(
        run.iter_config,
        run.tracker,
        run.config,
        resume=request.resume,
        ceiling_estimator=(
            _make_ceiling_estimator(
                workspace_dir=request.campaign.workspace_dir,
                driver_script=run.iter_config.driver_script,
                source_files=source_files,
                # The runtime the implementer lanes settle on is resolved further down, after this point, and the
                # analyst is a different role anyway: it reads the same provider/model ladder the standalone command
                # reads, so the raw selection is what it needs and an empty one means "take the default".
                analyst=AnalystAgent(
                    provider=request.agent.agent_backend or "",
                    model=request.agent.model or "",
                    timeout_sec=session_timeout_sec,
                    sandbox_mode=run.config.agent_sandbox_mode,
                ),
            )
            if request.roofline_ceiling.enabled
            else None
        ),
    )
    if request.resume:
        try:
            loop_runner.validate_resume_preflight()
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
    return loop_runner


@dataclass(frozen=True)
class CampaignTarget:
    """The kernel, driver and workspace a campaign optimizes, and the correctness threshold."""

    kernel: str | None
    driver: str | None
    workspace_dir: str
    snr_threshold: float


@dataclass(frozen=True)
class TimeBudget:
    """The campaign's runtime budget and its absolute deadline."""

    max_hours: float
    session_timeout_sec: int | None
    deadline_unix: float


@dataclass(frozen=True)
class TaskContext:
    """The kernel backend and the task context files handed to preparation."""

    kernel_backend: str | None
    program_md_file: str | None
    invocation_spec_file: str | None


@dataclass(frozen=True)
class RunArtifacts:
    """Where the run writes, how caches are bounded, and how the run is identified."""

    experiments_dir: str | None
    aiter_cache_max_gb: float
    experiment_id: str | None
    experience_id: str
    result_json: str | None


@dataclass(frozen=True)
class AgentRoleOptions:
    """Provider-facing knobs of the implementer, Analysis and supervisor roles."""

    permission_mode: str | None
    profile_timeout_sec: int
    supervisor_backend: str
    profiling: bool


@dataclass(frozen=True)
class LoopShape:
    """How the loop measures and how many candidates each round explores."""

    nproc_per_node: int
    lanes: int
    merge_stacking: bool
    bench_repeat: int
    commit_new_paths: tuple[str, ...]


@dataclass(frozen=True)
class TaskIdentity:
    """The task type and the names that identify the optimized operator."""

    task_type: str
    source_files: str
    target_functions: str
    framework: str
    operator_name: str


@dataclass(frozen=True)
class KnowledgeOptions:
    """The experience-KB and PR-KB switches, and the producer the records belong to."""

    experience_kb: bool
    kb_warmstart_enabled: bool
    producer: str
    pr_kb: bool | None


@dataclass(frozen=True)
class SpecialistProbeOptions:
    """Overrides for the planning specialists' scratch-tree probes."""

    specialist_probe: bool | None
    specialist_probe_max: int | None
    specialist_probe_budget_sec: float | None
    specialist_probe_scratch_root: str | None


@dataclass(frozen=True)
class ForgeLoopRequest:
    """Every option of one ``forge-loop`` invocation, grouped by concern."""

    campaign: CampaignTarget
    budget: TimeBudget
    git_branch: str | None
    gpu: GpuOptions
    task_context: TaskContext
    artifacts: RunArtifacts
    agent: AgentRuntimeOptions
    roles: AgentRoleOptions
    loop_shape: LoopShape
    roofline_ceiling: CeilingSwitch
    roofline_target: float
    prepare_task: bool
    identity: TaskIdentity
    knowledge: KnowledgeOptions
    probe: SpecialistProbeOptions
    baseline_json: str
    resume: bool


@click.command("forge-loop")
@option_group(
    "request",
    ForgeLoopRequest,
    option_group(
        "campaign",
        CampaignTarget,
        click.option("--kernel", default=None, help="Fresh campaign: kernel file to optimize"),
        click.option("--driver", default=None, help="Fresh campaign: validation/bench driver"),
        click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir"),
        click.option(
            "--snr-threshold",
            default=DEFAULT_SNR_THRESHOLD_DB,
            type=float,
            help="Fresh campaign: threshold in dB the driver's correctness "
            "suite must clear (stored immutably in the campaign config; "
            "ignored on --resume). A KEEP needs this and a measured gain "
            "over the incumbent; assembly adds the task's own suite.",
        ),
    ),
    option_group(
        "budget",
        TimeBudget,
        click.option(
            "--max-hours",
            default=1.0,
            type=float,
            callback=_validate_max_hours,
            help="Max runtime hours (default: 1.0, minimum 1.0). Budgets "
            ">2 hours enable Analysis profiling and, for single-lane "
            "rounds, Plan Critic review.",
        ),
        click.option(
            "--session-timeout-sec",
            default=None,
            type=click.IntRange(min=1),
            help="Wall-clock budget for one implementer session (seconds). Overrides "
            "the value computed from --max-hours; the claude backend cuts a "
            "session at this deadline and the session is told about it.",
        ),
        click.option(
            "--deadline-unix",
            default=0.0,
            type=float,
            help="Absolute UNIX deadline shared by preparation and optimization.",
        ),
    ),
    click.option(
        "--git-branch",
        default=None,
        help="Fresh campaign: development branch to optimize on (checked out "
        "before the immutable campaign config is snapshotted).",
    ),
    _gpu_options("Hardware SKU for KB identities, e.g. mi355x"),
    option_group(
        "task_context",
        TaskContext,
        click.option(
            "--kernel-backend",
            default=None,
            help=("Fresh campaign: kernel backend override. Unsupported kernel backends fall back to flydsl."),
        ),
        click.option(
            "--program-md-file", default=None, help="Fresh campaign: optional task context copied into the campaign"
        ),
        click.option(
            "--invocation-spec-file",
            default=None,
            help="Path to a Hyperloom invocation-spec JSON used by task preparation.",
        ),
    ),
    option_group(
        "artifacts",
        RunArtifacts,
        click.option(
            "--experiments-dir",
            default=None,
            help="Diagnostics/checkpoint root (profiles, optimization_potential, "
            "tracker checkpoint). Defaults to <workspace>/forge_experiments. "
            "Resume artifacts always live under the workspace regardless.",
        ),
        click.option(
            "--aiter-cache-max-gb",
            default=4.0,
            type=click.FloatRange(min=0.0),
            help=(
                "Per-attempt AITER cache soft limit in GiB (default: 4). "
                "LRU pruning targets 75% of the limit; 0 disables in-run pruning."
            ),
        ),
        click.option(
            "--experiment-id",
            default=None,
            help="Caller-owned experiment ID; recorded for external checkpoint recovery.",
        ),
        click.option(
            "--experience-id",
            default="",
            help="Unique KB run identity, independent of the checkpoint experiment ID.",
        ),
        click.option("--result-json", default=None, help="Write the result dict here (also printed)"),
    ),
    _agent_runtime_options,
    option_group(
        "roles",
        AgentRoleOptions,
        click.option("--permission-mode", default=None, help="Provider permission mode when supported"),
        click.option(
            "--profile-timeout-sec",
            default=ANALYSIS_TIMEOUT_SEC,
            type=int,
            help="Ceiling (seconds) for the single complete Analysis Agent "
            "session. The Agent persists phase and case artifacts for "
            "resume when the deadline is reached. Default 7200 (2 hours).",
        ),
        click.option(
            "--supervisor-backend",
            default="",
            callback=_validate_optional_agent_provider,
            help="Registered provider for the AVO supervisor. Omit to follow "
            "the provider of the Agent profile running the search, so one "
            "--agent-backend value controls every local agent.",
        ),
        click.option(
            "--profiling/--no-profiling",
            default=True,
            help="Allow Analysis hardware profiling and Implementer "
            "self-profiling guidance for long-horizon runs (>2 hours). "
            "Shorter runs keep Analysis static-only and omit that guidance. "
            "--no-profiling disables collection for every duration.",
        ),
    ),
    option_group(
        "loop_shape",
        LoopShape,
        click.option(
            "--nproc-per-node",
            default=1,
            type=click.IntRange(min=1),
            help="Ranks the driver self-launches via torchrun (collective tasks "
            "such as all-reduce). >1 profiles EVERY rank in its own "
            "rocprofv3 session, because wrapping the driver would only "
            "profile the launcher process, which runs no kernel. Default 1 "
            "(single-GPU, unchanged behavior).",
        ),
        click.option(
            "--lanes",
            default=3,
            type=click.IntRange(min=1, max=8),
            help="Implementer lanes per round. Above 1 the round's analysis is "
            "partitioned into that many non-overlapping plans, each run "
            "concurrently in its own workspace copy, and each candidate is "
            "measured on its own. Default 3: the lanes of a round run "
            "concurrently, so a lane costs a session rather than a share "
            "of the round's wall clock, and three is what the three "
            "specialist analyses can be divided into. The partition "
            "returns fewer when the evidence supports fewer. Above 1 "
            "needs a provider that declares session_env and is refused on "
            "one that does not; a provider without stop_hooks runs and is "
            "warned, because it can waste a round but not misreport one.",
        ),
        click.option(
            "--merge-stacking/--no-merge-stacking",
            default=True,
            help="Once consecutive iterations stop producing a new best, spend "
            "one iteration measuring two archived rejected gains applied "
            "together, chosen for winning on different cases. Costs a "
            "measurement but no Implementer session. Default on; this "
            "applies at every --lanes setting, so turn it off to compare "
            "against a run that predates it.",
        ),
        click.option(
            "--bench-repeat",
            default=1,
            type=click.IntRange(min=1),
            help="How many times each bench repeats its measurement in-process, "
            "reporting the per-case median. Default 1 (single shot). >1 "
            "shrinks run-to-run spread. Requires a driver that accepts "
            "--repeat; the flag is omitted entirely when this is 1.",
        ),
        click.option(
            "--commit-new-path",
            "commit_new_paths",
            multiple=True,
            help="Workspace-relative path or glob naming a file the agent may "
            "CREATE and still have committed with a KEEP (repeatable, e.g. "
            "--commit-new-path configs/*.json). Untracked files are "
            "otherwise never staged and never removed by a REVERT. A '*' "
            "does not cross a directory separator and '**' is rejected; "
            "name each level instead. Protected measurement paths are "
            "never admitted however they are spelled. Immutable per "
            "campaign: it is snapshotted into campaign_config.json and "
            "read back on --resume.",
        ),
    ),
    _roofline_ceiling_option(
        "Estimate the per-shape theoretical achievable latency the campaign "
        "measures its attainment against. 'off' (default) skips it. 'on' estimates "
        "it on a fresh campaign once the baseline is measured, which costs a "
        "profiler pass and an analyst session, and on --resume reuses the ceiling "
        "the campaign already estimated, estimating again only when that report "
        "can no longer be read. Pass it on every --resume that should keep it."
    ),
    click.option(
        "--roofline-target",
        default=0.0,
        type=float,
        help="Stop the campaign once mean per-case attainment (ceiling / measured, "
        "equal-weight across scored cases) reaches this fraction, e.g. 0.86. "
        "Requires a ceiling. Zero (default) disables the gate: a ceiling is an "
        "estimate whose arithmetic nothing checks, so stopping on one is opt-in. "
        "It never affects KEEP, which stays a measurement.",
    ),
    click.option(
        "--prepare-task/--no-prepare-task",
        default=True,
        help="Pre-loop task preparation (default on, fresh campaigns only). "
        "Before the loop, run a deterministic preflight of the driver "
        "against the loop's stdout contract; if it fails invoke ONE agent "
        "that authors/repairs the graph-timed measurement driver (never "
        "the kernel/source), then re-check. Skipped on --resume, whose "
        "driver contract is already fixed by the campaign.",
    ),
    option_group(
        "identity",
        TaskIdentity,
        click.option(
            "--task-type",
            default="",
            help="Task type (e.g. flydsl2flydsl, repository, image_kernel). "
            "'repository'/'image_kernel' enable multi-file / whole-repo "
            "handling; anything else keeps the single-file behavior.",
        ),
        click.option(
            "--source-files",
            default="",
            help="Comma/newline-separated implementation entry points used for "
            "orientation, profiling, JIT hints, and KB identity. This is not "
            "an edit allowlist; --kernel remains the anchor.",
        ),
        click.option(
            "--target-functions",
            default="",
            help="Comma-separated target kernel/function hints. Used for PMC "
            "filtering, source mapping, and agent orientation; it does not "
            "restrict which functions may be edited.",
        ),
        click.option(
            "--framework",
            default="",
            help="Explicit framework identity for the experience KB slug "
            "(vllm/sglang/aiter, or 'standalone' for a framework-less "
            "file). Authoritative when given; otherwise inferred from the "
            "file that defines the target operation, falling back to "
            "'unknown' when no known owner is found.",
        ),
        click.option(
            "--operator-name",
            default="",
            help="Logical operator identity used by profiling and the "
            "experience page key (for example, the traced operation name).",
        ),
    ),
    option_group(
        "knowledge",
        KnowledgeOptions,
        click.option(
            "--experience-kb/--no-experience-kb",
            default=True,
            help="Read and publish forge-loop experience KB entries (default on). "
            "Internal callers with their own KB lifecycle, such as "
            "forge-rewrite-by-flydsl, disable this explicitly.",
        ),
        click.option(
            "--kb-warmstart/--no-kb-warmstart",
            "kb_warmstart_enabled",
            default=True,
            help="Apply the best matching KB solution before iteration 1 (default on, "
            "and only with --experience-kb). A caller that prepares the workspace "
            "itself, such as forge-fuse, turns this off to keep publishing while "
            "never replaying a stored patch over a tree it already staged.",
        ),
        click.option(
            "--producer",
            default="",
            help="System owning the candidate stream these records belong to (default: "
            "the forge-loop's own). A producer has its own index in the KB "
            "identity scheme, so a pipeline driving this command as a subprocess "
            "can keep its records out of the kernel campaigns' ranking.",
        ),
        click.option(
            "--pr-kb/--no-pr-kb",
            default=None,
            help="Inject upstream pull-request references from the PR Monitor "
            "as Implementer prior knowledge (default off). Falls back to "
            "PR_KB_ENABLE when unset.",
        ),
    ),
    option_group(
        "probe",
        SpecialistProbeOptions,
        click.option(
            "--specialist-probe/--no-specialist-probe",
            default=None,
            help="Let the read-only planning specialists measure one variant per probe "
            "in a scratch tree, instead of only arguing about a dispatch constant "
            "(default on). Each probe re-runs the workspace driver for one case "
            "with declared constants overridden; it queues on the same device lock "
            "the fan-out lanes take, and never touches the canonical tree. Falls "
            "back to FORGE_SPECIALIST_PROBE when unset.",
        ),
        click.option(
            "--specialist-probe-max",
            default=None,
            type=click.IntRange(min=1),
            help="Probes ONE analysis round may make in total, shared by every "
            "specialist it dispatches (default 6). Every call counts, including "
            "one that is refused. Falls back to FORGE_SPECIALIST_PROBE_MAX when "
            "unset.",
        ),
        click.option(
            "--specialist-probe-budget-sec",
            default=None,
            type=click.FloatRange(min=1.0),
            help="Seconds on the device ONE analysis round may spend probing, shared by "
            "every specialist it dispatches (default 600). Cut down further at "
            "call time so no probe can leave a specialist without the time to "
            "write its analysis. Falls back to FORGE_SPECIALIST_PROBE_BUDGET_SEC "
            "when unset.",
        ),
        click.option(
            "--specialist-probe-scratch-root",
            default=None,
            help="Where the round scratch trees are created. Must be absolute and lie "
            "outside the workspace. Default: <experiments-dir>/specialist_probe, "
            "or a sibling of the workspace when that would land inside it. Falls "
            "back to FORGE_SPECIALIST_PROBE_SCRATCH_ROOT when unset.",
        ),
    ),
    click.option(
        "--baseline-json",
        default="",
        help="JSON file holding a scoring anchor measured outside this loop: "
        '{"wall_ms": <float>, "case_times": {"<case id>": <ms>}}. Every speedup '
        "this run reports then divides by those per-case times instead of by the "
        "kernel the run starts from, which is what a caller wants when that "
        "kernel already replaced something else (a rewrite's port replacing its "
        "source). The loop still benches the starting kernel, but as the search "
        "start rather than the anchor. Omit it and the loop anchors on its own "
        "first bench, as it always has.",
    ),
    click.option("--resume", is_flag=True, help="Resume the campaign stored in the exact workspace"),
)
def forge_loop(request: ForgeLoopRequest):
    """Run ONE Forge IterationLoop as a standalone subprocess (CLI-ized kernel backend)."""
    workspace_dir = request.campaign.workspace_dir
    resume = request.resume
    max_hours = request.budget.max_hours
    result_json = request.artifacts.result_json
    long_horizon = _is_long_horizon(max_hours)
    critic_enabled = bool(long_horizon)
    try:
        knowledge_config = KnowledgeConfig.from_env()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _reject_incompatible_loop_options(producer=request.knowledge.producer)

    from kernelforge.loop.campaign_config import CampaignConfigStore
    from kernelforge.loop.run_state import WorkspaceLock, WorkspaceLockError

    # Absolute deadline shared by task preparation and optimization.
    deadline = _ForgeDeadline.for_budget(request.budget)

    if request.roles.profile_timeout_sec <= 0:
        raise click.ClickException("--profile-timeout-sec must be positive")

    workspace = Path(workspace_dir).resolve()
    workspace_lock = WorkspaceLock(workspace / "forge_experiments" / "workspace.lock")
    try:
        workspace_lock.acquire()
    except WorkspaceLockError as error:
        raise click.ClickException(str(error)) from error
    click.get_current_context().call_on_close(workspace_lock.release)

    campaign_store = CampaignConfigStore(str(workspace))
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"
    _check_workspace_matches_resume(resume=resume, campaign_store=campaign_store)
    from kernelforge.cli.relay import resolve_relay_policy

    agent_runtime_overrides = _agent_runtime_overrides(request.agent)
    requested_relay_policy = resolve_relay_policy(
        Config.from_env(workspace=str(workspace), knowledge_config=knowledge_config, **agent_runtime_overrides)
    )
    resolution = _resolve_campaign(request, workspace, requested_relay_policy)
    campaign = resolution.campaign

    # Resume tooling may override the complete Analysis workflow deadline via FORGE_PROFILE_TIMEOUT_SEC;
    # --profile-timeout-sec is the default when the env is unset.
    _env_timeout = os.environ.get("FORGE_PROFILE_TIMEOUT_SEC")
    profile_timeout_sec = _profile_timeout_override(_env_timeout, request.roles.profile_timeout_sec)

    kernel = str((workspace / campaign.kernel_path).resolve())
    driver = str((workspace / campaign.driver_path).resolve())
    source_files_list = [str((workspace / path).resolve()) for path in campaign.source_files]
    target_functions_list = list(campaign.target_functions)
    snr_threshold = campaign.snr_threshold
    kernel_backend = campaign.kernel_backend
    task_type = campaign.task_type
    git_branch = campaign.git_branch
    operator_name = campaign.operator_name
    # Measurement semantics come from the campaign, never from this invocation's defaults.
    nproc_per_node = campaign.nproc_per_node
    bench_repeat = campaign.bench_repeat
    # From the campaign for the same reason: a resumed session that fell back to an empty allowlist could neither ship
    # nor remove the new file an earlier session was configured to.
    commit_new_paths = list(campaign.commit_new_paths)
    profiling_enabled = bool(request.roles.profiling and long_horizon)

    config = _session_config(request.probe, campaign, workspace, knowledge_config, agent_runtime_overrides)
    from kernelforge.cli.relay import probe_agent_profiles, resolve_session_relay

    try:
        session_relay = resolve_session_relay(campaign.agent_relay, config)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    startup_availability = probe_agent_profiles(session_relay, config=config, workspace=str(workspace))
    # Two roots by design: resume artifacts (run_state/candidates/best) always live under
    # <workspace>/forge_experiments (campaign_root); diagnostics and the external-recovery checkpoint go to
    # --experiments-dir when the caller supplies a distinct one (e.g. Hyperloom's output dir), else campaign_root.
    experiments_dir = request.artifacts.experiments_dir
    config.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else campaign_root
    config.experiments_dir.mkdir(parents=True, exist_ok=True)
    _isolate_aiter_cache(config.experiments_dir, request.artifacts.aiter_cache_max_gb, source_files_list)

    from kernelforge.loop.iteration.iteration_types import IterationConfig, LoopAgents, LoopCallbacks
    from kernelforge.tracker import ExperimentTracker, UsageAccumulator, UsageLedgerFile
    from kernelforge.loop.gate_settings import GateMeasurement
    from kernelforge.orchestrator.agent import ImplementerSetup, ImplementerTask, SessionOptions

    iter_config = IterationConfig(
        kernel_file=kernel,
        driver_script=driver,
        canonical_driver_sha256=campaign.driver_sha256,
        campaign_base_commit=campaign.base_commit,
        snr_threshold=snr_threshold,
        max_time_hours=max(0.05, max_hours),
        deadline_unix=deadline.deadline_unix - deadline.finalize_reserve_sec,
        git_branch=git_branch,
        workspace_dir=workspace_dir,
        experiment_id=request.artifacts.experiment_id or "",
        backend=kernel_backend.split("-", 1)[0],
        kernel_backend=kernel_backend,
        task_type=task_type,
        source_files=source_files_list,
        target_functions=target_functions_list,
        # Operator E2E time share (percent) for the baseline potential estimator.
        operator_name=operator_name,
        implementation_signature=campaign.implementation_signature,
        implementation_identity=dict(campaign.implementation_identity),
        # Rank count for collective tasks; selects the per-rank profiling backend.
        nproc_per_node=nproc_per_node,
        # Measurement fidelity: in-process repeats within each independent bench.
        bench_repeat=bench_repeat,
        lanes=request.loop_shape.lanes,
        merge_stacking=request.loop_shape.merge_stacking,
        # New files a KEEP may carry; a REVERT removes exactly the same set.
        commit_new_paths=commit_new_paths,
        # The attainment of the per-shape ceiling that ends the run; the ceiling itself is estimated inside the loop.
        roofline_target=_validate_roofline_target(request.roofline_target, request.roofline_ceiling.enabled),
    )
    baseline_json = request.baseline_json
    if baseline_json:
        # The anchor every speedup divides by, and the wall time published beside it, both come from the caller's
        # measurement. ``baseline_wall_ms`` is deliberately left unset so the loop still benches the kernel it starts
        # from -- that bench is the search start and the first incumbent, not the anchor.
        external_baseline_ms, external_baseline_case_times = _load_external_baseline(baseline_json)
        iter_config.baseline_case_times = dict(external_baseline_case_times)
        iter_config.pristine_baseline_wall_ms = external_baseline_ms
    tracker = ExperimentTracker(config.experiments_dir)
    # Published at a fixed path from the first call onwards, so a caller that never asked for --result-json, or that
    # loses this process outright, can still read what the run spent.
    usage_ledger = UsageLedgerFile(config.experiments_dir)
    usage = UsageAccumulator(on_update=usage_ledger.publish)
    usage_ledger.publish(usage.totals())

    run = _ForgeRun(
        request=request,
        campaign=campaign,
        campaign_store=campaign_store,
        config=config,
        iter_config=iter_config,
        tracker=tracker,
        usage=usage,
        deadline=deadline,
    )
    # The caller-owned experiment ID is an EXTERNAL recovery channel, deliberately independent of the internal
    # per-segment experiment identity (each resume segment gets a fresh ID so the campaign parent/child chain stays
    # intact).
    caller_experiment_id = run.caller_experiment_id
    _ensure_recovery_experiment(tracker, caller_experiment_id, kernel=kernel)
    program_md = _program_context(
        resolution,
        campaign_store,
        resume=resume,
        experience_kb=request.knowledge.experience_kb,
        workspace_dir=workspace_dir,
    )

    # Pre-loop task preparation (fresh campaigns only): ensure the driver conforms to the loop's stdout contract
    # BEFORE base_sha is captured, so any scaffolding the prep step commits becomes part of pristine (never the
    # solution diff).
    if request.prepare_task and not resume:
        _prepare_measurement_driver(run, program_md)

    if resolution.save_deferred:
        run = _persist_prepared_campaign(run, workspace, resolution.program_text)

    assembly = _prepare_assembly_task(run, program_md)
    assembly_preparation = assembly.preparation
    source_files_list = assembly.source_files
    kb_warmstart_enabled = assembly.kb_warmstart_enabled
    program_md = assembly.program_md

    # Resolved before the loop is built: the ceiling estimator captures it, and the raw option is ``None`` by default.
    session_timeout_sec = _forge_session_timeout_sec(max_hours, request.budget.session_timeout_sec)

    # Construct the loop only after task preparation has resolved the profiling contract; IterationLoop snapshots that
    # readiness in its runtime state.
    loop_runner = _build_loop_runner(run, source_files_list, session_timeout_sec)

    warm_start = _read_kb_warm_start(run, program_md, kb_warmstart_enabled)
    program_md = warm_start.program_md

    # Source mapping, profiling, profile analysis, and potential analysis are produced together by the commit-bound
    # Analysis Agent.
    iter_config.program_md = program_md

    # Hook-capable providers gate before stop; resumable providers apply the same canonical gate between turns.
    gate_on = True
    # No hard edit budget: a session making steady progress must not be cut off for editing a lot.
    max_blocks = 10
    # A turn cap never bounded time: it fired on 2.2% of sessions, so a session that neither converged nor capped ran
    # until something outside killed it.
    config.max_turns = FORGE_IMPLEMENTER_TURN_BACKSTOP
    print(
        f"  Implementer session budget: {session_timeout_sec}s "
        f"(campaign budget {max_hours:g}h; turn backstop {config.max_turns})"
    )
    # PR references are independent of the experience-KB lifecycle.
    pr_task_context, pr_kb_repo = _collect_pr_kb_context(run)

    selected_runtime = agent_runtime(config)
    implementer_setup = ImplementerSetup(
        ImplementerTask(
            program_md=program_md,
            kernel_backend_name=kernel_backend,
            pre_task_context=pr_task_context,
            pr_kb_repo=pr_kb_repo,
            task_type=task_type,
            source_files=source_files_list,
            # Function names, not paths: nothing to rebind onto a lane.
            target_functions=target_functions_list,
            profiling_enabled=profiling_enabled,
            commit_new_paths=iter_config.commit_new_paths,
        ),
        GateMeasurement(
            driver_script=driver,
            snr_threshold=snr_threshold,
            max_blocks=max_blocks,
            stage_timeout_sec=iter_config.validate_stage_timeout_sec,
            bench_timeout_sec=iter_config.bench_timeout_sec,
            bench_repeat=bench_repeat,
        ),
        SessionOptions(
            session_timeout_sec=session_timeout_sec,
            permission_mode=request.roles.permission_mode,
            agent_backend=selected_runtime.provider,
            insession_gate=gate_on,
        ),
    )
    from kernelforge.cli.relay import SearchActorBuildContext, start_search_actors
    from kernelforge.orchestrator.analysis import make_analysis_agent_service
    from kernelforge.orchestrator.orchestration import default_specialist_definitions

    specialist_definitions = default_specialist_definitions()
    actor_builder = SearchActorBuildContext(
        config=config,
        setup=implementer_setup,
        lane_binding=LaneBinding(iter_config.workspace_dir, driver, source_files_list),
        lanes=request.loop_shape.lanes,
        program_md=program_md,
        specialist_definitions=specialist_definitions,
        critic_enabled=critic_enabled,
        usage=usage,
        supervisor_backend=request.roles.supervisor_backend,
    )
    try:
        actors = start_search_actors(
            session_relay,
            actor_builder,
            availability=startup_availability,
            workspace=str(workspace),
            resume_profile_id=_resumed_profile_id(state_path),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(f"Agent startup availability check failed: {error}") from error
    start_bundle = actors.start_bundle
    effective_implementer = start_bundle.profile.provider
    effective_implementer_model = start_bundle.profile.model

    analysis_service = make_analysis_agent_service(
        config=actors.analysis_config(config),
        usage=usage,
        timeout_sec=profile_timeout_sec,
        profiling_enabled=profiling_enabled,
    )
    analysis_mode = "profiled" if profiling_enabled else "static-only"
    print("  Analysis Agent: " + analysis_mode)
    print(f"  Orchestration: enabled with parallel specialists ({', '.join(sorted(specialist_definitions))})")
    print(
        "  Plan Critic: "
        + ("enabled (long-horizon, same backend/model)" if critic_enabled else "disabled (requires --max-hours > 2)")
    )

    reporter = _RunReporter(
        run=run,
        loop_runner=loop_runner,
        warm_start=warm_start,
        assembly_preparation=assembly_preparation,
        implementer=_Implementer(effective_implementer, effective_implementer_model),
    )
    _build_result = reporter.build_result
    _publish_remote_best = reporter.publish_remote_best
    _checkpoint_on_best_committed = reporter.checkpoint_on_best_committed

    asyncio.run(
        loop_runner.run(
            LoopAgents(
                agent_fn=start_bundle.agent_fn,
                agent_factory=start_bundle.agent_factory if request.loop_shape.lanes > 1 else None,
                analysis_service=analysis_service,
                orchestration_service=start_bundle.orchestration_service,
                supervisor_fn=start_bundle.supervisor_fn,
                search_actors=actors.loop_search_actors,
            ),
            callbacks=LoopCallbacks(
                on_best_committed=_checkpoint_on_best_committed,
                on_best_ready=_publish_remote_best,
            ),
            usage=usage,
            workspace_lock_held=True,
        )
    )

    kb_experience = reporter.publish_final_best(llm_summary=deadline.remaining(reserve=30.0) >= 60.0)
    loop_runner._checkpoint_llm_usage()
    experiment_id = getattr(loop_runner.experiment, "experiment_id", None)
    loop_runner.llm_usage = usage.totals()
    _record_experiment_accounting(
        tracker,
        experiment_id=experiment_id,
        llm_usage=loop_runner.llm_usage,
        kb_experience=kb_experience,
    )

    # Final result: full kb_experience.
    result = _build_result(kb_experience=kb_experience)
    if result_json:
        atomic_write_json(result_json, result)
    click.echo(format_forge_loop_result(result))
    _write_pr_provenance(
        workspace_dir=workspace_dir,
        surfaced=iter_config.pr_reference_labels,
        winning_iteration=int(result.get("best_iteration") or 0),
        experiment_id=str(result.get("experiment_id") or ""),
    )
