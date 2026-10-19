# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-loop`` command: one Forge IterationLoop run as a standalone subprocess."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click

from kernelforge.cli.options import (
    CEILING_ON,
    _agent_runtime_options,
    _agent_runtime_overrides,
    _normalize_gpu_type,
    _roofline_ceiling_option,
    _validate_max_hours,
    _validate_optional_agent_provider,
    _validate_roofline_target,
)
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
from kernelforge.cli.relay import agent_relay_summary, relay_switch_possible
from kernelforge.cli.session_budget import _forge_session_timeout_sec, _is_long_horizon
from kernelforge.config import Config
from kernelforge.knowledge.kb_store.config import KnowledgeConfig
from kernelforge.loop.knowledge_integration import (
    WarmStartRollbackError,
    git_head,
    kb_reference_program_md,
    kb_read_status,
    kb_warmstart,
    mark_kb_reference_rejected,
    write_experience_to_kb,
)
from kernelforge.loop.recovery import (
    atomic_write_json,
    load_published_best,
    publish_warm_start_recovery,
    rollback_unpublished_warm_start,
)
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB, aggregate_regression_detail


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
    enabled: bool,
    workspace_dir: str,
    driver_script: str,
    source_files: Sequence[str],
    agent_provider: str,
    agent_model: str,
    session_timeout_sec: int,
    sandbox_mode: str,
):
    """Build the callable the loop uses to estimate a ceiling, or ``None``.

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
    if not enabled:
        return None

    async def estimate(*, case_ids: Sequence[str], case_ms: dict[str, float]):
        from kernelforge.mcp_server.tools.bench import bench_command
        from kernelforge.roofline_ceiling.command import resolve_analyst_backend
        from kernelforge.roofline_ceiling.estimate import estimate_ceiling

        return await estimate_ceiling(
            resolve_analyst_backend(agent_provider, agent_model, session_timeout_sec, sandbox_mode=sandbox_mode),
            workspace=workspace_dir,
            performance_command=bench_command(driver_script),
            kernel_files=[str(path) for path in source_files],
            driver_script=driver_script,
            known_case_ids=list(case_ids),
            known_case_ms=dict(case_ms),
            agent_model=agent_model,
            agent_timeout_sec=session_timeout_sec,
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


class _RunReporter:
    """How one forge-loop run reports its result: the file it writes, and the KB it publishes to.

    The runner calls three of these back mid-run, so they belong to one owner
    rather than to five closures over the same twenty values.
    """

    def __init__(
        self,
        *,
        loop_runner,
        workspace_dir,
        campaign_root,
        tracker,
        config,
        kernel,
        kernel_backend,
        base_sha,
        framework,
        operator_name,
        usage,
        result_json,
        experience_kb,
        experience_id,
        caller_experiment_id,
        kb_pristine_baseline_ms,
        kb_reused_speedup,
        source_files_list,
        target_functions_list,
        assembly_preparation,
        effective_implementer,
        effective_implementer_model,
        remote_publication,
    ) -> None:
        self.loop_runner = loop_runner
        self.workspace_dir = workspace_dir
        self.campaign_root = campaign_root
        self.tracker = tracker
        self.config = config
        self.kernel = kernel
        self.kernel_backend = kernel_backend
        self.base_sha = base_sha
        self.framework = framework
        self.operator_name = operator_name
        self.usage = usage
        self.result_json = result_json
        self.experience_kb = experience_kb
        self.experience_id = experience_id
        self.caller_experiment_id = caller_experiment_id
        self.kb_pristine_baseline_ms = kb_pristine_baseline_ms
        self.kb_reused_speedup = kb_reused_speedup
        self.source_files_list = source_files_list
        self.target_functions_list = target_functions_list
        self.assembly_preparation = assembly_preparation
        self.effective_implementer = effective_implementer
        self.effective_implementer_model = effective_implementer_model
        self.remote_publication = remote_publication

    def build_result(self, kb_experience) -> dict:
        """Assemble the loop result dict from live runner state."""
        loop_runner = self.loop_runner
        anchors = _search_anchors(loop_runner, pristine_fallback=self.kb_pristine_baseline_ms)
        search_start_ms = anchors.search_start_ms
        search_start_mean_case_speedup = anchors.search_start_mean_case_speedup
        pristine_ms = anchors.pristine_ms
        best = getattr(loop_runner, "best_wall_ms", None)
        total_speedup = getattr(loop_runner, "best_mean_case_speedup", None)
        incremental_speedup = (
            float(total_speedup) / float(search_start_mean_case_speedup)
            if total_speedup and search_start_mean_case_speedup
            else None
        )
        exp_id = getattr(loop_runner.experiment, "experiment_id", None)
        state_best = loop_runner.run_state.best
        best_iteration = getattr(state_best, "iteration", 0)
        best_commit = getattr(state_best, "commit_hash", "")
        # A validated warm-start is published before IterationLoop creates a run-state best.
        if not best_commit:
            published = load_published_best(self.workspace_dir) or {}
            if int(published.get("iteration", -1)) == 0:
                pristine_ms = published.get("pristine_baseline_ms") or published.get("baseline_wall_ms") or pristine_ms
                search_start_ms = published.get("search_start_ms") or published.get("best_wall_ms") or search_start_ms
                best = published.get("best_wall_ms")
                total_speedup = published.get("mean_case_speedup")
                search_start_mean_case_speedup = (
                    published.get("search_start_mean_case_speedup") or total_speedup or search_start_mean_case_speedup
                )
                incremental_speedup = 1.0
                best_iteration = 0
                best_commit = str(published.get("commit_hash") or "")
        # KEEP is decided on the mean of per-case speedups while these are aggregate wall times, so the two can
        # legitimately disagree by a hair.
        aggregate_regression = aggregate_regression_detail(
            baseline_ms=pristine_ms,
            best_ms=best,
            mean_case_speedup=total_speedup,
        )
        result = {
            "baseline_ms": pristine_ms,
            "pristine_baseline_ms": pristine_ms,
            "search_start_ms": search_start_ms,
            "best_ms": best,
            "mean_case_speedup": total_speedup,
            "search_start_mean_case_speedup": (search_start_mean_case_speedup),
            "aggregate_regression": aggregate_regression,
            "improved": bool(total_speedup and total_speedup > 1.0) and not aggregate_regression,
            "total_improved": bool(total_speedup and total_speedup > 1.0) and not aggregate_regression,
            "incremental_improved": bool(total_speedup and total_speedup > search_start_mean_case_speedup),
            "improved_during_search": bool(total_speedup and total_speedup > search_start_mean_case_speedup),
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
            "remote_publication": _remote_publication_view(
                self.remote_publication,
                best_commit,
            ),
            "best_manifest": str(self.campaign_root / "best" / "manifest.json"),
            "optimization_report": str(self.campaign_root / "optimization_report.md"),
            "optimization_history": str(self.campaign_root / "optimization_history.md"),
            "persistence_degraded": bool(getattr(loop_runner, "persistence_degraded", False)),
            "persistence_errors": list(getattr(loop_runner, "persistence_errors", [])),
            "iteration_count": 0,
            "kb_experience": kb_experience,
            "agent_backend": self.effective_implementer,
            "agent_model": self.effective_implementer_model,
            "agent_relay": agent_relay_summary(loop_runner.run_state),
            "llm_usage": getattr(loop_runner, "llm_usage", {}) or {},
        }
        if self.assembly_preparation is not None:
            from kernelforge.assembly.prepare import select_result

            select_result(result, self.assembly_preparation)
        if exp_id:
            try:
                completed_experiment = self.tracker.get(exp_id)
            except FileNotFoundError:
                # An incomplete run has no tracker record; the result must still be emitted so callers can reject it.
                return result
            result["iteration_count"] = len(completed_experiment.iterations)
            result["checkpoint"] = completed_experiment.checkpoint
        return result

    def write_result_json(self, result: dict) -> None:
        """Persist the result dict to --result-json (best-effort; never raises)."""
        if not self.result_json:
            return
        try:
            atomic_write_json(self.result_json, result)
        except Exception as e:  # noqa: BLE001 - a snapshot write must never break the loop
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
        if not self.experience_kb:
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
                config=self.config,
                loop_runner=self.loop_runner,
                workspace_dir=self.workspace_dir,
                kernel=self.kernel,
                kernel_backend=self.kernel_backend,
                gpu_target=self.config.gpu_target,
                base_sha=self.base_sha,
                pristine_baseline_ms=self.kb_pristine_baseline_ms,
                reused_speedup=self.kb_reused_speedup,
                source_files=self.source_files_list,
                target_functions=self.target_functions_list,
                framework=self.framework,
                experience_id=self.experience_id,
                operator_name=self.operator_name,
                llm_summary=llm_summary,
                incremental_summary=incremental_summary,
                snr_db_override=snr_db_override,
                usage=self.usage,
            )
        except Exception as e:  # noqa: BLE001 - checkpoint publish must never break the loop
            status = {"written": False, "reason": f"error:{e!r}"}
        _record_remote_publication_result(
            self.remote_publication,
            commit=commit,
            result=status,
        )
        return status

    def publish_remote_best(self, result) -> None:
        """Publish the KEEP the runner just made durable.

        The runner calls this after the KEEP commit, run state, event and local
        best artifact are durable, but before post-KEEP profiling.
        """
        if not getattr(result, "kept", False):
            return
        commit = str(getattr(result, "commit_hash", "") or git_head(self.workspace_dir))
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
        best_commit = str(getattr(result, "commit_hash", "") or git_head(self.workspace_dir))
        anchors = _search_anchors(self.loop_runner)
        search_start_ms = anchors.search_start_ms
        search_start_mean_case_speedup = anchors.search_start_mean_case_speedup
        baseline_ms = anchors.pristine_ms
        best_ms = getattr(result, "wall_ms", None)
        mean_case_speedup = getattr(result, "mean_case_speedup", None)
        aggregate_regression = aggregate_regression_detail(
            baseline_ms=baseline_ms,
            best_ms=best_ms,
            mean_case_speedup=mean_case_speedup,
        )
        checkpoint = {
            "state": "best_committed",
            "decision": "KEEP",
            "experiment_id": (
                self.caller_experiment_id or (experiment.experiment_id if experiment is not None else "")
            ),
            "base_commit": self.base_sha,
            "best_commit": best_commit,
            "best_iteration": int(getattr(result, "iteration", 0) or 0),
            "baseline_ms": baseline_ms,
            "pristine_baseline_ms": baseline_ms,
            "search_start_ms": search_start_ms,
            "best_ms": best_ms,
            "mean_case_speedup": mean_case_speedup,
            "search_start_mean_case_speedup": (search_start_mean_case_speedup),
            "aggregate_regression": aggregate_regression,
            "improved": bool(mean_case_speedup and mean_case_speedup > 1.0) and not aggregate_regression,
            "total_improved": bool(mean_case_speedup and mean_case_speedup > 1.0) and not aggregate_regression,
            "incremental_improved": bool(mean_case_speedup and mean_case_speedup > search_start_mean_case_speedup),
            "improved_during_search": bool(mean_case_speedup and mean_case_speedup > search_start_mean_case_speedup),
            "total_speedup": mean_case_speedup,
            "incremental_speedup": (
                float(mean_case_speedup) / float(search_start_mean_case_speedup)
                if mean_case_speedup and search_start_mean_case_speedup
                else None
            ),
            "validation_passed": bool(getattr(result, "validation_passed", False)),
            "validation_summary": str(getattr(result, "validation_summary", "") or ""),
            "snr_db": getattr(result, "snr_db", None),
        }
        if experiment is not None or self.caller_experiment_id:
            try:
                if experiment is not None:
                    self.tracker.set_checkpoint(experiment.experiment_id, checkpoint)
                # Mirror onto the caller-owned record so an external hard kill can still recover this KEEP; refreshed
                # on every resumed segment.
                if self.caller_experiment_id:
                    self.tracker.set_checkpoint(self.caller_experiment_id, checkpoint)
            except Exception as e:  # noqa: BLE001 - never break the loop
                print(f"  [checkpoint] skipped ({e})", flush=True)
        self.write_result_json(self.build_result(kb_experience=None))


def _prepare_measurement_driver(
    *,
    config,
    workspace_dir,
    kernel,
    driver,
    program_md,
    program_md_file,
    target_functions_list,
    source_files_list,
    kernel_backend,
    snr_threshold,
    invocation_spec_file,
    deadline_unix,
    finalize_reserve_sec,
    nproc_per_node,
    usage,
    result_json,
    require_time,
) -> None:
    """Make the driver conform to the measurement contract, or leave with the reason it cannot.

    A driver that already conforms is left alone; one that does not is handed
    to the prep agent, and a preparation that fails ends the run rather than
    measuring against a driver nobody vouched for.
    """
    from kernelforge.loop.task_preparer import (
        declared_case_ids,
        preflight_task,
        prepare_task_sync,
    )

    # Preparation runs the driver, so it needs the rank count before the loop that normally exports it exists.
    if nproc_per_node > 1:
        os.environ["FORGE_NPROC_PER_NODE"] = str(nproc_per_node)
    else:
        os.environ.pop("FORGE_NPROC_PER_NODE", None)

    require_time("task preparation", 10.0)
    # The spec's declared suite gates this too: a driver that times a subset of the task's cases is not "already
    # conforming", it just measures less than the task asks for, and accepting it here skips the only step that
    # would have repaired it.
    try:
        expected_case_ids = declared_case_ids(invocation_spec_file)
    except ValueError as exc:
        # Continuing would switch the driver's case check off and spend the whole run measuring a suite the
        # operator never got to state.
        raise click.ClickException(str(exc)) from exc
    pf = preflight_task(
        driver=driver,
        snr_threshold=snr_threshold,
        require_graph=True,
        require_profile=True,
        require_ranks=max(1, int(nproc_per_node or 1)),
        deadline_unix=deadline_unix - finalize_reserve_sec,
        expected_case_ids=expected_case_ids,
    )
    if pf.ok:
        print("  [prepare] task already conforms to the driver contract; skipping")
        _persist_declared_spec(invocation_spec_file or "", driver)
    else:
        print(f"  [prepare] task does not conform ({pf.summary()}); invoking prep agent...")
        # The budget that actually applies is min(wall, what the per-kernel deadline leaves), and it decides how
        # many attempts ever start.
        from kernelforge.loop import task_preparer as _tp

        _prep_wall = min(
            float(_tp.PREPARE_MAX_WALL_SEC),
            max(0.0, deadline_unix - finalize_reserve_sec - time.time()),
        )
        print(
            f"  [prepare] budget: wall={_prep_wall:.0f}s "
            f"attempt_cap={_tp.PER_ATTEMPT_CAP_SEC}s "
            f"max_attempts={_tp.PREPARE_MAX_ATTEMPTS}"
        )
        prep = prepare_task_sync(
            config=config,
            workspace_dir=workspace_dir,
            kernel=kernel,
            driver=driver,
            program_md=program_md,
            target_functions=target_functions_list,
            source_files=source_files_list,
            kernel_backend=kernel_backend,
            snr_threshold=snr_threshold,
            preflight=pf,
            invocation_spec_file=invocation_spec_file or "",
            expected_case_ids=expected_case_ids,
            # Let the default PREPARE_MAX_WALL_SEC (3000s, sized for a cold-JIT preflight) apply; prepare_task
            # clamps it to the per-kernel deadline_unix below.
            deadline_unix=deadline_unix - finalize_reserve_sec,
            # A collective task needs a driver that launches its own ranks; the preparer cannot infer that from
            # the kernel source.
            nproc_per_node=nproc_per_node,
            read_only_files=[path for path in (program_md_file, invocation_spec_file) if path],
            usage=usage,
        )
        if prep.ok:
            # Cap the file list: an external driver bundle can legitimately publish dozens of files, and one run
            # scrolled ~700 cache paths through the operator's log for a 3-file change.
            _shown = prep.wrote_files[:12]
            _extra = len(prep.wrote_files) - len(_shown)
            print(
                f"  [prepare] prepared measurement driver in {prep.attempts} attempt(s); "
                f"wrote {', '.join(_shown) or '(none)'}" + (f" (+{_extra} more)" if _extra > 0 else "")
            )
            # Telemetry only: never let a missing field break a good prep.
            _pf_sec = getattr(prep.final_preflight, "duration_sec", 0.0) or 0.0
            if _pf_sec:
                _stages = ", ".join(
                    f"{name}={detail['seconds']:.0f}s"
                    for name, detail in (getattr(prep.final_preflight, "details", {}) or {}).items()
                    if isinstance(detail, dict) and "seconds" in detail
                )
                print(f"  [prepare] preflight: {_pf_sec:.0f}s" + (f" ({_stages})" if _stages else ""))
            if prep.audit_dir:
                print(f"  [prepare] audit: {prep.audit_dir}")
        else:
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
            err_payload = json.dumps(err_result)
            if result_json:
                atomic_write_json(result_json, err_result)
            click.echo(f"__FORGE_RESULT__{err_payload}__FORGE_RESULT__")
            sys.exit(2)


def _collect_pr_kb_context(
    iter_config,
    *,
    pr_kb,
    workspace,
    workspace_dir,
    kernel_backend,
    campaign,
    operator_name,
    target_functions_list,
    finalize_reserve_sec,
    remaining,
) -> tuple[str, str]:
    """Inject PR references into the loop's prompt context, and say which repo they came from.

    Independent of the experience-KB lifecycle, and skipped outright when
    too little of the deadline is left to collect them.
    """
    pr_task_context = ""
    pr_kb_repo = ""
    if not _pr_kb_enabled(pr_kb):
        return pr_task_context, pr_kb_repo
    from kernelforge.knowledge.pr_knowledge.context import (
        REASON_LOCAL_FAILURE,
        REASON_SKIPPED_DEADLINE,
    )

    if remaining(reserve=finalize_reserve_sec) < 20.0:
        print("  [pr-kb] skipped (deadline)")
        iter_config.pr_kb_event = _pr_refs_event_fields(
            REASON_SKIPPED_DEADLINE,
            {},
        )
    else:
        pr_result = _collect_pr_references(
            workspace_dir=workspace_dir,
            kernel_backend=kernel_backend or "",
            git_remote=_git_remote_url(workspace),
            source_files=campaign.source_files,
            operator_name=operator_name or "",
            target_functions=target_functions_list,
            budget_sec=min(PR_KB_BUDGET_SEC, remaining(reserve=finalize_reserve_sec)),
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
    warm_start_result: dict | None
    kb_pristine_baseline_ms: float | None
    kb_reused_speedup: float | None


def _read_kb_warm_start(
    iter_config,
    *,
    config,
    kernel,
    driver,
    workspace_dir,
    kernel_backend,
    target_functions_list,
    source_files_list,
    framework,
    operator_name,
    snr_threshold,
    bench_repeat,
    resume,
    experience_kb,
    kb_warmstart_enabled,
    program_md,
    result_json,
    experience_id,
    finalize_reserve_sec,
    remaining,
    base_sha,
    caller_experiment_id,
    tracker,
) -> _WarmStart:
    """Adopt the best prior solution the KB offers, so iteration 1 starts from it rather than from scratch."""
    # KB warm-start: apply the best prior solution as the starting point and inject its experience into the prompt, so
    # the agent continues from the best-known state instead of from scratch.
    kb_pristine_baseline_ms = None
    kb_reused_speedup = None
    warm = {
        "candidate": False,
        "read_reason": "resume" if resume else "deadline",
        "read_error": "",
    }
    warm_start_result = None
    if experience_kb and kb_warmstart_enabled and not resume and remaining(reserve=finalize_reserve_sec) >= 600:
        try:
            warm = kb_warmstart(
                config=config,
                kernel=kernel,
                driver=driver,
                workspace_dir=workspace_dir,
                kernel_backend=kernel_backend,
                target_functions=target_functions_list,
                framework=framework,
                snr_threshold=snr_threshold,
                source_files=source_files_list,
                operator_name=operator_name,
                bench_repeat=bench_repeat,
            )
        except WarmStartRollbackError as error:
            failure = click.ClickException(f"warm-start rollback failed; workspace may be inconsistent: {error}")
            failure.exit_code = 2
            raise failure from error
        warm.setdefault(
            "read_reason",
            "hit" if warm.get("candidate") else "solution_pages_missing",
        )
        warm.setdefault("read_error", "")
    elif not resume and not experience_kb:
        warm["read_reason"] = "disabled"
        print("  [kb] experience KB disabled by caller")
    elif not resume:
        print("  [kb] warm-start skipped: absolute deadline reserve")
    if warm.get("candidate"):
        kb_pristine_baseline_ms = warm.get("pristine_ms")
        if warm.get("applied"):
            # The floor this campaign starts from.
            kb_reused_speedup = warm.get("mean_case_speedup")
            iter_config.publication_baseline_wall_ms = warm.get("pristine_ms")
            try:
                published = publish_warm_start_recovery(
                    workspace_dir=workspace_dir,
                    base_commit=base_sha,
                    warm=warm,
                    caller_experiment_id=caller_experiment_id,
                    experience_id=experience_id,
                    tracker=tracker,
                    result_json=result_json,
                )
                if published:
                    warm_start_result = dict(published)
                    iter_config.warm_start_publication = dict(published)
                if published and published.get("persistence_degraded"):
                    print(
                        "  [warm-start] recovery publication degraded: "
                        + "; ".join(published.get("persistence_errors") or []),
                        flush=True,
                    )
                print(
                    "  [warm-start] published recoverable best before iteration 1",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - translated into a rollback below
                try:
                    rollback_unpublished_warm_start(
                        workspace_dir,
                        base_commit=base_sha,
                        result_json=result_json,
                    )
                except Exception as rollback_error:
                    raise click.ClickException(
                        "failed to publish validated warm-start and could not "
                        f"restore pristine workspace: {rollback_error}"
                    ) from rollback_error
                warm["applied"] = False
                warm["keep_baseline_ms"] = warm.get("pristine_ms")
                mark_kb_reference_rejected(
                    workspace_dir,
                    int(warm.get("applied_rank") or 0),
                    "publication_failed",
                )
                warm["program_md_addition"] = kb_reference_program_md(
                    workspace_dir,
                    detect_applied=False,
                ) or warm.get(
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
        warm_start_result=warm_start_result,
        kb_pristine_baseline_ms=kb_pristine_baseline_ms,
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


def _prepare_assembly_task(
    *,
    kernel_backend,
    config,
    kernel,
    driver,
    source_files_list,
    campaign,
    iter_config,
    workspace,
    program_md,
    kb_warmstart_enabled,
    snr_threshold,
    deadline_unix,
    finalize_reserve_sec,
    resume,
) -> _AssemblyPreparation:
    """Export and bind the compiler output an assembly task is optimized against.

    A task on any other backend passes through untouched.
    """
    unchanged = _AssemblyPreparation(None, source_files_list, kb_warmstart_enabled, program_md)
    if kernel_backend != "assembly":
        return unchanged
    from kernelforge.assembly.prepare import prepare_assembly, seed_source_baseline

    try:
        assembly_preparation = asyncio.run(
            prepare_assembly(
                config=config,
                kernel=kernel,
                driver=driver,
                sources=source_files_list,
                base_commit=campaign.base_commit,
                threshold=snr_threshold,
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


def _warm_start_only_result(
    warm: dict,
    warm_start_result,
    *,
    usage,
    result_json,
    warm_start_improvement_flags,
) -> None:
    """Report a validated warm start and stop, for a caller that asked for nothing more."""
    if not isinstance(warm_start_result, dict):
        raise click.ClickException("validated KB warm-start has no recoverable result")
    pristine_ms = float(warm["pristine_ms"])
    best_ms = float(warm["keep_baseline_ms"])
    mean_case_speedup = float(warm["mean_case_speedup"])
    best_commit = str(warm.get("applied_commit") or "")
    publication_state = _initial_remote_publication_state(warm)
    publication = _remote_publication_view(
        publication_state,
        best_commit,
    )
    kb_experience = {
        "read": kb_read_status(warm),
        "write": publication_state["last_result"],
        "publication": publication,
    }
    result = {
        **warm_start_result,
        "pristine_baseline_ms": pristine_ms,
        "search_start_ms": best_ms,
        "mean_case_speedup": mean_case_speedup,
        "search_start_mean_case_speedup": mean_case_speedup,
        # Like the published manifest's badge, ``improved`` is withheld when the wall times contradict the score.
        **warm_start_improvement_flags(
            pristine_ms=pristine_ms,
            best_ms=best_ms,
            mean_case_speedup=mean_case_speedup,
        ),
        "incremental_improved": False,
        "improved_during_search": False,
        "total_speedup": mean_case_speedup,
        "incremental_speedup": 1.0,
        "remote_publication": publication,
        "kb_experience": kb_experience,
        "llm_usage": usage.totals(),
        "returned_after_read_kb": True,
    }
    payload = json.dumps(result)
    if result_json:
        atomic_write_json(result_json, result)
    print(
        "  [kb] validated warm-start accepted; returning before iteration 1",
        flush=True,
    )
    click.echo(f"__FORGE_RESULT__{payload}__FORGE_RESULT__")
    return


class _ForgeDeadline:
    """The run's absolute deadline, and the reserve finalization needs out of it.

    Every stage asks the same two questions -- how long is left, and is there
    enough left to start this -- so they are answered in one place rather than
    by two closures over the same pair of numbers.
    """

    def __init__(self, deadline_unix: float, finalize_reserve_sec: float) -> None:
        self._deadline_unix = deadline_unix
        self._finalize_reserve_sec = finalize_reserve_sec

    def remaining(self, *, reserve: float = 0.0) -> float:
        """Seconds left before the deadline, less whatever reserve the caller must leave."""
        return max(0.0, self._deadline_unix - time.time() - reserve)

    def require(self, phase: str, minimum: float = 1.0) -> None:
        """Refuse to start a phase the finalization reserve no longer leaves room for."""
        if self.remaining(reserve=self._finalize_reserve_sec) < minimum:
            raise click.ClickException(f"absolute Forge deadline exhausted before {phase}")


def _reject_incompatible_loop_options(
    *,
    return_after_read_kb: bool,
    kernel_backend: str,
    experience_kb: bool,
    kb_warmstart_enabled: bool,
    producer: str,
) -> None:
    """Refuse an option combination the loop cannot honour, before anything is set up."""
    if return_after_read_kb and kernel_backend == "assembly":
        raise click.UsageError("--return-after-read-kb is incompatible with assembly preparation")
    if return_after_read_kb and not experience_kb:
        raise click.UsageError("--return-after-read-kb cannot be used with --no-experience-kb")
    if return_after_read_kb and not kb_warmstart_enabled:
        raise click.UsageError("--return-after-read-kb cannot be used with --no-kb-warmstart")
    if producer:
        from kernelforge.knowledge.kb_store.identity.kernel_recipe import KERNEL_RECIPE_PRODUCERS

        if producer not in KERNEL_RECIPE_PRODUCERS:
            raise click.UsageError(f"--producer must be one of: {', '.join(sorted(KERNEL_RECIPE_PRODUCERS))}")


def _check_workspace_matches_resume(
    *,
    resume: bool,
    state_path,
    campaign_store,
    has_run_artifacts: bool,
) -> None:
    """Refuse a workspace that does not match what ``--resume`` says about it.

    Resuming needs the records a prior campaign left; starting fresh needs a
    workspace that has none, so a second campaign cannot be read as a
    continuation of the first.
    """
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
    except Exception as exc:  # noqa: BLE001 - usage accounting must not fail the run
        click.echo(
            f"Warning: failed to record LLM usage for experiment {experiment_id}: {exc}",
            err=True,
        )
    try:
        tracker.set_kb_experience(experiment_id, kb_experience)
    except Exception as exc:  # noqa: BLE001 - KB accounting must not fail the run
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
            task_id=Path(kernel).stem,
            description=f"External recovery channel for {caller_experiment_id}",
            experiment_id=caller_experiment_id,
        )


def _program_context(
    campaign,
    campaign_store,
    *,
    campaign_save_deferred: bool,
    program_text: str,
    resume: bool,
    experience_kb: bool,
    workspace_dir: str,
) -> str:
    """The program context the implementer is given, plus any KB reference a resume should carry."""
    if campaign_save_deferred:
        # The immutable config (and its program.md) is not on disk yet; use the in-memory program text captured from
        # --program-md-file (matching read_program_md's "" for a campaign without program context).
        program_md = program_text or ""
    else:
        try:
            program_md = campaign_store.read_program_md(campaign)
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error
    if resume and experience_kb:
        reference_pointer = kb_reference_program_md(workspace_dir)
        if reference_pointer:
            program_md = program_md + "\n\n" + reference_pointer
    return program_md


@click.command("forge-loop")
@click.option("--kernel", default=None, help="Fresh campaign: kernel file to optimize")
@click.option("--driver", default=None, help="Fresh campaign: validation/bench driver")
@click.option("--workspace", "workspace_dir", required=True, help="Git workspace dir")
@click.option(
    "--snr-threshold",
    default=DEFAULT_SNR_THRESHOLD_DB,
    type=float,
    help="Fresh campaign: threshold in dB the driver's correctness "
    "suite must clear (stored immutably in the campaign config; "
    "ignored on --resume). A KEEP needs this and a measured gain "
    "over the incumbent; assembly adds the task's own suite.",
)
@click.option(
    "--max-hours",
    default=1.0,
    type=float,
    callback=_validate_max_hours,
    help="Max runtime hours (default: 1.0, minimum 1.0). Budgets "
    ">2 hours enable Analysis profiling and, for single-lane "
    "rounds, Plan Critic review.",
)
@click.option(
    "--session-timeout-sec",
    default=None,
    type=click.IntRange(min=1),
    help="Wall-clock budget for one implementer session (seconds). Overrides "
    "the value computed from --max-hours; the claude backend cuts a "
    "session at this deadline and the session is told about it.",
)
@click.option(
    "--deadline-unix",
    default=0.0,
    type=float,
    help="Absolute UNIX deadline shared by preparation and optimization.",
)
@click.option(
    "--git-branch",
    default=None,
    help="Fresh campaign: development branch to optimize on (checked out "
    "before the immutable campaign config is snapshotted).",
)
@click.option(
    "--gpu-target",
    default=None,
    help="ROCm compilation architecture, e.g. gfx950 (also exported to env)",
)
@click.option(
    "--gpu-type",
    default=None,
    callback=_normalize_gpu_type,
    help="Hardware SKU for KB identities, e.g. mi355x",
)
@click.option(
    "--kernel-backend",
    default=None,
    help=("Fresh campaign: kernel backend override. Unsupported kernel backends fall back to flydsl."),
)
@click.option("--program-md-file", default=None, help="Fresh campaign: optional task context copied into the campaign")
@click.option(
    "--invocation-spec-file",
    default=None,
    help="Path to a Hyperloom invocation-spec JSON used by task preparation.",
)
@click.option(
    "--experiments-dir",
    default=None,
    help="Diagnostics/checkpoint root (profiles, optimization_potential, "
    "tracker checkpoint). Defaults to <workspace>/forge_experiments. "
    "Resume artifacts always live under the workspace regardless.",
)
@click.option(
    "--aiter-cache-max-gb",
    default=4.0,
    type=click.FloatRange(min=0.0),
    help=(
        "Per-attempt AITER cache soft limit in GiB (default: 4). "
        "LRU pruning targets 75% of the limit; 0 disables in-run pruning."
    ),
)
@click.option(
    "--experiment-id",
    default=None,
    help="Caller-owned experiment ID; recorded for external checkpoint recovery.",
)
@click.option(
    "--experience-id",
    default="",
    help="Unique KB run identity, independent of the checkpoint experiment ID.",
)
@click.option("--result-json", default=None, help="Write the result dict here (also printed)")
@_agent_runtime_options
@click.option("--permission-mode", default=None, help="Provider permission mode when supported")
@click.option(
    "--profile-timeout-sec",
    default=ANALYSIS_TIMEOUT_SEC,
    type=int,
    help="Ceiling (seconds) for the single complete Analysis Agent "
    "session. The Agent persists phase and case artifacts for "
    "resume when the deadline is reached. Default 7200 (2 hours).",
)
@click.option(
    "--supervisor-backend",
    default="",
    callback=_validate_optional_agent_provider,
    help="Registered provider for the AVO supervisor. Omit to follow "
    "the effectively resolved Implementer provider, so one "
    "--agent-backend value controls every local agent.",
)
@click.option(
    "--profiling/--no-profiling",
    default=True,
    help="Allow Analysis hardware profiling and Implementer "
    "self-profiling guidance for long-horizon runs (>2 hours). "
    "Shorter runs keep Analysis static-only and omit that guidance. "
    "--no-profiling disables collection for every duration.",
)
@click.option(
    "--nproc-per-node",
    default=1,
    type=click.IntRange(min=1),
    help="Ranks the driver self-launches via torchrun (collective tasks "
    "such as all-reduce). >1 profiles EVERY rank in its own "
    "rocprofv3 session, because wrapping the driver would only "
    "profile the launcher process, which runs no kernel. Default 1 "
    "(single-GPU, unchanged behavior).",
)
@click.option(
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
)
@click.option(
    "--merge-stacking/--no-merge-stacking",
    default=True,
    help="Once consecutive iterations stop producing a new best, spend "
    "one iteration measuring two archived rejected gains applied "
    "together, chosen for winning on different cases. Costs a "
    "measurement but no Implementer session. Default on; this "
    "applies at every --lanes setting, so turn it off to compare "
    "against a run that predates it.",
)
@click.option(
    "--bench-repeat",
    default=1,
    type=click.IntRange(min=1),
    help="How many times each bench repeats its measurement in-process, "
    "reporting the per-case median. Default 1 (single shot). >1 "
    "shrinks run-to-run spread. Requires a driver that accepts "
    "--repeat; the flag is omitted entirely when this is 1.",
)
@click.option(
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
)
@_roofline_ceiling_option(
    "Estimate the per-shape theoretical achievable latency the campaign "
    "measures its attainment against. 'off' (default) skips it. 'on' estimates "
    "it on a fresh campaign once the baseline is measured, which costs a "
    "profiler pass and an analyst session, and on --resume reuses the ceiling "
    "the campaign already estimated, estimating again only when that report "
    "can no longer be read. Pass it on every --resume that should keep it."
)
@click.option(
    "--roofline-target",
    default=0.0,
    type=float,
    help="Stop the campaign once mean per-case attainment (ceiling / measured, "
    "equal-weight across scored cases) reaches this fraction, e.g. 0.86. "
    "Requires a ceiling. Zero (default) disables the gate: a ceiling is an "
    "estimate whose arithmetic nothing checks, so stopping on one is opt-in. "
    "It never affects KEEP, which stays a measurement.",
)
@click.option(
    "--prepare-task/--no-prepare-task",
    default=True,
    help="Pre-loop task preparation (default on, fresh campaigns only). "
    "Before the loop, run a deterministic preflight of the driver "
    "against the loop's stdout contract; if it fails invoke ONE agent "
    "that authors/repairs the graph-timed measurement driver (never "
    "the kernel/source), then re-check. Skipped on --resume, whose "
    "driver contract is already fixed by the campaign.",
)
@click.option(
    "--task-type",
    default="",
    help="Task type (e.g. flydsl2flydsl, repository, image_kernel). "
    "'repository'/'image_kernel' enable multi-file / whole-repo "
    "handling; anything else keeps the single-file behavior.",
)
@click.option(
    "--source-files",
    default="",
    help="Comma/newline-separated implementation entry points used for "
    "orientation, profiling, JIT hints, and KB identity. This is not "
    "an edit allowlist; --kernel remains the anchor.",
)
@click.option(
    "--target-functions",
    default="",
    help="Comma-separated target kernel/function hints. Used for PMC "
    "filtering, source mapping, and agent orientation; it does not "
    "restrict which functions may be edited.",
)
@click.option(
    "--framework",
    default="",
    help="Explicit framework identity for the experience KB slug "
    "(vllm/sglang/aiter, or 'standalone' for a framework-less "
    "file). Authoritative when given; otherwise inferred from the "
    "file that defines the target operation, falling back to "
    "'unknown' when no known owner is found.",
)
@click.option(
    "--operator-name",
    default="",
    help="Logical operator identity used by profiling and the "
    "experience page key (for example, the traced operation name).",
)
@click.option(
    "--experience-kb/--no-experience-kb",
    default=True,
    help="Read and publish forge-loop experience KB entries (default on). "
    "Internal callers with their own KB lifecycle, such as "
    "forge-rewrite-by-flydsl, disable this explicitly.",
)
@click.option(
    "--kb-warmstart/--no-kb-warmstart",
    "kb_warmstart_enabled",
    default=True,
    help="Apply the best matching KB solution before iteration 1 (default on, "
    "and only with --experience-kb). A caller that prepares the workspace "
    "itself, such as forge-fuse, turns this off to keep publishing while "
    "never replaying a stored patch over a tree it already staged.",
)
@click.option(
    "--producer",
    default="",
    help="System owning the candidate stream these records belong to (default: "
    "the forge-loop's own). A producer has its own index in the KB "
    "identity scheme, so a pipeline driving this command as a subprocess "
    "can keep its records out of the kernel campaigns' ranking.",
)
@click.option(
    "--return-after-read-kb",
    "--return-after-read-KB",
    "return_after_read_kb",
    is_flag=True,
    default=False,
    help="Return before Iteration 1 when a KB solution applies cleanly, passes "
    "current correctness, and improves current performance.",
)
@click.option(
    "--pr-kb/--no-pr-kb",
    default=None,
    help="Inject upstream pull-request references from the PR Monitor "
    "as Implementer prior knowledge (default off). Falls back to "
    "PR_KB_ENABLE when unset.",
)
@click.option(
    "--specialist-probe/--no-specialist-probe",
    default=None,
    help="Let the read-only planning specialists measure one variant per probe "
    "in a scratch tree, instead of only arguing about a dispatch constant "
    "(default on). Each probe re-runs the workspace driver for one case "
    "with declared constants overridden; it queues on the same device lock "
    "the fan-out lanes take, and never touches the canonical tree. Falls "
    "back to FORGE_SPECIALIST_PROBE when unset.",
)
@click.option(
    "--specialist-probe-max",
    default=None,
    type=click.IntRange(min=1),
    help="Probes ONE analysis round may make in total, shared by every "
    "specialist it dispatches (default 6). Every call counts, including "
    "one that is refused. Falls back to FORGE_SPECIALIST_PROBE_MAX when "
    "unset.",
)
@click.option(
    "--specialist-probe-budget-sec",
    default=None,
    type=click.FloatRange(min=1.0),
    help="Seconds on the device ONE analysis round may spend probing, shared by "
    "every specialist it dispatches (default 600). Cut down further at "
    "call time so no probe can leave a specialist without the time to "
    "write its analysis. Falls back to FORGE_SPECIALIST_PROBE_BUDGET_SEC "
    "when unset.",
)
@click.option(
    "--specialist-probe-scratch-root",
    default=None,
    help="Where the round scratch trees are created. Must be absolute and lie "
    "outside the workspace. Default: <experiments-dir>/specialist_probe, "
    "or a sibling of the workspace when that would land inside it. Falls "
    "back to FORGE_SPECIALIST_PROBE_SCRATCH_ROOT when unset.",
)
@click.option(
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
)
@click.option("--resume", is_flag=True, help="Resume the campaign stored in the exact workspace")
def forge_loop(
    kernel,
    driver,
    workspace_dir,
    snr_threshold,
    max_hours,
    session_timeout_sec,
    deadline_unix,
    git_branch,
    gpu_target,
    gpu_type,
    kernel_backend,
    program_md_file,
    invocation_spec_file,
    experiments_dir,
    aiter_cache_max_gb,
    experiment_id,
    experience_id,
    result_json,
    model,
    agent_backend,
    agent_cli,
    agent_timeout_sec,
    agent_reasoning_effort,
    agent_sandbox_mode,
    agent_fallback_provider,
    agent_precheck,
    agent_options_json,
    permission_mode,
    supervisor_backend,
    profile_timeout_sec,
    profiling,
    roofline_ceiling,
    roofline_target,
    prepare_task,
    task_type,
    source_files,
    target_functions,
    framework,
    operator_name,
    experience_kb,
    kb_warmstart_enabled,
    producer,
    return_after_read_kb,
    pr_kb,
    resume,
    nproc_per_node,
    bench_repeat,
    lanes,
    merge_stacking,
    specialist_probe,
    specialist_probe_max,
    specialist_probe_budget_sec,
    specialist_probe_scratch_root,
    commit_new_paths,
    baseline_json,
):
    """Run ONE Forge IterationLoop as a standalone subprocess (CLI-ized kernel backend)."""
    long_horizon = _is_long_horizon(max_hours)
    critic_enabled = bool(long_horizon)
    try:
        knowledge_config = KnowledgeConfig.from_env()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    import dataclasses as _dataclasses
    import hashlib as _hashlib

    _reject_incompatible_loop_options(
        return_after_read_kb=return_after_read_kb,
        kernel_backend=kernel_backend,
        experience_kb=experience_kb,
        kb_warmstart_enabled=kb_warmstart_enabled,
        producer=producer,
    )

    from kernelforge.loop.campaign_config import (
        CampaignConfigStore,
        derive_campaign_implementation_contract,
    )
    from kernelforge.loop.run_state import WorkspaceLock, WorkspaceLockError

    # Absolute deadline shared by task preparation and optimization.
    if deadline_unix <= 0:
        deadline_unix = time.time() + max_hours * 3600.0
    finalize_reserve_sec = max(
        30.0,
        float(os.environ.get("FORGE_FINALIZE_RESERVE_SEC", "120") or 120),
    )

    deadline = _ForgeDeadline(deadline_unix, finalize_reserve_sec)

    if profile_timeout_sec <= 0:
        raise click.ClickException("--profile-timeout-sec must be positive")

    workspace = Path(workspace_dir).resolve()
    workspace_lock = WorkspaceLock(workspace / "forge_experiments" / "workspace.lock")
    try:
        workspace_lock.acquire()
    except WorkspaceLockError as error:
        raise click.ClickException(str(error)) from error
    click.get_current_context().call_on_close(workspace_lock.release)

    from kernelforge.loop.campaign_setup import resolve_campaign
    from kernelforge.cli.relay import resolve_default_relay_config

    campaign_store = CampaignConfigStore(str(workspace))
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"
    has_run_artifacts = (
        state_path.exists()
        or (campaign_root / "events.jsonl").exists()
        or any((campaign_root / "candidates").glob("iter_*"))
    )
    _check_workspace_matches_resume(
        resume=resume,
        state_path=state_path,
        campaign_store=campaign_store,
        has_run_artifacts=has_run_artifacts,
    )
    agent_runtime_overrides = _agent_runtime_overrides(
        model=model,
        agent_backend=agent_backend,
        agent_cli=agent_cli,
        agent_timeout_sec=agent_timeout_sec,
        agent_reasoning_effort=agent_reasoning_effort,
        agent_sandbox_mode=agent_sandbox_mode,
        agent_fallback_provider=agent_fallback_provider,
        agent_precheck=agent_precheck,
        agent_options_json=agent_options_json,
    )
    bootstrap_agent_config = Config.from_env(
        workspace=str(workspace),
        knowledge_config=knowledge_config,
        **agent_runtime_overrides,
    )
    requested_relay_config = resolve_default_relay_config(
        bootstrap_agent_config,
        blockage_threshold=3,
    )

    try:
        resolution = resolve_campaign(
            str(workspace),
            resume=resume,
            prepare_task=prepare_task,
            kernel=kernel,
            driver=driver,
            source_files=source_files or "",
            program_md_file=program_md_file,
            target_functions=target_functions or "",
            operator_name=operator_name or "",
            producer=producer or "",
            kernel_backend=kernel_backend or "",
            git_branch=git_branch or "",
            gpu_target=gpu_target or "",
            gpu_type=gpu_type,
            task_type=task_type or "",
            framework=framework or "",
            snr_threshold=snr_threshold,
            nproc_per_node=nproc_per_node,
            bench_repeat=bench_repeat,
            commit_new_paths=list(commit_new_paths),
            agent_relay=requested_relay_config,
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    campaign = resolution.campaign
    program_text = resolution.program_text
    campaign_save_deferred = resolution.save_deferred

    # Resume tooling may override the complete Analysis workflow deadline via FORGE_PROFILE_TIMEOUT_SEC;
    # --profile-timeout-sec is the default when the env is unset.
    _env_timeout = os.environ.get("FORGE_PROFILE_TIMEOUT_SEC")
    profile_timeout_sec = _profile_timeout_override(_env_timeout, profile_timeout_sec)

    kernel = str((workspace / campaign.kernel_path).resolve())
    driver = str((workspace / campaign.driver_path).resolve())
    source_files_list = [str((workspace / path).resolve()) for path in campaign.source_files]
    target_functions_list = list(campaign.target_functions)
    snr_threshold = campaign.snr_threshold
    gpu_target = campaign.gpu_target
    gpu_type = campaign.gpu_type
    kernel_backend = campaign.kernel_backend
    task_type = campaign.task_type
    git_branch = campaign.git_branch
    framework = campaign.framework
    operator_name = campaign.operator_name
    producer = campaign.producer
    # Measurement semantics come from the campaign, never from this invocation's defaults.
    nproc_per_node = campaign.nproc_per_node
    bench_repeat = campaign.bench_repeat
    # From the campaign for the same reason: a resumed session that fell back to an empty allowlist could neither ship
    # nor remove the new file an earlier session was configured to.
    commit_new_paths = list(campaign.commit_new_paths)
    profiling_enabled = bool(profiling and long_horizon)

    overrides = {"gpu_target": gpu_target}
    overrides["gpu_type"] = gpu_type
    overrides["producer"] = producer
    # Only what was actually asked for: an override present with a None value still wins over ``Config.from_env``'s
    # environment lookup, which is what made the FORGE_SPECIALIST_PROBE* variables dead on this path.
    for _name, _value in (
        ("specialist_probe", specialist_probe),
        ("specialist_probe_max", specialist_probe_max),
        ("specialist_probe_budget_sec", specialist_probe_budget_sec),
        ("specialist_probe_scratch_root", specialist_probe_scratch_root),
    ):
        if _value is not None:
            overrides[_name] = _value
    overrides.update(agent_runtime_overrides)
    if gpu_target:
        os.environ["GPU_TARGET"] = gpu_target
    config = Config.from_env(
        workspace=str(workspace),
        knowledge_config=knowledge_config,
        **overrides,
    )
    startup_availability = None
    if campaign.agent_relay is not None:
        from kernelforge.cli.relay import probe_agent_profiles

        try:
            startup_availability = probe_agent_profiles(
                campaign.agent_relay,
                config=config,
                workspace=str(workspace),
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise click.ClickException(f"Agent startup availability check failed: {error}") from error
    # Two roots by design: resume artifacts (run_state/candidates/best) always live under
    # <workspace>/forge_experiments (campaign_root); diagnostics and the external-recovery checkpoint go to
    # --experiments-dir when the caller supplies a distinct one (e.g. Hyperloom's output dir), else campaign_root.
    config.experiments_dir = Path(experiments_dir).resolve() if experiments_dir else campaign_root
    config.experiments_dir.mkdir(parents=True, exist_ok=True)

    # AITER cache isolation: give this attempt its own runtime-build cache so parallel forge processes never
    # share/evict each other's kernels.
    from kernelforge.loop.aiter_cache import (
        activate_aiter_cache_for_sources,
        configure_aiter_cache_isolation,
        seed_prebuilt_modules,
    )

    aiter_cache = configure_aiter_cache_isolation(
        config.experiments_dir,
        max_cache_bytes=int(aiter_cache_max_gb * 1024**3),
    )
    print(f"  [aiter-cache] isolated runtime builds under {aiter_cache.cache_root}")
    baseline_cache = activate_aiter_cache_for_sources(source_files_list)

    # Seed the pristine BASELINE shard with the package's prebuilt .so so the task-preparation preflight imports warm
    # modules instead of cold-compiling the CK instance-factory TU (>26 min on gfx950, which blows the preflight
    # timeout and leaves the driver stuck as a placeholder).
    if baseline_cache is not None:
        seed_stats = seed_prebuilt_modules(baseline_cache.aiter_jit_dir)
        print(
            f"  [aiter-cache] seeded {seed_stats['seeded']} prebuilt module(s)"
            f" (skipped {seed_stats['skipped']}, errors {seed_stats['errors']})"
            f" from {seed_stats['src'] or 'n/a'}"
        )

    from kernelforge.loop.runner import IterationLoop
    from kernelforge.loop.runner import IterationConfig
    from kernelforge.loop.scoring import warm_start_improvement_flags
    from kernelforge.tracker import ExperimentTracker, UsageAccumulator, UsageLedgerFile

    iter_config = IterationConfig(
        kernel_file=kernel,
        driver_script=driver,
        canonical_driver_sha256=campaign.driver_sha256,
        campaign_base_commit=campaign.base_commit,
        snr_threshold=snr_threshold,
        max_time_hours=max(0.05, max_hours),
        deadline_unix=deadline_unix - finalize_reserve_sec,
        git_branch=git_branch,
        workspace_dir=workspace_dir,
        experiment_id=experiment_id or "",
        backend=(kernel_backend or "").split("-", 1)[0],
        kernel_backend=kernel_backend or "",
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
        lanes=lanes,
        merge_stacking=merge_stacking,
        # New files a KEEP may carry; a REVERT removes exactly the same set.
        commit_new_paths=commit_new_paths,
        # The attainment of the per-shape ceiling that ends the run; the ceiling itself is estimated inside the loop.
        roofline_target=_validate_roofline_target(roofline_target, roofline_ceiling == CEILING_ON),
    )
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

    # The caller-owned experiment ID is an EXTERNAL recovery channel, deliberately independent of the internal
    # per-segment experiment identity (each resume segment gets a fresh ID so the campaign parent/child chain stays
    # intact).
    caller_experiment_id = (experiment_id or "").strip()
    _ensure_recovery_experiment(tracker, caller_experiment_id, kernel=kernel)
    program_md = _program_context(
        campaign,
        campaign_store,
        campaign_save_deferred=campaign_save_deferred,
        program_text=program_text,
        resume=resume,
        experience_kb=experience_kb,
        workspace_dir=workspace_dir,
    )

    # Pre-loop task preparation (fresh campaigns only): ensure the driver conforms to the loop's stdout contract
    # BEFORE base_sha is captured, so any scaffolding the prep step commits becomes part of pristine (never the
    # solution diff).
    if prepare_task and not resume:
        _prepare_measurement_driver(
            config=config,
            workspace_dir=workspace_dir,
            kernel=kernel,
            driver=driver,
            program_md=program_md,
            program_md_file=program_md_file,
            target_functions_list=target_functions_list,
            source_files_list=source_files_list,
            kernel_backend=kernel_backend,
            snr_threshold=snr_threshold,
            invocation_spec_file=invocation_spec_file,
            deadline_unix=deadline_unix,
            finalize_reserve_sec=finalize_reserve_sec,
            nproc_per_node=nproc_per_node,
            usage=usage,
            result_json=result_json,
            require_time=deadline.require,
        )

    # Fix the fresh campaign's canonical driver digest and pristine base_commit from the POST-preparation state and
    # persist the immutable config now.
    if campaign_save_deferred:
        prepared_driver_sha256 = _hashlib.sha256(Path(driver).read_bytes()).hexdigest()
        prepared_base_commit = git_head(str(workspace)) or campaign.base_commit
        prepared_signature, prepared_identity = derive_campaign_implementation_contract(
            workspace_dir=str(workspace),
            kernel_path=campaign.kernel_path,
            source_files=campaign.source_files,
            framework=campaign.framework,
            base_commit=prepared_base_commit,
        )
        campaign = _dataclasses.replace(
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
            campaign_store.save(campaign, program_md=program_text)
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error

    assembly = _prepare_assembly_task(
        kernel_backend=kernel_backend,
        config=config,
        kernel=kernel,
        driver=driver,
        source_files_list=source_files_list,
        campaign=campaign,
        iter_config=iter_config,
        workspace=workspace,
        program_md=program_md,
        kb_warmstart_enabled=kb_warmstart_enabled,
        snr_threshold=snr_threshold,
        deadline_unix=deadline_unix,
        finalize_reserve_sec=finalize_reserve_sec,
        resume=resume,
    )
    assembly_preparation = assembly.preparation
    source_files_list = assembly.source_files
    kb_warmstart_enabled = assembly.kb_warmstart_enabled
    program_md = assembly.program_md

    # Resolved before the loop is built: the ceiling estimator captures it, and the raw option is ``None`` by default.
    session_timeout_sec = _forge_session_timeout_sec(max_hours, session_timeout_sec)

    # Construct the loop only after task preparation has resolved the profiling contract; IterationLoop snapshots that
    # readiness in its runtime state.
    loop_runner = IterationLoop(
        iter_config,
        tracker,
        config,
        resume=resume,
        ceiling_estimator=_make_ceiling_estimator(
            enabled=roofline_ceiling == CEILING_ON,
            workspace_dir=workspace_dir,
            driver_script=iter_config.driver_script,
            source_files=source_files_list,
            # The runtime the implementer lanes settle on is resolved further down, after this point, and the analyst
            # is a different role anyway: it reads the same provider/model ladder the standalone command reads, so
            # the raw selection is what it needs and an empty one means "take the default".
            agent_provider=agent_backend or "",
            agent_model=model or "",
            session_timeout_sec=session_timeout_sec,
            sandbox_mode=config.agent_sandbox_mode,
        ),
    )

    if resume:
        try:
            loop_runner.validate_resume_preflight()
        except (OSError, ValueError) as error:
            raise click.ClickException(str(error)) from error

    # Pristine HEAD — anchors the cumulative solution diff written back at the end.
    base_sha = campaign.base_commit

    warm_start = _read_kb_warm_start(
        iter_config,
        config=config,
        kernel=kernel,
        driver=driver,
        workspace_dir=workspace_dir,
        kernel_backend=kernel_backend,
        target_functions_list=target_functions_list,
        source_files_list=source_files_list,
        framework=framework,
        operator_name=operator_name,
        snr_threshold=snr_threshold,
        bench_repeat=bench_repeat,
        resume=resume,
        experience_kb=experience_kb,
        kb_warmstart_enabled=kb_warmstart_enabled,
        program_md=program_md,
        result_json=result_json,
        experience_id=experience_id,
        finalize_reserve_sec=finalize_reserve_sec,
        remaining=deadline.remaining,
        base_sha=base_sha,
        caller_experiment_id=caller_experiment_id,
        tracker=tracker,
    )
    warm = warm_start.warm
    program_md = warm_start.program_md
    warm_start_result = warm_start.warm_start_result
    kb_pristine_baseline_ms = warm_start.kb_pristine_baseline_ms
    kb_reused_speedup = warm_start.kb_reused_speedup

    if return_after_read_kb and warm.get("applied"):
        _warm_start_only_result(
            warm,
            warm_start_result,
            usage=usage,
            result_json=result_json,
            warm_start_improvement_flags=warm_start_improvement_flags,
        )
        return

    # Source mapping, profiling, profile analysis, and potential analysis are produced together by the commit-bound
    # Analysis Agent.
    iter_config.program_md = program_md

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
    pr_task_context, pr_kb_repo = _collect_pr_kb_context(
        iter_config,
        pr_kb=pr_kb,
        workspace=workspace,
        workspace_dir=workspace_dir,
        kernel_backend=kernel_backend,
        campaign=campaign,
        operator_name=operator_name,
        target_functions_list=target_functions_list,
        finalize_reserve_sec=finalize_reserve_sec,
        remaining=deadline.remaining,
    )

    from kernelforge.cli.relay import (
        SearchActorBuildContext,
        build_search_actor_pool,
        config_for_profile,
    )
    from kernelforge.loop.agent_relay import AgentProfile, AgentRelayConfig
    from kernelforge.orchestrator.orchestration import default_specialist_definitions

    specialist_definitions = default_specialist_definitions()
    selected_runtime = config.agent_runtime()
    relay_config = campaign.agent_relay
    if relay_config is None:
        primary_profile = AgentProfile(
            selected_runtime.provider,
            selected_runtime.provider,
            selected_runtime.model,
            selected_runtime.reasoning_effort,
        )
        relay_config = AgentRelayConfig(
            primary_profile_id=primary_profile.profile_id,
            profiles=(primary_profile,),
        )

    actor_builder = SearchActorBuildContext(
        config=config,
        program_md=program_md,
        pr_task_context=pr_task_context,
        pr_kb_repo=pr_kb_repo,
        kernel_backend=kernel_backend,
        driver=driver,
        snr_threshold=snr_threshold,
        max_blocks=max_blocks,
        session_timeout_sec=session_timeout_sec,
        validation_timeout_sec=iter_config.validate_stage_timeout_sec,
        bench_timeout_sec=iter_config.bench_timeout_sec,
        bench_repeat=bench_repeat,
        permission_mode=permission_mode,
        task_type=task_type,
        source_files=source_files_list,
        target_functions=target_functions_list,
        profiling_enabled=profiling_enabled,
        commit_new_paths=iter_config.commit_new_paths,
        usage=usage,
        lanes=lanes,
        workspace_dir=iter_config.workspace_dir,
        specialist_definitions=specialist_definitions,
        critic_enabled=critic_enabled,
        relay_enabled=campaign.agent_relay is not None,
        primary_provider=relay_config.primary_profile.provider,
        supervisor_backend=supervisor_backend,
    )
    try:
        search_actors = build_search_actor_pool(
            relay_config,
            bundle_factory=actor_builder.build,
            availability=startup_availability,
            fallback_provider=config.agent_fallback_provider,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(f"Agent startup availability check failed: {error}") from error
    start_bundle = search_actors.bundle(search_actors.start_profile_id)
    agent_fn = start_bundle.agent_fn
    _lane_agent_factory = start_bundle.agent_factory
    orchestration_service = start_bundle.orchestration_service
    supervisor_fn = start_bundle.supervisor_fn
    effective_implementer = start_bundle.profile.provider
    effective_implementer_model = start_bundle.profile.model
    for profile_id, status in search_actors.availability.profiles.items():
        state = "available" if status.available else f"unavailable ({status.reason})"
        print(f"  Agent profile: {profile_id} / {status.model}: {state}")
    if search_actors.start_profile_id != relay_config.primary_profile_id:
        print(
            f"  WARNING: primary Agent profile {relay_config.primary_profile_id} is unavailable; "
            f"starting on fallback profile {search_actors.start_profile_id}"
        )
    if campaign.agent_relay is not None and not relay_switch_possible(search_actors.availability):
        print(
            "  WARNING: model relay disabled for this session: only one Agent profile passed its startup check "
            "(reasons above); a search blockage falls back to the Supervisor"
        )

    from kernelforge.orchestrator.analysis import make_analysis_agent_service

    analysis_config = (
        config_for_profile(config, start_bundle.profile, primary_provider=relay_config.primary_profile.provider)
        if campaign.agent_relay is not None
        else config
    )
    analysis_service = make_analysis_agent_service(
        config=analysis_config,
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

    remote_publication = _initial_remote_publication_state(warm)

    reporter = _RunReporter(
        loop_runner=loop_runner,
        workspace_dir=workspace_dir,
        campaign_root=campaign_root,
        tracker=tracker,
        config=config,
        kernel=kernel,
        kernel_backend=kernel_backend,
        base_sha=base_sha,
        framework=framework,
        operator_name=operator_name,
        usage=usage,
        result_json=result_json,
        experience_kb=experience_kb,
        experience_id=experience_id,
        caller_experiment_id=caller_experiment_id,
        kb_pristine_baseline_ms=kb_pristine_baseline_ms,
        kb_reused_speedup=kb_reused_speedup,
        source_files_list=source_files_list,
        target_functions_list=target_functions_list,
        assembly_preparation=assembly_preparation,
        effective_implementer=effective_implementer,
        effective_implementer_model=effective_implementer_model,
        remote_publication=remote_publication,
    )
    _build_result = reporter.build_result
    _attempt_remote_publication = reporter.attempt_remote_publication
    _publish_remote_best = reporter.publish_remote_best
    _checkpoint_on_best_committed = reporter.checkpoint_on_best_committed

    asyncio.run(
        loop_runner.run(
            agent_fn=agent_fn,
            agent_factory=_lane_agent_factory if lanes > 1 else None,
            analysis_service=analysis_service,
            orchestration_service=orchestration_service,
            supervisor_fn=supervisor_fn,
            on_best_committed=_checkpoint_on_best_committed,
            on_best_ready=_publish_remote_best,
            usage=usage,
            workspace_lock_held=True,
            search_actors=(search_actors if campaign.agent_relay is not None else None),
        )
    )

    # Final graceful write: upgrade the (possibly interim) per-run solution page with the precise LLM-generated
    # summary.
    final_best_commit = str(getattr(loop_runner.run_state.best, "commit_hash", "") or "")
    kb_write = _attempt_remote_publication(
        commit=final_best_commit,
        llm_summary=deadline.remaining(reserve=30.0) >= 60.0,
    )
    if remote_publication["pending_commit"]:
        kb_write = _attempt_remote_publication(
            commit=final_best_commit,
            llm_summary=False,
        )
    loop_runner._checkpoint_llm_usage()
    kb_experience = {
        "read": kb_read_status(warm),
        "write": kb_write,
        "publication": _remote_publication_view(
            remote_publication,
            final_best_commit,
        ),
    }
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
    payload = json.dumps(result)
    if result_json:
        atomic_write_json(result_json, result)
    # Sentinel-wrapped so the caller can extract it from mixed loop stdout.
    click.echo(f"__FORGE_RESULT__{payload}__FORGE_RESULT__")
    _write_pr_provenance(
        workspace_dir=workspace_dir,
        surfaced=iter_config.pr_reference_labels,
        winning_iteration=int(result.get("best_iteration") or 0),
        experiment_id=str(result.get("experiment_id") or ""),
    )
