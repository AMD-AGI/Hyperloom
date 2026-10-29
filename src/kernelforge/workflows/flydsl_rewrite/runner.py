# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-rewrite orchestrator: ingest -> seed -> preflight -> PORT -> OPTIMIZE -> report."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from kernelforge.utils.git import git, git_checkout_branch, git_head
from kernelforge.config import Config
from kernelforge.knowledge.kb_store.errors import (
    kb_store_secrets,
    sanitize_read_error,
)
from kernelforge.workflows.flydsl_rewrite import (
    driver_contract,
    flydsl_rewrite_driver_preparation,
    ingest,
    report,
    seed,
)
from kernelforge.workflows.flydsl_rewrite.applyback import ApplybackRequest, generate_applyback_patch
from kernelforge.workflows.flydsl_rewrite.attempt import (
    create_attempt_workspace,
    export_import_path,
)
from kernelforge.workflows.flydsl_rewrite.kb import (
    RewriteKbReadResult,
    RewriteKbTarget,
    scored_speedup,
    try_flydsl_kb_warmstart,
    write_flydsl_kb_solution,
)
from kernelforge.workflows.flydsl_rewrite.measurement import FlydslMeasurement
from kernelforge.workflows.flydsl_rewrite.optimize import OptimizeRequest, run_optimize
from kernelforge.workflows.flydsl_rewrite.port_loop import PortRequest, PortResult, run_port_loop
from kernelforge.workflows.flydsl_rewrite.request import RewriteRun
from kernelforge.workflows.flydsl_rewrite.budget import DEFAULT_REWRITE_BUDGET, RewriteBudgetPolicy
from kernelforge.agent_backends.usage import UsageAccumulator, combine_usage_totals
from kernelforge.agent_backends.usage_ledger import UsageLedgerFile

log = logging.getLogger(__name__)

# Pipeline-owned failure classes; driver contract failures use the classes ``driver_contract`` defines.
SOURCE_KERNEL_MISSING = "source_kernel_missing"
ATTEMPT_SETUP_FAILED = "attempt_setup_failed"
CANDIDATE_NAME_INVALID = "candidate_name_invalid"
INGEST_FAILED = "ingest_failed"
DEADLINE_BEFORE_PORT = "deadline_before_port"
PORT_FAILED = "port_failed"

# An untimeable candidate only costs the interim best; one proving the two bench paths measure different work
# invalidates every number the rewrite reports.
_FATAL_CANDIDATE_FAILURES = frozenset(
    {
        driver_contract.CASE_COVERAGE_MISMATCH,
        driver_contract.CANDIDATE_NOT_ISOLATED,
        driver_contract.CANDIDATE_MODE_UNSUPPORTED,
    }
)


def _git(workspace: str, *args: str) -> subprocess.CompletedProcess:
    return git("-C", workspace, *args, check=False)


def _progress(message: str) -> None:
    print(f"  [forge-rewrite] {message}", flush=True)


class _SetupFailed(Exception):
    """A stage refused the run before it could be scored; ``args`` are the reason and its failure class."""


#: Interpreter caches appear beside any module the driver imports, at any depth
#: under the path being added. The exclusion is anchored under that path because
#: a leading-wildcard pathspec matches nothing here and would silently stage an
#: empty set, and ``**/`` alone requires an intervening directory, which misses
#: the cache sitting next to the entry point.
_PYCACHE_EXCLUDES = (":(exclude){path}/__pycache__/**", ":(exclude){path}/**/__pycache__/**")


def _ensure_git_committed(
    workspace: str,
    message: str,
    paths: list[str],
    *,
    branch: str = "",
) -> None:
    """Ensure ``workspace`` is a git repo and commit ONLY ``paths`` on ``branch``.

    A path may be a directory, which commits what it holds. That is what the
    port needs: the driver validates an implementation as it stands on disk, so
    committing one declared file out of it would select something no stage ever
    measured.
    """
    if not (Path(workspace) / ".git").exists():
        _git(workspace, "init")
        _git(workspace, "config", "user.email", "forge-rewrite@local")
        _git(workspace, "config", "user.name", "forge-rewrite")
    if branch:
        message_out = git_checkout_branch(workspace, branch)
        log.info("forge-rewrite: producer branch %s: %s", branch, message_out)
    staged_ok = False
    for p in paths:
        if not p:
            continue
        # Force-add: the candidate lives under a dot-directory a caller's ignore rules may exclude, and forge-loop's
        # keep/revert silently no-ops on an untracked kernel. Interpreter caches are the one thing a path may hold
        # that no consumer wants; excluding them keeps a directory path usable here.
        excludes = [item.format(path=p.rstrip("/")) for item in _PYCACHE_EXCLUDES]
        r = _git(workspace, "add", "-f", "--", p, *excludes)
        if r.returncode != 0:
            log.warning("forge-rewrite: git add failed for %s: %s", p, (r.stderr or r.stdout).strip())
            continue
        staged_ok = True
    if not staged_ok:
        return
    # A non-zero commit here is the benign "nothing to commit" (idempotent re-run / already-committed unchanged file),
    # so we do NOT gate on its exit code — it conflates "nothing changed" (fine) with "add staged nothing" (broken)
    # into the same non-zero.
    _git(workspace, "commit", "-m", message)
    for p in paths:
        if p and _git(workspace, "ls-files", "--error-unmatch", "--", p).returncode != 0:
            log.warning(
                "forge-rewrite: %s is NOT git-tracked after commit "
                "(ignored / staging failed?); forge-loop keep/revert will no-op on it",
                p,
            )
            _progress(f"WARNING: {Path(p).name} is not git-tracked; forge-loop keep/revert may be a no-op")


def _run_usage(usage, optimize_result: dict | None = None, *, optimize_ran: bool = False) -> dict:
    """This run's cumulative spend: the in-process stages plus the nested forge-loop's own ledger.

    A forge-loop that was cut off or killed reports a ledger that stops at its
    last checkpoint, so the totals are published as partial rather than as a
    complete provider-priced answer.
    """
    opt_result = optimize_result or {}
    nested = opt_result.get("llm_usage")
    return combine_usage_totals(
        usage.totals(),
        nested if isinstance(nested, dict) else None,
        incomplete=optimize_ran and not opt_result.get("llm_usage_complete"),
    )


def _emit_final(result, result_json: str | None) -> dict:
    """Publish the run's result where every consumer reads it: the file and the sentinel line."""
    payload = report.emit_result(result, result_json)
    print(f"{report.SENTINEL}{payload}{report.SENTINEL}", flush=True)
    return result.to_dict()


def _plan_budget(
    *,
    started_at: float,
    deadline_unix: float | None,
    optimize_max_hours: float,
    applyback_enabled: bool,
) -> tuple[float, RewriteBudgetPolicy, float]:
    """The run's deadline, budget policy and search cutoff, announced as it is decided.

    A run that will not produce a patch spends nothing holding a reserve for
    one, so the policy drops the apply-back reserve when the caller declined it.
    """
    if not deadline_unix or deadline_unix <= 0:
        deadline_unix = started_at + optimize_max_hours * 3600.0
    budget = DEFAULT_REWRITE_BUDGET if applyback_enabled else DEFAULT_REWRITE_BUDGET.without_applyback()
    search_stop_unix = budget.search_stop_unix(deadline_unix)
    _progress(
        f"budget: remaining={max(0, int(deadline_unix - started_at))}s "
        f"search={max(0, int(search_stop_unix - started_at))}s applyback_reserve={budget.applyback_reserve_sec}s"
    )
    return deadline_unix, budget, search_stop_unix


def _with_port_fallbacks(opt: dict, *, best_ms, speedup, port_commit: str) -> dict:
    """The OPTIMIZE result, falling back to the ported kernel's own figures where it reported none."""
    filled = dict(opt)
    if filled.get("best_ms") is None:
        filled["best_ms"] = best_ms
    if filled.get("mean_case_speedup") is None:
        filled["mean_case_speedup"] = speedup
    if not filled.get("best_commit"):
        filled["best_commit"] = port_commit
    return filled


def _conforming_driver(
    spec,
    config: Config,
    preparation: flydsl_rewrite_driver_preparation.DriverPreparation,
    *,
    usage,
    prepare_driver: bool,
):
    """The driver's validated rewrite contract; raises _SetupFailed when it has none.

    A driver that does not already conform is handed to rewrite driver
    preparation first, unless the caller declined that.
    """
    driver_path = preparation.driver_path
    preflight = flydsl_rewrite_driver_preparation.preflight_rewrite_driver(
        spec,
        driver_path,
        deadline_unix=preparation.deadline_unix,
    )
    if not preflight.ok and prepare_driver:
        _progress(f"driver does not conform [{preflight.failure_class}]; invoking rewrite driver preparation")
        prepared = asyncio.run(
            flydsl_rewrite_driver_preparation.prepare_rewrite_driver(
                spec=spec,
                config=config,
                preparation=preparation,
                initial_preflight=preflight,
                usage=usage,
            )
        )
        if not prepared.ok or prepared.preflight is None:
            raise _SetupFailed(
                prepared.error or "rewrite driver preparation failed",
                prepared.failure_class or flydsl_rewrite_driver_preparation.DRIVER_PREPARATION_FAILED,
            )
        preflight = prepared.preflight
        _progress(f"prepared driver {Path(driver_path).name} in {prepared.attempts} attempt(s)")
    elif preflight.ok:
        _progress(f"supplied driver {Path(driver_path).name} already conforms; preparation skipped")
    if not preflight.ok:
        raise _SetupFailed(preflight.detail, preflight.failure_class)
    for warning in preflight.warnings:
        _progress(f"driver contract warning: {warning}")
    if preflight.source_ms is None:
        raise _SetupFailed(
            "the conforming rewrite driver reported no source baseline", driver_contract.REF_TIMING_UNPARSEABLE
        )
    return preflight


@dataclass(frozen=True)
class _RewriteKb:
    """This run's rewrite KB traffic: what it reads before PORT and publishes after it, unless the caller declined."""

    target: RewriteKbTarget
    enabled: bool

    def read(self, *, source_case_ms, search_stop_unix: float) -> RewriteKbReadResult:
        """What the KB can offer for this source+driver, restoring the seed when the read fails.

        The warm start builds a KB Store client from the store URL and bearer token,
        and the guard here catches whatever its own reader did not: the client is
        constructed outside that sanitizer's ``try``, so a construction failure of
        any type other than ``KBStoreError`` arrives here untouched.
        """
        if not self.enabled:
            return RewriteKbReadResult(read_reason="disabled")
        spec = self.target.spec
        kb_seed = Path(spec.flydsl_kernel).read_bytes() if Path(spec.flydsl_kernel).is_file() else None
        try:
            return asyncio.run(
                try_flydsl_kb_warmstart(
                    self.target,
                    source_case_ms=source_case_ms,
                    stop_at_unix=search_stop_unix,
                )
            )
        except Exception as error:  # noqa: BLE001 - isolation boundary: the run's rewrite KB read; it cold-starts
            if kb_seed is None:
                Path(spec.flydsl_kernel).unlink(missing_ok=True)
            else:
                Path(spec.flydsl_kernel).write_bytes(kb_seed)
            read_error = sanitize_read_error(error, secrets=kb_store_secrets(self.target.config))
            # Sanitized reason and stack frames only: the raw message may carry the KB store credentials.
            log.warning(
                "rewrite KB read failed, cold start: %s (%s)\n%s",
                read_error,
                type(error).__name__,
                "".join(traceback.format_tb(error.__traceback__)),
            )
            return RewriteKbReadResult(read_reason="read_error", read_error=read_error)

    def publish_port(self, port, measurement: FlydslMeasurement, port_commit: str) -> dict:
        """Record the ported kernel in the KB.

        A newly produced correct port is independently reusable even when it is
        slower than the source; a port that was warm-started from the KB is already
        there.
        """
        if not self.enabled:
            return {"written": False, "reason": "disabled"}
        if port.attempts == 0:
            return {"written": False, "reason": "kb_warmstart_reused"}
        written = write_flydsl_kb_solution(self.target, measurement, session_key=port_commit)
        _progress(f"PORT KB publish: {written.get('reason') or written.get('solution')}")
        return written

    def publish_final(self, opt: dict, source_ms, port, port_commit: str, optimize_session_key: str) -> dict:
        """Record the run's final best in the KB, replacing the record of the session that produced it."""
        if not self.enabled:
            return {"written": False, "reason": "disabled"}
        final_commit = str(opt.get("best_commit") or "")
        # PORT's reading measures the artifact being recorded only while the run's best is still the ported kernel.
        # Once OPTIMIZE has moved the best off that commit, it describes a kernel this record is not about.
        snr_db = port.snr_db if final_commit == port_commit else None
        return write_flydsl_kb_solution(
            self.target,
            FlydslMeasurement(source_ms, opt.get("best_ms"), opt.get("mean_case_speedup"), snr_db),
            # The run's final result belongs to the OPTIMIZE session when there was one, and to the PORT session
            # otherwise -- either way it replaces that session's record instead of standing beside it.
            session_key=optimize_session_key or port_commit,
        )


def _measure_flydsl_baseline(spec, driver_path: str, preflight, search_stop_unix: float, snr_db) -> FlydslMeasurement:
    """Time the ported FlyDSL kernel against the source kernel.

    An unavailable bench is not fatal -- the port is still correct, just
    untimed -- but a failure class the contract calls fatal ends the run.
    """
    untimed = FlydslMeasurement(preflight.source_ms, None, None, snr_db)
    if time.time() >= search_stop_unix:
        return untimed
    flydsl_budget = max(1, min(600, int(search_stop_unix - time.time())))
    candidate = driver_contract.preflight_candidate(
        spec,
        driver_path,
        reference_case_ids=preflight.reference_case_ids,
        timeout_sec=flydsl_budget,
    )
    for warning in candidate.warnings:
        _progress(f"driver contract warning: {warning}")
    if candidate.ok:
        speedup = scored_speedup(candidate.case_ms, preflight.source_case_ms)
        return FlydslMeasurement(preflight.source_ms, candidate.timing_ms, speedup, snr_db)
    if candidate.failure_class in _FATAL_CANDIDATE_FAILURES:
        raise _SetupFailed(candidate.detail, candidate.failure_class)
    _progress(f"candidate bench unavailable [{candidate.failure_class}]: {candidate.detail}")
    return untimed


class _Rewrite:
    """One rewrite run: what was asked, its spend, budget and base commit, and the scope its results are filed under."""

    def __init__(self, request: RewriteRun, config: Config):
        self.request = request
        self.config = config
        self.workspace = request.workspace.workspace_dir
        self.applyback_enabled = request.applyback.applyback
        # The nested forge-loop runs from the workspace and resolves this path against its own working directory, so
        # a relative one would send the two processes to different directories for the artifacts they share.
        self.experiments_dir = str(Path(request.workspace.experiments_dir).resolve())
        Path(self.experiments_dir).mkdir(parents=True, exist_ok=True)
        # Only this process's own stages feed the shared ledger: the nested forge-loop publishes its own share to the
        # same file, so folding the combined total in here would count that share twice.
        usage_ledger = UsageLedgerFile(self.experiments_dir)
        self.usage = UsageAccumulator(on_update=usage_ledger.publish)
        usage_ledger.publish(self.usage.totals())
        self.deadline_unix, budget, self.search_stop_unix = _plan_budget(
            started_at=time.time(),
            deadline_unix=request.budget.deadline_unix,
            optimize_max_hours=request.budget.max_hours,
            applyback_enabled=self.applyback_enabled,
        )
        # The framework patch must be based on the pristine caller-owned repository, before the standalone FlyDSL
        # seed/PORT commits are introduced.
        self.base_commit = git_head(self.workspace)
        # A resolvable HEAD is what the agent sessions need, not evidence that the caller wants a framework patch.
        # Asking is what makes one required: a consumer of the standalone kernel has no framework tree for a patch to
        # target, and its run must not be judged on a stage it declined.
        self.applyback_required = self.applyback_enabled and bool(self.base_commit)
        if not self.applyback_enabled:
            _progress("apply-back not requested; the standalone kernel is the whole deliverable")
        # Producer-owned scratch the consumer may reclaim.
        self.scope = report.ResultScope(request.subject.op_name, budget_policy=budget)
        self.optimize_session_key = ""

    def prepare(self) -> None:
        """Set the run up for PORT: its attempt directory, the seeded spec, a conforming driver and the KB's offer."""
        request = self.request
        subject = request.subject
        # A fresh directory each run stops a rerun inheriting a previous kernel; on the import path so drivers still
        # reach the candidate by module name.
        try:
            self.attempt = create_attempt_workspace(self.workspace)
            export_import_path(self.attempt)
        except OSError as error:
            raise _SetupFailed(f"could not create the attempt directory: {error}", ATTEMPT_SETUP_FAILED)
        self.scope = report.ResultScope(subject.op_name, self.attempt.temporary_paths, self.scope.budget_policy)
        _progress(f"attempt workspace {self.attempt.relative_root}")
        if not Path(subject.source_kernel).is_file():
            raise _SetupFailed(f"source kernel not found: {subject.source_kernel}", SOURCE_KERNEL_MISSING)
        self.driver_path = str(Path(request.driver.driver).resolve())
        if time.time() >= self.search_stop_unix:
            raise _SetupFailed("less than 20 minutes remain; no PORT session may start", DEADLINE_BEFORE_PORT)
        try:
            candidate_kernel = self.attempt.candidate_path(request.flydsl_kernel_name)
        except ValueError as error:
            raise _SetupFailed(str(error), CANDIDATE_NAME_INVALID)
        # Ingest auto-discovers the source_entry hint if omitted.
        try:
            spec = ingest.build_spec(subject, flydsl_kernel=str(candidate_kernel), workspace=self.workspace)
        except Exception as e:  # noqa: BLE001 - any ingest error must still be scorable
            log.warning("rewrite ingest failed: %s", e)
            raise _SetupFailed(f"ingest error: {type(e).__name__}: {e}", INGEST_FAILED)
        driver_contract.export_driver_environment(spec)
        _progress(
            f"op={spec.op_name} src={spec.source_kernel_name} entry={spec.source_entry or '<none>'} "
            f"driver={Path(self.driver_path).name} -> {spec.flydsl_kernel_relpath}"
        )
        seed.generate_seed(spec, spec.flydsl_kernel)
        _progress(f"seeded skeleton {spec.flydsl_kernel_relpath}")
        preparation = flydsl_rewrite_driver_preparation.DriverPreparation(
            self.driver_path, self.experiments_dir, self.search_stop_unix, request.driver.invocation_spec_file
        )
        preflight = _conforming_driver(
            spec, self.config, preparation, usage=self.usage, prepare_driver=request.driver.prepare_driver
        )
        _progress(
            f"source baseline: {preflight.source_ms:.4f} ms (full suite, cases={list(preflight.reference_case_ids)})"
        )
        _progress("driver contract OK: source timed, candidate mode recognized and not yet runnable")
        self.spec = spec
        self.preflight = preflight
        # An exact source+driver match may materialize a prior standalone FlyDSL file, but it must pass today's
        # FlyDSL-only and correctness gates before PORT is skipped.
        self.rewrite_kb = _RewriteKb(RewriteKbTarget(spec, self.driver_path, self.config), request.rewrite_kb_enabled)
        self.kb_read = self.rewrite_kb.read(
            source_case_ms=preflight.source_case_ms, search_stop_unix=self.search_stop_unix
        )

    def port(self) -> PortResult:
        """A correct FlyDSL port: the KB's warm start when it passed today's gates, otherwise a PORT campaign."""
        kb_read = self.kb_read
        if kb_read.applied:
            _progress(f"KB warm-start accepted: {kb_read.solution_slug} ({kb_read.best_ms} ms)")
            return PortResult(ok=True, attempts=0, snr_db=kb_read.snr_db)
        port_request = PortRequest(
            self.request.max_port_attempts,
            self.request.permission_mode,
            self.search_stop_unix,
            kb_read.reference_context,
        )
        return asyncio.run(run_port_loop(self.spec, self.driver_path, self.config, port_request, usage=self.usage))

    def result(
        self, port: PortResult, kb_write: dict, optimize_result: dict, *, optimize_ran: bool = False, **outcome
    ) -> report.RewriteResult:
        """The run's result once PORT has run: the port and source figures, its spend and the KB traffic so far."""
        return report.build_result(
            self.scope,
            report.RewriteOutcome(
                port_ok=port.ok,
                port_attempts=port.attempts,
                source_ms=self.preflight.source_ms,
                optimize_result=optimize_result,
                llm_usage=_run_usage(self.usage, optimize_result, optimize_ran=optimize_ran),
                kb_experience={"read": self.kb_read.to_dict(), "write": kb_write},
                **outcome,
            ),
        )

    def publish_setup_failure(self, failed: _SetupFailed) -> dict:
        """Publish a clean, scorable failure result instead of a traceback, so the caller can attribute it."""
        reason, failure_class = failed.args
        _progress(f"SETUP FAILED [{failure_class}]: {reason}")
        outcome = report.RewriteOutcome(
            port_ok=False,
            port_attempts=0,
            source_ms=None,
            optimize_result={},
            llm_usage=_run_usage(self.usage),
            failure_class=failure_class,
            failure_detail=reason,
        )
        return _emit_final(report.build_result(self.scope, outcome), self.request.result_json)

    def run_applyback(self, opt: dict, port: PortResult):
        """Turn the standalone best, now restored in the rewrite workspace, into a patch against the framework tree."""
        measurement = FlydslMeasurement(
            self.preflight.source_ms, opt.get("best_ms"), opt.get("mean_case_speedup"), port.snr_db
        )
        applyback_request = ApplybackRequest(
            self.base_commit,
            self.experiments_dir,
            str(opt.get("best_commit") or ""),
            measurement,
            self.deadline_unix,
            self.request.applyback_import_modules,
            self.request.applyback.max_applyback_attempts,
        )
        applyback = generate_applyback_patch(self.spec, self.config, applyback_request, usage=self.usage)
        if applyback.ok:
            _progress(f"apply-back patch ready: {applyback.patch_path}")
        else:
            _progress(f"APPLY-BACK FAILED: {applyback.error}")
        return applyback

    def publish_keep(self, payload: dict) -> None:
        """Publish an OPTIMIZE KEEP to the rewrite KB as the session's record, replacing the KEEP before it.

        Every KEEP is published, not just the run's final best, because an OPTIMIZE session can be terminated at its
        cutoff or killed outright. forge-loop cannot do this itself -- it runs here under --no-experience-kb because
        the rewrite identity is not its own -- so the rewrite layer watches its result file and publishes to its own
        store. Naming the record after the forge-loop session rather than the artifact makes each publication
        replace the last.
        """
        commit = str(payload.get("best_commit") or "")
        relpath = self.spec.flydsl_kernel_relpath
        # Read the kernel out of the commit, never off disk: the workspace still belongs to the running agent, and the
        # best is only restored there once OPTIMIZE is over.
        shown = _git(self.workspace, "show", f"{commit}:{relpath}")
        if shown.returncode != 0 or not shown.stdout.strip():
            _progress(f"KEEP publish skipped: {commit[:12]} has no {relpath}")
            return
        self.optimize_session_key = hashlib.sha256(str(payload.get("experiment_id") or commit).encode()).hexdigest()
        write = write_flydsl_kb_solution(
            self.rewrite_kb.target,
            # PORT's SNR belongs to the ported kernel, not to the KEEP that has since been optimized out of it, and
            # forge-loop's result file does not carry the accuracy it measured for this one. Unmeasured, so unclaimed.
            FlydslMeasurement(
                self.preflight.source_ms, payload.get("best_ms"), payload.get("mean_case_speedup"), snr_db=None
            ),
            session_key=self.optimize_session_key,
            content_override=shown.stdout.encode(),
        )
        _progress(f"KEEP KB publish ({commit[:12]}): {write.get('reason') or write.get('solution')}")

    def finish(self, port: PortResult, port_commit: str, measurement: FlydslMeasurement) -> dict:
        """Publish the correct port as the interim best, OPTIMIZE it, then publish the final best with its patch."""
        spec = self.spec
        source_ms = measurement.source_ms
        # The result JSON is written now, reflecting a SUCCESSFUL port (compiled + correct) with the ported kernel's
        # own time as the interim best.
        interim = self.result(
            port,
            self.rewrite_kb.publish_port(port, measurement, port_commit),
            {"best_ms": measurement.flydsl_ms, "mean_case_speedup": measurement.speedup},
            applyback_result={"ok": False, "error": "apply-back pending"} if self.applyback_enabled else None,
            applyback_required=self.applyback_required,
        )
        report.emit_result(interim, self.request.result_json)
        sp0 = interim.speedup
        _progress(
            f"interim (port only): flydsl={measurement.flydsl_ms} ms vs source={source_ms} ms -> "
            f"speedup={f'{sp0:.3f}x' if sp0 else 'unknown'} (persisted; OPTIMIZE will improve)"
        )
        opt: dict = {}
        optimize_ran = time.time() < self.search_stop_unix
        if optimize_ran:
            opt = run_optimize(
                spec,
                self.driver_path,
                self.config,
                OptimizeRequest(
                    self.experiments_dir,
                    self.request.optimize,
                    roofline_ceiling=self.request.roofline_ceiling,
                    permission_mode=self.request.permission_mode,
                    max_hours=max(1.0, (self.deadline_unix - time.time()) / 3600.0),
                    deadline_unix=self.deadline_unix,
                    stop_at_unix=self.search_stop_unix,
                    # Anchor the loop on the source, so every score it reports -- each KEEP published below and the
                    # run's final result -- already divides by the kernel this rewrite replaced.
                    source_ms=source_ms,
                    source_case_ms=self.preflight.source_case_ms,
                    on_new_best=self.publish_keep if self.request.rewrite_kb_enabled else None,
                ),
            )
        else:
            _progress("20-minute finalization reserve reached after PORT; skipping forge-loop")
        if optimize_ran and not opt.get("llm_usage_complete"):
            _progress(
                "WARNING: forge-loop did not report a final token ledger; the reported llm_usage covers "
                "only what it checkpointed and is published as partial"
            )
        opt = _with_port_fallbacks(
            opt, best_ms=measurement.flydsl_ms, speedup=measurement.speedup, port_commit=port_commit
        )
        kb_write = self.rewrite_kb.publish_final(opt, source_ms, port, port_commit, self.optimize_session_key)
        applyback = self.run_applyback(opt, port) if self.applyback_enabled else None
        result = self.result(
            port,
            kb_write,
            opt,
            optimize_ran=optimize_ran,
            applyback_result=applyback.to_dict() if applyback is not None else None,
            applyback_required=self.applyback_required,
        )
        sp = result.speedup
        _progress(
            f"DONE: flydsl_best={result.flydsl_best_ms} ms vs source={source_ms} ms -> "
            f"speedup={f'{sp:.3f}x' if sp else 'unknown'}"
        )
        return _emit_final(result, self.request.result_json)


def run_rewrite(request: RewriteRun, config: Config) -> dict:
    """Run the full rewrite pipeline; return (and sentinel-print) the result dict."""
    run = _Rewrite(request, config)
    try:
        run.prepare()
        port = run.port()
        if not port.ok:
            _progress(f"PORT FAILED after {port.attempts} attempts")
            port_failed = {"written": False, "reason": "port_failed"}
            result = run.result(port, port_failed, {}, failure_class=PORT_FAILED, failure_detail=port.error_tail)
            return _emit_final(result, request.result_json)
        _progress(f"PORT OK (attempt {port.attempts}, SNR={port.snr_db})")
        # Commit the correct port so forge-loop starts from a clean committed state.
        # The whole attempt directory, not the kernel alone: a port free to structure
        # its implementation may put part of it in a module beside the entry point,
        # and the driver validated all of it. Committing only the entry point would
        # select a candidate that was never measured, and leave the rest behind for
        # a consumer that reads the commit. The directory is the producer's own, so
        # this adds nothing the caller protects.
        _ensure_git_committed(
            run.workspace,
            "forge-rewrite: initial correct flydsl port",
            [run.attempt.relative_root],
            branch=request.optimize.git_branch,
        )
        port_commit = git_head(run.workspace)
        measurement = _measure_flydsl_baseline(
            run.spec, run.driver_path, run.preflight, run.search_stop_unix, port.snr_db
        )
    except _SetupFailed as failed:
        return run.publish_setup_failure(failed)
    return run.finish(port, port_commit, measurement)
