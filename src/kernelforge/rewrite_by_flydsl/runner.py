# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-rewrite orchestrator: ingest -> seed -> preflight -> PORT -> OPTIMIZE -> report.

This is the "another layer" that turns a source-language task into a FlyDSL task
and reuses forge-loop to optimize it. It owns only the rewrite-specific stages;
the optimization is delegated to forge-loop unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.config import Config
from kernelforge.knowledge.experience_integration import git_checkout_branch
from kernelforge.knowledge.experience_reader import sanitize_read_error
from kernelforge.rewrite_by_flydsl import (
    driver_contract,
    flydsl_rewrite_driver_preparation,
    ingest,
    report,
    seed,
)
from kernelforge.rewrite_by_flydsl.agent_kb import kb_store_secrets
from kernelforge.rewrite_by_flydsl.applyback import generate_applyback_patch
from kernelforge.rewrite_by_flydsl.attempt import (
    create_attempt_workspace,
    export_import_path,
)
from kernelforge.rewrite_by_flydsl.kb import (
    RewriteKbReadResult,
    try_flydsl_kb_warmstart,
    write_flydsl_kb_solution,
)
from kernelforge.rewrite_by_flydsl.optimize import run_optimize
from kernelforge.rewrite_by_flydsl.port_loop import PortResult, run_port_loop
from kernelforge.rewrite_by_flydsl.budget import DEFAULT_REWRITE_BUDGET
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

log = logging.getLogger(__name__)

# Pipeline-owned failure classes; driver contract failures use the classes
# ``driver_contract`` defines.
SOURCE_KERNEL_MISSING = "source_kernel_missing"
ATTEMPT_SETUP_FAILED = "attempt_setup_failed"
CANDIDATE_NAME_INVALID = "candidate_name_invalid"
INGEST_FAILED = "ingest_failed"
DEADLINE_BEFORE_PORT = "deadline_before_port"
PORT_FAILED = "port_failed"

# An untimeable candidate only costs the interim best; one proving the two bench
# paths measure different work invalidates every number the rewrite reports.
_FATAL_CANDIDATE_FAILURES = frozenset(
    {
        driver_contract.CASE_COVERAGE_MISMATCH,
        driver_contract.CANDIDATE_NOT_ISOLATED,
        driver_contract.CANDIDATE_MODE_UNSUPPORTED,
    }
)


def _git(workspace: str, *args: str) -> subprocess.CompletedProcess:
    return git("-C", workspace, *args, check=False)


def _ensure_git_committed(
    workspace: str,
    message: str,
    paths: list[str],
    *,
    branch: str = "",
) -> None:
    """Ensure ``workspace`` is a git repo and commit ONLY ``paths`` on ``branch``.

    forge-loop requires a git repo for its keep/revert pattern and benches the
    committed working tree, so the FlyDSL port kernel must be committed before
    OPTIMIZE. The commit lands on the producer's own branch — the one the nested
    loop then develops on — so the branch the caller handed us keeps the history
    it started with. We stage ONLY the rewrite-owned files, never ``git add -A``:
    the workspace may hold unrelated uncommitted changes, experiment outputs, or
    generated scaffolding, and sweeping those into a port commit would pollute
    the caller's history. Idempotent: inits and sets a local identity only when
    needed.
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
        # Force-add: the candidate lives under a dot-directory a caller's
        # ignore rules may exclude, and forge-loop's keep/revert silently
        # no-ops on an untracked kernel.
        r = _git(workspace, "add", "-f", "--", p)
        if r.returncode != 0:
            log.warning("forge-rewrite: git add failed for %s: %s", p, (r.stderr or r.stdout).strip())
            continue
        staged_ok = True
    if not staged_ok:
        return
    # A non-zero commit here is the benign "nothing to commit" (idempotent re-run /
    # already-committed unchanged file), so we do NOT gate on its exit code — it
    # conflates "nothing changed" (fine) with "add staged nothing" (broken) into the
    # same non-zero. What forge-loop actually needs is the INVARIANT that each path
    # is TRACKED afterwards (its `git add -u` keep/revert silently no-ops on an
    # untracked kernel). Verify that directly and warn loudly if it does not hold.
    _git(workspace, "commit", "-m", message)
    for p in paths:
        if p and _git(workspace, "ls-files", "--error-unmatch", "--", p).returncode != 0:
            log.warning(
                "forge-rewrite: %s is NOT git-tracked after commit "
                "(ignored / staging failed?); forge-loop keep/revert will no-op on it",
                p,
            )
            print(
                f"  [forge-rewrite] WARNING: {Path(p).name} is not git-tracked; forge-loop keep/revert may be a no-op",
                flush=True,
            )


def run_rewrite(
    *,
    op_name: str,
    source_kernel: str,
    driver: str,
    workspace: str,
    experiments_dir: str,
    target_functions: list[str],
    config: Config,
    source_entry: str = "",
    source_language: str = "",
    shapes: list[dict] | None = None,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    flydsl_kernel_name: str = "kernel.py",
    max_port_attempts: int = 3,
    optimize_max_hours: float = 1.0,
    permission_mode: str | None = None,
    supervisor_backend: str = "codex",
    profile_timeout_sec: int = 1800,
    optimize_git_branch: str = "forge-rewrite-optimize",
    result_json: str | None = None,
    deadline_unix: float | None = None,
    framework: str = "",
    prepare_driver: bool = True,
    invocation_spec_file: str = "",
    applyback_import_modules: list[str] | tuple[str, ...] = (),
    max_applyback_attempts: int = 2,
    rewrite_kb_enabled: bool = True,
) -> dict:
    """Run the full rewrite pipeline; return (and sentinel-print) the result dict."""
    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    if not deadline_unix or deadline_unix <= 0:
        deadline_unix = started_at + optimize_max_hours * 3600.0
    rewrite_budget = DEFAULT_REWRITE_BUDGET
    search_stop_unix = rewrite_budget.search_stop_unix(deadline_unix)
    print(
        "  [forge-rewrite] budget: "
        f"remaining={max(0, int(deadline_unix - started_at))}s "
        f"search={max(0, int(search_stop_unix - started_at))}s "
        f"applyback_reserve={rewrite_budget.applyback_reserve_sec}s",
        flush=True,
    )

    # The framework patch must be based on the pristine caller-owned repository,
    # before the standalone FlyDSL seed/PORT commits are introduced.
    base_result = _git(workspace, "rev-parse", "HEAD")
    rewrite_base_commit = (
        base_result.stdout.strip().splitlines()[0] if base_result.returncode == 0 and base_result.stdout.strip() else ""
    )

    # Producer-owned scratch the consumer may reclaim. Always reported, empty
    # until this run creates something.
    temporary_paths: list[str] = []

    # Emit a clean, scorable failure result (no traceback) on any setup error so
    # the caller can attribute it, instead of the process dying opaquely.
    def _setup_failed(reason: str, failure_class: str) -> dict:
        print(f"  [forge-rewrite] SETUP FAILED [{failure_class}]: {reason}", flush=True)
        result = report.build_result(
            op_name=op_name,
            port_ok=False,
            port_attempts=0,
            source_ms=None,
            optimize_result={},
            failure_class=failure_class,
            failure_detail=reason,
            temporary_paths=temporary_paths,
        )
        payload = report.emit_result(result, result_json)
        print(f"{report.SENTINEL}{payload}{report.SENTINEL}", flush=True)
        return result.to_dict()

    # A fresh directory each run stops a rerun inheriting a previous kernel; on
    # the import path so drivers still reach the candidate by module name.
    try:
        attempt = create_attempt_workspace(workspace)
        export_import_path(attempt)
    except OSError as error:
        return _setup_failed(f"could not create the attempt directory: {error}", ATTEMPT_SETUP_FAILED)
    temporary_paths = attempt.temporary_paths
    print(f"  [forge-rewrite] attempt workspace {attempt.relative_root}", flush=True)

    # (0) The source kernel to port FROM must exist. The driver path may not exist
    # yet when rewrite-specific preparation is enabled; it becomes the destination
    # for the isolated driver-authoring stage below.
    if not Path(source_kernel).is_file():
        return _setup_failed(f"source kernel not found: {source_kernel}", SOURCE_KERNEL_MISSING)
    driver_path = str(Path(driver).resolve())
    if time.time() >= search_stop_unix:
        return _setup_failed(
            "less than 20 minutes remain; no PORT session may start",
            DEADLINE_BEFORE_PORT,
        )

    # (1) Ingest -> normalized spec (auto-discovers the source_entry hint if omitted).
    try:
        candidate_kernel = attempt.candidate_path(flydsl_kernel_name)
    except ValueError as error:
        return _setup_failed(str(error), CANDIDATE_NAME_INVALID)
    try:
        spec = ingest.build_spec(
            op_name=op_name,
            source_kernel=source_kernel,
            flydsl_kernel=str(candidate_kernel),
            workspace=workspace,
            target_functions=target_functions,
            source_entry=source_entry,
            source_language=source_language,
            shapes=shapes,
            snr_threshold=snr_threshold,
        )
    except Exception as e:  # noqa: BLE001 - any ingest error must still be scorable
        return _setup_failed(f"ingest error: {type(e).__name__}: {e}", INGEST_FAILED)
    driver_contract.export_driver_environment(spec)
    print(
        f"  [forge-rewrite] op={spec.op_name} src={spec.source_kernel_name} "
        f"entry={spec.source_entry or '<none>'} driver={Path(driver_path).name} "
        f"-> {spec.flydsl_kernel_relpath}",
        flush=True,
    )

    # (2) Seed the FlyDSL skeleton. The attempt directory is new, so this is
    # always a fresh stub — which is what the candidate probe below relies on.
    seed.generate_seed(spec, spec.flydsl_kernel)
    print(f"  [forge-rewrite] seeded skeleton {spec.flydsl_kernel_relpath}", flush=True)

    # (3) Validate the rewrite-specific dual-path contract. A conforming supplied
    # driver stays untouched. A missing or invalid driver is authored in an
    # isolated workspace by the rewrite preparer; forge-loop's single-path
    # task_preparer is deliberately not involved.
    preflight = flydsl_rewrite_driver_preparation.preflight_rewrite_driver(
        spec,
        driver_path,
        deadline_unix=search_stop_unix,
    )
    if not preflight.ok and prepare_driver:
        print(
            f"  [forge-rewrite] driver does not conform "
            f"[{preflight.failure_class}]; invoking rewrite driver preparation",
            flush=True,
        )
        prepared = asyncio.run(
            flydsl_rewrite_driver_preparation.prepare_rewrite_driver(
                spec=spec,
                driver_path=driver_path,
                config=config,
                experiments_dir=experiments_dir,
                deadline_unix=search_stop_unix,
                invocation_spec_file=invocation_spec_file,
                initial_preflight=preflight,
            )
        )
        if not prepared.ok or prepared.preflight is None:
            return _setup_failed(
                prepared.error or "rewrite driver preparation failed",
                prepared.failure_class or flydsl_rewrite_driver_preparation.DRIVER_PREPARATION_FAILED,
            )
        preflight = prepared.preflight
        print(
            f"  [forge-rewrite] prepared driver {Path(driver_path).name} in {prepared.attempts} attempt(s)",
            flush=True,
        )
    elif preflight.ok:
        print(
            f"  [forge-rewrite] supplied driver {Path(driver_path).name} already conforms; preparation skipped",
            flush=True,
        )
    if not preflight.ok:
        return _setup_failed(preflight.detail, preflight.failure_class)

    for warning in preflight.warnings:
        print(f"  [forge-rewrite] driver contract warning: {warning}", flush=True)
    source_ms = preflight.source_ms
    if source_ms is None:
        return _setup_failed(
            "the conforming rewrite driver reported no source baseline",
            driver_contract.REF_TIMING_UNPARSEABLE,
        )
    print(
        f"  [forge-rewrite] source baseline: {source_ms:.4f} ms (full suite, "
        f"cases={list(preflight.reference_case_ids) or 'unreported'})",
        flush=True,
    )
    print(
        "  [forge-rewrite] driver contract OK: source timed, candidate mode recognized and not yet runnable", flush=True
    )

    # (5) KB warm-start / PORT: an exact source+driver match may materialize a
    # prior standalone FlyDSL file, but it must pass today's FlyDSL-only and
    # correctness gates before PORT is skipped. Performance is measured for
    # ranking/reporting and does not prevent reuse of a correct port.
    kb_seed = Path(spec.flydsl_kernel).read_bytes() if Path(spec.flydsl_kernel).is_file() else None
    if rewrite_kb_enabled:
        try:
            kb_read = asyncio.run(
                try_flydsl_kb_warmstart(
                    spec,
                    driver_path,
                    config,
                    source_ms=source_ms,
                    framework=framework,
                    stop_at_unix=search_stop_unix,
                )
            )
        except Exception as error:  # noqa: BLE001 - KB failure must cold-start
            if kb_seed is None:
                Path(spec.flydsl_kernel).unlink(missing_ok=True)
            else:
                Path(spec.flydsl_kernel).write_bytes(kb_seed)
            # The warm start builds a KB Store client from the store URL and
            # bearer token, and this guard catches whatever its own reader did
            # not: the client is constructed outside that sanitizer's ``try``, so
            # a construction failure of any type other than ``KBStoreError``
            # arrives here untouched. This reason is persisted as
            # ``kb_experience.read.read_error``, so it is redacted and bounded
            # here too. The exception type leads the message, so the cap can only
            # cut the tail of a long error body.
            kb_read = RewriteKbReadResult(
                read_reason="read_error",
                read_error=sanitize_read_error(
                    error,
                    secrets=kb_store_secrets(config),
                ),
            )
    else:
        kb_read = RewriteKbReadResult(read_reason="disabled")
    if kb_read.applied:
        port = PortResult(
            ok=True,
            attempts=0,
            snr_db=kb_read.snr_db,
        )
        print(
            f"  [forge-rewrite] KB warm-start accepted: {kb_read.solution_slug} ({kb_read.best_ms} ms)",
            flush=True,
        )
    else:
        port = asyncio.run(
            run_port_loop(
                spec,
                driver_path,
                config,
                max_attempts=max_port_attempts,
                permission_mode=permission_mode,
                stop_at_unix=search_stop_unix,
                pre_task_context=kb_read.reference_context,
            )
        )
    if not port.ok:
        print(f"  [forge-rewrite] PORT FAILED after {port.attempts} attempts", flush=True)
        result = report.build_result(
            op_name=op_name,
            port_ok=False,
            port_attempts=port.attempts,
            source_ms=source_ms,
            optimize_result={},
            kb_experience={
                "read": kb_read.to_dict(),
                "write": {"written": False, "reason": "port_failed"},
            },
            failure_class=PORT_FAILED,
            failure_detail=port.error_tail,
            temporary_paths=temporary_paths,
        )
        payload = report.emit_result(result, result_json)
        print(f"{report.SENTINEL}{payload}{report.SENTINEL}", flush=True)
        return result.to_dict()
    print(f"  [forge-rewrite] PORT OK (attempt {port.attempts}, SNR={port.snr_db})", flush=True)

    # Commit the correct port so forge-loop starts from a clean committed state.
    # Stage ONLY the ported kernel — never the whole workspace (see helper).
    _ensure_git_committed(
        workspace,
        "forge-rewrite: initial correct flydsl port",
        [spec.flydsl_kernel],
        branch=optimize_git_branch,
    )
    port_commit_result = _git(workspace, "rev-parse", "HEAD")
    port_commit = (
        port_commit_result.stdout.strip().splitlines()[0]
        if port_commit_result.returncode == 0 and port_commit_result.stdout.strip()
        else ""
    )

    # (5b) Interim result: measure the ported FlyDSL kernel and write the result
    # JSON NOW, reflecting a SUCCESSFUL port (compiled + correct) with the ported
    # kernel's own time as the interim best. This way a successful port's outcome
    # (and its baseline speedup vs the source) survives even if the OPTIMIZE phase
    # below is cut short (e.g. an outer hard timeout kills the process before the
    # final report). OPTIMIZE only ever IMPROVES on this.
    # The same run completes the driver contract: the candidate must now be
    # timeable over the cases the source was timed on.
    flydsl_baseline_ms = None
    if time.time() < search_stop_unix:
        flydsl_budget = max(1, min(600, int(search_stop_unix - time.time())))
        candidate = driver_contract.preflight_candidate(
            spec,
            driver_path,
            reference_case_ids=preflight.reference_case_ids,
            timeout_sec=flydsl_budget,
        )
        for warning in candidate.warnings:
            print(f"  [forge-rewrite] driver contract warning: {warning}", flush=True)
        if candidate.ok:
            flydsl_baseline_ms = candidate.timing_ms
        elif candidate.failure_class in _FATAL_CANDIDATE_FAILURES:
            return _setup_failed(candidate.detail, candidate.failure_class)
        else:
            print(
                f"  [forge-rewrite] candidate bench unavailable [{candidate.failure_class}]: {candidate.detail}",
                flush=True,
            )
    # A newly produced correct port is independently reusable even when it is
    # slower than the source. Publish it immediately through the rewrite-owned
    # KB path so an OPTIMIZE timeout cannot force the next run to repeat PORT.
    if rewrite_kb_enabled and port.attempts > 0:
        port_kb_write = write_flydsl_kb_solution(
            spec,
            driver_path,
            config,
            source_ms=source_ms,
            flydsl_best_ms=flydsl_baseline_ms,
            best_commit=port_commit,
            framework=framework,
            snr_db=port.snr_db,
            allow_non_improving=True,
        )
        print(
            f"  [forge-rewrite] PORT KB publish: {port_kb_write.get('reason') or port_kb_write.get('solution')}",
            flush=True,
        )
    elif rewrite_kb_enabled:
        port_kb_write = {
            "written": False,
            "reason": "kb_warmstart_reused",
        }
    else:
        port_kb_write = {"written": False, "reason": "disabled"}
    interim = report.build_result(
        op_name=op_name,
        port_ok=True,
        port_attempts=port.attempts,
        source_ms=source_ms,
        optimize_result={"best_ms": flydsl_baseline_ms},
        applyback_result={"ok": False, "error": "apply-back pending"},
        applyback_required=bool(rewrite_base_commit),
        kb_experience={
            "read": kb_read.to_dict(),
            "write": port_kb_write,
        },
        temporary_paths=temporary_paths,
    )
    if result_json:
        report.emit_result(interim, result_json)
    sp0 = interim.speedup
    print(
        f"  [forge-rewrite] interim (port only): flydsl={flydsl_baseline_ms} ms "
        f"vs source={source_ms} ms -> speedup={f'{sp0:.3f}x' if sp0 else 'unknown'} "
        f"(persisted; OPTIMIZE will improve)",
        flush=True,
    )

    # (6) OPTIMIZE: reuse forge-loop over the FlyDSL kernel (unchanged).
    opt: dict = {}
    if time.time() < search_stop_unix:
        remaining_hours = max(1.0, (deadline_unix - time.time()) / 3600.0)
        opt = run_optimize(
            spec,
            driver_path,
            config,
            experiments_dir=experiments_dir,
            max_hours=remaining_hours,
            git_branch=optimize_git_branch,
            permission_mode=permission_mode,
            supervisor_backend=supervisor_backend,
            profile_timeout_sec=profile_timeout_sec,
            deadline_unix=deadline_unix,
            stop_at_unix=search_stop_unix,
        )
    else:
        print(
            "  [forge-rewrite] 20-minute finalization reserve reached after PORT; skipping forge-loop",
            flush=True,
        )

    # (7) Report: FlyDSL best vs source baseline. If OPTIMIZE returned no best
    # (e.g. it produced no improving iteration, or its result was unparseable),
    # fall back to the ported-kernel baseline so the final result never regresses
    # below the interim port-only result.
    if opt.get("best_ms") is None:
        opt = {**opt, "best_ms": flydsl_baseline_ms}
    if not opt.get("best_commit"):
        opt = {**opt, "best_commit": port_commit}

    if rewrite_kb_enabled:
        kb_write = write_flydsl_kb_solution(
            spec,
            driver_path,
            config,
            source_ms=source_ms,
            flydsl_best_ms=opt.get("best_ms"),
            best_commit=str(opt.get("best_commit") or ""),
            framework=framework,
            snr_db=port.snr_db,
            allow_non_improving=port.attempts > 0,
        )
    else:
        kb_write = {"written": False, "reason": "disabled"}

    # The standalone best is now restored in the rewrite workspace. Run exactly
    # one repository-level agent session in a pristine temporary worktree and
    # publish its git-apply-compatible integration patch.
    applyback = generate_applyback_patch(
        spec,
        config,
        base_commit=rewrite_base_commit,
        experiments_dir=experiments_dir,
        framework=framework,
        best_commit=str(opt.get("best_commit") or ""),
        source_ms=source_ms,
        flydsl_best_ms=opt.get("best_ms"),
        reference_snr_db=port.snr_db,
        deadline_unix=deadline_unix,
        import_modules=applyback_import_modules,
        max_attempts=max_applyback_attempts,
    )
    if applyback.ok:
        print(
            f"  [forge-rewrite] apply-back patch ready: {applyback.patch_path}",
            flush=True,
        )
    else:
        print(
            f"  [forge-rewrite] APPLY-BACK FAILED: {applyback.error}",
            flush=True,
        )
    result = report.build_result(
        op_name=op_name,
        port_ok=True,
        port_attempts=port.attempts,
        source_ms=source_ms,
        optimize_result=opt,
        applyback_result=applyback.to_dict(),
        applyback_required=bool(rewrite_base_commit),
        kb_experience={
            "read": kb_read.to_dict(),
            "write": kb_write,
        },
        temporary_paths=temporary_paths,
    )
    payload = report.emit_result(result, result_json)
    sp = result.speedup
    print(
        f"  [forge-rewrite] DONE: flydsl_best={result.flydsl_best_ms} ms "
        f"vs source={source_ms} ms -> speedup={f'{sp:.3f}x' if sp else 'unknown'}",
        flush=True,
    )
    print(f"{report.SENTINEL}{payload}{report.SENTINEL}", flush=True)
    return result.to_dict()
