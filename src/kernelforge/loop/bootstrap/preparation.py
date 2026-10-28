# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Making one forge-loop task measurable: the driver contract, the declared invocation spec, and assembly binding."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from kernelforge.loop.bootstrap.errors import ForgeLoopError, TaskPreparationFailed
from kernelforge.tools.durable_io import atomic_write_json

if TYPE_CHECKING:
    from kernelforge.loop.bootstrap.run_context import _ForgeRun
    from kernelforge.loop.task_preparer import PrepareResult


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


def _exit_failed_preparation(run: _ForgeRun, prep: PrepareResult) -> NoReturn:
    """End the run with the reason preparation failed: on the console, in --result-json and in the published result."""
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
    result_json = run.request.artifacts.result_json
    if result_json:
        atomic_write_json(result_json, err_result)
    run.publish_result(err_result)
    raise TaskPreparationFailed(detail)


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
        raise ForgeLoopError(str(exc)) from exc
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
        _exit_failed_preparation(run, prep)
    _print_prepared_driver(prep)


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
        raise ForgeLoopError(f"assembly preparation failed; optimization was not started: {error}") from error
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
