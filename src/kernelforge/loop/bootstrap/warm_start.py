# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Experience KB warm start one forge-loop run adopts before iteration 1, and its recoverable publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kernelforge.loop.bootstrap.errors import ForgeLoopError
from kernelforge.loop.knowledge_integration import (
    WarmStartHarness,
    WarmStartRollbackError,
    kb_reference_program_md,
    kb_warmstart,
    mark_kb_reference_rejected,
)
from kernelforge.loop.recovery import CallerRecovery, publish_warm_start_recovery, rollback_unpublished_warm_start

if TYPE_CHECKING:
    from kernelforge.loop.bootstrap.run_context import _ForgeRun


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
            raise ForgeLoopError(
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
            raise ForgeLoopError(
                f"warm-start rollback failed; workspace may be inconsistent: {error}", exit_code=2
            ) from error
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
