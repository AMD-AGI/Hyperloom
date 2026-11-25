# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage GEAK source products and measure them inside the kernel task's lease."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.geak_rebench_verdict import (
    REBENCH_FAILED,
    REBENCH_FALLBACK_FAILED,
    REBENCH_NO_PROMOTE,
    REBENCH_UNMEASURED_GIVEN_UP,
    REBENCH_VALIDATED,
)

from hyperloom_common.io import atomic_write_json
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.common.overlay import overlay_is_loadable
from hyperloom.common.perf_metric import agentx_replay_session
from hyperloom.orchestrator.actions.executors.integrate._integrate_vocabulary import RESULT_GRADED
from hyperloom.orchestrator.kernel.geak_config import geak_spec_is_env
from hyperloom.orchestrator.state.task_registry import Task
from .geak_rebench import (
    MAX_UNMEASURED_REVALIDATIONS,
    REVALIDATION_UNMEASURED,
    source_digest,
    overlay_source_files,
    verify_source_artifacts,
)

_INTEGRATE_PATCH = "integrate_patch"


ORIGIN = "geak_native_revalidation"


def _source_bearing_specs(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The accepted kernel and head specs whose product is source, not an environment."""
    return [
        spec
        for spec in (result.get("accepted_kernels") or []) + (result.get("accepted_heads") or [])
        if isinstance(spec, dict) and not geak_spec_is_env(spec)
    ]


def _spec_declares_source(spec: dict[str, Any]) -> bool:
    """Whether a spec names a patch file at all."""
    return bool(spec.get("final_patch") or spec.get("patch_path"))


def _spec_patch_value(spec: dict[str, Any]) -> str:
    """The patch path a spec declares, as written."""
    return str(spec.get("final_patch") or spec.get("patch_path") or "").strip()


def _declared_patch_values(result: dict[str, Any], *, overlay: str) -> list[str]:
    """The patch paths a GEAK product declares; a combined patch stands for the whole product."""
    combined = str(result.get("final_patch") or "").strip()
    if combined:
        return [combined]
    specs = _source_bearing_specs(result)
    if not overlay and not all(_spec_declares_source(spec) for spec in specs):
        raise ValueError("GEAK accepted kernel has no deployable source patch or overlay")
    return [_spec_patch_value(spec) for spec in specs]


def _readable_patch_under(value: str, root: Path) -> Path:
    """Resolve a declared patch path to a non-empty file inside ``root``."""
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file() or not path.read_bytes().strip():
        raise ValueError(f"GEAK source patch is missing, empty, or outside its eval directory: {path}")
    return path


def _patch_sources(result: dict[str, Any], *, overlay: str) -> list[Path]:
    root = Path(str(result.get("eval_dir") or "")).expanduser().resolve()
    declared = _declared_patch_values(result, overlay=overlay)
    return [_readable_patch_under(value, root) for value in dict.fromkeys(item for item in declared if item)]


def prepare_source_revalidation(
    session_dir: Path, result: dict[str, Any], params: dict[str, Any], *, overlay: str
) -> dict[str, Any] | None:
    """Stage declared source patches; refuse incomplete products instead of dropping a layer."""
    if overlay and not overlay_is_loadable(overlay):
        raise ValueError("GEAK source candidate declares an unloadable overlay")
    patches = _patch_sources(result, overlay=overlay)
    if not patches:
        if not overlay and not any(
            params.get(name) for name in ("extra_server_args", "extra_envs", "remove_args", "unset_envs", "args_mode")
        ):
            raise ValueError("GEAK accepted kernels have no deployable source patch or overlay")
        return None
    identity: dict[str, Any] = {
        "patches": {str(path): source_digest(path) for path in patches},
        "launch": params,
        "overlay": overlay,
        "overlay_files": overlay_source_files(overlay),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    sid = f"geak-native-{digest[:24]}"
    workspace = runs_dir(session_dir, "specialist", sid)
    workspace.mkdir(parents=True, exist_ok=True)
    staged = []
    for index, path in enumerate(patches):
        destination = workspace / f"{index:03d}-{path.name}"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != identity["patches"][str(path)]:
            raise ValueError(f"GEAK source patch changed while staging: {path}")
        destination.write_bytes(content)
        staged.append(str(destination))
    params = {
        **params,
        "origin": ORIGIN,
        "specialist_task_id": sid,
        "patches": staged,
        "source_phase": "KERNEL_AGENT",
        "domain": "kernel",
        "provenance": ORIGIN,
        "require_accuracy_for_keep": True,
        "overlay_pythonpath": overlay,
        "native_geak_artifact_digest": digest,
        "native_geak_files": {path: source_digest(Path(path)) for path in staged},
        "native_geak_original_files": identity["patches"],
        "native_geak_overlay_files": identity["overlay_files"],
    }
    atomic_write_json(
        workspace / "specialist_done.json",
        {
            "patches_written": staged,
            "source_phase": "KERNEL_AGENT",
            "domain": "kernel",
            "proposal_set": [
                {
                    "name": "GEAK source candidate",
                    "extra_args": params.get("extra_server_args", ""),
                    "extra_envs": params.get("extra_envs", {}),
                }
            ],
        },
    )
    return params


def _revalidation_record(coord: Any, params: dict[str, Any]) -> dict[str, Any]:
    """The identity a source product is measured under; a product whose run was cut short keeps the one it had."""
    digest = params["native_geak_artifact_digest"]
    saved = dict((coord.shared_state.geak_pending or {}).get("native_review") or {})
    if saved.get("artifact_digest") == digest:
        return saved
    return {
        "artifact_digest": digest,
        "task_id": uuid.uuid4().hex,
        "params": params,
        "candidate": dict(coord.shared_state.geak_result),
    }


def _finish(coord: Any, record: dict[str, Any], *, status: str, error: str = "", measurement: Any = None) -> None:
    """Settle the candidate. A measurement that ended with no verdict is counted on the candidate, whatever ended
    it, a stop by the run included. At ``MAX_UNMEASURED_REVALIDATIONS`` of them the candidate is given up
    unmeasured: that is final, it is recorded on the kernel timeline, and the next KERNEL entry runs GEAK afresh."""
    settled = {
        **coord.shared_state.geak_result,
        "revalidation_status": status,
        "revalidation_error": error,
        "native_source_revalidation": dict(record, status=status),
        "canonical_revalidation": measurement,
    }
    if status == REBENCH_FAILED:
        settled[REVALIDATION_UNMEASURED] = settled.get(REVALIDATION_UNMEASURED, 0) + 1
        if settled[REVALIDATION_UNMEASURED] >= MAX_UNMEASURED_REVALIDATIONS:
            settled["revalidation_error_class"] = REBENCH_UNMEASURED_GIVEN_UP
            recorder = coord.phase_kernel.timeline()
            if recorder is not None:
                recorder.record_geak_rebench_conclusion(
                    final_status=status, final_error_class=REBENCH_UNMEASURED_GIVEN_UP, final_error=error
                )
    coord.shared_state.geak_result = settled
    coord.shared_state.geak_pending = {}
    coord.shared_state.save(coord.session_dir)


def finish_invalid_source(coord: Any, error: str) -> None:
    """Settle an undeployable product without invoking GEAK's proxy harness."""
    _finish(coord, {}, status=REBENCH_FALLBACK_FAILED, error=error)


def _record_canonical_adoption(coord: Any, params: dict[str, Any], measurement: dict[str, Any]) -> None:
    from hyperloom.orchestrator.state.shared_state import resolve_graded_comparison

    graded = resolve_graded_comparison(coord.shared_state, measurement, against_baseline=True)
    if graded.comparable:
        coord.phase_kernel.record_geak_adopted_kernels(
            coord.shared_state.geak_result,
            measured_tput=graded.candidate,
            baseline_tput=graded.reference,
            provenance="native_source_canonical_revalidation",
            overlay_loaded=bool(params.get("overlay_pythonpath")),
        )


def _unkept_status(result: dict[str, Any]) -> str:
    """The status of a product the integration did not keep: a verdict only where the executor says it graded one."""
    return REBENCH_NO_PROMOTE if result.get(RESULT_GRADED) else REBENCH_FAILED


async def revalidate_source(coord: Any, params: dict[str, Any]) -> None:
    """Measure a staged source product through the transactional integration and promotion paths.

    The measurement is the gate: a product the executor graded settles as validated or not promoted. One it did not
    grade settles as failed, which is no verdict: the next KERNEL entry measures it again if GEAK reported it as a
    win, and runs GEAK afresh otherwise, until :func:`_finish` gives the product up.
    """
    try:
        verify_source_artifacts(params)
    except (OSError, ValueError) as exc:
        _finish(coord, {}, status=REBENCH_FALLBACK_FAILED, error=str(exc))
        return
    record = _revalidation_record(coord, params)
    task = Task(
        task_id=record["task_id"],
        kind=_INTEGRATE_PATCH,
        state="running",
        params=params,
        idempotency_key=f"geak-source-{params['native_geak_artifact_digest']}",
    )
    if any(
        row.get("task_id") == task.task_id
        for row in (coord.shared_state.optimization_stack or [])
        if isinstance(row, dict)
    ):
        stack = coord.shared_state.optimization_stack
        if stack[-1].get("task_id") == task.task_id:
            _record_canonical_adoption(coord, params, coord.shared_state.current_best)
        _finish(coord, record, status=REBENCH_VALIDATED, measurement=coord.shared_state.current_best)
        return
    coord.shared_state.geak_pending = {
        **(coord.shared_state.geak_pending or {}),
        "status": "source_rebench_running",
        "native_review": record,
        "revalidation_task_id": task.task_id,
    }
    coord.shared_state.save(coord.session_dir)
    result = await coord.sub.execute_covered(task)
    measurement = result.get("bench_result") or result
    stack_before = len(coord.shared_state.optimization_stack or [])
    if coord.writeback.is_promotable_result(_INTEGRATE_PATCH, result):
        await coord.writeback.promote_to_shared_state(_INTEGRATE_PATCH, result, task=task)
    else:
        await coord.writeback.handle_unpromotable_result(task, result)
    kept = result.get("status") == "kept" and len(coord.shared_state.optimization_stack or []) > stack_before
    if kept:
        _record_canonical_adoption(coord, params, measurement)
    _finish(
        coord,
        record,
        status=REBENCH_VALIDATED if kept else _unkept_status(result),
        error=str(result.get("error") or result.get("reason") or ""),
        measurement=measurement,
    )


def source_rebench_params(
    coord: Any,
    *,
    reason: str,
    result: dict[str, Any],
    launch: dict[str, Any],
    overlay: str,
) -> dict[str, Any] | None:
    """Route native source products through the transactional patch executor."""
    from .geak_rebench import geak_candidate_matches
    from hyperloom.orchestrator.state.shared_state import stack_base_params

    state = coord.shared_state
    if not agentx_replay_session(state):
        return None
    pending = (state.geak_pending or {}).get("native_review") or {}
    if pending.get("params") and geak_candidate_matches(pending.get("candidate"), result):
        return dict(pending["params"])
    params = {
        "reason": reason,
        **launch,
        **stack_base_params({**(state.current_best or {}), **coord.writeback.current_best_launch_config()}),
        "config_path": state.baseline_config_path,
        "framework": state.framework,
        "model_path": state.model_path,
    }
    coord.proposals.inject_explore_runtime_params(params)
    try:
        return prepare_source_revalidation(coord.session_dir, result, params, overlay=overlay)
    except (OSError, ValueError) as exc:
        return {"skipped": True, "reason": "geak_native_source_invalid", "error": str(exc)}


def matching_review(state: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Retain a source product's revalidation identity only while GEAK still reports that product."""
    from .geak_rebench import geak_candidate_matches

    pending = state.geak_pending or {}
    review = pending.get("native_review") or {}
    if not review or not geak_candidate_matches(review.get("candidate"), result):
        return {}
    return {
        "native_review": review,
        "status": pending.get("status", "awaiting_rebench"),
        "revalidation_task_id": pending.get("revalidation_task_id", ""),
    }
