# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Review GEAK source products and measure them inside the kernel task's lease."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_overlay_is_loadable,
    geak_spec_is_env,
)
from hyperloom.orchestrator.state.task_registry import Task

ORIGIN = "geak_native_revalidation"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay_source_files(overlay: str) -> dict[str, str]:
    """Use the native launch's source evidence for every deployable overlay file."""
    from hyperloom.orchestrator.actions.executors._native_source import source_file_hashes

    if not overlay:
        return {}
    return source_file_hashes(
        sorted(p for p in Path(overlay).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    )


def _patch_sources(result: dict[str, Any], *, overlay: str) -> list[Path]:
    combined = str(result.get("final_patch") or "").strip()
    kernels = [
        spec
        for spec in (result.get("accepted_kernels") or []) + (result.get("accepted_heads") or [])
        if isinstance(spec, dict) and not geak_spec_is_env(spec)
    ]
    if (
        not combined
        and not overlay
        and any(not (spec.get("final_patch") or spec.get("patch_path")) for spec in kernels)
    ):
        raise ValueError("GEAK accepted kernel has no deployable source patch or overlay")
    raw = (
        [combined]
        if combined
        else [str(spec.get("final_patch") or spec.get("patch_path") or "").strip() for spec in kernels]
    )
    root = Path(str(result.get("eval_dir") or "")).expanduser().resolve()
    paths = []
    for value in dict.fromkeys(item for item in raw if item):
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not path.read_bytes().strip():
            raise ValueError(f"GEAK source patch is missing, empty, or outside its eval directory: {path}")
        paths.append(path)
    return paths


def prepare_source_revalidation(
    session_dir: Path, result: dict[str, Any], params: dict[str, Any], *, overlay: str
) -> dict[str, Any] | None:
    """Stage declared source patches; refuse incomplete products instead of dropping a layer."""
    if overlay and not _geak_overlay_is_loadable(overlay):
        raise ValueError("GEAK source candidate declares an unloadable overlay")
    patches = _patch_sources(result, overlay=overlay)
    if not patches:
        if not overlay and not any(
            params.get(name) for name in ("extra_server_args", "extra_envs", "remove_args", "unset_envs", "args_mode")
        ):
            raise ValueError("GEAK accepted kernels have no deployable source patch or overlay")
        return None
    identity = {
        "patches": {str(path): _sha256(path) for path in patches},
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
        "overlay_pythonpath": overlay,
        "native_geak_artifact_digest": digest,
        "native_geak_files": {path: _sha256(Path(path)) for path in staged},
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


def record_review(coord: Any, pending: Any, *, verdict: str, reasoning: str) -> bool:
    """Persist only an actual Critic verdict for the covered GEAK review in flight."""
    params = (pending.payload or {}).get("params") or {}
    if params.get("origin") != ORIGIN:
        return False
    saved = dict((coord.shared_state.geak_pending or {}).get("native_review") or {})
    if saved.get("proposal_msg_id") != pending.proposal_msg_id or saved.get("artifact_digest") != params.get(
        "native_geak_artifact_digest"
    ):
        return True
    saved.update(verdict=verdict, reasoning=reasoning)
    coord.shared_state.geak_pending = {**coord.shared_state.geak_pending, "native_review": saved}
    if verdict in {"approve", "advise"}:
        coord.shared_state.record_specialist_patch_verdict(str(params["specialist_task_id"]), verdict)
    coord.shared_state.save(coord.session_dir)
    return True


def _verify_artifacts(params: dict[str, Any]) -> None:
    for filename, expected in params["native_geak_files"].items():
        path = Path(filename)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"reviewed GEAK source patch changed or disappeared: {path}")
    overlay = str(params.get("overlay_pythonpath") or "")
    if overlay and overlay_source_files(overlay) != params.get("native_geak_overlay_files"):
        raise ValueError("reviewed GEAK overlay changed or disappeared")


async def _review(coord: Any, params: dict[str, Any]) -> dict[str, Any]:
    from hyperloom.orchestrator.loop.proposals import PendingProposal

    digest = params["native_geak_artifact_digest"]
    saved = dict((coord.shared_state.geak_pending or {}).get("native_review") or {})
    if saved.get("artifact_digest") == digest and saved.get("verdict") in {"approve", "advise"}:
        return saved
    payload = {
        "action_name": "integrate",
        "predicted_gain_pct": 0.0,
        "params": {**params, "covered_executor": "integrate_patch"},
        "rationale": "Review GEAK source patches and their complete serving configuration before canonical native AgentX validation; GEAK proxy scores are not acceptance evidence.",
    }
    message = Message.new("coordinator", "*", "proposal", {**payload, "needs_review": True})
    saved = {
        "artifact_digest": digest,
        "proposal_msg_id": message.msg_id,
        "verdict": "pending",
        "task_id": uuid.uuid4().hex,
        "params": params,
        "candidate": dict(coord.shared_state.geak_result),
    }
    coord.shared_state.geak_pending = {
        **(coord.shared_state.geak_pending or {}),
        "status": "awaiting_source_review",
        "native_review": saved,
    }
    coord.shared_state.save(coord.session_dir)
    coord.state.pending_proposals[message.msg_id] = PendingProposal(
        message.msg_id, "coordinator", "integrate", 0.0, payload
    )
    await coord.bus.append_and_seq(message)
    remaining = coord.shared_state.remaining_minutes()
    deadline = time.monotonic() + min(300.0, remaining * 60.0 if remaining is not None else 300.0)
    while time.monotonic() < deadline:
        saved = dict((coord.shared_state.geak_pending or {}).get("native_review") or {})
        if saved.get("verdict") != "pending":
            return saved
        await asyncio.sleep(0.25)
    saved.update(verdict="timeout", reasoning="Critic review did not complete within the revalidation window")
    return saved


def _finish(coord: Any, review: dict[str, Any], *, status: str, error: str = "", measurement: Any = None) -> None:
    coord.shared_state.geak_result = {
        **coord.shared_state.geak_result,
        "revalidation_status": status,
        "revalidation_error": error,
        "native_source_revalidation": dict(review, status=status),
        "canonical_revalidation": measurement,
    }
    coord.shared_state.geak_pending = {}
    coord.shared_state.save(coord.session_dir)


def finish_invalid_source(coord: Any, error: str) -> None:
    """Settle an undeployable product without invoking GEAK's proxy harness."""
    _finish(coord, {}, status="fallback_failed", error=error)


def _record_canonical_adoption(coord: Any, params: dict[str, Any], measurement: dict[str, Any]) -> None:
    from hyperloom.orchestrator.state.shared_state import resolve_graded_comparison

    graded = resolve_graded_comparison(coord.shared_state, measurement, against_baseline=True)
    if graded.comparable:
        coord._record_geak_adopted_kernels(
            coord.shared_state.geak_result,
            measured_tput=graded.candidate,
            baseline_tput=graded.reference,
            provenance="native_source_canonical_revalidation",
            overlay_loaded=bool(params.get("overlay_pythonpath")),
            source_applied=True,
        )
    coord.shared_state.resume_pending_revalidation = False


async def revalidate_source(coord: Any, params: dict[str, Any]) -> None:
    """Use real review and the existing transactional integration and promotion paths."""
    try:
        _verify_artifacts(params)
        review = await _review(coord, params)
        if review.get("verdict") not in {"approve", "advise"}:
            _finish(
                coord,
                review,
                status="no_promote",
                error=str(review.get("reasoning") or "Critic declined source integration"),
            )
            return
        _verify_artifacts(params)
    except (OSError, ValueError) as exc:
        _finish(coord, {}, status="fallback_failed", error=str(exc))
        return
    task = Task(
        task_id=review["task_id"],
        kind="integrate_patch",
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
        _finish(coord, review, status="validated", measurement=coord.shared_state.current_best)
        return
    coord.shared_state.geak_pending = {
        **coord.shared_state.geak_pending,
        "status": "source_rebench_running",
        "revalidation_task_id": task.task_id,
    }
    coord.shared_state.save(coord.session_dir)
    result = await coord.sub.execute_covered(task)
    measurement = result.get("bench_result") or result
    stack_before = len(coord.shared_state.optimization_stack or [])
    if coord._is_promotable_result("integrate_patch", result):
        await coord._promote_to_shared_state("integrate_patch", result, task=task)
    else:
        await coord._handle_unpromotable_result(task, result)
    kept = result.get("status") == "kept" and len(coord.shared_state.optimization_stack or []) > stack_before
    if kept:
        _record_canonical_adoption(coord, params, measurement)
    _finish(
        coord,
        review,
        status="validated" if kept else "no_promote",
        error=str(result.get("reason") or ""),
        measurement=measurement,
    )
