# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator-side programmatic handlers for kernel REQUEST kinds.

Handler signature::

    async def handler(payload: dict, *, session_dir: Path) -> dict:

Dispatch table is exposed via :data:`KERNEL_REQUEST_HANDLERS` for test monkey-patching.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Mapping, NamedTuple

from hyperloom.common import codex_session, llm_config
from hyperloom.common.checkpoint_quant import checkpoint_format
from hyperloom.common.coerce import to_str_list
from hyperloom.common.env import env_bool, forge_explicitly_enabled, is_truthy
from hyperloom.common.gpu_identity import is_gfx_arch
from hyperloom.common.io import JsonLineFormat, append_jsonl
from hyperloom.common.unified_diff import PatchOp, repo_relative_manifest
from hyperloom.orchestrator.trace_analysis._capture_shapes import (
    is_capture_fragment as _shared_is_capture_fragment,
)
from ..actions.executors._kernel_agent_tool import (
    HandlerResult,
    _maybe_apply_kernel_patch,
    _maybe_finalize_kernel_patch,
    _maybe_revert_kernel_patch,
    _run_subprocess,
    _shape_tool_result,
)
from ..actions.executors.trace_analyze import trace_analyze_handler
from ..actions.stop_attribution import stopped_by_the_run_class
from . import apply_kernel_patch, forge_fusion, forge_gemm_tuning
from .gemm_shape_coverage import canonical_dtype, traced_gemm_shapes
from .lane_budget import (
    INSUFFICIENT_BUDGET,
    LANE_FUSION,
    LANE_GEMM,
    LaneAllocation,
    allocate as _allocate_lane_budgets,
    gemm_per_tuner_timeout_sec,
)
from .patch_landing import bundle_belongs_to
from .patch_lifecycle import cleanup_verdict as _cleanup_verdict

from ._kernel_decisions import _entry_by_kernel_id, _honest_flag


log = logging.getLogger(__name__)

STACK_INCREMENTAL_KEEP_THRESHOLD_PCT = 0.5
KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT = 1.0
# A patch whose correctness was only established against a reference kernel;
# serving accuracy is what settles it.
_FRAMEWORK_APPLYBACK_ARTIFACT_KIND = "framework_applyback"
_INTEGRATE_ACCURACY_VALIDATION_TIER = "integrate_e2e_accuracy"


def _vram_guarded_server_args(extra_args: str) -> str:
    """Optionally cap ``--gpu-memory-utilization`` for the integrate re-baseline.

    When ``HL_INTEGRATE_VRAM_GUARD`` is on and the caller has not already pinned
    ``--gpu-memory-utilization``, append a conservative cap
    (``HL_INTEGRATE_VRAM_UTIL_CAP``, default 0.90) so a re-baseline server cannot
    OOM. A strict no-op when the flag is off or a util is already specified.
    """
    if not _honest_flag("HL_INTEGRATE_VRAM_GUARD"):
        return extra_args
    # ``--gpu-memory-utilization`` is vLLM-only; apply the cap only for vLLM.
    framework = (os.environ.get("FRAMEWORK") or "").strip().lower()
    if framework != "vllm":
        return extra_args
    if "gpu-memory-utilization" in (extra_args or ""):
        return extra_args
    try:
        cap = float(os.environ.get("HL_INTEGRATE_VRAM_UTIL_CAP", "0.90") or 0.90)
    except (TypeError, ValueError):
        cap = 0.90
    cap = min(max(cap, 0.1), 0.99)
    addition = f"--gpu-memory-utilization {cap:g}"
    return f"{extra_args} {addition}".strip() if extra_args else addition


def _confirm_source_imported(source_file: str, workspace: str | Path | None) -> bool | None:
    """Best-effort confirm the patched source was actually imported/compiled.

    Greps the re-baseline server log for evidence the patched module's basename
    was imported/loaded/compiled, so a measured E2E delta is attributed to code
    the workload really ran. Returns a tri-state:

    * ``True``  — the module basename appears in import/load/compile context.
    * ``False`` — the server log is readable and the basename never appears
      anywhere (positive evidence the patched file was not exercised).
    * ``None``  — unknown (no source_file, no readable log) — never penalized.
    """
    if not source_file or not workspace:
        return None
    ws = Path(workspace)
    logs = [p for p in (ws / "server.log", ws.parent / "server.log") if p.exists()]
    if not logs:
        try:
            logs = sorted(ws.rglob("server.log"))[:1]
        except OSError:
            logs = []
    if not logs:
        return None
    stem = Path(source_file).stem
    if not stem:
        return None
    try:
        text = logs[0].read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if stem not in text:
        return False
    # Confirmed only when the basename co-occurs with an import/compile cue.
    for line in text.splitlines():
        if stem in line and re.search(r"import|load|compil|build|\.py", line, re.IGNORECASE):
            return True
    # Present but not in an obvious import context.
    return None


def _confirm_sources_imported(
    source_files: list[str],
    workspace: str | Path | None,
) -> tuple[bool | None, dict[str, bool | None]]:
    """Confirm every file a patch wrote was exercised by the served process.

    A patch can span several files -- a new module, the dispatcher that routes
    to it, the original source it replaces -- so each is graded on its own with
    :func:`_confirm_source_imported` and the verdicts are combined:

    * ``True``  — every file shows import evidence.
    * ``False`` — no file appears in the log at all, which is the unambiguous
      "the served process never ran any of this" case.
    * ``None``  — anything mixed. A module can be imported lazily or folded
      into another, so partial evidence is recorded for audit rather than held
      against the patch.
    """
    ordered = list(dict.fromkeys(path for path in source_files if str(path or "").strip()))
    if not ordered:
        return None, {}
    per_file = {path: _confirm_source_imported(path, workspace) for path in ordered}
    verdicts = list(per_file.values())
    if all(verdict is True for verdict in verdicts):
        return True, per_file
    if all(verdict is False for verdict in verdicts):
        return False, per_file
    return None, per_file


HandlerFn = Callable[..., Awaitable[HandlerResult]]

_RUNTIME_GENERATED_SOURCE_MARKERS = (  # nosec B108 - marker strings, not filesystem writes.
    "/tmp/torchinductor",
    "/torchinductor_",
    "/.cache/torch/inductor",
    "/.triton/cache",
    "/triton/cache",
)
_COMPILE_GENERATED_NAME_MARKERS = (
    "triton_poi_",
    "triton_red_",
    "triton_tem_",
    "torchinductor",
    "inductor",
)


# forge is the only per-kernel backend. The default phase-level backend is the
# whole-pipeline GEAK delegate (``geak``); per-kernel selection is opt-in via
# KERNEL_OPT_BACKEND_ORDER=forge.
_DEFAULT_KERNEL_PHASE_BACKEND_ORDER = ("geak",)
# Outer subprocess cap for the whole GEMM-tuning run (all shapes/tuners) when there is
# no lane share to take, as in an unbounded session; sized for large models with many
# GEMM shapes. Override via HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC (or payload timeout_sec).
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 5 * 60 * 60


def _preapplied_snapshot_payload(payload: dict) -> dict:
    """Capture an already-applied worktree as the apply's final-content snapshot.

    A controller publication is git-applied before the validator runs, so the
    patch's final bytes are on disk already. Handing them over as a snapshot
    keeps the diff a manifest of changed paths only, which is what lets the
    normal apply run its backup, invalidation, fan-out and rebuild.



    Raises:
        RuntimeError: If the repo root is unknown or a written path is missing
            from the worktree. Either would let apply fall back to treating the
            diff itself as replacement source.
    """
    patch_path = Path(str(payload.get("patch_path") or ""))
    repo_root = Path(str(payload.get("repo") or payload.get("kernel_repo") or ""))
    if not repo_root.is_dir():
        raise RuntimeError(f"pre-applied patch needs its repo root, got {repo_root!s:.200}")
    snapshot = patch_path.parent / "preapplied_snapshot"
    for target in repo_relative_manifest(patch_path.read_text(encoding="utf-8", errors="replace")):
        if target.op is not PatchOp.WRITE:
            continue
        relative = target.path
        source = repo_root / relative
        if not source.is_file():
            raise RuntimeError(f"pre-applied patch writes {relative}, which is absent from {repo_root}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {**payload, "snapshot_dir": str(snapshot)}


def _find_selected_kernel_source(state: Any, kernel_id: str) -> str:
    """Look up a kernel's source file from the last trace-analyze result.

    Searches ``state.last_trace_analyze`` (preferring ``hot_kernels_top15``,
    falling back to ``hot_kernels``) for the entry matching ``kernel_id``.
    """
    kernels = (
        (state.last_trace_analyze or {}).get("hot_kernels_top15")
        or (state.last_trace_analyze or {}).get("hot_kernels")
        or []
    )
    for item in kernels:
        if not isinstance(item, dict):
            continue
        if str(item.get("kernel_id") or "") == kernel_id:
            return str(item.get("source_file") or "")
    return ""


class AmbiguousIntegrationTarget(Exception):
    """A bare ``kernel_id`` named several pending integration records.

    ``kernel_id`` is not unique across a nomination round, so it cannot select
    the record a KEEP/REVERT verdict is bound to.
    """


def _is_paired_gemm_measurement(payload: Mapping[str, Any]) -> bool:
    return payload.get("source") == "forge_gemm_paired"


def _integrate_mode(payload: Mapping[str, Any]) -> str:
    return str(payload.get("mode") or "patch").strip().lower()


def _fill_integrate_defaults_from_state(
    payload: dict,
    *,
    session_dir: Path,
) -> dict:
    """Pull ``base_tput`` / ``config_path`` / ``extra_server_args`` defaults from SharedState.

    Runs before the ``base_tput > 0`` hard-check in ``integrate_handler`` for
    bare ``{"kernel_id": ...}`` payloads. Always returns a shallow copy; never
    raises on a missing snapshot.



    Raises:
        AmbiguousIntegrationTarget: The payload carries no ``integration_id``
            that resolves and its ``kernel_id`` matches more than one pending
            record.
    """
    from ..state.shared_state import SharedState

    resolved = dict(payload)
    if _is_paired_gemm_measurement(resolved):
        reference = resolved.get("paired_reference")
        if (
            not isinstance(reference, dict)
            or float(reference.get("tput") or 0.0) <= 0
            or not resolved.get("config_path")
            or "extra_server_args" not in resolved
            or not isinstance(resolved.get("extra_envs"), dict)
        ):
            raise ValueError("GEMM paired measurement requires an explicit entry reference and recipe")
        resolved["base_tput"] = reference["tput"]
        return resolved
    state = SharedState.load_or_init(session_dir)

    records = state.pending_kernel_integration_records()
    pending_record = _pending_integration_record(records, str(resolved.get("integration_id") or ""))
    if pending_record is None and resolved.get("kernel_id"):
        pending_record = _sole_pending_record_for_kernel(records, resolved)
    if pending_record is not None:
        resolved.setdefault("integration_id", str(pending_record.get("integration_id") or ""))
        resolved.setdefault("kernel_id", str(pending_record.get("kernel_id") or ""))
        _fill_pending_record_identity(resolved, pending_record)
        _fill_fusion_sibling_defaults(resolved, pending_record)

    current_best = getattr(state, "current_best", None) or {}
    _fill_integrate_recipe_from_current_best(resolved, state, current_best)
    _merge_integrate_extra_envs(resolved, payload, current_best)
    _fill_integrate_kernel_attempt_defaults(resolved, state)
    return resolved


def _sole_pending_record_for_kernel(records: list[dict], resolved: Mapping[str, Any]) -> dict | None:
    requested_kernel_id = str(resolved.get("kernel_id") or "")
    requested_task_key = str(resolved.get("task_group_key") or "")
    candidates = [
        record
        for record in records
        if str(record.get("kernel_id") or "") == requested_kernel_id
        and (not requested_task_key or str(record.get("task_group_key") or "") == requested_task_key)
    ]
    if len(candidates) > 1:
        raise AmbiguousIntegrationTarget(
            f"integrate refused: kernel_id={requested_kernel_id!r} matches "
            f"{len(candidates)} pending integration records; an explicit "
            "integration_id is required to bind the KEEP/REVERT verdict. "
            f"candidates={sorted(str(record.get('integration_id') or '') for record in candidates)!r}"
        )
    return candidates[0] if candidates else None


def _fill_fusion_sibling_defaults(resolved: dict, pending_record: Mapping[str, Any]) -> None:
    if str(pending_record.get("source") or "") != "forge_fusion":
        return
    resolved.setdefault("source", "forge_fusion")
    resolved.setdefault("action_label", str(pending_record.get("action_label") or "fusion"))
    fusion_env_flags = pending_record.get("fusion_env_flags")
    if isinstance(fusion_env_flags, dict) and fusion_env_flags:
        requested = resolved.get("extra_envs")
        requested = dict(requested) if isinstance(requested, dict) else {}
        # The sibling's own flags win over anything already requested.
        resolved["extra_envs"] = {**requested, **{str(k): str(v) for k, v in fusion_env_flags.items()}}
    if "keep_threshold_pct" not in resolved and pending_record.get("keep_threshold_pct") is not None:
        try:
            resolved["keep_threshold_pct"] = float(pending_record.get("keep_threshold_pct"))
        except (TypeError, ValueError):
            pass


def _fill_integrate_recipe_from_current_best(
    resolved: dict,
    state: Any,
    current_best: Mapping[str, Any],
) -> None:
    from ..state.shared_state import resolve_grading_anchor_tput

    if float(resolved.get("base_tput", 0.0) or 0.0) <= 0:
        anchor_tput = resolve_grading_anchor_tput(state)
        if anchor_tput > 0:
            resolved["base_tput"] = anchor_tput

    if not resolved.get("config_path"):
        config_path = getattr(state, "baseline_config_path", "") or ""
        if config_path:
            resolved["config_path"] = config_path

    if not resolved.get("extra_server_args"):
        inherited_args = current_best.get("extra_server_args") or ""
        if inherited_args:
            resolved["extra_server_args"] = inherited_args
    for key in ("remove_args", "unset_envs"):
        resolved[key] = to_str_list(resolved.get(key, current_best.get(key)))
    resolved.setdefault("args_mode", current_best.get("args_mode") or "append")


def _merge_integrate_extra_envs(
    resolved: dict,
    payload: Mapping[str, Any],
    current_best: Mapping[str, Any],
) -> None:
    current_envs = current_best.get("extra_envs")
    current_envs = dict(current_envs) if isinstance(current_envs, dict) else {}
    for key in resolved["unset_envs"]:
        current_envs.pop(key, None)
    requested_envs = resolved.get("extra_envs")
    requested_envs = dict(requested_envs) if isinstance(requested_envs, dict) else {}
    if "unset_envs" not in payload:
        resolved["unset_envs"] = [key for key in resolved["unset_envs"] if key not in requested_envs]
    else:
        for key in resolved["unset_envs"]:
            requested_envs.pop(key, None)
    if current_envs or requested_envs or "extra_envs" in resolved:
        resolved["extra_envs"] = {**current_envs, **requested_envs}


def _fill_integrate_kernel_attempt_defaults(resolved: dict, state: Any) -> None:
    kernel_id = str(resolved.get("kernel_id") or "")
    if not kernel_id:
        return
    attempt = _entry_by_kernel_id(state, kernel_id) or {}
    if not resolved.get("task_group_key"):
        task_group_key = str(attempt.get("task_group_key") or "")
        if task_group_key:
            resolved["task_group_key"] = task_group_key
    last_kernel_opt = state.last_kernel_opt
    blocked_by_last_opt = (
        isinstance(last_kernel_opt, dict)
        and str(last_kernel_opt.get("kernel_id") or "") == kernel_id
        and bool(last_kernel_opt.get("vendor_playbook_deploy_blocked"))
    )
    if attempt.get("vendor_playbook_deploy_blocked") or blocked_by_last_opt:
        resolved["_vendor_playbook_deploy_blocked"] = True


def _fill_integrate_snapshot_from_bundle(resolved: dict, bundle: Any) -> None:
    """Backfill integrate inputs from a recorded multi-file artifact bundle.

    A bundle is bound to the sibling that produced it by ``integration_id``. When
    the caller already resolved this integrate from a specific pending record
    (i.e. ``resolved`` carries an ``integration_id``), a bundle stamped with a
    *different* id belongs to another sibling of the same nomination round and
    must not be merged in: doing so would land one sibling's multi-file write
    set under a second sibling's integrate. Under the one-patch era every
    kernel_id had exactly one bundle so this never arose; the fallbacks keyed on
    ``kernel_id`` (``last_kernel_opt`` / per-kernel ledger) now route several
    bundles through the same kernel_id, so the guard is load-bearing.

    A bundle with no ``integration_id`` of its own predates the contract and is
    accepted as before -- there is no id to disagree with.
    """
    if not isinstance(bundle, dict) or bundle.get("type") != "patch_snapshot":
        return
    if not bundle_belongs_to(bundle, resolved.get("integration_id")):
        # Cross-sibling bundle -- refuse rather than silently mixing write sets.
        return
    if not resolved.get("snapshot_dir") and bundle.get("snapshot_dir"):
        resolved["snapshot_dir"] = str(bundle["snapshot_dir"])
    if not resolved.get("patch_path") and bundle.get("patch_path"):
        resolved["patch_path"] = str(bundle["patch_path"])
    if not resolved.get("kernel_repo") and bundle.get("repo_root"):
        resolved["kernel_repo"] = str(bundle["repo_root"])
    if not resolved.get("producer_manifest") and bundle.get("producer_manifest"):
        resolved["producer_manifest"] = str(bundle["producer_manifest"])
    if not resolved.get("patch_write_paths"):
        write_paths = [str(path) for path in (bundle.get("write_paths") or []) if str(path or "").strip()]
        if write_paths:
            resolved["patch_write_paths"] = write_paths


def _fill_integrate_provenance(
    resolved: dict,
    *,
    framework_applyback: Any,
    integration_validation_status: Any,
) -> None:
    """Backfill artifact provenance for an integrate resolved from a ledger entry.

    These two fields arm the strict accuracy gate. A KEEP the ``source_file`` dedup
    drops from the pending queue resolves through a fallback instead, and without
    them a reference-only apply-back reads as an ordinary kernel patch.
    """
    if not resolved.get("artifact_kind") and isinstance(framework_applyback, dict):
        kind = str(framework_applyback.get("artifact_kind") or "")
        if kind:
            resolved["artifact_kind"] = kind
    if not resolved.get("integration_validation_status"):
        status = str(integration_validation_status or "")
        if status:
            resolved["integration_validation_status"] = status


class _ArtifactRecord(NamedTuple):
    """Where a recorded candidate keeps the patch, snapshot and source an integrate needs."""

    bundle: Any
    snapshot_dir: str
    deploy_patch_path: str
    deploy_repo_root: str
    artifact_path: str
    source_file: str
    framework_applyback: Any
    integration_validation_status: Any


def _best_kernel_artifact(last_kernel: Mapping[str, Any]) -> _ArtifactRecord:
    return _ArtifactRecord(
        bundle=last_kernel.get("best_artifact_bundle"),
        snapshot_dir=str(last_kernel.get("deploy_snapshot_dir") or ""),
        deploy_patch_path=str(last_kernel.get("deploy_patch_path") or ""),
        deploy_repo_root=str(last_kernel.get("deploy_repo_root") or ""),
        artifact_path=str(
            last_kernel.get("best_artifact_path")
            or last_kernel.get("patch_path")
            or last_kernel.get("optimized_path")
            or ""
        ),
        source_file=str(last_kernel.get("source_file") or ""),
        framework_applyback=last_kernel.get("framework_applyback"),
        integration_validation_status=last_kernel.get("integration_validation_status"),
    )


def _fill_integrate_from_artifact(resolved: dict, record: _ArtifactRecord) -> None:
    _fill_integrate_snapshot_from_bundle(resolved, record.bundle)
    if not resolved.get("snapshot_dir") and record.snapshot_dir:
        resolved["snapshot_dir"] = record.snapshot_dir
        if record.deploy_patch_path and not resolved.get("patch_path"):
            resolved["patch_path"] = record.deploy_patch_path
        if record.deploy_repo_root and not resolved.get("kernel_repo"):
            resolved["kernel_repo"] = record.deploy_repo_root
    if not resolved.get("patch_path") and record.artifact_path:
        resolved["patch_path"] = record.artifact_path
    if not resolved.get("source_file") and record.source_file:
        resolved["source_file"] = record.source_file
    _fill_integrate_provenance(
        resolved,
        framework_applyback=record.framework_applyback,
        integration_validation_status=record.integration_validation_status,
    )


def _pending_integration_record(records: list[dict], integration_id: str) -> dict | None:
    return next(
        (record for record in records if str(record.get("integration_id") or "") == integration_id),
        None,
    )


def _fill_pending_record_identity(resolved: dict, pending_record: Mapping[str, Any]) -> None:
    for key in ("task_group_key", "identity_route", "artifact_kind", "integration_validation_status"):
        resolved.setdefault(key, str(pending_record.get(key) or ""))


def _bind_integrate_to_pending_record(resolved: dict, pending_record: Mapping[str, Any]) -> None:
    resolved["kernel_id"] = str(pending_record.get("kernel_id") or resolved.get("kernel_id") or "")
    resolved["integration_id"] = str(pending_record.get("integration_id") or resolved.get("integration_id") or "")
    _fill_pending_record_identity(resolved, pending_record)
    _fill_integrate_snapshot_from_bundle(resolved, pending_record.get("artifact_bundle"))
    if not resolved.get("snapshot_dir") and pending_record.get("snapshot_dir"):
        resolved["snapshot_dir"] = str(pending_record["snapshot_dir"])
    if not resolved.get("patch_path"):
        resolved["patch_path"] = str(
            pending_record.get("deploy_patch_path") or pending_record.get("artifact_path") or ""
        )
    if not resolved.get("kernel_repo") and pending_record.get("deploy_repo_root"):
        resolved["kernel_repo"] = str(pending_record["deploy_repo_root"])
    if not resolved.get("source_file") and pending_record.get("source_file"):
        resolved["source_file"] = str(pending_record["source_file"])


def _missing_integration_inputs(resolved: dict, last_kernel: Mapping[str, Any]) -> HandlerResult | None:
    patch_path = str(resolved.get("patch_path") or "").strip()
    target_file = str(resolved.get("target_file") or resolved.get("source_file") or "").strip()
    if patch_path and target_file:
        return None
    missing = []
    if not patch_path:
        missing.append("patch_path")
    if not target_file:
        missing.append("target_file/source_file")
    return {
        "status": "failed",
        "error_class": "missing_integration_inputs",
        "error": "integrate requires an optimized artifact and target source before E2E",
        "decision": "REVERT",
        "kernel_id": str(resolved.get("kernel_id") or "") or None,
        "patch_path": patch_path or None,
        "target_file": target_file or None,
        "missing": missing,
        "last_kernel_opt": {
            k: last_kernel.get(k)
            for k in ("kernel_id", "best_artifact_path", "patch_path", "source_file")
            if k in last_kernel
        },
    }


def _resolve_integrate_payload(payload: dict, *, session_dir: Path) -> tuple[dict, HandlerResult | None]:
    """Fill integrate inputs from SharedState when Orchestration sends only kernel_id (artifact in ``last_kernel_opt``, source in ``last_trace_analyze``).

    Args:
        payload: The integrate request payload.
        session_dir: Session directory to load SharedState from.

    Returns:
        A tuple of ``(resolved_payload, error_result)`` where ``error_result``
        is a failure ``HandlerResult`` when required inputs are missing, else
        ``None``.
    """
    from ..state.shared_state import SharedState

    resolved = dict(payload)
    state = SharedState.load_or_init(session_dir)
    last_kernel = state.last_kernel_opt or {}
    pending_record = _pending_integration_record(
        state.pending_kernel_integration_records(),
        str(resolved.get("integration_id") or ""),
    )
    if pending_record is not None:
        _bind_integrate_to_pending_record(resolved, pending_record)
    kernel_id = str(resolved.get("kernel_id") or "")

    if kernel_id and str(last_kernel.get("kernel_id") or "") == kernel_id:
        # Snapshot deploy: prefer the original patch + snapshot dir so the whole
        # multi-file patch lands atomically.
        _fill_integrate_from_artifact(resolved, _best_kernel_artifact(last_kernel))

    # Multi-KEEP queue fallback: pull patch_path/source_file from the per-kernel
    # ledger for KEEPs other than the strongest pending one.
    if kernel_id:
        attempt = _entry_by_kernel_id(state, kernel_id) or {}
        _fill_integrate_from_artifact(
            resolved,
            _ArtifactRecord(
                bundle=attempt.get("last_artifact_bundle"),
                snapshot_dir=str(attempt.get("last_snapshot_dir") or ""),
                deploy_patch_path=str(attempt.get("last_deploy_patch_path") or ""),
                deploy_repo_root=str(attempt.get("last_deploy_repo_root") or ""),
                artifact_path=str(attempt.get("last_artifact_path") or ""),
                source_file=str(attempt.get("last_source_file") or ""),
                framework_applyback=attempt.get("last_framework_applyback"),
                integration_validation_status=attempt.get("last_integration_validation_status"),
            ),
        )

    if kernel_id and not (resolved.get("target_file") or resolved.get("source_file")):
        source = _find_selected_kernel_source(state, kernel_id)
        if source:
            resolved["source_file"] = source

    return resolved, _missing_integration_inputs(resolved, last_kernel)


def _normalize_precision(value: Any) -> str:
    """Normalize a precision label to a trimmed lower-case string.

    Args:
        value (Any): Raw precision value (e.g. ``"FP8"``, ``None``).

    Returns:
        str: The lower-cased, whitespace-stripped precision, or an empty
            string for falsy input.
    """
    return str(value or "").strip().lower()


def _requested_framework(payload: dict, state: Any, fallback: str = "") -> str:
    return _requested_setting(payload, state, "framework", fallback).lower()


def _requested_setting(payload: dict, state: Any, key: str, fallback: str = "") -> str:
    return str(payload.get(key) or getattr(state, key, "") or fallback).strip()


def _lane_budget(state: Any, lane: str, *, gemm_target_costs_sec: tuple[int, ...] = ()) -> LaneAllocation | None:
    """Derive one lane's share of the phase's remaining time.

    Wraps :func:`lane_budget.allocate`, which divides the remaining time between
    the lanes and returns, per lane, both a second budget and how many targets that
    budget can fund. Every lane draws its share from this single probe of session
    state, so the shares stay parts of one whole.
    """
    remaining = _lane_remaining_minutes(state)
    allocation = _allocate_lane_budgets(remaining, gemm_target_costs_sec=gemm_target_costs_sec)[lane]
    if remaining is not None and not allocation.is_fundable:
        return None
    return allocation


def _unfunded_lane_skip(lane: str) -> HandlerResult:
    """What a forge lane returns instead of starting when the phase cannot fund it: a skip, not a verdict."""
    reason = f"what is left of the KERNEL phase time budget cannot fund one {lane} target"
    log.info("forge %s lane skipped: %s", lane, reason)
    return {"status": "skipped", "error_class": INSUFFICIENT_BUDGET, "skip_reason": reason, "backend": "forge"}


def _lane_remaining_minutes(state: Any) -> float | None:
    """Minutes a lane may plan against: the tighter of session and phase.

    The session clock alone overfunds a phase that has already spent most of its
    own slice, and work planned past the phase exit is cut off partway through.
    """
    from ..phases.machine_state import phase_budget_remaining_seconds

    remaining_fn = getattr(state, "remaining_minutes", None)
    session = remaining_fn() if callable(remaining_fn) else None
    if session is None:
        return None
    phase_sec = phase_budget_remaining_seconds(state)
    if phase_sec is None:
        return float(session)
    return min(float(session), float(phase_sec) / 60.0)


def _gemm_tuning_timeout_sec(payload: dict, *, lane_budget_sec: int = 0) -> int:
    """Resolve the GEMM-tuning subprocess timeout in seconds.

    Reads ``payload['timeout_sec']`` then the
    ``HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC`` env var -- both operator inputs, so both
    outrank a derived budget -- then the lane's share of the phase, then the module
    default; the result is floored at 60 seconds.
    """
    raw = payload.get("timeout_sec") or os.environ.get(
        "HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC",
        "",
    )
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = lane_budget_sec if lane_budget_sec > 0 else _DEFAULT_GEMM_TUNING_TIMEOUT_SEC
    return max(60, value)


def _gemm_router_targets(
    *,
    model_path: str,
    framework: str,
    precision: str,
    quant_type: str,
    gpu_type: str,
    kernel_signature_log: str,
    has_untuned_csv: bool,
    has_shapes_json: bool,
    has_tunableop_input: bool,
) -> tuple[tuple[str, int], ...]:
    """Ask the forge router which tuners it would run and what each costs.

    The router is the only place a per-tuner runtime estimate exists, and it
    returns them already in the execution order the gemm lane ceiling assumes.
    """
    try:
        from kernelforge.gemm_tune.model_analyzer import analyze_model
        from kernelforge.gemm_tune.router import select_tuners

        specs = select_tuners(
            analyze_model(model_path),
            framework=framework,
            precision=precision,
            quant_type=quant_type,
            gpu_type=gpu_type,
            kernel_signature_log=kernel_signature_log or None,
            has_untuned_csv=has_untuned_csv,
            has_shapes_json=has_shapes_json,
            has_tunableop_input=has_tunableop_input,
        )
    except Exception:
        log.debug("GEMM: could not consult the tuner router for lane cost estimates", exc_info=True)
        return ()
    return tuple((str(spec.name), max(0, int(spec.estimated_minutes * 60))) for spec in specs if spec.should_run)


async def _priced_gemm_lane(state: Any, **router_inputs: Any) -> LaneAllocation | None:
    """The gemm lane's share, priced on the router's own per-tuner estimates for ``_gemm_router_targets`` inputs."""
    targets = await asyncio.to_thread(_gemm_router_targets, **router_inputs)
    return _lane_budget(state, LANE_GEMM, gemm_target_costs_sec=tuple(cost for _, cost in targets))


def _forge_fusion_timeout_sec(payload: dict, *, lane_budget_sec: int = 0) -> int:
    """Resolve the forge-fusion subprocess timeout in seconds.

    Args:
        payload (dict): Request payload that may carry ``timeout``/``timeout_sec``.
        lane_budget_sec (int): The fusion lane's share of the phase. ``0`` means there
            is no share to take, as in an unbounded session, and keeps the module
            default rather than collapsing an unattended lane to a one-second timeout.

    Returns:
        int: The resolved timeout in seconds (>= 1).
    """
    raw = (
        payload.get("timeout")
        or payload.get("timeout_sec")
        or os.environ.get(
            "FORGE_FUSION_TIMEOUT",
            "",
        )
    )
    try:
        value = int(float(raw))
    except (OverflowError, TypeError, ValueError):
        value = lane_budget_sec if lane_budget_sec > 0 else 7200
    return max(1, value)


def _positive_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _fusion_attn_tp_size(state: object, payload: dict, *, tp: int) -> int:
    """Attention TP shard behind the session's serving ``tp``.

    DP-attention splits the ``tp`` ranks into ``dp`` attention groups, so
    forge-fuse cannot infer the attention shard from ``--tp`` alone -- and it
    stamps the harness group/head dims from exactly this number.
    """
    explicit = _positive_int(payload.get("attn_tp"))
    if explicit:
        return explicit
    from hyperloom.inference_optimizer.roofline_ceiling import _parse_server_arg, resolve_runtime_workload

    server_args = str(payload.get("extra_server_args") or "").strip()
    if not server_args:
        try:
            server_args = resolve_runtime_workload(state, arm="current_best").server_args
        except Exception:  # noqa: BLE001 - best-effort for partial state / test doubles
            server_args = ""

    if "--enable-dp-attention" not in str(server_args).replace("=", " ").split():
        return tp
    # launch_infera_node injects --dp-size = tp when a dp-attention flag arrives
    # without one, and these args are read before that injection.
    dp = (
        _positive_int(_parse_server_arg(server_args, "--dp-size"))
        or _positive_int(_parse_server_arg(server_args, "--data-parallel-size"))
        or tp
    )
    return max(1, tp // dp)


def _fusion_session_serve_args(
    state: object,
    payload: dict,
    *,
    framework: str,
    model_path: str,
) -> dict[str, int]:
    """TP / attention TP / KV block size / max-model-len the smoke must match."""
    tp = _positive_int(payload.get("tp") or getattr(state, "tp", 0))
    max_model_len = _positive_int(payload.get("max_model_len") or getattr(state, "max_model_len", 0))
    block_size = _positive_int(payload.get("block_size"))
    if block_size <= 0 and "vllm" in (framework or "").strip().lower():
        from hyperloom.inference_optimizer.model_config_utils import (
            _sparse_kv_block_size,
        )

        block_size = _positive_int(_sparse_kv_block_size(model_path))
    args: dict[str, int] = {}
    if tp:
        args["tp"] = tp
        args["attn_tp"] = _fusion_attn_tp_size(state, payload, tp=tp)
    if block_size:
        args["block_size"] = block_size
    if max_model_len:
        args["max_model_len"] = max_model_len
    return args


def _gemm_tuning_workspace(payload: dict, *, session_dir: Path) -> Path:
    """Resolve the workspace directory for a GEMM-tuning run.

    Honors an explicit ``payload['workspace_path']``; otherwise builds a path
    under ``<session_dir>/runs/gemm_tuning/`` keyed by ``task_id`` /
    ``request_id`` (or a timestamped fallback).
    """
    raw = payload.get("workspace_path")
    if raw:
        return Path(raw)
    suffix = str(payload.get("task_id") or payload.get("request_id") or "").strip()
    if not suffix:
        suffix = f"request_{int(time.time())}"
    return Path(session_dir) / "runs" / "gemm_tuning" / suffix


def _parse_forge_gemm_sentinel(stdout: str) -> dict[str, Any] | None:
    """Parse FORGE_GEMM_TUNE_RESULT_BEGIN/END sentinel block from stdout."""
    m = re.search(
        r"FORGE_GEMM_TUNE_RESULT_BEGIN\s*\n(.*?)\nFORGE_GEMM_TUNE_RESULT_END",
        stdout,
        re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _read_forge_result_json(workspace: Path) -> dict[str, Any]:
    """Read forge's on-disk ``result.json`` from the tuning workspace.

    forge always writes the full report (including ``tuners_skipped``) to
    ``<output_dir>/result.json``, even when the stdout sentinel omits some
    fields. Returns ``{}`` when missing or unparseable.
    """
    try:
        path = workspace / "result.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {}


def _derive_gemm_skip_reason(tuners_skipped: Any) -> str:
    """Join forge per-tuner skip reasons into one concise human-readable string."""
    if not isinstance(tuners_skipped, list):
        return ""
    parts: list[str] = []
    for entry in tuners_skipped:
        if not isinstance(entry, dict):
            continue
        reason = str(entry.get("skip_reason") or "").strip()
        if not reason:
            continue
        tuner = str(entry.get("tuner") or "").strip()
        parts.append(f"{tuner}: {reason}" if tuner else reason)
    return "; ".join(parts)


_FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC = 30


def _forge_gemm_tune_probe_cmd() -> list[str]:
    """Return the exact interpreter and CLI prefix used by GEMM tuning."""
    return [sys.executable, "-m", "kernelforge.cli", "gemm-tune", "--help"]


def _forge_gemm_tune_available() -> bool:
    """Check exactly what ``_build_cmd`` will run, in the interpreter it runs in.

    The tuner is a subpackage of the ``kernelforge`` that ships in this
    distribution, invoked as ``sys.executable -m kernelforge.cli gemm-tune run``.
    Vendoring forge in-tree removes the cross-checkout failures this probe was
    built for, but not the reason it is a subprocess: ``find_spec`` proves the
    module is importable and says nothing about whether ``gemm-tune`` is
    registered on the CLI, which is the thing ``_build_cmd`` actually needs. So
    ask the subcommand itself -- in a subprocess, so a heavy CLI import cannot
    land in the orchestrator's own process. ``--help`` exits 0 only if
    ``kernelforge.cli`` imported and ``gemm-tune`` is registered on it.
    """
    try:
        proc = subprocess.run(
            _forge_gemm_tune_probe_cmd(),
            capture_output=True,
            text=True,
            timeout=_FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "forge-gemm-tune preflight timed out after %ss in %s",
            _FORGE_GEMM_PREFLIGHT_TIMEOUT_SEC,
            sys.executable,
        )
        return False
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("forge-gemm-tune preflight could not start in %s: %s", sys.executable, exc)
        return False
    if proc.returncode == 0:
        return True
    log.info(
        "forge-gemm-tune preflight failed (rc=%s): %s",
        proc.returncode,
        ((proc.stderr or proc.stdout or "").strip()[-400:] or "(no output)"),
    )
    return False


def _resolve_aiter_root_for_forge() -> str:
    """Resolve AITER's source root from split wheels or editable installs."""
    explicit = os.environ.get("AITER_ROOT_DIR", "").strip()
    if explicit:
        return explicit
    try:
        spec = importlib.util.find_spec("aiter_meta")
    except (ModuleNotFoundError, ValueError):
        spec = None
    locations = getattr(spec, "submodule_search_locations", None) or []
    for location in locations:
        root = Path(location)
        if (root / "csrc").is_dir():
            return str(root)
    try:
        spec = importlib.util.find_spec("aiter")
    except (ModuleNotFoundError, ValueError):
        spec = None
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    origin = getattr(spec, "origin", None)
    if origin:
        locations.append(str(Path(origin).parent))
    for location in locations:
        package = Path(location)
        root = package.parent
        if package.name == "aiter" and (root / "csrc").is_dir():
            return str(root)
    return ""


def _resolve_forge_precision_and_quant(state, payload: dict) -> tuple[str, str]:
    """Resolve the actual runtime precision and quant_type for forge tuning.

    Priority:
    1. Explicit payload override
    2. --quantization from current_best server args (actual runtime)
    3. state.precision (session-level, may be stale)
    4. Default: bf16

    Returns (precision, quant_type) tuple.
    """
    from hyperloom.inference_optimizer.roofline_ceiling import _parse_server_arg, resolve_runtime_workload

    quant_type = str(payload.get("quant_type") or "auto").strip()

    if payload.get("precision"):
        precision = _normalize_precision(payload["precision"])
        if precision == "fp8" and quant_type.lower() == "auto":
            quant_type = _requested_fp8_quant_type(state, payload)
        return precision, quant_type

    # Resolve from actual server args (baseline yaml + current_best overlay).
    current_best = getattr(state, "current_best", None) or {}
    server_args = resolve_runtime_workload(state, arm="current_best").server_args
    extra_envs = dict(current_best.get("extra_envs") or {})
    ref_envs = dict(getattr(state, "reference_envs", None) or {})
    per_token_signal = is_truthy(extra_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN")) or is_truthy(
        ref_envs.get("SGLANG_USE_AITER_FP8_PER_TOKEN")
    )

    quantization_arg = _parse_server_arg(server_args, "--quantization").lower()

    if quantization_arg == "fp8":
        precision = "fp8"
        # Hand forge the fp8 GEMM path the model runs: explicit per-token env wins,
        # else the checkpoint's static format, else "auto".
        if per_token_signal:
            quant_type = "per_token"
        else:
            quant_type = _requested_fp8_quant_type(state, payload)
        return precision, quant_type

    if quantization_arg in ("fp4", "mxfp4"):
        return quantization_arg, "fp4"

    # Fall back to session precision. When that is just the format a Quark or compressed-tensors checkpoint nests, it
    # is one label for layers the checkpoint can store differently (MXFP4 experts, fp8 attention), and forge can tell
    # them apart only when it is left to route each lane itself.
    precision = _normalize_precision(state.precision)
    model_path = _requested_setting(payload, state, "model_path")
    if precision and precision == checkpoint_format((_read_model_config(model_path) or {}).get("quantization_config")):
        return "auto", quant_type
    if not precision:
        precision = "bf16"
    if precision == "fp8" and quant_type.lower() == "auto":
        quant_type = _requested_fp8_quant_type(state, payload)
    return precision, quant_type


def _requested_fp8_quant_type(state, payload: dict) -> str:
    return _resolve_fp8_quant_type(
        _requested_setting(payload, state, "model_path"),
        _requested_setting(payload, state, "gpu_type"),
        _requested_framework(payload, state),
    )


#: An aiter dispatch line, hit or miss. Either one proves the process actually
#: routed a GEMM through aiter, which is what makes a server log usable as a
#: shape source; a log without one is silent about shapes no matter how recent
#: or how well its workspace matches.
_AITER_DISPATCH_MARKER = "shape is M:"
#: The MoE half of the same question. The resolved log is not only a dense-shape
#: source: ``kernelforge.gemm_tune.router`` reads it for MoE stage coverage and
#: 1-stage ASM detection, and those parse ``[fused_moe]`` dispatch lines, which
#: aiter prints from a different code path than the dense ``shape is M:`` ones.
#: Requiring the dense marker alone would reject a log that is fully informative
#: about MoE routing -- on a fleet where every model under tuning is MoE, that
#: is the common case, and the router would silently fall back to "tune CK 2-stage
#: unconditionally".
_AITER_MOE_DISPATCH_MARKERS = (b"[fused_moe]", b"Mxfp4 MoE backend")
#: Only the M of a dispatch line. Reading M straight off the serving log is the
#: one token source grounded in what the model actually ran -- forge's fallback
#: derives ``--tokens`` from ``conc`` alone, and real fleet logs reach M=15842,
#: far outside anything that derivation produces.
_AITER_M_RE = re.compile(rb"shape is M:(\d+),")
#: Read logs in chunks: a fleet server.log is ~17MB and the evidence question
#: is usually answered in the first few KB.
_LOG_SCAN_CHUNK = 1 << 20


#: Longest a dispatch prefix can be, so a match straddling a chunk boundary is
#: carried into the next read. ``shape is M:`` plus its digits is far shorter.
_LOG_SCAN_OVERLAP = 64


def _scan_serving_log_m(path) -> dict[int, int]:
    """Count the M values of the dense aiter dispatch lines in a serving log.

    Chunks overlap by ``_LOG_SCAN_OVERLAP`` so a match spanning a boundary is
    still seen, but the carry starts after the last match already counted:
    re-feeding a fixed tail would count any match landing in it twice, which
    skews the frequency ranking that picks ``--tokens``.

    Any read error yields no counts -- an unreadable candidate is not a usable
    shape source either way.
    """
    counts: dict[int, int] = {}
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_LOG_SCAN_CHUNK)
                if not chunk:
                    break
                buf = carry + chunk
                last_end = 0
                for match in _AITER_M_RE.finditer(buf):
                    last_end = match.end()
                    value = int(match.group(1))
                    if value > 0:
                        counts[value] = counts.get(value, 0) + 1
                carry = buf[max(last_end, len(buf) - _LOG_SCAN_OVERLAP) :]
    except (OSError, ValueError):
        return {}
    return counts


def _log_has_aiter_evidence(path) -> bool:
    """True when the log carries at least one aiter dispatch line, dense or MoE.

    Kept separate from :func:`_scan_serving_log_m` rather than folded into it as
    a ``first_only`` flag. Two reasons, both of which cost a real behaviour bug
    when the two were one function:

    * The M counter skips ``M:0``, so a log whose first dispatch line carried
      one read as "no evidence at all".
    * Evidence is not dense-only. ``[fused_moe]`` lines make a log fully usable
      for the MoE routing decisions that consume the same path.

    Stops at the first marker: a log with evidence usually proves it in the
    first few KB, and only a silent log is read to the end.
    """
    markers = (_AITER_DISPATCH_MARKER.encode(), *_AITER_MOE_DISPATCH_MARKERS)
    carry = b""
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_LOG_SCAN_CHUNK)
                if not chunk:
                    return False
                buf = carry + chunk
                if any(marker in buf for marker in markers):
                    return True
                carry = buf[-_LOG_SCAN_OVERLAP:]
    except OSError:
        return False


def _tokens_from_serving_log(path, limit: int = 16, reserve_largest: int = 4) -> str:
    """Derive forge's ``--tokens`` from the M values the server actually saw.

    Returns up to ``limit`` distinct M values, smallest first, as a
    comma-separated string -- empty when the log carries no dispatch lines.

    Selection is frequency-ranked, because tuning the M values the model spends
    its time at beats tuning the largest one it ever reached. But frequency
    alone is not enough: a serving warmup sweeps every M about equally often,
    so on real logs the counts come out uniform and the ranking degenerates
    into its tie-break. Measured on two fleet sessions, every distinct M
    carried an identical count (17 values x4, and 44 values x40), so a plain
    frequency cut kept the smallest M and dropped exactly the large prefill
    shapes -- 16384/24576/32768 and 57344/65536 -- that the runtime then
    missed. Reserve slots for the largest observed M so the prefill end
    survives the cut; GEMM time scales with M, so those are also where the
    end-to-end time actually is.
    """
    counts = _scan_serving_log_m(path)
    if not counts:
        return ""
    # Never let the reservation crowd out the frequency ranking: at most a
    # quarter of the budget goes to "largest", and always at least one slot.
    reserve = min(max(reserve_largest, 0), max(1, limit // 4))
    picked: list[int] = sorted(counts, reverse=True)[:reserve]
    for value, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(picked) >= limit:
            break
        if value not in picked:
            picked.append(value)
    return ",".join(str(m) for m in sorted(picked))


def _resolve_trace_shape_manifest(state, session_dir: Path) -> str:
    """Find the newest TraceShapeManifest this session produced.

    ``bypass_trace_analysis`` writes ``trace_shape_manifest.json`` next to its
    other bypass artifacts; forge calls it the preferred dense-shape source but
    nothing forwarded it, so the file was written and never read. Newest wins:
    a later trace reflects the currently resolved server args.
    """
    # Deduplicate: state.session_dir is usually the same path we were handed,
    # and an empty session then paid for two full-tree walks to find nothing.
    seen_roots: set[str] = set()
    roots: list[Path] = []
    for raw in (session_dir, Path(str(getattr(state, "session_dir", "") or session_dir))):
        if raw is None or not Path(raw).is_dir():
            continue
        key = str(Path(raw).resolve())
        if key in seen_roots:
            continue
        seen_roots.add(key)
        roots.append(Path(raw))
    for root in roots:
        best: tuple[float, str] | None = None
        for found in Path(root).glob("**/trace_shape_manifest.json"):
            try:
                mtime = found.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, str(found))
        if best is not None:
            return best[1]
    return ""


def _resolve_forge_server_log(state, session_dir: Path) -> str:
    """Find the server log matching the current runtime configuration.

    Priority: current_best workspace (matches the resolved server args)
    → baseline workspace → most recent server.log under runs/.

    Every candidate must carry aiter dispatch evidence. Picking on existence
    alone made the first *present* log win, so a ``current_best`` workspace
    whose log never routed a GEMM through aiter ended the search and the
    ``runs/`` fallback became unreachable -- the tuner was then handed a log
    that had no shapes in it at all.

    The server log is written by the benchmark server at startup and lives in
    the warmup_round benchmark directory (where the server process was first
    launched). When ``current_best.workspace`` points to the measure_round
    benchmark directory (one level sibling), the log is not there — so we also
    check sibling ``warmup_round/`` dirs and walk up to the parent run
    directory.
    """

    def _find_server_log_near(workspace_str: str) -> str | None:
        if not workspace_str:
            return None
        ws = Path(workspace_str)
        # Direct hit (server started in this exact dir).
        direct = ws / "server.log"
        if direct.is_file() and _log_has_aiter_evidence(direct):
            return str(direct)
        # Sibling warmup_round — benchmark dirs sit under
        # {run_hash}/{warmup_round|measure_round}/{benchmark_dir}/
        parent = ws.parent  # e.g. measure_round/
        if parent.name in ("warmup_round", "measure_round"):
            run_hash_dir = parent.parent
        else:
            run_hash_dir = parent
        warmup = run_hash_dir / "warmup_round"
        if warmup.is_dir():
            candidates: list[tuple[float, str]] = []
            for child in warmup.iterdir():
                sl = child / "server.log"
                if sl.is_file():
                    try:
                        candidates.append((sl.stat().st_mtime, str(sl)))
                    except OSError:
                        continue
            candidates.sort(reverse=True)
            for _mtime, candidate in candidates:
                if _log_has_aiter_evidence(candidate):
                    return candidate
        return None

    current_best = getattr(state, "current_best", None) or {}
    found = _find_server_log_near(str(current_best.get("workspace") or "").strip())
    if found:
        return found

    last_baseline = getattr(state, "last_baseline", None) or {}
    if isinstance(last_baseline, dict):
        found = _find_server_log_near(str(last_baseline.get("workspace") or "").strip())
        if found:
            return found

    # Fallback: the whole runs/ tree, newest first. Restricting this to a fixed
    # (baseline, explore, gemm_tuning, roofline) tuple skipped runs/integrate/,
    # which is where the GEMM validation runs put their logs -- those sessions
    # got "" plus a warning telling them to enable a flag that was already on.
    # Newest-first with an early return also means only the logs newer than the
    # winner are scanned, instead of every log in the tree.
    runs_dir = session_dir / "runs"
    candidates_by_age: list[tuple[float, Path]] = []
    if runs_dir.is_dir():
        for candidate_log in runs_dir.glob("**/server.log"):
            try:
                candidates_by_age.append((candidate_log.stat().st_mtime, candidate_log))
            except OSError:
                continue
        candidates_by_age.sort(key=lambda item: item[0], reverse=True)
        for _mtime, candidate_log in candidates_by_age:
            if _log_has_aiter_evidence(candidate_log):
                return str(candidate_log)

    # Separate the two ways this fails. No server.log at all is an upstream
    # gap; logs that exist but never dispatched through aiter means the serving
    # run had AITER_LOG_TUNED_CONFIG off. Both return "", but only the second is
    # actionable, and one silent "" hid it. Reuse the listing above rather than
    # walking the tree a second time.
    if candidates_by_age:
        log.warning(
            "GEMM: %d server.log file(s) under %s but none contain aiter dispatch "
            "lines (dense %r or MoE %s), so there is no runtime shape source. "
            "Serving runs need AITER_LOG_TUNED_CONFIG enabled for shapes to be "
            "observable",
            len(candidates_by_age),
            runs_dir,
            _AITER_DISPATCH_MARKER,
            " / ".join(repr(m.decode()) for m in _AITER_MOE_DISPATCH_MARKERS),
        )
    return ""


def _is_forge_compatible_shapes_json(path: Path) -> bool:
    """Validate that a shapes JSON file matches forge's expected format.

    Forge expects: [{"M": int, "N": int, "K": int}, ...]
    or {"shapes": [{"M": int, "N": int, "K": int}, ...]}
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("shapes", [])
        if not isinstance(data, list) or not data:
            return False
        sample = data[0]
        if not isinstance(sample, dict):
            return False
        # Must have M/N/K keys (case-insensitive).
        keys = {k.upper() for k in sample}
        return {"M", "N", "K"}.issubset(keys)
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _profile_shapes_are_fresh(state: Any) -> bool:
    """Return whether the latest profile matches the active workload/config."""
    return bool(state.profile_trace_matches_workload())


def _resolve_forge_shapes(
    state,
    session_dir: Path,
    *,
    require_fresh_profile: bool = False,
    precision: str = "",
) -> str:
    """Find TraceLens shapes JSON if available and in forge-compatible format.

    Forge dense tuners expect: [{"M": int, "N": int, "K": int}, ...]
    Only passes files that match this schema; incompatible formats are
    silently skipped so forge falls back to config.json shape derivation.

    ``precision`` scopes the traced shapes to the dtype the tuner will actually
    tune (a trace carries every GEMM dtype the model runs, e.g. FP8 projections
    alongside BF16 router heads). Empty means "no dtype scoping".

    When scoping is requested the candidate extraction runs first, because it is
    the only source whose dtype can be checked: a pre-rendered shapes artifact is
    a bare ``[{M,N,K}]`` list carrying no dtype or provenance, so an artifact
    recorded for another dtype would otherwise be handed to the tuner ahead of
    correctly-scoped shapes. Artifacts stay the fallback for the unscoped case
    and for when the trace yields nothing for this precision.
    """
    if require_fresh_profile and not _profile_shapes_are_fresh(state):
        log.info(
            "Forge GEMM shapes: latest TraceLens profile does not match the "
            "active workload/config; ignoring its shape artifacts"
        )
        return ""
    last_trace = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(last_trace, dict):
        return ""

    candidates: list[str] = []

    # Prefer explicit artifact fields when TraceLens exposes them.
    artifact_paths = last_trace.get("artifact_paths")
    for source, keys in (
        (last_trace, ("shapes_json", "shapes_path")),
        (artifact_paths if isinstance(artifact_paths, dict) else {}, ("shapes_json", "shapes", "gemm_shapes_json")),
    ):
        for key in keys:
            raw = str(source.get(key) or "").strip()
            if raw:
                candidates.append(raw)
    # Fallback: check beside candidates_path.
    candidates_path_str = last_trace.get("candidates_path") or ""
    if candidates_path_str:
        cand_file = Path(candidates_path_str)
        if cand_file.is_file():
            shapes_file = cand_file.parent / "shapes.json"
            candidates.append(str(shapes_file))

    # Extract the GEMM shapes observed by the latest TraceLens analysis. Older
    # traces can describe a backend that is no longer active.
    def _extracted() -> str:
        return _extract_gemm_shapes_from_candidates(
            str(last_trace.get("candidates_path") or ""),
            session_dir,
            precision=precision,
        )

    if canonical_dtype(precision):
        scoped = _extracted()
        if scoped:
            return scoped

    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and _is_forge_compatible_shapes_json(p):
            return str(p)

    return _extracted()


def _extract_gemm_shapes_from_candidates(candidates_path_str: str, session_dir: Path, *, precision: str = "") -> str:
    """Extract M,N,K from kernel_candidates.json hot_kernels input_shapes.

    Derives the GEMM dimensions actually observed during serving and writes a
    forge-compatible shapes JSON beside the candidates file, returning its path.

    ``precision`` scopes the result to one traced dtype. A trace records every
    GEMM the model runs, so an FP8 tuner would otherwise also receive the BF16
    router/head shapes; those rows are never looked up at serve time and they
    displace real FP8 shapes in the call-count ordering below. Empty keeps every
    dtype (historical behaviour).
    """
    import json as _json

    if not candidates_path_str:
        return ""
    cand_file = Path(candidates_path_str)
    if not cand_file.is_file():
        return ""

    try:
        data = _json.loads(cand_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    hot_kernels = data.get("hot_kernels", [])
    if not isinstance(hot_kernels, list):
        return ""

    # ``weight`` is the observed call count: decode GEMMs are invoked far more
    # often than prefill ones, so ordering by it puts the throughput-dominant
    # shapes first and they still get tuned when the tuner runs out of budget.
    # The same (M,N,K) can be reported by several kernels; keep the largest
    # count, otherwise a rare first sighting would outrank the hot one.
    weights: dict[tuple[int, int, int], int] = {}
    order: dict[tuple[int, int, int], int] = {}

    for kernel in hot_kernels:
        for key, weight in traced_gemm_shapes(kernel, precision=precision):
            if key not in order:
                order[key] = len(order)
            weights[key] = max(weights.get(key, 0), weight)

    if not weights:
        return ""

    # Most-called first; ties keep discovery order so output stays deterministic.
    ranked = sorted(weights, key=lambda key: (-weights[key], order[key]))
    shapes = [{"M": M, "N": N, "K": K} for M, N, K in ranked]

    out_path = cand_file.parent / "traced_gemm_shapes.json"
    try:
        out_path.write_text(_json.dumps(shapes, indent=2), encoding="utf-8")
    except OSError:
        return ""

    log.info(
        "extracted %d unique GEMM shapes from kernel_candidates.json -> %s",
        len(shapes),
        out_path,
    )
    return str(out_path)


# Map the resolved (precision, quant_type) to the aiter untuned-GEMM CSV the
# specialist phase records; fp8 "auto" resolves to blockscale (forge default).
_FORGE_UNTUNED_CSV_BY_QUANT: dict[str, str] = {
    "auto": "a8w8_blockscale_untuned_gemm.csv",
    "blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "block_scale": "a8w8_blockscale_untuned_gemm.csv",
    "a8w8_blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "fp8_blockscale": "a8w8_blockscale_untuned_gemm.csv",
    "per_token": "a8w8_untuned_gemm.csv",
    "per_tensor": "a8w8_untuned_gemm.csv",
    "a8w8": "a8w8_untuned_gemm.csv",
    "w8a8": "a8w8_untuned_gemm.csv",
    "w8a8_fp8": "a8w8_untuned_gemm.csv",
    "fp8_w8a8": "a8w8_untuned_gemm.csv",
    "bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "a8w8_bpreshuffle": "a8w8_bpreshuffle_untuned_gemm.csv",
    "blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "a8w8_blockscale_bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "blockscale+bpreshuffle": "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    "fp4": "a4w4_blockscale_untuned_gemm.csv",
    "mxfp4": "a4w4_blockscale_untuned_gemm.csv",
    "a4w4": "a4w4_blockscale_untuned_gemm.csv",
    "a4w4_blockscale": "a4w4_blockscale_untuned_gemm.csv",
}


def _csv_has_data_rows(path: Path) -> bool:
    """Return True when ``path`` is a CSV carrying at least one data row.

    The aiter recorder leaves header-only or empty files for quant types the
    server never exercised; those must not be passed to forge as a real shape
    source.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if "M" not in header.upper():
                return False
            for line in f:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _csv_k_values(path: Path) -> set[int]:
    """Return the distinct integer ``K`` (contraction-dim) values in a CSV.

    The aiter recorder writes a header containing ``M,N,K`` (optionally with
    extra columns such as ``q_dtype_w``). ``K`` is the GEMM contraction dim,
    which for a transformer layer equals its input dim (``hidden_size`` for
    QKV/gate-up/o projections, ``intermediate_size`` for the down projection).
    """
    ks: set[int] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split(",")
            cols = {name.strip().upper(): i for i, name in enumerate(header)}
            kidx = cols.get("K")
            if kidx is None:
                return ks
            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= kidx:
                    continue
                try:
                    ks.add(int(float(parts[kidx])))
                except ValueError:
                    continue
    except OSError:
        return ks
    return ks


def _read_model_config(model_path: str) -> dict | None:
    """Load a HF ``config.json`` as a dict; ``None`` when unavailable/unreadable."""
    if not model_path:
        return None
    # ``model_path`` may be an HF repo id; resolve to the local weights dir
    # (shared resolver) so the config read works for repo-id launches.
    from hyperloom.inference_optimizer.model_config_utils import (
        resolve_local_model_dir,
    )

    cfg = (resolve_local_model_dir(model_path) or Path(model_path)) / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _model_hidden_size(model_path: str) -> int | None:
    """Read ``hidden_size`` from a HF ``config.json``; ``None`` when unavailable."""
    data = _read_model_config(model_path)
    if data is None:
        return None
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg_dict in candidates:
        for key in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
            val = cfg_dict.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


def _resolve_fp8_quant_type(model_path: str, gpu_type: str = "", framework: str = "") -> str:
    """Pick the fp8 dense tuner quant_type from the checkpoint's static format.

    forge accepts an explicit ``quant_type``; rather than letting it fall back to
    its internal blockscale default, hand it the path the model actually runs:

    - ``blockscale_bpreshuffle`` when the checkpoint uses block-quantized
      weights AND the target GPU is gfx950 (MI355X) AND framework is sglang --
      sglang/aiter automatically upgrades blockscale to the bpreshuffle kernel
      on CDNA4. vLLM does NOT use this path (it reads
      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE).
    - ``blockscale`` when the checkpoint uses block-quantized weights on gfx942,
      on vllm, or when GPU type is unknown.
    - ``per_token`` for a plain fp16/bf16 checkpoint served under dynamic
      ``--quantization fp8`` (the a8w8 per-token path).
    - ``auto`` when ``config.json`` cannot be read, so forge sniffs the
      ``kernel_signature_log`` itself (preserves the legacy behaviour and keeps
      the no-readable-config case unchanged).
    """
    data = _read_model_config(model_path)
    if data is None:
        return "auto"
    candidates: list[dict] = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    is_blockscale = False
    for cfg_dict in candidates:
        qc = cfg_dict.get("quantization_config")
        if isinstance(qc, dict):
            if qc.get("weight_block_size"):
                is_blockscale = True
                break
            method = str(qc.get("quant_method") or qc.get("fmt") or "").lower()
            if "block" in method:
                is_blockscale = True
                break
    if is_blockscale:
        if _is_gfx950(gpu_type) and framework.lower() == "sglang":
            return "blockscale_bpreshuffle"
        return "blockscale"
    return "per_token"


def _is_gfx950(gpu_type: str) -> bool:
    """True when gpu_type resolves to gfx950 (CDNA4 / MI355X)."""
    key = (gpu_type or "").strip().lower()
    if is_gfx_arch(key, "gfx950"):
        return True
    if not key or key == "auto":
        return _is_gfx950_rocminfo()
    return False


@functools.lru_cache(maxsize=1)
def _is_gfx950_rocminfo() -> bool:
    """Cached rocminfo probe for gfx950 arch."""
    try:
        out = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
        return "gfx950" in out.lower()
    except (OSError, subprocess.SubprocessError):
        return False


def _csv_matches_model(csv_path: Path, model_path: str) -> bool:
    """Return True when an untuned CSV plausibly belongs to ``model_path``.

    A real per-model dense untuned CSV always contains GEMMs whose ``K`` equals
    the model ``hidden_size``. When ``hidden_size`` is known and absent from the
    CSV's ``K`` column, the CSV was recorded for a different model and is
    rejected so forge derives shapes from the model config instead.

    Returns True when validation is not possible (``hidden_size`` unreadable or
    the CSV exposes no ``K`` column) to avoid false rejections.
    """
    hidden = _model_hidden_size(model_path)
    if hidden is None:
        return True
    k_values = _csv_k_values(csv_path)
    if not k_values:
        return True
    return hidden in k_values


def _resolve_forge_untuned_csv(session_dir: Path, precision: str, quant_type: str, model_path: str = "") -> str:
    """Find an aiter untuned-GEMM CSV in a specialist worktree.

    Specialist runs may materialize or modify these files under
    ``runs/specialist/<hash>/worktree/aiter/configs/*_untuned_gemm.csv``; this
    resolver picks the newest non-empty CSV matching the resolved quant type.
    Because an unchanged checkout can also contain static upstream rows, this is
    a fallback behind explicit benchmark input and the latest runtime profile.

    When ``model_path`` is given, candidate CSVs whose GEMM shapes do not match
    the model are rejected so forge derives per-model shapes from ``config.json``.
    Returns the CSV path, or "" when none is available.
    """
    precision = (precision or "").strip().lower()
    quant_type = (quant_type or "").strip().lower()

    fname = _FORGE_UNTUNED_CSV_BY_QUANT.get(quant_type)
    if fname is None:
        log.warning(
            "Forge GEMM shapes: unknown quant_type=%r for precision=%r; not guessing an untuned CSV",
            quant_type,
            precision,
        )
        return ""

    from hyperloom.inference_optimizer.session.session_paths import runs_root

    specialist_dir = runs_root(session_dir) / "specialist"
    if not specialist_dir.is_dir():
        return ""

    best: Path | None = None
    best_mtime = -1.0
    for csv_path in specialist_dir.glob(f"*/worktree/aiter/configs/{fname}"):
        if not _csv_has_data_rows(csv_path):
            continue
        if not _csv_matches_model(csv_path, model_path):
            continue
        try:
            mtime = csv_path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = csv_path

    return str(best) if best is not None else ""


def _path_is_existing_file(value: str) -> bool:
    """Safe ``Path.is_file()`` that never raises on an over-long pathname.

    A caller may hand us inline JSON content instead of a path; ``is_file()``
    raises ``OSError(ENAMETOOLONG)`` on such input. Treat any OSError as
    "not a file".
    """
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _normalize_tokens(value: Any) -> str:
    """Return a clean comma-separated token string for forge's ``--tokens``.

    forge parses ``--tokens`` as ``int(t) for t in value.split(",")``, so accept
    lists and bracketed strings and emit a bare comma-separated list.
    """
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip().strip("[](){}")
        if not text:
            return ""
        items = list(text.split(","))
    out: list[str] = []
    for it in items:
        s = str(it).strip().strip("'\"")
        if not s:
            continue
        try:
            out.append(str(int(float(s))))
        except (TypeError, ValueError):
            continue
    return ",".join(out)


def _normalize_forge_shapes_json(value: Any, workspace: Path) -> str:
    """Return a usable shapes-JSON *file path*, materializing inline content.

    Callers sometimes pass GEMM shapes as inline JSON in ``shapes_json`` instead
    of a file path; forge treats it strictly as a path. Normalize here:

    - existing file path -> returned unchanged
    - list/dict, or a string that parses as JSON -> written to
      ``<workspace>/forge_shapes.json`` and that path returned
    - anything else (empty / unparseable / non-existent path) -> ""
    """
    if value in (None, ""):
        return ""

    # Already-parsed inline content.
    if isinstance(value, (list, dict)):
        parsed: Any = value
    else:
        text = str(value).strip()
        if not text:
            return ""
        if _path_is_existing_file(text):
            return text
        # Inline JSON content (possibly Python-repr with single quotes).
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                try:
                    import ast

                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return ""
        else:
            # Non-JSON string that is not an existing file.
            return ""

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        out = workspace / "forge_shapes.json"
        out.write_text(json.dumps(parsed), encoding="utf-8")
        return str(out)
    except (OSError, TypeError, ValueError):
        return ""


# Forge tuner families whose deliverable is an aiter tuned-GEMM CSV, i.e. the
# ones whose rows are resolved through aiter's padded (M, N, K) lookup.
_AITER_CSV_TUNER_FRAMEWORKS = ("sglang", "vllm-aiter")


#: Wall-clock the aiter CK tuner needs per shape, measured on gfx950 at
#: ``--mp 1`` (12 shapes / 1462s in production, ~135s each). Used to size the
#: shape budget so a wider ladder cannot push the tuner past its timeout and
#: return nothing at all.
_AITER_TUNE_SEC_PER_SHAPE = 150


def _align_forge_shapes_for_aiter(
    shapes_json: str,
    *,
    forge_framework: str,
    workspace: Path,
    budget_sec: int = 0,
    mp: int = 1,
) -> tuple[str, dict[str, Any] | None]:
    """Re-key profiled GEMM shapes onto the M values aiter actually looks up.

    Captured shapes carry the raw runtime M, which for prefill is the
    data-dependent scheduled-token count and so never recurs between runs. aiter
    resolves a tuned row by trying the raw M and then two padded M variants, so a
    CSV keyed on raw M is unreachable and the tuner's micro win never reaches the
    server. Padding the shapes first makes each tuned row serve the whole bucket
    that pads onto it.

    Returns the shapes-JSON path to hand forge plus an alignment report, or the
    input path and ``None`` when alignment does not apply.
    """
    if forge_framework not in _AITER_CSV_TUNER_FRAMEWORKS:
        return shapes_json, None
    if not env_bool("HYPERLOOM_GEMM_ALIGN_SHAPES", True):
        return shapes_json, None

    from .gemm_shape_coverage import align_shapes_to_aiter_keys, load_shapes_json, write_shapes_json

    observed = load_shapes_json(shapes_json)
    if not observed:
        return shapes_json, None
    try:
        max_shapes = int(os.environ.get("HYPERLOOM_GEMM_ALIGN_MAX_SHAPES") or 64)
    except (TypeError, ValueError):
        max_shapes = 64
    max_shapes = max(1, max_shapes)
    if budget_sec > 0:
        try:
            per_shape = int(os.environ.get("HYPERLOOM_GEMM_TUNE_SEC_PER_SHAPE") or _AITER_TUNE_SEC_PER_SHAPE)
        except (TypeError, ValueError):
            per_shape = _AITER_TUNE_SEC_PER_SHAPE
        # Reserve a third of the window for JIT builds and the report step.
        affordable = int(budget_sec * 0.66 * max(1, mp) // max(1, per_shape))
        max_shapes = min(max_shapes, max(len(observed), affordable))
    aligned, report = align_shapes_to_aiter_keys(observed, max_shapes=max_shapes)
    if not aligned or report.get("unchanged"):
        return shapes_json, {**report, "applied": False, "source_shapes_json": shapes_json}
    try:
        out = write_shapes_json(aligned, workspace / "forge_shapes.aiter_aligned.json")
    except OSError:
        return shapes_json, {**report, "applied": False, "source_shapes_json": shapes_json}
    log.info(
        "Forge GEMM shapes: re-keyed %d observed shape(s) onto %d aiter lookup key(s) (observed M=%s -> aligned M=%s)",
        report.get("observed"),
        report.get("aligned"),
        report.get("observed_m"),
        report.get("aligned_m"),
    )
    return out, {**report, "applied": True, "source_shapes_json": shapes_json}


#: Markers that identify which kernels aiter is serving, read off a server log.
#: ``bf16_tuned_gemm.csv`` means dense linears resolve through
#: ``aiter/tuned_gemm.py`` (Forge's ``sglang_dense_bf16`` writes that table via
#: ``AITER_CONFIG_GEMM_BF16``); the fused-MoE markers mean the MoE layers run on
#: aiter's CK kernels (Forge's ``fmoe_ck``, via ``AITER_CONFIG_FMOE``) rather
#: than vLLM's Triton ``fused_moe``.
_AITER_SERVING_MARKERS = {
    "bf16_dense": ("bf16_tuned_gemm.csv",),
    "fused_moe": ("[aiter] [fused_moe]", "Mxfp4 MoE backend"),
}

#: Which dtypes fall in each of aiter's width buckets. Mirrors ``bit16_list`` /
#: ``bit8_list`` / ``bit4_list`` in
#: ``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py``.
_AITER_BIT16_DTYPES = frozenset({"bfloat16", "float16"})
_AITER_BIT8_DTYPES = frozenset({"float8_e4m3fn", "float8_e4m3fnuz", "int8"})
_AITER_BIT4_DTYPES = frozenset({"float4_e2m1fn_x2", "uint32", "int4"})


def _aiter_moe_dtype_pair_supported(q_dtype_a: str, q_dtype_w: str) -> bool:
    """Return whether aiter's CK MoE codegen has a kernel family for this pair.

    ``get_gemm1_kernels_list`` / ``get_gemm2_kernels_list`` pick a family from the
    activation/weight widths and raise ``Unsupported data type combination`` for
    anything else. Notably a BF16 activation against FP4 weights -- which the
    serving path runs happily -- matches no family, so handing it to the tuner
    trades a silent no-op for a hard error.
    """
    act = q_dtype_a.replace("torch.", "")
    weight = q_dtype_w.replace("torch.", "")
    if act in _AITER_BIT16_DTYPES and weight in _AITER_BIT16_DTYPES:
        return True
    if act in _AITER_BIT8_DTYPES and weight in _AITER_BIT8_DTYPES:
        return True
    # The a8w4 family is FP8-only on the activation side; INT8 does not qualify.
    if act.startswith("float8") and weight in _AITER_BIT4_DTYPES:
        return True
    return act in _AITER_BIT4_DTYPES and weight in _AITER_BIT4_DTYPES


def _aiter_fused_moe_dispatch_keys(server_log: str) -> list[dict[str, Any]]:
    """Return the distinct MoE problems a server log shows aiter dispatching.

    Deduplicated on everything but the token count, most-missed first. One model
    routinely yields several problems -- the same checkpoint dispatches both a
    BF16-activation and an FP8-activation variant, and the EP path appends a
    masked fake-expert slot so ``expert``/``topk`` arrive one higher than the
    model config states. Neither is derivable from the config, which is why the
    log is the authoritative source for what to tune.
    """
    if not server_log:
        return []
    from kernelforge.gemm_tune.evidence import moe_dispatch_keys, parse_log_file

    return moe_dispatch_keys(parse_log_file(server_log))


def _aiter_ck_moe_tuner_supports(server_log: str) -> bool:
    """Return whether aiter's CK MoE tuner can tune anything the server dispatched.

    The tuner builds its kernel candidates from the activation/weight dtype pair
    and rejects some combinations the serving path happily runs. Measured on
    gpt-oss-120b at TP=1, a BF16-activation / FP4-weight MoE (the
    ``AITER_MXFP4_BF16`` backend) benchmarks fine but fails candidate generation
    with ``Unsupported data type combination: b16, fp4x2``, so routing it to
    ``fmoe_ck`` would only trade silent no-op for a hard tuner error.

    A single checkpoint can dispatch several dtype pairs at once, so this asks
    whether *any* of them is tunable; per-problem filtering happens where the
    tuning input is written.
    """
    if not server_log:
        return False
    keys = _aiter_fused_moe_dispatch_keys(server_log)
    if not keys:
        # MoE evidence without a parseable problem tuple: let Forge decide.
        return True
    return any(_aiter_moe_dtype_pair_supported(key["q_dtype_a"], key["q_dtype_w"]) for key in keys)


#: forge wordings that carry nothing deployable, each distinct from an honest
#: ``no_improvement``. ``build_report`` checks ``has_candidate`` first, so a run
#: holding a usable env reports ``candidate`` even when a sibling crashed --
#: these arrive only with nothing to deploy.
_FORGE_BARREN_MICRO_DECISIONS = (
    "failed",
    "empty_output",
    "partial_failure",
    "partial_output",
)


def _fmoe_token_list(tokens: Any) -> list[int]:
    """Positive token counts to sweep, keyed off whatever the caller sends.

    Accepts forge's comma-separated string (what :func:`_normalize_tokens`
    produces) or a sequence. Unparseable and non-positive entries are dropped
    rather than raising -- one bad entry is not worth the run -- and ``[1]`` is
    the floor so there is always a token to key on.
    """
    if isinstance(tokens, str):
        raw: list[str] = [part.strip() for part in tokens.split(",")]
    elif isinstance(tokens, (list, tuple, set, frozenset)):
        raw = [str(item).strip() for item in tokens]
    elif tokens is None:
        raw = []
    else:
        raw = [str(tokens).strip()]

    out: set[int] = set()
    for item in raw:
        if not item:
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.add(value)
    return sorted(out) or [1]


def _write_fmoe_untuned_csv_from_log(
    server_log: str,
    tokens: Any,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    """Turn the MoE problems observed in ``server_log`` into a tuning input CSV.

    Returns ``(csv_path, report)``; ``csv_path`` is "" when nothing tunable was
    observed. Writing the observed tuple verbatim is the whole point: the
    quantisation pair, the per-partition ``inter_dim`` and the EP-inflated
    expert/topk counts are all properties of what the serving framework chose,
    and every attempt to re-derive them from the model config is a guess that has
    already produced tables no runtime lookup could reach.

    Problems whose dtype pair aiter's codegen rejects are dropped rather than
    passed through, because one unsupported row aborts the whole tuner run.
    """
    from kernelforge.gemm_tune.evidence import MOE_SHAPE_FIELDS, moe_untuned_csv_text

    report: dict[str, Any] = {
        "observed": 0,
        "tunable": 0,
        "dropped_unsupported": [],
        "keys": [],
    }
    keys = _aiter_fused_moe_dispatch_keys(server_log)
    report["observed"] = len(keys)
    if not keys:
        return "", report

    tunable: list[dict[str, Any]] = []
    for key in keys:
        pair = (key["q_dtype_a"], key["q_dtype_w"])
        if _aiter_moe_dtype_pair_supported(*pair):
            tunable.append(key)
            report["keys"].append({name: key[name] for name in MOE_SHAPE_FIELDS})
        else:
            combo = f"{pair[0]}/{pair[1]}"
            if combo not in report["dropped_unsupported"]:
                report["dropped_unsupported"].append(combo)
    report["tunable"] = len(tunable)
    if not tunable:
        return "", report

    token_list = _fmoe_token_list(tokens)
    rendered = [moe_untuned_csv_text(key, tokens=token_list).splitlines() for key in tunable]
    lines = rendered[0][:1] + [row for key_lines in rendered for row in key_lines[1:]]

    csv_path = workspace / "untuned_fmoe_from_runtime.csv"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        # A full disk or a read-only workspace must cost the MoE tuner its input,
        # not the whole tuning run: the dense tuners take their shapes from
        # elsewhere and can still produce something useful.
        report["write_error"] = f"{type(exc).__name__}: {exc}"
        log.warning("Forge GEMM shapes: cannot write %s: %s", csv_path, exc)
        return "", report
    log.info(
        "Forge GEMM shapes: derived %d MoE problem(s) x %d token(s) from %s%s",
        len(tunable),
        len(token_list),
        server_log,
        (f"; dropped {report['dropped_unsupported']} as untunable by aiter" if report["dropped_unsupported"] else ""),
    )
    return str(csv_path), report


def _aiter_serving_evidence(server_log: str) -> set[str]:
    """Return which aiter kernel families a server log shows in use.

    Routing is driven by the log rather than by precision alone because only the
    log says which backend the model actually got: the same checkpoint runs on
    aiter or on the native path depending on the recipe's env.
    """
    found: set[str] = set()
    if not server_log:
        return found
    try:
        with open(server_log, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for family, markers in _AITER_SERVING_MARKERS.items():
                    if family not in found and any(marker in line for marker in markers):
                        found.add(family)
                if len(found) == len(_AITER_SERVING_MARKERS):
                    break
    except OSError:
        return set()
    return found


def _forge_framework_for_vllm(
    *,
    framework: str,
    precision: str,
    quant_type: str,
    tunableop_input: str,
    aiter_bf16_dense: bool = False,
    aiter_fused_moe: bool = False,
) -> str:
    """Route vLLM runs served by aiter to Forge's AITER tuner family.

    Forge's vLLM branch only offers ``vllm_moe_triton`` and
    ``vllm_dense_tunableop``, which target kernels an aiter-served model never
    executes. Its sglang branch carries the tuners that do write the tables aiter
    reads, and the router already accepts ``vllm-aiter`` as an alias for it.
    """
    if framework != "vllm" or tunableop_input:
        return framework
    if _is_vllm_block_fp8(precision, quant_type):
        return "vllm-aiter"
    if aiter_bf16_dense and precision in ("bf16", "fp16"):
        return "vllm-aiter"
    if aiter_fused_moe:
        return "vllm-aiter"
    return framework


def _resolve_vllm_aiter_routing(
    *,
    model_path: str,
    server_log: str,
    tp: int,
) -> dict[str, bool]:
    """Resolve the aiter-routing flags for a vLLM run from runtime evidence."""
    flags = {"aiter_bf16_dense": False, "aiter_fused_moe": False}
    evidence = _aiter_serving_evidence(server_log)
    if not evidence:
        return flags

    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(model_path) or {}
    if not summary:
        return flags
    is_moe = bool(summary.get("is_moe"))

    # Dense BF16 routing is for dense checkpoints; a MoE model's dense side
    # rides along with its MoE routing instead.
    flags["aiter_bf16_dense"] = "bf16_dense" in evidence and not is_moe

    if "fused_moe" in evidence and is_moe and _aiter_ck_moe_tuner_supports(server_log):
        # Only route MoE when aiter's CK fused-MoE can actually serve this
        # checkpoint at this TP -- otherwise the tuner has no reachable target.
        from hyperloom.inference_optimizer.model_config_utils import (
            model_supports_aiter_ck_fused_moe,
        )

        flags["aiter_fused_moe"] = model_supports_aiter_ck_fused_moe(model_path, tp)

    _warn_if_moe_routing_is_coarser_than_the_log(server_log, flags)
    return flags


def _warn_if_moe_routing_is_coarser_than_the_log(server_log: str, flags: dict[str, bool]) -> None:
    """Say so when one log shows both MoE backends and routing picks one.

    The decision above is a substring scan: seeing an aiter fused-MoE marker
    anywhere routes the whole run to the aiter tuner family, and
    ``vllm_moe_triton`` then never runs. A run can dispatch both -- aiter CK over
    part of the token range and vLLM's Triton path over the rest -- and forge's
    own parser records exactly that as ``impl="mixed"``. Whichever way the single
    flag falls, the range served by the other backend is left untuned.

    Reported rather than acted on here: changing this routing changes which
    tuners run for every aiter-served vLLM model, which is a bigger step than
    the tuner-side addition that already covers the CK half. Forge adds
    ``fmoe_ck`` from the same evidence, so the gap this warns about is the
    Triton half.
    """
    if not flags.get("aiter_fused_moe"):
        return
    try:
        from kernelforge.gemm_tune.evidence import parse_log_file
    except ImportError:
        # Same reasoning as apply_verification._parse: kernelforge is in this
        # wheel, so a miss is a broken install, and a bare return makes the
        # missing routing warning indistinguishable from a clean run.
        log.warning(
            "kernelforge.gemm_tune is not importable, so the aiter/vLLM MoE "
            "routing check is skipped -- it ships with Hyperloom, so this means "
            'an incomplete install; reinstall with pip install -e ".[forge]"'
        )
        return
    moe = (parse_log_file(server_log).get("dispatch") or {}).get("moe") or {}
    if moe.get("impl") == "mixed" or moe.get("vllm_config_hit"):
        log.warning(
            "gemm routing: %s shows both aiter CK and vLLM Triton MoE dispatch "
            "(impl=%s, stages=%s); routing sends the whole run to the aiter "
            "tuner family, so the token range Triton serves goes untuned",
            server_log,
            moe.get("impl"),
            moe.get("stages_seen"),
        )


async def _served_capture_workload(
    executor: Any,
    run: _ShapeCaptureRun,
    capture_state: Any,
    task_params: dict[str, Any],
) -> tuple[dict[str, Any], HandlerResult | None]:
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    ctx = RunnerContext(
        task=Task(
            task_id=run.task_id,
            kind="gemm_shape_capture",
            state="running",
            params=task_params,
            idempotency_key=f"{run.task_id}-run",
        ),
        lease=None,
    )
    ctx.extra = {
        "shared_state": capture_state,
        "session_dir": run.session_dir,
        "workspace": run.capture_dir,
    }
    try:
        benchmark_result = await executor(ctx)
    except Exception as exc:  # noqa: BLE001 - convert capture launch faults to a stable result
        return {}, _shape_capture_failure(f"vLLM TunableOp shape capture raised {exc!r}", run.capture_dir)
    return (benchmark_result if isinstance(benchmark_result, dict) else {}), None


async def _capture_block_fp8_profile_shapes(run: _ShapeCaptureRun, state: Any, payload: dict) -> HandlerResult:
    from ..actions.executors.roofline import RooflineExecutor

    task_params = _capture_task_params(run, state, payload)
    task_params["workspace_path"] = str(run.capture_dir / "tracelens")
    last_baseline = getattr(state, "last_baseline", None)
    if isinstance(last_baseline, dict):
        benchmark_script = str(last_baseline.get("benchmark_script") or "").strip()
        if benchmark_script:
            task_params["benchmark_script"] = benchmark_script
    benchmark_result, fault = await _served_capture_workload(
        RooflineExecutor(shared_state=state), run, state, task_params
    )
    if fault is not None:
        return fault

    steady_state_trace = str(benchmark_result.get("steady_state_trace") or "").strip()
    shapes_json, shape_count = "", 0
    if steady_state_trace:
        shapes_json, shape_count = _extract_vllm_block_fp8_profile_shapes(
            Path(steady_state_trace),
            output_dir=run.capture_dir,
        )
    if benchmark_result.get("status") == "succeeded" and shape_count > 0:
        return {
            "status": "ok",
            "shapes_json": shapes_json,
            "shape_capture_workspace": str(run.capture_dir),
            "shape_count": shape_count,
            "capture_mode": "block_fp8_profile",
            "source_profile_trace": steady_state_trace,
        }
    return _shape_capture_failure(
        f"vLLM block-FP8 profile capture produced no structured GEMM shapes{_benchmark_failure_detail(benchmark_result)}",
        run.capture_dir,
        shape_count=shape_count,
        capture_mode="block_fp8_profile",
    )


async def _capture_tunableop_recording(run: _ShapeCaptureRun, state: Any, payload: dict) -> HandlerResult:
    import copy

    from ..actions.executors.baseline import BaselineExecutor

    config_path = str(payload.get("config_path") or getattr(state, "baseline_config_path", "") or "").strip()
    if not config_path or not Path(config_path).is_file():
        return _shape_capture_failure(
            "vLLM TunableOp shape capture requires an existing baseline_config_path",
            run.capture_dir,
        )
    try:
        capture_port = _resolve_shape_capture_port(payload.get("shape_capture_port"))
    except ValueError as exc:
        return _shape_capture_failure(str(exc), run.capture_dir)

    try:
        timeout_sec = int(
            payload.get("shape_capture_timeout_sec")
            or os.environ.get("HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC")
            or _DEFAULT_CAPTURE_TIMEOUT_SEC
        )
    except (TypeError, ValueError):
        timeout_sec = _DEFAULT_CAPTURE_TIMEOUT_SEC

    # unset_envs is applied after extra_envs, so any PYTORCH_TUNABLEOP_* name left here strips the recording.
    kept_unset = [name for name in run.unset_envs if not name.startswith("PYTORCH_TUNABLEOP_")]
    task_params = _capture_task_params(run, state, payload)
    task_params.update(
        {
            "extra_envs": _tunableop_recording_envs(run.envs, capture_port, run.capture_dir),
            "unset_envs": [*kept_unset, "HL_TUNABLEOP_MODE", "HL_TUNABLEOP_FILE", "HL_TUNABLEOP_VERBOSE"],
            "remove_args": [*run.remove_args, "--port"],
            "config_path": config_path,
            "timeout_sec": max(60, timeout_sec),
            "baseline_double_run": False,
        }
    )
    capture_state = copy.deepcopy(state)
    capture_state.baseline_eager_fallback = False
    benchmark_result, fault = await _served_capture_workload(
        BaselineExecutor(session_dir=run.session_dir, shared_state=capture_state),
        run,
        capture_state,
        task_params,
    )
    if fault is not None:
        return fault

    untuned_base = run.capture_dir / _TUNABLEOP_UNTUNED_NAME
    row_count = _merge_tunableop_untuned_files(untuned_base)
    if benchmark_result.get("status") == "succeeded" and row_count > 0:
        return {
            "status": "ok",
            "tunableop_input": str(untuned_base),
            "shape_capture_workspace": str(run.capture_dir),
            "shape_count": row_count,
        }
    try:
        untuned_base.unlink(missing_ok=True)
    except OSError:
        log.debug("shape capture: failed to remove incomplete TunableOp recording %s", untuned_base, exc_info=True)
    return _shape_capture_failure(
        f"vLLM TunableOp shape capture produced no complete workload recording"
        f"{_benchmark_failure_detail(benchmark_result)}",
        run.capture_dir,
        shape_count=row_count,
    )


#: Filenames PyTorch TunableOp reads and writes inside one capture directory.
_TUNABLEOP_UNTUNED_NAME = "tunableop_untuned.csv"


_TUNABLEOP_RESULTS_NAME = "tunableop_results.csv"


#: How long a recording pass may serve the workload when nothing else says.
_DEFAULT_CAPTURE_TIMEOUT_SEC = 1800


#: The vLLM operators whose profiled arguments carry a block-FP8 GEMM's shape.
_VLLM_BLOCK_FP8_TRACE_OPS = (
    "w8a8_triton_block_scaled_mm",
    "rocm_aiter_gemm_a8w8_blockscale",
    "rocm_aiter_triton_gemm_a8w8_blockscale",
)


def _is_vllm_block_fp8(precision: str, quant_type: str) -> bool:
    """Return whether vLLM runs the block-scaled FP8 linear kernel path."""
    return precision == "fp8" and quant_type.strip().lower() in {
        "blockscale",
        "block_scale",
        "a8w8_blockscale",
        "fp8_blockscale",
    }


def _vllm_block_fp8_profile_capture_required(
    *,
    framework: str,
    precision: str,
    quant_type: str,
    shapes_json: str,
    tunableop_input: str,
    dry_run: bool,
) -> bool:
    """Return whether block-FP8 needs a profiled runtime-shape capture pass."""
    if (
        framework != "vllm"
        or not _is_vllm_block_fp8(precision, quant_type)
        or dry_run
        or shapes_json
        or tunableop_input
        or not env_bool("HYPERLOOM_GEMM_SHAPE_CAPTURE", True)
    ):
        return False
    from ..actions.executors._multi_node_env import is_multi_node

    return not is_multi_node()


def _trace_event_block_fp8_shape(event: Any) -> tuple[int, int, int] | None:
    """Extract one (M, N, K) tuple from a profiled block-FP8 linear event."""
    if not isinstance(event, dict):
        return None
    name = str(event.get("name") or "").lower()
    if not any(marker in name for marker in _VLLM_BLOCK_FP8_TRACE_OPS):
        return None
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    dims = args.get("Input Dims") or args.get("Input dims") or args.get("input_shapes")
    if not isinstance(dims, list) or len(dims) < 2:
        return None
    a_dims, b_dims = dims[0], dims[1]
    if not isinstance(a_dims, list) or not isinstance(b_dims, list) or len(a_dims) < 2 or len(b_dims) < 2:
        return None
    try:
        m = int(a_dims[-2])
        k = int(a_dims[-1])
        if int(b_dims[-1]) == k:
            n = int(b_dims[-2])
        elif int(b_dims[-2]) == k:
            n = int(b_dims[-1])
        else:
            return None
    except (TypeError, ValueError):
        return None
    return (m, n, k) if min(m, n, k) > 0 else None


def _extract_vllm_block_fp8_profile_shapes(
    trace_input: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[str, int]:
    """Convert Kineto block-FP8 events into Forge's structured shapes JSON."""
    import gzip

    def _is_capture_sidecar(path: Path) -> bool:
        # Shared classifier, so a layout the kernel-agent routes demote is also
        # kept out of the shape harvest here; the exact-``capture_traces`` test
        # this replaced missed ``graph_capture_profile/``.
        return _shared_is_capture_fragment(path, trace_input if trace_input.is_dir() else trace_input.parent)

    shapes: set[tuple[int, int, int]] = set()
    # ``Path("")`` normalizes to ``Path(".")``, which would otherwise walk the
    # whole process CWD and harvest shapes from unrelated traces.
    if str(trace_input) in ("", "."):
        return "", 0
    if trace_input.is_file():
        trace_paths = [] if _is_capture_sidecar(trace_input) else [trace_input]
    elif trace_input.is_dir():
        trace_paths = [
            path
            for path in sorted(trace_input.rglob("*.json")) + sorted(trace_input.rglob("*.json.gz"))
            if not _is_capture_sidecar(path)
        ]
    else:
        return "", 0
    for path in trace_paths:
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
                    data = json.load(stream)
            else:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        events = data.get("traceEvents") if isinstance(data, dict) else None
        if not isinstance(events, list):
            continue
        for event in events:
            shape = _trace_event_block_fp8_shape(event)
            if shape is not None:
                shapes.add(shape)
    if not shapes:
        return "", 0
    destination = output_dir or (trace_input if trace_input.is_dir() else trace_input.parent)
    out = destination / "forge_shapes.json"
    payload = [{"M": m, "N": n, "K": k} for m, n, k in sorted(shapes)]
    try:
        destination.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return "", 0
    return str(out), len(payload)


def _reuse_vllm_block_fp8_roofline_shapes(
    state: Any,
    *,
    workspace: Path,
    current_workload: dict[str, Any] | None = None,
) -> HandlerResult | None:
    """Reuse block-FP8 runtime shapes from the latest Roofline profile trace."""
    last_trace_analyze = getattr(state, "last_trace_analyze", None)
    if not isinstance(last_trace_analyze, dict):
        return None
    source_trace = str(last_trace_analyze.get("steady_state_trace") or "").strip()
    if not source_trace:
        log.info(
            "vLLM block-FP8 shape capture: latest Roofline has no selected "
            "steady-state trace; running a standard Roofline fallback"
        )
        return None
    profile_trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    analyzed_trace = str(last_trace_analyze.get("trace_input") or "").strip()
    try:
        profile_trace_id = str(Path(profile_trace).expanduser().resolve(strict=False))
        analyzed_trace_id = str(Path(analyzed_trace).expanduser().resolve(strict=False))
    except OSError:
        profile_trace_id = profile_trace
        analyzed_trace_id = analyzed_trace
    if not profile_trace or not analyzed_trace or profile_trace_id != analyzed_trace_id:
        log.info(
            "vLLM block-FP8 shape capture: steady-state trace provenance does "
            "not match the latest profile; running a standard Roofline fallback"
        )
        return None
    if str(getattr(state, "last_profile_status", "") or "").strip().lower() != "succeeded":
        log.info(
            "vLLM block-FP8 shape capture: latest Roofline profile is not successful; "
            "running a standard Roofline fallback"
        )
        return None
    profile_workload = getattr(state, "last_profile_workload", None)
    expected_workload = current_workload or state.current_profile_workload_context()
    recorded_workload = profile_workload if isinstance(profile_workload, dict) else {}
    if recorded_workload != expected_workload:
        mismatches = sorted(
            key
            for key in set(recorded_workload) | set(expected_workload)
            if recorded_workload.get(key) != expected_workload.get(key)
        )
        log.info(
            "vLLM block-FP8 shape capture: Roofline workload mismatch (%s); running a standard Roofline fallback",
            ", ".join(mismatches) or "missing profile workload metadata",
        )
        return None
    shapes_json, shape_count = _extract_vllm_block_fp8_profile_shapes(
        Path(source_trace),
        output_dir=workspace,
    )
    if shape_count == 0:
        log.info(
            "vLLM block-FP8 shape capture: Roofline trace %s contains no reusable "
            "block-FP8 shapes; running a standard Roofline fallback",
            source_trace,
        )
        return None
    log.info(
        "vLLM block-FP8 shape capture: reusing %d shape(s) from Roofline trace %s",
        shape_count,
        source_trace,
    )
    return {
        "status": "ok",
        "shapes_json": shapes_json,
        "shape_capture_workspace": str(workspace),
        "shape_count": shape_count,
        "capture_mode": "roofline_profile_reuse",
        "source_profile_trace": source_trace,
    }


def _vllm_dense_shape_capture_required(
    *,
    framework: str,
    model_path: str,
    shapes_json: str,
    tunableop_input: str,
    dry_run: bool,
) -> bool:
    """Return whether Forge needs an automatic TunableOp recording pass."""
    if (
        framework != "vllm"
        or dry_run
        or shapes_json
        or tunableop_input
        or not env_bool("HYPERLOOM_GEMM_SHAPE_CAPTURE", True)
    ):
        return False
    from ..actions.executors._multi_node_env import is_multi_node

    if is_multi_node():
        return False

    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(model_path)
    if not summary or bool(summary.get("is_moe")):
        return False
    try:
        hidden_size = int(summary.get("hidden_size") or 0)
        intermediate_size = int(summary.get("intermediate_size") or 0)
    except (TypeError, ValueError):
        return False
    return hidden_size > 0 and intermediate_size > 0


def _pick_shape_capture_port() -> int:
    """Pick a free local port distinct from the production serving port."""
    import socket

    for _ in range(5):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port != 8888:
            return port
    return 18888


def _resolve_shape_capture_port(value: Any) -> int:
    """Resolve an isolated capture port and reject the production port."""
    if value in (None, ""):
        return _pick_shape_capture_port()
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid shape_capture_port: {value!r}") from exc
    if port <= 0 or port > 65535 or port == 8888:
        raise ValueError(f"shape_capture_port must be 1..65535 and not 8888: {port}")
    return port


def _is_tunableop_untuned_row(line: str) -> bool:
    """Recognize a native PyTorch TunableOp offline-input row."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "Validator")):
        return False
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) < 2 or "TunableOp" not in fields[0] or not fields[1]:
        return False
    dimensions = [int(value) for value in re.findall(r"\d+", fields[1])]
    return sum(value > 0 for value in dimensions) >= 3


def _merge_tunableop_untuned_files(base_path: Path) -> int:
    """Merge per-device TunableOp recordings into one deterministic input."""
    rows: list[str] = []
    seen: set[str] = set()
    pattern = f"{base_path.stem}*{base_path.suffix}"
    for path in sorted(base_path.parent.glob(pattern)):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            row = line.strip()
            if _is_tunableop_untuned_row(row) and row not in seen:
                seen.add(row)
                rows.append(row)
    if not rows:
        return 0

    tmp_path = base_path.with_suffix(f"{base_path.suffix}.tmp")
    try:
        tmp_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        os.replace(tmp_path, base_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            log.debug(
                "shape capture: failed to remove temporary TunableOp file %s",
                tmp_path,
                exc_info=True,
            )
        return 0
    return len(rows)


class _ShapeCaptureRun(NamedTuple):
    """One shape-capture pass: where it records, and the recipe it serves the workload under."""

    capture_dir: Path
    session_dir: Path
    task_id: str
    envs: dict[str, str]
    server_args: str
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str


def _shape_capture_failure(
    error: str,
    capture_dir: Path,
    *,
    shape_count: int | None = None,
    capture_mode: str = "",
) -> HandlerResult:
    failure: HandlerResult = {
        "status": "failed",
        "decision": "REVERT",
        "requires_e2e_validation": False,
        "error_class": "shape_capture_failed",
        "error": error,
        "shape_capture_workspace": str(capture_dir),
    }
    if shape_count is not None:
        failure["shape_count"] = shape_count
    if capture_mode:
        failure["capture_mode"] = capture_mode
    return failure


def _benchmark_failure_detail(benchmark_result: Mapping[str, Any]) -> str:
    error = str(benchmark_result.get("error") or benchmark_result.get("error_class") or "").strip()
    return f": {error}" if error else ""


def _fill_capture_workload_envs(envs: dict[str, str], state: Any, payload: dict) -> None:
    for env_name, state_name in (
        ("TP", "tp"),
        ("CONC", "conc"),
        ("ISL", "isl"),
        ("OSL", "osl"),
        ("MAX_MODEL_LEN", "max_model_len"),
    ):
        value = payload.get(state_name)
        if value in (None, ""):
            value = envs.get(env_name)
        if value in (None, ""):
            value = getattr(state, state_name, 0)
        try:
            resolved = int(value or 0)
        except (TypeError, ValueError):
            resolved = 0
        if resolved > 0:
            envs[env_name] = str(resolved)


def _shape_capture_run(state: Any, payload: dict, workspace: Path, session_dir: Path) -> _ShapeCaptureRun:
    capture_dir = workspace / "shape_capture" / f"attempt-{time.time_ns()}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    current_best = getattr(state, "current_best", None) or {}

    inherited_envs = dict(current_best.get("extra_envs") or {})
    inherited_envs.update(dict(payload.get("extra_envs") or {}))
    envs = {str(key): str(value) for key, value in inherited_envs.items()}
    _fill_capture_workload_envs(envs, state, payload)

    inherited_unset = payload.get("unset_envs", current_best.get("unset_envs")) or []
    inherited_remove = payload.get("remove_args", current_best.get("remove_args")) or []
    return _ShapeCaptureRun(
        capture_dir=capture_dir,
        session_dir=session_dir,
        task_id=f"{str(payload.get('task_id') or workspace.name)}-shape-capture",
        envs=envs,
        server_args=(
            str(payload.get("extra_server_args") or "")
            if "extra_server_args" in payload
            else str(current_best.get("extra_server_args") or "")
        ),
        remove_args=[inherited_remove] if isinstance(inherited_remove, str) else [str(arg) for arg in inherited_remove],
        unset_envs=[inherited_unset] if isinstance(inherited_unset, str) else [str(key) for key in inherited_unset],
        args_mode=str(payload.get("args_mode") or current_best.get("args_mode") or "append"),
    )


def _capture_task_params(run: _ShapeCaptureRun, state: Any, payload: dict) -> dict[str, Any]:
    from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id

    return {
        "output_dir": str(run.capture_dir),
        "framework": "vllm",
        "model_path": str(payload.get("model_path") or getattr(state, "model_path", "") or ""),
        "gpu_type": str(payload.get("gpu_type") or getattr(state, "gpu_type", "") or ""),
        "extra_server_args": run.server_args,
        "extra_envs": run.envs,
        "remove_args": run.remove_args,
        "unset_envs": run.unset_envs,
        "args_mode": run.args_mode,
        # Either arm of the capture is a sub-step of the KERNEL phase's own
        # event, not a dispatched measurement, so it records into that event
        # rather than leaving one of its own.
        INLINE_EVENT_PARAM: kernel_event_id(int(getattr(state, "macro_cycle", 0) or 0)),
    }


def _tunableop_recording_envs(inherited: Mapping[str, str], port: int, capture_dir: Path) -> dict[str, str]:
    envs = {
        key: value for key, value in inherited.items() if not key.startswith(("PYTORCH_TUNABLEOP_", "HL_TUNABLEOP_"))
    }
    envs.update(
        {
            "PORT": str(port),
            "RUN_EVAL": "false",
            "HL_TUNABLEOP_MODE": "",
            "HL_TUNABLEOP_FILE": "",
            "HL_TUNABLEOP_VERBOSE": "",
            "PYTORCH_TUNABLEOP_ENABLED": "1",
            "PYTORCH_TUNABLEOP_TUNING": "0",
            "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
            "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(capture_dir / _TUNABLEOP_UNTUNED_NAME),
            "PYTORCH_TUNABLEOP_FILENAME": str(capture_dir / _TUNABLEOP_RESULTS_NAME),
        }
    )
    return envs


async def _capture_vllm_tunableop_shapes(
    *,
    state: Any,
    session_dir: Path,
    payload: dict,
    workspace: Path,
) -> HandlerResult:
    """Record real vLLM GEMMs using TunableOp or a block-FP8 profiler trace."""
    run = _shape_capture_run(state, payload, workspace, session_dir)
    if payload.get("_shape_capture_mode") == "block_fp8_profile":
        return await _capture_block_fp8_profile_shapes(run, state, payload)
    return await _capture_tunableop_recording(run, state, payload)


async def _invoke_forge_gemm_tune(input_payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Run ``kernelforge gemm-tune`` for ``input_payload`` and return its result, or a failed result."""
    try:
        cmd = forge_gemm_tuning.build_command(input_payload)
    except ValueError as exc:
        return {
            "status": "failed",
            "micro_decision": "failed",
            "error_class": exc.__class__.__name__,
            "error": repr(exc),
        }
    aiter_root = _resolve_aiter_root_for_forge()
    if aiter_root:
        cmd = ["env", f"AITER_ROOT_DIR={aiter_root}", *cmd]
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout, label="forge_gemm_tuning")
    except subprocess.TimeoutExpired as exc:
        # Reaped by the process-group kill in _run_subprocess; shape a failed result.
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        return {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout}s: {cmd_repr[:1500]}",
        }
    result = _parse_forge_gemm_sentinel(stdout)
    return result if result is not None else _shape_tool_result(rc, stdout, stderr)


def _bridge_forge_decision(
    result: dict[str, Any],
    *,
    raw_model_path: str,
    session_dir: Path,
) -> None:
    micro = str(result.get("micro_decision") or "").strip().lower()
    if micro == "candidate" and result.get("recommended_env"):
        result.setdefault("decision", "KEEP")
        # Make the tuned CSV durable + recipe-portable (mirrors integrate_patch's
        # source-layer snapshot): copy it into the serving aiter config dir,
        # repoint the env there, and snapshot it so the KEEP survives with the
        # recipe instead of referencing the ephemeral tuner-workspace path.
        # Keep the logical ID here: a resolved HF snapshot basename is a commit
        # hash, which would make durable artifact names unstable across revisions.
        _durable_envs, _snap_dir = _persist_forge_gemm_csv_durably(
            dict(result["recommended_env"]),
            model_path=raw_model_path,
            session_dir=session_dir,
        )
        result.setdefault("extra_envs", _durable_envs)
        if _snap_dir:
            result.setdefault("source_snapshot", _snap_dir)
        # Derive best_speedup from tuners_run when absent.
        if "best_speedup" not in result:
            best = 1.0
            for t in result.get("tuners_run") or []:
                if isinstance(t, dict):
                    sp = float(t.get("best_micro_speedup") or 1.0)
                    if sp > best:
                        best = sp
            if best > 1.0:
                result["best_speedup"] = best
        # Micro-only result: E2E validation still needed.
        result.setdefault("requires_e2e_validation", True)
    elif micro in ("no_improvement", "skipped"):
        # Left unadorned on purpose: the wordings below are only legible against it.
        result.setdefault("decision", "REVERT")
    elif micro in _FORGE_BARREN_MICRO_DECISIONS:
        result.setdefault("decision", "REVERT")
        if micro == "failed":
            result.setdefault("status", "failed")
        result.setdefault("error_class", f"forge_{micro}")


class _ForgeGemmRun(NamedTuple):
    """One forge GEMM tuning run: what it tunes for, where it works, and the log it reads."""

    session_dir: Path
    workspace: Path
    framework: str
    tuning_framework: str
    precision: str
    quant_type: str
    gpu_type: str
    tp: int
    conc: int
    mp: int
    model_dir: str
    model_id: str
    server_log: str
    thorough: bool


class _ForgeShapeSources(NamedTuple):
    """Every shape input a forge GEMM run hands its tuners."""

    tokens: str
    shapes_json: str
    untuned_csv: str
    moe_untuned_csv: str
    moe_key_report: dict[str, Any]
    tunableop_input: str
    shapes_manifest: str
    demand_json: str
    capture: HandlerResult | None


def _requested_existing_file(payload: dict, key: str) -> str:
    path = str(payload.get(key) or "").strip()
    return path if _path_is_existing_file(path) else ""


def _forge_gemm_run(state: Any, payload: dict, session_dir: Path) -> _ForgeGemmRun:
    from hyperloom.common.model_paths import resolve_serving_model_path
    from hyperloom.common.visible_devices import detect_gpu_count
    from hyperloom.inference_optimizer.model_config_utils import resolve_local_model_dir

    workspace = _gemm_tuning_workspace(payload, session_dir=session_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    # Resolve precision from actual runtime, not just session-level state.
    precision, quant_type = _resolve_forge_precision_and_quant(state, payload)

    # Bootstrap already walked HL_MODEL_BASE and the hub cache to decide what to
    # serve; probing only the hub cache here would reject a repo id that the
    # running server resolved fine.
    model_id = _requested_setting(payload, state, "model_path", os.environ.get("MODEL_PATH") or "")
    model_dir = resolve_local_model_dir(resolve_serving_model_path(model_id) or model_id) if model_id else None

    tp = int(payload.get("tp") or state.tp or os.environ.get("TP") or 1)
    # Default mp = all visible GPUs.
    detected_gpus = detect_gpu_count() or tp
    mp = int(payload.get("mp") or os.environ.get("FORGE_GEMM_TUNE_MP") or detected_gpus)
    return _ForgeGemmRun(
        session_dir=session_dir,
        workspace=workspace,
        framework=_requested_framework(payload, state, "sglang"),
        tuning_framework="",
        precision=precision,
        quant_type=quant_type,
        gpu_type=_requested_setting(payload, state, "gpu_type", os.environ.get("GPU_TYPE") or "mi300x").lower(),
        tp=tp,
        conc=int(payload.get("conc") or state.conc or os.environ.get("CONC") or 64),
        mp=mp,
        model_dir=str(model_dir) if model_dir is not None else "",
        model_id=model_id,
        server_log="",
        # Exhaustive search when budget allows (>= 24h) and mp >= 4.
        thorough=bool(float(getattr(state, "max_minutes", 0) or 0) >= 1440 and mp >= 4),
    )


async def _forge_shape_sources(state: Any, payload: dict, run: _ForgeGemmRun) -> _ForgeShapeSources:
    tokens = _normalize_tokens(payload.get("tokens"))
    if not tokens and run.server_log:
        tokens = _normalize_tokens(await asyncio.to_thread(_tokens_from_serving_log, run.server_log))
        if tokens:
            log.info("GEMM: derived --tokens=%s from observed M in %s", tokens, run.server_log)

    moe_untuned_csv = _requested_existing_file(payload, "moe_untuned_csv")
    moe_key_report: dict[str, Any] = {}
    if not moe_untuned_csv:
        moe_untuned_csv, moe_key_report = _write_fmoe_untuned_csv_from_log(run.server_log, tokens, run.workspace)

    shapes_json = _normalize_forge_shapes_json(payload.get("shapes_json"), run.workspace)
    untuned_csv = _requested_existing_file(payload, "untuned_csv")
    if not shapes_json and not untuned_csv and run.framework != "vllm":
        shapes_json = _resolve_forge_shapes(state, run.session_dir, require_fresh_profile=True, precision=run.precision)
        if not shapes_json:
            untuned_csv = _resolve_forge_untuned_csv(run.session_dir, run.precision, run.quant_type, run.model_dir)

    return _ForgeShapeSources(
        tokens=tokens,
        shapes_json=shapes_json,
        untuned_csv=untuned_csv,
        moe_untuned_csv=moe_untuned_csv,
        moe_key_report=moe_key_report,
        tunableop_input=str(payload.get("tunableop_input") or "").strip(),
        shapes_manifest="",
        demand_json=_requested_existing_file(payload, "demand_json"),
        capture=None,
    )


def _profiled_block_fp8_shapes(
    state: Any,
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
) -> _ForgeShapeSources:
    if not _vllm_block_fp8_profile_capture_required(
        framework=run.framework,
        precision=run.precision,
        quant_type=run.quant_type,
        shapes_json=sources.shapes_json,
        tunableop_input=sources.tunableop_input,
        dry_run=bool(payload.get("dry_run")),
    ):
        return sources
    traced_shapes = _resolve_forge_shapes(state, run.session_dir, require_fresh_profile=True, precision=run.precision)
    if traced_shapes:
        return sources._replace(shapes_json=traced_shapes, untuned_csv="")
    reused = _reuse_vllm_block_fp8_roofline_shapes(
        state,
        workspace=run.workspace,
        current_workload=state.current_profile_workload_context(payload),
    )
    if reused is None:
        return sources
    return sources._replace(shapes_json=str(reused["shapes_json"]), untuned_csv="", capture=reused)


async def _captured_forge_shapes(
    state: Any,
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
    priced_lane: Any,
) -> tuple[_ForgeShapeSources, HandlerResult | None]:
    sources = _profiled_block_fp8_shapes(state, payload, run, sources)
    block_fp8_capture = _vllm_block_fp8_profile_capture_required(
        framework=run.framework,
        precision=run.precision,
        quant_type=run.quant_type,
        shapes_json=sources.shapes_json,
        tunableop_input=sources.tunableop_input,
        dry_run=bool(payload.get("dry_run")),
    )
    # Keyed on the routed framework: a run handed to the AITER tuner family has
    # no use for a TunableOp recording pass, and paying for one costs a full
    # extra server boot.
    tunableop_capture = (
        _vllm_dense_shape_capture_required(
            framework=run.tuning_framework,
            model_path=run.model_dir,
            shapes_json=sources.shapes_json,
            tunableop_input=sources.tunableop_input,
            dry_run=bool(payload.get("dry_run")),
        )
        and not block_fp8_capture
    )
    if not block_fp8_capture and not tunableop_capture:
        return sources, None
    # A capture costs a full extra server boot, wasted on a lane the phase cannot fund once it has the shapes.
    lane_after_capture = await priced_lane(
        has_untuned_csv=tunableop_capture and bool(sources.untuned_csv),
        has_shapes_json=block_fp8_capture,
        has_tunableop_input=tunableop_capture,
    )
    if lane_after_capture is None:
        return sources, _unfunded_lane_skip(LANE_GEMM)

    capture_payload = dict(payload)
    if block_fp8_capture:
        capture_payload["_shape_capture_mode"] = "block_fp8_profile"
    capture = await _capture_vllm_tunableop_shapes(
        state=state,
        session_dir=run.session_dir,
        payload=capture_payload,
        workspace=run.workspace,
    )
    if capture.get("status") != "ok":
        return sources, _stamp_forge_gemm_provenance(capture, run)
    captured = sources._replace(
        tunableop_input=str(capture.get("tunableop_input") or "").strip(),
        capture=capture,
    )
    captured_shapes = str(capture.get("shapes_json") or "").strip()
    if not captured_shapes:
        return captured, None
    # Forge dense tuners prefer untuned_csv over shapes_json. A fresh profile
    # capture is workload-matched and must supersede any stale specialist CSV
    # resolved before the capture pass.
    return captured._replace(shapes_json=captured_shapes, untuned_csv=""), None


async def _forge_dense_shape_manifest(
    state: Any,
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
) -> str:
    requested = str(payload.get("shapes_manifest") or "").strip()
    if requested:
        return requested if _path_is_existing_file(requested) else ""
    if sources.untuned_csv or sources.shapes_json:
        log.debug(
            "GEMM: not scavenging a trace shape manifest; this run already has "
            "a workload-matched dense-shape source (%s)",
            "untuned_csv" if sources.untuned_csv else "shapes_json",
        )
        return ""
    # Off the event loop for the same reason: a ``**/`` walk of a session tree
    # that holds thousands of run artifacts.
    scavenged = await asyncio.to_thread(_resolve_trace_shape_manifest, state, run.session_dir)
    return scavenged if _path_is_existing_file(scavenged) else ""


def _forge_gemm_tune_input(
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
    timeout: int,
    tuner_ceiling: int,
) -> dict[str, Any]:
    return {
        "model_path": run.model_dir,
        "framework": run.tuning_framework,
        "precision": run.precision,
        "quant_type": run.quant_type,
        "gpu_type": run.gpu_type,
        "tp": run.tp,
        "conc": run.conc,
        "mp": run.mp,
        "output_dir": str(run.workspace),
        # Passing the same value to both made the producer's own
        # min(per_tuner, remaining) an identity, so the first tuner could
        # consume the entire session and every later one was skipped for lack of
        # time. The per-target cap must stay strictly below the global one.
        "timeout": gemm_per_tuner_timeout_sec(timeout),
        # Bounds the whole session across all tuners.
        "global_timeout": timeout,
        "skip_gpu_check": True,
        "tokens": sources.tokens,
        "untuned_csv": sources.untuned_csv,
        "moe_untuned_csv": sources.moe_untuned_csv,
        "shapes_json": sources.shapes_json,
        "shapes_manifest": sources.shapes_manifest,
        "demand_json": sources.demand_json,
        "tunableop_input": sources.tunableop_input,
        "kernel_signature_log": run.server_log,
        "tuner": str(payload.get("tuner") or ""),
        # How many routed tuners the lane's share pays for. Omitted when none
        # could be derived, which leaves the producer's own routing intact.
        **({"max_tuners": tuner_ceiling} if tuner_ceiling > 0 else {}),
        "thorough": run.thorough,
    }


def _stamp_forge_gemm_provenance(
    result: HandlerResult,
    run: _ForgeGemmRun,
    tuning_framework: str = "",
) -> HandlerResult:
    result.setdefault("backend", "forge")
    result.setdefault("engine", "forge")
    result.setdefault("workspace", str(run.workspace))
    result.setdefault("precision", run.precision)
    result.setdefault("framework", run.framework)
    if tuning_framework:
        result.setdefault("tuning_framework", tuning_framework)
    result.setdefault("model_path", run.model_id)
    return result


def _report_forge_shape_provenance(
    result: HandlerResult,
    sources: _ForgeShapeSources,
    shape_alignment: dict[str, Any] | None,
) -> None:
    if sources.moe_key_report:
        result.setdefault("moe_key_source", sources.moe_key_report)
    if shape_alignment is not None:
        result.setdefault("shape_alignment", shape_alignment)
    if sources.capture is None:
        return
    result.setdefault(
        "shape_capture",
        {
            "status": "ok",
            "workspace": sources.capture.get("shape_capture_workspace"),
            "tunableop_input": sources.tunableop_input,
            "shapes_json": sources.shapes_json,
            "shape_count": sources.capture.get("shape_count"),
            "capture_mode": sources.capture.get("capture_mode", "tunableop"),
            "source_profile_trace": sources.capture.get("source_profile_trace"),
        },
    )


def _fill_forge_tuner_outcomes(result: HandlerResult, workspace: Path) -> None:
    on_disk = _read_forge_result_json(workspace)
    for key in ("tuners_skipped", "failed_tuners"):
        if not result.get(key) and on_disk.get(key):
            result[key] = on_disk[key]
    if not result.get("skip_reason"):
        reason = _derive_gemm_skip_reason(result.get("tuners_skipped"))
        if reason:
            result["skip_reason"] = reason
    failed = result.get("failed_tuners")
    for row in failed if isinstance(failed, list) else []:
        if isinstance(row, dict):
            log.warning(
                "forge gemm tuner %s failed (%s): %s",
                row.get("tuner") or "?",
                row.get("error_class") or "?",
                row.get("error") or "",
            )
    rows = result.get("tuners_run")
    rows = rows if isinstance(rows, list) else []
    for key in ("error_class", "error"):
        if result.get(key):
            continue
        reported = next((str(row[key]) for row in rows if isinstance(row, dict) and row.get(key)), "")
        if reported:
            result[key] = reported


async def _run_forge_gemm_tuning(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Deterministic GEMM tuning via the ``kernelforge gemm-tune`` CLI.

    Supports bf16/fp8/fp4 + sglang/vllm. Only micro-benchmarks;
    returns recommended_env for Hyperloom E2E validation.

    ``model_path`` accepts either a local directory or a Hugging Face repo ID.
    Forge receives a validated local directory, while result provenance and
    durable artifact names retain the original logical model identifier.
    Missing inputs return ``model_path_missing``; inputs that cannot resolve to
    a local directory return ``model_path_unavailable`` as a ``skipped`` result,
    because forge never ran and so has no verdict to report.
    """
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    # Importing kernelforge.cli is deliberately isolated in a subprocess, but
    # that subprocess may still take until the bounded timeout to fail. Keep the
    # synchronous probe off the orchestrator reactor.
    if not await asyncio.to_thread(_forge_gemm_tune_available):
        return {
            "status": "failed",
            "error_class": "forge_gemm_tune_not_found",
            "error": (
                "forge-gemm-tune is not runnable in this interpreter: "
                f"'{sys.executable} -m kernelforge.cli gemm-tune --help' failed. "
                "kernelforge ships with this distribution, so this means a "
                "partial install: reinstall with 'pip install -e .[forge]'."
                f" (interpreter: {sys.executable!r})"
            ),
            "backend": "forge",
        }

    run = _forge_gemm_run(state, payload, session_dir)
    if not run.model_id:
        return {"status": "failed", "error_class": "model_path_missing", "error": "model_path is required"}
    if not run.model_dir:
        # Forge needs the config on disk to derive shapes, so it cannot run --
        # but not running one tuning backend is a skip, not a session failure.
        # Reporting it as failed would spend a REVERT verdict on an experiment
        # that never started.
        return {
            "status": "skipped",
            "error_class": "model_path_unavailable",
            "skip_reason": (
                f"Model path {run.model_id!r} is neither an existing local "
                "directory nor an available Hugging Face cache snapshot"
            ),
            "backend": "forge",
        }
    # Off the event loop: this walks runs/ and byte-scans server logs that measure
    # ~17MB apiece on the fleet. Inline, it stalled every other coroutine on this
    # orchestrator -- heartbeats included -- for the duration.
    server_log = str(payload.get("kernel_signature_log") or "").strip() or await asyncio.to_thread(
        _resolve_forge_server_log, state, session_dir
    )
    run = run._replace(
        server_log=server_log,
        tuning_framework=_forge_framework_for_vllm(
            framework=run.framework,
            precision=run.precision,
            quant_type=run.quant_type,
            tunableop_input=str(payload.get("tunableop_input") or "").strip(),
            **_resolve_vllm_aiter_routing(model_path=run.model_dir, server_log=server_log, tp=run.tp),
        ),
    )
    priced_gemm_lane = functools.partial(
        _priced_gemm_lane,
        state,
        model_path=run.model_dir,
        framework=run.tuning_framework,
        precision=run.precision,
        quant_type=run.quant_type,
        gpu_type=run.gpu_type,
        kernel_signature_log=run.server_log,
    )

    sources = await _forge_shape_sources(state, payload, run)
    sources, capture_failure = await _captured_forge_shapes(state, payload, run, sources, priced_gemm_lane)
    if capture_failure is not None:
        return capture_failure
    sources = sources._replace(shapes_manifest=await _forge_dense_shape_manifest(state, payload, run, sources))

    gemm_lane = await priced_gemm_lane(
        has_untuned_csv=bool(sources.untuned_csv),
        has_shapes_json=bool(sources.shapes_json or sources.shapes_manifest or sources.demand_json),
        has_tunableop_input=bool(sources.tunableop_input),
    )
    if gemm_lane is None:
        return _unfunded_lane_skip(LANE_GEMM)
    timeout = _gemm_tuning_timeout_sec(payload, lane_budget_sec=gemm_lane.budget_sec)
    shape_alignment: dict[str, Any] | None = None
    if sources.shapes_json:
        aligned_shapes, shape_alignment = _align_forge_shapes_for_aiter(
            sources.shapes_json,
            forge_framework=run.tuning_framework,
            workspace=run.workspace,
            budget_sec=timeout,
            mp=run.mp,
        )
        sources = sources._replace(shapes_json=aligned_shapes)

    requested_tuners = _positive_int(payload.get("max_tuners"))
    tuner_ceiling = 0 if str(payload.get("tuner") or "").strip() else (requested_tuners or gemm_lane.max_targets)
    result = await _invoke_forge_gemm_tune(_forge_gemm_tune_input(payload, run, sources, timeout, tuner_ceiling), timeout)
    _stamp_forge_gemm_provenance(result, run, tuning_framework=run.tuning_framework)
    _report_forge_shape_provenance(result, sources, shape_alignment)
    _fill_forge_tuner_outcomes(result, run.workspace)
    _bridge_forge_decision(result, raw_model_path=run.model_id, session_dir=run.session_dir)
    return result


def _persist_forge_gemm_csv_durably(extra_envs: dict, *, model_path: str, session_dir: Path) -> tuple[dict, str]:
    """Make forge GEMM tuned CSVs durable + recipe-portable.

    The forge KEEP references tuned CSVs by their ephemeral tuner-workspace paths,
    so a recipe replayed after the workspace is gone (or on another box) loses the
    tuning and aiter falls back to its default config. Mirror integrate_patch's
    durability: copy each CSV into the serving aiter config tree, repoint the env
    there, and snapshot the realized files via :func:`snapshot_source_layer` so
    they travel with the recipe.

    The copy lands one level below ``configs/model_configs/`` on purpose. aiter
    merges every ``model_configs/*{table}*.csv`` it can glob whenever the env var
    is unset, and that glob is not recursive. Writing directly into that
    directory would hand the table to every later server start -- including after
    E2E rejected the candidate, and including servers for other models, since the
    scan does not discriminate by model. Replay does not need the scan: it
    restores the env var explicitly (see ``prelude._warm_kernel_extra_envs``) and
    defers a GEMM column that has no env at all.

    The snapshot lands under ``<session_dir>/optimization_stack/src/`` (the same
    durable, run-cleanup-surviving location integrate_patch uses) -- NOT under the
    ephemeral ``runs/gemm_tuning`` workspace, which would be cleaned away and
    defeat the cross-environment recipe-portability this exists for.

    Best-effort: on any error the env is returned unchanged (never breaks the KEEP).
    Returns ``(extra_envs, source_snapshot_dir)``.
    """
    # Below model_configs/, out of reach of aiter's non-recursive auto-merge glob.
    _FORGE_DURABLE_SUBDIR = "hyperloom"
    _forge_durable_env_stems = {
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm",
        "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm",
        "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm",
        "AITER_CONFIG_FMOE": "tuned_fmoe",
    }
    slug = (
        "".join(c if (c.isalnum() or c in "._-") else "_" for c in Path(model_path).name).strip("_").lower() or "model"
    )

    pending: list[tuple[str, str, Path]] = []
    for env_key, stem in _forge_durable_env_stems.items():
        src_csv = str(extra_envs.get(env_key) or "").strip()
        if not src_csv or not Path(src_csv).is_file():
            continue
        rel = f"configs/model_configs/{_FORGE_DURABLE_SUBDIR}/{stem}_{slug}.csv"
        pending.append((env_key, rel, Path(src_csv)))
    if not pending:
        return extra_envs, ""

    # Step 1 -- commit durable copies + env repoints. This is what makes the
    # KEEP survive: each CSV lands in aiter's config dir and the env points
    # there instead of the ephemeral tuner workspace.
    try:
        import importlib.util

        spec = importlib.util.find_spec("aiter")
        if spec is None or not spec.origin:
            return extra_envs, ""
        aiter_pkg = Path(spec.origin).resolve().parent
        updated = dict(extra_envs)
        rel_paths: list[str] = []
        for env_key, rel, src_path in pending:
            dst = aiter_pkg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            updated[env_key] = str(dst)
            rel_paths.append(rel)
    except Exception:
        log.exception("forge gemm CSV durable-copy failed; keeping workspace path")
        return extra_envs, ""

    # Step 2 -- recipe-portability snapshot. Separate best-effort concern: a
    # snapshot failure must NOT discard the copy + repoint committed above.
    snap_dir = ""
    try:
        from ..source_snapshot import snapshot_source_layer

        snap = snapshot_source_layer(
            framework_root=aiter_pkg,
            base_sha=None,
            rel_paths=rel_paths,
            dest_dir=Path(session_dir) / "optimization_stack" / "src" / f"forge_gemm_{slug}",
            provenance="kernelforge.gemm_tune",
            extra={
                "env_keys": [env_key for env_key, _, _ in pending],
                "model": slug,
            },
        )
        snap_dir = str((snap or {}).get("snapshot_dir") or "")
    except Exception:
        log.exception("forge gemm CSV snapshot failed; durable copy + repoint kept")
    return updated, snap_dir


async def run_gemm_tuning_handler(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Run the forge GEMM tuner and record the run in the audit trace.

    Args:
        payload: The GEMM-tuning request payload.
        session_dir: Session directory for workspace and state.

    Returns:
        A ``HandlerResult`` describing the tuning outcome.
    """
    result = await _run_forge_gemm_tuning(payload, session_dir=session_dir)
    result.setdefault("task_id", payload.get("task_id"))
    result.setdefault("macro_cycle", payload.get("macro_cycle"))
    _trace_gemm_tuning_run(result, session_dir=session_dir)
    return result


# forge-fusion (autonomous kernel fusion)
def _forge_fusion_available() -> bool:
    """Check that KernelForge's fusion pipeline is importable.

    Probes the subpackage rather than ``kernelforge``: an installation
    predating the fusion absorption would satisfy the parent import and only
    fail once the subprocess rejected ``forge-fuse``. PATH is not consulted
    because the tool is invoked through ``sys.executable -m``.
    """
    try:
        return importlib.util.find_spec("kernelforge.fusion") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _resolve_fusion_decode_trace(state) -> str:
    """Reuse this run's PRELUDE/roofline decode trace for fusion discovery.

    forge-fusion's discover stage needs a CUDA-graph-disabled decode kineto trace.
    PRELUDE/roofline already captured one and promoted it into
    ``state.last_profile_trace`` alongside the workload it was measured on
    (``SharedState.record_profile_workload``), so the lane reuses that rather than
    re-profiling. ``state.last_profile_trace`` is the only source: the fusion
    opportunities discovered below are attributed to the run that produced the trace,
    and a caller-supplied path would file this run's decisions under another workload.

    ``last_profile_trace`` holds a single trace file for AgentX profiles and for any
    round that merged its ranks, and the capture directory otherwise --
    ``_preferred_main_trace_path`` in ``actions/executors/profile.py`` returns
    ``trace_dir`` when ``require_single_rank`` is off and no ``merged-*`` file exists,
    which is every non-AgentX sglang/vLLM profile. A directory therefore resolves to the
    newest capture in it, which is this run's too: the directory is the round's own
    ``trace_dir``.
    """
    trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    if not trace:
        return ""
    path = Path(trace)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        return ""
    captures = sorted(
        list(path.glob("*.trace.json.gz")) + list(path.glob("*.trace.json")) + list(path.glob("*.json.gz")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(captures[0]) if captures else ""


def _active_forge_fusion_env_flags(state: Any) -> dict[str, str]:
    """Return active env flags only when forge-fusion itself is current_best."""
    current_best = getattr(state, "current_best", None) or {}
    if str(current_best.get("action") or "") != "fusion":
        return {}
    envs = current_best.get("extra_envs")
    if not isinstance(envs, dict):
        return {}
    active: dict[str, str] = {}
    for key, val in envs.items():
        name = str(key)
        value = str(val)
        if "_FUSED" not in name.upper():
            continue
        if value.strip().lower() in ("", "0", "false", "no", "off", "none"):
            continue
        active[name] = value
    return active


def _resolve_forge_agent(
    payload: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the Forge agent backend and model as one decision.

    Shared by forge-fusion and the rewrite lane, which uses the same model
    ladder via :func:`llm_config.resolve_forge_llm_model`. A valid explicit
    ``agent_backend`` or ``llm_model`` in the request wins; otherwise
    :func:`llm_config.preferred_agent_backend` decides, so this role cannot
    disagree with the specialists, the TraceLens runner or the Forge registry
    about which backend a box is configured for.

    Model id precedence (after the backend is chosen) is owned by
    :func:`llm_config.resolve_forge_llm_model`.



    Raises:
        ValueError: If ``agent_backend`` is not ``"claude"`` or ``"codex"``.
    """
    source = env if env is not None else os.environ
    known_backends = {llm_config.AGENT_BACKEND_CLAUDE, llm_config.AGENT_BACKEND_CODEX}
    explicit_backend = str(payload.get("agent_backend") or "").strip().lower()
    if explicit_backend and explicit_backend not in known_backends:
        raise ValueError(f"agent_backend={payload.get('agent_backend')!r} is invalid; choose 'claude' or 'codex'")

    agent_backend = explicit_backend or llm_config.preferred_agent_backend(source)
    llm_model = llm_config.resolve_forge_llm_model(
        agent_backend,
        env=source,
        explicit=str(payload.get("llm_model") or ""),
    )
    return agent_backend, llm_model


def _resolve_forge_fusion_sandbox_mode(
    payload: Mapping[str, Any],
    *,
    agent_backend: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve and validate the sandbox policy recorded for forge-fusion.

    Codex delegates both defaults and validation to the canonical Hyperloom
    resolver, including its operator opt-in for ``bypass``. Claude records
    ``workspace-write`` as the stable audit default; an explicit override is
    validated by that same resolver so both backends share one policy vocabulary
    and unsafe bypass cannot reach the subprocess.



    Raises:
        CodexSessionUnavailableError: If the mode is unknown or bypass lacks
            the operator mode confirmation.
    """
    explicit = str(payload.get("agent_sandbox_mode") or "").strip()
    if agent_backend == "claude" and not explicit:
        return codex_session.DEFAULT_CODEX_SANDBOX_MODE
    return codex_session.resolve_codex_sandbox_mode(
        sandbox_mode=explicit,
        env=dict(env) if env is not None else None,
    )


async def _run_forge_fusion(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Autonomous kernel fusion via the forge-fusion CLI.

    Resolves one provider-compatible agent backend, model, and validated sandbox
    policy, then runs forge-fuse through :func:`forge_fusion.run_forge_fusion`,
    which returns the fusion contract result. A KEPT fusion carries a source patch + env flags
    and ``requires_e2e_validation`` so the integrate gate confirms the
    end-to-end gain. Reuses the PRELUDE decode trace (no re-profiling).
    """
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    active_fusion_flags = _active_forge_fusion_env_flags(state)
    if active_fusion_flags:
        return {
            "status": "complete",
            "backend": "forge",
            "engine": "forge_fusion",
            "micro_decision": "already_active",
            "decision": "REVERT",
            "kept": False,
            "requires_e2e_validation": False,
            "active_env_flags": active_fusion_flags,
            "reason": (
                "current_best is already a forge-fusion KEEP; "
                "skip forge-fusion to avoid rerunning the same adopted source patch"
            ),
            "source": "forge_fusion",
        }

    if not _forge_fusion_available():
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "forge_fusion_not_found",
            "error": ("KernelForge fusion pipeline not found. Install via 'pip install <KernelForge>[claude,codex]'."),
            "decision": "REVERT",
            "kept": False,
        }

    model_path = _requested_setting(payload, state, "model_path", os.environ.get("MODEL_PATH") or "")
    if not model_path:
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "model_path_missing",
            "error": "model_path is required",
            "decision": "REVERT",
            "kept": False,
        }

    trace_path = _resolve_fusion_decode_trace(state)
    if not trace_path:
        recorded = str(getattr(state, "last_profile_trace", "") or "").strip()
        return {
            "status": "skipped",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "decode_trace_missing",
            "error": (
                "no decode trace available for fusion discovery "
                f"(state.last_profile_trace={recorded or '(empty)'}; run profile/roofline first)"
            ),
            "decision": "REVERT",
            "kept": False,
        }

    framework = _requested_framework(payload, state, "sglang")
    gpu = str(payload.get("gpu") or "0").strip()
    try:
        agent_backend, llm_model = _resolve_forge_agent(payload)
    except ValueError as exc:
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "invalid_agent_backend",
            "error": str(exc),
            "decision": "REVERT",
            "kept": False,
        }
    try:
        agent_sandbox_mode = _resolve_forge_fusion_sandbox_mode(
            payload,
            agent_backend=agent_backend,
        )
    except RuntimeError as exc:
        return {
            "status": "failed",
            "backend": "forge",
            "engine": "forge_fusion",
            "error_class": "invalid_agent_sandbox_mode",
            "error": str(exc),
            "decision": "REVERT",
            "kept": False,
        }
    max_turns = int(payload.get("max_turns") or os.environ.get("FORGE_FUSION_MAX_TURNS") or 100)
    fusion_lane = _lane_budget(state, LANE_FUSION)
    if fusion_lane is None:
        return {**_unfunded_lane_skip(LANE_FUSION), "engine": "forge_fusion"}
    timeout = _forge_fusion_timeout_sec(payload, lane_budget_sec=fusion_lane.budget_sec)
    requested_recipes = _positive_int(payload.get("max_recipes"))
    # An explicit payload value stays an operator/test escape hatch.
    fusion_recipe_ceiling = requested_recipes if requested_recipes > 0 else fusion_lane.max_targets

    workspace = session_dir / "runs" / "fusion" / str(payload.get("task_id") or "kernel_entry_fusion")
    workspace.mkdir(parents=True, exist_ok=True)

    input_payload = {
        "trace_path": trace_path,
        "model_path": model_path,
        "framework": framework,
        "output_dir": str(workspace),
        "discover_mode": str(payload.get("discover_mode") or "llm"),
        "agent_backend": agent_backend,
        "llm_model": llm_model,
        "agent_sandbox_mode": agent_sandbox_mode,
        "max_turns": max_turns,
        "gpu": gpu,
        "timeout": timeout,
        # Multi-patch (one independent sibling per recipe) is the default; the
        # combine escape hatch (a single merged patch) must be requested explicitly.
        "fuse_all_confirmed": bool(payload.get("fuse_all_confirmed", False)),
        # How many recipes the lane's share pays for. Omitted when none could be
        # derived, which leaves forge-fuse on every discovered recipe.
        **({"max_recipes": fusion_recipe_ceiling} if fusion_recipe_ceiling > 0 else {}),
        "verbose": bool(payload.get("verbose", False)),
        **_fusion_session_serve_args(state, payload, framework=framework, model_path=model_path),
    }
    result = await forge_fusion.run_forge_fusion(input_payload)

    result.setdefault("backend", "forge")
    result.setdefault("engine", "forge_fusion")
    result.setdefault("workspace", str(workspace))
    result.setdefault("framework", framework)
    result.setdefault("model_path", model_path)
    result.setdefault("agent_backend", agent_backend)
    result.setdefault("llm_model", llm_model)
    result.setdefault("agent_sandbox_mode", agent_sandbox_mode)
    result.setdefault("source", "forge_fusion")
    return result


async def run_fusion_handler(payload: dict, *, session_dir: Path) -> HandlerResult:
    """Run autonomous kernel fusion via forge-fusion (serving-validated).

    Registered as the ``run_fusion`` kernel request. Authors serving-safe fused
    kernels and returns a source patch + env flags for the integrate gate.
    """
    return await _run_forge_fusion(payload, session_dir=session_dir)


# A tuner error is a diagnostic pointer, not the diagnosis: the full text lives
# in the run's own result.json and tune.log. 400 characters is enough to carry
# the argparse line or the aiter marker that says which of the two it was.
_TRACE_TUNER_ERROR_MAXLEN = 400

# Emitted even when null. ``kept`` is null on every row observed so far, and an
# absent key would be indistinguishable from ``false``.
_TRACE_TUNER_ALWAYS_KEYS = ("tuner", "best_micro_speedup", "kept")


def _trace_tuner_row(tuner: dict[str, Any]) -> dict[str, Any]:
    """One per-tuner entry for the audit row, keeping why it ended as it did.

    ``tuner``/``best_micro_speedup``/``kept`` alone cannot separate a tuner that
    crashed from one that ran and found nothing -- the single question the audit
    trail exists to answer. Without the reason, a tuner whose runs argparse
    rejected in seconds reads as a clean ``no_improvement`` (#1211).
    """
    error = tuner.get("error")
    if isinstance(error, str) and len(error) > _TRACE_TUNER_ERROR_MAXLEN:
        error = error[:_TRACE_TUNER_ERROR_MAXLEN] + "..."
    row = {
        "tuner": tuner.get("tuner") or tuner.get("name"),
        "best_micro_speedup": tuner.get("best_micro_speedup"),
        "kept": tuner.get("kept"),
        "status": tuner.get("status"),
        "elapsed_s": tuner.get("elapsed_s"),
        "error_class": tuner.get("error_class"),
        "error": error,
    }
    # A clean run stays as compact as before: everything added here is dropped
    # when it is null, so a successful row gains only status and elapsed_s.
    return {k: v for k, v in row.items() if k in _TRACE_TUNER_ALWAYS_KEYS or v is not None}


def _trace_gemm_tuning_run(result: Any, *, session_dir: Path) -> None:
    """Append one ``gemm_tuning.jsonl`` audit row for a GEMM-tuning run.

    Distils the run result into a compact source-attribution row (engine,
    decision, speedup, per-tuner summary) appended to
    ``reports/trace/gemm_tuning.jsonl``. Best-effort; any failure is swallowed.
    """
    if not isinstance(result, dict):
        return
    from datetime import datetime, timezone

    from hyperloom.inference_optimizer.session.session_paths import gemm_tuning_steps_path

    engine = str(result.get("engine") or result.get("backend") or "").strip().lower() or "unknown"
    tuners: list[dict[str, Any]] = [
        _trace_tuner_row(t) for t in (result.get("tuners_run") or []) if isinstance(t, dict)
    ]
    # The envelope reported no error class even when a tuner had named one, so a
    # crashed run and a barren one looked alike at the top level too. Take the
    # first one a tuner supplied rather than leaving the field null.
    error_class = result.get("error_class") or next((t["error_class"] for t in tuners if t.get("error_class")), None)
    row = {
        "kind": "gemm_tuning",
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "engine": engine,
        "backend": result.get("backend"),
        "status": result.get("status"),
        "decision": result.get("decision"),
        "micro_decision": result.get("micro_decision"),
        "best_speedup": result.get("best_speedup"),
        "precision": result.get("precision"),
        "framework": result.get("framework"),
        "gpu_type": result.get("gpu_type"),
        "tuned_file": result.get("tuned_file"),
        "workspace": result.get("workspace"),
        "requires_e2e_validation": result.get("requires_e2e_validation"),
        "tuners_run": tuners,
        # A crash stays in the audit even when a sibling tuner won and the run was
        # kept -- otherwise a KEEP row hid the failures behind it.
        "failed_tuners": result.get("failed_tuners") or None,
        "error_class": error_class,
    }
    row = {k: v for k, v in row.items() if v is not None}
    try:
        append_jsonl(gemm_tuning_steps_path(session_dir), row, make_parents=True, fmt=JsonLineFormat(sort_keys=True))
    except OSError:
        log.debug("full-trace: gemm_tuning audit append failed", exc_info=True)


# TraceLens picks its steady-state window by writing split chunks and selecting
# one file; the TraceLens-free reader picks a window in memory and never writes
# chunks. Both answer "is the window this analysis rests on trustworthy", so the
# event normalizes them onto one shape and keeps the raw form under ``selected``.


def _summarize_dropped_patches(dropped: Any) -> dict[str, int]:
    """Count dropped patch entries by reason, logging each one as it is counted.

    ``parse_outcome`` names why every unusable entry was refused, but a reason
    nobody reports is indistinguishable from forge never having offered the
    entry at all.
    """
    counts: dict[str, int] = {}
    for entry in dropped or ():
        reason = str(getattr(entry, "reason", "") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
        log.warning(
            "nomination landing: refused patch kernel=%s reason=%s",
            str(getattr(entry, "kernel_name", "") or "<unnamed>"),
            reason,
        )
    return counts


def _raw_kernel_backend_order() -> list[str]:
    """Return the effective kernel backend order.

    Forge is deliberately not request-selectable.  The only supported forge
    opt-in is exactly ``KERNEL_OPT_BACKEND_ORDER=forge``; every other value,
    missing value or legacy alias stays on the GEAK whole-phase backend.
    """
    if forge_explicitly_enabled():
        return ["forge"]
    return list(_DEFAULT_KERNEL_PHASE_BACKEND_ORDER)


def geak_selected() -> bool:
    """Whether ``geak`` (the whole-pipeline e2e delegate) is in the kernel backend order.

    ``geak`` is not a per-kernel backend: when it appears in the order it
    means "delegate the whole KERNEL_AGENT phase to the GEAK e2e optimizer".
    It therefore *owns* the phase whenever present (any other backends in the
    order are ignored for the kernel phase), so an order of just ``geak``
    runs only the GEAK e2e optimizer. ``forge`` is the per-kernel backend.

    Returns:
        bool: ``True`` when ``geak`` is in the resolved order.
    """
    return "geak" in _raw_kernel_backend_order()


def _sweep_integrate_aiter_locks(*, reason: str) -> dict[str, Any]:
    """Best-effort orphaned-lock sweep immediately before an integrate boot."""
    from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

    try:
        stats = sweep_stale_aiter_locks_if_dead()
    except Exception as exc:  # noqa: BLE001 - cache hygiene must not hide benchmark results
        log.warning("integrate_handler: aiter lock sweep failed before %s: %r", reason, exc)
        return {"errors": 1, "exception": repr(exc)}
    if stats.get("skipped_live"):
        log.info(
            "integrate_handler: aiter lock sweep skipped before %s; a compiler is alive",
            reason,
        )
    elif stats.get("deleted"):
        log.warning(
            "integrate_handler: reaped %d orphaned aiter lock(s) across %s before %s",
            stats.get("deleted"),
            stats.get("dirs") or [stats.get("dir")],
            reason,
        )
    return stats


_INTEGRATE_LOG_NAMES = ("server.log", "benchmark_stderr.log", "benchmark_stdout.log")


def _workspace_log_sizes(workspace: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name in _INTEGRATE_LOG_NAMES:
        path = workspace / name
        try:
            sizes[name] = path.stat().st_size if path.is_file() else 0
        except OSError:
            sizes[name] = 0
    return sizes


def _compiled_registry_error_text(
    workspace: Path,
    *,
    after_sizes: dict[str, int] | None = None,
) -> str:
    """Return the integrate log excerpt that shows a compiled-registry miss.

    The excerpt names the kernel aiter could not find, which is the only thing that
    identifies the module at fault when the round's env does not.
    """
    from ..actions.executors._aiter_jit import is_aiter_jit_registry_mismatch

    for name in _INTEGRATE_LOG_NAMES:
        path = workspace / name
        if not path.is_file():
            continue
        start = 0 if after_sizes is None else max(0, int(after_sizes.get(name, 0)))
        try:
            with path.open("rb") as handle:
                if after_sizes is None:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 65536))
                else:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    if start >= size:
                        handle.seek(max(0, size - 65536))
                    else:
                        handle.seek(start)
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        if is_aiter_jit_registry_mismatch(text):
            return text
    return ""


def _workspace_has_compiled_registry_error(
    workspace: Path,
    *,
    after_sizes: dict[str, int] | None = None,
) -> bool:
    """True when an integrate workspace log shows a compiled-registry miss."""
    return bool(_compiled_registry_error_text(workspace, after_sizes=after_sizes))


def _integrate_extra_envs(ctx: Any) -> dict[str, str] | None:
    task = getattr(ctx, "task", None)
    params = getattr(task, "params", None)
    if isinstance(params, dict):
        envs = params.get("extra_envs")
        if isinstance(envs, dict):
            return envs
    return None


def _lock_cleanup_allows_retry(cleanup: dict[str, Any]) -> bool:
    if cleanup.get("skipped_live") or cleanup.get("errors"):
        return False
    return bool(cleanup.get("deleted")) or (cleanup.get("scanned", 0) == 0 and not cleanup.get("skipped_fresh"))


async def _run_integrate_rebaseline_with_lock_retry(
    executor: Any,
    ctx: Any,
    *,
    workspace: Path,
    reason: str,
) -> dict[str, Any]:
    """Run one integrate baseline and retry once after a JIT registry miss or baton stall."""
    from ..actions.executors._aiter_jit import (
        drop_serving_so_for_envs,
        find_aiter_baton_wait,
        result_is_aiter_jit_registry_mismatch,
    )

    prelaunch_sweep = _sweep_integrate_aiter_locks(reason=reason)
    first_started_unix = time.time()
    result = await executor(ctx)
    if not isinstance(result, dict) or result.get("status") == "succeeded":
        return result

    if result_is_aiter_jit_registry_mismatch(result) or _workspace_has_compiled_registry_error(workspace):
        envs = _integrate_extra_envs(ctx)
        log_sizes = _workspace_log_sizes(workspace)
        # The env does not always reach the module at fault: a round that pins one
        # table leaves every other AITER_CONFIG_* unset, and each of those resolves to
        # the shipped default plus its matching model_configs overlays, and
        # AITER_CONFIG_FMOE maps to no serving module at all. Read the module out of
        # the kernel the error named, or the drop unlinks nothing and the retry
        # repeats the failure.
        from ..actions.executors._aiter_jit import registry_mismatch_modules

        named_modules = registry_mismatch_modules(
            str(result.get("error") or ""),
            _compiled_registry_error_text(workspace),
        )
        cleanup = drop_serving_so_for_envs(
            envs,
            backup_dir=workspace / "aiter_jit_backup",
            also_modules=named_modules,
        )
        log.warning(
            "integrate_handler: classified %s as aiter_jit_registry_mismatch; "
            "retrying once after so drop (env modules + %s from the error)",
            reason,
            list(named_modules) or "none",
        )
        retry_result = await executor(ctx)
        if not isinstance(retry_result, dict):
            return retry_result
        retry_result["aiter_jit_registry_mismatch_retry"] = {
            "cleanup": cleanup,
            "retry_attempted": True,
            "retry_succeeded": retry_result.get("status") == "succeeded",
        }
        if result_is_aiter_jit_registry_mismatch(retry_result) or (
            retry_result.get("status") != "succeeded"
            and _workspace_has_compiled_registry_error(workspace, after_sizes=log_sizes)
        ):
            retry_result["error_class"] = "aiter_jit_registry_mismatch"
        return retry_result

    evidence = find_aiter_baton_wait(
        workspace,
        since_unix=first_started_unix - 1.0,
    )
    if evidence is None:
        return result

    cleanup = _sweep_integrate_aiter_locks(reason=f"{reason} stale-lock retry")
    result["error_class"] = "stale_jit_lock"
    result["stale_jit_lock"] = {
        "evidence": evidence,
        "prelaunch_sweep": prelaunch_sweep,
        "post_failure_sweep": cleanup,
        "retry_attempted": False,
    }

    if not _lock_cleanup_allows_retry(cleanup):
        return result

    log.warning(
        "integrate_handler: classified %s as stale_jit_lock; retrying once after cleanup",
        reason,
    )
    retry_started_unix = time.time()
    retry_result = await executor(ctx)
    if not isinstance(retry_result, dict):
        return retry_result
    retry_evidence = (
        find_aiter_baton_wait(
            workspace,
            since_unix=retry_started_unix - 1.0,
        )
        if retry_result.get("status") != "succeeded"
        else None
    )
    retry_result["stale_jit_lock_retry"] = {
        "evidence": evidence,
        "cleanup": cleanup,
        "retry_attempted": True,
        "retry_succeeded": retry_result.get("status") == "succeeded",
    }
    if retry_evidence is not None:
        retry_result["error_class"] = "stale_jit_lock"
        retry_result["stale_jit_lock_retry"]["retry_evidence"] = retry_evidence
    return retry_result


class _SourceImport(NamedTuple):
    """Whether the served process imported the patched source, and what it cost."""

    confirmed: bool | None
    evidence: dict[str, bool | None]
    decision: str
    downgraded: bool


def _applied_patch_result(
    payload: dict[str, Any],
    *,
    session_dir: str,
    preapplied_git_patch: bool,
) -> dict[str, Any]:
    # A controller publication is git-applied to the worktree before the validator
    # runs, so its final bytes are already on disk; apply reads them as a snapshot
    # instead of mistaking the diff for replacement source. Skipping the apply
    # outright would also skip the cache invalidation, multi-node fan-out and
    # rebuild the measurement depends on. Only an in-process caller can set this:
    # an agent's integrate params land in ``payload`` verbatim, so the payload
    # cannot carry the trust.
    return _maybe_apply_kernel_patch(
        _preapplied_snapshot_payload(payload) if preapplied_git_patch else payload,
        session_dir=session_dir,
        kernel_id=payload.get("kernel_id"),
    )


def _integrate_failure(
    payload: dict[str, Any],
    apply_result: dict[str, Any],
    *,
    error_class: str,
    error: str,
    decision: str,
    **extra: Any,
) -> HandlerResult:
    return {
        "status": "failed",
        "error_class": error_class,
        "error": error,
        "decision": decision,
        "kernel_id": payload.get("kernel_id"),
        "patch_path": payload.get("patch_path"),
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "apply_result": apply_result,
        "revert_result": _maybe_revert_kernel_patch(apply_result),
        **extra,
    }


def _unmeasured_integration_failure(
    payload: dict[str, Any],
    *,
    apply_result: dict[str, Any],
    bench_result: Any,
) -> HandlerResult:
    rebaseline_error_class = (
        str((bench_result or {}).get("error_class") or "").strip() if isinstance(bench_result, dict) else ""
    ) or "bench_exception"
    # A run the stop class claims measured nothing, so the patch has no verdict to answer for.
    stopped = stopped_by_the_run_class(rebaseline_error_class)
    return _integrate_failure(
        payload,
        apply_result,
        error_class=stopped.error_class if stopped is not None else rebaseline_error_class,
        error=stopped.interrupted if stopped is not None else "re-baseline did not succeed",
        decision="NEEDS_REVIEW" if stopped is not None else "REVERT",
        rebaseline_detail=bench_result,
    )


def _confirm_source_import(
    payload: dict[str, Any],
    *,
    workspace: Any,
    decision: str,
) -> _SourceImport:
    if not _honest_flag("HL_CONFIRM_SOURCE_IMPORTED"):
        return _SourceImport(confirmed=None, evidence={}, decision=decision, downgraded=False)
    # Grade the whole write set; the single target is the fallback for a
    # patch whose bundle declared none.
    written = [str(path) for path in (payload.get("patch_write_paths") or []) if str(path or "").strip()]
    if not written:
        written = [str(payload.get("target_file") or payload.get("source_file") or "")]
    confirmed, evidence = _confirm_sources_imported(written, workspace)
    downgraded = (
        decision == "KEEP" and confirmed is False and _honest_flag("HL_CONFIRM_SOURCE_IMPORTED_STRICT")
    )
    return _SourceImport(
        confirmed=confirmed,
        evidence=evidence,
        decision="NEEDS_REVIEW" if downgraded else decision,
        downgraded=downgraded,
    )


class _IntegrateRebaseline(NamedTuple):
    """The re-baseline run that measures one patched build end to end."""

    workspace: Path
    server_args: str
    executor: Any
    ctx: Any


class _IntegrateVerdict(NamedTuple):
    """Everything an integrate reports about the candidate it measured."""

    payload: dict
    bench_result: dict[str, Any]
    performance: Any
    cleanup: dict[str, Any]
    accuracy_gate: dict[str, Any] | None
    source_import: _SourceImport
    rebuild_check: HandlerResult
    server_args: str


def _request_overrides_serving_controls(payload: Mapping[str, Any]) -> bool:
    return (
        bool(to_str_list(payload.get("remove_args")))
        or bool(to_str_list(payload.get("unset_envs")))
        or str(payload.get("args_mode") or "append").strip().lower() == "replace"
    )


def _integrate_request_refusal(payload: dict, *, requested_controls: bool) -> HandlerResult | None:
    if payload.get("_vendor_playbook_deploy_blocked"):
        # A vendor-playbook KEEP (e.g. mori dispatch/combine launch-config
        # tuning) has no deployable artifact: best_artifact_path is a copy of
        # a KernelForge task-bundle config file, not a rewrite of the real
        # installed operator, and apply_kernel_patch's legacy full-file
        # replace would happily overwrite the real site-packages module with
        # it (PR #1191 review finding #1). Refuse before touching the
        # filesystem rather than letting a config-file copy silently
        # corrupt a live install.
        return {
            "status": "failed",
            "error_class": "vendor_playbook_not_deployable",
            "error": (
                "integrate refused: kernel_id="
                f"{payload.get('kernel_id')!r} is a vendor-playbook result "
                "(closed-source operator launch-config tuning); it has no "
                "deployable artifact and must not be applied as a source patch"
            ),
            "decision": "NEEDS_REVIEW",
            "kernel_id": payload.get("kernel_id"),
        }
    if float(payload.get("base_tput", 0.0)) <= 0:
        return {
            "status": "failed",
            "error": "integrate_handler requires base_tput > 0 to compute KEEP/REVERT",
        }
    paired_measurement = _is_paired_gemm_measurement(payload)
    if paired_measurement and any(
        payload.get(key) for key in ("patch_path", "target_file", "source_file", "snapshot_dir")
    ):
        raise ValueError("GEMM paired measurement cannot apply a kernel artifact")
    if _integrate_mode(payload) != "env_only":
        return None
    if (
        payload.get("extra_envs")
        or str(payload.get("extra_server_args") or "").strip()
        or requested_controls
        or paired_measurement
    ):
        return None
    return {
        "status": "failed",
        "error_class": "env_only_missing_envs",
        "error": "env_only integrate requires runtime configuration or an explicit paired reference",
    }


def _unapplied_patch_failure(payload: dict, apply_result: dict) -> HandlerResult | None:
    status = apply_result.get("status")
    if status == "ok":
        return None
    if status == "failed":
        apply_reason = str(apply_result.get("error") or apply_result.get("reason") or "").strip()
        error_class = "apply_failed"
        error = f"kernel patch apply failed: {apply_reason}" if apply_reason else "kernel patch apply failed"
    else:
        error_class = "patch_not_applied"
        error = "kernel patch was not applied; refusing to run E2E benchmark"
    return {
        "status": "failed",
        "error_class": error_class,
        "error": error,
        "decision": "REVERT",
        "apply_result": apply_result,
        "kernel_id": payload.get("kernel_id"),
        "patch_path": payload.get("patch_path"),
        "target_file": payload.get("target_file") or payload.get("source_file"),
    }


def _integrate_rebaseline(
    payload: dict,
    state: Any,
    *,
    session_dir: Path,
    performance_policy: dict[str, float],
) -> _IntegrateRebaseline:
    from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id
    from hyperloom.inference_optimizer.session.session_paths import fs_safe_id, unique_runs_dir
    from ..actions.executors.baseline import BaselineExecutor
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    # VRAM barrier (HL_HONEST_E2E umbrella, default ON; opt out with
    # HL_HONEST_E2E=0 or HL_INTEGRATE_VRAM_GUARD=0): cap re-baseline util on
    # vLLM so the integrate server cannot OOM on a tighter node.
    extra_args = _vram_guarded_server_args(str(payload.get("extra_server_args") or "").strip())
    # A fusion sibling's kernel_id is its operator name (``llm:<recipe>``), which
    # only reaches this handler now that fusion lands through the generic queue.
    # It is a legal id but not a legal directory name everywhere -- fold it.
    task_id = f"integrate-{fs_safe_id(payload.get('kernel_id'))}"
    workspace = unique_runs_dir(session_dir, "integrate", task_id)
    keep_policy = (
        {} if _is_paired_gemm_measurement(payload) else {"post_measure_accuracy_keep_policy": performance_policy}
    )
    task = Task(
        task_id=task_id,
        kind="baseline",
        state="running",
        params={
            "config_path": payload.get("config_path"),
            "output_dir": str(workspace),
            "extra_server_args": extra_args,
            "extra_envs": dict(payload.get("extra_envs") or {}),
            "remove_args": to_str_list(payload.get("remove_args")),
            "unset_envs": to_str_list(payload.get("unset_envs")),
            "args_mode": str(payload.get("args_mode") or "append"),
            # The only artifact that patches FlyDSL sources, so the only run that
            # needs the JIT cache key widened.
            "flydsl_source_dirs": (str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND),
            "defer_accuracy_until_after_measure": True,
            "post_measure_accuracy_min_tput": performance_policy["base_tput"]
            * (1.0 + performance_policy["keep_threshold_pct"] / 100.0),
            **keep_policy,
            # Synthetic kind="baseline": candidate A/B validation against the
            # already-anchored reference. It runs eval for the kernel accuracy
            # gate but never establishes a replacement quality reference.
            "quality_ref_exempt": True,
            # A sub-step of the KERNEL phase's own event, not a dispatched
            # measurement, so it records into that event rather than leaving a
            # baseline event of its own.
            INLINE_EVENT_PARAM: kernel_event_id(int(getattr(state, "macro_cycle", 0) or 0)),
        },
        idempotency_key=f"{task_id}-rebaseline",
    )
    return _IntegrateRebaseline(
        workspace=workspace,
        server_args=extra_args,
        executor=BaselineExecutor(session_dir=session_dir, shared_state=state),
        ctx=RunnerContext(task=task, lease=None),
    )


@contextlib.contextmanager
def _forced_cpp_itfs_rebuild(apply_result: dict) -> Iterator[None]:
    if not (apply_result.get("cpp_itfs_cache_backup") or {}).get("is_cpp_itfs"):
        yield
        return
    previous = os.environ.get("AITER_REBUILD")
    os.environ["AITER_REBUILD"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AITER_REBUILD", None)
        else:
            os.environ["AITER_REBUILD"] = previous


async def _multi_node_restart_failure(
    rebaseline: _IntegrateRebaseline,
    payload: dict,
    apply_result: dict,
) -> HandlerResult | None:
    from ..actions.executors._multi_node_env import is_multi_node

    if not is_multi_node():
        return None
    from ..actions.executors._multi_node_server_lifecycle import (
        ServerRestartFailed,
        restart_server_for_round,
    )

    try:
        await restart_server_for_round(
            extra_server_args=rebaseline.server_args,
            framework=os.environ.get("FRAMEWORK") or None,
            model_path=(str(payload.get("model_path") or "").strip() or os.environ.get("MODEL_PATH") or None),
            tp=int(os.environ.get("TP") or 0) or None,
            ep=int(os.environ.get("EP") or 0) or None,
            force_full_restart=True,
        )
    except ServerRestartFailed as exc:
        return _integrate_failure(
            payload,
            apply_result,
            error_class="mn_server_restart_failed_post_patch",
            error=str(exc),
            decision="REVERT",
        )
    ctx = rebaseline.ctx
    ctx.extra = {**(getattr(ctx, "extra", None) or {}), "mn_round_restarted": True}
    return None


def _cpp_itfs_rebuild_check(apply_result: dict) -> HandlerResult:
    from ..actions.executors._multi_node_env import is_multi_node

    cpp_itfs_backup = apply_result.get("cpp_itfs_cache_backup") or {}
    if not cpp_itfs_backup.get("is_cpp_itfs") or is_multi_node():
        return {"verified": True, "status": "skipped"}
    return apply_kernel_patch.verify_cpp_itfs_rebuilt(cpp_itfs_backup)


def _applyback_verdict_pending(payload: Mapping[str, Any]) -> bool:
    return (
        str(payload.get("artifact_kind") or "") == _FRAMEWORK_APPLYBACK_ARTIFACT_KIND
        and str(payload.get("integration_validation_status") or "") != "passed"
    )


log = logging.getLogger(__name__)


# Mirrors the completion ceiling the inferencex eval shim installs; kept in sync
# so the feasibility check reasons about the budget the eval will really ask for.
_EVAL_DEFAULT_MAX_TOKENS = 4096


def _eval_generation_budget() -> int:
    """Completion tokens the eval harness reserves per sample.

    Mirrors the clamp installed by the inferencex shim: ``HYPERLOOM_EVAL_MAX_TOKENS`` when it parses as a
    positive integer, else the shim's own default.
    """
    raw = (os.environ.get("HYPERLOOM_EVAL_MAX_TOKENS") or "").strip()
    if not raw:
        return _EVAL_DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _EVAL_DEFAULT_MAX_TOKENS
    return value if value >= 0 else _EVAL_DEFAULT_MAX_TOKENS


def _integrate_eval_score(bench_result: dict[str, Any], workspace: Path) -> dict[str, Any]:
    from ..actions.executors._accuracy_gate import parse_eval_results

    measured = bench_result.get("accuracy")
    attached = {
        "accuracy": float(measured) if isinstance(measured, (int, float)) else None,
        "task": str(bench_result.get("accuracy_task") or ""),
        "metric": str(bench_result.get("accuracy_metric") or ""),
        "source_file": str(bench_result.get("accuracy_source") or ""),
    }
    if attached["accuracy"] is not None:
        return attached
    eval_out = parse_eval_results(workspace, framework=os.environ.get("FRAMEWORK") or None)
    parsed = eval_out.get("accuracy")
    if not isinstance(parsed, (int, float)):
        return attached
    return {
        "accuracy": float(parsed),
        "task": str(eval_out.get("task") or ""),
        "metric": str(eval_out.get("metric") or ""),
        "source_file": str(eval_out.get("source_file") or ""),
    }


def _grade_integrate_accuracy(
    bench_result: dict[str, Any],
    *,
    baseline_accuracy: float,
    workspace: Path,
    strict: bool = False,
    server_args: str = "",
) -> dict[str, Any]:
    """Grade a kernel re-baseline's accuracy against the session baseline.

    The staged re-baseline already ran the serving eval after hot throughput passed, so the score is read
    back rather than re-measured.
    """
    from ..actions.executors._accuracy_gate import (
        accuracy_keep_block,
        accuracy_passed,
        require_kernel_accuracy_default,
        resolve_served_context,
        served_context_hosts_eval,
    )

    scored = _integrate_eval_score(bench_result, workspace)
    new_accuracy = scored["accuracy"]

    accuracy_pass: bool | None = None
    if new_accuracy is not None and baseline_accuracy > 0:
        accuracy_pass = accuracy_passed(baseline_accuracy, new_accuracy)

    blocked, reason, degraded = accuracy_keep_block(
        accuracy_pass,
        required=True if strict else require_kernel_accuracy_default(),
        baseline_accuracy=baseline_accuracy,
    )
    if strict and degraded:
        blocked = True
        reason = "accuracy gate produced no eval result and this artifact has no other end-to-end correctness evidence"
    # A verdict can be missing because the eval broke, or because the serving
    # configuration cannot answer an eval request at all. Only the first says
    # anything about the patch. The second reproduces on every retry, so
    # charging it to the patch discards a kernel over a configuration choice.
    infeasible = False
    if accuracy_pass is None:
        fits, why = served_context_hosts_eval(
            served_max_model_len=resolve_served_context(
                server_args=server_args,
                env_max_model_len=os.environ.get("MAX_MODEL_LEN", 0),
            ),
            eval_max_tokens=_eval_generation_budget(),
        )
        if not fits:
            infeasible = True
            reason = why
            log.warning(
                "integrate_handler: the accuracy gate cannot run under this "
                "serving configuration, so no kernel can clear it until the "
                "configuration changes: %s",
                why,
            )
    log.info(
        "integrate_handler: accuracy gate pass=%s blocked=%s degraded=%s new=%s baseline=%.4f source=%s",
        accuracy_pass,
        blocked,
        degraded,
        "n/a" if new_accuracy is None else f"{new_accuracy:.4f}",
        baseline_accuracy,
        scored["source_file"] or "none",
    )
    return {
        "blocked": blocked,
        "accuracy_pass": accuracy_pass,
        "reason": reason,
        "degraded": degraded,
        "infeasible": infeasible,
        "baseline_accuracy": baseline_accuracy,
        **scored,
    }


def _integrate_accuracy_verdict(
    payload: dict,
    bench_result: dict[str, Any],
    rebaseline: _IntegrateRebaseline,
    performance: Any,
    baseline_accuracy: float,
) -> tuple[dict[str, Any] | None, str]:
    if performance.decision != "KEEP":
        return None, performance.decision
    gate = _grade_integrate_accuracy(
        bench_result,
        baseline_accuracy=baseline_accuracy,
        workspace=rebaseline.workspace,
        strict=_applyback_verdict_pending(payload),
        server_args=rebaseline.server_args,
    )
    if not gate["blocked"] or gate["infeasible"]:
        return gate, "KEEP"
    # A measured regression is hard negative evidence -> REVERT. A missing
    # verdict is only an evidence gap -> NEEDS_REVIEW.
    return gate, ("REVERT" if gate["accuracy_pass"] is False else "NEEDS_REVIEW")


def _settled_patch(decision: str, apply_result: dict, *, caller_owns_durability: bool) -> dict[str, Any]:
    if decision != "KEEP":
        finalize_result: HandlerResult = {"status": "skipped", "reason": "non-KEEP decision"}
    elif caller_owns_durability:
        finalize_result = {"status": "skipped", "reason": "caller owns the KEEP's durability"}
    else:
        finalize_result = _maybe_finalize_kernel_patch(apply_result)
    revert_result = (
        {"status": "skipped", "reason": "KEEP decision"}
        if decision == "KEEP"
        else _maybe_revert_kernel_patch(apply_result)
    )
    top_status, patch_cleanup_status, patch_cleanup_action = _cleanup_verdict(
        decision=decision,
        revert_result=revert_result,
        finalize_result=finalize_result,
        revert_required=decision != "KEEP" and bool(apply_result.get("manifest_path")),
    )
    return {
        "status": top_status,
        "decision": decision,
        "patch_cleanup_status": patch_cleanup_status,
        "patch_cleanup_action": patch_cleanup_action,
        "apply_result": apply_result,
        "revert_result": revert_result,
        "finalize_result": finalize_result,
    }


def _integrate_result(verdict: _IntegrateVerdict) -> HandlerResult:
    payload = verdict.payload
    bench_result = verdict.bench_result
    result: dict[str, Any] = {
        **verdict.cleanup,
        "kernel_id": payload.get("kernel_id"),
        "patch_path": payload.get("patch_path"),
        "target_file": payload.get("target_file") or payload.get("source_file"),
        "base_tput": float(payload.get("base_tput", 0.0)),
        "new_tput": float(bench_result.get("output_throughput") or 0.0),
        "gain_pct": verdict.performance.gain_pct,
        "graded_objective": verdict.performance.graded.objective,
        "bench_result": bench_result,
        "report_path": bench_result.get("report_path"),
        "workspace": bench_result.get("workspace"),
        "extra_server_args": verdict.server_args,
        "extra_envs": dict(payload.get("extra_envs") or {}),
        "rebuild_check": verdict.rebuild_check,
        "task_group_key": str(payload.get("task_group_key") or ""),
        "identity_route": str(payload.get("identity_route") or ""),
        "integration_id": str(payload.get("integration_id") or ""),
    }
    # Fusion provenance rides through so the KEEP writeback lifts the stack row as
    # ``fusion`` (not ``integrate``) and sets ``last_fusion_integrate`` -- both are
    # read by the idempotency short-circuit and the remote-recipe fusion export.
    if str(payload.get("source") or "") == "forge_fusion":
        result["source"] = "forge_fusion"
        result["action_label"] = str(payload.get("action_label") or "fusion")
    if _integrate_mode(payload) == "env_only":
        result["advisory"] = (
            "env_only mode: the E2E benchmark measured the configuration change only, not a code patch."
        )
    if result["status"] == "failed":
        result["error_class"] = "patch_revert_incomplete"
        result["error"] = str(result["revert_result"].get("error") or "Kernel patch revert did not complete")
    _explain_integrate_decision(result, verdict)
    return result


def _explain_integrate_decision(result: HandlerResult, verdict: _IntegrateVerdict) -> None:
    performance = verdict.performance
    graded = performance.graded
    source_import = verdict.source_import
    if graded.graded_on_intvty and graded.verdict == "REVERT":
        result["decision_reason"] = "intvty_regression"
    keep_threshold_pct = float(verdict.payload.get("keep_threshold_pct", 1.0))
    if performance.stack_positive_keep and performance.gain_pct <= keep_threshold_pct:
        result["decision_reason"] = "stack_positive_increment"
        result["stack_incremental_gain_pct"] = performance.stack_incremental_gain_pct
        result["stack_incremental_keep_threshold_pct"] = STACK_INCREMENTAL_KEEP_THRESHOLD_PCT
    if source_import.confirmed is not None:
        result["source_import_confirmed"] = source_import.confirmed
    if len(source_import.evidence) > 1:
        result["source_import_evidence"] = source_import.evidence
    if source_import.downgraded:
        result["decision_reason"] = "source_not_confirmed_imported"
    gate = verdict.accuracy_gate
    if gate is not None:
        result["accuracy"] = gate["accuracy"]
        result["baseline_accuracy"] = gate["baseline_accuracy"]
        result["accuracy_pass"] = gate["accuracy_pass"]
        result["accuracy_gate"] = gate
        if gate["blocked"]:
            result["decision_reason"] = (
                "accuracy_regression" if gate["accuracy_pass"] is False else "accuracy_evidence_missing"
            )
    if not _applyback_verdict_pending(verdict.payload):
        return
    result["artifact_kind"] = _FRAMEWORK_APPLYBACK_ARTIFACT_KIND
    # Only a KEEP settles the outstanding verdict. A non-KEEP is left unstamped:
    # the attempt ledger already distinguishes a rejection from a retryable
    # fault, and this field must not blur the two.
    if result["decision"] == "KEEP":
        result["integration_validation_status"] = "passed"
        result["validation_tier"] = _INTEGRATE_ACCURACY_VALIDATION_TIER


async def integrate_handler(
    payload: dict,
    *,
    session_dir: Path,
    preapplied_git_patch: bool = False,
) -> HandlerResult:
    """Apply a kernel patch + re-baseline + KEEP/REVERT decision.

    Applies an optimized kernel artifact, re-runs the active Magpie baseline,
    and KEEPs only when measured E2E throughput clears the threshold AND the
    re-baseline's accuracy holds (source + artifacts are backed up first so
    non-KEEP can restore without a rebuild). Accuracy is graded only for a
    candidate that already cleared the throughput bar -- see
    :func:`_grade_integrate_accuracy`.

    Payload: ``base_tput`` must be > 0 at decision time, but is auto-filled from
    SharedState when a baseline has been recorded, so a bare ``{kernel_id}`` (or
    ``{integration_id}``) payload is accepted. Optional: patch_path,
    target_file, snapshot_dir, kernel_repo, config_path, extra_server_args,
    extra_envs, source, task_group_key, keep_threshold_pct (1.0). The baseline
    benchmark launch applies the invocation's fixed benchmark timeout policy.

    Returns the KEEP/REVERT decision and re-baseline metrics (``status``,
    ``decision``, ``base_tput``, ``new_tput``, ``gain_pct``, ``kernel_id``,
    ``patch_path``, ``report_path``, ``workspace``), plus ``accuracy`` /
    ``baseline_accuracy`` / ``accuracy_pass`` / ``accuracy_gate`` when the gate
    was graded. ``base_tput`` / ``new_tput`` remain output throughput;
    ``gain_pct`` follows ``graded_objective`` and ``bench_result`` retains the
    E2E measurement.
    """
    from ..actions.executors._accuracy_gate import EVAL_KIND_CONTEXT_TOO_SMALL
    from ..actions.executors.benchmark_result import is_valid_measurement
    from ..measurement.integrate_performance import assess_integrate_performance
    from ..state.shared_state import SharedState

    requested_controls = _request_overrides_serving_controls(payload)
    # Fill defaults from SharedState before the ``base_tput > 0`` check so a bare
    # {kernel_id} payload isn't failed with a phantom "missing base_tput".
    payload = _fill_integrate_defaults_from_state(payload, session_dir=session_dir)
    refusal = _integrate_request_refusal(payload, requested_controls=requested_controls)
    if refusal is not None:
        return refusal

    # Coordinator-internal: config-only measurements opt in explicitly; agent
    # integrate requests retain main's patch-resolution contract.
    mode = _integrate_mode(payload)
    paired_measurement = _is_paired_gemm_measurement(payload)
    if mode != "env_only":
        payload, missing_inputs = _resolve_integrate_payload(payload, session_dir=session_dir)
        if missing_inputs is not None:
            return missing_inputs

    state = SharedState.load_or_init(session_dir)
    apply_result = _applied_patch_result(payload, session_dir=session_dir, preapplied_git_patch=preapplied_git_patch)
    if mode == "env_only" and apply_result.get("status") == "skipped":
        apply_result = {
            "status": "ok",
            "reason": "env_only_no_patch_applied",
            "kernel_id": payload.get("kernel_id"),
        }
    log.info("integrate_handler: apply_result=%s", apply_result)
    unapplied = _unapplied_patch_failure(payload, apply_result)
    if unapplied is not None:
        return unapplied

    base_tput = float(payload.get("base_tput", 0.0))
    performance_policy = {
        "base_tput": base_tput,
        "keep_threshold_pct": float(payload.get("keep_threshold_pct", 1.0)),
        "stack_incremental_keep_threshold_pct": STACK_INCREMENTAL_KEEP_THRESHOLD_PCT,
    }
    rebaseline = _integrate_rebaseline(payload, state, session_dir=session_dir, performance_policy=performance_policy)
    with _forced_cpp_itfs_rebuild(apply_result):
        # The sweep runs even when the regular JIT cache is warm: cpp_itfs attention
        # uses the separate AITER_ROOT_DIR/build tree and can carry a stale baton
        # from a timed-out Forge driver.
        _sweep_integrate_aiter_locks(reason="integrate server startup")
        restart_failure = await _multi_node_restart_failure(rebaseline, payload, apply_result)
        if restart_failure is not None:
            return restart_failure
        try:
            bench_result = await _run_integrate_rebaseline_with_lock_retry(
                rebaseline.executor,
                rebaseline.ctx,
                workspace=rebaseline.workspace,
                reason=f"integrate {payload.get('kernel_id') or 'anonymous'} rebaseline",
            )
        except Exception as exc:  # noqa: BLE001
            return _integrate_failure(
                payload, apply_result, error_class="rebaseline_exception", error=repr(exc), decision="REVERT"
            )
    if not is_valid_measurement(bench_result):
        return _unmeasured_integration_failure(payload, apply_result=apply_result, bench_result=bench_result)

    rebuild_check = _cpp_itfs_rebuild_check(apply_result)
    if not rebuild_check.get("verified", True):
        return _integrate_failure(
            payload,
            apply_result,
            error_class="cpp_itfs_rebuild_not_verified",
            error=(
                "re-baseline did not produce a freshly-built cpp_itfs lib.so; refusing to score a possibly-stale binary"
            ),
            decision="NEEDS_REVIEW",
            rebuild_check=rebuild_check,
        )

    if paired_measurement:
        return {
            "status": "ok",
            "decision": "NEEDS_REVIEW",
            "base_tput": base_tput,
            "new_tput": float(bench_result.get("output_throughput") or 0.0),
            "bench_result": bench_result,
            "workspace": bench_result.get("workspace"),
        }
    performance = assess_integrate_performance(state, bench_result, **performance_policy)
    if performance.graded.degrade_reason:
        log.info("integrate_handler: grading unavailable (%s)", performance.graded.degrade_reason)
    # Read after the re-baseline, not before it: the executor shares this state
    # object and a salvaged sibling accuracy during the run republishes the
    # reference this candidate is graded against.
    accuracy_gate, accuracy_decision = _integrate_accuracy_verdict(
        payload, bench_result, rebaseline, performance, float(state.baseline_accuracy or 0.0)
    )
    if accuracy_gate is not None and accuracy_gate["blocked"] and accuracy_gate["infeasible"]:
        # A gate that cannot run under this serving configuration measured nothing
        # about the patch, so it is an integration fault: faults carry their own
        # budget and never spend one of the attempts a patch gets to prove itself.
        return _integrate_failure(
            payload,
            apply_result,
            error_class=EVAL_KIND_CONTEXT_TOO_SMALL,
            error=accuracy_gate["reason"],
            decision="NEEDS_REVIEW",
            gain_pct=performance.gain_pct,
            accuracy_gate=accuracy_gate,
        )

    # Only the strict sub-flag enforces the confirmation; an unknown never penalizes.
    confirmation = _confirm_source_import(
        payload,
        workspace=bench_result.get("workspace"),
        decision=accuracy_decision,
    )
    return _integrate_result(
        _IntegrateVerdict(
            payload=payload,
            bench_result=bench_result,
            performance=performance,
            cleanup=_settled_patch(confirmation.decision, apply_result, caller_owns_durability=preapplied_git_patch),
            accuracy_gate=accuracy_gate,
            source_import=confirmation,
            rebuild_check=rebuild_check,
            server_args=rebaseline.server_args,
        )
    )


# Kernel-agent programmatic dispatch table.
KERNEL_REQUEST_HANDLERS: dict[str, HandlerFn] = {
    "trace_analyze": trace_analyze_handler,
    # No run_gemm_tuning / run_fusion entries: KernelPhase awaits those handlers directly.
    "integrate": integrate_handler,
    "apply_patch": integrate_handler,  # alias — same flow
}


def get_handler(kind: str) -> HandlerFn | None:
    """Look up the programmatic handler registered for a request kind.

    Args:
        kind (str): The kernel request ``kind`` to resolve.

    Returns:
        HandlerFn | None: The registered handler coroutine function, or
            ``None`` when no handler is registered for ``kind``.
    """
    return KERNEL_REQUEST_HANDLERS.get(kind)


__all__ = [
    "KERNEL_REQUEST_HANDLERS",
    "get_handler",
    "integrate_handler",
    "run_gemm_tuning_handler",
]
