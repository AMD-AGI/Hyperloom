# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrate request's own inputs, filled in from what the session already knows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from hyperloom_common.coerce import to_str_list
from hyperloom_common.unified_diff import PatchOp, repo_relative_manifest

from ..actions.executors._kernel_agent_tool import HandlerResult
from . import kernel_decisions
from .patch_landing import bundle_belongs_to


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

    records = kernel_decisions.pending_kernel_integration_records(state)
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

    return resolved


def _fill_integrate_snapshot_from_bundle(resolved: dict, bundle: Any) -> None:
    """Backfill integrate inputs from a recorded multi-file artifact bundle.

    A bundle is bound to the sibling that produced it by ``integration_id``: a
    bundle stamped with a different id than the integrate being resolved belongs
    to another sibling of the same nomination round, and merging it would land
    that sibling's write set under this one's integrate. A bundle with no
    ``integration_id`` of its own carries no id to disagree with.
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


def _missing_integration_inputs(resolved: dict) -> HandlerResult | None:
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
    }


def _resolve_integrate_payload(payload: dict, *, session_dir: Path) -> tuple[dict, HandlerResult | None]:
    """Fill integrate inputs from SharedState when Orchestration sends only kernel_id (artifact from the queued integration, source from ``last_trace_analyze``).

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
    pending_record = _pending_integration_record(
        kernel_decisions.pending_kernel_integration_records(state),
        str(resolved.get("integration_id") or ""),
    )
    if pending_record is not None:
        _bind_integrate_to_pending_record(resolved, pending_record)
    kernel_id = str(resolved.get("kernel_id") or "")

    if kernel_id and not (resolved.get("target_file") or resolved.get("source_file")):
        source = _find_selected_kernel_source(state, kernel_id)
        if source:
            resolved["source_file"] = source

    return resolved, _missing_integration_inputs(resolved)


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
