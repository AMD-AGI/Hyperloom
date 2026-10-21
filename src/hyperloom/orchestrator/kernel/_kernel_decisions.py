# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-decision write-owner functions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from kernelforge.knowledge.kb_store.identity.implementation import normalize_operator_name
from kernelforge.knowledge.kb_store.identity.kernel_recipe import KERNEL_RECIPE_PRODUCERS

from hyperloom.common.coerce import to_float
from hyperloom.common.env import env_flag

from .patch_landing import (
    DEFAULT_PATCH_BUDGET,
    INTEGRATING_STACK_ACTIONS,
    VERDICT_STATUSES,
    IntegratedStack,
    clamp_by_budget,
    evict_terminal,
    patch_budget,
    record_matches_task,
    record_source_path,
)
from hyperloom.common.timeutil import now_iso as _now_iso
from ..state.kernel_decision_settings import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
    _MAX_INTEGRATE_FAULT_ATTEMPTS,
    effective_hot_kernel_gpu_pct,
    effective_hot_kernel_min_gpu_pct,
    resolve_hot_kernel_min_gpu_pct,
)

log = logging.getLogger(__name__)

# "Honest E2E" hardening flags.
_HONEST_E2E_UMBRELLA_ENV = "HL_HONEST_E2E"


def _honest_flag(specific_env: str) -> bool:
    """Resolve a per-fix honest-E2E flag against the umbrella flag."""
    return env_flag(specific_env, default=env_flag(_HONEST_E2E_UMBRELLA_ENV, default=True))


def _stable_kernel_task_key(
    *,
    task_group_key: str,
    kernel_id: str,
    source_file: str,
) -> str:
    """Return the persistent task identity; ordinals are fallback-only."""
    key = str(task_group_key or "").strip()
    if key:
        return key
    return json.dumps(
        {
            "version": 1,
            "kind": "legacy-kernel",
            "kernel_id": str(kernel_id or ""),
            "source_file": str(source_file or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _kernel_integration_id(
    *,
    task_key: str,
    source_file: str,
    artifact_path: str,
    artifact_bundle: dict[str, Any],
) -> str:
    """Return an immutable patch identity independent of trace ordinals."""
    payload = {
        "task_key": task_key,
        "source_file": source_file,
        "artifact_path": artifact_path,
        "artifact_bundle": artifact_bundle,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return f"kernel-integration:{digest}"


def _text(value: Any) -> str:
    return str(value or "")


def _promoted_needs_review(entry: dict[str, Any], decision: str, micro_speedup: float) -> bool:
    promotion_threshold = to_float(os.environ.get("HL_VERIFIED_MICRO_PROMOTE_THRESHOLD"))
    return (
        _honest_flag("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW")
        and decision == "NEEDS_REVIEW"
        and _text(entry.get("last_backend")).lower() == "geak"
        and entry.get("last_correctness_passed") is True
        and micro_speedup >= (1.10 if promotion_threshold is None else promotion_threshold)
    )


def _existing_integration_id(
    queue: dict[str, Any],
    *,
    source_file: str,
    artifact_path: str,
    task_key: str = "",
    artifact_bundle: dict[str, Any] | None = None,
) -> str:
    return next(
        (
            candidate_id
            for candidate_id, candidate in queue.items()
            if isinstance(candidate, dict)
            and _text(candidate.get("source_file")) == source_file
            and _text(candidate.get("artifact_path")) == artifact_path
            and (
                artifact_bundle is None
                or (
                    dict(candidate.get("artifact_bundle") or {}) == artifact_bundle
                    and (_text(candidate.get("task_key")) == task_key or bool(artifact_path or artifact_bundle))
                )
            )
        ),
        "",
    )


def _queue_kernel_keep(
    state,
    *,
    task_key: str,
    kernel_id: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Persist one KEEP patch snapshot without coupling it to an ordinal slot."""
    if entry.get("vendor_playbook_deploy_blocked"):
        # A vendor-playbook KEEP has no deployable artifact -- refusing to queue it here means
        # integrate_handler() only runs for an LLM-initiated request that names the kernel_id directly.
        return None
    decision = _text(entry.get("last_decision")).upper()
    micro_speedup = to_float(entry.get("last_micro_speedup")) or 0.0
    if decision != "KEEP" and not _promoted_needs_review(entry, decision, micro_speedup):
        return None
    artifact_path = _text(entry.get("last_artifact_path"))
    artifact_bundle = dict(entry.get("last_artifact_bundle") or {})
    source_file = _text(entry.get("last_source_file"))
    queue = state.pending_kernel_integrations
    integration_id = _existing_integration_id(
        queue, task_key=task_key, source_file=source_file, artifact_path=artifact_path, artifact_bundle=artifact_bundle
    ) or _kernel_integration_id(
        task_key=task_key,
        source_file=source_file,
        artifact_path=artifact_path,
        artifact_bundle=artifact_bundle,
    )
    if integration_id in queue:
        queued = queue[integration_id]
        if isinstance(queued, dict):
            # The patch snapshot is immutable, but trace-local routing metadata must follow the task when ordinals
            # are reassigned on a later profile.
            queued.update(
                task_key=task_key,
                kernel_id=_text(kernel_id or entry.get("current_kernel_id")),
                task_group_key=_text(entry.get("task_group_key") or queued.get("task_group_key")),
                identity_route=_text(entry.get("identity_route") or queued.get("identity_route")),
                legacy_task_group_keys=list(
                    entry.get("legacy_task_group_keys") or queued.get("legacy_task_group_keys") or []
                ),
                trace_gpu_pct=entry.get("last_gpu_pct", queued.get("trace_gpu_pct", 0.0)),
            )
        return queue[integration_id]
    queue[integration_id] = {
        "integration_id": integration_id,
        "task_key": task_key,
        "task_group_key": _text(entry.get("task_group_key")),
        "identity_route": _text(entry.get("identity_route")),
        "legacy_task_group_keys": list(entry.get("legacy_task_group_keys") or []),
        "kernel_id": _text(kernel_id or entry.get("current_kernel_id")),
        "source_file": source_file,
        "artifact_path": artifact_path,
        "artifact_bundle": artifact_bundle,
        "snapshot_dir": _text(entry.get("last_snapshot_dir")),
        "deploy_patch_path": _text(entry.get("last_deploy_patch_path")),
        "deploy_repo_root": _text(entry.get("last_deploy_repo_root")),
        "micro_speedup": micro_speedup,
        "optimization_decision": decision,
        "trace_gpu_pct": entry.get("last_gpu_pct", 0.0),
        "created_at": _text(entry.get("last_ts")) or _now_iso(),
        "status": "pending",
        "correctness_source": _text(entry.get("last_correctness_source")),
        "artifact_kind": _text((entry.get("last_framework_applyback") or {}).get("artifact_kind")),
        "integration_validation_status": _text(entry.get("last_integration_validation_status")),
        "framework_applyback": dict(entry.get("last_framework_applyback") or {}),
    }
    return queue[integration_id]


_FUSION_LANE = "forge_fusion"


def enqueue_nominated_patch(state, *, patch, keep_threshold_pct: float = 3.0) -> dict[str, Any] | None:
    """Queue one self-nominated sibling patch for the shared integrate lane."""
    if not isinstance(getattr(state, "pending_kernel_integrations", None), dict):
        state.pending_kernel_integrations = {}
    kernel_name = _text(getattr(patch, "kernel_name", "")).strip()
    artifact_path = _text(getattr(patch, "patch_path", "")).strip()
    source_file = _text(getattr(patch, "target_file", "")).strip()
    if not artifact_path or not source_file:
        # Nothing to apply / no same-source key to collapse on: refusing to queue keeps a malformed sibling off the
        # serial lane rather than dispatching a patch that can only fail the apply gate.
        return None
    env_flag = _text(getattr(patch, "env_flag", "")).strip()
    fusion_env_flags = {flag: "1" for flag in env_flag.split() if flag}
    micro_speedup = to_float(getattr(patch, "micro_speedup", 0.0)) or 0.0
    queue = state.pending_kernel_integrations
    # The lane owns its own task_key / kernel_id namespace so its integration_ids and kernel-rejection identities
    # stay distinct; the prefix is a state key only -- no consumer reads it (they gate on the ``source`` field), so
    # it cannot mis-route.
    task_key = f"{_FUSION_LANE}:{kernel_name or source_file}"
    integration_id = _existing_integration_id(
        queue, source_file=source_file, artifact_path=artifact_path
    ) or _kernel_integration_id(
        task_key=task_key,
        source_file=source_file,
        artifact_path=artifact_path,
        artifact_bundle={},
    )
    record = {
        "integration_id": integration_id,
        "task_key": task_key,
        "task_group_key": "",
        "identity_route": "",
        "legacy_task_group_keys": [],
        "kernel_id": kernel_name or _FUSION_LANE,
        "source_file": source_file,
        "artifact_path": artifact_path,
        "artifact_bundle": {},
        "snapshot_dir": _text(getattr(patch, "snapshot_dir", "")),
        "deploy_patch_path": artifact_path,
        "deploy_repo_root": _text(getattr(patch, "kernel_repo", "")),
        "base_commit": _text(getattr(patch, "base_commit", "")),
        "micro_speedup": micro_speedup,
        "optimization_decision": "KEEP",
        "trace_gpu_pct": 0.0,
        "created_at": _now_iso(),
        "status": "pending",
        "correctness_source": "",
        "artifact_kind": "",
        "integration_validation_status": "",
        "framework_applyback": {},
        # The generic drain / writeback read these back to lift the KEEP as action="fusion".
        "source": _FUSION_LANE,
        "action_label": "fusion",
        "fusion_env_flags": fusion_env_flags,
        "keep_threshold_pct": float(keep_threshold_pct),
    }
    if integration_id not in queue:
        queue[integration_id] = record
    else:
        # Re-enqueue of the same sibling: refresh the mutable fields (the patch snapshot identity is immutable) so a
        # re-run's env/threshold win.
        queued = queue[integration_id]
        if isinstance(queued, dict):
            if _text(queued.get("status")) in VERDICT_STATUSES:
                # A settled verdict is not re-litigated: reviving it to pending both misreports it and puts it beyond
                # evict_terminal's reach.
                log.info(
                    "nomination re-offers %s patch %s (%s); keeping the verdict, not re-queueing",
                    queued.get("status"),
                    kernel_name or source_file,
                    artifact_path,
                )
                return None
            queued["status"] = "pending"
            queued["micro_speedup"] = micro_speedup
            queued["fusion_env_flags"] = fusion_env_flags
            queued["keep_threshold_pct"] = float(keep_threshold_pct)
            queued["source"] = _FUSION_LANE
            queued["action_label"] = "fusion"
    return queue[integration_id]


def _patch_budget_for(state) -> int:
    """How many sibling patches one round may land, env-overridable."""
    return patch_budget(os.environ.get("HL_KERNEL_PATCH_BUDGET"), default=DEFAULT_PATCH_BUDGET)


def _ensure_kernel_task_state(state) -> None:
    """Initialise the stable ledger and re-queue the KEEPs recorded in it."""
    if not isinstance(getattr(state, "kernel_opt_task_attempts", None), dict):
        state.kernel_opt_task_attempts = {}
    if not isinstance(getattr(state, "pending_kernel_integrations", None), dict):
        state.pending_kernel_integrations = {}
    # The queue's only deletion point.
    state.pending_kernel_integrations = evict_terminal(
        state.pending_kernel_integrations,
        budget=_patch_budget_for(state),
    )
    for task_key, stable_entry in state.kernel_opt_task_attempts.items():
        if not isinstance(stable_entry, dict):
            continue
        _queue_kernel_keep(
            state,
            task_key=task_key,
            kernel_id=str(stable_entry.get("current_kernel_id") or stable_entry.get("kernel_id") or ""),
            entry=stable_entry,
        )


def _pending_record_settled(state, record: dict[str, Any], integrated, attempted_entries: list[dict[str, Any]]) -> bool:
    task_group_key = _text(record.get("task_group_key"))
    task_group_aliases = {str(alias) for alias in (record.get("legacy_task_group_keys") or []) if str(alias)}
    kernel_id = _text(record.get("kernel_id"))
    source_file = record_source_path(record)
    artifact_path = _text(record.get("artifact_path"))
    stable_entry = state.kernel_opt_task_attempts.get(_text(record.get("task_key"))) or {}
    if kernel_id in set(state.rejected_kernel_ids or []) and (
        not task_group_key or bool(stable_entry.get("rejected_reason") if isinstance(stable_entry, dict) else "")
    ):
        return True
    if integrated.covers(
        kernel_ids=(kernel_id,),
        task_group_key=task_group_key,
        source_file=source_file,
        task_group_aliases=task_group_aliases,
    ):
        return True
    return any(
        record_matches_task(
            attempted,
            kernel_id=kernel_id,
            task_group_key=task_group_key,
            source_file=source_file,
            task_group_aliases=task_group_aliases,
        )
        # Match only on a real patch_path: an attempted entry with a blank patch_path would match {"",
        # artifact_path}, and one empty-path attempt would drop the whole sibling family from the pending list.
        and (not artifact_path or _text(attempted.get("patch_path")) == artifact_path)
        for attempted in attempted_entries
    )


def pending_kernel_integration_records(state) -> list[dict[str, Any]]:
    """Return pending KEEP snapshots, preserving patches across ordinal reuse."""
    _ensure_kernel_task_state(state)
    integrated = IntegratedStack.of(state.optimization_stack)
    attempted_entries = [
        entry
        for entry in (state.kernel_integrate_attempts or {}).values()
        if isinstance(entry, dict) and not (entry.get("retryable") and not entry.get("rejected"))
    ]
    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    for integration_id, raw_record in state.pending_kernel_integrations.items():
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        if _text(record.get("status")) not in ("", "pending"):
            continue
        if _pending_record_settled(state, record, integrated, attempted_entries):
            continue
        impact = to_float(record.get("trace_gpu_pct")) or 0.0
        if impact <= 0.0:
            impact = _kernel_trace_impact_pct(state, _text(record.get("kernel_id")))
        record["integration_id"] = _text(record.get("integration_id")) or integration_id
        candidates.append((impact, (to_float(record.get("micro_speedup")) or 0.0), integration_id, record))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Collapse only genuine same-source siblings: two whole-file overwrites of one file cannot both land, so the
    # strongest wins.
    claimed_sources: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for _impact, _micro, _integration_id, record in candidates:
        source_file = record_source_path(record)
        if source_file and source_file in claimed_sources:
            continue
        if source_file:
            claimed_sources.add(source_file)
        deduped.append(record)
    # Cap how many siblings dispatch this round.
    fit, _deferred = clamp_by_budget(deduped, _patch_budget_for(state))
    return fit


def _resolve_kernel_patch_identity(
    state,
    payload: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    """Resolve a kernel patch's identity tuple from a result/intent payload."""
    payload = payload or {}
    kernel_id = str(payload.get("kernel_id") or "")
    patch_path = str(payload.get("patch_path") or payload.get("best_artifact_path") or "")
    # The last_kernel_opt back-fill is a single-patch convenience: it lends the one just-optimized patch to a result
    # that omitted its own path.
    if (
        not patch_path
        and kernel_id
        and not str(payload.get("integration_id") or "")
        and str((state.last_kernel_opt or {}).get("kernel_id") or "") == kernel_id
    ):
        patch_path = str(
            (state.last_kernel_opt or {}).get("best_artifact_path")
            or (state.last_kernel_opt or {}).get("patch_path")
            or ""
        )
    target_file = str(payload.get("target_file") or payload.get("source_file") or "")
    extra_args = str(payload.get("extra_server_args") or "").strip()
    return kernel_id, patch_path, target_file, extra_args


def kernel_patch_key(state, payload: dict[str, Any] | None) -> str:
    """Compute the dedup key for a kernel patch."""
    kernel_id, patch_path, _target_file, extra_args = _resolve_kernel_patch_identity(state, payload)
    if not kernel_id or not patch_path:
        return ""
    return "|".join([kernel_id, patch_path, extra_args])


def find_rejected_kernel_patch(
    state,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Look up a previously-rejected patch matching ``payload``."""
    key = kernel_patch_key(state, payload)
    if not key:
        return None
    for entry in state.rejected_kernel_patches:
        if isinstance(entry, dict) and entry.get("key") == key:
            return entry
    return None


def _stamp_integration_validation(
    state,
    *,
    kernel_id: str,
    task_key: str,
    integration_status: str,
    validation_tier: str,
) -> None:
    """Settle an artifact's outstanding integration verdict in the ledgers."""
    entries = []
    if task_key:
        entries.append((state.kernel_opt_task_attempts or {}).get(task_key))
    if kernel_id:
        entries.append(_entry_by_kernel_id(state, kernel_id))
    for attempt in entries:
        if not isinstance(attempt, dict):
            continue
        attempt["integration_status"] = "integrated"
        if integration_status:
            attempt["last_integration_validation_status"] = integration_status
        if validation_tier:
            attempt["validation_tier"] = validation_tier


def _bind_result_to_pending(state, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    integration_id = _text(result.get("integration_id"))
    pending = (state.pending_kernel_integrations or {}).get(integration_id) if integration_id else None
    if not isinstance(pending, dict):
        return result, None
    # A sibling result may omit its own patch path -- resolve it from the bound record rather than from
    # last_kernel_opt, which would borrow a different sibling's identity and key the ledger wrong.
    if not (result.get("patch_path") or result.get("best_artifact_path")):
        result = {**result, "patch_path": _text(pending.get("artifact_path"))}
    if not (result.get("target_file") or result.get("source_file")):
        result = {**result, "target_file": record_source_path(pending)}
    return result, pending


def _attempt_totals(attempts: list[Any]) -> tuple[float, int]:
    rows = [a for a in attempts if isinstance(a, dict)]
    gains = [a["gain_pct"] for a in rows if isinstance(a.get("gain_pct"), (int, float))]
    return float(max(gains, default=0.0)), sum(1 for a in rows if a.get("is_fault"))


def _record_integration_keep(
    state, result: dict[str, Any], pending_record, kernel_id: str, task_group_key: str
) -> None:
    integration_status = _text(result.get("integration_validation_status"))
    validation_tier = _text(result.get("validation_tier"))
    if isinstance(pending_record, dict):
        pending_record.update(status="integrated", integrated_at=_now_iso())
        if integration_status:
            pending_record["integration_validation_status"] = integration_status
        if validation_tier:
            pending_record["validation_tier"] = validation_tier
    if integration_status or validation_tier:
        _stamp_integration_validation(
            state,
            kernel_id=kernel_id,
            task_key=_pending_task_key(pending_record, task_group_key),
            integration_status=integration_status,
            validation_tier=validation_tier,
        )


def _pending_task_key(pending_record, task_group_key: str) -> str:
    return _text((pending_record.get("task_key") if isinstance(pending_record, dict) else "") or task_group_key)


def _reject_integration(
    state, entry: dict[str, Any], *, reason: str, keep_threshold_pct: float, pending_record
) -> None:
    rejected = {
        "key": entry["key"],
        "kernel_id": entry["kernel_id"],
        "task_group_key": entry["task_group_key"],
        "patch_path": entry["patch_path"],
        "target_file": entry["target_file"],
        "extra_server_args": entry["extra_server_args"],
        "attempt_count": entry["attempt_count"],
        "fault_count": entry["fault_count"],
        "best_gain_pct": entry["best_gain_pct"],
        "keep_threshold_pct": keep_threshold_pct,
        "last_decision": entry["last_decision"],
        "last_error_class": entry["last_error_class"],
        "reason": reason,
        "ts": _now_iso(),
    }
    key, kernel_id, task_group_key = entry["key"], entry["kernel_id"], entry["task_group_key"]
    state.rejected_kernel_patches = [
        r for r in state.rejected_kernel_patches if not (isinstance(r, dict) and r.get("key") == key)
    ]
    state.rejected_kernel_patches.append(rejected)
    # A grouped task's members stay out of ``rejected_kernel_ids``: the ids are synthetic per trace and a member can
    # be re-dispatched under another task.
    if kernel_id and not task_group_key and kernel_id not in state.rejected_kernel_ids:
        state.rejected_kernel_ids.append(kernel_id)
    entry["rejected"] = rejected
    state.kernel_integrate_attempts[key] = entry
    if isinstance(pending_record, dict):
        pending_record.update(status="rejected", rejected_at=_now_iso(), rejected_reason=reason)
    task_key = _pending_task_key(pending_record, task_group_key)
    stable_attempt = (state.kernel_opt_task_attempts or {}).get(task_key) if task_key else None
    if isinstance(stable_attempt, dict):
        stable_attempt.update(
            integration_status="rejected", integration_rejected_reason=reason, integration_rejected_at=_now_iso()
        )


def record_kernel_integrate_result(
    state,
    result: dict[str, Any],
    *,
    max_attempts: int = 3,
    keep_threshold_pct: float = 1.0,
    max_fault_attempts: int | None = None,
) -> dict[str, Any] | None:
    """Persist one integrate E2E result and reject exhausted patch attempts."""
    if max_fault_attempts is None:
        max_fault_attempts = _MAX_INTEGRATE_FAULT_ATTEMPTS

    if not isinstance(result, dict):
        return None
    result, pending_record = _bind_result_to_pending(state, result)
    key = kernel_patch_key(state, result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = _resolve_kernel_patch_identity(state, result)
    task_group_key = _text(
        result.get("task_group_key") or (_entry_by_kernel_id(state, kernel_id) or {}).get("task_group_key")
    )
    pending = pending_record if isinstance(pending_record, dict) else {}
    integration_id = _text(pending.get("integration_id") or result.get("integration_id"))
    identity_route = _text(result.get("identity_route") or pending.get("identity_route"))
    is_fault = state._is_integrate_fault(result)
    entry = dict(state.kernel_integrate_attempts.get(key) or {})
    attempts = list(entry.get("attempts") or [])
    attempts.append(
        {
            "decision": result.get("decision"),
            "status": result.get("status"),
            "error_class": result.get("error_class"),
            "is_fault": is_fault,
            "new_tput": result.get("new_tput"),
            "gain_pct": result.get("gain_pct"),
            "accuracy": result.get("accuracy"),
            "accuracy_pass": result.get("accuracy_pass"),
            "decision_reason": result.get("decision_reason"),
            "artifact_kind": _text(result.get("artifact_kind")),
            "validation_tier": _text(result.get("validation_tier")),
            "workspace": result.get("workspace"),
            "report_path": result.get("report_path"),
            "ts": _now_iso(),
            "cycle": int(getattr(state, "macro_cycle", 0) or 0),
        }
    )
    # Quota accounting: faults and gate verdicts draw from separate budgets.
    best_gain, fault_count = _attempt_totals(attempts)
    verdict_attempt_count = len(attempts) - fault_count
    entry.update(
        {
            "key": key,
            "kernel_id": kernel_id,
            "task_group_key": task_group_key,
            "identity_route": identity_route,
            "integration_id": integration_id,
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_server_args": extra_args,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "fault_count": fault_count,
            "verdict_attempt_count": verdict_attempt_count,
            "best_gain_pct": best_gain,
            "last_decision": result.get("decision"),
            "last_status": result.get("status"),
            "last_error_class": result.get("error_class"),
            "last_was_fault": is_fault,
            "updated_at": _now_iso(),
        }
    )
    # Clear any stale retryable flag; re-set below only for un-exhausted faults.
    entry.pop("retryable", None)
    state.kernel_integrate_attempts[key] = entry

    if result.get("decision") == "KEEP":
        _record_integration_keep(state, result, pending_record, kernel_id, task_group_key)
        return entry

    # Integration fault: never measured fairly.
    if is_fault:
        if fault_count < max_fault_attempts:
            entry["retryable"] = True
            state.kernel_integrate_attempts[key] = entry
            return entry
        reason = f"fault_attempts_exhausted_{max_fault_attempts}"
    else:
        # Gate verdict path: a genuine REVERT, or too many non-fault attempts without a KEEP.
        should_reject = result.get("decision") == "REVERT" or verdict_attempt_count >= max_attempts
        if not should_reject:
            return entry
        reason = (
            "revert_decision" if result.get("decision") == "REVERT" else f"max_e2e_attempts_{max_attempts}_without_keep"
        )
    _reject_integration(
        state, entry, reason=reason, keep_threshold_pct=keep_threshold_pct, pending_record=pending_record
    )
    return entry


def record_gemm_tuning(state, result: dict[str, Any]) -> None:
    """Capture the GEAK GEMM tuning result for sequencing and prompts."""
    if not isinstance(result, dict):
        result = {"status": "failed", "error": "non-dict gemm tuning result"}
    entry = dict(result)
    entry.setdefault("ts", _now_iso())
    state.last_gemm_tuning = entry
    attempts = list(state.gemm_tuning_attempts or [])
    attempts.append(entry)
    state.gemm_tuning_attempts = attempts[-_DEFAULT_ATTEMPTS_HISTORY:]


def _canonical_kernel_recipe_operator(kernel_id: str) -> str:
    """The ``kernel_name`` dimension out of a ``kernel:<producer>:<kernel_name>:...`` id, or ``\"\"`` when ``kernel_id`` is not that scheme.

    forge-loop / flydsl / fusion land their integrations under this six-dimension recipe id (see
    ``kernelforge.knowledge.kb_store.identity.kernel_recipe``), not the roofline trace's synthetic ``kNNN`` id. The ``kernel_name``
    dimension is already ``normalize_operator_name``-clean at write time, so it is returned as-is.
    """
    parts = str(kernel_id or "").split(":")
    if len(parts) != 7 or parts[0] != "kernel" or parts[1] not in KERNEL_RECIPE_PRODUCERS:
        return ""
    return parts[2]


def _forge_loop_entries_by_operator_in_optimization_stack(state) -> dict[str, dict[str, Any]]:
    """Map normalized operator name -> its integrating optimization_stack entry (forge-loop/flydsl/fusion).

    These lanes key their ``optimization_stack`` entries by the long-form recipe id
    (``kernel:forge-loop:<operator>:<framework>:<framework_version>:<backend>:<gpu>``), which never equals a roofline
    trace's synthetic ``kNNN`` kernel_id even though both name the same kernel. Comparing on the operator name — run
    through the same ``normalize_operator_name`` the recipe id was built with — is the one identity the two sides
    share.
    """
    entries: dict[str, dict[str, Any]] = {}
    for e in state.optimization_stack or []:
        if not isinstance(e, dict) or e.get("action") not in INTEGRATING_STACK_ACTIONS:
            continue
        operator = _canonical_kernel_recipe_operator(str(e.get("kernel_id") or ""))
        if operator:
            entries[normalize_operator_name(operator)] = e
    return entries


def _forge_loop_operators_in_optimization_stack(state) -> set[str]:
    """Normalized operator names an integrating kernel-recipe lane has already landed; see the sibling ``_entries`` function."""
    return set(_forge_loop_entries_by_operator_in_optimization_stack(state))


def _kernel_trace_impact_pct(state, kernel_id: str) -> float:
    """Return TraceLens gpu_pct for a kernel_id; unknown kernels sort last."""
    kid = str(kernel_id or "").strip()
    if not kid:
        return 0.0
    trace = state.last_trace_analyze or {}
    for row in trace.get("hot_kernels_top15") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("kernel_id") or "").strip() != kid:
            continue
        try:
            return float(row.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def next_pending_keep_kernel_id(state) -> str:
    """Return next KEEP kernel_id awaiting integrate (\"\" if drained)."""
    pending = pending_keep_kernel_ids(state)
    return pending[0] if pending else ""


def pending_keep_kernel_ids(state) -> list[str]:
    """All KEEP kernel_ids awaiting integrate, sorted impact-first."""
    return [
        str(record.get("kernel_id") or "")
        for record in pending_kernel_integration_records(state)
        if str(record.get("kernel_id") or "")
    ]


def has_keep_pending_integrate(state) -> bool:
    """Whether any KEEP kernel is still awaiting integrate."""
    return bool(next_pending_keep_kernel_id(state))


def index_attempts_by_kernel_id(attempts: Any) -> dict[str, dict]:
    """Re-index a stable-keyed attempt ledger by trace-local ``current_kernel_id``."""
    latest: dict[str, tuple[str, dict]] = {}
    for entry in (attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        kernel_id = str(entry.get("current_kernel_id") or "")
        if not kernel_id:
            continue
        ts = str(entry.get("last_ts") or entry.get("ts") or "")
        if ts >= latest.get(kernel_id, ("", {}))[0]:
            latest[kernel_id] = (ts, entry)
    return {kernel_id: entry for kernel_id, (_ts, entry) in latest.items()}


def _entry_by_kernel_id(state, kernel_id: str) -> dict | None:
    """The stable-ledger entry currently holding ``kernel_id``, or ``None``."""
    return index_attempts_by_kernel_id(state.kernel_opt_task_attempts).get(kernel_id)


def kernel_opt_attempts_count(state) -> int:
    """Number of distinct kernel tasks with recorded kernel_opt attempts."""
    _ensure_kernel_task_state(state)
    return len(state.kernel_opt_task_attempts or {})


_HotRow = tuple[tuple[bool, int, float], str, str, list[str], str, tuple[str, str, float]]


def _hot_kernel_task_groups(task_groups: Any) -> tuple[dict[str, tuple[list[str], str]], dict[str, set[str]]]:
    kid_to_group: dict[str, tuple[list[str], str]] = {}
    group_key_aliases: dict[str, set[str]] = {}
    for group in task_groups:
        if not isinstance(group, dict):
            continue
        members = [str(m) for m in (group.get("kernel_ids") or []) if m]
        group_key = _text(group.get("task_group_key"))
        aliases = {group_key, *[str(alias) for alias in (group.get("legacy_task_group_keys") or []) if str(alias)]}
        group_key_aliases[group_key] = {alias for alias in aliases if alias}
        for member in members:
            kid_to_group[member] = (members, group_key)
    return kid_to_group, group_key_aliases


def _hot_kernel_rows(
    hot: list[Any], min_gpu_pct: float, kid_to_group: dict[str, tuple[list[str], str]]
) -> list[_HotRow]:
    rows: list[_HotRow] = []
    for k in hot:
        if not isinstance(k, dict):
            continue
        if k.get("reusable_native_kernel") is not True:
            continue
        # Bypass path tags a kernel non-dispatchable when its shape is geometry-only (launch_grid/tile_name) and would
        # fail the kernel-opt gate.
        if k.get("shape_dispatchable") is False:
            continue
        gpu_pct = to_float(k.get("gpu_pct")) or 0.0
        # Vendor-playbook groups (mori's dispatch+combine) are gated on the sum of the group's members, not each
        # member's own share, and may pin a per-playbook floor -- see effective_hot_kernel_gpu_pct's docstring.
        if effective_hot_kernel_gpu_pct(k) < effective_hot_kernel_min_gpu_pct(k, min_gpu_pct):
            continue
        kid = _text(k.get("kernel_id"))
        if not kid:
            continue
        src = _text(k.get("source_file"))
        group_info = kid_to_group.get(kid)
        members = sorted(group_info[0]) if group_info else [kid]
        # Identity of the underlying kernel, independent of the synthetic per-row kernel_id.
        identity = (src, _text(k.get("name") or k.get("operation")), gpu_pct)
        rank = int(k.get("tracelens_pitem_rank") or 0)
        order = (rank <= 0, rank, -float(k.get("impact_score") or 0.0))
        rows.append((order, kid, src, members, group_info[1] if group_info else "", identity))
    rows.sort(key=lambda x: x[0])
    return rows


def _deduped_hot_rows(rows: list[_HotRow], top_n: int) -> list[_HotRow]:
    ranked: list[_HotRow] = []
    seen_groups: set[str | tuple[str, ...]] = set()
    seen_identities: set[tuple[str, str, float]] = set()
    for row in rows:
        dedup_key: str | tuple[str, ...] = row[4] or tuple(row[3])
        if dedup_key in seen_groups:
            continue
        # Fallback dedup: when the trace carries no ``task_groups`` metadata every row degenerates to its own group,
        # so the SAME kernel appearing under several synthetic ids (identical source_file+name+gpu_pct, e.g.
        # k001/k002) is treated as several distinct hot kernels.
        identity = row[5]
        if identity[0] and identity[1]:
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
        seen_groups.add(dedup_key)
        ranked.append(row)
    # TraceLens' order: task priority, then member impact. Rows it never ranked (bypass route, injected collectives)
    # follow in emitted order under their own top_n, so they cannot fall behind TraceLens' cap.
    return [r for r in ranked if not r[0][0]][:top_n] + [r for r in ranked if r[0][0]][:top_n]


def _attempt_for_member(attempts: dict[str, Any], member_id: str) -> dict[str, Any]:
    """Ledger entry covering ``member_id``, tolerating synthetic-id churn."""
    for value in attempts.values():
        if not isinstance(value, dict):
            continue
        if member_id in {
            _text(value.get("current_kernel_id")),
            _text(value.get("kernel_id")),
            _text(value.get("task_group_primary_kernel_id")),
        }:
            return value
        for key in ("task_group_kernel_ids", "opfanout_collapsed_ids"):
            if member_id in {str(m) for m in (value.get(key) or []) if m}:
                return value
    return {}


def _member_is_rejected(attempts: dict[str, Any], rejected: set[str], member_id: str) -> bool:
    """True when ``member_id`` (or its ledger twin) is out of play."""
    if member_id in rejected:
        return True
    attempt = _attempt_for_member(attempts, member_id)
    if not attempt:
        return False
    if _text(attempt.get("rejected_reason")).strip():
        return True
    return _text(attempt.get("integration_status")).strip().lower() == "rejected"


def _group_attempt(attempts: dict[str, Any], aliases: set[str]) -> dict[str, Any] | None:
    return next(
        (
            attempt
            for attempt in attempts.values()
            if isinstance(attempt, dict)
            and {_text(attempt.get("stable_task_key")), _text(attempt.get("task_group_key"))} & aliases
        ),
        None,
    )


def _matches_current_task(
    attempts: dict[str, Any], member_id: str, group_key: str, source: str, aliases: set[str]
) -> bool:
    if group_key:
        return _group_attempt(attempts, aliases) is not None
    attempt = _attempt_for_member(attempts, member_id)
    if not attempt:
        return False
    recorded_source = _text(attempt.get("last_source_file"))
    return not source or not recorded_source or source == recorded_source


def untried_hot_reusable_kernels(
    state,
    *,
    min_gpu_pct: float | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Hot kernels still owing a ``kernel_opt`` attempt (reusable, gpu_pct >= min_gpu_pct, untouched); capped to top_n by TraceLens rank, one kernel_id per task_group."""
    info = state.last_trace_analyze or {}
    hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
    if not isinstance(hot, list):
        return []
    if min_gpu_pct is None:
        min_gpu_pct = resolve_hot_kernel_min_gpu_pct()
    if top_n is None:
        try:
            top_n = int(os.environ.get("HYPERLOOM_KERNEL_OPT_GATE_TOP_N", _DEFAULT_HOT_KERNEL_GATE_TOP_N))
        except (TypeError, ValueError):
            top_n = _DEFAULT_HOT_KERNEL_GATE_TOP_N
    kid_to_group, group_key_aliases = _hot_kernel_task_groups(info.get("task_groups") or [])

    integrated = IntegratedStack.of(state.optimization_stack)
    integrated_operators = _forge_loop_operators_in_optimization_stack(state)
    rejected = set(state.rejected_kernel_ids or [])
    _ensure_kernel_task_state(state)
    attempts = state.kernel_opt_task_attempts or {}

    def _still_untried(row: _HotRow) -> bool:
        _order, _kid, src, members, group_key, identity = row
        aliases = group_key_aliases.get(group_key) or {group_key}
        if members and all(
            _member_is_rejected(attempts, rejected, member)
            and _matches_current_task(attempts, member, group_key, src, aliases)
            for member in members
        ):
            return False
        if integrated.covers(
            kernel_ids=members,
            task_group_key=group_key,
            source_file=src,
            task_group_aliases=group_key_aliases.get(group_key),
        ):
            return False
        # A kernel-recipe lane (forge-loop/flydsl/fusion) landed under its own long-form recipe id, which never
        # equals this row's synthetic kNNN id or its trace source_file -- see _forge_loop_operators_in_optimization_stack.
        row_name = _text(identity[1])
        if row_name and normalize_operator_name(row_name) in integrated_operators:
            return False
        stable_attempt = _group_attempt(attempts, aliases) if group_key else None
        if stable_attempt is not None and int(stable_attempt.get("attempts", 0)) > 0:
            return False
        if group_key:
            return True
        # Resolve through ``_attempt_for_member`` rather than comparing ids inline: a row's own id is not the only id
        # it covers.
        return not any(
            _matches_current_task(attempts, member, group_key, src, aliases)
            and int((_attempt_for_member(attempts, member) or {}).get("attempts", 0)) > 0
            for member in members
        )

    rows = _deduped_hot_rows(_hot_kernel_rows(hot, min_gpu_pct, kid_to_group), max(1, int(top_n)))
    return [row[1] for row in rows if _still_untried(row)]
