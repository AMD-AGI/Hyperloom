# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-decision write-owner functions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from kernelforge.knowledge.kb_store.identity.implementation import normalize_operator_name
from kernelforge.knowledge.kb_store.identity.kernel_recipe import KERNEL_RECIPE_PRODUCERS

from hyperloom.common.env import env_flag

from ._hot_kernel_scan import HotKernelGate, KernelAttemptLedger, TaskGroups, ranked_hot_rows
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
from hyperloom.common.coerce import to_float
from hyperloom.common.timeutil import now_iso as _now_iso
from ..state.kernel_decision_settings import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _MAX_INTEGRATE_FAULT_ATTEMPTS,
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


def _queued_integration_id(queue: dict[str, Any], is_held_by: Callable[[dict[str, Any]], bool]) -> str:
    """The id of the first queued record that already holds the patch ``is_held_by`` describes, else ``\"\"``."""
    return next(
        (integration_id for integration_id, record in queue.items() if isinstance(record, dict) and is_held_by(record)),
        "",
    )


@dataclass(frozen=True)
class _KeepSnapshot:
    """The immutable patch identity one KEEP offers the integrate lane, with the verdict that produced it."""

    task_key: str
    kernel_id: str
    source_file: str
    artifact_path: str
    artifact_bundle: dict[str, Any]
    decision: str
    micro_speedup: float

    def is_held_by(self, record: dict[str, Any]) -> bool:
        """Whether a queued record already holds this exact patch."""
        return (
            str(record.get("source_file") or "") == self.source_file
            and str(record.get("artifact_path") or "") == self.artifact_path
            and dict(record.get("artifact_bundle") or {}) == self.artifact_bundle
            and (str(record.get("task_key") or "") == self.task_key or bool(self.artifact_path or self.artifact_bundle))
        )


def _verified_micro_promote_threshold() -> float:
    """The micro speedup a verified NEEDS_REVIEW must clear before it queues as a KEEP."""
    return to_float(os.environ.get("HL_VERIFIED_MICRO_PROMOTE_THRESHOLD", "1.10"), default=1.10)


def _promotes_verified_micro(entry: dict[str, Any], *, decision: str, micro_speedup: float) -> bool:
    """Whether a verified GEAK micro win lifts a NEEDS_REVIEW verdict into the integrate queue."""
    return (
        _honest_flag("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW")
        and decision == "NEEDS_REVIEW"
        and str(entry.get("last_backend") or "").lower() == "geak"
        and entry.get("last_correctness_passed") is True
        and micro_speedup >= _verified_micro_promote_threshold()
    )


def _keep_snapshot(entry: dict[str, Any], *, task_key: str, kernel_id: str) -> _KeepSnapshot | None:
    """The patch a ledger entry offers for integration, or ``None`` when its verdict offers none."""
    decision = str(entry.get("last_decision") or "").upper()
    micro_speedup = to_float(entry.get("last_micro_speedup"), default=0.0)
    if decision != "KEEP" and not _promotes_verified_micro(entry, decision=decision, micro_speedup=micro_speedup):
        return None
    return _KeepSnapshot(
        task_key=task_key,
        kernel_id=str(kernel_id or entry.get("current_kernel_id") or ""),
        source_file=str(entry.get("last_source_file") or ""),
        artifact_path=str(entry.get("last_artifact_path") or ""),
        artifact_bundle=dict(entry.get("last_artifact_bundle") or {}),
        decision=decision,
        micro_speedup=micro_speedup,
    )


def _queued_keep_record(snapshot: _KeepSnapshot, entry: dict[str, Any], integration_id: str) -> dict[str, Any]:
    """The queue row one KEEP snapshot is first filed as."""
    applyback = dict(entry.get("last_framework_applyback") or {})
    return {
        "integration_id": integration_id,
        "task_key": snapshot.task_key,
        "task_group_key": str(entry.get("task_group_key") or ""),
        "identity_route": str(entry.get("identity_route") or ""),
        "legacy_task_group_keys": list(entry.get("legacy_task_group_keys") or []),
        "kernel_id": snapshot.kernel_id,
        "source_file": snapshot.source_file,
        "artifact_path": snapshot.artifact_path,
        "artifact_bundle": snapshot.artifact_bundle,
        "snapshot_dir": str(entry.get("last_snapshot_dir") or ""),
        "deploy_patch_path": str(entry.get("last_deploy_patch_path") or ""),
        "deploy_repo_root": str(entry.get("last_deploy_repo_root") or ""),
        "micro_speedup": snapshot.micro_speedup,
        "optimization_decision": snapshot.decision,
        "trace_gpu_pct": entry.get("last_gpu_pct", 0.0),
        "created_at": str(entry.get("last_ts") or _now_iso()),
        "status": "pending",
        "correctness_source": str(entry.get("last_correctness_source") or ""),
        "artifact_kind": str(applyback.get("artifact_kind") or ""),
        "integration_validation_status": str(entry.get("last_integration_validation_status") or ""),
        "framework_applyback": applyback,
    }


def _rebind_queued_routing(queued: Any, snapshot: _KeepSnapshot, entry: dict[str, Any]) -> None:
    """Point a queued patch at the task it now belongs to.

    The patch snapshot is immutable, but trace-local routing metadata must follow the task when ordinals are
    reassigned on a later profile.
    """
    if not isinstance(queued, dict):
        return
    queued["task_key"] = snapshot.task_key
    queued["kernel_id"] = snapshot.kernel_id
    queued["task_group_key"] = str(entry.get("task_group_key") or queued.get("task_group_key") or "")
    queued["identity_route"] = str(entry.get("identity_route") or queued.get("identity_route") or "")
    queued["legacy_task_group_keys"] = list(
        entry.get("legacy_task_group_keys") or queued.get("legacy_task_group_keys") or []
    )
    queued["trace_gpu_pct"] = entry.get("last_gpu_pct", queued.get("trace_gpu_pct", 0.0))


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
    snapshot = _keep_snapshot(entry, task_key=task_key, kernel_id=kernel_id)
    if snapshot is None:
        return None
    queue = state.pending_kernel_integrations
    integration_id = _queued_integration_id(queue, snapshot.is_held_by) or _kernel_integration_id(
        task_key=snapshot.task_key,
        source_file=snapshot.source_file,
        artifact_path=snapshot.artifact_path,
        artifact_bundle=snapshot.artifact_bundle,
    )
    if integration_id not in queue:
        queue[integration_id] = _queued_keep_record(snapshot, entry, integration_id)
    else:
        _rebind_queued_routing(queue[integration_id], snapshot, entry)
    return queue[integration_id]


@dataclass(frozen=True)
class _NominatedPatch:
    """One self-nominated sibling patch, resolved to the identity its lane queues it under.

    Each lane owns its own task_key / kernel_id namespace so the two lanes get distinct integration_ids and distinct
    kernel-rejection identities; the prefix is a state key only -- no consumer reads it (they gate on the ``source``
    field), so it cannot mis-route.
    """

    lane: str
    kernel_name: str
    source_file: str
    artifact_path: str
    snapshot_dir: str
    kernel_repo: str
    base_commit: str
    micro_speedup: float
    env_flag: str
    keep_threshold_pct: float

    @property
    def is_fusion(self) -> bool:
        """Whether this patch lands as ``action="fusion"`` rather than the generic integrate."""
        return self.lane == "fusion"

    @property
    def lane_prefix(self) -> str:
        """The state-key namespace this lane queues under."""
        return "forge_fusion" if self.is_fusion else "forge_rewrite"

    @property
    def task_key(self) -> str:
        """The lane-scoped task this patch is dispatched as."""
        return f"{self.lane_prefix}:{self.kernel_name or self.source_file}"

    @property
    def env_flags(self) -> dict[str, str]:
        """The environment the fusion lane activates the patch with."""
        return {flag: "1" for flag in self.env_flag.split() if flag}

    @property
    def label(self) -> str:
        """How this patch is named in the log."""
        return self.kernel_name or self.source_file

    def is_held_by(self, record: dict[str, Any]) -> bool:
        """Whether a queued record already holds this exact patch."""
        return (
            str(record.get("source_file") or "") == self.source_file
            and str(record.get("artifact_path") or "") == self.artifact_path
        )


def _nominated_patch(patch: Any, *, lane: str, keep_threshold_pct: float) -> _NominatedPatch | None:
    """The queueable identity of a nominated sibling, or ``None`` when it names no patch to apply."""
    artifact_path = str(getattr(patch, "patch_path", "") or "").strip()
    source_file = str(getattr(patch, "target_file", "") or "").strip()
    if not artifact_path or not source_file:
        return None
    return _NominatedPatch(
        lane=lane,
        kernel_name=str(getattr(patch, "kernel_name", "") or "").strip(),
        source_file=source_file,
        artifact_path=artifact_path,
        snapshot_dir=str(getattr(patch, "snapshot_dir", "") or ""),
        kernel_repo=str(getattr(patch, "kernel_repo", "") or ""),
        base_commit=str(getattr(patch, "base_commit", "") or ""),
        micro_speedup=to_float(getattr(patch, "micro_speedup", 0.0), default=0.0),
        env_flag=str(getattr(patch, "env_flag", "") or "").strip(),
        keep_threshold_pct=float(keep_threshold_pct),
    )


def _warn_rewrite_lane_drops_env_flags(nominated: _NominatedPatch) -> None:
    """Warn that the rewrite lane cannot activate a patch's env flags.

    It lands as the generic action="integrate", which does not carry them into the re-baseline.
    """
    if nominated.is_fusion or not nominated.env_flags:
        return
    log.warning(
        "nomination rewrite patch %s carries env_flag %r; rewrite lane cannot "
        "activate it and the KEEP re-baseline will run the un-gated path",
        nominated.label,
        nominated.env_flag,
    )


def _nominated_patch_record(nominated: _NominatedPatch, integration_id: str) -> dict[str, Any]:
    """The queue row one nominated sibling is first filed as."""
    record: dict[str, Any] = {
        "integration_id": integration_id,
        "task_key": nominated.task_key,
        "task_group_key": "",
        "identity_route": "",
        "legacy_task_group_keys": [],
        "kernel_id": nominated.kernel_name or nominated.lane_prefix,
        "source_file": nominated.source_file,
        "artifact_path": nominated.artifact_path,
        "artifact_bundle": {},
        "snapshot_dir": nominated.snapshot_dir,
        "deploy_patch_path": nominated.artifact_path,
        "deploy_repo_root": nominated.kernel_repo,
        "base_commit": nominated.base_commit,
        "micro_speedup": nominated.micro_speedup,
        "optimization_decision": "KEEP",
        "trace_gpu_pct": 0.0,
        "created_at": _now_iso(),
        "status": "pending",
        "correctness_source": "",
        "artifact_kind": "",
        "integration_validation_status": "",
        "framework_applyback": {},
    }
    if nominated.is_fusion:
        record.update(_fusion_lane_fields(nominated))
    return record


def _fusion_lane_fields(nominated: _NominatedPatch) -> dict[str, Any]:
    """The fields the generic drain / writeback read back to lift a fusion KEEP as ``action="fusion"``."""
    return {
        "source": "forge_fusion",
        "action_label": "fusion",
        "fusion_env_flags": nominated.env_flags,
        "keep_threshold_pct": nominated.keep_threshold_pct,
    }


def _reopen_queued_nomination(queued: Any, nominated: _NominatedPatch) -> bool:
    """Re-open a queued sibling for another integrate attempt; false when its settled verdict stands.

    The patch snapshot identity is immutable, so only the mutable fields are refreshed and a re-run's env/threshold
    win. A settled verdict is not re-litigated: reviving it to pending both misreports it and puts it beyond
    ``evict_terminal``'s reach.
    """
    if not isinstance(queued, dict):
        return True
    if str(queued.get("status") or "") in VERDICT_STATUSES:
        log.info(
            "nomination re-offers %s patch %s (%s); keeping the verdict, not re-queueing",
            queued.get("status"),
            nominated.label,
            nominated.artifact_path,
        )
        return False
    queued["status"] = "pending"
    queued["micro_speedup"] = nominated.micro_speedup
    if nominated.is_fusion:
        queued.update(_fusion_lane_fields(nominated))
    return True


def enqueue_nominated_patch(
    state, *, patch, lane: str = "fusion", keep_threshold_pct: float = 3.0
) -> dict[str, Any] | None:
    """Queue one self-nominated sibling patch for the shared integrate lane."""
    if not isinstance(getattr(state, "pending_kernel_integrations", None), dict):
        state.pending_kernel_integrations = {}
    nominated = _nominated_patch(patch, lane=lane, keep_threshold_pct=keep_threshold_pct)
    if nominated is None:
        # Nothing to apply / no same-source key to collapse on: refusing to queue keeps a malformed sibling off the
        # serial lane rather than dispatching a patch that can only fail the apply gate.
        return None
    _warn_rewrite_lane_drops_env_flags(nominated)
    queue = state.pending_kernel_integrations
    integration_id = _queued_integration_id(queue, nominated.is_held_by) or _kernel_integration_id(
        task_key=nominated.task_key,
        source_file=nominated.source_file,
        artifact_path=nominated.artifact_path,
        artifact_bundle={},
    )
    if integration_id not in queue:
        queue[integration_id] = _nominated_patch_record(nominated, integration_id)
    elif not _reopen_queued_nomination(queue[integration_id], nominated):
        return None
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


@dataclass(frozen=True)
class _PendingPatchIdentity:
    """The identities one queued patch is matched on when deciding whether it still needs dispatch."""

    task_key: str
    kernel_id: str
    task_group_key: str
    task_group_aliases: set[str]
    source_file: str
    artifact_path: str

    @classmethod
    def of(cls, record: dict[str, Any]) -> _PendingPatchIdentity:
        """Read one queued record's identities."""
        return cls(
            task_key=str(record.get("task_key") or ""),
            kernel_id=str(record.get("kernel_id") or ""),
            task_group_key=str(record.get("task_group_key") or ""),
            task_group_aliases={str(alias) for alias in (record.get("legacy_task_group_keys") or []) if str(alias)},
            source_file=record_source_path(record),
            artifact_path=str(record.get("artifact_path") or ""),
        )


class _PatchPriority(NamedTuple):
    """The order a queued patch dispatches in: trace impact first, then its measured micro win."""

    trace_gpu_pct: float
    micro_speedup: float


def _dispatch_priority(state, record: dict[str, Any], kernel_id: str) -> _PatchPriority:
    """Where this patch sorts in the round's dispatch order."""
    impact = to_float(record.get("trace_gpu_pct"), default=0.0)
    if impact <= 0.0:
        impact = _kernel_trace_impact_pct(state, kernel_id)
    return _PatchPriority(impact, to_float(record.get("micro_speedup"), default=0.0))


@dataclass(frozen=True)
class _SettledIntegrations:
    """What has already landed on the stack and what has already had its integrate attempt."""

    integrated: IntegratedStack
    attempted: list[dict[str, Any]]

    @classmethod
    def of(cls, state) -> _SettledIntegrations:
        """Read both ledgers once for a whole pass over the pending queue."""
        return cls(
            integrated=IntegratedStack.of(state.optimization_stack),
            attempted=[
                entry
                for entry in (state.kernel_integrate_attempts or {}).values()
                if isinstance(entry, dict) and not (entry.get("retryable") and not entry.get("rejected"))
            ],
        )

    def covers(self, identity: _PendingPatchIdentity) -> bool:
        """Whether this patch has already landed, or already spent the attempt it was queued for."""
        if self.integrated.covers(
            kernel_ids=(identity.kernel_id,),
            task_group_key=identity.task_group_key,
            source_file=identity.source_file,
            task_group_aliases=identity.task_group_aliases,
        ):
            return True
        return any(self._attempt_matches(attempted, identity) for attempted in self.attempted)

    @staticmethod
    def _attempt_matches(attempted: dict[str, Any], identity: _PendingPatchIdentity) -> bool:
        """Whether a settled attempt was this patch's.

        Matched only on a real patch_path: an attempted entry with a blank patch_path would match {"",
        artifact_path}, and one empty-path attempt would drop the whole sibling family from the pending list.
        """
        return record_matches_task(
            attempted,
            kernel_id=identity.kernel_id,
            task_group_key=identity.task_group_key,
            source_file=identity.source_file,
            task_group_aliases=identity.task_group_aliases,
        ) and (not identity.artifact_path or str(attempted.get("patch_path") or "") == identity.artifact_path)


def _patch_kernel_is_rejected(state, identity: _PendingPatchIdentity) -> bool:
    """Whether this patch's kernel is out of play, by its own id or by its task's recorded rejection."""
    if identity.kernel_id not in set(state.rejected_kernel_ids or []):
        return False
    stable_entry = state.kernel_opt_task_attempts.get(identity.task_key) or {}
    return not identity.task_group_key or bool(
        stable_entry.get("rejected_reason") if isinstance(stable_entry, dict) else ""
    )


def _strongest_patch_per_source(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse genuine same-source siblings: two whole-file overwrites of one file cannot both land."""
    claimed_sources: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        source_file = record_source_path(record)
        if source_file and source_file in claimed_sources:
            continue
        if source_file:
            claimed_sources.add(source_file)
        deduped.append(record)
    return deduped


def pending_kernel_integration_records(state) -> list[dict[str, Any]]:
    """Return pending KEEP snapshots, preserving patches across ordinal reuse."""
    _ensure_kernel_task_state(state)
    settled = _SettledIntegrations.of(state)
    candidates: list[tuple[_PatchPriority, dict[str, Any]]] = []
    for integration_id, raw_record in state.pending_kernel_integrations.items():
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        if str(record.get("status") or "pending") != "pending":
            continue
        identity = _PendingPatchIdentity.of(record)
        if _patch_kernel_is_rejected(state, identity) or settled.covers(identity):
            continue
        record["integration_id"] = str(record.get("integration_id") or integration_id)
        candidates.append((_dispatch_priority(state, record, identity.kernel_id), record))
    candidates.sort(key=lambda item: item[0], reverse=True)
    # Cap how many siblings dispatch this round.
    fit, _deferred = clamp_by_budget(
        _strongest_patch_per_source([record for _priority, record in candidates]), _patch_budget_for(state)
    )
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


@dataclass(frozen=True)
class _IntegrateSettlement:
    """One integrate result bound to the queued patch and the ledger identity it settles."""

    result: dict[str, Any]
    pending_record: dict[str, Any] | None
    key: str
    kernel_id: str
    patch_path: str
    target_file: str
    extra_server_args: str
    task_group_key: str
    integration_id: str
    identity_route: str

    @property
    def task_key(self) -> str:
        """The stable-ledger key this settlement writes its verdict under."""
        queued = self.pending_record
        return str((queued.get("task_key") if isinstance(queued, dict) else "") or self.task_group_key or "")


def _with_queued_paths(result: dict[str, Any], pending_record: Any) -> dict[str, Any]:
    """The result with the paths its queued record owns filled in.

    A sibling result may omit its own patch path -- resolving it from the bound record rather than from
    last_kernel_opt is what keeps it from borrowing a different sibling's identity and keying the ledger wrong.
    """
    if not isinstance(pending_record, dict):
        return result
    if not str(result.get("patch_path") or result.get("best_artifact_path") or ""):
        result = {**result, "patch_path": str(pending_record.get("artifact_path") or "")}
    if not str(result.get("target_file") or result.get("source_file") or ""):
        result = {**result, "target_file": record_source_path(pending_record)}
    return result


def _bound_pending_record(state, result: dict[str, Any]) -> Any:
    """The queued record this result names, matched on ``integration_id`` alone."""
    integration_id = str(result.get("integration_id") or "")
    return (state.pending_kernel_integrations or {}).get(integration_id) if integration_id else None


def _integrate_settlement(state, result: dict[str, Any]) -> _IntegrateSettlement | None:
    """Resolve what one integrate result settles, or ``None`` when it names no patch."""
    pending_record = _bound_pending_record(state, result)
    result = _with_queued_paths(result, pending_record)
    key = kernel_patch_key(state, result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = _resolve_kernel_patch_identity(state, result)
    queued = pending_record if isinstance(pending_record, dict) else {}
    return _IntegrateSettlement(
        result=result,
        pending_record=pending_record if isinstance(pending_record, dict) else None,
        key=key,
        kernel_id=kernel_id,
        patch_path=patch_path,
        target_file=target_file,
        extra_server_args=extra_args,
        task_group_key=str(
            result.get("task_group_key") or (_entry_by_kernel_id(state, kernel_id) or {}).get("task_group_key") or ""
        ),
        integration_id=str(queued.get("integration_id") or result.get("integration_id") or ""),
        identity_route=str(result.get("identity_route") or queued.get("identity_route") or ""),
    )


def _integrate_attempt(state, result: dict[str, Any], *, is_fault: bool) -> dict[str, Any]:
    """The one attempt row this integrate result adds to the patch's history."""
    return {
        "decision": result.get("decision"),
        "status": result.get("status"),
        "error_class": result.get("error_class"),
        "is_fault": is_fault,
        "new_tput": result.get("new_tput"),
        "gain_pct": result.get("gain_pct"),
        "accuracy": result.get("accuracy"),
        "accuracy_pass": result.get("accuracy_pass"),
        "decision_reason": result.get("decision_reason"),
        "artifact_kind": str(result.get("artifact_kind") or ""),
        "validation_tier": str(result.get("validation_tier") or ""),
        "workspace": result.get("workspace"),
        "report_path": result.get("report_path"),
        "ts": _now_iso(),
        "cycle": int(getattr(state, "macro_cycle", 0) or 0),
    }


def _record_integrate_attempt(state, settlement: _IntegrateSettlement, *, is_fault: bool) -> dict[str, Any]:
    """Append this result to the patch's attempt ledger and return the ledger entry."""
    result = settlement.result
    entry = dict(state.kernel_integrate_attempts.get(settlement.key) or {})
    attempts = list(entry.get("attempts") or [])
    attempts.append(_integrate_attempt(state, result, is_fault=is_fault))
    best_gain = max(
        (
            float(a.get("gain_pct"))
            for a in attempts
            if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
        ),
        default=0.0,
    )
    # Quota accounting: faults and gate verdicts draw from separate budgets.
    fault_count = sum(1 for a in attempts if isinstance(a, dict) and a.get("is_fault"))
    entry.update(
        {
            "key": settlement.key,
            "kernel_id": settlement.kernel_id,
            "task_group_key": settlement.task_group_key,
            "identity_route": settlement.identity_route,
            "integration_id": settlement.integration_id,
            "patch_path": settlement.patch_path,
            "target_file": settlement.target_file,
            "extra_server_args": settlement.extra_server_args,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "fault_count": fault_count,
            "verdict_attempt_count": len(attempts) - fault_count,
            "best_gain_pct": best_gain,
            "last_decision": result.get("decision"),
            "last_status": result.get("status"),
            "last_error_class": result.get("error_class"),
            "last_was_fault": is_fault,
            "updated_at": _now_iso(),
        }
    )
    # Clear any stale retryable flag; the caller re-sets it only for un-exhausted faults.
    entry.pop("retryable", None)
    state.kernel_integrate_attempts[settlement.key] = entry
    return entry


def _settle_kept_integration(state, settlement: _IntegrateSettlement) -> None:
    """Mark the queued patch and its stable task as integrated."""
    result = settlement.result
    validation_tier = str(result.get("validation_tier") or "")
    integration_status = str(result.get("integration_validation_status") or "")
    queued = settlement.pending_record
    if isinstance(queued, dict):
        queued["status"] = "integrated"
        queued["integrated_at"] = _now_iso()
        if integration_status:
            queued["integration_validation_status"] = integration_status
        if validation_tier:
            queued["validation_tier"] = validation_tier
    if integration_status or validation_tier:
        _stamp_integration_validation(
            state,
            kernel_id=settlement.kernel_id,
            task_key=settlement.task_key,
            integration_status=integration_status,
            validation_tier=validation_tier,
        )


def _rejection_reason(
    result: dict[str, Any],
    entry: dict[str, Any],
    *,
    max_attempts: int,
    max_fault_attempts: int,
) -> str:
    """Why this patch is out of attempts, or ``""`` while it still has one.

    A genuine REVERT retires the patch at once; otherwise it is too many non-fault attempts without a KEEP, and
    faults draw on their own separate quota.
    """
    if entry.get("last_was_fault"):
        if int(entry.get("fault_count") or 0) < max_fault_attempts:
            return ""
        return f"fault_attempts_exhausted_{max_fault_attempts}"
    if result.get("decision") == "REVERT":
        return "revert_decision"
    if int(entry.get("verdict_attempt_count") or 0) >= max_attempts:
        return f"max_e2e_attempts_{max_attempts}_without_keep"
    return ""


def _reject_kernel_patch(
    state,
    settlement: _IntegrateSettlement,
    entry: dict[str, Any],
    reason: str,
    *,
    keep_threshold_pct: float,
) -> None:
    """Retire this patch: record the rejection and close the queued record and its stable task on it."""
    result = settlement.result
    key = settlement.key
    rejected = {
        "key": key,
        "kernel_id": settlement.kernel_id,
        "task_group_key": settlement.task_group_key,
        "patch_path": settlement.patch_path,
        "target_file": settlement.target_file,
        "extra_server_args": settlement.extra_server_args,
        "attempt_count": entry.get("attempt_count"),
        "fault_count": entry.get("fault_count"),
        "best_gain_pct": entry.get("best_gain_pct"),
        "keep_threshold_pct": keep_threshold_pct,
        "last_decision": result.get("decision"),
        "last_error_class": result.get("error_class"),
        "reason": reason,
        "ts": _now_iso(),
    }
    state.rejected_kernel_patches = [
        r for r in state.rejected_kernel_patches if not (isinstance(r, dict) and r.get("key") == key)
    ]
    state.rejected_kernel_patches.append(rejected)
    # A grouped task's members stay out of ``rejected_kernel_ids``: the ids are synthetic per trace and a member can
    # be re-dispatched under another task.
    kernel_id = settlement.kernel_id
    if kernel_id and not settlement.task_group_key and kernel_id not in state.rejected_kernel_ids:
        state.rejected_kernel_ids.append(kernel_id)
    entry["rejected"] = rejected
    state.kernel_integrate_attempts[key] = entry
    queued = settlement.pending_record
    if isinstance(queued, dict):
        queued["status"] = "rejected"
        queued["rejected_at"] = _now_iso()
        queued["rejected_reason"] = reason
    _stamp_task_rejection(state, settlement.task_key, reason)


def _stamp_task_rejection(state, task_key: str, reason: str) -> None:
    """Close the stable task a rejected patch belonged to."""
    if not task_key:
        return
    stable_attempt = (state.kernel_opt_task_attempts or {}).get(task_key)
    if isinstance(stable_attempt, dict):
        stable_attempt["integration_status"] = "rejected"
        stable_attempt["integration_rejected_reason"] = reason
        stable_attempt["integration_rejected_at"] = _now_iso()


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
    settlement = _integrate_settlement(state, result)
    if settlement is None:
        return None
    is_fault = state._is_integrate_fault(settlement.result)
    entry = _record_integrate_attempt(state, settlement, is_fault=is_fault)
    if settlement.result.get("decision") == "KEEP":
        _settle_kept_integration(state, settlement)
        return entry
    reason = _rejection_reason(
        settlement.result, entry, max_attempts=max_attempts, max_fault_attempts=max_fault_attempts
    )
    if not reason:
        if is_fault:
            # Integration fault: never measured fairly, so it is retried rather than counted against the gate.
            entry["retryable"] = True
            state.kernel_integrate_attempts[settlement.key] = entry
        return entry
    _reject_kernel_patch(state, settlement, entry, reason, keep_threshold_pct=keep_threshold_pct)
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
    groups = TaskGroups.of(info.get("task_groups") or [])
    rows = ranked_hot_rows(hot, groups, HotKernelGate.resolve(min_gpu_pct, top_n))
    _ensure_kernel_task_state(state)
    ledger = KernelAttemptLedger.of(state, groups, _forge_loop_operators_in_optimization_stack(state))
    return [row.kernel_id for row in rows if ledger.still_owes_attempt(row)]
