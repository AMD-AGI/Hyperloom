# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-decision write-owner functions.

SharedState is a passive persisted record; the functions that *own kernel
decisions* (recording kernel-opt / integrate / gemm-tuning outcomes,
kernel-patch identity, pending-keep bookkeeping, hot-kernel reuse) live here.
They take ``state`` as their first argument and read/mutate it; SharedState
keeps thin forwarding shims so existing callers keep working.

Also carries the "honest E2E" hardening-flag helper (``_honest_flag`` + its
constants), shared by both this cluster and the request handlers that stayed
in the origin module.

Dependencies on retry/default settings come from
``state.kernel_decision_settings`` so this module does not import
``shared_state``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from hyperloom.common.env import env_bool

from ._recorder_trace import trace_recording_skipped
from .patch_landing import (
    DEFAULT_PATCH_BUDGET,
    clamp_by_budget,
    evict_terminal,
    patch_budget,
    record_source_path,
)
from ..state.kernel_decision_settings import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
    _DEFAULT_KERNEL_OPT_MAX_PARTIAL,
    _MAX_INTEGRATE_FAULT_ATTEMPTS,
    _now_iso,
    effective_hot_kernel_gpu_pct,
    effective_hot_kernel_min_gpu_pct,
    resolve_hot_kernel_min_gpu_pct,
    resolve_kernel_opt_max_failures,
)
from ..trace.trace_env import env_flag


log = logging.getLogger(__name__)

#: Collective primitives the dedicated lane can measure. Each needs an
#: independent reference implementation in the generated driver.
SUPPORTED_COLLECTIVE_OPS = frozenset({"all_reduce", "reduce_scatter", "all_gather"})

#: Batch-filter skip reasons that mean no backend ever saw the kernel. Two
#: readers depend on the same answer -- the dispatcher reports such a skip
#: instead of falling through to its validation guards, and
#: :func:`record_kernel_opt` leaves the attempt ledger alone -- so they read one
#: table.
#:
#: ``not_live_in_flight`` is here and its two siblings are not, which is the
#: whole reason the liveness check reports which of them applies: a kernel held
#: back because a sibling dispatch is in flight has had no backend look at it,
#: while ``not_live_rejected`` and ``not_live_attempts_exhausted`` describe a
#: kernel that spent its attempts. A single ``not_live`` covered all three, so
#: this exemption could not take the first without also retiring the last two --
#: which left the in-flight case charging a kernel for a dispatch that never
#: happened.
_UNATTEMPTED_SKIP_PREFIXES: tuple[str, ...] = (
    "below_min_gpu_pct",
    "group_exhausted",
    "group_in_flight",
    "group_task_complete",
    "not_live_in_flight",
    "opfanout_merged_into",
)


def unattempted_skip_reason(reason: str) -> bool:
    """Whether ``reason`` means the kernel was never handed to a backend."""
    return str(reason or "").startswith(_UNATTEMPTED_SKIP_PREFIXES)


# "Honest E2E" hardening flags. The umbrella flag ``HL_HONEST_E2E`` turns the
# whole mode on; each fix also has a per-fix override that wins over the umbrella
# (set it to an explicit falsey value to opt a single fix out of the umbrella).
_HONEST_E2E_UMBRELLA_ENV = "HL_HONEST_E2E"


def _honest_flag(specific_env: str) -> bool:
    """Resolve a per-fix honest-E2E flag against the umbrella flag.

    Returns ``True`` when the per-fix env ``specific_env`` is truthy, OR when it
    is unset and the umbrella ``HL_HONEST_E2E`` is truthy. An explicit falsey
    per-fix value always wins (lets one fix opt out of the umbrella). The
    umbrella defaults ON. Opt the whole cohort back out with ``HL_HONEST_E2E=0``
    (or a single fix via its per-fix env).

    The per-fix layer uses :func:`trace_env.env_flag` (the canonical superset
    vocabulary that recognizes ``0/false/no/off`` as an *explicit* False and
    falls back to its ``default`` for unset/unrecognized values), so an
    unrecognized per-fix value defers to the umbrella exactly as before. The
    umbrella layer is :func:`common.env.env_bool` (default ON).

    Args:
        specific_env: The per-fix environment variable name.

    Returns:
        bool: Whether the gated behavior should be enabled.
    """
    return env_flag(specific_env, default=env_bool(_HONEST_E2E_UMBRELLA_ENV, True))


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


def _queue_kernel_keep(
    state,
    *,
    task_key: str,
    kernel_id: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Persist one KEEP patch snapshot without coupling it to an ordinal slot."""
    if entry.get("vendor_playbook_deploy_blocked"):
        # A vendor-playbook KEEP has no deployable artifact -- see
        # record_kernel_opt()'s comment (PR #1191 review finding #1).
        # Refusing to queue it here means _auto_enqueue_pending_integrations()
        # never dispatches an integrate for it; integrate_handler() still
        # checks this flag independently for an LLM-initiated request that
        # names the kernel_id directly.
        return None
    decision = str(entry.get("last_decision") or "").upper()
    try:
        micro_speedup = float(entry.get("last_micro_speedup") or 0.0)
    except (TypeError, ValueError):
        micro_speedup = 0.0
    try:
        promotion_threshold = float(
            os.environ.get(
                "HL_VERIFIED_MICRO_PROMOTE_THRESHOLD",
                "1.10",
            )
            or 1.10
        )
    except ValueError:
        promotion_threshold = 1.10
    promoted_needs_review = (
        _honest_flag("HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW")
        and decision == "NEEDS_REVIEW"
        and str(entry.get("last_backend") or "").lower() == "geak"
        and entry.get("last_correctness_passed") is True
        and micro_speedup >= promotion_threshold
    )
    if decision != "KEEP" and not promoted_needs_review:
        return None
    artifact_path = str(entry.get("last_artifact_path") or "")
    artifact_bundle = dict(entry.get("last_artifact_bundle") or {})
    source_file = str(entry.get("last_source_file") or "")
    queue = state.pending_kernel_integrations
    existing_integration_id = next(
        (
            candidate_id
            for candidate_id, candidate in queue.items()
            if isinstance(candidate, dict)
            and str(candidate.get("source_file") or "") == source_file
            and str(candidate.get("artifact_path") or "") == artifact_path
            and dict(candidate.get("artifact_bundle") or {}) == artifact_bundle
            and (str(candidate.get("task_key") or "") == task_key or bool(artifact_path or artifact_bundle))
        ),
        "",
    )
    integration_id = existing_integration_id or _kernel_integration_id(
        task_key=task_key,
        source_file=source_file,
        artifact_path=artifact_path,
        artifact_bundle=artifact_bundle,
    )
    if integration_id not in queue:
        queue[integration_id] = {
            "integration_id": integration_id,
            "task_key": task_key,
            "task_group_key": str(entry.get("task_group_key") or ""),
            "identity_route": str(entry.get("identity_route") or ""),
            "legacy_task_group_keys": list(entry.get("legacy_task_group_keys") or []),
            "kernel_id": str(kernel_id or entry.get("current_kernel_id") or ""),
            "source_file": source_file,
            "artifact_path": artifact_path,
            "artifact_bundle": artifact_bundle,
            "snapshot_dir": str(entry.get("last_snapshot_dir") or ""),
            "deploy_patch_path": str(entry.get("last_deploy_patch_path") or ""),
            "deploy_repo_root": str(entry.get("last_deploy_repo_root") or ""),
            "micro_speedup": micro_speedup,
            "optimization_decision": decision,
            "trace_gpu_pct": entry.get("last_gpu_pct", 0.0),
            "created_at": str(entry.get("last_ts") or _now_iso()),
            "status": "pending",
            "correctness_source": str(entry.get("last_correctness_source") or ""),
            "artifact_kind": str((entry.get("last_framework_applyback") or {}).get("artifact_kind") or ""),
            "integration_validation_status": str(entry.get("last_integration_validation_status") or ""),
            "framework_applyback": dict(entry.get("last_framework_applyback") or {}),
        }
    else:
        # The patch snapshot is immutable, but trace-local routing metadata must
        # follow the task when ordinals are reassigned on a later profile.
        queued = queue[integration_id]
        if isinstance(queued, dict):
            queued["task_key"] = task_key
            queued["kernel_id"] = str(kernel_id or entry.get("current_kernel_id") or "")
            queued["task_group_key"] = str(entry.get("task_group_key") or queued.get("task_group_key") or "")
            queued["identity_route"] = str(entry.get("identity_route") or queued.get("identity_route") or "")
            queued["legacy_task_group_keys"] = list(
                entry.get("legacy_task_group_keys") or queued.get("legacy_task_group_keys") or []
            )
            queued["trace_gpu_pct"] = entry.get(
                "last_gpu_pct",
                queued.get("trace_gpu_pct", 0.0),
            )
    return queue[integration_id]


def enqueue_nominated_patch(state, *, patch, keep_threshold_pct: float = 3.0) -> dict[str, Any] | None:
    """Queue one self-nominated fusion sibling for the shared integrate lane.

    The fusion lane used to integrate inline (apply + e2e re-baseline + KEEP)
    right where the run finished. Under the nomination contract each kept recipe
    is instead written here as one ``status="pending"`` record and drained by the
    SWEEP-entry integrate loop, so the same-file collapse, cross-file
    independence, and patch budget the rewrite/gemm lanes already get
    (``pending_kernel_integration_records``) apply to fusion siblings too.

    Three fusion-specific facts ride on the record because the generic drain and
    writeback cannot infer them:

    * ``fusion_env_flags`` -- the fused path is env-gated; unset, the patch is the
      eager path, so ``_fill_integrate_defaults_from_state`` merges these into the
      re-baseline server's envs or a real win is measured un-fused and REVERTED.
    * ``keep_threshold_pct`` -- fusion keeps its own e2e bar (default 3.0%),
      distinct from the generic integrate default, so the KEEP/REVERT verdict is
      unchanged from the pre-contract inline path.
    * ``action_label="fusion"`` -- the promoted stack row must read ``fusion`` (not
      ``integrate``) or the idempotency short-circuit and the remote-recipe fusion
      export both go blind; ``last_fusion_integrate`` is set off the same signal.

    Eviction-safe: a distinct ``integration_id`` at ``status="pending"`` survives
    ``_ensure_kernel_task_state`` (evict_terminal keeps non-terminal records; the
    re-queue loop only ADDS). Idempotent on ``(source_file, artifact_path)``.

    Args:
        state: SharedState (mutated in place).
        patch: A ``NominatedPatch`` (duck-typed) from ``parse_outcome``.
        keep_threshold_pct: The fusion-specific e2e KEEP bar to carry.

    Returns:
        The queued record, or ``None`` when the patch has no artifact to apply.
    """
    if not isinstance(getattr(state, "pending_kernel_integrations", None), dict):
        state.pending_kernel_integrations = {}
    kernel_name = str(getattr(patch, "kernel_name", "") or "").strip()
    artifact_path = str(getattr(patch, "patch_path", "") or "").strip()
    source_file = str(getattr(patch, "target_file", "") or "").strip()
    if not artifact_path or not source_file:
        # Nothing to apply / no same-source key to collapse on: refusing to queue
        # keeps a malformed sibling off the serial lane rather than dispatching a
        # patch that can only fail the apply gate.
        return None
    env_flag = str(getattr(patch, "env_flag", "") or "").strip()
    fusion_env_flags = {flag: "1" for flag in env_flag.split() if flag}
    try:
        micro_speedup = float(getattr(patch, "micro_speedup", 0.0) or 0.0)
    except (TypeError, ValueError):
        micro_speedup = 0.0
    queue = state.pending_kernel_integrations
    existing_integration_id = next(
        (
            candidate_id
            for candidate_id, candidate in queue.items()
            if isinstance(candidate, dict)
            and str(candidate.get("source_file") or "") == source_file
            and str(candidate.get("artifact_path") or "") == artifact_path
        ),
        "",
    )
    task_key = f"forge_fusion:{kernel_name}" if kernel_name else f"forge_fusion:{source_file}"
    integration_id = existing_integration_id or _kernel_integration_id(
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
        "kernel_id": kernel_name or "forge_fusion",
        "source_file": source_file,
        "artifact_path": artifact_path,
        "artifact_bundle": {},
        "snapshot_dir": str(getattr(patch, "snapshot_dir", "") or ""),
        "deploy_patch_path": artifact_path,
        "deploy_repo_root": str(getattr(patch, "kernel_repo", "") or ""),
        "base_commit": str(getattr(patch, "base_commit", "") or ""),
        "micro_speedup": micro_speedup,
        "optimization_decision": "KEEP",
        "trace_gpu_pct": 0.0,
        "created_at": _now_iso(),
        "status": "pending",
        "correctness_source": "",
        "artifact_kind": "",
        "integration_validation_status": "",
        "framework_applyback": {},
        # Fusion-specific: the generic drain / writeback read these back.
        "source": "forge_fusion",
        "action_label": "fusion",
        "fusion_env_flags": fusion_env_flags,
        "keep_threshold_pct": float(keep_threshold_pct),
    }
    if integration_id not in queue:
        queue[integration_id] = record
    else:
        # Re-enqueue of the same sibling: refresh the fusion-specific fields (the
        # patch snapshot identity is immutable) so a re-run's env/threshold win.
        queued = queue[integration_id]
        if isinstance(queued, dict):
            queued["status"] = "pending"
            queued["micro_speedup"] = micro_speedup
            queued["fusion_env_flags"] = fusion_env_flags
            queued["keep_threshold_pct"] = float(keep_threshold_pct)
            queued["source"] = "forge_fusion"
            queued["action_label"] = "fusion"
    return queue[integration_id]


def _patch_budget_for(state) -> int:
    """How many sibling patches one round may land, env-overridable.

    ``HL_KERNEL_PATCH_BUDGET`` lets an operator widen or narrow the ceiling; the
    default is the module constant. Kept here so both the eviction cap and the
    pending-record clamp read one number.
    """
    return patch_budget(os.environ.get("HL_KERNEL_PATCH_BUDGET"), default=DEFAULT_PATCH_BUDGET)


def _ensure_kernel_task_state(state) -> None:
    """Initialise the stable ledger and re-queue the KEEPs recorded in it."""
    if not isinstance(getattr(state, "kernel_opt_task_attempts", None), dict):
        state.kernel_opt_task_attempts = {}
    if not isinstance(getattr(state, "pending_kernel_integrations", None), dict):
        state.pending_kernel_integrations = {}
    # The queue's only deletion point. Terminal records (integrated / rejected /
    # dispatch-failed) were once only status-flipped and never removed, so the
    # dict grew every round and was rescanned in full each time. Reap them here,
    # keeping a bounded tail for triage, before the re-queue below repopulates
    # the live KEEPs.
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


def pending_kernel_integration_records(state) -> list[dict[str, Any]]:
    """Return pending KEEP snapshots, preserving patches across ordinal reuse."""
    _ensure_kernel_task_state(state)
    integrated_sources = _source_files_in_optimization_stack(state)
    integrated_entries = [
        entry
        for entry in (state.optimization_stack or [])
        if isinstance(entry, dict) and entry.get("action") in {"integrate", "collective"}
    ]
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
        if str(record.get("status") or "pending") != "pending":
            continue
        task_group_key = str(record.get("task_group_key") or "")
        task_group_aliases = {str(alias) for alias in (record.get("legacy_task_group_keys") or []) if str(alias)}
        kernel_id = str(record.get("kernel_id") or "")
        # One spelling on both sides: the integrated-stack scan below writes
        # ``target_file or source_file`` too, so reading only ``source_file``
        # here would let a same-source sibling slip past the exclusion.
        source_file = record_source_path(record)
        artifact_path = str(record.get("artifact_path") or "")
        stable_entry = state.kernel_opt_task_attempts.get(str(record.get("task_key") or "")) or {}
        if kernel_id in set(state.rejected_kernel_ids or []) and (
            not task_group_key or bool(stable_entry.get("rejected_reason") if isinstance(stable_entry, dict) else "")
        ):
            continue
        if source_file and source_file in integrated_sources:
            continue
        if any(
            _record_matches_task(
                integrated,
                kernel_id=kernel_id,
                task_group_key=task_group_key,
                source_file=source_file,
                task_group_aliases=task_group_aliases,
            )
            for integrated in integrated_entries
        ):
            continue
        if any(
            _record_matches_task(
                attempted,
                kernel_id=kernel_id,
                task_group_key=task_group_key,
                source_file=source_file,
                task_group_aliases=task_group_aliases,
            )
            # Match only on a real patch_path: an attempted entry with a blank
            # patch_path used to match {"", artifact_path}, so one empty-path
            # attempt dropped the whole sibling family from the pending list.
            and (not artifact_path or str(attempted.get("patch_path") or "") == artifact_path)
            for attempted in attempted_entries
        ):
            continue
        try:
            impact = float(record.get("trace_gpu_pct") or 0.0)
        except (TypeError, ValueError):
            impact = 0.0
        if impact <= 0.0:
            impact = _kernel_trace_impact_pct(state, kernel_id)
        try:
            micro = float(record.get("micro_speedup") or 0.0)
        except (TypeError, ValueError):
            micro = 0.0
        record["integration_id"] = str(record.get("integration_id") or integration_id)
        candidates.append((impact, micro, integration_id, record))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Collapse only genuine same-source siblings: two whole-file overwrites of one
    # file cannot both land, so the strongest wins. Siblings on *different* files
    # are kept -- that is the whole point of a batch. The source key is read the
    # same way it is written (target_file or source_file), so a record whose path
    # lives under target_file is no longer mistaken for source-less.
    claimed_sources: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for _impact, _micro, _integration_id, record in candidates:
        source_file = record_source_path(record)
        if source_file and source_file in claimed_sources:
            continue
        if source_file:
            claimed_sources.add(source_file)
        deduped.append(record)
    # Cap how many siblings dispatch this round. The overflow is deferred, not
    # dropped: it stays pending (its queue record is untouched) and is
    # reconsidered next macro cycle, because a patch below this round's cut may
    # clear the next round's. The integrate lane is serial, so this ceiling is a
    # wall-clock guard, not a preference.
    fit, _deferred = clamp_by_budget(deduped, _patch_budget_for(state))
    return fit


def _vendor_playbook_id_from_result(result: dict[str, Any]) -> str:
    """Return the vendor-playbook group id a ``kernel_opt`` result belongs to.

    ``forge_submit._submit_vendor_playbook()`` stamps ``vendor_playbook_id``
    on every raw per-backend attempt dict it returns (winner and reused
    sibling alike); ``kernel_optimization.py`` carries those attempt dicts
    through verbatim in ``result["attempts"]``. Empty when this kernel_opt
    result did not go through the vendor-playbook path.
    """
    for attempt in result.get("attempts") or []:
        if isinstance(attempt, dict):
            vid = str(attempt.get("vendor_playbook_id") or "").strip()
            if vid:
                return vid
    return ""


def _resolve_kernel_patch_identity(
    state,
    payload: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    """Resolve a kernel patch's identity tuple from a result/intent payload.

    Pulls ``kernel_id`` / patch path / target file / extra server args
    from the envelope, back-filling the patch path from
    :attr:`last_kernel_opt` when the payload omits it but names a
    matching kernel. Extra launch args are read from the canonical
    ``extra_server_args`` field.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or LLM
            intent envelope (``None`` treated as empty).

    Returns:
        tuple[str, str, str, str]: ``(kernel_id, patch_path,
            target_file, extra_args)``; any unresolved component is an
            empty string.
    """
    payload = payload or {}
    kernel_id = str(payload.get("kernel_id") or "")
    patch_path = str(payload.get("patch_path") or payload.get("best_artifact_path") or "")
    # The last_kernel_opt back-fill is a single-patch convenience: it lends the
    # one just-optimized patch to a result that omitted its own path. A batch
    # sibling carries an integration_id and must never borrow it -- last_kernel_opt
    # holds whichever sibling finished most recently, so borrowing would key this
    # result under another sibling's identity. When a sibling omits its path it
    # is resolved from its own pending record, not from here.
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
    """Compute the dedup key for a kernel patch.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or intent
            envelope.

    Returns:
        str: ``"<kernel_id>|<patch_path>|<extra_args>"``, or ``""`` when
            either ``kernel_id`` or ``patch_path`` cannot be resolved.
    """
    kernel_id, patch_path, _target_file, extra_args = _resolve_kernel_patch_identity(state, payload)
    if not kernel_id or not patch_path:
        return ""
    return "|".join([kernel_id, patch_path, extra_args])


def find_rejected_kernel_patch(
    state,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Look up a previously-rejected patch matching ``payload``.

    Args:
        payload (dict[str, Any] | None): The kernel_opt result or intent
            envelope identifying the patch.

    Returns:
        dict[str, Any] | None: The matching rejected-patch entry, or
            ``None`` when the key is unresolvable or not on record.
    """
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


def record_kernel_integrate_result(
    state,
    result: dict[str, Any],
    *,
    max_attempts: int = 3,
    keep_threshold_pct: float = 1.0,
    max_fault_attempts: int | None = None,
) -> dict[str, Any] | None:
    """Persist one integrate E2E result and reject exhausted patch attempts.

    Appends the attempt to the per-key ``kernel_integrate_attempts``
    ledger. Two terminal paths are kept distinct:

    * **Gate verdict** — a genuine REVERT (gain below threshold / accuracy
      regression), or ``max_attempts`` non-fault attempts without a KEEP,
      moves the patch into ``rejected_kernel_patches`` and records its
      ``kernel_id`` in ``rejected_kernel_ids`` (terminal).
    * **Integration fault** — an environment/apply/bench crash (see
      :meth:`SharedState._is_integrate_fault`) that never fairly measured the
      patch. Faults do *not* consume the REVERT quota; they get an independent
      ``max_fault_attempts`` budget and are marked ``retryable`` so the
      pending-integrate driver re-enqueues them, only being rejected once
      that fault budget is exhausted.

    Args:
        result (dict[str, Any]): The integrate E2E result envelope.
        max_attempts (int): Max non-fault attempts before rejecting a
            non-KEEP patch (default 3).
        keep_threshold_pct (float): The gain threshold recorded on the
            rejection row for context (default 1.0).
        max_fault_attempts (int): Independent budget for total integration-
            fault attempts (initial + retries) before they are rejected as
            ``fault_attempts_exhausted`` (default 2 = one retry).

    Returns:
        dict[str, Any] | None: The updated attempts entry (carrying a
            ``rejected`` sub-dict when rejection fired, or
            ``retryable=True`` for an un-exhausted fault), or ``None`` when
            ``result`` is not a dict or its patch key is unresolvable.
    """
    if max_fault_attempts is None:
        max_fault_attempts = _MAX_INTEGRATE_FAULT_ATTEMPTS

    if not isinstance(result, dict):
        return None
    # Bind the result to its queued record first, by integration_id alone. An
    # integration_id is now required: the old fallback grabbed the first sibling
    # matching only the kernel name and accepted an empty artifact_path, which
    # could stamp integrated/rejected onto the wrong member of a batch. Every
    # auto-dispatched integrate carries its integration_id (the drain and
    # auto-enqueue drivers both pass it), and the LLM path that once dispatched by
    # bare kernel_id is now closed, so the fallback has no legitimate caller left.
    integration_id = str(result.get("integration_id") or "")
    pending_record = (state.pending_kernel_integrations or {}).get(integration_id) if integration_id else None
    if isinstance(pending_record, dict):
        # A sibling result may omit its own patch path -- resolve it from the
        # bound record rather than from last_kernel_opt, which would borrow a
        # different sibling's identity and key the ledger wrong.
        if not str(result.get("patch_path") or result.get("best_artifact_path") or ""):
            result = {**result, "patch_path": str(pending_record.get("artifact_path") or "")}
        if not str(result.get("target_file") or result.get("source_file") or ""):
            result = {**result, "target_file": record_source_path(pending_record)}
    key = kernel_patch_key(state, result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = _resolve_kernel_patch_identity(state, result)
    task_group_key = str(
        result.get("task_group_key") or (_entry_by_kernel_id(state, kernel_id) or {}).get("task_group_key") or ""
    )
    if isinstance(pending_record, dict):
        integration_id = str(pending_record.get("integration_id") or integration_id)
    identity_route = str(
        result.get("identity_route")
        or (pending_record.get("identity_route") if isinstance(pending_record, dict) else "")
        or ""
    )
    is_fault = state._is_integrate_fault(result)
    entry = dict(state.kernel_integrate_attempts.get(key) or {})
    attempts = list(entry.get("attempts") or [])
    attempt = {
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
    attempts.append(attempt)
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

    # Record the integrate outcome into the breakdown recorder (idempotent per
    # kernel_id, best-effort).
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        sdir = getattr(state, "_session_dir", None)
        if not sdir or not kernel_id:
            # Checked before the recorder is reached, so the recorder's own
            # guard never rules on it. On a KEEP this is the adoption that
            # credits the integrate, and nothing downstream can tell its
            # absence from a step that earned nothing.
            trace_recording_skipped(
                "kernel_e2e",
                reason="no session_dir" if not sdir else "no kernel_id",
                entity=kernel_id,
            )
        else:
            _dec = str(result.get("decision") or "").upper()
            instrument.record_kernel_e2e(
                sdir,
                kernel_id=kernel_id,
                integrated=(_dec == "KEEP"),
                e2e_gain_pct=result.get("gain_pct"),
                validated=True if _dec == "KEEP" else None,
                decision=_dec,
                patch_path=patch_path,
                target_file=target_file,
                extra_server_args=extra_args,
                result=result,
                # The id recovered above, not the one on the result: a result
                # that reached us without one still belongs to the pending
                # integrate we matched it to, and that is the integrate whose
                # readings must not be written over by a later one.
                occurrence=integration_id or None,
                validation_tier=(str(result.get("validation_tier") or "integrate_e2e") if _dec == "KEEP" else ""),
            )
    except Exception as exc:  # noqa: BLE001
        trace_recording_skipped(
            "kernel_e2e",
            reason="caller raised before the recorder",
            entity=kernel_id,
            error=exc,
        )

    if result.get("decision") == "KEEP":
        validation_tier = str(result.get("validation_tier") or "")
        integration_status = str(result.get("integration_validation_status") or "")
        if isinstance(pending_record, dict):
            pending_record["status"] = "integrated"
            pending_record["integrated_at"] = _now_iso()
            if integration_status:
                pending_record["integration_validation_status"] = integration_status
            if validation_tier:
                pending_record["validation_tier"] = validation_tier
        if integration_status or validation_tier:
            _stamp_integration_validation(
                state,
                kernel_id=kernel_id,
                task_key=str(
                    (pending_record.get("task_key") if isinstance(pending_record, dict) else "") or task_group_key or ""
                ),
                integration_status=integration_status,
                validation_tier=validation_tier,
            )
        return entry

    # Integration fault: never measured fairly. Retry on its own budget instead
    # of burning the REVERT quota, only rejecting once that budget is exhausted.
    if is_fault:
        if fault_count < max_fault_attempts:
            entry["retryable"] = True
            state.kernel_integrate_attempts[key] = entry
            return entry
        reason = f"fault_attempts_exhausted_{max_fault_attempts}"
    else:
        # Gate verdict path: a genuine REVERT, or too many non-fault attempts
        # without a KEEP.
        should_reject = result.get("decision") == "REVERT" or verdict_attempt_count >= max_attempts
        if not should_reject:
            return entry
        reason = (
            "revert_decision" if result.get("decision") == "REVERT" else f"max_e2e_attempts_{max_attempts}_without_keep"
        )
    rejected = {
        "key": key,
        "kernel_id": kernel_id,
        "task_group_key": task_group_key,
        "patch_path": patch_path,
        "target_file": target_file,
        "extra_server_args": extra_args,
        "attempt_count": len(attempts),
        "fault_count": fault_count,
        "best_gain_pct": best_gain,
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
    # A grouped task's members stay out of ``rejected_kernel_ids``: the ids are
    # synthetic per trace and a member can be re-dispatched under another task.
    # The task-level rejection below is the terminal fact, so consumers must
    # read the ledger row (``integration_status`` /
    # ``integration_rejected_reason``) rather than this set alone.
    if kernel_id and not task_group_key and kernel_id not in state.rejected_kernel_ids:
        state.rejected_kernel_ids.append(kernel_id)
    entry["rejected"] = rejected
    state.kernel_integrate_attempts[key] = entry
    task_key = str((pending_record.get("task_key") if isinstance(pending_record, dict) else "") or task_group_key or "")
    if isinstance(pending_record, dict):
        pending_record["status"] = "rejected"
        pending_record["rejected_at"] = _now_iso()
        pending_record["rejected_reason"] = reason
    if task_key:
        stable_attempt = (state.kernel_opt_task_attempts or {}).get(task_key)
        if isinstance(stable_attempt, dict):
            stable_attempt["integration_status"] = "rejected"
            stable_attempt["integration_rejected_reason"] = reason
            stable_attempt["integration_rejected_at"] = _now_iso()
    return entry


def record_kernel_opt(state, result: dict[str, Any]) -> None:
    """Capture the ``run_optimization`` handler result for the next Orch turn.

    Empty ``kernel_id`` is a no-op, a non-KEEP cannot overwrite a pending KEEP,
    and a ``kernel_id`` is retired after >= ``max_partial`` PARTIALs
    (``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL``).

    Args:
        result (dict[str, Any]): The ``run_optimization`` handler result
            envelope; non-dicts and empty ``kernel_id`` are no-ops.
    """
    if not isinstance(result, dict):
        return
    # Capture an empty-queue skip (no eligible kernels) as a non-failure
    # breadcrumb so the summary can surface it.
    is_no_eligible_dispatch_skip = (
        str(result.get("status") or "").lower() == "skipped"
        and str(result.get("reason") or "") == "no_eligible_kernels"
    )
    if is_no_eligible_dispatch_skip:
        state.last_kernel_opt_dispatch_skip = {
            "reason": "no_eligible_kernels",
            "kernels_considered": int(result.get("kernels_considered") or 0),
            "message": str(result.get("message") or ""),
            "ts": _now_iso(),
        }
    elif str(result.get("kernel_id") or ""):
        state.last_kernel_opt_dispatch_skip = {}
    # Author-time breakdown capture: record geak/forge invocations before the
    # metadata-less early return so no failed attempt becomes invisible.
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        sdir = getattr(state, "_session_dir", None)
        instrument.record_kernel_invocations(sdir, result)
        # Record dispatch and per-backend attempts.
        _kid = str(result.get("kernel_id") or "")
        if not sdir or not _kid:
            trace_recording_skipped(
                "kernel_dispatch",
                reason="no session_dir" if not sdir else "no kernel_id",
                entity=_kid,
            )
        else:
            _attempts = result.get("attempts")
            _attempts = _attempts if isinstance(_attempts, list) else []
            _backends = []
            for _a in _attempts:
                if isinstance(_a, dict):
                    _b = str(_a.get("backend") or "").lower()
                    if _b and _b not in _backends:
                        _backends.append(_b)
            if not _backends:
                _sel = result.get("selected_backends") or result.get("backends")
                if isinstance(_sel, list):
                    _backends = [str(b).lower() for b in _sel if b]
            # A backend that failed before dispatching attempts still counts as
            # dispatched. Mirror record_kernel_backend_result's failure-detect
            # so the synthetic FAILED attempt and the dispatch flag stay
            # consistent.
            _status = str(result.get("status") or "").lower()
            _err_class = str(result.get("error_class") or "")
            _decision = str((result.get("proposal") or {}).get("decision") or "").upper()
            _failed_predispatch = (not _attempts) and (
                _status in {"failed", "error", "crashed", "timeout"} or (_decision == "REVERT" and bool(_err_class))
            )
            if _failed_predispatch and not _backends:
                # Never default an unattributable failure to GEAK; "unknown"
                # reflects a pre-dispatch gating failure with no backend launched.
                _b = str(result.get("backend") or "").lower() or "unknown"
                _backends = [_b]
            _dispatched = bool(_attempts) or _failed_predispatch
            instrument.record_kernel_dispatch(
                sdir,
                kernel_id=_kid,
                dispatched=_dispatched,
                backends=_backends,
                # ``reason`` first: for an undispatched row it is the only field
                # that names *which* gate declined -- below the GPU-share floor,
                # merged into an op-fanout representative, a group already in
                # flight. ``status`` is "skipped" for all of them, so reading it
                # first collapses the distinction this lane exists to draw.
                skip_reason=(
                    ""
                    if _dispatched
                    else str(result.get("reason") or result.get("error_class") or result.get("status") or "")
                ),
                orchestration_commit=str(getattr(state, "code_revision", "") or ""),
            )
            instrument.record_kernel_backend_result(sdir, result)
    except Exception as exc:  # noqa: BLE001
        trace_recording_skipped(
            "kernel_dispatch",
            reason="caller raised before the recorder",
            entity=str(result.get("kernel_id") or ""),
            error=exc,
        )
    kernel_id = str(result.get("kernel_id") or "")
    if not kernel_id:
        # Metadata-less failure: preserve prior streaming-record KEEP.
        return
    # The batch filter dropped this kernel before any backend ran, and it named
    # the kernel so the report can say which one. Writing a ledger row for it
    # would spend the one dispatch this kernel gets on a decision nobody made:
    # the row drops it out of untried_hot_reusable_kernels(), which is what the
    # KERNEL-entry dispatch and the phase-advance gate both ask, and the summary
    # reads a row with no decision as IN_FLIGHT and reports it as a failure.
    if str(result.get("status") or "").lower() == "skipped" and unattempted_skip_reason(
        str(result.get("reason") or "")
    ):
        return
    _ensure_kernel_task_state(state)

    verification = result.get("verification") or {}
    proposal = result.get("proposal") or {}
    decision = str(proposal.get("decision", "")).upper()
    micro_speedup = verification.get("micro_speedup", 0.0)
    try:
        micro_float = float(micro_speedup)
    except (TypeError, ValueError):
        micro_float = 0.0
    best_artifact_path = str(verification.get("best_artifact_path", "") or "")
    deploy_snapshot_dir = str(verification.get("deploy_snapshot_dir", "") or "")
    deploy_patch_path = str(verification.get("deploy_patch_path", "") or "")
    deploy_repo_root = str(verification.get("deploy_repo_root", "") or "")
    best_artifact_bundle = dict(verification.get("best_artifact_bundle") or {})
    source_file = str(result.get("source_file") or (result.get("candidate") or {}).get("source_file") or "")
    task_group_id = str(result.get("task_group_id") or "")
    task_group_key = str(result.get("task_group_key") or "")
    stable_task_key = _stable_kernel_task_key(
        task_group_key=task_group_key,
        kernel_id=kernel_id,
        source_file=source_file,
    )
    task_group_kernel_ids = [str(item) for item in (result.get("task_group_kernel_ids") or []) if str(item)]
    # Siblings the batch filter merged into this representative. Recorded for the
    # same reason a task_group's members are: the work queue resolves a member to
    # whichever ledger row covers it, and one that resolves to none keeps owing an
    # attempt no dispatch will ever make.
    opfanout_collapsed_ids = [str(item) for item in (result.get("opfanout_collapsed_ids") or []) if str(item)]
    status = str(result.get("status") or "").lower()
    err_class = str(result.get("error_class") or "")
    # Pure infra failure = backend ladder with no verdict; kept distinct from
    # REVERT/PARTIAL so retirement counters don't double-count.
    is_infra_failure = decision == "" and (
        status in {"failed", "error", "timeout"}
        or err_class
        in {
            "subtask_exception",
            "handler_exception",
            "subprocess_timeout",
            "kernel_agent_root_missing",
            "missing_integration_inputs",
        }
    )
    ts = _now_iso()

    prior_entry = dict(_entry_by_kernel_id(state, kernel_id) or {})
    legacy_task_keys = {str(item) for item in (result.get("legacy_task_group_keys") or []) if str(item)}
    stable_prior_entry = state.kernel_opt_task_attempts.get(stable_task_key)
    migrated_stable_key = ""
    if not isinstance(stable_prior_entry, dict):
        migrated_stable_key, stable_prior_entry = next(
            (
                (task_key, candidate_entry)
                for task_key, candidate_entry in state.kernel_opt_task_attempts.items()
                if isinstance(candidate_entry, dict)
                and (
                    task_key in legacy_task_keys
                    or (
                        (
                            not str(result.get("identity_route") or "")
                            or not str(candidate_entry.get("identity_route") or "")
                            or str(candidate_entry.get("identity_route") or "")
                            != str(result.get("identity_route") or "")
                        )
                        and bool(
                            legacy_task_keys
                            & {
                                str(alias)
                                for alias in (candidate_entry.get("legacy_task_group_keys") or [])
                                if str(alias)
                            }
                        )
                    )
                )
            ),
            ("", None),
        )
    if isinstance(stable_prior_entry, dict):
        prior_entry = dict(stable_prior_entry)
        if migrated_stable_key and migrated_stable_key != stable_task_key:
            state.kernel_opt_task_attempts.pop(migrated_stable_key, None)
    prior_task_group_key = task_group_key if migrated_stable_key else str(prior_entry.get("task_group_key") or "")
    task_identity_changed = bool(task_group_key and prior_task_group_key and task_group_key != prior_task_group_key)
    if task_identity_changed:
        stale_member_ids = {
            member_id
            for member_id in [
                kernel_id,
                *task_group_kernel_ids,
            ]
            if str((_entry_by_kernel_id(state, member_id) or {}).get("task_group_key") or "")
            not in {"", task_group_key}
        }
        state.rejected_kernel_ids = [
            member_id for member_id in (state.rejected_kernel_ids or []) if member_id not in stale_member_ids
        ]
        entry = {}
    else:
        entry = prior_entry
    history = list(entry.get("history") or [])
    history.append(
        {
            "decision": decision,
            "micro": micro_float,
            "status": status,
            "ts": ts,
        }
    )
    history = history[-10:]
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    # Per-source attempts so a Python wrapper and its device file don't share a
    # retry quota.
    per_source = dict(entry.get("attempts_per_source") or {})
    src_key = source_file or ""
    per_source[src_key] = int(per_source.get(src_key, 0)) + 1
    entry["attempts_per_source"] = per_source
    if decision == "PARTIAL":
        entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
    elif decision == "KEEP":
        # Success resets streaks so a future regression isn't auto-retired on
        # stale history.
        entry["partial_count"] = 0
        entry["failure_count"] = 0
    if is_infra_failure:
        entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
    entry["last_decision"] = decision
    entry["last_status"] = status
    entry["last_micro_speedup"] = micro_float
    entry["last_artifact_path"] = best_artifact_path
    entry["last_artifact_bundle"] = best_artifact_bundle
    entry["last_snapshot_dir"] = deploy_snapshot_dir
    entry["last_deploy_patch_path"] = deploy_patch_path
    entry["last_deploy_repo_root"] = deploy_repo_root
    entry["last_source_file"] = source_file
    entry["kernel_id"] = kernel_id
    entry["current_kernel_id"] = kernel_id
    entry["stable_task_key"] = stable_task_key
    entry["identity_route"] = str(result.get("identity_route") or "")
    entry["operator_identity"] = dict(result.get("operator_identity") or {})
    if legacy_task_keys:
        entry["legacy_task_group_keys"] = sorted(
            {
                *[str(alias) for alias in (entry.get("legacy_task_group_keys") or []) if str(alias)],
                *legacy_task_keys,
            }
        )
    entry["last_gpu_pct"] = _kernel_trace_impact_pct(state, kernel_id)
    # Record backend + correctness so the GEAK-only verified-NEEDS_REVIEW
    # promotion gate can identify a correctness-verified GEAK win (consumed only
    # when HL_PROMOTE_VERIFIED_MICRO_NEEDS_REVIEW is enabled).
    entry["last_backend"] = str(verification.get("best_backend") or "")
    entry["last_correctness_passed"] = verification.get("correctness_passed")
    # Provenance for how correctness was established and whether the artifact
    # still owes a framework integration verdict.
    entry["last_correctness_source"] = str(verification.get("correctness_source") or "")
    entry["last_integration_validation_status"] = str(verification.get("integration_validation_status") or "")
    entry["last_framework_applyback"] = dict(verification.get("framework_applyback") or {})
    entry["last_ts"] = ts
    entry["history"] = history
    # A vendor-playbook artifact (e.g. mori's dispatch/combine launch-config
    # tuning, see agents/kernel/tools/_vendor_operator_playbooks.py) is a
    # KernelForge task-bundle config file, not a rewrite of the real,
    # installed operator source -- there is no supported path from it back
    # to the live serving install, and apply_kernel_patch's legacy
    # full-file-replace strategy would happily overwrite the real
    # site-packages module with it if ever asked to (PR #1191 review
    # finding #1). KEEP is still reported (the measured speedup is real and
    # worth surfacing), but deploy/integrate is refused downstream --
    # see _queue_kernel_keep() and integrate_handler()'s defense-in-depth
    # check.
    vendor_playbook_id = _vendor_playbook_id_from_result(result)
    entry["vendor_playbook_id"] = vendor_playbook_id
    entry["vendor_playbook_deploy_blocked"] = bool(vendor_playbook_id)

    # last_kernel_opt overwrite policy: KEEP always wins; non-KEEP writes only
    # when there is no pending KEEP to protect.
    prev = state.last_kernel_opt or {}
    prev_decision = str(prev.get("decision", "")).upper()
    prev_kid = str(prev.get("kernel_id", ""))
    integrated_ids = _kernel_ids_in_optimization_stack(state)
    prev_pending = (
        prev_decision == "KEEP"
        and bool(prev_kid)
        and prev_kid not in (state.rejected_kernel_ids or [])
        and prev_kid not in integrated_ids
    )
    if decision == "KEEP" or not prev_pending:
        state.last_kernel_opt = {
            "kernel_id": kernel_id,
            "decision": decision,
            "reasons": proposal.get("reasons", []),
            "micro_speedup": micro_float,
            "compile_passed": verification.get("compile_passed"),
            "correctness_passed": verification.get("correctness_passed"),
            "correctness_source": str(verification.get("correctness_source") or ""),
            "integration_validation_status": str(verification.get("integration_validation_status") or ""),
            "framework_applyback": dict(verification.get("framework_applyback") or {}),
            "best_artifact_path": best_artifact_path,
            "best_artifact_bundle": best_artifact_bundle,
            "deploy_snapshot_dir": deploy_snapshot_dir,
            "deploy_patch_path": deploy_patch_path,
            "deploy_repo_root": deploy_repo_root,
            "source_file": source_file,
            "task_group_key": task_group_key,
            "vendor_playbook_id": vendor_playbook_id,
            "vendor_playbook_deploy_blocked": bool(vendor_playbook_id),
            "ts": ts,
        }

    max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
    env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
    if env_v:
        try:
            max_partial = max(1, int(env_v))
        except (TypeError, ValueError):
            # Malformed env override -> keep the default partial-attempt cap.
            pass

    # Backend ladder failures are often transient infra faults; real REVERT
    # still retires immediately below, but infra failures get a retry.
    max_failures = resolve_kernel_opt_max_failures()

    # High-impact infra-retry (HL_HONEST_E2E umbrella, default ON; opt out with
    # HL_HONEST_E2E=0 or HL_INFRA_RETRY_HIGH_IMPACT=0). An infra non-finish
    # (timeout / preprocess / agent-crash; no verdict) means the attempt didn't
    # finish, not that the kernel can't be improved. Give a high-GPU%-share
    # kernel more attempts to COMPLETE rather than retiring it as a REVERT would.
    infra_failure_cap = max_failures
    if _honest_flag("HL_INFRA_RETRY_HIGH_IMPACT"):
        try:
            _impact_pct = float(_kernel_trace_impact_pct(state, kernel_id) or 0.0)
        except Exception:  # noqa: BLE001 - impact is best-effort
            _impact_pct = 0.0
        # Stamp impact so the dispatch-side cap widens from the same record.
        entry["last_gpu_pct"] = _impact_pct
        try:
            _min_gpu = float(os.environ.get("HL_INFRA_RETRY_MIN_GPU_PCT", "5.0") or 5.0)
        except ValueError:
            _min_gpu = 5.0
        try:
            _infra_max = int(os.environ.get("HL_INFRA_RETRY_MAX", "4") or 4)
        except ValueError:
            _infra_max = 4
        if _impact_pct >= _min_gpu:
            infra_failure_cap = max(max_failures, _infra_max)

    should_reject = (
        decision == "REVERT"
        or int(entry.get("partial_count", 0)) >= max_partial
        or int(entry.get("failure_count", 0)) >= infra_failure_cap
    )
    if should_reject:
        # A grouped task's members must stay out of ``rejected_kernel_ids`` --
        # the ids are synthetic per trace and one member's REVERT would blacklist
        # the shared kernel name, tombstoning its siblings by association. The
        # integrate path already exempts grouped tasks (see
        # record_kernel_integrate_result); the kernel-opt REVERT did not, which
        # was the real sibling guilt-by-association risk. The task-level
        # rejected_reason below remains the terminal fact for grouped tasks.
        if kernel_id and not task_group_key and kernel_id not in state.rejected_kernel_ids:
            state.rejected_kernel_ids.append(kernel_id)
        entry["rejected_reason"] = (
            "revert_decision"
            if decision == "REVERT"
            else (
                f"max_partial_attempts_{max_partial}_without_keep"
                if int(entry.get("partial_count", 0)) >= max_partial
                else f"max_failures_{max_failures}_without_keep"
            )
        )

    if opfanout_collapsed_ids:
        entry["opfanout_collapsed_ids"] = opfanout_collapsed_ids
    if task_group_id:
        entry["task_group_id"] = task_group_id
        entry["task_group_key"] = task_group_key
        entry["task_group_primary_kernel_id"] = str(result.get("task_group_primary_kernel_id") or kernel_id)
        entry["task_group_kernel_ids"] = task_group_kernel_ids
        entry["task_group_shape_case_ids"] = [
            str(item) for item in (result.get("task_group_shape_case_ids") or []) if str(item)
        ]
        entry["task_group_shape_case_count"] = int(result.get("task_group_shape_case_count") or 0)
    state.kernel_opt_task_attempts[stable_task_key] = dict(entry)
    _queue_kernel_keep(
        state,
        task_key=stable_task_key,
        kernel_id=kernel_id,
        entry=entry,
    )

    # One grouped result owns one keyed optimization ledger entry. Neither KEEP
    # nor REVERT is copied to sibling ordinal IDs: the group key and member list
    # prevent redispatch without leaving unscoped rejection tombstones that
    # could suppress a different task after the next trace reranks kernel IDs.


def record_gemm_tuning(state, result: dict[str, Any]) -> None:
    """Capture the GEAK GEMM tuning result for sequencing and prompts.

    Snapshots the result into ``last_gemm_tuning`` and appends it to the
    capped ``gemm_tuning_attempts`` history. A non-dict result is
    normalized into a failure record.

    Args:
        result (dict[str, Any]): The GEMM tuning result envelope.
    """
    if not isinstance(result, dict):
        result = {"status": "failed", "error": "non-dict gemm tuning result"}
    entry = dict(result)
    entry.setdefault("ts", _now_iso())
    state.last_gemm_tuning = entry
    attempts = list(state.gemm_tuning_attempts or [])
    attempts.append(entry)
    state.gemm_tuning_attempts = attempts[-_DEFAULT_ATTEMPTS_HISTORY:]
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.record_gemm_tuning_operation(
            getattr(state, "_session_dir", None),
            payload={"task_id": str(entry.get("task_id") or "kernel_entry_gemm_tuning")},
            result=entry,
        )
    except Exception as exc:  # noqa: BLE001
        trace_recording_skipped(
            "gemm_tuning",
            reason="caller raised before the recorder",
            entity=str(entry.get("task_id") or ""),
            error=exc,
        )


def is_collective_candidate(candidate: dict[str, Any]) -> bool:
    """Return whether a trace row requires the dedicated collective lane.

    The contract's ``collective`` kind is a name/path heuristic: it also fires on
    a single-GPU ``block_reduce`` and on anything whose source sits under a
    ``dist/`` directory. Withholding those from the other lanes would strand them,
    because the collective lane is opt-in and only admits an injected
    nccl-summary row whose primitive it can actually measure. So the ownership
    test is the lane's own admission test, not the heuristic.
    """
    contract = candidate.get("kernel_contract")
    if not isinstance(contract, dict):
        return False
    if str(contract.get("kind") or "") != "collective":
        return False
    if str(contract.get("collective_op") or "") not in SUPPORTED_COLLECTIVE_OPS:
        return False
    if str(candidate.get("candidate_source") or "").strip() != "nccl_summary":
        return False
    return candidate.get("is_multigpu") is True


def _kernel_ids_in_optimization_stack(state) -> set[str]:
    """kernel_ids already absorbed into optimization_stack by a kernel lane.

    Returns:
        set[str]: The set of ``kernel_id`` values that appear on an
            ``integrate`` or ``collective`` entry of
            :attr:`optimization_stack`.
    """
    return {
        str(e.get("kernel_id"))
        for e in (state.optimization_stack or [])
        if isinstance(e, dict) and e.get("action") in {"integrate", "collective"} and e.get("kernel_id")
    }


def _source_files_in_optimization_stack(state) -> set[str]:
    """source_file paths already touched by an integrating kernel lane; enforces "same source_file, only strongest KEEP integrated" (apply_kernel_patch is a whole-file overwrite).

    Returns:
        set[str]: The set of ``target_file`` / ``source_file`` paths
            referenced by ``integrate`` or ``collective`` entries of
            :attr:`optimization_stack`.
    """
    sources: set[str] = set()
    for e in state.optimization_stack or []:
        if not isinstance(e, dict) or e.get("action") not in {"integrate", "collective"}:
            continue
        src = record_source_path(e)
        if src:
            sources.add(src)
    return sources


def _record_matches_task(
    record: dict[str, Any],
    *,
    kernel_id: str,
    task_group_key: str,
    source_file: str,
    task_group_aliases: set[str] | None = None,
) -> bool:
    """Match persisted integration state to the current stable task identity."""
    recorded_key = str(record.get("task_group_key") or "")
    if task_group_key and recorded_key:
        accepted_keys = {task_group_key, *(task_group_aliases or set())}
        return recorded_key in accepted_keys
    if str(record.get("kernel_id") or "") != kernel_id:
        return False
    recorded_source = record_source_path(record)
    if source_file and recorded_source:
        return source_file == recorded_source
    return True


def _kernel_ids_with_integrate_attempts(state) -> set[str]:
    """kernel_ids that already received a *terminal* E2E integrate verdict.

    A kernel_id whose only integrate attempts are un-exhausted integration
    faults (``retryable``) is intentionally excluded so the pending-integrate
    driver re-enqueues it for a fault retry. A kernel_id is treated as
    attempted once *any* of its entries reached a non-retryable terminal
    state (KEEP / real REVERT / fault budget exhausted); a terminal entry on
    one patch key wins over a retryable entry on another.

    Returns:
        set[str]: The kernel_ids with at least one non-retryable terminal
            integrate entry.
    """
    terminal: set[str] = set()
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("kernel_id") or "").strip()
        if not kid:
            continue
        if entry.get("retryable") and not entry.get("rejected"):
            continue
        terminal.add(kid)
    return terminal


def integrate_attempt_count_for_kernel(state, kernel_id: str) -> int:
    """Total *recorded* integrate attempts for a kernel_id.

    Sums ``attempt_count`` across every ``kernel_integrate_attempts`` entry
    sharing this ``kernel_id`` (one kernel may produce more than one patch
    key). The count only advances inside
    :func:`record_kernel_integrate_result`, so it is a reliable in-flight
    signal for the KERNEL-phase auto-integrate driver: an unchanged count
    means a dispatched integrate has not yet been recorded (still in flight),
    an advanced count means it completed.

    Args:
        kernel_id (str): The kernel identifier to total attempts for.

    Returns:
        int: Recorded integrate attempts (0 when unknown/blank).
    """
    kid = str(kernel_id or "").strip()
    if not kid:
        return 0
    total = 0
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("kernel_id") or "").strip() != kid:
            continue
        try:
            total += int(entry.get("attempt_count") or 0)
        except (TypeError, ValueError):
            continue
    return total


def integrate_attempt_count_for_integration(
    state,
    integration_id: str,
) -> int:
    """Return recorded attempts for one immutable pending patch."""
    ident = str(integration_id or "").strip()
    if not ident:
        return 0
    total = 0
    for entry in (state.kernel_integrate_attempts or {}).values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("integration_id") or "") != ident:
            continue
        try:
            total += int(entry.get("attempt_count") or 0)
        except (TypeError, ValueError):
            continue
    return total


def _kernel_trace_impact_pct(state, kernel_id: str) -> float:
    """Return TraceLens gpu_pct for a kernel_id; unknown kernels sort last.

    Args:
        kernel_id (str): The kernel identifier to look up in the latest
            trace-analyze ``hot_kernels_top15``.

    Returns:
        float: The kernel's ``gpu_pct`` impact, or ``0.0`` when blank,
            unknown, or unparseable.
    """
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
    """Return next KEEP kernel_id awaiting integrate ("" if drained).

    Ordering favors trace impact (``gpu_pct``) over kernel micro speedup:
    E2E validation should test the highest-impact hot kernel first, not
    merely the patch with the largest isolated microbenchmark win.

    Returns:
        str: The highest-impact pending KEEP ``kernel_id``, or ``""``
            when the queue is drained.
    """
    pending = pending_keep_kernel_ids(state)
    return pending[0] if pending else ""


def pending_keep_kernel_ids(state) -> list[str]:
    """All KEEP kernel_ids awaiting integrate, sorted impact-first.

    Kernels that already have an integrate attempt (including
    ``NEEDS_REVIEW``) are excluded so a noisy near-threshold result does not
    automatically rerun the same patch up to the historical max-attempt cap.
    Positive ``NEEDS_REVIEW`` rows are handled by stack validation instead.

    Returns:
        list[str]: Pending KEEP ``kernel_id`` values sorted impact-first
            (trace ``gpu_pct``, then micro speedup), one per source file.
    """
    return [
        str(record.get("kernel_id") or "")
        for record in pending_kernel_integration_records(state)
        if str(record.get("kernel_id") or "")
    ]


def has_keep_pending_integrate(state) -> bool:
    """Whether any KEEP kernel is still awaiting integrate.

    Returns:
        bool: ``True`` when :meth:`next_pending_keep_kernel_id` is
            non-empty.
    """
    return bool(next_pending_keep_kernel_id(state))


def index_attempts_by_kernel_id(attempts: Any) -> dict[str, dict]:
    """Re-index a stable-keyed attempt ledger by trace-local ``current_kernel_id``.

    The ordinal id is not an identity — reranking moves it between operators, so
    two stable entries can claim the same one. The latest-stamped entry wins,
    which is the one currently occupying the ordinal slot.

    Args:
        attempts: A ``kernel_opt_task_attempts`` mapping, or anything falsy.

    Returns:
        ``{current_kernel_id: attempt}``, holding the ledger's own entry dicts.
    """
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
    """Number of distinct kernel tasks with recorded kernel_opt attempts.

    Returns:
        int: The size of the ``kernel_opt_task_attempts`` ledger (one entry
            per stable task identity, not per ordinal kernel_id).
    """
    _ensure_kernel_task_state(state)
    return len(state.kernel_opt_task_attempts or {})


def untried_hot_reusable_kernels(
    state,
    *,
    min_gpu_pct: float | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Hot kernels still owing a ``kernel_opt`` attempt (reusable, gpu_pct >= min_gpu_pct, untouched); capped to top_n by gpu_pct, one kernel_id per task_group.

    Args:
        min_gpu_pct (float | None): Minimum GPU-share threshold; when
            ``None`` it is read from ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT``.
        top_n (int | None): Cap on enforced kernels by gpu_pct; when
            ``None`` it is read from ``HYPERLOOM_KERNEL_OPT_GATE_TOP_N``.

    Returns:
        list[str]: The untried hot-reusable ``kernel_id`` values (one per
            task_group), sorted strongest-first.
    """
    info = state.last_trace_analyze or {}
    hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
    task_groups = info.get("task_groups") or []
    if not isinstance(hot, list):
        return []

    if min_gpu_pct is None:
        min_gpu_pct = resolve_hot_kernel_min_gpu_pct()
    if top_n is None:
        try:
            top_n = int(
                os.environ.get(
                    "HYPERLOOM_KERNEL_OPT_GATE_TOP_N",
                    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
                )
            )
        except (TypeError, ValueError):
            top_n = _DEFAULT_HOT_KERNEL_GATE_TOP_N
    top_n = max(1, int(top_n))

    kid_to_group: dict[str, tuple[list[str], str]] = {}
    group_key_aliases: dict[str, set[str]] = {}
    for g in task_groups:
        if not isinstance(g, dict):
            continue
        members = [str(m) for m in (g.get("kernel_ids") or []) if m]
        group_key = str(g.get("task_group_key") or "")
        aliases = {
            group_key,
            *[str(alias) for alias in (g.get("legacy_task_group_keys") or []) if str(alias)],
        }
        group_key_aliases[group_key] = {alias for alias in aliases if alias}
        for m in members:
            kid_to_group[m] = (members, group_key)

    integrated_sources = _source_files_in_optimization_stack(state)
    integrated_entries = [
        entry
        for entry in (state.optimization_stack or [])
        if isinstance(entry, dict) and entry.get("action") in {"integrate", "collective"}
    ]
    rejected = set(state.rejected_kernel_ids or [])
    _ensure_kernel_task_state(state)
    attempts = state.kernel_opt_task_attempts or {}

    # Sort by gpu_pct desc so dedup picks the strongest member of each
    # task_group.
    rows: list[tuple[float, str, str, list[str], str, tuple[str, str, float]]] = []
    for k in hot:
        if not isinstance(k, dict):
            continue
        if k.get("reusable_native_kernel") is not True:
            continue
        if is_collective_candidate(k):
            continue
        # Bypass path tags a kernel non-dispatchable when its shape is
        # geometry-only (launch_grid/tile_name) and would fail the kernel-opt
        # gate. Skip those so they never re-enter the untried queue. Absent field
        # (TraceLens path) is treated as dispatchable to avoid regressing it.
        if k.get("shape_dispatchable") is False:
            continue
        try:
            gpu_pct = float(k.get("gpu_pct") or 0.0)
        except (TypeError, ValueError):
            gpu_pct = 0.0
        # Vendor-playbook groups (mori's dispatch+combine) are gated on the
        # sum of the group's members, not each member's own share, and may
        # pin a per-playbook floor -- see effective_hot_kernel_gpu_pct's
        # docstring. Ranking below still sorts on the per-row gpu_pct.
        if effective_hot_kernel_gpu_pct(k) < effective_hot_kernel_min_gpu_pct(k, min_gpu_pct):
            continue
        kid = str(k.get("kernel_id") or "")
        if not kid:
            continue
        src = str(k.get("source_file") or "")
        group_info = kid_to_group.get(kid)
        members = sorted(group_info[0]) if group_info else [kid]
        group_key = group_info[1] if group_info else ""
        # Identity of the underlying kernel, independent of the synthetic
        # per-row kernel_id. Used only as a dedup fallback (see below).
        identity = (src, str(k.get("name") or k.get("operation") or ""), gpu_pct)
        rows.append((gpu_pct, kid, src, members, group_key, identity))
    rows.sort(key=lambda x: x[0], reverse=True)

    ranked: list[tuple[float, str, str, list[str], str, tuple[str, str, float]]] = []
    seen_groups: set[str | tuple[str, ...]] = set()
    seen_identities: set[tuple[str, str, float]] = set()
    for row in rows:
        dedup_key: str | tuple[str, ...] = row[4] or tuple(row[3])
        if dedup_key in seen_groups:
            continue
        # Fallback dedup: when the trace carries no ``task_groups`` metadata
        # every row degenerates to its own group, so the SAME kernel appearing
        # under several synthetic ids (identical source_file+name+gpu_pct, e.g.
        # k001/k002) is treated as several distinct hot kernels. The first gets
        # attempted and rejected while its twin stays forever "untried", so
        # kernel_work_pending() never goes False and KERNEL_AGENT spins until
        # the wall-clock cap -- while the twin is not even in the candidate
        # registry, so no agent can ever act on it. Collapse by identity.
        identity = row[5]
        if identity[0] and identity[1]:
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
        seen_groups.add(dedup_key)
        ranked.append(row)
    ranked = ranked[:top_n]

    untried: list[str] = []

    def _attempt_for_member(member_id: str) -> dict[str, Any]:
        """Ledger entry covering ``member_id``, tolerating synthetic-id churn.

        ``current_kernel_id`` tracks whichever synthetic id the agent last
        asked for, so for a kernel that shows up under several ids (k001/k002
        for one CK GEMM) it flip-flops. Matching on it alone makes the entry
        invisible under the *other* id, which is exactly how a rejected kernel
        gets re-reported as untried forever. Fall back to the membership the
        ledger itself records -- a task_group's members, and the op-fanout
        siblings the batch filter merged into the row's representative. The
        latter matters because the merge is reported as an unattempted skip,
        which writes no row of its own: without it the sibling resolves to no
        entry at all and stays in this queue for a dispatch that cannot happen.
        """
        for value in attempts.values():
            if not isinstance(value, dict):
                continue
            if member_id in {
                str(value.get("current_kernel_id") or ""),
                str(value.get("kernel_id") or ""),
                str(value.get("task_group_primary_kernel_id") or ""),
            }:
                return value
            for key in ("task_group_kernel_ids", "opfanout_collapsed_ids"):
                if member_id in {str(m) for m in (value.get(key) or []) if m}:
                    return value
        return {}

    def _member_is_rejected(member_id: str) -> bool:
        """True when ``member_id`` (or its ledger twin) is out of play."""
        if member_id in rejected:
            return True
        attempt = _attempt_for_member(member_id)
        if not attempt:
            return False
        if str(attempt.get("rejected_reason") or "").strip():
            return True
        return str(attempt.get("integration_status") or "").strip().lower() == "rejected"

    def _matches_current_task(member_id: str, group_key: str, source: str) -> bool:
        if group_key:
            aliases = group_key_aliases.get(group_key) or {group_key}
            return any(
                isinstance(attempt, dict)
                and (
                    str(attempt.get("stable_task_key") or "") == group_key
                    or str(attempt.get("task_group_key") or "") == group_key
                    or str(attempt.get("stable_task_key") or "") in aliases
                    or str(attempt.get("task_group_key") or "") in aliases
                )
                for attempt in attempts.values()
            )
        attempt = _attempt_for_member(member_id)
        if not isinstance(attempt, dict) or not attempt:
            return False
        recorded_source = str(attempt.get("last_source_file") or "")
        return not source or not recorded_source or source == recorded_source

    for _pct, kid, src, members, group_key, _identity in ranked:
        if members and all(
            _member_is_rejected(member) and _matches_current_task(member, group_key, src) for member in members
        ):
            continue
        if any(
            _record_matches_task(
                integrated,
                kernel_id=member,
                task_group_key=group_key,
                source_file=src,
                task_group_aliases=group_key_aliases.get(group_key),
            )
            for member in members
            for integrated in integrated_entries
        ):
            continue
        if src and src in integrated_sources:
            continue
        stable_attempt = next(
            (
                attempt
                for attempt in attempts.values()
                if group_key
                and isinstance(attempt, dict)
                and (
                    str(attempt.get("stable_task_key") or "") == group_key
                    or str(attempt.get("task_group_key") or "") == group_key
                    or str(attempt.get("stable_task_key") or "") in (group_key_aliases.get(group_key) or {group_key})
                    or str(attempt.get("task_group_key") or "") in (group_key_aliases.get(group_key) or {group_key})
                )
            ),
            None,
        )
        if stable_attempt is not None and int(stable_attempt.get("attempts", 0)) > 0:
            continue
        # Resolve through ``_attempt_for_member`` rather than comparing ids
        # inline: a row's own id is not the only id it covers. An op-fanout
        # representative covers the siblings the batch filter merged into it,
        # and those merges are reported as unattempted skips that write no row
        # of their own -- so a sibling compared by id alone finds nothing and
        # keeps owing an attempt no dispatch will make.
        if not group_key and any(
            _matches_current_task(member, group_key, src)
            and int((_attempt_for_member(member) or {}).get("attempts", 0)) > 0
            for member in members
        ):
            continue
        untried.append(kid)
    return untried
