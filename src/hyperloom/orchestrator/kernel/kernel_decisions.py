# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-decision write-owner functions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, NamedTuple

from kernelforge.knowledge import KERNEL_RECIPE_PRODUCERS, normalize_operator_name

from hyperloom.common.coerce import to_float, to_int
from hyperloom.common.env import env_flag

from .patch_landing import (
    INTEGRATING_STACK_ACTIONS,
    VERDICT_STATUSES,
    IntegratedStack,
    evict_terminal,
    patch_budget,
    record_matches_task,
    record_source_path,
)
from hyperloom.common.timeutil import now_iso as _now_iso
from ..state.kernel_decision_settings import (
    _DEFAULT_ATTEMPTS_HISTORY,
    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
    _MAX_INTEGRATE_E2E_ATTEMPTS,
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


def _existing_integration_id(queue: dict[str, Any], *, source_file: str, artifact_path: str) -> str:
    """The queued record already holding this patch, keyed by the pair that identifies it."""
    return next(
        (
            candidate_id
            for candidate_id, candidate in queue.items()
            if isinstance(candidate, dict)
            and _text(candidate.get("source_file")) == source_file
            and _text(candidate.get("artifact_path")) == artifact_path
        ),
        "",
    )


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
    micro_speedup = patch.micro_speedup
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


def _patch_budget_for() -> int:
    """How many sibling patches one round may land, env-overridable."""
    return patch_budget(os.environ.get("HL_KERNEL_PATCH_BUDGET"))


def _evict_settled_integrations(state) -> None:
    """Reap stale terminal records from the integration queue -- the queue's only deletion point."""
    state.pending_kernel_integrations = evict_terminal(
        state.pending_kernel_integrations,
        budget=_patch_budget_for(),
    )


def _pending_record_settled(state, record: dict[str, Any], integrated, attempted_entries: list[dict[str, Any]]) -> bool:
    """True when something already ruled on this queued patch, so it is no longer pending work."""
    task_group_key = _text(record.get("task_group_key"))
    task_group_aliases = {str(alias) for alias in (record.get("legacy_task_group_keys") or []) if str(alias)}
    kernel_id = _text(record.get("kernel_id"))
    source_file = record_source_path(record)
    artifact_path = _text(record.get("artifact_path"))
    if kernel_id in set(state.rejected_kernel_ids or []) and not task_group_key:
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
    _evict_settled_integrations(state)
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
    return deduped[: _patch_budget_for()]


def _resolve_kernel_patch_identity(payload: dict[str, Any] | None) -> tuple[str, str, str, str]:
    """Resolve a kernel patch's identity tuple from a result/intent payload."""
    payload = payload or {}
    kernel_id = str(payload.get("kernel_id") or "")
    patch_path = str(payload.get("patch_path") or payload.get("best_artifact_path") or "")
    target_file = str(payload.get("target_file") or payload.get("source_file") or "")
    extra_args = str(payload.get("extra_server_args") or "").strip()
    return kernel_id, patch_path, target_file, extra_args


def kernel_patch_key(payload: dict[str, Any] | None) -> str:
    """Compute the dedup key for a kernel patch."""
    kernel_id, patch_path, _target_file, extra_args = _resolve_kernel_patch_identity(payload)
    if not kernel_id or not patch_path:
        return ""
    return "|".join([kernel_id, patch_path, extra_args])


def find_rejected_kernel_patch(
    state,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Look up a previously-rejected patch matching ``payload``."""
    key = kernel_patch_key(payload)
    if not key:
        return None
    for entry in state.rejected_kernel_patches:
        if isinstance(entry, dict) and entry.get("key") == key:
            return entry
    return None


def _bind_result_to_pending(state, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Bind an integrate result to its queued record by integration_id, filling in what it left out."""
    integration_id = _text(result.get("integration_id"))
    pending = (state.pending_kernel_integrations or {}).get(integration_id) if integration_id else None
    if not isinstance(pending, dict):
        return result, None
    # A sibling result may omit its own patch path -- resolve it from the record it is bound to.
    if not (result.get("patch_path") or result.get("best_artifact_path")):
        result = {**result, "patch_path": _text(pending.get("artifact_path"))}
    if not (result.get("target_file") or result.get("source_file")):
        result = {**result, "target_file": record_source_path(pending)}
    return result, pending


def _attempt_totals(attempts: list[Any]) -> tuple[float, int]:
    """The best gain this patch ever measured and how many of its attempts were integration faults."""
    rows = [a for a in attempts if isinstance(a, dict)]
    gains = [a["gain_pct"] for a in rows if isinstance(a.get("gain_pct"), (int, float))]
    return float(max(gains, default=0.0)), sum(1 for a in rows if a.get("is_fault"))


def _record_integration_keep(result: dict[str, Any], pending_record) -> None:
    """Settle the queued record of a patch the integrate gate kept."""
    if not isinstance(pending_record, dict):
        return
    pending_record.update(status="integrated", integrated_at=_now_iso())
    integration_status = _text(result.get("integration_validation_status"))
    validation_tier = _text(result.get("validation_tier"))
    if integration_status:
        pending_record["integration_validation_status"] = integration_status
    if validation_tier:
        pending_record["validation_tier"] = validation_tier


def _reject_integration(state, entry: dict[str, Any], *, reason: str, pending_record) -> None:
    """Record a patch's rejection against its attempt entry, the rejection ledger and its queued record."""
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


def record_kernel_integrate_result(state, result: dict[str, Any]) -> dict[str, Any] | None:
    """Persist one integrate E2E result and reject exhausted patch attempts."""
    if not isinstance(result, dict):
        return None
    result, pending_record = _bind_result_to_pending(state, result)
    key = kernel_patch_key(result)
    if not key:
        return None
    kernel_id, patch_path, target_file, extra_args = _resolve_kernel_patch_identity(result)
    task_group_key = _text(result.get("task_group_key"))
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
            "graded_objective": result.get("graded_objective"),
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
        _record_integration_keep(result, pending_record)
        return entry

    # Integration fault: never measured fairly.
    if is_fault:
        if fault_count < _MAX_INTEGRATE_FAULT_ATTEMPTS:
            entry["retryable"] = True
            state.kernel_integrate_attempts[key] = entry
            return entry
        reason = f"fault_attempts_exhausted_{_MAX_INTEGRATE_FAULT_ATTEMPTS}"
    else:
        # Gate verdict path: a genuine REVERT, or too many non-fault attempts without a KEEP.
        should_reject = result.get("decision") == "REVERT" or verdict_attempt_count >= _MAX_INTEGRATE_E2E_ATTEMPTS
        if not should_reject:
            return entry
        reason = (
            "revert_decision"
            if result.get("decision") == "REVERT"
            else f"max_e2e_attempts_{_MAX_INTEGRATE_E2E_ATTEMPTS}_without_keep"
        )
    _reject_integration(state, entry, reason=reason, pending_record=pending_record)
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


def _forge_loop_operators_in_optimization_stack(state) -> set[str]:
    """Normalized operator names an integrating kernel-recipe lane (forge-loop/flydsl/fusion) has already landed.

    These lanes key their ``optimization_stack`` entries by the long-form recipe id
    (``kernel:forge-loop:<operator>:<framework>:<framework_version>:<backend>:<gpu>``), which never equals a roofline
    trace's synthetic ``kNNN`` kernel_id even though both name the same kernel. Comparing on the operator name — run
    through the same ``normalize_operator_name`` the recipe id was built with — is the one identity the two sides
    share.
    """
    operators: set[str] = set()
    for e in state.optimization_stack or []:
        if not isinstance(e, dict) or e.get("action") not in INTEGRATING_STACK_ACTIONS:
            continue
        operator = _canonical_kernel_recipe_operator(str(e.get("kernel_id") or ""))
        if operator:
            operators.add(normalize_operator_name(operator))
    return operators


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
        return to_float(row.get("gpu_pct"), 0.0)
    return 0.0


def pending_keep_kernel_ids(state) -> list[str]:
    """All KEEP kernel_ids awaiting integrate, sorted impact-first."""
    return [
        str(record.get("kernel_id") or "")
        for record in pending_kernel_integration_records(state)
        if str(record.get("kernel_id") or "")
    ]


def has_keep_pending_integrate(state) -> bool:
    """Whether any KEEP kernel is still awaiting integrate."""
    return bool(pending_keep_kernel_ids(state))


def _attempt_records(state) -> list[dict[str, Any]]:
    """Every record that says a kernel patch was produced or measured: queued patches and integrate verdicts."""
    return [*state.pending_kernel_integrations.values(), *state.kernel_integrate_attempts.values()]


def _attempted_operators(state, records: list[dict[str, Any]]) -> set[str]:
    """Normalized operator names some backend already produced a patch for.

    A Forge rewrite or fusion and a GEAK adoption name the kernel by its symbol or by a kernel-recipe id, never by
    the trace's synthetic ``kNNN`` ordinal, so the operator name is the identity they share with a trace row.
    """
    names = [str(record.get("kernel_id") or "") for record in records]
    integration = state.kernel_rewrite_controller_result.get("integration", {})
    names.extend(str(row.get("operator_id") or "") for row in integration.get("results", []))
    return {normalize_operator_name(_canonical_kernel_recipe_operator(name) or name) for name in names if name.strip()}


class _HotRow(NamedTuple):
    """One reusable hot trace row the gate ranks, with the identities an attempt can match it by."""

    order: tuple[bool, int, float]
    kernel_id: str
    source_file: str
    members: list[str]
    group_key: str
    identity: tuple[str, str, float]


def _resolve_hot_kernel_top_n() -> int:
    """The gate's per-ranking cap, ``HYPERLOOM_KERNEL_OPT_GATE_TOP_N``, never below one."""
    return max(1, to_int(os.environ.get("HYPERLOOM_KERNEL_OPT_GATE_TOP_N"), _DEFAULT_HOT_KERNEL_GATE_TOP_N))


def _hot_rows(
    hot: list[Any],
    kid_to_group: dict[str, tuple[list[str], str]],
    min_gpu_pct: float,
) -> list[_HotRow]:
    """The reusable, dispatchable trace rows at or above the GPU-share floor, in TraceLens' order."""
    rows: list[_HotRow] = []
    for k in hot:
        if not isinstance(k, dict) or k.get("reusable_native_kernel") is not True:
            continue
        # Bypass path tags a kernel non-dispatchable when its shape is geometry-only (launch_grid/tile_name).
        if k.get("shape_dispatchable") is False:
            continue
        # Vendor-playbook groups (mori's dispatch+combine) are gated on the sum of the group's members, not each
        # member's own share, and may pin a per-playbook floor -- see effective_hot_kernel_gpu_pct's docstring.
        if effective_hot_kernel_gpu_pct(k) < effective_hot_kernel_min_gpu_pct(k, min_gpu_pct):
            continue
        kid = str(k.get("kernel_id") or "")
        if not kid:
            continue
        gpu_pct = to_float(k.get("gpu_pct"), 0.0)
        src = str(k.get("source_file") or "")
        group_info = kid_to_group.get(kid)
        rank = int(k.get("tracelens_pitem_rank") or 0)
        rows.append(
            _HotRow(
                order=(rank <= 0, rank, -float(k.get("impact_score") or 0.0)),
                kernel_id=kid,
                source_file=src,
                members=sorted(group_info[0]) if group_info else [kid],
                group_key=group_info[1] if group_info else "",
                # Identity of the underlying kernel, independent of the synthetic per-row kernel_id.
                identity=(src, str(k.get("name") or k.get("operation") or ""), gpu_pct),
            )
        )
    rows.sort(key=lambda row: row.order)
    return rows


def _rank_hot_rows(rows: list[_HotRow], top_n: int) -> list[_HotRow]:
    """One row per task group, capped to ``top_n`` per ranking.

    Rows TraceLens never ranked (bypass route, injected collectives) follow in emitted order under their own
    ``top_n``, so they cannot fall behind TraceLens' cap.
    """
    ranked: list[_HotRow] = []
    seen_groups: set[str | tuple[str, ...]] = set()
    seen_identities: set[tuple[str, str, float]] = set()
    for row in rows:
        dedup_key: str | tuple[str, ...] = row.group_key or tuple(row.members)
        if dedup_key in seen_groups:
            continue
        # Without ``task_groups`` metadata every row is its own group, so the same kernel under several synthetic ids
        # (identical source_file+name+gpu_pct) is collapsed on its identity instead.
        if row.identity[0] and row.identity[1]:
            if row.identity in seen_identities:
                continue
            seen_identities.add(row.identity)
        seen_groups.add(dedup_key)
        ranked.append(row)
    return [r for r in ranked if not r.order[0]][:top_n] + [r for r in ranked if r.order[0]][:top_n]


def _task_group_index(
    task_groups: list[Any],
) -> tuple[dict[str, tuple[list[str], str]], dict[str, set[str]]]:
    """Map each member kernel_id to its group's ``(members, task_group_key)``, and each group key to its aliases."""
    kid_to_group: dict[str, tuple[list[str], str]] = {}
    group_key_aliases: dict[str, set[str]] = {}
    for g in task_groups:
        if not isinstance(g, dict):
            continue
        members = [str(m) for m in (g.get("kernel_ids") or []) if m]
        group_key = str(g.get("task_group_key") or "")
        group_key_aliases[group_key] = {
            str(alias) for alias in [group_key, *(g.get("legacy_task_group_keys") or [])] if str(alias)
        }
        for m in members:
            kid_to_group[m] = (members, group_key)
    return kid_to_group, group_key_aliases


def untried_hot_reusable_kernels(state) -> list[str]:
    """Reusable hot kernels (gpu_pct >= the configured minimum) no backend has produced a patch for yet.

    Capped by TraceLens rank, one kernel_id per task_group. A kernel counts as tried once a patch for
    it was queued for integration, gated end to end, landed on the optimization stack, or published by the rewrite
    controller.
    """
    info = state.last_trace_analyze or {}
    hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
    if not isinstance(hot, list):
        return []
    kid_to_group, group_key_aliases = _task_group_index(info.get("task_groups") or [])

    integrated = IntegratedStack.of(state.optimization_stack)
    records = _attempt_records(state)
    tried_operators = _attempted_operators(state, records) | _forge_loop_operators_in_optimization_stack(state)

    untried: list[str] = []
    for row in _rank_hot_rows(
        _hot_rows(hot, kid_to_group, resolve_hot_kernel_min_gpu_pct()), _resolve_hot_kernel_top_n()
    ):
        aliases = group_key_aliases.get(row.group_key)
        if integrated.covers(
            kernel_ids=row.members,
            task_group_key=row.group_key,
            source_file=row.source_file,
            task_group_aliases=aliases,
        ):
            continue
        if row.identity[1] and normalize_operator_name(row.identity[1]) in tried_operators:
            continue
        if any(
            record_matches_task(
                record,
                kernel_id=member,
                task_group_key=row.group_key,
                source_file=row.source_file,
                task_group_aliases=aliases,
            )
            for record in records
            for member in row.members
        ):
            continue
        untried.append(row.kernel_id)
    return untried
