# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator result writeback: settle finished tasks into SharedState and rebuild that state on resume."""

from __future__ import annotations
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple
import yaml
from hyperloom.common.coerce import to_float, to_str_list
from hyperloom.common.env import is_truthy
from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.orchestrator.lever import (
    FRAMEWORK_STACK_ACTION,
    LEVER_CONFIG,
    LEVER_ENABLEMENT,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    patch_lever_kind,
    lever_kind_for_task,
    patch_owner_phase,
)
from ..knowledge.remote_recipe.sanitize import HOST_ORIGIN_KEY
from ..actions._recorder_trace import trace_recording_skipped
from hyperloom.inference_optimizer.session.optimization_journal import (
    Verdict,
    classify_change_kind,
    operation_kind_for,
)
from ..actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from ..actions.executors.explore import STACK_REVALIDATE_SOURCE, is_stack_revalidation
from hyperloom.inference_optimizer.grid_server_args import strip_benchmark_harness_flags
from hyperloom.inference_optimizer.inferencex_checkout import has_benchmark_lib
from hyperloom.inference_optimizer.framework_registry import server_args_env_name
from ..actions.executors._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS
from ..actions.executors._subprocess_kill import AGENTX_PREFLIGHT_ERROR_CLASS
from ..actions.executors._ray_serving import RAY_INFEASIBLE_MARKER
from hyperloom.inference_optimizer.breakdown.stop_reasons import (
    AGENTX_PREFLIGHT_STOP_REASON,
    PATCH_RECOVERY_INCOMPLETE_STOP_REASON,
)
from ..phases.machine_state import PHASE_ENABLEMENT, record_lifecycle_event
from ..actions.stop_attribution import stopped_by_the_run_class
from ..bringup import ARGV_INVALID
from ..framework.artifacts import candidate_key
from ..state._shared_state.attempt_audit import _AUDIT_ACTIONS
from ..state.shared_state import SharedState, resolve_graded_comparison
from ..bus.message_bus import Message
from hyperloom.inference_optimizer.session.resume_guard import confirmed_stopped_scope
from hyperloom.inference_optimizer.grid_server_args import (
    dedupe_extra_server_args,
    merge_cumulative_extra_server_args,
)
from ..state.round_store import ABANDONED, BOOTED, FAILED
from ..state.task_registry import Task
from ..actions.executors.benchmark_result import is_valid_measurement
from ..actions.executors.integrate_patch import KEEP_STATUSES
from ..actions.executors._accuracy_gate import (
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    EVAL_KIND_ACCURACY_UNAVAILABLE,
    accuracy_meets_floor,
)
from ..knowledge.agent_kb import PatchKB
from .proposals import PendingProposal
from ..measurement.integrate_performance import integrate_measurement_fields
from ..measurement.runtime_findings import KEEP_REASON_CORRECTNESS_FIX
from ._geak_harness_replay import GeakHarnessReplay
from ._geak_revalidation import GeakRevalidation
from ._launch_identity import LaunchIdentity
from ._promote_outcome import _PromoteOutcome

if TYPE_CHECKING:
    from hyperloom.common.perf_metric import GradedComparison
    from .coordinator import Coordinator
import logging as _logging

log = _logging.getLogger(__name__)


# Combined baseline-failure backstop: fast-fail after this many TOTAL baseline failures.
_BASELINE_MAX_TOTAL_FAILURES: int = 3
_BASELINE_FAILED_STOP_REASON = "baseline_failed"
#: Baseline failures a retry cannot change, each mapped to the stop reason it ends the run with on its FIRST
#: occurrence, with no streak.
_STOPS_ON_FIRST_OCCURRENCE: dict[str, str] = {
    # A retry recomposes the same argv the installed parser already refused.
    ARGV_INVALID: ARGV_INVALID,
    # The server booted and measured a cold cache. The next attempt launches the same server and no framework patch
    # changes the operator's flag; ``_authorable_launch_log`` keeps it out of enablement.
    PREFIX_CACHE_OFF_ERROR_CLASS: _BASELINE_FAILED_STOP_REASON,
}
# Default resume-drift floor (%): a re-measured current_best below this fraction of its recorded tput is flagged as
# drift.
_DEFAULT_RESUME_DRIFT_FLOOR_PCT: float = 95.0


def _is_ray_cluster_infeasible(result_payload: Mapping[str, Any]) -> bool:
    """Whether a baseline failed because the Ray cluster can never place its round."""
    return RAY_INFEASIBLE_MARKER in str(result_payload.get("error") or "")


def _authorable_launch_log(result_payload: dict[str, Any] | None, *, err_class: str) -> str:
    """The failed boot's text the next enablement round authors against, or ``""``.

    A failure that never reached a framework error is no wall: as
    integrate_patch's "before" boot it passes any patch that starts the server.
    """
    if not isinstance(result_payload, dict) or _is_ray_cluster_infeasible(result_payload):
        return ""
    if err_class in (
        "fast_exit_arg_error",
        AGENTX_PREFLIGHT_ERROR_CLASS,
        PREFIX_CACHE_OFF_ERROR_CLASS,
    ) or stopped_by_the_run_class(err_class):
        return ""
    parts: list[str] = []
    for key in ("error", "stderr", "log_tail", "log_excerpt", "traceback", "reason"):
        val = result_payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, (list, tuple)):
            joined = "\n".join(str(x) for x in val if str(x).strip())
            if joined.strip():
                parts.append(joined.strip())
    return "\n".join(parts).strip()


def _graded_source(measurement: Mapping[str, Any], output_tput: float) -> dict[str, Any]:
    """*measurement* with the caller's resolved output throughput stamped in.

    A winner record does not always carry its own throughput, so grading must
    not read back out what the caller already resolved.
    """
    return {**measurement, "output_throughput": float(output_tput)}


# Statuses meaning a patch was applied and measured, then rolled back or rejected on measured grounds.
_INTEGRATE_REVERT_STATUSES: frozenset[str] = frozenset({"reverted", "accuracy_unavailable_reject", "regression"})


def _integrate_marker_verdict(status: str, phase: str) -> str:
    """Classify one integrate window from its verdict and its durable phase.

    The pair is the whole truth about an attempt's obligations, so the live
    promotion path and the resume recovery path read it here rather than each
    keeping its own table.

    Returns:
        ``"retained"`` when the candidate was deliberately left in the tree and
        the user's stash was popped, ``"settled"`` when the tree was restored
        to its pre-attempt state, ``"inflight"`` when an apply-only attempt is
        still waiting for its benchmark, and ``"incomplete"`` when the two
        disagree and no obligation can be discharged from them.
    """
    if phase == "accepted" and status in KEEP_STATUSES:
        return "retained"
    if phase == "restored" and status not in KEEP_STATUSES and status != "applied_no_bench":
        return "settled"
    if status == "applied_no_bench" and phase in {"applied", "applied_with_restored_stash"}:
        return "inflight"
    return "incomplete"


def _confirmed_task_outcome(task: Task) -> dict[str, Any] | None:
    """Read a task's own record of the result its executor produced.

    ``SubAgentRunner`` writes the full outcome onto the task row before the
    ``delegated_result`` reaches the bus, so a crash inside that window leaves
    the verdict durable here and nowhere else. An unconfirmed physical cleanup
    makes that outcome diagnostic only — the work it describes may still hold
    resources — so it is not returned.

    TODO: ``bringup.reconcile._terminal_by_observation`` reads the same evidence
    shape under slightly different rules; the two want one reader.
    """
    for entry in reversed(getattr(task, "history", None) or []):
        if not isinstance(entry, dict):
            continue
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("outcome"), dict):
            continue
        if evidence.get("cleanup_confirmed") is not True:
            return None
        result = evidence["outcome"].get("result")
        return result if isinstance(result, dict) else None
    return None


def _integrate_stack_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the members a KEEP promotes.

    Only the kernel-stack lane produces multi-member results, and it binds them to their ledger rows before
    promoting, so the members are read off the result rather than bound again here.
    """
    from ..phases.kernel_stack import resolve_stack_members

    raw_members = result.get("stack_kernel_ids")
    membership = {"stack_validation": isinstance(raw_members, list) and len(raw_members) > 1, **result}
    members = resolve_stack_members(membership)
    return {"stack_kernel_ids": list(members), "stack_validation": membership["stack_validation"]}


def _patch_member_list(raw: Any) -> list[Any]:
    """The members of one patch field, whether it names a single path or several."""
    if isinstance(raw, (str, Path)):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _declared_patch_members(result: Mapping[str, Any], params: Mapping[str, Any]) -> list[Any] | None:
    """The patch members a settled KEEP names, or ``None`` when it names no field at all.

    An executor verdict's ``patches_applied`` is authoritative, including an
    explicit empty list left after every candidate was rejected: a field that
    names nothing is still an answer, and the workspace must not be searched
    behind it.
    """
    if "patches_applied" in result:
        return _patch_member_list(result.get("patches_applied"))
    declared: list[Any] = []
    named = "patches" in params
    for key in ("patches", "prior_patches", "patch_path", "patch"):
        if key in result:
            named = True
            declared.extend(_patch_member_list(result.get(key)))
    declared.extend(_patch_member_list(params.get("patches")))
    return declared if named else None


def _workspace_patch_files(workspace: str) -> list[Path]:
    """The patch files a run workspace holds.

    A recovery path for the older result envelopes that persisted only
    ``workspace``; a raw framework-agent diff comes back in ``patches_applied``.
    """
    root = Path(workspace) if workspace else None
    if root is None or not root.is_dir():
        return []
    found: list[Path] = []
    for base in (root, root / "patches", root / "worktree" / "patches"):
        if base.is_dir():
            found.extend(sorted(base.glob("*.patch")))
            found.extend(sorted(base.glob("*.diff")))
    return found


def _existing_patch_paths(candidates: list[Any]) -> tuple[list[Path], list[str]]:
    """Split declared patch members into the files that are on disk and the ones that are not."""
    resolved: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for raw in candidates:
        if isinstance(raw, Mapping):
            raw = raw.get("patch_path") or raw.get("patch_ref") or raw.get("patch_file") or ""
        raw_text = str(raw or "").strip()
        if not raw_text:
            missing.append("<empty-patch-member>")
            continue
        path = Path(raw_text)
        if not path.is_file():
            missing.append(raw_text)
            continue
        try:
            canonical = path.resolve()
        except OSError:
            canonical = path
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    return resolved, missing


def _keep_origin_provenance(result: Mapping[str, Any], sources: list[Path], *, realized: bool) -> dict[str, Any]:
    """How a staged KEEP's overlay was captured, and which checkout it came from.

    ``apply_root`` becomes a per-ref answer once the refs exist; the rest is
    recorded so a section that would not replay can still be read back.
    """
    return {
        "base_sha": str(result.get("base_sha") or ""),
        "complete": bool(result.get("source_snapshot_complete", True)),
        "artifacts_outside_root": int(result.get("source_artifacts_outside_root") or 0),
        "realized": bool(realized),
        HOST_ORIGIN_KEY: {
            "apply_root": str(result.get("framework_root") or ""),
            "snapshot": str(result.get("source_snapshot") or ""),
            "manifest": str(result.get("source_manifest") or ""),
            "sources": [str(path) for path in sources],
        },
    }


def _profile_trace_path(result: Mapping[str, Any]) -> Any:
    """The trace a profile round produced, surfaced so Orch hands trace_analyze a real path.

    A round whose trace input was never ready has none, whatever files it left
    behind.
    """
    if result.get("trace_input_ready") is False:
        return None
    return result.get("main_trace_path") or (result.get("trace_files") or [None])[0]


def _unstageable_patch_sources(row: Mapping[str, Any]) -> list[str]:
    """The patch members a staged outbox row names that are not on disk at drain time."""
    missing = [str(path) for path in (row.get("missing_patch_sources") or []) if str(path).strip()]
    if row.get("include_patches"):
        missing.extend(
            str(path)
            for path in (row.get("patch_sources") or [])
            if str(path).strip() and not Path(str(path)).is_file()
        )
    return list(dict.fromkeys(missing))


def _integrate_keep_tput(result: Mapping[str, Any], measurement: Mapping[str, Any]) -> float | None:
    """The throughput an integrate KEEP is promoted on, or ``None`` when it measured none.

    A bench_result of its own outranks the envelope, which keeps the figure the
    attempt started from under the same name.
    """
    if measurement is not result and "output_throughput" in measurement:
        measured = measurement["output_throughput"]
    else:
        measured = result.get("output_throughput")
        if measured is None:
            measured = result.get("new_tput")
    if not isinstance(measured, (int, float)) or measured <= 0:
        return None
    return float(measured)


def _lever_for_keep(task_params: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Name the lever a settled KEEP moved, reading the delivery first.

    What came back outranks what was asked for: a mandate that goes out naming
    one lever routinely returns another, and the result carries the applier's
    own markers.
    """
    return patch_lever_kind(result) or patch_lever_kind(task_params)


#: The one owner label a patch KEEP stages under. The three-column layout has a
#: single ``patch`` column, so explore- and framework-agent lifts both stage
#: under this marker. Attribution keeps its own explore/framework split on the
#: lever kind.
_PATCH_KEEP_OWNER = "PATCH"

#: Levers whose overlays feed the one patch column. ``kernel`` publishes through
#: its own column and is absent; a KEEP that names none of these falls back to
#: the authoring phase to decide.
_PATCH_COLUMN_LEVERS = frozenset({LEVER_CONFIG, LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR, LEVER_ENABLEMENT})


def _is_patch_column_keep(task_params: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Whether a settled KEEP's overlay belongs in the patch column.

    A patch-column lever answers on its own; a KEEP that names no such lever
    (kernel, or none recorded) falls back to the authoring phase, exactly as the
    old owner label did before explore and framework shared one column. This is
    now one boolean rather than a per-agent owner, because there is one column.
    """
    lever = _lever_for_keep(task_params, result)
    if lever in _PATCH_COLUMN_LEVERS:
        return True
    phase = str(task_params.get("source_phase") or result.get("source_phase") or "").strip().upper()
    return phase in {"EXPLORE", "FRAMEWORK_AGENT"}


class IntegrateRecoveryIncomplete(RuntimeError):
    """An integration cannot publish a result while its recovery is unconfirmed."""


def _source_layer_handles(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source-layer fields a ``source_patch`` lift must carry.

    Every field here has to survive the hop from the executor result into the
    optimization_stack entry, where ``build_env_spec`` reads it. Assembling them
    in one place is what keeps a second lift path from forwarding a subset and
    degrading the overlay it feeds.

    Args:
        result: An integrate_patch KEEP result.

    Returns:
        dict[str, Any]: Handles to merge into the lift. ``source_snapshot_complete``
            is present only when the result recorded it, so entries predating the
            field still fall back to reading their manifest.
    """
    handles: dict[str, Any] = {
        "source_snapshot": result.get("source_snapshot") or "",
        "source_manifest": result.get("source_manifest") or "",
        "target_files": [str(path) for path in (result.get("target_files") or []) if str(path).strip()],
        "framework_root": result.get("framework_root") or "",
        "base_sha": result.get("base_sha") or "",
        "source_import_root": result.get("source_import_root") or "",
    }
    if "source_snapshot_complete" in result:
        handles["source_snapshot_complete"] = bool(result["source_snapshot_complete"])
    return handles


def _baseline_perf_axes(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """The axes the baseline's own perf snapshot overrides on ``current_best``."""
    if not snapshot:
        return {}
    axes: dict[str, Any] = {
        "total_throughput": snapshot["total_throughput"],
        "e2e_norm_intvty_p90": snapshot["e2e_norm_intvty_p90"],
    }
    for name in ("input_throughput", "tpot_p90_ms", "e2e_norm_intvty_p50", "duration_seconds", "request_error_rate"):
        if snapshot.get(name) is not None:
            axes[name] = snapshot[name]
    return axes


def _integrate_verdict(status: str, *, lifted: bool, kept: bool) -> Verdict:
    """The settlement verdict a finished integrate_patch attempt earns.

    ``kept_inert`` means applied with every switch off: nothing was adopted,
    yet the patch stays on disk as registered levers, so it is not a failure
    either.
    """
    if status == "kept_inert":
        return Verdict.RECORDED
    if kept:
        return Verdict.ADOPTED if lifted else Verdict.REFUSED
    if status in _INTEGRATE_REVERT_STATUSES:
        return Verdict.REVERTED
    return Verdict.FAILED


def _integrate_keep_recipe_delta(result: Mapping[str, Any], task_params: Mapping[str, Any]) -> dict[str, Any]:
    """The recipe delta a settled integrate_patch KEEP re-applies.

    What the applier reports it applied outranks what the mandate asked for.
    """
    return {
        "extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "remove_args": to_str_list(result.get("remove_args_applied") or task_params.get("remove_args")),
        "unset_envs": to_str_list(result.get("unset_envs_applied") or task_params.get("unset_envs")),
        "args_mode": str(result.get("args_mode") or task_params.get("args_mode") or "append").strip().lower(),
    }


def _integrate_keep_origin(result: Mapping[str, Any], task_params: Mapping[str, Any]) -> dict[str, Any]:
    """Who a settled integrate_patch KEEP is attributed to.

    The mandate's stamp and the deliverable's markers travel together, and the
    result wins on a collision.
    """
    domain = str(task_params.get("domain") or task_params.get("source_domain") or result.get("domain") or "").strip()
    provenance = str(task_params.get("provenance") or result.get("provenance") or "").strip()
    if domain and not provenance.startswith("specialist:"):
        provenance = f"specialist:{domain}"
    fields: dict[str, Any] = {"provenance": provenance or "integrate_patch", "scope": "source_patch"}
    lever_kind = _lever_for_keep(task_params, result)
    if lever_kind:
        fields["lever_kind"] = lever_kind
    source_phase = str(task_params.get("source_phase") or result.get("source_phase") or "").strip()
    if source_phase:
        fields["source_phase"] = source_phase
    if domain:
        fields["domain"] = domain
    if task_params.get("gap_layer"):
        fields["gap_layer"] = str(task_params.get("gap_layer"))
    if task_params.get("framework_agent_authoring"):
        fields["framework_agent_authoring"] = True
    # A verified correctness fix was bounded by the integrate gate, not by a gain.
    if result.get("keep_reason") == KEEP_REASON_CORRECTNESS_FIX:
        fields["keep_reason"] = KEEP_REASON_CORRECTNESS_FIX
        fields["resolves_finding"] = str(result.get("resolves_finding") or "")
        fields["attribution_eligible"] = False
    return fields


def _integrate_keep_variant(
    result: Mapping[str, Any],
    task: "Task | None",
    measurement: Mapping[str, Any],
    tput: float,
) -> dict[str, Any]:
    """The winning variant a settled integrate_patch KEEP lifts.

    The durable source-layer handles are what keep ``current_best``
    relaunchable and reproducible in the GEAK baseline.
    """
    task_params = (getattr(task, "params", None) or {}) if task is not None else {}
    specialist_task_id = str(result.get("specialist_task_id") or task_params.get("specialist_task_id") or "")
    return {
        "name": specialist_task_id or "integrate_patch_keep",
        "task_id": getattr(task, "task_id", "") if task is not None else "",
        "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "candidate_extra_envs": dict(result.get("extra_envs_applied") or {}),
        "recipe_delta": _integrate_keep_recipe_delta(result, task_params),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "tput": tput,
        **integrate_measurement_fields(measurement),
        **_source_layer_handles(result),
        **_integrate_keep_origin(result, task_params),
    }


class _ExploreLift(NamedTuple):
    """What accepting one explore round's winners changed."""

    promoted: bool
    last_lifted: dict[str, Any] | None


def _explore_audit_extras(result: Mapping[str, Any]) -> dict[str, Any]:
    """What an explore round's audit row records about it."""
    winners = result.get("winners") or []
    best_winner = result.get("best_variant")
    return {
        "round_id": str(result.get("round_id") or ""),
        "winners_count": (len(winners) if isinstance(winners, list) else 0),
        "losers_count": len(result.get("losers") or []),
        "skipped_dup_count": len(result.get("skipped_dup") or []),
        "best_variant_name": (best_winner.get("name") if isinstance(best_winner, dict) else None),
        "best_gain_pct_vs_base": result.get("best_gain_pct"),
        "output_throughput": result.get("output_throughput"),
        "explore_grid_exhausted": bool(result.get("explore_grid_exhausted")),
    }


class _LiftRequest(NamedTuple):
    """A measured winner a caller asks the lift to accept."""

    task_kind: str
    tput: float
    variant: dict[str, Any]
    gap_canonical_id: str
    entry_extra: Mapping[str, Any]


class _ResolvedLaunch(NamedTuple):
    """The cumulative launch a winner lifts ``current_best`` to."""

    variant_name: Any
    candidate_args: str
    full_args: str
    overlay: str


def _is_prebaseline_enablement(task_kind: str, bv: Mapping[str, Any], baseline_tput: Any) -> bool:
    """Whether this lift is an enablement patch landing before any runnable baseline.

    Such a patch has no runnable reference to grade against, so its
    configuration is retained without gain attribution.
    """
    return (
        task_kind == "integrate_patch"
        and float(baseline_tput or 0.0) <= 0.0
        and bv.get("baseline_enablement") is True
        and bv.get("attribution_eligible") is False
    )


def _resolved_launch(previous: Mapping[str, Any], bv: Mapping[str, Any]) -> _ResolvedLaunch:
    """The cumulative launch a winner lifts ``current_best`` to.

    Every args string passes through :func:`strip_benchmark_harness_flags`, the
    previous ``current_best`` included since it is re-merged onto the winner.
    A removal or replace winner publishes its own effective cumulative config
    from ExploreExecutor, so prepending the prior launch would reintroduce
    flags it deliberately removed. An authored-kernel overlay stays active
    until another KEEP replaces it.
    """
    candidate_args = strip_benchmark_harness_flags(bv.get("candidate_extra_server_args") or bv.get("extra_server_args"))
    full_args = strip_benchmark_harness_flags(bv.get("extra_server_args"))
    controls_effective = bool(
        bv.get("remove_args") or bv.get("unset_envs") or str(bv.get("args_mode") or "").strip().lower() == "replace"
    )
    if controls_effective:
        full_args = dedupe_extra_server_args(full_args)
    else:
        full_args = merge_cumulative_extra_server_args(
            strip_benchmark_harness_flags(previous.get("extra_server_args")),
            candidate_args,
            full_args,
        )
    return _ResolvedLaunch(
        variant_name=bv.get("name"),
        candidate_args=candidate_args,
        full_args=full_args,
        overlay=str(bv.get("final_overlay") or "").strip() or str(previous.get("final_overlay") or "").strip(),
    )


def _already_stacked(stack: list[Any], request: _LiftRequest, variant_name: Any) -> bool:
    """Whether the winner's configuration is already applied on the stack.

    Keyed by ``(action, variant_name)`` and by fingerprint, so a rerun under a
    new name cannot re-apply a config the stack already carries.
    """
    rows = [row for row in stack if isinstance(row, dict)]
    applied = {(str(row.get("action")), str(row.get("variant_name"))) for row in rows}
    if (request.task_kind, str(variant_name or "")) in applied:
        return True
    fingerprint = str(request.variant.get("fingerprint") or "")
    return bool(fingerprint) and any(fingerprint == str(row.get("fingerprint") or "") for row in rows)


def _variant_fingerprint(bv: Mapping[str, Any], candidate_args: str, full_args: str) -> str:
    """The winner's stable join key, computed from its config when it carries none.

    Breakdown attribution maps an explore gain to its specialist provenance
    through this key.
    """
    recorded = str(bv.get("fingerprint") or "").strip()
    if recorded:
        return recorded
    from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint

    return canonical_fingerprint(
        candidate_args or full_args,
        dict(bv.get("extra_envs") or {}),
        remove_args=bv.get("remove_args"),
        unset_envs=bv.get("unset_envs"),
        args_mode=str(bv.get("args_mode") or "append"),
    )


def _variant_attribution(bv: Mapping[str, Any]) -> dict[str, Any]:
    """What a winner's stack row is attributed to, and whether it may be republished."""
    fields: dict[str, Any] = {}
    # Authored-kernel names separate a config gain from a gain measured with a
    # kernel loaded. Absent on every flags-only variant.
    kernels = [str(name).strip() for name in (bv.get("accepted_kernels") or []) if str(name).strip()]
    if kernels:
        fields["accepted_kernels"] = kernels
    for key in ("baseline_enablement", "attribution_eligible", "recipe_publishable", "framework_agent_authoring"):
        if key in bv:
            fields[key] = bool(bv.get(key))
    for key in ("domain", "gap_layer", "keep_reason", "resolves_finding"):
        if bv.get(key):
            fields[key] = str(bv.get(key))
    provenance = str(bv.get("provenance") or "").strip()
    if provenance:
        fields["provenance"] = provenance
    if bv.get("task_id"):
        fields["task_id"] = str(bv.get("task_id"))
    return fields


def _variant_launch_controls(bv: Mapping[str, Any]) -> dict[str, Any]:
    """The removal and replace controls a winner's stack row re-applies."""
    fields: dict[str, Any] = {}
    for key in ("remove_args", "unset_envs", "args_mode"):
        if bv.get(key):
            fields[key] = bv.get(key)
    recipe_delta = bv.get("recipe_delta")
    if isinstance(recipe_delta, Mapping):
        fields["recipe_delta"] = {
            "extra_server_args": str(recipe_delta.get("extra_server_args") or "").strip(),
            "extra_envs": dict(recipe_delta.get("extra_envs") or {}),
            "remove_args": to_str_list(recipe_delta.get("remove_args")),
            "unset_envs": to_str_list(recipe_delta.get("unset_envs")),
            "args_mode": str(recipe_delta.get("args_mode") or "append").strip().lower(),
        }
    if bv.get("effective_extra_server_args"):
        fields["effective_extra_server_args"] = dedupe_extra_server_args(
            strip_benchmark_harness_flags(bv.get("effective_extra_server_args"))
        )
    return fields


def _variant_source_layer(bv: Mapping[str, Any]) -> dict[str, Any]:
    """The durable source-layer handles a winner's stack row carries."""
    fields: dict[str, Any] = {}
    for key in ("source_snapshot", "source_manifest", "source_import_root", "framework_root", "base_sha"):
        value = bv.get(key)
        if value:
            fields[key] = str(value)
    # Copied unconditionally: False is the meaningful value that marks a
    # snapshot unusable, so truthiness must not drop it.
    if "source_snapshot_complete" in bv:
        fields["source_snapshot_complete"] = bool(bv["source_snapshot_complete"])
    target_files = [str(path) for path in (bv.get("target_files") or []) if str(path).strip()]
    if target_files:
        fields["target_files"] = target_files
    scope = str(bv.get("scope") or "").strip()
    if scope:
        fields["scope"] = scope
    return fields


def _preserves_measured_eval_trigger(state: Any, result_payload: Mapping[str, Any]) -> bool:
    """Whether the stored enablement trigger outranks this eval failure's own.

    A run that never executed the eval characterizes nothing: it reports
    ``accuracy_unavailable`` with no task, metric or source. Letting it
    overwrite a trigger that did measure an accuracy downgrades real
    ``accuracy_below_floor`` evidence, and the next enablement attempt loses the
    measurement it is supposed to reproduce. Contract fingerprints cannot gate
    this: ``RUN_EVAL`` is itself a contract field, so the eval-less run's
    fingerprint never matches the measured one.
    """
    if result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY) != EVAL_KIND_ACCURACY_UNAVAILABLE:
        return False
    if to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY)) is not None:
        return False
    stored_kind = str(state.enablement.baseline_eval_kind or "")
    return bool(stored_kind) and stored_kind != EVAL_KIND_ACCURACY_UNAVAILABLE


def _adopt_eval_trigger(state: Any, result_payload: Mapping[str, Any]) -> None:
    """Make this eval failure the evidence the enablement lane authors against.

    Seeds ``enablement_launch_log`` from the evidence so the FRAMEWORK pump
    dispatches even when the failure carries no boot log.
    """
    enablement = state.enablement
    cfg = result_payload.get("materialized_config")
    if isinstance(cfg, str) and cfg:
        enablement.probe_config_path = cfg
    fingerprint = result_payload.get(BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY)
    if isinstance(fingerprint, str) and fingerprint:
        enablement.eval_contract_fingerprint = fingerprint
    incoming_kind = result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY)
    if isinstance(incoming_kind, str) and incoming_kind:
        enablement.baseline_eval_kind = incoming_kind
    observed = to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY))
    if observed is not None:
        enablement.observed_accuracy = float(observed)
    enablement.observed_task = str(result_payload.get("accuracy_task") or "")
    enablement.observed_metric = str(result_payload.get("accuracy_metric") or "")
    evidence = str(result_payload.get(BASELINE_EVAL_EVIDENCE_KEY) or "")
    if evidence:
        enablement.baseline_eval_evidence = evidence[:4000]
        enablement.record_launch_observation(
            log=evidence, observation_path=str(result_payload.get("boot_observation_path") or "")
        )


class _StackKey(NamedTuple):
    """Where a KEEP sits on the optimization stack."""

    action: str
    variant: str


def _stacked_keep_keys(stack: Any) -> set[_StackKey]:
    """Every place the optimization stack already carries a KEEP."""
    return {
        _StackKey(str(entry.get("action") or ""), str(entry.get("variant_name") or ""))
        for entry in (stack or [])
        if isinstance(entry, dict)
    }


def _orphan_stack_key(kind: str, result: Mapping[str, Any]) -> _StackKey | None:
    """Where a recorded KEEP would sit on the stack, or ``None`` for a kind that never stacks.

    ``framework_agent`` stacks under the framework family label, keyed by the
    canonical candidate key, so it is reconciled on both.
    """
    if kind == "integrate_patch":
        return _StackKey(kind, str(result.get("specialist_task_id") or ""))
    if kind == "framework_agent":
        candidate = result.get("candidate")
        return _StackKey(FRAMEWORK_STACK_ACTION, candidate_key(candidate if isinstance(candidate, dict) else None))
    if kind == "explore":
        best = result.get("best_variant") or {}
        return _StackKey(kind, str((best.get("name") if isinstance(best, dict) else "") or ""))
    return None


class _IntegrateMarker(NamedTuple):
    """What a crashed integrate window's own durable records say it reached."""

    phase: str
    outcome: dict[str, Any] | None
    verdict: str


class _KeptResultScan(NamedTuple):
    """What a search of the event log found for one integrate window.

    ``scanned`` is false when the log could not be read, which establishes
    nothing -- a different answer from having found no KEEP.
    """

    scanned: bool
    kept: dict[str, Any] | None


def _integrate_marker_of(pending: Mapping[str, Any], task: Task | None) -> _IntegrateMarker:
    """Read a crashed integrate window's marker: the phase it reached, its outcome, their verdict."""
    recovery = pending.get("recovery")
    phase = str(recovery.get("phase") or "") if isinstance(recovery, dict) else ""
    outcome = _confirmed_task_outcome(task) if task is not None else None
    verdict = _integrate_marker_verdict(str((outcome or {}).get("status") or ""), phase) if outcome else ""
    return _IntegrateMarker(phase=phase, outcome=outcome, verdict=verdict)


def _task_recorded_a_result(task: Task) -> bool:
    """Whether a task row's history shows the runner already wrote a result for it."""
    return any(
        any(key in (row.get("evidence") or {}) for key in ("result_keys", "result", "outcome"))
        for row in task.history
        if isinstance(row, dict)
    )


def _replayed_explore_variant(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The winning variant a recorded explore KEEP replays, or ``None`` when it names none."""
    best = result.get("best_variant")
    if not isinstance(best, dict) or not best.get("name"):
        return None
    return dict(best)


def _replayed_recipe_delta(result: Mapping[str, Any]) -> dict[str, Any]:
    """The recipe delta a recorded patch KEEP re-applies."""
    return {
        "extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "remove_args": to_str_list(result.get("remove_args_applied") or result.get("remove_args")),
        "unset_envs": to_str_list(result.get("unset_envs_applied") or result.get("unset_envs")),
        "args_mode": str(result.get("args_mode") or "append").strip().lower(),
    }


def _replay_origin_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Who a recorded patch KEEP is attributed to: provenance, authoring phase, domain, gap layer.

    A KEEP whose authoring phase was never recorded cannot be republished as a
    recipe, and neither can an enablement landing: the first because nothing can
    say which column it belongs to, the second because it describes this host.
    """
    domain = str(result.get("domain") or result.get("source_domain") or "").strip()
    provenance = str(result.get("provenance") or "").strip()
    if domain and not provenance.startswith("specialist:"):
        provenance = f"specialist:{domain}"
    fields: dict[str, Any] = {"provenance": provenance or "integrate_patch", "scope": "source_patch"}
    lever_kind = patch_lever_kind(result)
    if lever_kind:
        fields["lever_kind"] = lever_kind
    source_phase = patch_owner_phase(result)
    if source_phase:
        fields["source_phase"] = source_phase
    else:
        fields["recipe_publishable"] = False
    if domain:
        fields["domain"] = domain
    gap_layer = str(result.get("gap_layer") or "").strip()
    if gap_layer:
        fields["gap_layer"] = gap_layer
    if result.get("framework_agent_authoring"):
        fields["framework_agent_authoring"] = True
    if result.get("enablement") or result.get("enablement_landing"):
        fields["recipe_publishable"] = False
    return fields


def _replayed_integrate_variant(result: Mapping[str, Any], tput: float) -> dict[str, Any] | None:
    """The winning variant a recorded integrate_patch KEEP replays, or ``None`` when it names none.

    Carries the same durable source-layer handles as the live KEEP lift, so a
    ``source_patch`` recovered on this path is equally reproducible in the GEAK
    baseline and no path is left snapshot-less.
    """
    from hyperloom.common.perf_metric import constraint_fields, graded_axes_of

    specialist_task_id = str(result.get("specialist_task_id") or "")
    if not specialist_task_id:
        return None
    bench = result.get("bench_result") or result
    return {
        "name": specialist_task_id,
        "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
        "candidate_extra_envs": dict(result.get("extra_envs_applied") or {}),
        "recipe_delta": _replayed_recipe_delta(result),
        "extra_envs": dict(result.get("extra_envs_applied") or {}),
        "tput": float(tput),
        **graded_axes_of(bench),
        # ``graded_axes_of`` carries the throughput axes only; the latency and
        # power budgets grade on these and fail closed without them.
        **constraint_fields(bench),
        "workspace": result.get("workspace"),
        **_source_layer_handles(result),
        **_replay_origin_fields(result),
    }


# task.params fields fingerprinted by the self-loop guard.
_BASELINE_FINGERPRINT_KEYS: tuple[str, ...] = (
    "benchmark_script",
    "result_dir",
    "extra_server_args",
    "extra_envs",
    "model_path",
    "gpu_type",
    "config_path",
    "disable_run_eval",
)


#: Server flags that assert a workload-shape field, each with what it asserts.
_WORKLOAD_ARG_FLAGS: dict[str, dict[str, bool]] = {
    "--enable-chunked-prefill": {"chunked_prefill_enabled": True},
    "--disable-chunked-prefill": {"chunked_prefill_enabled": False},
    "--enable-torch-compile": {"enable_torch_compile": True},
}
#: Server flags whose following token carries the value of a workload-shape count.
_WORKLOAD_ARG_COUNTS: dict[str, str] = {
    "--max-running-requests": "max_running_requests",
    "--max-num-seqs": "max_num_seqs",
}


def _server_arg_workload_fields(extra_server_args: str) -> dict[str, Any]:
    """The workload-shape fields a Magpie extra-server-args line declares.

    A count whose CLI value is not an integer leaves its field unset: the tag
    describes the workload, and an unparsable number describes nothing.
    """
    out: dict[str, Any] = {}
    tokens = extra_server_args.split()
    for index, token in enumerate(tokens):
        out.update(_WORKLOAD_ARG_FLAGS.get(token, {}))
        count_field = _WORKLOAD_ARG_COUNTS.get(token)
        if count_field is None or index + 1 >= len(tokens):
            continue
        try:
            out[count_field] = int(tokens[index + 1])
        except ValueError:
            continue
    return out


def _parse_baseline_workload_extra(yaml_path: str) -> dict[str, Any]:
    """Extract KB workload-tag fields from a baseline-materialized Magpie YAML."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    out: dict[str, Any] = {}
    bm = cfg.get("benchmark") if isinstance(cfg, dict) else None
    if not isinstance(bm, dict):
        return out
    for key in ("workload_mode", "quant_scheme"):
        value = bm.get(key)
        if value not in (None, "", 0):
            out[key] = value
    envs = bm.get("envs") if isinstance(bm.get("envs"), dict) else {}
    declared = envs.get(server_args_env_name(bm.get("framework")))
    out.update(_server_arg_workload_fields(declared.strip() if isinstance(declared, str) else ""))
    # Torch compile may also be a separate env var.
    if "enable_torch_compile" not in out:
        tc_env = envs.get("ENABLE_TORCH_COMPILE")
        if isinstance(tc_env, str):
            out["enable_torch_compile"] = is_truthy(tc_env)
    return out


class WritebackCollaborator(GeakRevalidation, GeakHarnessReplay, LaunchIdentity):
    """Handles result writeback: promotion of measured results, KEEP/REVERT settlement, and resume replay."""

    @staticmethod
    def baseline_params_fingerprint(params: dict[str, Any] | None) -> dict[str, Any]:
        """Project ``params`` to the keys that determine baseline behavior."""
        params = params or {}
        out: dict[str, Any] = {}
        for key in _BASELINE_FINGERPRINT_KEYS:
            if key == "extra_envs":
                envs = params.get(key) or {}
                if isinstance(envs, dict):
                    out[key] = sorted([str(k), str(v)] for k, v in envs.items())
                else:
                    out[key] = None
                continue
            value = params.get(key)
            out[key] = None if value is None else str(value)
        return out

    def __init__(self, coordinator: "Coordinator") -> None:
        super().__init__(coordinator)
        self._lifecycle_last_save: float = 0.0
        self._lifecycle_save_min_interval_s: float = 2.0
        self._resumed_from: dict[str, Any] = {}

    def emit_lifecycle(
        self,
        *,
        step: str,
        status: str,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Record + persist one operator-facing lifecycle event.

        Terminal events (END/ERROR) flush immediately; non-terminal markers are
        debounced within ``_lifecycle_save_min_interval_s``.

        Args:
            step: The machine step name (resolved to a human label downstream).
            status: The lifecycle status (e.g. START / END / ERROR / ENTER).
            artifacts: Optional mapping of produced artifact paths.
            detail: Optional free-text detail.
            duration_s: Optional elapsed seconds for the step.
        """
        record_lifecycle_event(
            self.shared_state,
            step=step,
            status=status,
            artifacts=artifacts,
            detail=detail,
            duration_s=duration_s,
        )
        terminal = status in ("END", "ERROR")
        now = time.monotonic()
        if terminal or (now - self._lifecycle_last_save >= self._lifecycle_save_min_interval_s):
            self.shared_state.save(self.session_dir)
            self._lifecycle_last_save = now

    @staticmethod
    def _keep_patch_sources(
        result: Mapping[str, Any],
        task: "Task | None",
    ) -> tuple[list[Path], list[str]]:
        """Locate authoritative patch files without reconstructing their diff."""
        params = (getattr(task, "params", None) or {}) if task is not None else {}
        declared = _declared_patch_members(result, params)
        workspace = str(result.get("workspace") or "").strip()
        candidates = declared if declared is not None else _workspace_patch_files(workspace)
        return _existing_patch_paths(candidates)

    @staticmethod
    def _provenance_with_apply_roots(
        provenance: Mapping[str, Any],
        refs: list[str],
    ) -> dict[str, Any]:
        """Turn this KEEP's single apply root into a per-ref answer.

        One KEEP lands in one checkout, so every ref it staged shares that root;
        recording it per ref is what lets a Recipe whose KEEPs came from
        different trees stay replayable without anything reconciling them.
        """
        row = {key: value for key, value in provenance.items() if key != HOST_ORIGIN_KEY}
        origin = dict(provenance.get(HOST_ORIGIN_KEY) or {})
        apply_root = str(origin.pop("apply_root", "") or "").strip()
        if apply_root and refs:
            origin["apply_roots"] = {str(ref): apply_root for ref in refs if str(ref).strip()}
        if origin:
            row[HOST_ORIGIN_KEY] = origin
        return row

    def _stage_agent_keep(
        self,
        *,
        owner: str,
        stack_index: int,
        result: Mapping[str, Any],
        task: "Task | None",
        include_patches: bool,
        provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Stage one KEEP into the patch column.

        Returns ``False`` when staging must be retried. The overlay bytes are
        captured now rather than at CLOSE because the worktree they were
        harvested from does not outlive the round. Config and kernel are
        published once from the settled stack at CLOSE.
        """
        if not str(owner or "").strip():
            return False
        patch_kb = PatchKB.open()
        if not patch_kb.active:
            return False
        sources, missing = self._keep_patch_sources(result, task)
        if include_patches and missing:
            log.warning(
                "patch kb: KEEP at stack index %d references missing patch members: %s",
                stack_index,
                missing,
            )
            return False
        if not include_patches:
            return True
        refs: list[str] = []
        if sources:
            refs = patch_kb.stage_patches(sources, stack_index=stack_index)
            if len(refs) != len(sources) or any(not ref for ref in refs):
                return False
        # How the overlay was captured travels with it, so a later session can
        # tell a complete capture from one that could not account for every path,
        # and can apply each overlay to the checkout it was taken from.
        if provenance and not patch_kb.stage_provenance(
            stack_index=stack_index,
            **self._provenance_with_apply_roots(provenance, refs),
        ):
            return False
        return True

    def _enqueue_agent_keep_outbox(
        self,
        *,
        stack_index: int,
        result: Mapping[str, Any],
        task: "Task | None",
        include_patches: bool,
    ) -> None:
        """Persist an idempotent section handoff to run after state durability.

        The caller has already decided this KEEP feeds the patch column, so
        every stored field (row id, outbox owner, the stack's
        ``kb_required_owner``) uses the single patch-owner marker: the
        three-column layout has one patch column, not a per-agent one.
        """
        from ..knowledge.remote_recipe import KnowledgeSections

        if (
            str(os.environ.get("KNOWLEDGE_STORE_MODE") or "local").strip() != "remote"
            or KnowledgeSections.from_env() is None
        ):
            return
        sources, missing = self._keep_patch_sources(result, task) if include_patches else ([], [])
        # The realized diff is what the tree ended up holding, so it replaces the
        # delivered patch rather than joining it -- staging both would apply the
        # same change twice. It is chosen here, not at drain, because drain only
        # ever sees the row.
        realized = Path(str(result.get("source_realized_patch") or "").strip() or ".")
        realized_supersedes = include_patches and realized.is_file()
        if realized_supersedes:
            sources, missing = [realized], []
        row = {
            "id": f"{_PATCH_KEEP_OWNER}:{int(stack_index)}",
            "owner": _PATCH_KEEP_OWNER,
            "stack_index": int(stack_index),
            "include_patches": bool(include_patches),
            "patch_sources": [str(path) for path in sources],
            "missing_patch_sources": missing,
            "provenance": _keep_origin_provenance(result, sources, realized=realized_supersedes),
        }
        outbox = list(self.shared_state.kb_stage_outbox or [])
        if not any(isinstance(existing, dict) and existing.get("id") == row["id"] for existing in outbox):
            outbox.append(row)
        if include_patches and (sources or missing):
            self._claim_patch_section(int(stack_index))
        self.shared_state.kb_stage_outbox = outbox

    def _claim_patch_section(self, stack_index: int) -> None:
        """Record that a stack row cannot publish until its patch section is staged."""
        stack = list(self.shared_state.optimization_stack or [])
        if 0 <= stack_index < len(stack) and isinstance(stack[stack_index], dict):
            stack[stack_index]["kb_required_owner"] = _PATCH_KEEP_OWNER
            self.shared_state.optimization_stack = stack

    def _dead_letter_keep_section(
        self,
        dead_letter: list[dict[str, Any]],
        row: Mapping[str, Any],
        missing: list[str],
    ) -> list[dict[str, Any]]:
        """Drop a staged KEEP section whose patch sources are gone; release the row that claimed it.

        Returns:
            The dead-letter ledger carrying this row's latest failure.
        """
        owner = str(row.get("owner") or "").upper()
        index = int(row.get("stack_index") or 0)
        stack = self.shared_state.optimization_stack or []
        claimed = 0 <= index < len(stack) and isinstance(stack[index], dict)
        if claimed and str(stack[index].get("kb_required_owner") or "").upper() == owner:
            stack[index].pop("kb_required_owner", None)
        log.warning(
            "%s kb: dropping owner section %s because patch sources are unavailable: %s",
            owner,
            row.get("id"),
            missing,
        )
        failed = {**dict(row), "missing_patch_sources": missing, "reason": "patch_source_missing"}
        return [item for item in dead_letter if item.get("id") != failed.get("id")] + [failed]

    def _drain_agent_keep_outbox(self) -> None:
        """Run section writes only after the authoritative state save."""
        pending = list(self.shared_state.kb_stage_outbox or [])
        if not pending:
            return
        retained: list[dict[str, Any]] = []
        dead_letter = [dict(row) for row in (self.shared_state.kb_stage_dead_letter or []) if isinstance(row, dict)]
        stack = list(self.shared_state.optimization_stack or [])
        for row in pending:
            if not isinstance(row, dict):
                continue
            missing = _unstageable_patch_sources(row)
            if missing:
                dead_letter = self._dead_letter_keep_section(dead_letter, row, missing)
            elif not self._stage_agent_keep(
                owner=str(row.get("owner") or ""),
                stack_index=int(row.get("stack_index") or 0),
                result={"patches": list(row.get("patch_sources") or [])},
                task=SimpleNamespace(params={}),
                include_patches=bool(row.get("include_patches")),
                provenance=row.get("provenance"),
            ):
                retained.append(row)
        self.shared_state.kb_stage_outbox = retained
        self.shared_state.kb_stage_dead_letter = dead_letter[-200:]
        self.shared_state.optimization_stack = stack
        self.shared_state.save(self.session_dir)

    def validate(
        self,
        new_tput: float,
        measurement: Mapping[str, Any],
        *,
        source: str = "writeback",
        measurement_basis: str = "e2e_rebench",
        ts: str | None = None,
    ) -> bool:
        """Validate cumulative gain against baseline, updating the watermark.

        Args:
            new_tput: The newly measured output throughput.
            measurement: The measurement the promotion came from. Candidate and
                baseline are read off it as a pair, so a total-graded session
                cannot divide an output numerator by a total denominator.
            source: Which promotion path produced this figure, recorded so the
                breakdown can name it.
            measurement_basis: How the reading was obtained — ``e2e_rebench``
                for a full-stack revalidation, ``e2e_decision_round`` for the
                round an explore variant was graded on.
            ts: Author-time stamp the caller already minted for this
                promotion; defaults to now.

        Returns:
            Whether a comparable measurement updated the validation watermark.
        """
        if self.shared_state.baseline_tput <= 0:
            return False
        graded_source = _graded_source(measurement, new_tput)
        graded = resolve_graded_comparison(self.shared_state, graded_source, against_baseline=True)
        if not graded.comparable:
            log.info("cumulative gain held: measurement not comparable (%s)", graded.degrade_reason)
            return False
        validated_gain = (
            (graded.candidate - graded.reference) / graded.reference * 100.0 if graded.reference > 0 else 0.0
        )
        ts = str(ts or datetime.now(timezone.utc).isoformat())
        self.shared_state.cumulative_gain_validated = float(validated_gain)
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
        self.shared_state.validated_recipe_generation = int(self.shared_state.working_recipe_generation or 0)
        # The breakdown's own total is the sum of its ledger, so without this
        # record there is nothing for it to disagree with.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import stack_event

            stack_event.record_validation(
                stack_len=self.shared_state.cumulative_gain_validated_stack_len,
                baseline_tput=graded.reference,
                validated_tput=graded.candidate,
                validated_gain_pct=float(validated_gain),
                source=source,
                measurement_basis=measurement_basis,
                graded_objective=graded.objective,
                # The figures grading actually read, not the raw measurement: the caller's resolved output
                # throughput is stamped into it, so the axes recorded here are the ones the verdict was reached on.
                measurement=graded_source,
                ts=ts,
                ttft_mean_ms=measurement.get("ttft_mean_ms"),
                e2el_mean_ms=measurement.get("e2el_mean_ms"),
                ttft_e2el_source=str(measurement.get("ttft_e2el_source") or ""),
                server_launch_flags=str(measurement.get("resolved_server_launch_flags") or ""),
                workspace=measurement.get("workspace"),
            )
        except Exception as exc:
            log.debug("stack validation record failed", exc_info=True)
            # Losing this one costs the export its only independent check on
            # the ledger: with no promoted figure to compare against, the
            # session total falls back to the sum of the very steps it is
            # meant to be checking.
            trace_recording_skipped(
                "session_validation",
                reason="caller raised before the recorder",
                entity=source,
                error=exc,
            )
        return True

    async def _validate_and_watermark(
        self,
        new_tput: float,
        measurement: Mapping[str, Any],
        *,
        source: str = "writeback",
        measurement_basis: str = "e2e_rebench",
        watermark_reason: str,
    ) -> bool:
        """Validate cumulative gain and fire a watermark roofline when it updates."""
        if self.validate(new_tput, measurement, source=source, measurement_basis=measurement_basis):
            await self._coord.phase_kernel.maybe_enqueue_watermark_roofline(reason=watermark_reason)
            return True
        return False

    async def record_integrate_keep(self, result: dict[str, Any]) -> None:
        """Promote a kernel integrate KEEP into the optimization stack.

        Stamps ``cumulative_gain_validated`` and fires a watermark roofline once
        the lift is accepted. No-op without a positive ``new_tput`` or when the
        lift refuses the winner.

        Args:
            result (dict[str, Any]): The integrate-patch executor result.
        """
        measurement = result.get("bench_result") or result
        new_tput = _integrate_keep_tput(result, measurement)
        if new_tput is None:
            return
        stack_fields = _integrate_stack_fields(result)
        # A fusion sibling drained through the shared integrate lane must still
        # land on the stack as ``action="fusion"``: the idempotency short-circuit
        # (``_active_forge_fusion_env_flags``) and the remote-recipe fusion export
        # (``build_kernel_fusion_value``) both key on that label, and the latter
        # also gates on ``last_fusion_integrate`` being a KEEP. The generic path
        # is otherwise unchanged.
        is_fusion = str(result.get("source") or "") == "forge_fusion"
        action_label = str(result.get("action_label") or "").strip() if is_fusion else ""
        lift_kind = action_label or "integrate"
        variant: dict[str, Any] = {
            "name": result.get("kernel_id"),
            "candidate_extra_server_args": result.get("extra_server_args"),
            "extra_envs": {str(k): str(v) for k, v in (result.get("extra_envs") or {}).items()},
            "source_phase": str(self.shared_state.phase or "KERNEL_AGENT"),
            **integrate_measurement_fields(measurement),
        }
        if is_fusion:
            variant["provenance"] = "forge_fusion"
        lifted = self.lift_to_current_best(
            lift_kind,
            float(new_tput),
            variant,
            gap_canonical_id=str(result.get("gap_canonical_id") or "").strip(),
            entry_extra={
                "integration_id": result.get("integration_id"),
                "kernel_id": result.get("kernel_id"),
                "task_group_key": result.get("task_group_key"),
                "identity_route": result.get("identity_route"),
                "patch_path": result.get("patch_path"),
                "target_file": result.get("target_file"),
                "gain_pct": result.get("gain_pct"),
                **stack_fields,
                "backend": result.get("backend"),
                "engine": result.get("engine"),
                # Provenance for a fusion sibling; readers key the stack row on
                # ``action == "fusion"`` above, this just records the producer.
                **({"backend": "forge", "engine": "forge_fusion"} if is_fusion else {}),
                # Preserve committed source-layer identity for recipe export.
                **{key: result[key] for key in ("scope", "operator_id", "base_sha", "keep_commit") if result.get(key)},
            },
        )
        if not lifted:
            return
        if is_fusion:
            self.shared_state.last_fusion_integrate = {**result, "decision": "KEEP"}
        self._coord.recipe_journal.journal_integrate_keep(result, lift_kind=lift_kind, new_tput=float(new_tput))
        await self._validate_and_watermark(
            new_tput, measurement, source="integrate_keep", watermark_reason="integrate_keep_watermark"
        )

    def is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        """Decide whether a settled task result should be promoted.

        Per-kind rules: baseline/profile require a valid measurement, sweep
        requires ``status == "succeeded"``, ``replay_warm_recipe`` always routes
        through promotion (it owns its own failure bookkeeping), and everything
        else is promotable unless ``status == "failed"``.

        Args:
            task_kind (str): The task's kind.
            result (dict[str, Any]): The task result payload.

        Returns:
            bool: ``True`` when the result should go through
                :meth:`promote_to_shared_state`.
        """
        if task_kind == "baseline":
            # A baseline whose accuracy eval failed measured throughput but must
            # not anchor; route it to handle_unpromotable_result for enablement.
            # One refused for a cold prefix cache measured too, but must not anchor
            # either; handle_unpromotable_result stops the run on it.
            if bool(result.get("baseline_eval_failed")) or result.get("error_class") == PREFIX_CACHE_OFF_ERROR_CLASS:
                return False
            return is_valid_measurement(result)
        if task_kind == "profile":
            measurement_status = result.get("measurement_status")
            if measurement_status is not None and str(measurement_status) != "succeeded":
                return False
            # Magpie marks a profiled replay diagnostic_only, so a native AgentX profile
            # is never a valid measurement. The trace is what a profile task produces, and
            # ``ProfileExecutor`` already accepts the round on this field.
            if result.get("round_succeeded") is True:
                return True
            return is_valid_measurement(result)
        # replay_warm_recipe always routes through promote_warm_replay (owns its own failure bookkeeping).
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def record_intervention_for_task(
        self,
        task: "Task",
        result: Any,
        verdict: Verdict,
    ) -> None:
        """Log a settled task's change_type into SharedState.intervention_mix.

        explore → ``config`` when adopted, else ``config_attempt``;
        integrate_patch → ``code_patch`` when adopted, else ``code_patch_attempt``.

        Args:
            task: The settled task whose kind selects the intervention class.
            result: The task result dict; non-dict results are ignored.
            verdict: The settlement verdict.
        """
        if not isinstance(result, dict):
            return
        adopted = verdict is Verdict.ADOPTED
        kind = (task.kind or "").strip()
        if kind == "explore":
            best = result.get("best_variant")
            delta_pct = best.get("gain_pct") if adopted and isinstance(best, dict) else None
            self.shared_state.record_intervention(
                change_type="config" if adopted else "config_attempt",
                action="explore",
                task_id=task.task_id,
                delta_pct=delta_pct if isinstance(delta_pct, (int, float)) else None,
            )
            return
        if kind == "integrate_patch":
            if not str(result.get("status") or "").strip():
                return
            self.shared_state.record_intervention(
                change_type="code_patch" if adopted else "code_patch_attempt",
                action="integrate_patch",
                task_id=task.task_id,
                delta_pct=result.get("delta_pct"),
            )

    def _persist_eval_failure(self, result_payload: dict[str, Any]) -> None:
        """Persist an eval-rooted baseline failure so enablement can re-run it."""
        state = self.shared_state
        was_validation_pending = bool(state.enablement.validation_pending)
        stored_kind = str(state.enablement.baseline_eval_kind or "")
        preserve_measured_trigger = _preserves_measured_eval_trigger(state, result_payload)
        state.enablement.origin = "eval"
        state.enablement.pending = True
        # A failed revalidation reopens the authoring loop; its cost is charged
        # against the round ledger by the caller that settles the round.
        if was_validation_pending:
            state.enablement.validation_pending = False
            enablement_event.record_revalidation_outcome(
                generation=int(state.enablement.revalidation_generation or 0),
                promoted=False,
                task_id=str(state.enablement.revalidation_task_id or ""),
                accuracy=result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY),
                accuracy_floor=state.enablement.accuracy_floor,
                error_class=str(result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY) or ""),
                reason="revalidation eval failed",
            )
        floor = to_float(result_payload.get(BASELINE_EVAL_ACCURACY_FLOOR_KEY))
        if floor is not None:
            state.enablement.accuracy_floor = float(floor)
        if preserve_measured_trigger:
            log.info(
                "enablement: keeping measured trigger kind=%s (accuracy=%s task=%s); "
                "an eval-less baseline reported accuracy_unavailable and must not "
                "overwrite it",
                stored_kind,
                state.enablement.observed_accuracy,
                state.enablement.observed_task,
            )
        else:
            _adopt_eval_trigger(state, result_payload)
        self._record_enablement_eval_trigger()

    def _record_enablement_eval_trigger(self) -> None:
        """Open the enablement lane's event on the eval failure that triggered it.

        Recorded here rather than derived at export: ``origin`` is cleared when
        the lane succeeds, so the fields it reads do not all survive the run.
        Only the first trigger is kept, which is what stops an eval-less
        re-baseline downgrading a measured trigger to an empty one.
        """
        state = self.shared_state
        enablement_event.record_trigger(
            origin=enablement_event.ORIGIN_EVAL,
            mode=str(state.enablement_mode or ""),
            kind=str(state.enablement.baseline_eval_kind or ""),
            evidence=str(state.enablement.baseline_eval_evidence or ""),
            observed_accuracy=state.enablement.observed_accuracy,
            accuracy_floor=state.enablement.accuracy_floor,
            observed_task=state.enablement.observed_task,
            observed_metric=state.enablement.observed_metric,
            eval_contract_fingerprint=state.enablement.eval_contract_fingerprint,
            probe_config_path=state.enablement.probe_config_path,
        )

    async def _reopen_revalidation_window(self) -> None:
        """Leave an enablement revalidation window open for a round the run stopped.

        A round the run stopped measured nothing, so the window stays open and
        no observation is charged.

        The generation advances for the same reason opening a window does: the
        next enqueue's idempotency key must not resolve to the row the run
        stopped. The tracked id goes with it, since the row it names is spent and
        the next enqueue records its own.

        Two callers reach the same round by different routes: the writeback, when
        the reaped row's result is routed, and the resume recovery, when the row
        was cancelled at dispatch and so produced no result to route at all.
        """
        state = self.shared_state
        state.enablement.revalidation_generation = int(state.enablement.revalidation_generation or 0) + 1
        state.enablement.revalidation_task_id = ""
        await self._coord.enablement_lane.settle_enablement_round(ABANDONED, reason="revalidation_stopped_by_the_run")

    async def _record_revalidation_not_promoted(
        self,
        *,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
        stopped_by_the_run: bool,
    ) -> None:
        """Close out an enablement revalidation baseline that did not promote.

        A genuine failure -- boot, OOM, timeout, eval -- closes the revalidation
        window, reopens the authoring loop, and charges its observation to the
        round ledger.

        A round the run stopped is none of those things; what it gets instead, and
        why, is :meth:`_reopen_revalidation_window`.

        Args:
            task: The revalidation baseline task that came back unpromotable.
            result_payload: Its result, read for the launch/traceback text.
            err_class: Its ``error_class``, for the log line.
            stopped_by_the_run: Whether the run stopped the round rather than the
                round saying anything about the baseline.
        """
        state = self.shared_state
        generation = int(state.enablement.revalidation_generation or 0)
        if stopped_by_the_run:
            await self._reopen_revalidation_window()
        else:
            state.enablement.revalidation_task_id = ""
            state.enablement.validation_pending = False
            await self._coord.enablement_lane.settle_enablement_round(FAILED, reason=err_class or "revalidation_failed")
        # A window the run stopped measured nothing, so recording it as a failed
        # revalidation would charge the lane for a clock.
        enablement_event.record_revalidation_outcome(
            generation=generation,
            promoted=False,
            task_id=str(task.task_id or ""),
            accuracy_floor=state.enablement.accuracy_floor,
            error_class="" if stopped_by_the_run else str(err_class or ""),
            reason="stopped by the run" if stopped_by_the_run else "revalidation baseline failed",
        )
        launch_log = _authorable_launch_log(result_payload, err_class=str(err_class or ""))
        if launch_log:
            observed = str(result_payload.get("boot_observation_path") or "")
            state.enablement.record_launch_observation(log=launch_log, observation_path=observed)
        log.warning(
            "enablement revalidation task %s %s (error_class=%s); pending=%s rearm=%s",
            task.task_id,
            "was stopped by the run" if stopped_by_the_run else "failed",
            err_class,
            bool(state.enablement.validation_pending),
            not bool(state.stop_reason),
        )

    def _generic_baseline_failure_stops(
        self,
        task: Task,
        err_class: str,
        result_payload: Mapping[str, Any],
        *,
        retries_suppressed: bool,
    ) -> bool:
        """Count a baseline failure of no specially handled class; whether it ends the run.

        Args:
            task: The failed baseline task.
            err_class: The failure's ``error_class``.
            result_payload: The failed result.
            retries_suppressed: A pending eval or an open enablement lane owns the
                retries, so the streak alone does not stop the run.

        Returns:
            ``True`` when the run should stop as ``baseline_failed``.
        """
        # A process that exited with the same output as last time (modulo numbers)
        # will do so again: stop at two instead of spending a third attempt. The
        # executor signs only ``subprocess_nonzero``, from the process's whole
        # output rather than the 2000-character ``error`` tail; other classes carry
        # a fixed sentence that two unrelated failures share.
        signature = str(result_payload.get("output_signature") or "") if err_class == "subprocess_nonzero" else ""
        repeated = bool(
            signature
            and self.shared_state.baseline_failure_streak >= 1
            and signature == self.shared_state.baseline_last_failure_signature
        )
        self.shared_state.baseline_failure_streak += 1
        self.shared_state.baseline_last_failure_signature = signature
        streak = self.shared_state.baseline_failure_streak
        if _is_ray_cluster_infeasible(result_payload):
            # The Ray cluster cannot place the round, and the runtime has already
            # repaired what it safely could (a lone local head without
            # serving_slot). A retry meets the same cluster and no framework patch
            # changes it, so stop on the first one, like the AgentX preflight.
            # Waiting for a second would not be bounded: with no launch log to
            # author against, the enablement lane this failure opens never
            # schedules another baseline.
            log.error(
                "baseline %s failed because the Ray cluster cannot place it; stopping the run: %s",
                task.task_id,
                result_payload.get("error") or err_class,
            )
            return True
        return not retries_suppressed and (streak >= 3 or (streak >= 2 and repeated))

    def _arms_baseline_failure_gates(self, err_class: str) -> bool:
        """Whether a failed baseline goes through the streak and stop gates.

        Only while no baseline has succeeded yet, except a cache-off refusal:
        a re-baseline refused for it stops the run as the first baseline does,
        since the next round would launch the same server.
        """
        return self.shared_state.baseline_tput <= 0 or err_class == PREFIX_CACHE_OFF_ERROR_CLASS

    async def handle_unpromotable_result(
        self,
        task: Task,
        result: dict[str, Any] | None,
    ) -> None:
        """Record a failed / unpromotable task result into SharedState: append to last_action_failures (+ a failed attempts row for _AUDIT_ACTIONS) and apply the baseline failure_streak/stop_reason gates.

        Args:
            task: The failed/unpromotable task.
            result: The task result payload; ``None`` is treated as an empty
                result.
        """
        result_payload = dict(result or {})
        if task.kind == "conc_sweep" and not result_payload.get("status"):
            result_payload["status"] = "failed"
        if task.kind == "explore" and bool((task.params or {}).get("geak_fallback")):
            self._close_failed_geak_revalidation(result_payload)
        # Per-action audit (failed attempt) for the in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp baseline-params fingerprint for the self-loop denial helper.
            if task.kind == "baseline":
                audit_extras["fingerprint"] = self.baseline_params_fingerprint(task.params)
            self.shared_state.record_action_attempt(
                action=task.kind,
                task_id=task.task_id,
                status="failed",
                decision="no_promote",
                result=result_payload,
                extras=audit_extras,
            )
        # Global rolling failure log (every kind, including kernel_agent-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        if task.kind == "conc_sweep":
            # No audit attempt here: ``record_action_attempt`` returns early for
            # any kind outside ``_AUDIT_ACTIONS``, which conc_sweep is not in.
            # The conc_sweep event carries these facts instead.
            self.shared_state.record_conc_sweep(result_payload)
        # An upstream-PR candidate task that settles failed/empty never reaches
        # the promote branch that writes the terminal progress row; stamp
        # no_result_failed so the pump does not re-select it every tick.
        self._coord.phase_framework.record_unpromoted_candidate(task, result_payload)
        baseline_event_payload: dict[str, Any] | None = None
        err_class = result_payload.get("error_class", "")
        # Only arm the streaks and stop gates while no baseline has succeeded yet.
        if task.kind == "baseline" and self._arms_baseline_failure_gates(err_class):
            baseline_event_payload = await self._charge_baseline_failure(task, result_payload, err_class)
        # Mirror the promote-path roofline failure handling: bump streak, clear gate, warn.
        if task.kind == "roofline":
            self.shared_state.roofline_failure_streak += 1
            if self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
            log.warning(
                "Auto-roofline %s failed (reason=%s phase=%s "
                "error_class=%s); continuing in degraded mode "
                "(specialists / explore proceed without a fresh "
                "analysis_md). No retry, no fallback.",
                task.task_id,
                str((task.params or {}).get("reason") or ""),
                result_payload.get("phase"),
                result_payload.get("error_class"),
            )
        elif task.kind == "profile" and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            # With enable_roofline off the initial analysis is a profile; leaving its id set would hold PRELUDE.
            self.shared_state.auto_roofline_pending_task_id = ""
        self.shared_state.save(self.session_dir)
        if baseline_event_payload is not None:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "event",
                    baseline_event_payload,
                )
            )

    def _tracks_open_revalidation(self, task: Task) -> bool:
        """Whether this baseline is the enablement revalidation the lane is still waiting on."""
        enablement = self.shared_state.enablement
        if not bool(getattr(enablement, "validation_pending", False)):
            return False
        if (task.params or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON:
            return True
        tracked = str(getattr(enablement, "revalidation_task_id", "") or "").strip()
        return bool(tracked) and tracked == str(task.task_id or "")

    async def _charge_baseline_failure(
        self,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
    ) -> dict[str, Any]:
        """Charge a failed baseline to the streaks and the stop gates; the event it reports.

        A round the run itself stopped -- the session budget reaped it, or the
        orchestrator cancelled the action -- measured nothing, so it says
        nothing about whether this baseline boots. Charging it to the streaks
        would let three stops the run chose end the session as
        ``baseline_failed``, blaming the model for the clock. The executor
        already refuses to grade such a round; the ledger has to agree.
        """
        from ..actions.executors._accuracy_gate import eval_enablement_allowed
        from ..actions.executors._multi_node_env import is_multi_node

        stopped_by_the_run = stopped_by_the_run_class(err_class) is not None
        eval_failed = bool(result_payload.get(BASELINE_EVAL_FAILED_KEY))
        if self._tracks_open_revalidation(task):
            await self._record_revalidation_not_promoted(
                task=task,
                result_payload=result_payload,
                err_class=err_class,
                stopped_by_the_run=stopped_by_the_run,
            )
        # Single-node eval-pending failure: throughput measured fine and the eval
        # is expected to re-run under enablement, so do not spend the
        # baseline_failed budget yet. Multi-node keeps the strict backstop, and
        # so does a session that never admitted the eval lane -- nothing would
        # re-run the eval, so holding the budget just stalls the run. Baseline
        # failures are also the ENABLEMENT phase's own input, and its attempt cap
        # bounds them.
        eval_pending_suppress = eval_failed and not is_multi_node() and eval_enablement_allowed(self.shared_state)
        in_enablement = str(self.shared_state.phase or "").strip().upper() == PHASE_ENABLEMENT
        retries_suppressed = bool(eval_pending_suppress or in_enablement)
        if eval_failed:
            self._persist_eval_failure(result_payload)
        self._stop_the_run_on_baseline_failure(task, result_payload, err_class, retries_suppressed=retries_suppressed)
        # Combined backstop: count ALL baseline failures so mixed error_classes
        # that split the per-class streaks still fast-fail.
        if not stopped_by_the_run:
            self.shared_state.baseline_total_failures += 1
        if (
            self.shared_state.baseline_total_failures >= _BASELINE_MAX_TOTAL_FAILURES
            and not self.shared_state.stop_reason
            and not retries_suppressed
        ):
            self.shared_state.set_stop_reason(_BASELINE_FAILED_STOP_REASON)
        self._arm_eager_baseline_fallback(task, err_class)
        self._stash_enablement_launch_log(result_payload, err_class)
        return {
            "kind": "baseline_not_promoted",
            "task_id": task.task_id,
            "failure_streak": self.shared_state.baseline_failure_streak,
            "arg_error_streak": self.shared_state.baseline_arg_error_streak,
            "stop_reason": self.shared_state.stop_reason,
            "result_status": result_payload.get("status"),
            "error_class": err_class,
            "baseline_eval_failed": eval_failed,
        }

    def _stop_the_run_on_baseline_failure(
        self,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
        *,
        retries_suppressed: bool,
    ) -> None:
        """Set the stop reason a failed baseline of this class earns, if it earns one.

        A fast arg error gets its own streak so deterministic failures do not
        burn the slow-baseline retry budget.
        """
        if stopped_by_the_run_class(err_class) is not None:
            log.warning(
                "baseline %s was stopped by the run (%s); the failure streak stays at %d "
                "because nothing about the baseline was measured",
                task.task_id,
                err_class,
                self.shared_state.baseline_failure_streak,
            )
        elif err_class in _STOPS_ON_FIRST_OCCURRENCE:
            self.shared_state.set_stop_reason(_STOPS_ON_FIRST_OCCURRENCE[err_class])
        elif err_class == "fast_exit_arg_error":
            self.shared_state.baseline_arg_error_streak += 1
            if self.shared_state.baseline_arg_error_streak >= 2:
                self.shared_state.set_stop_reason("baseline_arg_error")
        elif err_class == AGENTX_PREFLIGHT_ERROR_CLASS:
            # The delegated client must pass its environment preflight before framework optimization.
            log.error(
                "baseline %s failed the agentic client preflight; setup must "
                "resolve it before resuming. This is an environment gap, not something a "
                "framework patch can close: %s",
                task.task_id,
                result_payload.get("error") or err_class,
            )
            self.shared_state.set_stop_reason(AGENTX_PREFLIGHT_STOP_REASON)
        # ``baseline_arg_error_streak`` is deliberately left alone on every other
        # class: a boot that failed some other way is no evidence the arguments
        # were fixed.
        elif self._generic_baseline_failure_stops(
            task, err_class, result_payload, retries_suppressed=retries_suppressed
        ):
            self.shared_state.set_stop_reason(_BASELINE_FAILED_STOP_REASON)

    def _arm_eager_baseline_fallback(self, task: Task, err_class: str) -> None:
        """Arm the one-shot disable-cuda-graph retry a capture failure is often recoverable by."""
        if err_class != "cuda_graph_capture_failed" or self.shared_state.baseline_eager_fallback:
            return
        self.shared_state.baseline_eager_fallback = True
        log.warning(
            "baseline %s hit cuda-graph capture failure; arming "
            "disable-cuda-graph fallback for the next baseline retry",
            task.task_id,
        )

    def _stash_enablement_launch_log(self, result_payload: dict[str, Any], err_class: str) -> None:
        """Stash a failed boot's launch text for the FRAMEWORK pump to author against.

        Only the first trigger is kept, so a serial enablement does not rewrite
        the reason the lane opened with the gap it is now on.
        """
        launch_log = _authorable_launch_log(result_payload, err_class=err_class)
        if not launch_log:
            return
        self.shared_state.enablement.record_launch_observation(
            log=launch_log, observation_path=str(result_payload.get("boot_observation_path") or "")
        )
        if str(self.shared_state.enablement.origin or ""):
            return
        enablement_event.record_trigger(
            origin=enablement_event.ORIGIN_BOOT,
            mode=str(self.shared_state.enablement_mode or ""),
            kind=str(err_class or ""),
            evidence=launch_log,
        )

    def lift_to_current_best(
        self,
        task_kind: str,
        best_tput: float,
        bv: dict[str, Any],
        *,
        gap_canonical_id: str = "",
        entry_extra: Mapping[str, Any] | None = None,
    ) -> bool:
        """Accept prebaseline enablement config or lift an improving measured winner.

        The only writer of ``optimization_stack``, and the only config-KEEP
        writer of ``current_best`` (the baseline anchor is the other one). Also
        the only place a winner is merged onto the previous config instead of
        replacing it, so ``unset_envs`` and cumulative args stay correct across
        the whole stack.

        Args:
            task_kind: The action kind that produced the winner (stamped on the
                stack entry / current_best).
            best_tput: The winning variant's measured throughput.
            bv: The winning variant dict (args, envs, metrics, provenance).
            gap_canonical_id: When known, stamped onto the stack entry so
                provenance resolves by gap id rather than name.
            entry_extra: Per-action metadata for the stack entry only;
                ``current_best`` stays a pure config record.

        Returns:
            ``True`` when the configuration was accepted, ``False`` when a
            performance winner was not comparable or did not beat the anchor.
        """
        cand_source = _graded_source(bv, best_tput)
        graded = (
            None
            if _is_prebaseline_enablement(task_kind, bv, self.shared_state.baseline_tput)
            else resolve_graded_comparison(self.shared_state, cand_source)
        )
        if graded is not None and self._refuses_graded_winner(task_kind, bv, graded):
            return False
        previous = self.shared_state.current_best
        launch = _resolved_launch(previous, bv)
        request = _LiftRequest(
            task_kind=task_kind,
            tput=float(best_tput),
            variant=bv,
            gap_canonical_id=gap_canonical_id,
            entry_extra=entry_extra or {},
        )
        if launch.candidate_args or launch.variant_name:
            self._append_stack_entry(request, launch, graded)
        self._write_current_best(request, launch, cand_source, previous)
        return True

    def _refuses_graded_winner(self, task_kind: str, bv: Mapping[str, Any], graded: "GradedComparison") -> bool:
        """Whether grading turns this winner down, logging the reason it was refused.

        AgentX sessions fail closed when the interactivity axis could not apply:
        the output figure on a degraded pair is diagnostic only, and promoting
        on it would record a throughput KEEP the session never asked for. A
        verified correctness fix was already bounded by the integrate gate.
        """
        from hyperloom.common.perf_metric import VERDICT_KEEP

        if not graded.comparable:
            log.info("current_best held: %s winner not comparable (%s)", task_kind, graded.degrade_reason)
            return True
        if graded.veto_reason:
            log.warning(
                "current_best held: %s winner refused by %s (%s)", task_kind, graded.vetoed_by, graded.veto_reason
            )
            return True
        correctness_fix = bv.get("keep_reason") == KEEP_REASON_CORRECTNESS_FIX
        if graded.graded_on_intvty and graded.verdict != VERDICT_KEEP and not correctness_fix:
            log.info(
                "current_best held: %s winner refused by %s, intvty %.1f->%.1f tput %.1f->%.1f",
                task_kind,
                graded.refused_by,
                graded.reference,
                graded.candidate,
                graded.tput_reference,
                graded.tput_candidate,
            )
            return True
        if graded.reference > 0 and graded.candidate <= graded.reference and not correctness_fix:
            log.info(
                "current_best held at %.1f %s: %s winner measured %.1f (no lift)",
                graded.reference,
                graded.objective,
                task_kind,
                graded.candidate,
            )
            return True
        return False

    def _append_stack_entry(
        self,
        request: _LiftRequest,
        launch: _ResolvedLaunch,
        graded: "GradedComparison | None",
    ) -> None:
        """Record the winner on the optimization stack, unless it is already applied.

        ``graded`` is ``None`` for a pre-baseline enablement patch: it was never
        graded against an anchor, so it has no adoption to claim.
        """
        if _already_stacked(self.shared_state.optimization_stack, request, launch.variant_name):
            return
        stack_entry = self._stack_entry(request, launch)
        self.shared_state.optimization_stack.append(stack_entry)
        # Mirror the append into gain_per_stack_entry so the two lists stay index-aligned.
        self.shared_state.append_stack_gain_entry(
            action=request.task_kind,
            variant_name=launch.variant_name,
            new_tput=request.tput,
            extra_server_args=launch.full_args,
        )
        if graded is None:
            return
        # ``graded.reference`` is the anchor this winner had to beat, and this
        # call is the last moment it exists: the current_best write that follows
        # overwrites it.
        from hyperloom.inference_optimizer.breakdown.recorder import stack_event

        stack_event.record_adoption(
            stack_index=len(self.shared_state.optimization_stack) - 1,
            entry=stack_entry,
            throughput_before=graded.reference,
            throughput_after=graded.candidate,
            baseline_tput=self.shared_state.baseline_tput,
            objective=graded.objective,
            degrade_reason=graded.degrade_reason,
        )

    def _stack_entry(self, request: _LiftRequest, launch: _ResolvedLaunch) -> dict[str, Any]:
        """The stack row a lifted winner is recorded as."""
        bv = request.variant
        entry: dict[str, Any] = {
            "stack_entry_id": uuid.uuid4().hex,
            "action": request.task_kind,
            "variant_name": launch.variant_name,
            "candidate_extra_server_args": launch.candidate_args,
            "candidate_extra_envs": dict(bv.get("candidate_extra_envs") or {}),
            "extra_server_args": launch.full_args,
            "extra_envs": dict(bv.get("extra_envs") or {}),
            # Carry the promoting lane's accuracy verdict onto the stack so CLOSE
            # reads one place instead of reconstructing which lane promoted the
            # champion. ``None`` means "not gated".
            "accuracy": bv.get("accuracy"),
            "tput": request.tput,
            "workspace": bv.get("workspace"),
            "ts": datetime.now(timezone.utc).isoformat(),
            # Stable filter label for "what kind of optimization" (backend / param
            # / env), so the stack can be sliced like the timeline.
            "operation_kind": operation_kind_for(
                request.task_kind,
                classify_change_kind(
                    request.task_kind,
                    {"extra_server_args": launch.candidate_args, "extra_envs": dict(bv.get("extra_envs") or {})},
                ),
            ),
            **self._variant_authorship(request.task_kind, bv, launch.variant_name),
            **_variant_attribution(bv),
            **_variant_launch_controls(bv),
            **_variant_source_layer(bv),
        }
        fingerprint = _variant_fingerprint(bv, launch.candidate_args, launch.full_args)
        if fingerprint:
            entry["fingerprint"] = fingerprint
        if request.gap_canonical_id:
            entry["gap_canonical_id"] = request.gap_canonical_id
        if launch.overlay:
            entry["final_overlay"] = launch.overlay
        for key, value in request.entry_extra.items():
            if value not in (None, "", [], {}):
                entry[str(key)] = value
        return entry

    def _variant_authorship(self, task_kind: str, bv: Mapping[str, Any], variant_name: Any) -> dict[str, Any]:
        """Which phase and which lever a winner is credited to.

        The phase fallback inherits whatever is live at writeback time, which is
        not what authored the winner.
        """
        fields: dict[str, Any] = {}
        source_phase = str(
            bv.get("source_phase") or (self.shared_state.phase if task_kind != "integrate_patch" else "") or ""
        ).strip()
        if source_phase:
            fields["source_phase"] = source_phase
        lever_kind = lever_kind_for_task(task_kind, bv)
        if lever_kind:
            fields["lever_kind"] = lever_kind
        else:
            log.warning(
                "lift: no lever_kind for a %s winner (variant=%s); the stack entry will report as unattributed",
                task_kind,
                variant_name,
            )
        return fields

    def _write_current_best(
        self,
        request: _LiftRequest,
        launch: _ResolvedLaunch,
        cand_source: Mapping[str, Any],
        previous: Mapping[str, Any],
    ) -> None:
        """Make the lifted winner the session's ``current_best``.

        Envs start from the previous stack top so a source-layer KEEP, which
        applies none of its own, does not clear prior explore and env layers.
        """
        from hyperloom.common.perf_metric import constraint_fields, graded_axes_of

        bv = request.variant
        envs = dict(previous.get("extra_envs") or {})
        for key in to_str_list(bv.get("unset_envs")):
            envs.pop(key, None)
        envs.update(dict(bv.get("extra_envs") or {}))
        current_best: dict[str, Any] = {
            "action": request.task_kind,
            "tput": request.tput,
            "variant_name": launch.variant_name,
            "extra_server_args": launch.full_args,
            "extra_envs": envs,
            "final_overlay": launch.overlay,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": bv.get("ttft_mean_ms"),
            **constraint_fields(bv),
            "tpot_mean_ms": bv.get("tpot_mean_ms"),
            "workspace": bv.get("workspace"),
            # The axes of the measurement this KEEP was graded on. Without them
            # the next round's anchor has no snapshot, and the session degrades
            # to output grading permanently after the first KEEP.
            **graded_axes_of(cand_source),
        }
        for key in ("remove_args", "unset_envs", "args_mode"):
            if bv.get(key):
                current_best[key] = bv.get(key)
        if bv.get("effective_extra_server_args"):
            current_best["effective_extra_server_args"] = dedupe_extra_server_args(
                strip_benchmark_harness_flags(bv.get("effective_extra_server_args"))
            )
        if (bv.get("remove_args") or bv.get("unset_envs")) and not current_best.get("args_mode"):
            current_best["args_mode"] = "replace"
        self.shared_state.current_best = current_best
        self.shared_state.working_recipe_generation = int(self.shared_state.working_recipe_generation or 0) + 1
        self._stamp_current_best_measurement(bv)

    def _should_run_prelude_bootstrap(self, tput: Any) -> bool:
        """Whether to enqueue the post-baseline PRELUDE bootstrap chain.

        Returns ``False`` when there is no positive baseline throughput, when a
        roofline task is already pending, or when a stop is already pending
        (e.g. the baseline accuracy test produced no result ->
        ``baseline_accuracy_failed``). In the stop case the run is about to halt
        at the Coordinator's end-of-tick check, so no new bootstrap work (warm
        replay / roofline / scout / static recon) must be enqueued or dispatched
        in the meantime.

        Args:
            tput: The promoted baseline throughput.

        Returns:
            bool: True only when the bootstrap chain should run.
        """
        if not (isinstance(tput, (int, float)) and tput > 0):
            return False
        if (self.shared_state.auto_roofline_pending_task_id or "").strip():
            return False
        if (self.shared_state.stop_reason or "").strip():
            return False
        return True

    async def promote_to_shared_state(
        self,
        task_kind: str,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> _PromoteOutcome:
        """Lift specific action-result fields into the persistent SharedState (baseline/profile/roofline/grid).

        Args:
            task_kind: The settled task's kind, selecting the promote branch.
            result: The task result dict.
            task: The originating task, used for audit fingerprints and
                pending-roofline gating.

        Returns:
            The promotion outcome; its ``verdict`` is the settlement verdict.
        """
        # ``integrate_patch`` settles "apply_failed" / "reverted", which promote,
        # so a promoted result is still the only record of why a patch failed.
        error_class = str(result.get("error_class") or "").strip()
        if error_class and task is not None:
            self.shared_state.record_action_failure(
                action=task_kind,
                task_id=task.task_id,
                result=result,
            )
        # ``replay_warm_recipe`` is mirrored by _promote_replay_warm_recipe
        # instead: its executor settles on "succeeded" and the keep decision is
        # only reached further down this call, so mirroring it here published
        # every replay as discarded -- including the ones that went on to be
        # pushed onto the stack.
        handler_name = self._PROMOTE_HANDLERS.get(task_kind)
        if handler_name is None:
            return _PromoteOutcome(verdict=Verdict.RECORDED)
        outcome = _PromoteOutcome(verdict=Verdict.RECORDED)
        await getattr(self, handler_name)(result, task, outcome)
        # sweep / conc_sweep already recorded + saved + returned via their handler.
        if outcome.early_return:
            return outcome
        # Audit trail: one succeeded-attempt record with branch-supplied decision/extras.
        if outcome.audit_decision is not None and task_kind in _AUDIT_ACTIONS:
            self.shared_state.record_action_attempt(
                action=task_kind,
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision=outcome.audit_decision,
                result=result,
                extras=outcome.audit_extras,
            )
            outcome.changed = True
        if outcome.changed:
            self.shared_state.save(self.session_dir)
            self._drain_agent_keep_outbox()
        return outcome

    _PROMOTE_HANDLERS: dict[str, str] = {
        "baseline": "_promote_baseline",
        "replay_warm_recipe": "_promote_replay_warm_recipe",
        "profile": "_promote_profile",
        "roofline": "_promote_roofline",
        "explore": "_promote_explore",
        "integrate_patch": "_promote_integrate_patch",
        "conc_sweep": "_promote_conc_sweep",
    }

    def _anchor_held_reason(self, result: dict, *, is_revalidation: bool) -> str:
        """Why a baseline result does not replace the anchor, or ``""`` when it does.

        The anchor is the best measurement of the unmodified stack. A later, lower re-baseline must not replace it, or
        downstream gains end up measured against a different reference than the one they claim. A revalidation is
        exempt: the enablement patch changed the stack, so the prior anchor no longer describes anything reproducible.
        """
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and tput > 0):
            return "no_measurement"
        prior_anchor = float(self.shared_state.baseline_tput or 0.0)
        if prior_anchor <= 0.0 or is_revalidation:
            return ""
        # A stamped cumulative gain is measured against this anchor; moving it afterwards leaves every stamp
        # describing a reference that no longer exists, which the report then classifies as `inconsistent`.
        if self.shared_state.has_validated_gain_stamp:
            return "anchor_load_bearing"
        # A hot pass measured against a recorded COLD anchor is a correction, not a regression, so it lands even when
        # the number is lower. The two are not comparable -- that is what the cold marker says -- and a cold figure
        # can read higher than the hot one that replaces it whenever the "cold" pass was not really cold: weights in
        # page cache and a JIT cache a prior run populated leave it paying none of the startup its depressed
        # reputation assumes. Without this the session cannot escape the marker. PRELUDE refuses to finish while it
        # is set, the retry that would clear it is rejected for measuring lower, and the run re-measures whole
        # baseline rounds until the clock kills it.
        hot_pass_ran = result.get("measure_round_runtime_sec")
        corrects_a_cold_anchor = (
            bool(self.shared_state.baseline_measure_round_dropped)
            and isinstance(hot_pass_ran, (int, float))
            and hot_pass_ran > 0
        )
        if float(tput) > prior_anchor or corrects_a_cold_anchor:
            return ""
        return "below_anchor"

    async def _promote_baseline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a baseline result: anchor tput / accuracy / config and bootstrap PRELUDE."""
        from hyperloom.common.perf_metric import stamp_output_per_gpu

        stamp_output_per_gpu(result, self.shared_state.tp)
        changed = False
        tput = result.get("output_throughput")
        measured_ok = isinstance(tput, (int, float)) and tput > 0
        promoting_tid = str(getattr(task, "task_id", "") or "").strip()
        task_params = getattr(task, "params", None) or {}
        tracked_tid = str(getattr(self.shared_state.enablement, "revalidation_task_id", "") or "").strip()
        is_revalidation = bool(
            task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON
            or (tracked_tid and tracked_tid == promoting_tid)
        )
        prior_anchor = float(self.shared_state.baseline_tput or 0.0)
        held_reason = self._anchor_held_reason(result, is_revalidation=is_revalidation)
        anchor_accepted = not held_reason
        if measured_ok:
            self._anchor_baseline_tput(
                result, held_reason=held_reason, prior_anchor=prior_anchor, promoting_tid=promoting_tid
            )
            self.shared_state.baseline_failure_streak = 0
            # A genuine baseline is what revalidates an enablement KEEP.
            await self._settle_pending_revalidation(
                result, is_revalidation=is_revalidation, promoting_tid=promoting_tid
            )
            changed = True
        if anchor_accepted:
            if self._promote_anchor_run_facts(result, task_params):
                changed = True
            # Once the stack carries a validated layer, current_best belongs to
            # the stack top and a baseline must not reset it to the bare
            # reference config.
            if not (self.shared_state.optimization_stack or []):
                self._anchor_current_best(result)
                changed = True
            outcome.verdict = Verdict.ADOPTED
            outcome.audit_decision = "promoted"
        elif measured_ok:
            outcome.verdict = Verdict.REVERTED
            outcome.audit_decision = "no_promote"
        else:
            outcome.verdict = Verdict.FAILED
            outcome.audit_decision = "discarded"
        audit_extras = self._baseline_audit_extras(result, task_params, held_reason, prior_anchor)
        # seed the gaps[] ledger from baseline.
        await self._coord.gap_refresh.refresh_gaps(
            reason="baseline_done", workload_id=self._coord.recipe_journal.workload_canonical_id()
        )
        if self.shared_state.baseline_tput > 0:
            await self.drain_queued_baselines(reason="baseline_established")
        # Standalone baseline-arm roofline ceiling (pure CPU): backs up the
        # snapshot ceiling in case the later roofline step fails. Abstains under
        # AgentX, where the ceiling is derived from the inert ISL/OSL and the
        # conc-sweep chart already refuses to draw it for the same reason.
        from hyperloom.orchestrator.actions.executors._workload_envs import agentx_active

        if measured_ok and not agentx_active(self.shared_state):
            self.shared_state.record_baseline_roofline_ceiling()
        # Only the run that first establishes the anchor bootstraps; a later
        # re-baseline must not re-fire replay, scout or recon.
        if prior_anchor <= 0.0 and self._should_run_prelude_bootstrap(tput):
            await self._bootstrap_prelude_after_baseline(tput)
        outcome.changed = changed
        outcome.audit_extras = audit_extras

    def _anchor_baseline_tput(
        self,
        result: dict,
        *,
        held_reason: str,
        prior_anchor: float,
        promoting_tid: str,
    ) -> None:
        """Make this measured baseline the session's anchor, unless the anchor is held.

        The anchor is the hot measure round; the cold warmup round is discarded
        so gain math never mixes a cold "before" with a hot "after".
        """
        measured = float(result.get("output_throughput"))
        if held_reason:
            log.info(
                "baseline anchor: keeping %.1f; re-baseline measured %.1f (task=%s reason=%s)",
                prior_anchor,
                measured,
                promoting_tid,
                held_reason,
            )
            return
        self.shared_state.baseline_tput = measured

    async def _settle_pending_revalidation(
        self,
        result: dict,
        *,
        is_revalidation: bool,
        promoting_tid: str,
    ) -> None:
        """Close an open enablement revalidation window on this promoted baseline.

        An unrelated baseline promoting while the window is open only anchors
        the throughput: it must not consume or clear the pending state.
        """
        enablement = self.shared_state.enablement
        tracked_tid = str(getattr(enablement, "revalidation_task_id", "") or "").strip()
        if not bool(getattr(enablement, "validation_pending", False)):
            enablement.origin = ""
            enablement.pending = False
            return
        if not is_revalidation:
            log.info(
                "enablement revalidation pending: unrelated baseline promoted "
                "(task=%s tracked=%s); not consuming pending state",
                promoting_tid,
                tracked_tid,
            )
            enablement.origin = ""
            enablement.pending = False
            return
        accuracy = result.get("accuracy")
        floor = float(getattr(enablement, "accuracy_floor", 0.0) or 0.0)
        generation = int(getattr(enablement, "revalidation_generation", 0) or 0)
        if accuracy_meets_floor(accuracy, floor) or bool(self.shared_state.eval_disabled):
            await self._close_enablement_lane(generation, accuracy, floor, promoting_tid)
            return
        await self._rearm_enablement_after_subfloor(generation, accuracy, floor, promoting_tid)

    async def _close_enablement_lane(
        self,
        generation: int,
        accuracy: Any,
        floor: float,
        promoting_tid: str,
    ) -> None:
        """Close the enablement lane on a revalidation that met its accuracy floor.

        This promote is the lane's terminal: the KEEP that preceded it was
        provisional, so the round that landed it did not close the lane.
        """
        enablement = self.shared_state.enablement
        enablement.succeeded = True
        enablement.validation_pending = False
        enablement.revalidation_task_id = ""
        enablement.origin = ""
        enablement.pending = False
        await self._coord.enablement_lane.settle_enablement_round(BOOTED, reason="revalidation_promoted")
        enablement_event.record_revalidation_outcome(
            generation=generation,
            promoted=True,
            task_id=str(promoting_tid or ""),
            accuracy=accuracy,
            accuracy_floor=floor,
        )
        await self._coord.enablement_lane.close_lane_event(
            outcome=enablement_event.OUTCOME_SUCCEEDED,
            reason="revalidation promoted",
        )

    async def _rearm_enablement_after_subfloor(
        self,
        generation: int,
        accuracy: Any,
        floor: float,
        promoting_tid: str,
    ) -> None:
        """Rearm the specialist loop after a revalidation measured below its floor.

        The frozen trigger identity is left alone: the lane still owes the
        accuracy it opened on.
        """
        log.warning(
            "enablement revalidation: accuracy %.4f below floor %.4f; rearming",
            accuracy if isinstance(accuracy, (int, float)) else float("nan"),
            floor,
        )
        enablement = self.shared_state.enablement
        enablement.validation_pending = False
        enablement.revalidation_task_id = ""
        await self._coord.enablement_lane.settle_enablement_round(FAILED, reason="revalidation_below_floor")
        enablement_event.record_revalidation_outcome(
            generation=generation,
            promoted=False,
            task_id=str(promoting_tid or ""),
            accuracy=accuracy,
            accuracy_floor=floor,
            reason="accuracy below floor",
        )

    def _promote_anchor_run_facts(self, result: dict, task_params: Mapping[str, Any]) -> bool:
        """Promote the accuracy, config and wall-clock the anchor run measured; whether any moved.

        All of them describe the anchor run, so they only move when the anchor
        itself does; otherwise the recorded reference tput and the config it was
        measured with drift apart.
        """
        from hyperloom.common.perf_metric import perf_snapshot_from_mapping

        state = self.shared_state
        state.baseline_benchmark_script = str(task_params.get("benchmark_script") or "").strip()
        state.baseline_perf = perf_snapshot_from_mapping(result) or {}
        changed = False
        accuracy = result.get("accuracy")
        if isinstance(accuracy, (int, float)):
            state.baseline_accuracy = float(accuracy)
            changed = True
        # Persist the materialized YAML so downstream tasks reuse the exact workload contract.
        materialized = result.get("materialized_config")
        if isinstance(materialized, str) and materialized:
            state.baseline_config_path = materialized
            changed = True
            # Workload-shape extras off the YAML feed the lesson and pitfall attrs.
            parsed = _parse_baseline_workload_extra(materialized)
            if parsed:
                state.baseline_workload_extra = parsed
        # ExploreExecutor derives the overtime kill deadline from this wall-clock.
        runtime_sec = result.get("subprocess_runtime_sec")
        if isinstance(runtime_sec, (int, float)) and runtime_sec > 0:
            state.baseline_runtime_sec = float(runtime_sec)
            changed = True
        # Whether this baseline had to keep its cold figure because the budget
        # could not fund a hot pass and a variant to use it. Carried on the
        # session because the decision it drives is the session's: PRELUDE routes
        # to CLOSE rather than optimizing against a denominator that was never
        # the baseline.
        measure_round_dropped = bool(result.get("measure_round_dropped"))
        if measure_round_dropped != bool(state.baseline_measure_round_dropped):
            state.baseline_measure_round_dropped = measure_round_dropped
            changed = True
        if self._promote_anchor_round_split(result):
            changed = True
        return changed

    def _promote_anchor_round_split(self, result: dict) -> bool:
        """Promote the anchor's warm and boot wall-clock splits; whether either moved.

        Both are cleared by a later baseline that carries neither, so a stale
        split can never be subtracted from a fresh total and reported as this
        workload's boot, and a resumed session with a fresh clock is not held to
        an earlier leg's shortfall.
        """
        state = self.shared_state
        changed = False
        # Present only on the double-run baseline path; else explore uses the cold anchor.
        warm_runtime = result.get("measure_round_runtime_sec")
        if isinstance(warm_runtime, (int, float)) and warm_runtime > 0:
            state.baseline_warm_runtime_sec = float(warm_runtime)
            changed = True
        elif float(state.baseline_warm_runtime_sec or 0.0) != 0.0:
            state.baseline_warm_runtime_sec = 0.0
            changed = True
        post_ready = result.get("post_ready_runtime_sec")
        if isinstance(post_ready, (int, float)) and post_ready > 0:
            state.baseline_post_ready_runtime_sec = float(post_ready)
            changed = True
        elif float(state.baseline_post_ready_runtime_sec or 0.0) != 0.0:
            state.baseline_post_ready_runtime_sec = 0.0
            changed = True
        return changed

    def _anchor_current_best(self, result: dict) -> None:
        """Make the anchor run the session's ``current_best``.

        ``current_best.tput`` follows the same hot baseline contract as the
        anchor, so the gain numerator and denominator stay aligned.
        """
        from hyperloom.common.perf_metric import constraint_fields

        tput = result.get("output_throughput")
        warmup_anchor = result.get("warmup_round_tput")
        measured = float(tput) if isinstance(tput, (int, float)) else None
        anchor_tput = float(self.shared_state.baseline_tput or 0.0)
        current_best = {
            "action": "baseline",
            "tput": anchor_tput if anchor_tput > 0 else measured,
            "hot_tput": measured,
            "cold_tput": (
                float(warmup_anchor) if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0 else None
            ),
            "ttft_mean_ms": result.get("ttft_mean_ms"),
            **constraint_fields(result),
            "tpot_mean_ms": result.get("tpot_mean_ms"),
            "input_throughput": result.get("input_throughput"),
            "total_throughput": result.get("total_token_throughput"),
            "tpot_p90_ms": result.get("tpot_p90_ms"),
            "e2e_norm_intvty_p90": result.get("e2e_norm_intvty_p90"),
            "e2e_norm_intvty_p50": result.get("e2e_norm_intvty_p50"),
            "duration_seconds": result.get("duration_seconds"),
            "request_error_rate": result.get("request_error_rate"),
            "output_tput_per_gpu": result.get("output_tput_per_gpu"),
            "workspace": result.get("workspace"),
            **_baseline_perf_axes(self.shared_state.baseline_perf),
        }
        # The measured corpus shape replaces the canonical seed. Only an aiperf
        # result carries the distributions, so their presence is what marks the
        # measurement as AgentX-produced.
        if result.get("isl_distribution"):
            from hyperloom.inference_optimizer.agentx.mapping import map_corpus_shape

            self.shared_state.agentx_corpus_shape = map_corpus_shape(result)
        self.shared_state.current_best = current_best
        # Reads the current_best just assigned, so it has to follow it.
        self._stamp_current_best_measurement(result)
        from hyperloom.orchestrator.measurement.baseline_budget import baseline_budget_stop_reason

        if stop_reason := baseline_budget_stop_reason(self.shared_state, result):
            self.shared_state.set_stop_reason(stop_reason)

    def _baseline_audit_extras(
        self,
        result: dict,
        task_params: Mapping[str, Any],
        held_reason: str,
        prior_anchor: float,
    ) -> dict[str, Any]:
        """What a settled baseline's audit row records about it."""
        tput = result.get("output_throughput")
        measured = float(tput) if isinstance(tput, (int, float)) else None
        extras: dict[str, Any] = {
            "materialized_config": result.get("materialized_config"),
            "accuracy": result.get("accuracy"),
            "baseline_tput": measured,
            # Stamp the canonical params fingerprint for the self-loop denial helper.
            "fingerprint": self.baseline_params_fingerprint(task_params),
            "is_revalidation": bool(task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON),
            "enablement_succeeded": bool(getattr(self.shared_state.enablement, "succeeded", False)),
            "enablement_accuracy_floor": float(getattr(self.shared_state.enablement, "accuracy_floor", 0.0) or 0.0),
        }
        if held_reason and measured is not None and measured > 0:
            extras["anchor_kept_tput"] = prior_anchor
            extras["anchor_held"] = {"reason": held_reason, "prior": prior_anchor, "measured": measured}
        # Present only when the probe cut a runaway eval short; explains a ~0 accuracy.
        if result.get("eval_probe"):
            extras["eval_probe"] = result["eval_probe"]
        return extras

    async def _bootstrap_prelude_after_baseline(self, tput: Any) -> None:
        """Fire the post-baseline PRELUDE chain, in the order it has to run in.

        History injection, then warm replay, then auto-analysis, then the
        research scout and the static recon.
        """
        # History injection fires regardless of --no-warm-replay.
        self._coord.phase_prelude.inject_warm_recipe_history_into_ledger()
        # Warm-recipe replay, anchored on the hot baseline_tput contract.
        await self._coord.phase_prelude.maybe_enqueue_warm_replay(
            baseline_tput=float(self.shared_state.baseline_tput or tput),
        )
        # Auto-analysis (roofline / profile); may defer.
        await self._coord.phase_prelude.maybe_enqueue_prelude_initial_analysis_after_baseline(
            baseline_tput=float(tput),
        )
        # Research scout and static recon: parallel, read-only, CPU-only. Recon
        # seeds bridge candidates as gaps[] before the optimisation phase starts.
        await self._coord.phase_internal.maybe_enqueue_prelude_research_scout()
        await self._coord.phase_internal.maybe_enqueue_prelude_static_recon()

    async def drain_queued_baselines(self, *, reason: str) -> list[str]:
        """Cancel redundant queued baselines, preserving enablement revalidation.

        Args:
            reason: Stamped onto the cancellation history and the observation.

        Returns:
            The cancelled task ids (empty when the queue held none).
        """
        spared = await self._enablement_revalidation_task_ids()
        try:
            cancelled = await self.tasks.cancel_family(
                ["baseline"],
                reason=reason,
                exclude_task_ids=spared,
            )
        except Exception:
            log.exception("baseline drain: cancel_family failed")
            return []
        if not cancelled:
            return []
        log.info(
            "baseline drain: cancelled %d queued baseline task(s) (reason=%s, spared=%d)",
            len(cancelled),
            reason,
            len(spared),
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "baseline_drain",
                "reason": reason,
                "cancelled_task_ids": cancelled,
                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
            },
        )
        return cancelled

    async def _enablement_revalidation_task_ids(self) -> set[str]:
        """Return queued enablement-revalidation baseline task IDs."""
        spared: set[str] = set()
        tracked = str(getattr(self.shared_state.enablement, "revalidation_task_id", "") or "").strip()
        if tracked:
            spared.add(tracked)
        for task in await self.tasks.queued():
            if str(getattr(task, "kind", "") or "") != "baseline":
                continue
            if (getattr(task, "params", None) or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON:
                spared.add(str(getattr(task, "task_id", "") or ""))
        return {t for t in spared if t}

    async def _promote_replay_warm_recipe(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Separate promote path so replay doesn't overwrite baseline_tput/current_best."""
        outcome.verdict = self._coord.phase_prelude.promote_warm_replay(result, task=task)
        # PRELUDE initial roofline was deferred while replay ran.
        await self._coord.phase_prelude.maybe_enqueue_prelude_initial_analysis_after_baseline()

    def _record_failed_profile(self, trace_path: Any) -> None:
        """Mark the session's last profile as failed, keeping a trace the round did produce.

        A round that failed after capturing a usable trace still leaves
        ``trace_analyze`` something to read.
        """
        self.shared_state.last_profile_status = "failed"
        self.shared_state.last_profile_workload = {}
        if not trace_path:
            self.shared_state.last_profile_trace = ""
            self.shared_state.last_profile_launch_evidence_path = ""
        self.shared_state.last_profile_args = ""
        self.shared_state.last_profile_workload_action = ""

    def _adopt_profile_trace(self, result: dict, task: "Task | None", trace_path: Any) -> str:
        """Make this trace the session's last profile, and invalidate the stale analysis.

        Returns:
            The server args the trace was taken with.
        """
        self.shared_state.last_profile_trace = str(trace_path)
        self.shared_state.last_profile_launch_evidence_path = str(result.get("launch_evidence_path") or "")
        self.shared_state.last_profile_status = "succeeded"
        profile_args = ""
        if task is not None:
            # Tag the workload with the arm it measured, so a later same-arm check can trust it.
            task_params = task.params or {}
            profile_args = str(task_params.get("base_extra_args") or "")
            self.shared_state.record_profile_workload(
                task_params,
                arm=("baseline" if str(task_params.get("reason") or "") == "prelude_initial" else ""),
            )
        else:
            self.shared_state.last_profile_workload = self.shared_state.current_profile_workload_context()
            self.shared_state.last_profile_workload_action = str(
                (self.shared_state.current_best or {}).get("action") or ""
            )
        self.shared_state.last_profile_args = str(
            self.shared_state.last_profile_workload.get("server_args") or profile_args
        )
        self.shared_state.last_trace_analyze = {}
        return profile_args

    async def _promote_profile(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a profile result: trace path / status, optional current_best, roofline anchor."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # Defensive skipped arm: audit as skipped + drop the gate.
        if str(result.get("status") or "") == "skipped":
            audit_decision = "skipped"
            audit_extras = {
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
        else:
            audit_decision = "promoted"
            audit_extras = {
                "trace_path": None,
                "profile_args": None,
                "output_throughput": result.get("output_throughput"),
            }
        trace_path = _profile_trace_path(result)
        profile_status = str(result.get("status") or "")
        measurement_status = str(result.get("measurement_status") or profile_status)
        audit_extras["measurement_status"] = measurement_status
        audit_extras["trace_capture_status"] = result.get("trace_capture_status")
        audit_extras["trace_capture_reason"] = (result.get("trace_capture") or {}).get("reason")
        if profile_status == "failed" or result.get("error_class") == "no_trace_files":
            outcome.verdict = Verdict.FAILED
            self._record_failed_profile(trace_path)
            changed = True
        elif trace_path:
            audit_extras["trace_path"] = str(trace_path)
            audit_extras["profile_args"] = self._adopt_profile_trace(result, task, trace_path)
            changed = True
        # Host-side rewrite evidence is independent of the trace: it answers
        # "which host-side work is redundant", which no kernel timeline can, and
        # a run whose trace was unusable can still have produced good evidence.
        # So promote it on its own, outside the trace_path branch.
        from ..actions.executors._framework_rewrite_evidence import promote_evidence_path

        evidence_path = promote_evidence_path(self.shared_state, result)
        if evidence_path:
            audit_extras["framework_rewrite_evidence"] = evidence_path
            audit_extras["framework_rewrite_candidate_count"] = result.get("framework_rewrite_candidate_count")
            changed = True
        # On a successful profile, re-anchor last_roofline_tput and clear the pending field.
        if measurement_status == "succeeded":
            anchor_tput = self._coord.phase_kernel.current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
                changed = True
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
            changed = True
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    def _anchor_watermark_on_roofline(self, result: dict, task: "Task | None") -> dict[str, Any]:
        """Re-anchor the watermark step on a succeeded roofline; describe its snapshot.

        The anchor is what stops the watermark firing again until throughput
        climbs another 10%, so a roofline that produced no analysis leaves it
        where it is: anchoring on an empty snapshot would buy a whole cycle of
        silence for one that says nothing.
        """
        # Prefer the executor's last_trace_analyze snapshot over the result dict.
        last_ta = self.shared_state.last_trace_analyze or {}
        self.shared_state.roofline_failure_streak = 0
        if str(last_ta.get("analysis_md_text") or ""):
            anchor_tput = self._coord.phase_kernel.current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
        else:
            log.warning(
                "roofline %s produced no analysis; leaving the watermark anchor at %.4f so a real one can still be taken",
                task.task_id if task else "?",
                float(self.shared_state.last_roofline_tput or 0.0),
            )
        snapshot_id = last_ta.get("roofline_snapshot_id")
        return {
            "snapshot_id": result.get("snapshot_id") if snapshot_id is None else snapshot_id,
            "last_profile_trace": (self.shared_state.last_profile_trace or result.get("last_profile_trace")),
            "analysis_md_path": (last_ta.get("analysis_md_path") or result.get("analysis_md_path")),
            "profile_workspace": result.get("profile_workspace"),
            "degraded": bool(result.get("degraded", False)),
        }

    def _charge_roofline_failure(self, result: dict, task: "Task | None") -> dict[str, Any]:
        """Charge a failed roofline to the streak; describe what it failed on.

        There is no retry and no fallback: specialists and explore go on without
        a fresh ``analysis_md``.
        """
        self.shared_state.roofline_failure_streak += 1
        log.warning(
            "Auto-roofline %s failed (reason=%s phase=%s "
            "error_class=%s); continuing in degraded mode "
            "(specialists / explore proceed without a fresh "
            "analysis_md). No retry, no fallback.",
            task.task_id if task else "?",
            str((task.params or {}).get("reason") or "") if task is not None else "",
            result.get("phase"),
            result.get("error_class"),
        )
        return {
            "phase": result.get("phase"),
            "error_class": result.get("error_class"),
            "error": result.get("error"),
        }

    async def _promote_roofline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a roofline result: audit + failure streak + roofline anchor (reads last_trace_analyze)."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # The composite roofline action runs profile + trace_analyze atomically;
        # its executor writes last_profile_* + last_trace_analyze, so here we just record the audit row.
        status = str(result.get("status") or "")
        if status == "skipped":
            # Defensive arm: clean no-op, no streak/watermark touch.
            audit_decision = "skipped"
            audit_extras = {
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
        elif status == "succeeded":
            audit_decision = "promoted"
            audit_extras = self._anchor_watermark_on_roofline(result, task)
            changed = True
        else:
            outcome.verdict = Verdict.FAILED
            audit_decision = "discarded"
            audit_extras = self._charge_roofline_failure(result, task)
            changed = True
        # Clear the pending pointer (matched by task id).
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
            changed = True
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_explore(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote an explore result: ledger increment, winners, current_best lift, resume revalidation."""
        changed = False
        # The Coordinator is single-writer for explore_search.accepted,
        # current_best and optimization_stack.
        update = result.get("explore_search_update")
        if isinstance(update, dict):
            self.shared_state.apply_explore_search_update(update)
            changed = True
        if self._record_framework_lever_attributions(result):
            changed = True
        winners = result.get("winners") or []
        # A stack revalidation confirms the existing stack and current_best
        # rather than adding a variant, so it never promotes: the accept and
        # lift path below is skipped, or a rebench would append a duplicate
        # stack entry and re-lift current_best.
        is_revalidation_task = task is not None and is_stack_revalidation(task.params)
        if is_revalidation_task:
            stale = await self._warn_on_stale_revalidation(task)
            # A GEAK revalidation (2b) rules on its own rebench and publishes the
            # verdict itself; a native one reconciles the watermark here.
            if bool((task.params or {}).get("geak_fallback")):
                await self._rule_on_geak_rebench(result, task=task, outcome=outcome)
                return
            await self._reconcile_revalidated_watermark(result, stale=stale)
            changed = True
        lift = _ExploreLift(promoted=False, last_lifted=None)
        if isinstance(winners, list) and winners and not is_revalidation_task:
            lift = self._accept_explore_winners(result, task, outcome)
            changed = True
        self.shared_state.note_explore_outcome(promoted=lift.promoted)
        # A round with no measured variant is not a data point for the plateau window.
        if not is_revalidation_task and (winners or result.get("losers")):
            self.shared_state.gain_gated_action_count += 1
        if lift.last_lifted is not None:
            # The last successful lift owns every axis of the validated measurement.
            await self._validate_and_watermark(
                float(lift.last_lifted["tput"]),
                lift.last_lifted,
                source="explore_keep",
                measurement_basis="e2e_decision_round",
                watermark_reason="explore_keep_watermark",
            )
        else:
            changed = True
        if lift.promoted:
            outcome.audit_decision = "promoted"
            outcome.verdict = Verdict.ADOPTED
        elif winners and not is_revalidation_task:
            outcome.audit_decision = "no_promote"
            outcome.verdict = Verdict.REFUSED
        else:
            outcome.audit_decision = "discarded"
            outcome.verdict = Verdict.RECORDED if is_revalidation_task else Verdict.REVERTED
        outcome.changed = changed
        outcome.audit_extras = _explore_audit_extras(result)

    def _record_framework_lever_attributions(self, result: Mapping[str, Any]) -> bool:
        """Record what each framework lever this round measured was worth; whether any landed.

        Recorded whether or not the lever's own variant won: a rewrite measured
        at +0.1% is as useful to know as one measured at +8%, and without the
        number the lever would be re-benched every round.
        """
        recorded = False
        for attribution in result.get("framework_lever_attributions") or []:
            if not isinstance(attribution, dict):
                continue
            if self.shared_state.record_framework_lever_attribution(
                str(attribution.get("switch") or ""),
                gain_pct=attribution.get("gain_pct"),
                source=str(attribution.get("source") or ""),
            ):
                recorded = True
        return recorded

    async def _warn_on_stale_revalidation(self, task: "Task") -> bool:
        """Whether a revalidation measured an older recipe than the one now live.

        A lift since the row was enqueued means the grid ran an older
        ``current_best``, and crediting it to the newer one is the mismatch
        CLOSE refuses to publish. Rows enqueued before generations existed carry
        no stamp.
        """
        measured_generation = (task.params or {}).get("recipe_generation")
        working_generation = int(self.shared_state.working_recipe_generation or 0)
        if measured_generation is None or int(measured_generation) == working_generation:
            return False
        log.warning(
            "stack revalidation %s measured recipe generation %s; current_best is now %s -- not validating",
            task.task_id,
            measured_generation,
            working_generation,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "stale_stack_revalidation",
                "task_id": task.task_id,
                "measured_recipe_generation": int(measured_generation),
                "working_recipe_generation": working_generation,
                "geak_fallback": bool((task.params or {}).get("geak_fallback")),
            },
        )
        return True

    async def _reconcile_revalidated_watermark(self, result: dict, *, stale: bool) -> None:
        """Move the validation watermark onto what a stack revalidation measured.

        A failed or empty rebench leaves the watermark where it is, so the round
        reports a keep warning rather than claiming a validation it has no
        measurement for.
        """
        measured = result.get("output_throughput")
        if stale or not isinstance(measured, (int, float)) or measured <= 0:
            return
        if self.shared_state.baseline_tput <= 0:
            return
        self.validate(measured, result, source=STACK_REVALIDATE_SOURCE)
        recorded = self.shared_state.current_best.get("tput")
        if not isinstance(recorded, (int, float)) or recorded <= 0:
            return
        floor = _DEFAULT_RESUME_DRIFT_FLOOR_PCT
        if float(measured) >= float(recorded) * floor / 100.0:
            return
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "current_best_drift",
                "severity": "high",
                "measured_tput": float(measured),
                "recorded_tput": float(recorded),
                "floor_pct": floor,
            },
        )

    def _record_accepted_winners(self, result: Mapping[str, Any], winners: list[dict[str, Any]]) -> None:
        """Enter this round's winners on the explore ledger."""
        round_id = str(result.get("round_id") or "")
        for winner in winners:
            accepted = dict(winner)
            accepted.setdefault("accepted_at_round", round_id)
            accepted.setdefault("provenance", winner.get("provenance") or "llm_direct")
            self.shared_state.record_explore_accepted(accepted)
            # A specialist-provenance KEEP zeroes that domain's rounds_since_last_keep counter.
            provenance = str(accepted.get("provenance") or "")
            if provenance.startswith("specialist:"):
                self.shared_state.note_domain_keep(provenance.split(":", 1)[1].strip())

    def _accept_explore_winners(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> _ExploreLift:
        """Enter this round's explore winners and lift the ones that still beat the anchor.

        Winners arrive already graded by the executor, on the decision round
        that judged them; the lift still refuses one that no longer beats the
        live anchor. They are lifted cumulatively, in application order and each
        on its own measurement, because the graded axis can improve even when
        output throughput falls.
        """
        winners = [winner for winner in result.get("winners") or [] if isinstance(winner, dict)]
        self._record_accepted_winners(result, winners)
        gap_canonical_id = str((task.params or {}).get("gap_canonical_id") or "").strip() if task is not None else ""
        promoted = False
        last_lifted: dict[str, Any] | None = None
        for winner in winners:
            winner_tput = winner.get("tput")
            if not isinstance(winner_tput, (int, float)) or float(winner_tput) <= 0:
                continue
            entry = dict(winner)
            if task is not None:
                entry["task_id"] = str(task.task_id or "")
            if not self.lift_to_current_best("explore", float(winner_tput), entry, gap_canonical_id=gap_canonical_id):
                continue
            promoted = True
            last_lifted = entry
            if winner.get("fingerprint"):
                outcome.adopted_variants.add(str(winner["fingerprint"]))
        return _ExploreLift(promoted=promoted, last_lifted=last_lifted)

    def _confirmed_integrate_completion(
        self, result: dict, task: "Task | None", *, kept_flag: bool
    ) -> dict[str, Any] | None:
        """Return the matching, completed marker without performing recovery."""
        pending = self.shared_state.pending_integrate
        task_id = str(getattr(task, "task_id", "") or "")
        matches = isinstance(pending, dict) and bool(task_id) and pending.get("task_id") == task_id
        failed = (
            result.get("error_class") == "integrate_restore_incomplete"
            or bool(result.get("recovery_errors"))
            or bool(result.get("stash_restore_error"))
        )
        if not matches and not failed:
            return None
        recovery = pending.get("recovery") if matches else None
        phase = str(recovery.get("phase") or "") if isinstance(recovery, dict) else ""
        status = str(result.get("status") or "")
        if not failed:
            if matches and "recovery" not in pending and kept_flag:
                return pending
            verdict = _integrate_marker_verdict(status, phase)
            if verdict in ("retained", "settled"):
                return pending
            if verdict == "inflight":
                return None
        self.shared_state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
        self.shared_state.save(self.session_dir)
        raise IntegrateRecoveryIncomplete(
            f"integrate recovery is incomplete or inconsistent for task={task_id!r}: "
            f"phase={phase!r}, status={status!r}, errors={result.get('recovery_errors')!r}, "
            f"stash_restore_error={result.get('stash_restore_error')!r}"
        )

    async def _promote_integrate_patch(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote only after recovery is confirmed; clear only its completed marker."""
        changed = self._record_authored_levers(result)
        stack_len_before = len(self.shared_state.optimization_stack or [])
        status = str(result.get("status") or "")
        measurement = result.get("bench_result") or result
        new_tput = measurement.get("output_throughput", result.get("output_throughput"))
        kept_flag = status == "kept" and isinstance(new_tput, (int, float)) and float(new_tput) > 0
        completed_pending = self._confirmed_integrate_completion(result, task, kept_flag=kept_flag)
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        enablement_landing = bool(
            result.get("enablement") or task_params.get("enablement") or task_params.get("enablement_landing")
        )
        lifted = False
        if kept_flag:
            lifted = await self._lift_integrate_keep(result, task, measurement, enablement_landing=enablement_landing)
            changed = True
        if completed_pending is not None and self.shared_state.pending_integrate is completed_pending:
            self.shared_state.pending_integrate = {}
            changed = True
        outcome.verdict = _integrate_verdict(status, lifted=lifted, kept=kept_flag)
        # A measured integrate counts either way; KEEP/REVERT is a later judgement.
        if new_tput is not None:
            self.shared_state.gain_gated_action_count += 1
        if (
            lifted
            and not enablement_landing
            and len(self.shared_state.optimization_stack or []) > stack_len_before
            and _is_patch_column_keep(task_params, result)
        ):
            self._enqueue_agent_keep_outbox(
                stack_index=len(self.shared_state.optimization_stack) - 1,
                result=result,
                task=task,
                include_patches=True,
            )
        outcome.changed = changed

    def _record_authored_levers(self, result: Mapping[str, Any]) -> bool:
        """Register this patch's framework-rewrite switches as search levers; whether any landed.

        Done for both KEEP verdicts: a bundle that cleared the gate is on and
        gets leave-one-out attribution, while an inert KEEP is dormant and gets
        additive attribution. ``kept_inert`` deliberately does not lift
        current_best -- the code is applied but every switch is off, so the
        running configuration is unchanged.
        """
        levers = result.get("framework_levers")
        if not (isinstance(levers, list) and levers):
            return False
        lever_outcome = str(result.get("framework_lever_outcome") or "")
        return bool(
            self.shared_state.record_authored_framework_levers(
                levers,
                default_on=(lever_outcome == "default_on"),
                specialist_task_id=str(result.get("specialist_task_id") or ""),
                stack_delta_pct=result.get("delta_pct"),
            )
        )

    async def _lift_integrate_keep(
        self,
        result: dict,
        task: "Task | None",
        measurement: Mapping[str, Any],
        *,
        enablement_landing: bool,
    ) -> bool:
        """Lift a settled integrate_patch KEEP onto current_best; whether it landed.

        A patch landing before any runnable baseline establishes the baseline
        environment. It stays in the configuration stack for reproducibility but
        is ineligible for gain attribution, because no runnable "before"
        measurement exists to attribute a gain against -- and so it claims no
        validated watermark either.
        """
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        new_tput = float(measurement.get("output_throughput", result.get("output_throughput")))
        variant = _integrate_keep_variant(result, task, measurement, new_tput)
        prebaseline = enablement_landing and float(self.shared_state.baseline_tput or 0.0) <= 0.0
        if enablement_landing:
            variant["recipe_publishable"] = False
        if prebaseline:
            variant["baseline_enablement"] = True
            variant["attribution_eligible"] = False
        lifted = self.lift_to_current_best(
            "integrate_patch",
            new_tput,
            variant,
            gap_canonical_id=str(task_params.get("gap_canonical_id") or result.get("gap_canonical_id") or "").strip(),
        )
        if prebaseline:
            log.info(
                "integrate_patch KEEP accepted as pre-baseline enablement; "
                "retained in config stack without gain attribution "
                "(task=%s specialist=%s)",
                str(getattr(task, "task_id", "") or ""),
                str(result.get("specialist_task_id") or task_params.get("specialist_task_id") or ""),
            )
        elif lifted:
            await self._validate_and_watermark(
                new_tput,
                measurement,
                source="integrate_patch_keep",
                watermark_reason="integrate_patch_keep_watermark",
            )
        return lifted

    async def _promote_conc_sweep(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a conc_sweep result: record_conc_sweep + save; discovery-only.

        No audit attempt is written: ``record_action_attempt`` returns early for
        any kind outside ``_AUDIT_ACTIONS``, which conc_sweep is not in. The
        conc_sweep event carries the skip reason, the budget verdict and the
        summary instead.
        """
        outcome.early_return = True
        # Write last_conc_sweep so the SWEEP exit logic can distinguish an honest sweep_done from a no-pair sweep_failed.
        self.shared_state.record_conc_sweep(result)
        self.shared_state.save(self.session_dir)

    # ------------------------------------------------------------------
    # Resume / replay (folded in from the former ResumeCollaborator).
    # Three semantic boundaries live below: the live-promote / replay path
    # (``_replay_keep_from_result``), the resume-reconcile path
    # (``_resume_consistency_pass`` + its recover helpers), and the
    # current_best lift path (``current_best_launch_config`` /
    # ``build_env_spec``).
    # ------------------------------------------------------------------
    def detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking).

        Returns:
            A dict with ``is_resume``, ``event_count``, ``state_json_present``
            and ``rebuilt`` (the last set later by :meth:`replay_for_resume`).
        """
        event_count = self.bus.count_sync()
        state_path = SharedState.state_path(self.session_dir)
        self._resumed_from = {
            "is_resume": event_count > 0 or state_path.exists(),
            "event_count": event_count,
            "state_json_present": state_path.exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }
        return self._resumed_from

    async def replay_for_resume(self) -> dict[str, Any]:
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``. Idempotent; a proposal is undecided when no review_verdict targets it.

        Returns:
            A dict summarising the replay: ``is_resume``, ``event_count``,
            ``state_json_present``, ``pending_restored`` (count rebuilt) and
            ``verdicts_seen``.
        """
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if not target:
                continue
            # Verdicts with a verdict_map but no summary are treated as needs_review.
            summary = v.payload.get("verdict") or ""
            if not summary and isinstance(v.payload.get("verdict_map"), dict):
                summary = "needs_review"
            verdict_by_target[target] = summary
            decided_ids.add(target)

        rebuilt = 0
        self.state.pending_proposals.clear()
        for p in proposal_msgs:
            if p.msg_id in decided_ids:
                continue
            payload = p.payload or {}
            self.state.pending_proposals[p.msg_id] = PendingProposal(
                proposal_msg_id=p.msg_id,
                from_agent=p.from_agent,
                action_name=str(payload.get("action_name", "")),
                predicted_gain_pct=float(payload.get("predicted_gain_pct", 0.0)),
                payload=dict(payload),
            )
            rebuilt += 1

        self._resumed_from["rebuilt"] = True
        self._resumed_from["pending_restored"] = rebuilt
        return {
            "is_resume": self._resumed_from["is_resume"],
            "event_count": self._resumed_from["event_count"],
            "state_json_present": self._resumed_from["state_json_present"],
            "pending_restored": rebuilt,
            "verdicts_seen": len(verdicts),
        }

    def _publish_recovered_keeps(self, report: dict[str, Any]) -> None:
        """Materialise the KB sections the recovered KEEPs owe, once the state is durable.

        The outbox must never publish a config the state file has not made
        authoritative yet, so a save that failed leaves the sections pending.
        """
        state = self.shared_state
        resume_state_durable = True
        try:
            state.save(self.session_dir)
        except Exception:
            resume_state_durable = False
            log.exception("Coordinator: pre-outbox resume save failed")
            report["warnings"].append({"kind": "resume_pre_outbox_save_failed"})
        pending_before = len(state.kb_stage_outbox or [])
        if not (pending_before and resume_state_durable):
            return
        self._drain_agent_keep_outbox()
        pending_after = len(state.kb_stage_outbox or [])
        if pending_after:
            report["warnings"].append({"kind": "kb_stage_outbox_incomplete", "pending": pending_after})
        else:
            report["fixes"].append({"kind": "reconciled_kb_stage_outbox", "count": pending_before})

    async def _compensate_validation_watermark(self, report: dict[str, Any]) -> None:
        """Enqueue the one full-stack rebench that re-measures unvalidated KEEPs."""
        state = self.shared_state
        if not state.optimization_stack_has_unvalidated_keeps():
            return
        stack = [e for e in (state.optimization_stack or []) if isinstance(e, dict)]
        generation = int(state.working_recipe_generation or 0)
        report["warnings"].append(
            {
                "kind": "resume_unvalidated_keeps",
                "validated_stack_len": int(state.cumulative_gain_validated_stack_len or 0),
                "stack_len": len(stack),
                "working_recipe_generation": generation,
                "validated_recipe_generation": state.validated_recipe_generation,
            }
        )
        fix = await self.enqueue_internal_stack_rebench(
            reason="resume_unvalidated_keeps",
            idempotency_key=f"resume-stack-revalidate-g{generation}",
        )
        report["fixes"].append({"kind": "queued_resume_stack_rebench", **fix})

    async def _resume_consistency_pass(self) -> dict[str, Any]:
        """One-shot resume audit + recovery for stack/current_best consistency.

        Recovers half-applied / orphaned KEEPs through the same lift the live
        path uses, then compensates the validation watermark by enqueuing a
        single full-stack end-to-end rebench. Idempotent — only runs on a
        resumed session and every recovery step dedupes, so a second pass is a
        no-op.
        """
        if not self._resumed_from.get("is_resume"):
            return {"skipped": True, "reason": "not_resume"}
        state = self.shared_state
        report: dict[str, Any] = {
            "skipped": False,
            "fixes": [],
            "warnings": [],
        }
        active_inferencex = str(state.active_inferencex_path or "").strip()
        if active_inferencex:
            if has_benchmark_lib(active_inferencex):
                os.environ["INFERENCEX_PATH"] = active_inferencex
            elif Path(active_inferencex).is_dir():
                # Not an InferenceX tree: dropped, so the checkout this launch resolved stays the one every round uses.
                report["warnings"].append({"kind": "active_inferencex_checkout_invalid", "path": active_inferencex})
                state.active_inferencex_path = ""
            else:
                report["warnings"].append({"kind": "active_inferencex_checkout_missing", "path": active_inferencex})
                state.set_stop_reason("active_inferencex_checkout_missing")
        # Loading the state file is the one boundary the persisted
        # ``optimization_stack`` crosses, so bind its rows to their members here:
        # a row that cannot name what it integrated makes every later read of the
        # stack a guess, and every reader below inherits that guess.
        stack_phase = self._coord.phase_kernel_stack
        try:
            stack_phase.stack_resolved_kernel_ids()
        except ValueError as exc:
            stack_phase.halt_stack_recovery(exc)
        # (0) Interrupted stack attempt: its members may still be applied to the
        # framework tree, and everything this pass or the leg benchmarks before
        # SWEEP entry would measure them. SWEEP entry settles through the same
        # recovery.
        await self._resume_recover_interrupted_stack(report)
        # (1) Half-applied integrate window: replay the
        # missing stack append or roll back the partial patch BEFORE anything
        # reads the stack, so the rest of the pass sees the recovered truth.
        await self._resume_recover_pending_integrate(report)
        if state.pending_integrate:
            state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
            state.save(self.session_dir)
            raise IntegrateRecoveryIncomplete(
                "pending integrate recovery is incomplete; refusing resume before measurement"
            )
        # (1b) In-flight targeted build: an off-loop compile cannot survive a
        # coordinator restart, so kill the orphan group, GC its attempt dir,
        # sweep its jit locks, fail the row, and clear the sentinel.
        await self._resume_recover_pending_targeted_build(report)
        # (1c) Combined PRELUDE replay: no benchmark verdict survived the
        # restart, so restore both Recipe and Kernel trees before continuing.
        await self._resume_recover_pending_warm_replay(report)
        # (1d) Orphaned revalidation tasks: if enablement_validation_pending is set
        # but the tracked revalidation task is already terminal, unstick the window
        # so a fresh revalidation can be enqueued -- closed and charged to the
        # stall counter, or reopened uncharged when the run cancelled the row.
        await self._resume_recover_pending_revalidation(report)
        # (2) Orphaned KEEPs: replay integrate_patch KEEPs
        # that crashed before the append landed; surface ambiguous ones loudly.
        await self._resume_recover_orphaned_keeps(report)

        # (3) A config with no stack behind it cannot be reproduced.
        if state.current_best and not state.optimization_stack:
            report["warnings"].append({"kind": "current_best_without_stack"})

        self._publish_recovered_keeps(report)
        await self._compensate_validation_watermark(report)
        try:
            state.save(self.session_dir)
        except Exception:
            log.exception("Coordinator: resume consistency save failed")
        await self.bus.record_observation("coordinator", "observation", {"kind": "resume_consistency", **report})
        return report

    def _replay_keep_from_result(self, kind: str, result: dict[str, Any]) -> bool:
        """Replay a recorded KEEP delegated-result into current_best/stack.

        Reconstructs the winning-variant dict from a persisted ``delegated_result``
        and routes it through :meth:`lift_to_current_best`, which dedupes by
        ``(action, variant_name)`` — so replay is idempotent. Used by both the
        pending-integrate (Gap C) and orphaned-KEEP (Gap B) resume recovery
        paths. Returns ``True`` only when a new stack entry was appended.

        Args:
            kind: The originating action kind (``integrate_patch`` / ``explore``
                / ``framework``).
            result: The recorded delegated result payload for that KEEP.

        Returns:
            ``True`` when the replay appended a new stack entry, else ``False``.
        """
        if not isinstance(result, dict):
            return False
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and float(tput) > 0):
            return False
        if kind == "explore":
            bv = _replayed_explore_variant(result)
        elif kind == "integrate_patch":
            bv = _replayed_integrate_variant(result, float(tput))
        else:
            return False
        if bv is None:
            return False
        before = len(self.shared_state.optimization_stack or [])
        if not self.lift_to_current_best(
            kind,
            float(tput),
            bv,
            gap_canonical_id=str(result.get("gap_canonical_id") or ""),
        ):
            return False
        return len(self.shared_state.optimization_stack or []) > before

    def _resume_rollback_pending_integrate(self, pending: dict[str, Any], gone_scope: str | None) -> dict[str, Any]:
        """Restore a pending integration from its durable file and stash evidence.

        Incomplete evidence or an unverifiable restore remains pending; the
        resume entry point refuses to continue to measurement in that state.

        Args:
            pending: The ``pending_integrate`` sentinel dict.
            gone_scope: What an operator confirmed stopped with the task (``confirmed_stopped_scope``), if anything.

        Returns:
            A summary ``{"reversed": [...], "failed": [...]}``.
        """
        from ..actions.executors.integrate_patch import restore_pending_integrate
        from ..actions.executors.source_tree_identity import settle_replaced_tree, tree_replaced

        if tree_replaced(pending, gone_scope):
            return settle_replaced_tree(pending)
        return restore_pending_integrate(pending)

    @staticmethod
    def _gc_attempt_runtime(attempt_dir: str) -> bool:
        """Remove an attempt-runtime dir (best-effort).

        Returns True when a directory was present and removal was attempted.
        """
        import shutil

        path = Path(str(attempt_dir or "").strip())
        if not attempt_dir or not path.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True

    async def _resume_recover_interrupted_stack(self, report: dict[str, Any]) -> None:
        """Settle a stack attempt a crash or halt left behind, before anything here can benchmark.

        The recovery halts the session if the tree cannot be settled, which is
        the point: the alternative is measuring a tree whose contents no resume
        can account for.
        """
        if await self._coord.phase_kernel_stack.recover_interrupted_stack_validation():
            report["fixes"].append({"kind": "interrupted_stack_validation_recovered"})

    def _clear_pending_integrate(self, pending: dict[str, Any], *, gc_runtime: bool) -> None:
        """Drop a discharged sentinel, and its attempt runtime once nothing runs from it."""
        self.shared_state.pending_integrate = {}
        self.shared_state.save(self.session_dir)
        if gc_runtime and pending.get("attempt_venv_root"):
            self._gc_attempt_runtime(str(Path(pending["attempt_venv_root"]).parent))

    def _clear_settled_integrate_window(
        self,
        pending: dict[str, Any],
        report: dict[str, Any],
        marker: _IntegrateMarker,
    ) -> bool:
        """Clear a sentinel whose records show the tree was put back; whether it was cleared.

        A confirmed outcome carrying no result still counts when the phase says
        the tree is clean: the attempt restored it and then died before
        recording what it decided. An unconfirmed or absent outcome is a
        different case and is left to the caller.
        """
        task_id = str(pending.get("task_id") or "")
        if marker.verdict == "settled":
            fix = {
                "kind": "settled_pending_integrate",
                "task_id": task_id,
                "status": str((marker.outcome or {}).get("status") or ""),
            }
        elif marker.phase == "restored" and marker.outcome is not None:
            fix = {"kind": "settled_pending_integrate", "task_id": task_id, "phase": marker.phase}
        else:
            return False
        report["fixes"].append(fix)
        self._clear_pending_integrate(pending, gc_runtime=True)
        return True

    async def _scan_for_published_integrate_keep(self, task_id: str) -> _KeptResultScan:
        """Search the event log for the KEEP an integrate window published before it crashed."""
        try:
            for msg in await self.bus.tail(topic="delegated_result", n=-1):
                payload = msg.payload or {}
                if str(payload.get("task_id") or "") != task_id:
                    continue
                res = payload.get("result") or {}
                # Require an explicit integrate_patch kind: an empty-kind wildcard
                # could misclassify a non-integrate event that happens to share
                # this task_id as a kept integrate result, skipping rollback of a
                # half-applied patch.
                if (
                    isinstance(res, dict)
                    and str(res.get("kind") or payload.get("kind") or "") == "integrate_patch"
                    and str(res.get("status") or "").lower() == "kept"
                ):
                    return _KeptResultScan(scanned=True, kept=res)
        except Exception:
            log.exception("Coordinator: pending_integrate kept-result scan failed")
            return _KeptResultScan(scanned=False, kept=None)
        return _KeptResultScan(scanned=True, kept=None)

    async def _recover_pending_integrate_from_task(
        self,
        pending: dict[str, Any],
        task: Task | None,
        report: dict[str, Any],
    ) -> bool:
        """Settle an integrate window from durable records rather than the bus.

        Args:
            pending: The ``pending_integrate`` sentinel dict.
            task: The sentinel's ``integrate_patch`` task row, if it still exists.
            report: The resume report dict to append fixes/warnings to.

        Returns:
            ``True`` when the sentinel was discharged or deliberately retained,
            ``False`` when the caller should roll the window back (a no-op when it changed nothing in the tree).
        """
        state = self.shared_state
        task_id = str(pending.get("task_id") or "")
        # Event retention can erase a KEEP. A promoted stack row is
        # independent positive evidence; a terminal/missing task without
        # its result is not evidence that a candidate was rejected.
        stack = state.optimization_stack or []
        if any(isinstance(row, dict) and row.get("task_id") == task_id for row in stack):
            state.pending_integrate = {}
            state.save(self.session_dir)
            report["fixes"].append({"kind": "retained_promoted_pending_integrate", "task_id": task_id})
            return True
        marker = _integrate_marker_of(pending, task)
        if marker.verdict == "retained":
            appended = self._replay_keep_from_result("integrate_patch", marker.outcome or {})
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
            self._clear_pending_integrate(pending, gc_runtime=False)
            return True
        if not (pending.get("patches") or pending.get("artifacts")):
            return False  # env/args only: short of a recorded KEEP there is nothing to hold, only to clear
        if self._clear_settled_integrate_window(pending, report, marker):
            return True
        if marker.phase == "accepted":
            # A KEEP verdict left the candidate in the tree on purpose and popped the user's stash, so neither rollback
            # nor clear is correct here. Hold the obligation: dropping it would strand unrecorded candidate code, and
            # standing in a baseline number for the lost measurement would poison current_best and misattribute the
            # candidate's gain on the next stack rebench.
            report["warnings"].append(
                {"kind": "pending_integrate_outcome_unknown", "task_id": task_id, "phase": marker.phase}
            )
            return True
        if marker.verdict == "incomplete":
            # A confirmed outcome that discharged nothing: the teardown the
            # attempt owed is still owed, so retry it rather than hold the
            # sentinel until someone edits state.json. An online restore that
            # failed lands here on a task the runner recorded as succeeded.
            return False
        if confirmed_stopped_scope(task.history if task is not None else None) is not None:
            return False  # confirmed stopped with no verdict: nothing will ever finish this window
        if task is None or task.state not in ("queued", "running", "failed"):
            report["warnings"].append({"kind": "pending_integrate_outcome_unknown", "task_id": task_id})
            return True
        if _task_recorded_a_result(task):
            report["warnings"].append({"kind": "pending_integrate_result_missing", "task_id": task_id})
            return True
        return False

    async def _resume_recover_pending_integrate(self, report: dict[str, Any]) -> None:
        """Recover a crashed integrate_patch window from the sentinel.

        The verdict is taken from the first source that still carries one: a
        published KEEP on the bus, then the task row's own completion evidence,
        which the runner writes before it publishes. Whatever is found is
        classified by the same ``(status, phase)`` table the live promotion path
        reads, then replayed, cleared, or rolled back accordingly. A scan that
        could not read the event log establishes nothing, so it must neither
        roll back nor clear the sentinel.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        pending = state.pending_integrate or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        if not task_id:
            report["warnings"].append({"kind": "pending_integrate_task_missing"})
            return
        scan = await self._scan_for_published_integrate_keep(task_id)
        if not scan.scanned:
            report["warnings"].append({"kind": "pending_integrate_scan_failed", "task_id": task_id})
            return
        if scan.kept is not None:
            appended = self._replay_keep_from_result("integrate_patch", scan.kept)
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
            self._clear_pending_integrate(pending, gc_runtime=False)
            return
        task = await self.tasks.find(task_id, kind="integrate_patch")
        if await self._recover_pending_integrate_from_task(pending, task, report):
            return
        gone_scope = confirmed_stopped_scope(task.history if task is not None else None)
        summary = self._resume_rollback_pending_integrate(pending, gone_scope)
        if summary.get("failed"):
            report["warnings"].append({"kind": "pending_integrate_rollback_failed", "task_id": task_id, **summary})
            state.save(self.session_dir)
            return
        if summary.get("reversed") or summary.get("artifacts_reverted") or summary.get("tree_replaced"):
            report["fixes"].append({"kind": "rolled_back_pending_integrate", "task_id": task_id, **summary})
        else:
            report["fixes"].append({"kind": "cleared_stale_pending_integrate", "task_id": task_id})
        self._clear_pending_integrate(pending, gc_runtime=True)

    async def _resume_recover_pending_warm_replay(
        self,
        report: dict[str, Any],
    ) -> None:
        """Rollback a combined PRELUDE set whose verdict was lost to a crash."""
        state = self.shared_state
        pending = state.warm_replay_pending or {}
        if not isinstance(pending, dict) or not pending:
            return
        rollback = self._coord.phase_prelude.rollback_combined_warm({}, None)
        errors = list(rollback.get("errors") or [])
        if errors:
            report["warnings"].append(
                {
                    "kind": "resume_warm_rollback_failed",
                    "task_id": pending.get("task_id"),
                    "errors": errors,
                }
            )
            state.set_stop_reason("warm_replay_rollback_failed")
            state.save(self.session_dir)
            return
        task_id = str(pending.get("task_id") or "").strip()
        task_state = ""
        if task_id:
            try:
                from ..state.task_registry import TaskNotFound

                try:
                    task = await self.tasks.get(task_id)
                except TaskNotFound:
                    task = None
                if task is not None:
                    task_state = str(task.state or "")
                    if task_state == "queued":
                        await self.tasks.transition(
                            task_id,
                            "cancelled",
                            evidence={"reason": "resume_interrupted_warm_replay"},
                        )
                        task_state = "cancelled"
                    elif task_state == "running":
                        await self.tasks.transition(
                            task_id,
                            "failed",
                            evidence={"failure_class": "resume_interrupted"},
                        )
                        task_state = "failed"
            except Exception as exc:  # noqa: BLE001
                state.warm_replay_pending = {
                    **dict(pending),
                    "status": "task_invalidation_failed",
                }
                report["warnings"].append(
                    {
                        "kind": "resume_warm_task_invalidation_failed",
                        "task_id": task_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                state.set_stop_reason("warm_replay_rollback_failed")
                state.save(self.session_dir)
                return
        state.warm_replay_outcome = {
            **dict(state.warm_replay_outcome or {}),
            "status": "failed",
            "reason": "interrupted_combined_validation_rolled_back",
            # This terminal branch never runs ``promote_warm_replay``, which is
            # what normally stamps ``settled_at``; stamp it here so the SBD
            # warm_replay event reports the real span instead of collapsing its
            # end_time back onto ``enqueued_at`` (a zero-duration replay).
            "settled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kernel": {
                "status": "reverted",
                "reason": "interrupted_combined_validation",
            },
        }
        report["fixes"].append(
            {
                "kind": "recovered_pending_warm_replay",
                "task_id": pending.get("task_id"),
                "task_state": task_state,
            }
        )
        state.save(self.session_dir)

    async def _resume_recover_pending_targeted_build(self, report: dict[str, Any]) -> None:
        """Reclaim an off-loop build that was in flight when the coordinator died.

        A detached compile cannot be re-adopted across a restart: kill its
        recorded process group, rmtree the attempt dir, sweep its per-attempt
        aiter JIT locks (a killed compile leaves a pid-less lock that wedges
        every later build of that module), mark the row failed with a
        ``timeout`` failure_class for the framework channel, and clear the
        sentinel. Best-effort throughout.
        """
        import shutil
        import signal

        from ..enablement.runtime.targeted_build import kill_build_pgroup

        state = self.shared_state
        pending = state.pending_targeted_build or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        summary: dict[str, Any] = {"kind": "reclaimed_pending_targeted_build", "task_id": task_id}

        try:
            pgid = int(pending.get("pgid") or 0)
        except (TypeError, ValueError):
            pgid = 0
        if pgid > 0:
            kill_build_pgroup(pgid, sig=signal.SIGKILL)
            summary["killed_pgid"] = pgid

        attempt_root = str(pending.get("attempt_root") or "").strip()
        if attempt_root and Path(attempt_root).exists():
            shutil.rmtree(attempt_root, ignore_errors=True)
            summary["removed_attempt_root"] = attempt_root

        jit_dir = str(pending.get("aiter_jit_dir") or "").strip()
        if jit_dir:
            try:
                from ..actions.executors._aiter_jit import sweep_stale_aiter_locks_if_dead

                sweep_stale_aiter_locks_if_dead(aiter_jit_dir=Path(jit_dir))
                summary["swept_jit_dir"] = jit_dir
            except Exception:
                log.debug("resume: targeted-build jit sweep failed for %s", jit_dir, exc_info=True)

        state.enablement.last_build_failure = {
            "failure_class": "timeout",
            "failure_summary": "targeted build interrupted by coordinator restart",
        }
        if task_id:
            try:
                task = await self.tasks.get(task_id)
                if getattr(task, "state", "") == "running":
                    await self.tasks.transition(task_id, "failed", evidence={"failure_class": "resume_interrupted"})
                    summary["failed_row"] = True
            except Exception:
                log.debug("resume: targeted-build row fail raced for %s", task_id, exc_info=True)

        state.pending_targeted_build = {}
        report["fixes"].append(summary)

    async def _resume_recover_pending_revalidation(self, report: dict[str, Any]) -> None:
        """Unstick enablement_validation_pending when the tracked revalidation task is terminal.

        If the coordinator died while a revalidation baseline was running, the
        task row may already be in a terminal state on resume.  Without this
        recovery the pending flag stays set indefinitely and the next
        revalidation cannot be enqueued (tracked_tid is still the old row).

        Which recovery depends on how the row ended, not on this being the resume
        path. A row the run cancelled -- which is what the queue scan does to a
        revalidation the wall-clock budget can no longer fit -- measured nothing
        and produced no result to route, so it is no evidence about the baseline
        and gets :meth:`_reopen_revalidation_window`, the same verdict the
        writeback reaches for the same round. Anything else ended having had its
        chance: the window closes and the round charges an observation carrying
        no evidence, which spends an evidence-stall credit.
        """
        state = self.shared_state
        if not bool(state.enablement.validation_pending):
            return
        tracked_tid = str(state.enablement.revalidation_task_id or "").strip()
        if not tracked_tid:
            return
        try:
            from ..state.task_registry import TERMINAL_STATES, TaskNotFound

            row_state = ""
            try:
                row = await self.tasks.get(tracked_tid)
                row_state = str(getattr(row, "state", "") or "")
                is_terminal = row_state in TERMINAL_STATES
            except TaskNotFound:
                is_terminal = True
            if not is_terminal:
                return
            if row_state == "cancelled":
                await self._reopen_revalidation_window()
                report["fixes"].append({"kind": "reopened_revalidation_the_run_cancelled", "task_id": tracked_tid})
                log.info(
                    "resume: revalidation task %s was cancelled by the run; window left open at generation %d",
                    tracked_tid,
                    int(state.enablement.revalidation_generation or 0),
                )
                return
            state.enablement.validation_pending = False
            state.enablement.revalidation_task_id = ""
            report["fixes"].append({"kind": "cleared_orphaned_revalidation_pending", "task_id": tracked_tid})
            log.info(
                "resume: cleared stale enablement_validation_pending for terminal revalidation task %s",
                tracked_tid,
            )
        except Exception:
            log.debug("resume: revalidation pending recovery check failed", exc_info=True)

    async def _surface_orphaned_keep(self, report: dict[str, Any], kind: str, variant: str, task_id: Any) -> None:
        """Alert on an explore or framework KEEP the stack lost, without resurrecting it.

        Which stack such a KEEP landed on is known only to the round that ran
        it, so a replay here cannot tell recovery from re-applying an eviction.
        """
        report["warnings"].append(
            {"kind": "orphaned_keep", "orphan_kind": kind, "variant": variant, "task_id": task_id}
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "orphaned_keep", "severity": "medium", "orphan_kind": kind, "variant": variant},
        )

    async def _recover_orphaned_integrate_keep(
        self,
        result: dict[str, Any],
        report: dict[str, Any],
        variant: str,
        task_id: Any,
    ) -> bool:
        """Replay an integrate_patch KEEP the stack lost; whether the stack now carries it.

        A KEEP whose run workspace is gone can no longer be replayed, so it is
        discarded with an operator alert instead.
        """
        kind = "integrate_patch"
        workspace = str(result.get("workspace") or "").strip()
        if workspace and not Path(workspace).exists():
            report["warnings"].append(
                {
                    "kind": "orphaned_keep_discarded",
                    "orphan_kind": kind,
                    "variant": variant,
                    "task_id": task_id,
                    "reason": "workspace_missing",
                }
            )
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {"kind": "orphaned_keep_discarded", "severity": "medium", "orphan_kind": kind, "variant": variant},
            )
            return False
        if self._replay_keep_from_result(kind, result):
            report["fixes"].append({"kind": "replayed_orphaned_keep", "orphan_kind": kind, "variant": variant})
            return True
        report["warnings"].append({"kind": "orphaned_keep_replay_noop", "orphan_kind": kind, "variant": variant})
        return False

    async def _resume_recover_orphaned_keeps(self, report: dict[str, Any]) -> None:
        """Recover / surface KEEPs present in the event log but absent from the stack (Gap B).

        ``integrate_patch`` KEEPs are well-defined (a ``kept`` status means the
        single-variant bench + accuracy gate passed and the patch was committed),
        so a kept-but-absent one is a crash before the append landed → replay it
        (idempotent), unless its run workspace is gone → discard + alert. ``explore``
        / ``framework`` KEEPs are ambiguous (the stack they landed on is only
        known to the round that ran), so they are surfaced as a ``medium``
        alert rather than resurrected. Whatever the stack ends up as
        is re-validated by the Gap A full-stack rebench.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        try:
            stack_keys = _stacked_keep_keys(self.shared_state.optimization_stack)
            seen: set[_StackKey] = set()
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                res = payload.get("result") or {}
                if not isinstance(res, dict) or str(res.get("status") or "").lower() != "kept":
                    continue
                kind = str(payload.get("kind") or "")
                key = _orphan_stack_key(kind, res)
                if key is None or not key.variant or key in stack_keys or key in seen:
                    continue
                seen.add(key)
                if kind != "integrate_patch":
                    await self._surface_orphaned_keep(report, kind, key.variant, payload.get("task_id"))
                elif await self._recover_orphaned_integrate_keep(res, report, key.variant, payload.get("task_id")):
                    stack_keys.add(key)
        except Exception:
            log.exception("Coordinator: orphaned KEEP resume recovery failed")

    async def enqueue_internal_stack_rebench(
        self,
        *,
        reason: str,
        idempotency_key: str = "stack-revalidate",
    ) -> dict[str, Any]:
        """Enqueue one full-stack end-to-end rebench of the cumulative config.

        Builds a single-variant ``explore`` task from ``current_best``'s launch
        args/envs, benched against ``baseline_tput`` so the measured delta
        becomes the validated cumulative gain. Tagged ``source=stack_revalidate``
        so ``promote_to_shared_state`` reconciles ``cumulative_gain_validated_stack_len``
        from the measured throughput.

        Args:
            reason: Human-readable reason stamped on the task params.
            idempotency_key: Key for the native stack rebench.

        Returns:
            A summary ``{"task_id", "existing"}`` or ``{"skipped", "reason"}``.
        """
        benchmark_script = self.shared_state.accepted_baseline_script()
        # The grid is frozen from ``current_best`` now; a lift before the result lands makes it a measurement of an
        # older Recipe.
        recipe_generation = int(self.shared_state.working_recipe_generation or 0)
        launch = self.current_best_launch_config()
        args = launch["extra_server_args"]
        envs = launch["extra_envs"]
        overlay = launch["final_overlay"]
        cb_now = self.shared_state.current_best
        cb_remove = cb_now.get("remove_args")
        cb_unset = cb_now.get("unset_envs")
        cb_replace = str(cb_now.get("args_mode") or "").strip().lower() == "replace"
        if not (args or envs or cb_remove or cb_unset or cb_replace):
            return {"skipped": True, "reason": "empty_config"}
        params: dict[str, Any] = {
            "source": STACK_REVALIDATE_SOURCE,
            "reason": reason,
            "recipe_generation": recipe_generation,
            "grid": [
                {
                    "name": STACK_REVALIDATE_SOURCE,
                    "extra_args": args,
                    "extra_envs": dict(envs),
                    # Carry the overlay so an authored-kernel native stack rebuild
                    # loads the built kernels (inert when empty).
                    "overlay_pythonpath": overlay,
                    "provenance": STACK_REVALIDATE_SOURCE,
                    "note": "full-stack end-to-end revalidation",
                }
            ],
            # The rebench reproduces the whole stack, so its gain is cumulative-vs-baseline.
            "base_tput": float(self.shared_state.baseline_tput or 0.0),
        }
        if cb_remove:
            params["base_remove_args"] = [cb_remove] if isinstance(cb_remove, str) else list(cb_remove or [])
        if cb_unset:
            params["base_unset_envs"] = [cb_unset] if isinstance(cb_unset, str) else list(cb_unset or [])
        if cb_replace:
            params["base_args_mode"] = "replace"
        if self.shared_state.baseline_config_path:
            params["config_path"] = self.shared_state.baseline_config_path
        if benchmark_script:
            params["benchmark_script"] = benchmark_script
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("explore")
        self._coord.proposals.inject_explore_runtime_params(params)
        task, existing = await self.tasks.create_or_return_existing(
            kind="explore",
            params=params,
            idempotency_key=idempotency_key,
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
            dispatch_class="coordinator",
        )
        return {"task_id": task.task_id, "existing": bool(existing)}

    async def _resume_reenter_kernel_if_needed(self) -> None:
        """Idempotently re-fire the KERNEL_AGENT entry hook on resume.

        Phase-entry side effects (the GEAK delegation and its ``result.json``
        crash-recovery) are bound
        to a phase *transition* via ``_on_phase_entered``; a resume only restores
        ``phase`` from state.json and never re-enters the current phase. Without
        this, a session that crashed mid ``KERNEL_AGENT`` sits idle until the
        phase budget cap fires, then hands SWEEP an empty result — the whole
        delegation is silently lost.

        General across every crash timing (not case-by-case): the decision is
        driven purely by whether THIS KERNEL phase's history row already carries
        a ``geak`` completion record, so it self-classifies:

          * completed-this-phase -> the kernel work is done; the phase machine
            exits KERNEL automatically on the next tick (agent settled, no pending work)
            with no e2e re-run;
          * not-completed -> re-enter ``on_enter_kernel``; its own entry guard
            promotes an existing OK ``result.json`` (crash-before-handback) and
            re-runs the e2e only when there is genuinely nothing to recover
            (run_e2e itself then continues from the pinned eval_dir on disk).

        No-op unless resumed while parked in ``KERNEL_AGENT`` with the GEAK
        backend selected.
        """
        from ..phases.machine_state import PHASE_KERNEL_AGENT

        if not self._resumed_from.get("is_resume"):
            return
        state = self.shared_state
        if (state.phase or "").strip().upper() != PHASE_KERNEL_AGENT:
            return
        if not (self._coord.phase_machine.kernel_enabled() and self._coord.phase_kernel.geak_enabled()):
            return
        history = state.phase_history or []
        row = history[-1] if history else {}
        evidence = row.get("evidence") if isinstance(row, dict) else {}
        completed_this_phase = isinstance(evidence, dict) and isinstance(evidence.get("geak"), dict)
        if completed_this_phase:
            # The delegation landed during this phase but the SWEEP transition
            # never persisted (crash between hook and the next tick). The exit
            # rule sees geak_result set (terminal) and no work pending, so the
            # machine advances on the next tick without any hint.
            log.info("resume: KERNEL GEAK already completed this phase; machine will exit on next tick.")
            return
        log.info(
            "resume: re-entering KERNEL GEAK delegation (no completion "
            "evidence on the current phase row); recover-from-disk or re-run."
        )
        from hyperloom.orchestrator.phases.machine import Transition

        await self._coord.phase_kernel.on_enter_kernel(
            Transition(from_phase="resume", to_phase="KERNEL_AGENT", reason="resume", evidence={})
        )

    @property
    def resumed_from(self) -> dict[str, Any]:
        """Read-only snapshot of resume detection (set by ``__init__``).

        Returns:
            A copy of the resume-detection dict so callers cannot mutate
            internal state.
        """
        return dict(self._resumed_from)

    # Bounded test interface
    async def replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session (replay log + abandon orphan dispatches)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()
        await self._resume_consistency_pass()
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT
        # (phase-entry side effects are bound to transitions, not resume). Runs
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
