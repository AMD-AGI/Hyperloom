# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lifting a measured winner into ``current_best``, and the stack entry it leaves behind."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, NamedTuple

from hyperloom.common.coerce import to_str_list
from hyperloom.inference_optimizer.grid_server_args import (
    dedupe_extra_server_args,
    merge_cumulative_extra_server_args,
    strip_benchmark_harness_flags,
)
from hyperloom.inference_optimizer.session.optimization_journal import classify_change_kind, operation_kind_for
from hyperloom.orchestrator.lever import FRAMEWORK_STACK_ACTION, lever_kind_for_task
from hyperloom.plugs import progress as _progress

from ..framework.artifacts import candidate_key
from ..measurement.runtime_findings import KEEP_REASON_CORRECTNESS_FIX
from ..state.shared_state import resolve_graded_comparison
from .launch_controls import kept_launch_controls
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyperloom.common.perf_metric import GradedComparison

log = logging.getLogger(__name__)


class _LiftRequest(NamedTuple):
    task_kind: str
    tput: float
    variant: dict[str, Any]
    gap_canonical_id: str
    entry_extra: Mapping[str, Any]


class _ResolvedLaunch(NamedTuple):
    variant_name: Any
    candidate_args: str
    full_args: str
    overlay: str


def _resolved_launch(previous: Mapping[str, Any], bv: Mapping[str, Any]) -> _ResolvedLaunch:
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


def _graded_source(measurement: Mapping[str, Any], output_tput: float) -> dict[str, Any]:
    """*measurement* with the caller's resolved output throughput stamped in.

    A winner record does not always carry its own throughput, so grading must
    not read back out what the caller already resolved.
    """
    return {**measurement, "output_throughput": float(output_tput)}


def _already_stacked(stack: list[Any], request: _LiftRequest, variant_name: Any) -> bool:
    rows = [row for row in stack if isinstance(row, dict)]
    applied = {(str(row.get("action")), str(row.get("variant_name"))) for row in rows}
    if (request.task_kind, str(variant_name or "")) in applied:
        return True
    fingerprint = str(request.variant.get("fingerprint") or "")
    return bool(fingerprint) and any(fingerprint == str(row.get("fingerprint") or "") for row in rows)


def _variant_fingerprint(bv: Mapping[str, Any], candidate_args: str, full_args: str) -> str:
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


class _StackKey(NamedTuple):
    action: str
    variant: str


def _stacked_keep_keys(stack: Any) -> set[_StackKey]:
    return {
        _StackKey(str(entry.get("action") or ""), str(entry.get("variant_name") or ""))
        for entry in (stack or [])
        if isinstance(entry, dict)
    }


def _orphan_stack_key(kind: str, result: Mapping[str, Any]) -> _StackKey | None:
    if kind == "integrate_patch":
        return _StackKey(kind, str(result.get("specialist_task_id") or ""))
    if kind == "framework_agent":
        candidate = result.get("candidate")
        return _StackKey(FRAMEWORK_STACK_ACTION, candidate_key(candidate if isinstance(candidate, dict) else None))
    if kind == "explore":
        best = result.get("best_variant") or {}
        return _StackKey(kind, str((best.get("name") if isinstance(best, dict) else "") or ""))
    return None


class CurrentBestLiftMixin:
    """Accepting a winner as the run's new current best."""

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
            if (
                task_kind == "integrate_patch"
                and float(self.shared_state.baseline_tput or 0.0) <= 0.0
                and bv.get("baseline_enablement") is True
                and bv.get("attribution_eligible") is False
            )
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
        from hyperloom.common.perf_metric import VERDICT_KEEP

        if not graded.comparable:
            log.info("current_best held: %s winner not comparable (%s)", task_kind, graded.degrade_reason)
            return True
        if graded.veto_reason:
            log.warning(
                "current_best held: %s winner refused by %s (%s)", task_kind, graded.vetoed_by, graded.veto_reason
            )
            _progress.warn("winner_vetoed", f"{task_kind} winner refused by {graded.vetoed_by}: {graded.veto_reason}")
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
        from hyperloom.common.perf_metric import baseline_on_axis
        from hyperloom.inference_optimizer.breakdown.recorder import stack_event

        stack_event.record_adoption(
            stack_index=len(self.shared_state.optimization_stack) - 1,
            entry=stack_entry,
            throughput_before=graded.reference,
            throughput_after=graded.candidate,
            baseline_tput=baseline_on_axis(self.shared_state, graded.objective),
            objective=graded.objective,
            degrade_reason=graded.degrade_reason,
        )

    def _stack_entry(self, request: _LiftRequest, launch: _ResolvedLaunch) -> dict[str, Any]:
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
        current_best.update(kept_launch_controls(bv, previous))
        if bv.get("effective_extra_server_args"):
            current_best["effective_extra_server_args"] = dedupe_extra_server_args(
                strip_benchmark_harness_flags(bv.get("effective_extra_server_args"))
            )
        if (bv.get("remove_args") or bv.get("unset_envs")) and not current_best.get("args_mode"):
            current_best["args_mode"] = "replace"
        self.shared_state.current_best = current_best
        self.shared_state.working_recipe_generation = int(self.shared_state.working_recipe_generation or 0) + 1
        self._stamp_current_best_measurement(bv)
        _progress.best_updated(request.tput, self.shared_state, bv.get("request_id"))
