# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from hyperloom.common.coerce import to_float, to_str_list
from hyperloom.common.io import append_jsonl
from hyperloom.common.launch_log_evidence import (
    launch_argv_from_log,
    observed_sglang_server_identity_from_log,
)
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_CONFIG,
    LEVER_KERNEL,
    LEVER_UPSTREAM_PR,
    owner_from_lever,
    patch_lever_kind,
    patch_owner_phase,
)
from ..kernel._recorder_trace import trace_recording_skipped
from ..state.optimization_journal import (
    Journal,
    JournalEntry,
    OUTCOME_KEEP,
    OUTCOME_NO_PROMOTE,
    OUTCOME_REVERT,
    PROMOTION_REFUSED_KEY,
    classify_change_kind,
    derive_journal_outcome,
    operation_kind_for,
    summarize_change,
)
from ..actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from ..actions.executors._grid_server_args import strip_benchmark_harness_flags
from ..actions.stop_attribution import stopped_by_the_run_class
from ..state.shared_state import SharedState, resolve_grading_anchor_tput
from hyperloom.inference_optimizer.protocol.intent import Intent
from ..bus.message_bus import Message
from .coordinator_helpers import (
    _MIN_KERNEL_ENGAGED_GAIN_PCT,
    _accepted_config_as_variant,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _merge_cumulative_extra_server_args,
    _parse_baseline_workload_extra,
    _geak_accepted_kernel_specs,
    _geak_has_accepted_kernel,
    _geak_overlay_digest,
    _geak_overlay_is_loadable,
    _geak_result_has_material,
    _geak_revalidation_decision,
    _geak_spec_name,
    _geak_sweep_measured_tput,
    _normalize_geak_overlay_dir,
)
from ..policy.gate import (
    PolicyDenied,
)
from ..state.task_registry import Task
from ..actions.executors.benchmark_result import is_valid_measurement
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
from ..knowledge.agent_kb import ExploreAgentKB, FrameworkAgentKB

from .coordinator import (
    _AUDIT_ACTIONS,
    _BASELINE_MAX_TOTAL_FAILURES,
    _DEFAULT_RESUME_DRIFT_FLOOR_PCT,
    _ENABLEMENT_MAX_STALL,
    _SEVERITY_CRASH,
    _SEVERITY_REGRESS,
    PendingProposal,
    _extract_enablement_launch_log,
)
import logging as _logging

log = _logging.getLogger(__name__)

# Upstream-PR KEEPs are stacked under the ``framework`` attribution family
# label rather than under their task kind, because that label is what
# ``phase_breakdown`` and the action-family table publish.
_FRAMEWORK_STACK_ACTION = "framework"

#: Task kind -> the lever it moves, for winners whose params carried no stamp.
#: ``integrate_patch`` is absent: it lands every lever, so its stamp is the only
#: evidence and a missing one is a real gap rather than something to guess at.
#: ``geak_e2e`` is absent for the same reason: it promotes on a proven kernel
#: overlay OR on a config/env-only win, and only the promoting site knows which.
#: It stamps ``lever_kind`` on the winner from the same overlay proof
#: ``_geak_stack_entry_extra`` uses, so guessing ``kernel`` here would let the
#: lever buckets contradict ``_geak_contribution`` for the very same row.
_LEVER_BY_TASK_KIND = {
    "explore": LEVER_CONFIG,
    "sweep": LEVER_CONFIG,
    "conc_sweep": LEVER_CONFIG,
    "gemm_tuning": LEVER_KERNEL,
    "collective": LEVER_KERNEL,
    "fusion": LEVER_KERNEL,
    "integrate": LEVER_KERNEL,
    # Reachable when a session recorded before the action was retired is
    # resumed and its orphaned KEEPs are reconciled against the stack.
    "framework_agent": LEVER_UPSTREAM_PR,
    _FRAMEWORK_STACK_ACTION: LEVER_UPSTREAM_PR,
}


def _lever_for_keep(task_params: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Name the lever a settled KEEP moved, reading the delivery first.

    What came back outranks what was asked for: a mandate that goes out naming
    one lever routinely returns another, and the result carries the applier's
    own markers.
    """
    return patch_lever_kind(result) or patch_lever_kind(task_params)


def _keep_owner_section(task_params: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """Name the section a KEEP stages into, preferring the lever it moved."""
    by_lever = owner_from_lever(_lever_for_keep(task_params, result))
    if by_lever:
        return by_lever
    return str(task_params.get("source_phase") or result.get("source_phase") or "").strip().upper()


def _lever_kind_for_lift(task_kind: str, bv: Any) -> str:
    """Resolve the lever a winner moved.

    Args:
        task_kind: The action kind that produced the winner.
        bv: The winning variant dict, read for a ``lever_kind`` stamp.

    Returns:
        One of :data:`LEVER_KINDS`, or ``""`` when nothing named a lever --
        which the caller logs rather than papering over.
    """
    # The kind decides where it can only move one lever; ``integrate_patch``
    # lands every lever, so there the producer's stamp is the only evidence.
    by_kind = _LEVER_BY_TASK_KIND.get(str(task_kind or "").strip(), "")
    if by_kind:
        return by_kind
    return patch_lever_kind(bv if isinstance(bv, dict) else None)


@dataclass
class _PromoteOutcome:
    """Mutable carrier threaded through the per-kind promote handlers;
    ``early_return`` skips the shared audit/save tail (sweep / conc_sweep)."""

    changed: bool = False
    audit_decision: str | None = None
    audit_extras: dict[str, Any] = field(default_factory=dict)
    early_return: bool = False


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


def _predicted_gain(*sources: dict[str, Any] | None) -> float | None:
    """First non-zero ``predicted_gain_pct`` (``to_float``-parsed) across ordered sources.

    Sources are checked in order; a non-zero prediction wins. Returns ``None``
    when none carry a usable value so the journal row stays ``predicted``-free
    for unpredicted (default-grid) changes rather than recording a fake 0.
    """
    for src in sources:
        if not isinstance(src, dict):
            continue
        val = to_float(src.get("predicted_gain_pct"))
        if val is not None and val != 0.0:
            return val
    return None


class WritebackCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _emit_lifecycle(
        self,
        *,
        step: str,
        status: str,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Record + persist one operator-facing lifecycle event.

        Best-effort by design: operator-facing logging must never break the
        orchestration loop, so any failure is swallowed at debug level.

        Args:
            step: The machine step name (resolved to a human label downstream).
            status: The lifecycle status (e.g. START / END / ERROR / ENTER).
            artifacts: Optional mapping of produced artifact paths.
            detail: Optional free-text detail.
            duration_s: Optional elapsed seconds for the step.
        """
        try:
            self.shared_state.record_lifecycle_event(
                step=step,
                status=status,
                artifacts=artifacts,
                detail=detail,
                duration_s=duration_s,
            )
            # Terminal events (END/ERROR) always flush; non-terminal markers are
            # debounced by ``_lifecycle_save_min_interval_s``.
            terminal = status in ("END", "ERROR")
            now = time.monotonic()
            if terminal or (now - self._lifecycle_last_save >= self._lifecycle_save_min_interval_s):
                self.shared_state.save(self.session_dir)
                self._coord._lifecycle_last_save = now
        except Exception:  # noqa: BLE001
            log.debug(
                "Coordinator: lifecycle emit failed (step=%s status=%s)",
                step,
                status,
                exc_info=True,
            )

    async def _record_policy_denied(
        self,
        source: str,
        intent: Intent,
        denied: PolicyDenied,
        *,
        action_name: str | None = None,
    ) -> None:
        """Record a PolicyGate denial.

        Publishes a ``policy_denied`` observation and records the denial streak.
        The streak is a fact for LLM self-correction only: there is no
        auto-prune and no ``policy_loop`` stop triggered from it.

        Args:
            source (str): The agent whose intent was denied.
            intent (Intent): The denied intent.
            denied (PolicyDenied): The denial carrying rule / hint / reason.
            action_name (str | None): Explicit action name override; falls back
                to ``intent.payload['action_name']``.
        """
        # Surface every PolicyGate denial in the process log (not just the bus)
        # so security rejections are observable in ops logs.
        log.warning(
            "PolicyGate denied intent: source=%s type=%s rule=%s reason=%s",
            source,
            intent.type.value,
            denied.rule,
            str(denied),
        )
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                source,
                "observation",
                {
                    "kind": "policy_denied",
                    "intent_type": intent.type.value,
                    "rule": denied.rule,
                    "hint": denied.hint,
                    "reason": str(denied),
                },
                priority=0,
            )
        )
        resolved_action = action_name or str((intent.payload or {}).get("action_name") or "")
        # Streak counter is a fact for LLM self-correction only; the system does not auto-prune or stop on it.
        self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )

    async def _record_observation(self, source: str, topic: str, payload: dict) -> None:
        """Append a broadcast observation message to the bus.

        Args:
            source (str): The agent recording the observation.
            topic (str): The bus topic to publish under.
            payload (dict): The observation payload.
        """
        await self.bus.append_and_seq(Message.new(source, "*", topic, payload))

    @staticmethod
    def _keep_patch_sources(
        result: Mapping[str, Any],
        task: "Task | None",
    ) -> tuple[list[Path], list[str]]:
        """Locate authoritative patch files without reconstructing their diff."""
        candidates: list[Any] = []
        params = (getattr(task, "params", None) or {}) if task is not None else {}
        explicit_present = False
        # An executor verdict's patches_applied is authoritative, including an
        # explicit empty list after all candidates were rejected.
        if "patches_applied" in result:
            explicit_present = True
            raw = result.get("patches_applied")
            if isinstance(raw, (str, Path)):
                candidates.append(raw)
            elif isinstance(raw, (list, tuple)):
                candidates.extend(raw)
        else:
            for key in ("patches", "prior_patches", "patch_path", "patch"):
                if key not in result:
                    continue
                explicit_present = True
                raw = result.get(key)
                if isinstance(raw, (str, Path)):
                    candidates.append(raw)
                elif isinstance(raw, (list, tuple)):
                    candidates.extend(raw)
            if "patches" in params:
                explicit_present = True
                raw_params = params.get("patches")
                if isinstance(raw_params, (str, Path)):
                    candidates.append(raw_params)
                elif isinstance(raw_params, (list, tuple)):
                    candidates.extend(raw_params)

        # Raw framework-agent diffs are normally returned in
        # ``patches_applied``. The shallow workspace scan is a recovery path for
        # older result envelopes that only persisted ``workspace``.
        workspace_value = str(result.get("workspace") or "").strip()
        workspace = Path(workspace_value) if workspace_value else None
        if not explicit_present and workspace is not None and workspace.is_dir():
            for base in (workspace, workspace / "patches", workspace / "worktree" / "patches"):
                if not base.is_dir():
                    continue
                candidates.extend(sorted(base.glob("*.patch")))
                candidates.extend(sorted(base.glob("*.diff")))

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
            if canonical in seen:
                continue
            seen.add(canonical)
            resolved.append(canonical)
        return resolved, missing

    def _stage_agent_keep(
        self,
        *,
        owner: str,
        stack_index: int,
        result: Mapping[str, Any],
        task: "Task | None",
        include_patches: bool,
    ) -> bool:
        """Stage one KEEP into its owner section.

        Returns ``False`` when patch staging must be retried. Final config is
        published once from durable ``current_best`` at CLOSE.
        """
        normalized = str(owner or "").strip().upper()
        facade = (
            ExploreAgentKB.open()
            if normalized == "EXPLORE"
            else FrameworkAgentKB.open()
            if normalized == "FRAMEWORK_AGENT"
            else None
        )
        if facade is None or not facade.active:
            return False
        sources, missing = self._keep_patch_sources(result, task)
        if include_patches and missing:
            log.warning(
                "%s kb: KEEP at stack index %d references missing patch members: %s",
                normalized,
                stack_index,
                missing,
            )
            return False
        if include_patches and sources:
            refs = facade.stage_patches(sources, stack_index=stack_index)
            if len(refs) != len(sources) or any(not ref for ref in refs):
                return False
        return True

    def _enqueue_agent_keep_outbox(
        self,
        *,
        owner: str,
        stack_index: int,
        result: Mapping[str, Any],
        task: "Task | None",
        include_patches: bool,
    ) -> None:
        """Persist an idempotent section handoff to run after state durability."""
        from ..knowledge.remote_recipe import KnowledgeSections

        if (
            str(os.environ.get("KNOWLEDGE_STORE_MODE") or "local").strip() != "remote"
            or KnowledgeSections.from_env() is None
        ):
            return
        normalized = str(owner or "").strip().upper()
        if normalized not in {"EXPLORE", "FRAMEWORK_AGENT"}:
            return
        sources, missing = self._keep_patch_sources(result, task) if include_patches else ([], [])
        row = {
            "id": f"{normalized}:{int(stack_index)}",
            "owner": normalized,
            "stack_index": int(stack_index),
            "include_patches": bool(include_patches),
            "patch_sources": [str(path) for path in sources],
            "missing_patch_sources": missing,
        }
        outbox = list(getattr(self.shared_state, "kb_stage_outbox", []) or [])
        if not any(isinstance(existing, dict) and existing.get("id") == row["id"] for existing in outbox):
            outbox.append(row)
        if include_patches and (sources or missing):
            stack = list(self.shared_state.optimization_stack or [])
            if 0 <= int(stack_index) < len(stack) and isinstance(
                stack[int(stack_index)],
                dict,
            ):
                stack[int(stack_index)]["kb_required_owner"] = normalized
                self.shared_state.optimization_stack = stack
        self.shared_state.kb_stage_outbox = outbox

    def _drain_agent_keep_outbox(self) -> None:
        """Run section writes only after the authoritative state save."""
        pending = list(getattr(self.shared_state, "kb_stage_outbox", []) or [])
        if not pending:
            return
        retained: list[dict[str, Any]] = []
        dead_letter = [
            dict(row) for row in (getattr(self.shared_state, "kb_stage_dead_letter", []) or []) if isinstance(row, dict)
        ]
        stack = list(self.shared_state.optimization_stack or [])
        for row in pending:
            if not isinstance(row, dict):
                continue
            missing = [str(path) for path in (row.get("missing_patch_sources") or []) if str(path).strip()]
            if row.get("include_patches"):
                missing.extend(
                    str(path)
                    for path in (row.get("patch_sources") or [])
                    if str(path).strip() and not Path(str(path)).is_file()
                )
            missing = list(dict.fromkeys(missing))
            if missing:
                failed = {
                    **dict(row),
                    "missing_patch_sources": missing,
                    "reason": "patch_source_missing",
                }
                dead_letter = [item for item in dead_letter if item.get("id") != failed.get("id")]
                dead_letter.append(failed)
                index = int(row.get("stack_index") or 0)
                owner = str(row.get("owner") or "").upper()
                if (
                    0 <= index < len(stack)
                    and isinstance(stack[index], dict)
                    and str(stack[index].get("kb_required_owner") or "").upper() == owner
                ):
                    stack[index].pop("kb_required_owner", None)
                log.warning(
                    "%s kb: dropping owner section %s because patch sources are unavailable: %s",
                    owner,
                    row.get("id"),
                    missing,
                )
                continue
            task = SimpleNamespace(params={})
            if not self._stage_agent_keep(
                owner=str(row.get("owner") or ""),
                stack_index=int(row.get("stack_index") or 0),
                result={"patches": list(row.get("patch_sources") or [])},
                task=task,
                include_patches=bool(row.get("include_patches")),
            ):
                retained.append(row)
        self.shared_state.kb_stage_outbox = retained
        self.shared_state.kb_stage_dead_letter = dead_letter[-200:]
        self.shared_state.optimization_stack = stack
        self.shared_state.save(self.session_dir)

    def _record_kernel_opt_partial(self, result: dict[str, Any]) -> None:
        """Streaming callback for ``_run_optimization_batch`` sub-attempts: write each per-kernel entry to kernel_opt_task_attempts immediately so the next-tick prompt is accurate mid-batch.

        Args:
            result: One sub-attempt's per-kernel result dict.
        """
        try:
            self.shared_state.record_kernel_opt(result)
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            # Never let a per-sub-attempt hiccup poison the gather.
            log.exception(
                "_record_kernel_opt_partial failed for kernel_id=%s",
                (result or {}).get("kernel_id") if isinstance(result, dict) else None,
            )

    def _update_cumulative_gain_validated(
        self,
        new_tput: float,
        *,
        source: str = "writeback",
        measurement_basis: str = "e2e_rebench",
        ts: str | None = None,
    ) -> None:
        """Update cumulative_gain_validated, its timestamp, and stack-length watermark.

        Call only when ``baseline_tput > 0`` and ``new_tput`` is a positive
        measured throughput.  The caller remains responsible for any surrounding
        guard (e.g. ``if self.shared_state.baseline_tput > 0``).

        Args:
            new_tput: The newly measured throughput to promote as the validated
                gain anchor.
            source: Which promotion path produced this figure, recorded so the
                breakdown can name it.
            measurement_basis: ``e2e_rebench`` when ``new_tput`` came from a
                full-stack revalidation, ``e2e_decision_round`` when it is the
                round an explore variant was graded on, ``derived_speedup``
                when it was inferred from a micro-benchmark.
            ts: Author-time stamp the caller already minted for this
                promotion; defaults to now.
        """
        validated_gain = (float(new_tput) - self.shared_state.baseline_tput) / self.shared_state.baseline_tput * 100.0
        ts = str(ts or datetime.now(timezone.utc).isoformat())
        self.shared_state.cumulative_gain_validated = float(validated_gain)
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(self.shared_state.optimization_stack)
        # The breakdown's own total is the sum of its ledger, so without this
        # record there is nothing for it to disagree with.
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_session_validation(
                self.session_dir,
                baseline_tput=float(self.shared_state.baseline_tput),
                validated_tput=float(new_tput),
                validated_gain_pct=float(validated_gain),
                stack_len=self.shared_state.cumulative_gain_validated_stack_len,
                source=source,
                measurement_basis=measurement_basis,
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("record_session_validation failed", exc_info=True)
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

    async def _record_integrate_keep(self, result: dict[str, Any]) -> None:
        """Promote a kernel integrate KEEP into the optimization stack.

        Stamps ``cumulative_gain_validated`` and fires a watermark roofline once
        the lift is accepted. No-op without a positive ``new_tput`` or when the
        lift refuses the winner.

        Args:
            result (dict[str, Any]): The integrate-patch executor result.
        """
        new_tput = result.get("new_tput")
        if not isinstance(new_tput, (int, float)) or new_tput <= 0:
            return
        lifted = self._lift_to_current_best(
            "integrate",
            float(new_tput),
            {
                "name": result.get("kernel_id"),
                "candidate_extra_server_args": result.get("extra_server_args"),
                "extra_envs": {str(k): str(v) for k, v in (result.get("extra_envs") or {}).items()},
                "source_phase": str(getattr(self.shared_state, "phase", "") or "KERNEL_AGENT"),
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
                "tpot_mean_ms": result.get("tpot_mean_ms"),
                "workspace": result.get("workspace"),
            },
            gap_canonical_id=str(result.get("gap_canonical_id") or "").strip(),
            entry_extra={
                "integration_id": result.get("integration_id"),
                "kernel_id": result.get("kernel_id"),
                "task_group_key": result.get("task_group_key"),
                "identity_route": result.get("identity_route"),
                "patch_path": result.get("patch_path"),
                "target_file": result.get("target_file"),
                "gain_pct": result.get("gain_pct"),
                "stack_kernel_ids": [str(k) for k in (result.get("stack_kernel_ids") or []) if str(k)],
            },
        )
        if lifted and self.shared_state.baseline_tput > 0:
            # Integrate KEEP is already rebench-validated: promote into cumulative_gain_validated + watermark.
            self._update_cumulative_gain_validated(new_tput)
            await self._maybe_enqueue_watermark_roofline(
                reason="integrate_keep_watermark",
            )

    def _is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
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
                :meth:`_promote_to_shared_state`.
        """
        if not isinstance(result, dict):
            return False
        if task_kind == "baseline":
            # A baseline whose accuracy eval failed measured throughput but must
            # not anchor; route it to _handle_unpromotable_result for enablement.
            if bool(result.get("baseline_eval_failed")):
                return False
            return is_valid_measurement(result)
        if task_kind == "profile":
            return is_valid_measurement(result)
        if task_kind == "sweep":
            return result.get("status") == "succeeded"
        # replay_warm_recipe always routes through _promote_warm_replay (owns its own failure bookkeeping).
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def _record_intervention_for_task(
        self,
        task: "Task",
        result: Any,
    ) -> None:
        """Log a completed task's change_type into SharedState.intervention_mix (explore → config; integrate_patch → code_patch_attempt or code_patch when kept). Best-effort.

        Args:
            task: The completed task whose kind selects the intervention class.
            result: The task result dict; non-dict results are ignored.
        """
        if not isinstance(result, dict):
            return
        kind = (task.kind or "").strip()
        if kind == "explore":
            # Winner surrogate: result.winners present OR best_variant set.
            winners = result.get("winners") or []
            best = result.get("best_variant")
            if not winners and not best:
                # An explore round that KEPT nothing still counts as a config-only attempt.
                self.shared_state.record_intervention(
                    change_type="config_attempt",
                    action="explore",
                    task_id=task.task_id,
                    delta_pct=None,
                )
                return
            delta_pct = None
            if isinstance(best, dict):
                delta_pct = best.get("gain_pct")
            self.shared_state.record_intervention(
                change_type="config",
                action="explore",
                task_id=task.task_id,
                delta_pct=delta_pct if isinstance(delta_pct, (int, float)) else None,
            )
            return
        if kind == "integrate_patch":
            status = str(result.get("status") or "").strip().lower()
            if not status:
                return
            if status != "kept":
                self.shared_state.record_intervention(
                    change_type="code_patch_attempt",
                    action="integrate_patch",
                    task_id=task.task_id,
                    delta_pct=result.get("delta_pct"),
                )
                return
            self.shared_state.record_intervention(
                change_type="code_patch",
                action="integrate_patch",
                task_id=task.task_id,
                delta_pct=result.get("delta_pct"),
            )

    def _persist_eval_failure(self, result_payload: dict[str, Any]) -> None:
        """Persist an eval-rooted baseline failure so enablement can re-run it.

        Records origin, floor, probe config, contract fingerprint, evidence,
        kind and observed accuracy, and seeds ``enablement_launch_log`` so the
        FRAMEWORK pump dispatches even when the failure carries no boot log.

        A run that never executed the eval characterizes nothing: it reports
        ``accuracy_unavailable`` with no task/metric/source. Such a run must
        still register as a failed round (origin / pending / stall accounting
        below), but it must NOT overwrite a stored trigger that actually
        measured an accuracy -- otherwise an eval-less re-baseline downgrades
        real ``accuracy_below_floor`` evidence to an empty
        ``accuracy_unavailable`` and the next enablement attempt loses the
        measurement it is supposed to reproduce. Contract fingerprints cannot
        gate this: ``RUN_EVAL`` is itself a contract field, so the eval-less
        run's fingerprint never matches the measured one.
        """
        state = self.shared_state
        was_validation_pending = bool(state.enablement.validation_pending)
        incoming_kind = result_payload.get(BASELINE_EVAL_FAILURE_KIND_KEY)
        measured_incoming = to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY)) is not None
        stored_kind = str(state.enablement.baseline_eval_kind or "")
        preserve_measured_trigger = (
            incoming_kind == EVAL_KIND_ACCURACY_UNAVAILABLE
            and not measured_incoming
            and bool(stored_kind)
            and stored_kind != EVAL_KIND_ACCURACY_UNAVAILABLE
        )
        state.enablement.origin = "eval"
        state.enablement.pending = True
        # A failed revalidation reopens the authoring loop and counts as a
        # no-progress round so the enablement_stalled cap can still terminate.
        if was_validation_pending:
            state.enablement.validation_pending = False
            state.enablement.stall_streak = int(state.enablement.stall_streak or 0) + 1
            if state.enablement.stall_streak >= _ENABLEMENT_MAX_STALL and not state.stop_reason:
                state.set_stop_reason("enablement_stalled")
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
            return
        cfg = result_payload.get("materialized_config")
        if isinstance(cfg, str) and cfg:
            state.enablement.probe_config_path = cfg
        fp = result_payload.get(BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY)
        if isinstance(fp, str) and fp:
            state.enablement.eval_contract_fingerprint = fp
        if isinstance(incoming_kind, str) and incoming_kind:
            state.enablement.baseline_eval_kind = incoming_kind
        observed = to_float(result_payload.get(BASELINE_EVAL_OBSERVED_ACCURACY_KEY))
        if observed is not None:
            state.enablement.observed_accuracy = float(observed)
        state.enablement.observed_task = str(result_payload.get("accuracy_task") or "")
        state.enablement.observed_metric = str(result_payload.get("accuracy_metric") or "")
        evidence = str(result_payload.get(BASELINE_EVAL_EVIDENCE_KEY) or "")
        if evidence:
            state.enablement.baseline_eval_evidence = evidence[:4000]
            state.enablement.launch_log = evidence

    def _reopen_revalidation_window(self) -> None:
        """Leave an enablement revalidation window open for a round the run stopped.

        A round the run stopped measured nothing, so it says nothing about whether
        the KEEP'd patch still revalidates. The window therefore stays open --
        only an eval-origin KEEP ever opens one, and closing it here would strand
        a patch nothing revalidated -- and the stall streak is not charged,
        because reaching the ``enablement_stalled`` cap on the evidence of a clock
        is exactly what the baseline failure streak already exempts this round
        from.

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
        state.enablement.inflight_task_id = ""

    def _record_revalidation_not_promoted(
        self,
        *,
        task: Task,
        result_payload: dict[str, Any],
        err_class: str,
        stopped_by_the_run: bool,
    ) -> None:
        """Close out an enablement revalidation baseline that did not promote.

        A genuine failure -- boot, OOM, timeout, eval -- is a no-progress round:
        it closes the revalidation window, reopens the authoring loop, and counts
        toward the ``enablement_stalled`` cap so repeated KEEP-then-fail cycles
        terminate.

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
        if stopped_by_the_run:
            self._reopen_revalidation_window()
        else:
            state.enablement.revalidation_task_id = ""
            state.enablement.validation_pending = False
            state.enablement.stall_streak = int(state.enablement.stall_streak or 0) + 1
            if state.enablement.stall_streak >= _ENABLEMENT_MAX_STALL and not state.stop_reason:
                state.set_stop_reason("enablement_stalled")
            else:
                state.enablement.inflight_task_id = ""
        launch_log = _extract_enablement_launch_log(result_payload)
        if launch_log:
            state.enablement.launch_log = launch_log
        log.warning(
            "enablement revalidation task %s %s (error_class=%s); stall_streak=%d pending=%s rearm=%s",
            task.task_id,
            "was stopped by the run" if stopped_by_the_run else "failed",
            err_class,
            int(state.enablement.stall_streak or 0),
            bool(state.enablement.validation_pending),
            not bool(state.stop_reason),
        )

    async def _handle_unpromotable_result(
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
        if task.kind in {"conc_sweep", "replay_warm_recipe", "integrate_patch"}:
            try:
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                result_payload.setdefault("workload", self._v4_workload())
                instrument.record_action_operation(
                    self.session_dir,
                    action=task.kind,
                    task_id=task.task_id,
                    status="failed",
                    decision="discarded",
                    result=result_payload,
                    phase=str(getattr(self.shared_state, "phase", "") or ""),
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    tick=int(getattr(self.shared_state, "tick", 0) or 0),
                )
            except Exception:  # noqa: BLE001
                log.debug("v4 action failure capture failed", exc_info=True)
        any_changed = False
        params = task.params or {}
        if task.kind == "explore" and bool(params.get("geak_fallback")):
            from ..phases.geak_rebench import geak_rebench_should_apply_result

            if geak_rebench_should_apply_result(
                self.shared_state,
                task,
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
            ):
                geak_result = (
                    dict(self.shared_state.geak_result)
                    if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                    else {}
                )
                geak_result["revalidation_status"] = "failed"
                geak_result["revalidation_error_class"] = str(result_payload.get("error_class") or "")
                geak_result["revalidation_error"] = str(
                    result_payload.get("error") or result_payload.get("reason") or ""
                )[:500]
                self.shared_state.geak_result = geak_result
                # ``geak_pending`` is a live-work slot, not a diagnostic
                # archive.  Keeping a terminal failure here prevents the
                # KERNEL -> SWEEP transition forever.  The settled verdict and
                # its diagnostics survive in ``geak_result`` instead.
                self.shared_state.geak_pending = {}
                self.shared_state.resume_pending_revalidation = False
                any_changed = True
        # Per-action audit (failed attempt) for the in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp baseline-params fingerprint for the self-loop denial helper.
            if task.kind == "baseline":
                audit_extras["fingerprint"] = _baseline_params_fingerprint(task.params)
            self.shared_state.record_action_attempt(
                action=task.kind,
                task_id=task.task_id,
                status="failed",
                decision="no_promote",
                result=result_payload,
                extras=audit_extras,
            )
            any_changed = True
        # Global rolling failure log (every kind, including kernel_agent-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        any_changed = True
        if task.kind == "conc_sweep":
            self.shared_state.record_action_attempt(
                action="conc_sweep",
                task_id=task.task_id,
                status=str(result_payload.get("status") or "failed"),
                decision="discarded",
                result=result_payload,
                extras={
                    "was_skipped": bool(result_payload.get("was_skipped", False)),
                    "skip_reason": result_payload.get("skip_reason"),
                    "budget_exhausted": bool(result_payload.get("budget_exhausted", False)),
                    "total_budget_sec": result_payload.get("total_budget_sec"),
                    "elapsed_sec": result_payload.get("elapsed_sec"),
                    "best_speedup": ((result_payload.get("summary") or {}).get("best_speedup")),
                    "best_conc": ((result_payload.get("summary") or {}).get("best_conc")),
                    "successful_pairs": ((result_payload.get("summary") or {}).get("successful_pairs")),
                    "report_path": result_payload.get("report_json_path"),
                },
            )
            self.shared_state.record_conc_sweep(result_payload)
        # An upstream-PR candidate task that settles failed/empty never reaches
        # the promote branch that writes the terminal progress row; stamp
        # no_result_failed so the pump does not re-select it every tick.
        if task.kind == "integrate_patch" and (task.params or {}).get("framework_agent_candidate_id"):
            cand = (task.params or {}).get("candidate")
            cand_id = self._framework_candidate_key(cand if isinstance(cand, dict) else None)
            if cand_id:
                self._stamp_framework_progress(
                    candidate_id=cand_id,
                    batch_id=str((task.params or {}).get("batch_id") or ""),
                    status="no_result_failed",
                    kept=False,
                    rationale=str(result_payload.get("reason") or result_payload.get("error") or "")[:500],
                    provenance="executor",
                    extra={"status": str(result_payload.get("status") or "")},
                )
        # Baseline-specific gates: streak counter + stop_reason + baseline_not_promoted event.
        # Fast arg errors get their own streak so they don't burn the
        # slow-baseline retry budget on deterministic failures.
        baseline_event_payload: dict[str, Any] | None = None
        # Only arm/streak while no baseline has succeeded yet (tput <= 0).
        if task.kind == "baseline" and self.shared_state.baseline_tput <= 0:
            err_class = result_payload.get("error_class", "")
            # A round the run itself stopped -- the session budget reaped it, or
            # the orchestrator cancelled the action -- measured nothing, so it
            # says nothing about whether this baseline boots. Charging it to the
            # streaks would let three stops the run chose end the session as
            # ``baseline_failed``, blaming the model for the clock. The executor
            # already refuses to grade such a round; the ledger has to agree.
            stopped_by_the_run = stopped_by_the_run_class(err_class) is not None
            # While a serial enablement is actively engaged, baseline boots
            # re-fail on purpose (each round clears a deeper gap), so the
            # ``baseline_failed`` fast-fail must NOT fire here; the
            # ``enablement_stalled`` cap is the correct fast-fail instead.
            # ``fast_exit_arg_error`` stays gated on its own streak regardless.
            from ..phases.machine_state import enablement_engaged as _enablement_engaged  # noqa: PLC0415

            enablement_engaged = _enablement_engaged(self.shared_state)
            eval_failed = bool(result_payload.get(BASELINE_EVAL_FAILED_KEY))
            # Revalidation task failed for any reason (boot/OOM/timeout/eval): clear
            # pending state, preserve the frozen trigger identity, increment stall.
            reval_tid = str(getattr(self.shared_state.enablement, "revalidation_task_id", "") or "").strip()
            is_revalidation = bool(
                (task.params or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON
                or (reval_tid and reval_tid == str(task.task_id or ""))
            )
            if is_revalidation and bool(getattr(self.shared_state.enablement, "validation_pending", False)):
                self._record_revalidation_not_promoted(
                    task=task,
                    result_payload=result_payload,
                    err_class=err_class,
                    stopped_by_the_run=stopped_by_the_run,
                )
                any_changed = True
            from ..actions.executors._accuracy_gate import eval_enablement_allowed  # noqa: PLC0415
            from ..actions.executors._multi_node_env import is_multi_node  # noqa: PLC0415

            # Single-node eval-pending failure: throughput measured fine and the
            # eval is expected to re-run under enablement, so do not spend the
            # baseline_failed budget yet. Multi-node keeps the strict backstop,
            # and so does a session that never admitted the eval lane — nothing
            # would re-run the eval, so holding the budget just stalls the run.
            eval_pending_suppress = eval_failed and not is_multi_node() and eval_enablement_allowed(self.shared_state)
            if eval_failed:
                self._persist_eval_failure(result_payload)
            if stopped_by_the_run:
                log.warning(
                    "baseline %s was stopped by the run (%s); the failure streak stays at %d "
                    "because nothing about the baseline was measured",
                    task.task_id,
                    err_class,
                    self.shared_state.baseline_failure_streak,
                )
            elif err_class == "fast_exit_arg_error":
                self.shared_state.baseline_arg_error_streak += 1
                if self.shared_state.baseline_arg_error_streak >= 2:
                    self.shared_state.set_stop_reason("baseline_arg_error")
            else:
                self.shared_state.baseline_failure_streak += 1
                self.shared_state.baseline_arg_error_streak = 0
                if (
                    self.shared_state.baseline_failure_streak >= 3
                    and not enablement_engaged
                    and not eval_pending_suppress
                ):
                    self.shared_state.set_stop_reason("baseline_failed")
            # Combined backstop: count ALL baseline failures so mixed
            # error_classes that split the per-class streaks still fast-fail.
            if not stopped_by_the_run:
                self.shared_state.baseline_total_failures += 1
            if (
                self.shared_state.baseline_total_failures >= _BASELINE_MAX_TOTAL_FAILURES
                and not self.shared_state.stop_reason
                and not enablement_engaged
                and not eval_pending_suppress
            ):
                self.shared_state.set_stop_reason("baseline_failed")
            # One-shot eager fallback: a (non-OOM) cuda-graph capture failure is
            # often recoverable by disabling cuda-graph capture.
            if err_class == "cuda_graph_capture_failed" and not self.shared_state.baseline_eager_fallback:
                self.shared_state.baseline_eager_fallback = True
                log.warning(
                    "baseline %s hit cuda-graph capture failure; arming "
                    "disable-cuda-graph fallback for the next baseline retry",
                    task.task_id,
                )
            # Stash the launch/traceback text for the FRAMEWORK pump (fast arg errors excluded).
            if err_class != "fast_exit_arg_error":
                launch_log = _extract_enablement_launch_log(result_payload)
                if launch_log:
                    self.shared_state.enablement.launch_log = launch_log
            baseline_event_payload = {
                "kind": "baseline_not_promoted",
                "task_id": task.task_id,
                "failure_streak": self.shared_state.baseline_failure_streak,
                "arg_error_streak": self.shared_state.baseline_arg_error_streak,
                "stop_reason": self.shared_state.stop_reason,
                "result_status": result_payload.get("status"),
                "error_class": err_class,
                "baseline_eval_failed": eval_failed,
            }
            any_changed = True
        # Mirror the promote-path roofline failure handling: bump streak, clear gate, warn.
        if task.kind == "roofline":
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            if self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
            any_changed = True
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
        if any_changed:
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

    def _source_session_id(self) -> str:
        """Return the hyperloom-local session id used as source_session_id on KB fact writes.

        NOT a KB-side session id; prefers recipe_kb_session_id, falls back to session_dir.name.

        Returns:
            The hyperloom-local session id (recipe_kb_session_id when set, else
            ``session_dir.name``).
        """
        return str(getattr(self.shared_state, "recipe_kb_session_id", "") or "") or self.session_dir.name

    async def _fact_write_hook(
        self,
        *,
        task: "Task",
        result: Any,
        kept: bool,
    ) -> None:
        """Per-task fact-write entry point (per_variant for explore grids, else per-task); best-effort, never raises.

        Args:
            task: The completed task being recorded.
            result: The task's :class:`SubAgentResult` (or result dict).
            kept: Whether the task's result was KEEP-promoted.
        """
        result_dict = result.result if hasattr(result, "result") else (result or {})
        if not isinstance(result_dict, dict):
            result_dict = {}
        source_session_id = self._source_session_id()
        per_variant = result_dict.get("per_variant_outcomes")
        if task.kind == "explore" and isinstance(per_variant, list) and per_variant:
            for vo in per_variant:
                try:
                    self._record_fact_per_variant(
                        task=task,
                        source_session_id=source_session_id,
                        variant_outcome=vo,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "fact-write per-variant failed (task=%s)",
                        task.task_id,
                    )
        else:
            try:
                self._record_fact_per_task(
                    task=task,
                    source_session_id=source_session_id,
                    result_dict=result_dict,
                    kept=kept,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "fact-write per-task failed (task=%s)",
                    task.task_id,
                )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive; never crash on save
            log.exception("fact-write SharedState.save failed")

    def _ensure_journal(self) -> Journal:
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume).

        Returns:
            The per-session :class:`Journal` instance (created on first call,
            with the baseline backfilled on subsequent calls).
        """
        existing = getattr(self, "_journal", None)
        if existing is None:
            ss = self.shared_state
            self._coord._journal = Journal.load_or_create(
                self.session_dir,
                session_id=str(getattr(ss, "recipe_kb_session_id", "") or "")
                or str(getattr(ss, "session_id", "") or "")
                or self.session_dir.name,
                model=str(getattr(ss, "model_name", "") or ""),
                hardware=str(getattr(ss, "gpu_type", "") or ""),
                framework=str(getattr(ss, "framework", "") or ""),
                baseline_throughput=float(getattr(ss, "baseline_tput", 0.0) or 0.0),
            )
        else:
            # Backfill baseline once the baseline executor finishes.
            existing.update_baseline(float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0))
        return self._journal

    def _pitfall_severity_for(
        self,
        result_dict: dict[str, Any] | None,
    ) -> str | None:
        """Decide whether a failed result warrants a pitfall row.

        ``crash`` / ``oom`` / ``hang`` / ``detokenizer_stall`` on ``error_class``,
        or ``crash`` / ``oom`` / ``hang`` on ``status``, yield
        ``SEVERITY_CRASH``; a ``gain_pct`` at or below
        ``PITFALL_REGRESS_THRESHOLD_PCT`` (-5.0) yields ``SEVERITY_REGRESS``;
        otherwise ``None``.

        Args:
            result_dict: The failed task's result dict; non-dict yields ``None``.

        Returns:
            The pitfall severity (``SEVERITY_CRASH`` / ``SEVERITY_REGRESS``), or
            ``None`` when no pitfall is warranted.
        """
        if not isinstance(result_dict, dict):
            return None
        error_class = str(result_dict.get("error_class") or "").lower()
        # ``detokenizer_stall`` is a hang in all but name; record it as a
        # crash-severity pitfall so the offending config is not re-proposed.
        if error_class in ("crash", "oom", "hang", "detokenizer_stall"):
            return _SEVERITY_CRASH
        status = str(result_dict.get("status") or "").lower()
        if status in ("crash", "oom", "hang"):
            return _SEVERITY_CRASH
        gain = result_dict.get("gain_pct")
        try:
            gain_pct = float(gain) if gain is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        if gain_pct is not None and gain_pct <= self.PITFALL_REGRESS_THRESHOLD_PCT:
            return _SEVERITY_REGRESS
        return None

    def _journal_entry_phase(self) -> str:
        """Return the current phase label for journal entries.

        Returns:
            str: The uppercased phase name, or ``"UNKNOWN"`` when unset.
        """
        return str(getattr(self.shared_state, "phase", "") or "").strip().upper() or "UNKNOWN"

    def _record_fact_impl(
        self,
        *,
        task: "Task",
        source_session_id: str,
        is_keep: bool,
        change: str,
        gain_pct: float | None,
        throughput_after: float | None,
        best_config_candidate: dict[str, Any] | None,
        evidence_refs: list[str],
        pitfall_severity_dict: dict[str, Any],
        variant_name: str | None = None,
    ) -> None:
        """Shared KB write for _record_fact_per_task and _record_fact_per_variant.

        Writes one KB lesson (on KEEP with positive gain) or one KB pitfall
        (on REVERT/failure) to the recipe row, then returns.  Call only after
        the journal entry has been appended and ``recipe_kb`` is confirmed
        non-None by the caller.

        Args:
            task: The completed task (provides task_id).
            source_session_id: Hyperloom-local session id stamped on provenance.
            is_keep: True when the outcome is a validated KEEP.
            change: Summarized change string (used in the statement).
            gain_pct: Measured gain percentage, or ``None``.
            throughput_after: Measured throughput after the change, or ``None``.
            best_config_candidate: Pre-extracted best-config dict (differs
                between per-task and per-variant callers).
            evidence_refs: List of evidence reference strings to stamp on the
                provenance (caller builds task-only or task+variant refs).
            pitfall_severity_dict: The dict passed to ``_pitfall_severity_for``
                (per-task passes ``result_dict``; per-variant passes a merged
                metrics + outcome dict).
            variant_name: Variant name, present only for per-variant calls;
                added to ``provenance_details`` when non-None.
        """
        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        workload_tags = self._coord._collect_workload_tags()
        extra = workload_tags if workload_tags else None
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        provenance_base: dict[str, Any] = {
            "source_session_id": source_session_id,
            "source_task_id": task.task_id,
            "evidence": list(evidence_refs or []),
            "applicable_models": list(models or []),
            "applicable_hardware": list(hardware or []),
            "extra": dict(extra or {}),
            "now": now_iso,
        }
        if variant_name is not None:
            provenance_base["source_variant_name"] = variant_name

        if is_keep and gain_pct is not None and gain_pct > 0:
            statement = self._coord._build_statement(
                change=change,
                kind="lesson",
            )
            impact = self._coord._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(getattr(self.shared_state, "optimization_stack", []) or []),
                measured_at=now_iso,
            )
            live = self._read_local_recipe_row()
            recipe_overrides = self._kb_best_config_overrides_for_keep(
                live=live,
                best_config_candidate=best_config_candidate,
                throughput_after=throughput_after,
            )
            self._kb_amend_recipe(
                append_lesson={
                    "statement": statement,
                    "measured_impact": impact,
                },
                recipe_overrides=recipe_overrides or None,
                provenance_details=provenance_base,
            )
            return

        severity = self._pitfall_severity_for(pitfall_severity_dict)
        if severity is not None:
            description = self._coord._build_statement(
                change=change,
                severity=severity,
                kind="pitfall",
            )
            self._kb_amend_recipe(
                append_pitfall={
                    "description": description,
                    "severity": severity,
                },
                provenance_details=provenance_base,
            )

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact (source_session_id is hyperloom-local).

        Args:
            task: The completed task being recorded.
            source_session_id: The hyperloom-local session id stamped on the
                fact provenance.
            result_dict: The task result dict.
            kept: Whether the result was KEEP-promoted (KEEP → lesson, else
                pitfall/REVERT).
        """
        journal = self._ensure_journal()
        # integrate_patch reports its delta under ``delta_pct``;
        # fall back to it so a reverted/kept patch shows its REAL measured delta
        # in the journal instead of a null gain.
        gain_pct = to_float(result_dict.get("gain_pct"))
        if gain_pct is None:
            gain_pct = to_float(result_dict.get("delta_pct"))
        throughput_after = to_float(result_dict.get("output_throughput"))
        kind = classify_change_kind(task.kind, None)
        change = summarize_change(task.kind, None, result_dict)
        # Journal outcome follows the executor's per-status verdict for source-
        # patch kinds (a ``reverted`` patch is promotable but NOT a KEEP); other
        # kinds keep the binary promotable→KEEP behaviour. See
        # ``derive_journal_outcome`` (fixes the "fake KEEP" bug).
        outcome = derive_journal_outcome(task.kind, result_dict, promotable=kept)
        is_keep = outcome == OUTCOME_KEEP
        if is_keep:
            error_class = None
            reason = None
        else:
            error_class = str(result_dict.get("error_class") or "") or None
            # A skip states its own cause under ``skip_reason``; without it the
            # timeline shows a step that did nothing and never says why.
            reason = str(result_dict.get("reason") or result_dict.get("skip_reason") or "") or None
        journal.append_entry(
            JournalEntry(
                phase=self._journal_entry_phase(),
                lever_kind=_lever_kind_for_lift(kind, result_dict if isinstance(result_dict, dict) else None),
                iter=int(self.shared_state.tick or 0),
                kind=kind,
                change=change,
                outcome=outcome,
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                error_class=error_class,
                reason=reason,
                task_id=task.task_id,
                tick=int(self.shared_state.tick or 0),
                predicted_gain_pct=_predicted_gain(
                    result_dict,
                    getattr(task, "params", None),
                ),
            )
        )

        if self.recipe_kb is None:
            return

        self._record_fact_impl(
            task=task,
            source_session_id=source_session_id,
            is_keep=is_keep,
            change=change,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            best_config_candidate=self._extract_kept_best_config(
                task=task,
                result_dict=result_dict,
            ),
            # evidence_refs (log:task-...) gives traceability since source_session_id lands in attrs.
            evidence_refs=[f"log:task-{task.task_id}"],
            pitfall_severity_dict=result_dict,
        )

    def _build_statement(
        self,
        *,
        change: str,
        kind: str,
        severity: str | None = None,
    ) -> str:
        """Build the lesson statement / pitfall description hashed into the KB canonical_id; MUST exclude volatile fields (e.g. gain_pct) so N sessions merge instead of producing N rows. Identity = framework + change + model/hw.

        Args:
            change: The summarized change description.
            kind: ``"lesson"`` or ``"pitfall"`` — selects the rendered form.
            severity: The pitfall severity, rendered only when ``kind`` is
                ``"pitfall"``.

        Returns:
            The identity-stable statement / description string.
        """
        framework = str(getattr(self.shared_state, "framework", "") or "").strip()
        fw_tag = f"[{framework or '?'}] "
        model = self.shared_state.model_name or "?"
        hw = self.shared_state.gpu_type or "?"
        if kind == "lesson":
            return f"{fw_tag}{change} on {model}/{hw}"
        # kind == "pitfall"
        return f"{fw_tag}{change} → {severity or '?'} on {model}/{hw}"

    @staticmethod
    def _build_measured_impact(
        *,
        gain_pct: float | None,
        throughput_after: float | None,
        stack_depth: int,
        measured_at: str,
    ) -> dict[str, Any]:
        """Structured ``measured_impact`` payload (dict not legacy string so consumers parse without regex); stack_depth = stack length before this lesson lands.

        Args:
            gain_pct: The measured gain percent, or ``None``.
            throughput_after: Throughput after the change, or ``None``.
            stack_depth: Optimization-stack length before this lesson lands.
            measured_at: ISO timestamp of the measurement.

        Returns:
            A compact ``measured_impact`` dict with ``None`` fields stripped.
        """
        out: dict[str, Any] = {
            "gain_pct": float(gain_pct) if gain_pct is not None else None,
            "stack_depth_at_apply": int(stack_depth),
            "measured_at": measured_at,
        }
        if throughput_after is not None:
            out["throughput_after"] = float(throughput_after)
        # Strip None for compactness (prompt section uses .get).
        return {k: v for k, v in out.items() if v is not None}

    def _record_fact_per_variant(
        self,
        *,
        task: "Task",
        source_session_id: str,
        variant_outcome: dict[str, Any],
    ) -> None:
        """Per-variant fact write — mirror of _record_fact_per_task for explore per-variant decisions.

        Args:
            task: The completed explore task.
            source_session_id: The hyperloom-local session id stamped on the
                fact provenance.
            variant_outcome: One per-variant outcome row (name, outcome,
                metrics).
        """
        journal = self._ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw == "KEEP":
            outcome = OUTCOME_KEEP
        # ``KEEP_UNSTABLE`` is only reachable for a session recorded before the
        # per-KEEP confirmation round was removed; it still reads as a revert.
        elif outcome_raw in ("REVERT", "FAILED", "KEEP_UNSTABLE"):
            outcome = OUTCOME_REVERT
        elif outcome_raw == "SKIPPED_DEDUP":
            return  # nothing to journal
        else:
            outcome = OUTCOME_NO_PROMOTE
        variant_name = str(variant_outcome.get("variant_name") or "")
        metrics = variant_outcome.get("metrics") or {}
        gain_pct = to_float(metrics.get("gain_pct") if isinstance(metrics, dict) else None)
        throughput_after = to_float(metrics.get("output_throughput") if isinstance(metrics, dict) else None)
        variant_attrs = variant_outcome.get("variant") or {}
        kind = classify_change_kind(
            task.kind,
            variant_attrs if isinstance(variant_attrs, dict) else None,
        )
        # Ensure the change summary is variant-specific (else every explore variant writes an identical row).
        change_attrs = dict(variant_attrs) if isinstance(variant_attrs, dict) else {}
        if (
            not (change_attrs.get("extra_server_args") or change_attrs.get("extra_envs") or change_attrs.get("name"))
            and variant_name
        ):
            change_attrs["name"] = variant_name
        change = summarize_change(task.kind, change_attrs, None)
        error_class = None
        reason = None
        if outcome == OUTCOME_REVERT:
            error_class = str(variant_outcome.get("error_class") or "") or None
            reason = str(variant_outcome.get("reason") or "") or None
        # Proposer attribution + per-variant measurement detail, carried from the
        # explore executor's per_variant_outcomes so the decision row records who
        # proposed the change and how it measured (beyond headline gain/tput).
        detail_metrics = {
            k: metrics[k]
            for k in (
                "runtime_sec",
                "wall_clock_ratio_vs_baseline",
                "estimated_output_throughput",
            )
            if isinstance(metrics, dict) and metrics.get(k) is not None
        }
        journal.append_entry(
            JournalEntry(
                phase=self._journal_entry_phase(),
                lever_kind=_lever_kind_for_lift(kind, variant_outcome if isinstance(variant_outcome, dict) else None),
                iter=int(self.shared_state.tick or 0),
                kind=kind,
                change=change,
                outcome=outcome,
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                error_class=error_class,
                reason=reason,
                task_id=task.task_id,
                variant_name=variant_name,
                provenance=str(variant_outcome.get("provenance") or ""),
                scope=str(variant_outcome.get("scope") or ""),
                fingerprint=str(variant_outcome.get("fingerprint") or ""),
                metrics=detail_metrics,
                tick=int(self.shared_state.tick or 0),
                predicted_gain_pct=_predicted_gain(
                    variant_outcome,
                    variant_attrs if isinstance(variant_attrs, dict) else None,
                    getattr(task, "params", None),
                ),
            )
        )

        if self.recipe_kb is None:
            return

        self._record_fact_impl(
            task=task,
            source_session_id=source_session_id,
            is_keep=(outcome == OUTCOME_KEEP),
            change=change,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            best_config_candidate=self._extract_kept_best_config(
                task=task,
                variant_attrs=change_attrs,
            ),
            # Workload-shape tags — see _record_fact_per_task.
            evidence_refs=[f"log:task-{task.task_id}", f"variant:{variant_name}"],
            pitfall_severity_dict={
                **(metrics if isinstance(metrics, dict) else {}),
                "error_class": variant_outcome.get("error_class"),
                "status": variant_outcome.get("outcome"),
            },
            variant_name=variant_name,
        )

    def _collect_workload_tags(self) -> dict[str, Any]:
        """Return the workload-shape KB tag dict for the current session; shared by recipe attrs + lesson/pitfall writes so the warm-start reader filters symmetrically.

        Returns:
            A dict of workload-shape KB tags (framework, model, parallelism,
            runtime versions, baseline workload extras) with empty values
            omitted.
        """
        ss = self.shared_state
        out: dict[str, Any] = {}
        framework = str(getattr(ss, "framework", "") or "").strip()
        if framework:
            out["framework"] = framework
        model_class = str(getattr(ss, "model_class", "") or "").strip()
        if model_class:
            out["model_class"] = model_class
        # model_family is not part of the seven-dimension Recipe identity.
        model_name = str(getattr(ss, "model_name", "") or "").strip()
        if model_name:
            out["model_name"] = model_name
        for src_attr, dst_key in (
            ("precision", "precision"),
            ("tp", "tp"),
            ("ep", "ep"),
            ("conc", "conc"),
            ("isl", "isl"),
            ("osl", "osl"),
            ("max_model_len", "max_model_len"),
        ):
            v = getattr(ss, src_attr, None)
            if v not in (None, "", 0):
                out[dst_key] = v
        # EP env fallback when SharedState.ep is unset (legacy SDK callers).
        if "ep" not in out:
            raw_ep = (os.environ.get("EP") or "").strip()
            try:
                n = int(raw_ep) if raw_ep else 0
            except ValueError:
                n = 0
            if n > 0:
                out["ep"] = n
        # PP — no SharedState field (no CLI surface); env-only.
        raw_pp = (os.environ.get("PP") or "").strip()
        try:
            pp_n = int(raw_pp) if raw_pp else 0
        except ValueError:
            pp_n = 0
        if pp_n > 0:
            out["pp"] = pp_n
        # runtime version tags from stack_fingerprint_meta (cli writes at boot, resume reads verbatim).
        fp_meta = getattr(ss, "stack_fingerprint_meta", None) or {}
        if isinstance(fp_meta, dict):
            # framework_version is whichever of sglang/vllm is active.
            fw_lc = framework.lower()
            if fw_lc in ("sglang", "vllm"):
                v = str(fp_meta.get(fw_lc) or "").strip()
                if v and v != "unknown":
                    out["framework_version"] = v
            for src_key, dst_key in (
                ("rocm", "rocm_version"),
                ("aiter", "aiter_version"),
                ("image_digest", "image_digest"),
            ):
                v = str(fp_meta.get(src_key) or "").strip()
                if v and v != "unknown":
                    out[dst_key] = v
        # per-baseline workload extras from materialized YAML; keep bool False (don't drop an "explicitly disabled" signal).
        wl_extra = getattr(ss, "baseline_workload_extra", None) or {}
        if isinstance(wl_extra, dict):
            for k in ("max_running_requests", "max_num_seqs"):
                v = wl_extra.get(k)
                if isinstance(v, int) and v > 0:
                    out[k] = v
            for k in ("chunked_prefill_enabled", "enable_torch_compile"):
                v = wl_extra.get(k)
                if isinstance(v, bool):
                    out[k] = v
            for k in ("quant_scheme", "workload_mode"):
                v = wl_extra.get(k)
                if isinstance(v, str) and v.strip():
                    out[k] = v.strip()
        return out

    def _build_kernel_optimizations_from_state(self) -> list[dict[str, Any]]:
        """Collect KEEP'd kernel optimizations + their E2E verdict by joining kernel_opt_task_attempts (micro) and kernel_integrate_attempts (E2E) on kernel_id; non-integrated KEEPs surface integrated=False. Returns KernelOptimization-shaped dicts.

        Returns:
            A list of KernelOptimization-shaped dicts for each KEEP'd kernel,
            joined with its E2E integrate verdict where available.
        """
        ss = self.shared_state
        opt_attempts = getattr(ss, "kernel_opt_task_attempts", {}) or {}
        integ_attempts = getattr(ss, "kernel_integrate_attempts", {}) or {}
        if not isinstance(opt_attempts, dict):
            return []

        # Index integrate results by kernel_id (last write wins; entry carries rolled-up best_gain_pct).
        integ_by_kid: dict[str, dict[str, Any]] = {}
        integ_by_task: dict[str, dict[str, Any]] = {}
        if isinstance(integ_attempts, dict):
            for entry in integ_attempts.values():
                if not isinstance(entry, dict):
                    continue
                kid = str(entry.get("kernel_id") or "")
                if kid:
                    integ_by_kid[kid] = entry
                task_group_key = str(entry.get("task_group_key") or "")
                if task_group_key:
                    integ_by_task[task_group_key] = entry

        out: list[dict[str, Any]] = []
        for ledger_id, e in opt_attempts.items():
            if not isinstance(e, dict):
                continue
            if str(e.get("last_decision", "")).upper() != "KEEP":
                continue
            try:
                micro = float(e.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            kid = str(e.get("current_kernel_id") or e.get("kernel_id") or ledger_id)
            task_group_key = str(e.get("task_group_key") or "")
            integ = integ_by_task.get(task_group_key) if task_group_key else integ_by_kid.get(kid)
            e2e_gain = 0.0
            e2e_tput = 0.0
            e2e_decision = ""
            integrated = False
            if isinstance(integ, dict):
                integrated = True
                # Integrate-layer verdict (E2E); lets warm-start skip a micro-win/E2E-loss kernel.
                e2e_decision = str(integ.get("last_decision") or "").upper()
                try:
                    e2e_gain = float(integ.get("best_gain_pct") or 0.0)
                except (TypeError, ValueError):
                    e2e_gain = 0.0
                # Last attempt's E2E re-bench throughput.
                for att in reversed(list(integ.get("attempts") or [])):
                    if isinstance(att, dict) and att.get("new_tput") is not None:
                        try:
                            e2e_tput = float(att.get("new_tput") or 0.0)
                        except (TypeError, ValueError):
                            e2e_tput = 0.0
                        break
            out.append(
                {
                    "kernel_id": kid,
                    "source_file": str(e.get("last_source_file") or ""),
                    "artifact_path": str(e.get("last_artifact_path") or ""),
                    "micro_speedup": micro,
                    "decision": "KEEP",
                    "e2e_gain_pct": e2e_gain,
                    "e2e_tput": e2e_tput,
                    "e2e_decision": e2e_decision,
                    "integrated": integrated,
                    "ts": str(e.get("last_ts") or ""),
                }
            )
        return out

    def _collect_attempt_provenance(
        self,
    ) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
        """Map proven optimizations to their research-hint origin from the gaps[] attempts ledger; returns (kept_sources by name/kernel, kept_by_gap by canonical_id, reverted_rows). Fail-soft.

        Returns:
            A ``(kept_sources, kept_by_gap, reverted_rows)`` tuple: KEEP'd
            provenance keyed by variant/kernel name, KEEP'd provenance keyed by
            gap canonical_id, and reverted-attempt rows.
        """
        kept_sources: dict[str, str] = {}
        kept_by_gap: dict[str, str] = {}
        reverted_rows: list[dict[str, Any]] = []
        gaps = getattr(self.shared_state, "gaps", []) or []
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            provenance = str(gap.get("provenance") or "").strip()
            canonical = str(gap.get("canonical_id") or "").strip()
            for attempt in gap.get("attempts") or []:
                if not isinstance(attempt, dict):
                    continue
                variant = str(attempt.get("variant_name") or "").strip()
                kernel = str(attempt.get("kernel_id") or "").strip()
                outcome = str(attempt.get("outcome") or "").strip().upper()
                if outcome == "KEEP" and provenance:
                    if variant:
                        kept_sources.setdefault(variant, provenance)
                    if kernel:
                        kept_sources.setdefault(kernel, provenance)
                    if canonical:
                        kept_by_gap.setdefault(canonical, provenance)
                elif outcome == "REVERT" and (variant or kernel):
                    row: dict[str, Any] = {
                        "name": variant or kernel,
                        "reason": "reverted",
                        "gain_pct": attempt.get("gain_pct"),
                    }
                    if provenance:
                        row["source"] = provenance
                    reverted_rows.append(row)
        return kept_sources, kept_by_gap, reverted_rows

    def _build_recipe_attrs_from_state(self) -> dict[str, Any]:
        """Materialise the recipe-shaped view of :class:`SharedState` (defensive getattr).

        Returns:
            A recipe-shaped attrs dict (best_config, what_worked, what_failed,
            kernel_optimizations, workload tags, session row) for KB recipe
            writes.
        """
        ss = self.shared_state
        current_best = getattr(ss, "current_best", {}) or {}
        opt_stack = getattr(ss, "optimization_stack", []) or []
        gain_per_stack = getattr(ss, "gain_per_stack_entry", []) or []
        last_failures = getattr(ss, "last_action_failures", []) or []
        # RecipeKB best_config keys on the canonical extra_server_args field.
        best_config: dict[str, Any] = {}
        if isinstance(current_best, dict):
            cb_args = current_best.get("extra_server_args")
            if cb_args:
                best_config["extra_server_args"] = str(cb_args)
            for key in ("extra_envs", "name", "tput", "accuracy"):
                if key in current_best:
                    best_config[key] = current_best[key]
        # Prefer the last validated stack layer for launch args (current_best may carry a corrupted string).
        if opt_stack:
            last_entry = opt_stack[-1]
            if isinstance(last_entry, dict):
                stack_args = str(
                    last_entry.get("candidate_extra_server_args") or last_entry.get("extra_server_args") or "",
                ).strip()
                if stack_args:
                    best_config["extra_server_args"] = stack_args
        sediment_on = bool(getattr(ss, "recipe_sediment_enabled", True))
        kept_sources, kept_by_gap, reverted_rows = self._collect_attempt_provenance() if sediment_on else ({}, {}, [])
        what_worked: list[dict[str, Any]] = []
        for idx, entry in enumerate(opt_stack):
            if not isinstance(entry, dict):
                continue
            gain_per: float | None = None
            if idx < len(gain_per_stack):
                gain_per = gain_per_stack[idx]
            name = str(entry.get("variant_name") or entry.get("name") or entry.get("kernel_id") or "")
            row: dict[str, Any] = {
                "name": name,
                "extra_server_args": str(entry.get("extra_server_args") or ""),
                "extra_envs": dict(entry.get("extra_envs") or {}),
                "gain_pct": gain_per,
            }
            # Prefer the entry's gap-id provenance (naming-independent); fall back to name/kernel_id match.
            entry_gap = str(entry.get("gap_canonical_id") or "").strip()
            src = (
                (kept_by_gap.get(entry_gap) if entry_gap else None)
                or kept_sources.get(name)
                or kept_sources.get(str(entry.get("kernel_id") or ""))
            )
            if src:
                row["source"] = src
            what_worked.append(row)
        what_failed: list[dict[str, Any]] = []
        for failure in last_failures[-10:]:
            if isinstance(failure, dict):
                what_failed.append(
                    {
                        "name": str(failure.get("name") or failure.get("action") or ""),
                        "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                    }
                )
        for rev in reverted_rows:
            what_failed.append(rev)
        kernel_optimizations = self._coord._build_kernel_optimizations_from_state()
        cumulative_validated = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
        validated_stack_len = int(getattr(ss, "cumulative_gain_validated_stack_len", 0) or 0)
        # Workload-shape tags for shape-filtered warm-start queries (shared via _collect_workload_tags).
        workload_tags = self._coord._collect_workload_tags()
        # framework_version left unset here (manifest-derived); the T0 backfill writes it.
        return {
            "best_config": best_config,
            "best_throughput": float(current_best.get("tput", 0.0)) if isinstance(current_best, dict) else 0.0,
            "what_worked": what_worked,
            "what_failed": what_failed,
            "kernel_optimizations": kernel_optimizations,
            "last_profiled": str(getattr(ss, "cumulative_gain_validated_ts", "") or ""),
            "workload": workload_tags,
            "sessions": [
                {
                    "session_id": str(getattr(ss, "recipe_kb_session_id", "") or self.session_dir.name),
                    "gain_pct": cumulative_validated,
                    "stack_len": validated_stack_len or len(opt_stack),
                    # arbor-shape provenance so the session row is self-describing (before/after tput + knobs).
                    "throughput_before": float(getattr(ss, "baseline_tput", 0.0) or 0.0),
                    "throughput_after": (
                        float(current_best.get("tput", 0.0)) if isinstance(current_best, dict) else 0.0
                    ),
                    "date": datetime.now(timezone.utc).isoformat(),
                    "actions_taken": [
                        nm
                        for nm in (
                            str(e.get("variant_name") or e.get("name") or e.get("action") or "").strip()
                            for e in opt_stack
                            if isinstance(e, dict)
                        )
                        if nm
                    ],
                }
            ],
        }

    def _record_remote_recipe_audit(
        self,
        *,
        source: str,
        status: str,
        canonical_id: str,
        session_id: str,
        optimized_throughput: float = 0.0,
        reason: str = "",
        error_type: str = "",
    ) -> None:
        """Append one best-effort, secret-free KB Store publish audit row."""
        try:
            from hyperloom.inference_optimizer.session.session_paths import (
                recipe_snapshot_audit_jsonl,
            )

            row: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "op": "write",
                "method": "write",
                "mode": "remote",
                "backend": "kb-store",
                "remote": "kb-store",
                "resolution": status,
                "success": status in {"written", "skipped"},
                "generator": source,
                "phase": "CLOSE",
                "status": status,
                "reason": reason,
                "request": {
                    "canonical_id": canonical_id or None,
                    "session_id": session_id or None,
                },
                "result": {
                    "canonical_id": canonical_id,
                    "session_id": session_id,
                    "created": status == "written",
                    "best_throughput": optimized_throughput,
                },
                "provenance": {
                    "component": "remote_recipe",
                    "source": source,
                },
            }
            if error_type:
                row["error"] = {"type": error_type}
            append_jsonl(
                recipe_snapshot_audit_jsonl(self.session_dir),
                row,
                make_parents=True,
                sort_keys=True,
            )
        except Exception:  # noqa: BLE001 - audit cannot break finalization
            log.debug("Remote Recipe KB audit append failed", exc_info=True)

    def ensure_recipe_finalized(
        self,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Idempotently publish terminal Recipe state and persist its outcome."""
        state = self.shared_state
        prior = dict(getattr(state, "recipe_finalize_outcome", {}) or {})
        prior_status = str(getattr(state, "recipe_finalize_status", "") or prior.get("status") or "")
        if prior_status in {"written", "skipped", "disabled"}:
            return prior

        attempts = int(getattr(state, "recipe_finalize_attempts", 0) or 0) + 1
        state.recipe_finalize_attempts = attempts
        state.recipe_finalize_status = "pending"
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — publication can still proceed
            log.exception("Recipe finalize pending-state save failed")

        try:
            raw = self.finalize_recipe_and_journal(source=source) or {}
            outcome = {
                **dict(raw),
                "source": source,
                "attempt": attempts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # noqa: BLE001 — persist retryable failure
            log.exception("Recipe finalize raised")
            outcome = {
                "status": "error",
                "reason": type(exc).__name__,
                "source": source,
                "attempt": attempts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        raw_status = str(outcome.get("status") or "error")
        state.recipe_finalize_status = "failed" if raw_status == "error" else raw_status
        state.recipe_finalize_outcome = outcome
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — T4 can still retry in-process
            log.exception("Recipe finalize outcome save failed")
        return outcome

    def finalize_recipe_and_journal(
        self,
        *,
        source: str = "close",
    ) -> dict[str, Any]:
        """Finalize Recipe state and return a secret-free observable outcome."""
        try:
            journal = self._ensure_journal()
            ss = self.shared_state
            cb = getattr(ss, "current_best", {}) or {}
            final_tput = float(cb.get("tput", 0.0)) if isinstance(cb, dict) else 0.0
            total_gain = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
            journal.finalize(
                final_throughput=final_tput if final_tput > 0 else None,
                total_gain_pct=total_gain,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("optimization_journal.finalize failed")

        if bool(getattr(getattr(self, "knowledge_plane", None), "kb_disabled", False)):
            log.info("Recipe KB finalize skipped (--degraded-kb)")
            return {
                "status": "skipped",
                "reason": "degraded_kb",
                "backend": "disabled",
            }

        from ..knowledge.config import KnowledgeConfig, KnowledgeStoreMode

        try:
            config = getattr(getattr(self, "knowledge_plane", None), "config", None) or KnowledgeConfig.from_env()
        except Exception as exc:  # noqa: BLE001 - KB remains best-effort
            log.exception("Recipe KB finalize configuration failed (non-fatal)")
            return {
                "status": "error",
                "reason": f"configuration:{type(exc).__name__}",
                "backend": "unknown",
            }
        # Every Recipe sink funnels through agentx_kb_write_blocked; see it for
        # why an agentic measurement must not enter a cross-session store. Placed
        # ahead of the mode branch because in REMOTE mode _kb_amend_recipe returns
        # early, which made the write below the only Recipe writer and the one
        # door that gate could not see.
        from hyperloom.orchestrator.actions.executors._workload_envs import (
            agentx_kb_write_blocked,
        )

        if agentx_kb_write_blocked(self.shared_state):
            log.info(
                "Recipe KB finalize skipped (AgentX): the recipe identity has no mode "
                "or workload dimension, so an agentic-replay result would overwrite a "
                "synthetic best_throughput and be tagged isl/osl=%s/%s.",
                getattr(self.shared_state, "isl", "?"),
                getattr(self.shared_state, "osl", "?"),
            )
            # Backend stays as configured: "disabled" here would be
            # indistinguishable in telemetry from a KB that was actually down,
            # and reason= already carries why nothing was written.
            return {
                "status": "skipped",
                "reason": "agentx",
                "backend": str(getattr(config, "mode", "") or "unknown"),
            }
        if config.mode is KnowledgeStoreMode.REMOTE:
            # Remote mode has one Recipe sink: the KB Store final session
            # writer. T0 and runtime amendment are intentionally absent.
            remote_cid = ""
            remote_sid = str(
                getattr(self.shared_state, "recipe_kb_session_id", "")
                or getattr(self.shared_state, "session_id", "")
                or self.session_dir.name
            )
            try:
                from ..knowledge.remote_recipe import HyperloomRemoteKB

                remote_cid = self._workload_canonical_id()
                remote_result = HyperloomRemoteKB.from_env().write(
                    remote_cid,
                    self.shared_state,
                    session_id=remote_sid,
                )
                log.info(
                    "Remote Recipe KB finalize: status=%s reason=%s cid=%s sid=%s",
                    remote_result.status,
                    remote_result.reason,
                    remote_cid,
                    remote_result.session_id,
                )
                self._record_remote_recipe_audit(
                    source=source,
                    status=remote_result.status,
                    canonical_id=remote_cid,
                    session_id=remote_result.session_id,
                    optimized_throughput=remote_result.optimized_throughput,
                    reason=remote_result.reason,
                )
                return {
                    "status": remote_result.status,
                    "reason": remote_result.reason,
                    "backend": "kb-store",
                    "canonical_id": str(getattr(remote_result, "canonical_id", "") or remote_cid),
                    "session_id": str(getattr(remote_result, "session_id", "") or remote_sid),
                }
            except Exception as exc:  # noqa: BLE001 - remote transport is best-effort
                self._record_remote_recipe_audit(
                    source=source,
                    status="error",
                    canonical_id=remote_cid,
                    session_id=remote_sid,
                    error_type=type(exc).__name__,
                )
                log.exception("Remote Recipe KB finalize failed (non-fatal)")
                return {
                    "status": "error",
                    "reason": type(exc).__name__,
                    "backend": "kb-store",
                    "canonical_id": remote_cid,
                    "session_id": remote_sid,
                }

        # Local mode never consults ambient KB_STORE_* credentials.
        if self.recipe_kb is None:
            return {
                "status": "skipped",
                "reason": "no_recipe_backend",
                "backend": "local",
            }
        ss = self.shared_state
        model_name = getattr(ss, "model_name", "") or ""
        gpu_type = getattr(ss, "gpu_type", "") or ""
        if not model_name or not gpu_type:
            log.info(
                "recipe KB finalize_recipe: missing model/hardware (model=%r hardware=%r); skipping update_recipe",
                model_name,
                gpu_type,
            )
            return {
                "status": "skipped",
                "reason": "missing_model_or_hardware",
                "backend": "local",
            }
        try:
            attrs = self._coord._build_recipe_attrs_from_state()
            # Hoist workload tags flat into top-level recipe attrs (shallow-merged) for warm-start filters.
            workload_tags = attrs.get("workload") or {}

            # sessions[] read-modify-write: read anchor, drop prior entry with our session_id (resume safety), append ours, write back.
            my_sessions = list(attrs["sessions"] or [])
            my_session_ids = {str((s or {}).get("session_id") or "") for s in my_sessions if isinstance(s, dict)}
            # v2: read-modify-write the recipe row; sessions[] merged in-process under the cid flock so concurrent finalises don't tear.
            merged_sessions: list[dict[str, Any]] = list(my_sessions)
            existing_row: dict[str, Any] = {}
            if self.recipe_kb is not None:
                try:
                    cid = self._workload_canonical_id()
                    # Read exactly the local store's authority row.
                    existing_row = self.recipe_kb.get_authoritative_recipe(canonical_id=cid) or {}
                    existing_sessions: list[dict[str, Any]] = []
                    for row in existing_row.get("sessions") or []:
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("session_id") or "") in my_session_ids:
                            # Resume/retry of the same session — our new entry supersedes the prior one.
                            continue
                        existing_sessions.append(dict(row))
                    merged_sessions = existing_sessions + my_sessions
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.info(
                        "recipe read failed (%s); finalize will append "
                        "the current session only; the next finalize "
                        "will catch up.",
                        exc,
                    )

            # KEEP'd kernel optimizations ride the extras channel; merge with prior rows, dedup by kernel_id.
            kopts_new = list(attrs.get("kernel_optimizations") or [])
            new_kids = {str((k or {}).get("kernel_id") or "") for k in kopts_new if isinstance(k, dict)}
            merged_kopts: list[dict[str, Any]] = list(kopts_new)
            for prior in existing_row.get("kernel_optimizations") or []:
                if not isinstance(prior, dict):
                    continue
                if str(prior.get("kernel_id") or "") in new_kids:
                    continue
                merged_kopts.append(dict(prior))

            extras_payload = dict(workload_tags or {})
            if merged_kopts:
                extras_payload["kernel_optimizations"] = merged_kopts

            overrides: dict[str, Any] = {
                "what_worked": attrs["what_worked"],
                "what_failed": attrs["what_failed"],
                "last_profiled": attrs["last_profiled"],
                "sessions": merged_sessions,
                "extras": extras_payload,
            }
            # Overwrite best_config/best_throughput only on a real improvement: requires has_validated_win AND my_tput > live_tput.
            my_tput = float(attrs.get("best_throughput") or 0.0)
            cb_now = getattr(ss, "current_best", {}) or {}
            cb_args_now = str(cb_now.get("extra_server_args") or "").strip() if isinstance(cb_now, dict) else ""
            validated_gain = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
            has_validated_win = bool(
                (getattr(ss, "optimization_stack", []) or []) or validated_gain > 0.0 or cb_args_now
            )
            try:
                live_tput = float(existing_row.get("best_throughput") or 0.0)
            except (TypeError, ValueError):
                live_tput = 0.0
            if has_validated_win and my_tput > live_tput:
                overrides["best_config"] = attrs["best_config"]
                overrides["best_throughput"] = my_tput
            self._kb_amend_recipe(
                recipe_overrides=overrides,
                provenance_details={
                    "phase": "close_finalize",
                    "evidence": [
                        f"log:session-{getattr(ss, 'recipe_kb_session_id', '') or self.session_dir.name}",
                    ],
                },
            )
            return {
                "status": "written",
                "reason": "",
                "backend": "local",
            }
        # Catch-all keeps CLOSE step 2.5 defensive against programmer bugs.
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("update_recipe raised unexpectedly")
            return {
                "status": "error",
                "reason": type(exc).__name__,
                "backend": "local",
            }

    async def _record_specialist_result(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
        run_error: str = "",
    ) -> None:
        """Common bookkeeping for any specialist task termination (dispatcher loop + intent routing); idempotent on round_id, failures logged not raised.

        Args:
            task: The terminated specialist task.
            done_payload: The specialist's done payload (proposal_set, domain,
                summary, etc.).
            source: The emitting agent string (``specialist:<task_id>``).
            run_error: Dispatch failure text when the specialist produced no
                usable payload.
        """
        task_params = task.params or {}
        domain = str(done_payload.get("domain") or task_params.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = bool(done_payload.get("empty")) or len(proposals) == 0

        round_entry = self._build_specialist_round_entry(
            task=task,
            done_payload=done_payload,
            source=source,
            run_error=run_error,
        )
        # Advisory multi-model scoring of the proposal_set; informational only, gates nothing. Defensive.
        _scorer = getattr(self, "_proposal_scorer", None)
        if _scorer is not None and proposals:
            try:
                scores = await _scorer.score(
                    gap={
                        "domain": domain,
                        "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                        "gap_symptom": task_params.get("gap_symptom"),
                        "gap_evidence": task_params.get("gap_evidence"),
                        "summary": done_payload.get("summary", ""),
                    },
                    proposals=proposals,
                    task_id=task.task_id,
                    tick=int(getattr(self.shared_state, "tick", 0) or 0),
                    phase=(getattr(self.shared_state, "phase", "") or "") or None,
                )
                if scores and (scores.get("models") or scores.get("errors")):
                    round_entry["ensemble_scores"] = scores
                    input_err = (scores.get("errors") or {}).get("input")
                    if input_err and not scores.get("models"):
                        log.warning(
                            "specialist bookkeeping: proposal scoring skipped for task=%s: %s",
                            task.task_id,
                            input_err,
                        )
            except Exception:  # noqa: BLE001 — advisory; never block
                log.exception(
                    "specialist bookkeeping: proposal scoring failed for task=%s (continuing without scores)",
                    task.task_id,
                )
        try:
            self.shared_state.record_specialist_round(round_entry)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: record_specialist_round failed for task=%s",
                task.task_id,
            )

        # Per-anchor coverage ledger: every specialist completion is
        # one "round" — tick all anchors, then zero the one that just ran so a
        # long-idle domain's counter climbs until the hard-trigger forces it.
        try:
            self.shared_state.bump_domain_round_counters()
            self.shared_state.note_specialist_dispatched(domain)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: domain round-counter update failed for task=%s",
                task.task_id,
            )

        try:
            self.shared_state.update_last_specialist(
                {
                    "task_id": task.task_id,
                    "domain": domain,
                    "gap_canonical_id": str(
                        done_payload.get("gap_canonical_id") or task_params.get("gap_canonical_id") or ""
                    ),
                    "empty": is_empty,
                    "proposals_total": len(proposals),
                    "confidence": done_payload.get("confidence"),
                    "summary": str(done_payload.get("summary") or "")[:480],
                    "reason": str(run_error or done_payload.get("reason") or "")[:480],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: update_last_specialist failed for task=%s",
                task.task_id,
            )

        # Persist so a resume picks up the bookkeeping without re-running the specialist.
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: SharedState.save failed for task=%s",
                task.task_id,
            )

        # Routed via ``_coord`` so a test / caller that overrides
        # ``coordinator._record_observation`` still wins (bare-name delegation
        # resolves it back onto this class otherwise).
        await self._coord._record_observation(
            source or "coordinator",
            "observation",
            {
                "kind": "specialist_done_recorded",
                "task_id": task.task_id,
                "domain": domain,
                "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                "proposals_total": len(proposals),
                "empty": is_empty,
            },
        )

        # Multi-node only: auto-materialise the proposal_set into a
        # benchmarked explore task. No-op single-node (LLM drives explore
        # directly there) and no-op when the proposal_set is empty / has
        # no applicable variants. See :meth:`_maybe_materialize_mn_explore`.
        try:
            await self._maybe_materialize_mn_explore(
                task=task,
                domain=domain,
                proposals=proposals,
            )
        except Exception:  # noqa: BLE001 — defensive; never block bookkeeping
            log.exception(
                "mn_auto_materialize: bridge raised for task=%s (continuing)",
                task.task_id,
            )

        # Harvest research-scout output (hints, competitor target, gap seeds, PR dedup). Fail-soft.
        if domain == "research_scout_specialist":
            try:
                await self._coord._harvest_research_scout(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout harvest failed for task=%s",
                    task.task_id,
                )

        # Consume static-recon bridge candidates into gaps[] so the
        # freeform specialist picks them up with a precise mandate. Fail-soft.
        if domain == "static_recon_specialist":
            try:
                self._coord._consume_static_recon(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "static-recon consume failed for task=%s",
                    task.task_id,
                )

        # Aggregate research evidence from any research domain that
        # self-reports a ``research`` block, so FRAMEWORK / explore lanes
        # reuse the session-wide seen-set. Idempotent for research_scout
        # (already harvested above). Fail-soft.
        try:
            self._coord._aggregate_research_evidence(done_payload)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "research evidence aggregation failed for task=%s",
                task.task_id,
            )

        # Refresh the gaps ledger after a specialist round closes; record the verdict as a gap attempt.
        gap_cid = str(done_payload.get("gap_canonical_id") or "").strip()
        if gap_cid:
            try:
                self.shared_state.append_gap_attempt(
                    gap_cid,
                    {
                        "action": "specialist",
                        "variant_name": domain,
                        "outcome": "EMPTY" if is_empty else "PROPOSALS",
                        "proposals_total": len(proposals),
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "specialist bookkeeping: append_gap_attempt failed for gap=%s",
                    gap_cid,
                )
        try:
            await self._refresh_gaps(reason="specialist_done")
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "specialist bookkeeping: _refresh_gaps failed for task=%s",
                task.task_id,
            )
        if bool((task.params or {}).get("enablement")) and isinstance(done_payload.get("needs_targeted_build"), dict):
            try:
                await self._maybe_enqueue_specialist_requested_build(
                    task_id=str(task.task_id or ""),
                    payload=done_payload,
                )
            except Exception:  # noqa: BLE001
                log.exception(
                    "specialist build request failed for task=%s",
                    task.task_id,
                )
        # Push specialist-authored patches to the Critic so integrate_patch can pass.
        try:
            await self._maybe_autosubmit_specialist_patches(
                task=task,
                done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "B3: specialist patch autosubmit failed for task=%s",
                task.task_id,
            )
        # Relaxed FRAMEWORK rule: a config-lever deliverable (no source patch,
        # but a proposal_set of serving flags / env vars) is routed through the
        # same integrate_patch gate via its config_changes channel.
        try:
            await self._maybe_autosubmit_framework_config(
                task=task,
                done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK config autosubmit failed for task=%s",
                task.task_id,
            )

    def _aggregate_research_evidence(self, done_payload: dict[str, Any]) -> None:
        """Aggregate research evidence (PR ids / diffs / NVIDIA refs) into the
        session-wide seen-set, de-duped across the session.

        Applies to every domain that self-reports a ``research`` block
        (candidate discovery + research_scout), so FRAMEWORK / explore lanes
        do not re-fetch the same references. Fail-soft: never raises (the caller
        also guards, but keep this self-contained so partial payloads degrade
        gracefully).
        """
        block = done_payload.get("research")
        if not isinstance(block, dict):
            return
        pr_ids: list[Any] = []
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs"):
            vals = block.get(key)
            if isinstance(vals, list):
                pr_ids.extend(vals)
        if not pr_ids:
            return
        try:
            added = self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "depth: register_seen_pr_ids failed during research aggregation",
            )
            return
        if added:
            log.info(
                "depth: aggregated %d new research reference(s) into seen-set",
                added,
            )

    async def _harvest_research_scout(self, done_payload: dict[str, Any]) -> None:
        """Persist top-level scout output and re-seed Orchestration.

        The scout is a text-hints-only collector. Any ``competitor_target``
        numbers it emits are intentionally ignored here: measured competitor
        baselines are sourced from InferenceX, not authored by the scout, so
        LLM-written numbers must never be persisted as a consumable target.

        Args:
            done_payload: The completed research-scout task payload.
        """
        from ..knowledge import research_hints as _research_hints

        hints = done_payload.get("new_findings") or []
        if not isinstance(hints, list):
            hints = []
        try:
            added, dropped = _research_hints.append_hints(
                self.session_dir,
                hints,
            )
            if dropped:
                log.info(
                    "research-scout: dropped %d sourceless hint(s)",
                    dropped,
                )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: append_hints failed")
            added = 0
        # Share inspected PR ids with the FRAMEWORK dedup set.
        pr_ids: list[Any] = []
        for hint in hints:
            if isinstance(hint, dict) and hint.get("source"):
                pr_ids.append(hint["source"])
        proposals = done_payload.get("proposal_set") or []
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                for key in ("pr_evidence", "source_evidence"):
                    refs = proposal.get(key)
                    if isinstance(refs, list):
                        pr_ids.extend(refs)
        try:
            self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: register_seen_pr_ids failed")
        # Seed high-priority hints as gaps[] so the config arm tries them early.
        try:
            self._seed_gaps_from_research_hints()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: gap seeding failed")
        compacted = await self._coord._maybe_checkpoint_orchestration(
            tick=int(getattr(self.shared_state, "tick", 0) or 0),
            force=True,
        )
        if not compacted:
            self._coord._reset_orchestration_conversation()
        log.info(
            "research-scout harvested: hints_added=%d seen_pr_ids=%d",
            added,
            len(self.shared_state.research_scout_seen_pr_ids or []),
        )

    def _lift_to_current_best(
        self,
        task_kind: str,
        best_tput: float,
        bv: dict[str, Any],
        *,
        gap_canonical_id: str = "",
        entry_extra: Mapping[str, Any] | None = None,
    ) -> bool:
        """Lift a winner only when it improves the current throughput anchor.

        The only writer of ``optimization_stack``, and the only config-KEEP
        writer of ``current_best`` (the baseline anchor is the other one). Also
        the only place a winner is merged onto the previous config instead of
        replacing it, so ``unset_envs`` and cumulative args stay correct across
        the whole stack. The stack append is skipped when the winner is already
        applied, keyed by ``(action, variant_name)`` or by ``fingerprint``, so a
        rerun of an already-stacked config cannot double-apply it.

        Every args string read here passes through
        :func:`strip_benchmark_harness_flags`, the previous ``current_best``
        included since it is re-merged onto the winner.

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
            ``True`` when the winner was lifted, ``False`` when it was refused
            for not beating the current anchor.
        """
        anchor = resolve_grading_anchor_tput(self.shared_state)
        if anchor > 0 and float(best_tput) <= anchor:
            log.info(
                "current_best held at %.1f: %s winner measured %.1f (no lift)",
                anchor,
                task_kind,
                float(best_tput),
            )
            return False
        previous = self.shared_state.current_best or {}
        base_args = ""
        if isinstance(previous, dict):
            base_args = strip_benchmark_harness_flags(previous.get("extra_server_args"))
        # An authored-kernel overlay stays active until another KEEP replaces it.
        _overlay = str((bv.get("final_overlay") if isinstance(bv, dict) else "") or "").strip()
        if not _overlay and isinstance(previous, dict):
            _overlay = str(previous.get("final_overlay") or "").strip()
        candidate_args = ""
        if isinstance(bv, dict):
            candidate_args = strip_benchmark_harness_flags(
                bv.get("candidate_extra_server_args") or bv.get("extra_server_args")
            )
        full_args = ""
        if isinstance(bv, dict):
            full_args = strip_benchmark_harness_flags(bv.get("extra_server_args"))
        controls_effective = bool(
            isinstance(bv, dict)
            and (
                bv.get("remove_args")
                or bv.get("unset_envs")
                or str(bv.get("args_mode") or "").strip().lower() == "replace"
            )
        )
        # Build cumulative launch args without double-stacking; helper dedupes repeated --flag pairs (last wins).
        if controls_effective:
            # Removal/replace winners publish their effective cumulative config
            # from ExploreExecutor. Prepending the prior current_best would
            # reintroduce flags the variant deliberately removed.
            full_args = _dedupe_extra_server_args(full_args)
        else:
            full_args = _merge_cumulative_extra_server_args(
                base_args,
                candidate_args,
                full_args,
            )

        variant_name = bv.get("name") if isinstance(bv, dict) else None
        if candidate_args or variant_name:
            existing = {
                (str(e.get("action")), str(e.get("variant_name")))
                for e in self.shared_state.optimization_stack
                if isinstance(e, dict)
            }
            key = (task_kind, str(variant_name or ""))
            # A rerun under a new name must not re-apply an already-stacked config.
            candidate_fp = str(bv.get("fingerprint") or "")
            already_stacked = key in existing or (
                bool(candidate_fp)
                and any(
                    candidate_fp == str(e.get("fingerprint") or "")
                    for e in self.shared_state.optimization_stack
                    if isinstance(e, dict)
                )
            )
            if not already_stacked:
                source_phase = str(
                    (bv.get("source_phase") if isinstance(bv, dict) else "")
                    or (getattr(self.shared_state, "phase", "") if task_kind != "integrate_patch" else "")
                    or ""
                ).strip()
                # The phase fallback above inherits whatever is live at
                # writeback time, which is not what authored the winner.
                lever_kind = _lever_kind_for_lift(task_kind, bv)
                if not lever_kind:
                    log.warning(
                        "lift: no lever_kind for a %s winner (variant=%s); the stack entry will report as unattributed",
                        task_kind,
                        variant_name,
                    )
                stack_entry: dict[str, Any] = {
                    "action": task_kind,
                    "variant_name": variant_name,
                    "candidate_extra_server_args": candidate_args,
                    "candidate_extra_envs": (
                        dict(bv.get("candidate_extra_envs") or {}) if isinstance(bv, dict) else {}
                    ),
                    "extra_server_args": full_args,
                    "extra_envs": (dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}),
                    # Carry the promoting lane's accuracy verdict onto the stack
                    # so CLOSE reads one place instead of reconstructing which
                    # lane promoted the champion. ``None`` means "not gated".
                    "accuracy": (bv.get("accuracy") if isinstance(bv, dict) else None),
                    "tput": float(best_tput),
                    "workspace": (bv.get("workspace") if isinstance(bv, dict) else None),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                if source_phase:
                    stack_entry["source_phase"] = source_phase
                if lever_kind:
                    stack_entry["lever_kind"] = lever_kind
                if gap_canonical_id:
                    stack_entry["gap_canonical_id"] = gap_canonical_id
                # Stamp the variant's stable join key (and source) so breakdown
                # attribution maps this explore gain to its specialist provenance.
                fp_val = ""
                prov_val = ""
                if isinstance(bv, dict):
                    fp_val = str(bv.get("fingerprint") or "").strip()
                    if not fp_val:
                        from ..actions.executors._canonical_fingerprint import (
                            canonical_fingerprint,
                        )

                        fp_val = canonical_fingerprint(
                            candidate_args or full_args,
                            dict(bv.get("extra_envs") or {}),
                            remove_args=bv.get("remove_args"),
                            unset_envs=bv.get("unset_envs"),
                            args_mode=str(bv.get("args_mode") or "append"),
                        )
                    prov_val = str(bv.get("provenance") or "").strip()
                if fp_val:
                    stack_entry["fingerprint"] = fp_val
                if prov_val:
                    stack_entry["provenance"] = prov_val
                # Carry the authored-kernel names onto the stack entry so
                # attribution can separate a config gain from a gain measured
                # with a kernel loaded. Absent on every flags-only variant.
                if isinstance(bv, dict):
                    _stack_kernels = [str(k).strip() for k in (bv.get("accepted_kernels") or []) if str(k).strip()]
                    if _stack_kernels:
                        stack_entry["accepted_kernels"] = _stack_kernels
                if isinstance(bv, dict):
                    recipe_delta = bv.get("recipe_delta")
                    if isinstance(recipe_delta, Mapping):
                        stack_entry["recipe_delta"] = {
                            "extra_server_args": str(recipe_delta.get("extra_server_args") or "").strip(),
                            "extra_envs": dict(recipe_delta.get("extra_envs") or {}),
                            "remove_args": to_str_list(recipe_delta.get("remove_args")),
                            "unset_envs": to_str_list(recipe_delta.get("unset_envs")),
                            "args_mode": str(recipe_delta.get("args_mode") or "append").strip().lower(),
                        }
                    for _ctrl_key in ("remove_args", "unset_envs", "args_mode"):
                        if bv.get(_ctrl_key):
                            stack_entry[_ctrl_key] = bv.get(_ctrl_key)
                    if bv.get("task_id"):
                        stack_entry["task_id"] = str(bv.get("task_id"))
                    if bv.get("effective_extra_server_args"):
                        stack_entry["effective_extra_server_args"] = _dedupe_extra_server_args(
                            strip_benchmark_harness_flags(bv.get("effective_extra_server_args"))
                        )
                # Stable filter label for "what kind of optimization" (backend /
                # param / env), so the stack can be sliced like the timeline.
                _stack_envs = dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
                stack_entry["operation_kind"] = operation_kind_for(
                    task_kind,
                    classify_change_kind(
                        task_kind,
                        {"extra_server_args": candidate_args, "extra_envs": _stack_envs},
                    ),
                )
                _stack_scope = str(bv.get("scope") or "").strip() if isinstance(bv, dict) else ""
                if _stack_scope:
                    stack_entry["scope"] = _stack_scope
                if isinstance(bv, dict):
                    for _src_key in (
                        "source_snapshot",
                        "source_manifest",
                        "source_import_root",
                        "framework_root",
                        "base_sha",
                    ):
                        val = bv.get(_src_key)
                        if val:
                            stack_entry[_src_key] = str(val)
                    # Copied unconditionally: False is the meaningful value that
                    # marks a snapshot unusable, so truthiness must not drop it.
                    if "source_snapshot_complete" in bv:
                        stack_entry["source_snapshot_complete"] = bool(bv["source_snapshot_complete"])
                    target_files = [str(path) for path in (bv.get("target_files") or []) if str(path).strip()]
                    if target_files:
                        stack_entry["target_files"] = target_files
                    for _attr_key in ("baseline_enablement", "attribution_eligible"):
                        if _attr_key in bv:
                            stack_entry[_attr_key] = bool(bv.get(_attr_key))
                    if "recipe_publishable" in bv:
                        stack_entry["recipe_publishable"] = bool(bv.get("recipe_publishable"))
                    if "framework_agent_authoring" in bv:
                        stack_entry["framework_agent_authoring"] = bool(bv.get("framework_agent_authoring"))
                    for _origin_key in ("domain", "gap_layer"):
                        if bv.get(_origin_key):
                            stack_entry[_origin_key] = str(bv.get(_origin_key))
                if _overlay:
                    stack_entry["final_overlay"] = _overlay
                for _extra_key, _extra_val in (entry_extra or {}).items():
                    if _extra_val not in (None, "", [], {}):
                        stack_entry[str(_extra_key)] = _extra_val
                self.shared_state.optimization_stack.append(stack_entry)
                # Mirror append into gain_per_stack_entry so the two lists stay index-aligned.
                self.shared_state.append_stack_gain_entry(
                    action=task_kind,
                    variant_name=variant_name,
                    new_tput=best_tput,
                    extra_server_args=full_args,
                )

        # Merge envs: start from previous stack top envs so source-layer KEEPs
        # (config_changes_applied={}) do not clear prior explore/env layers.
        _prev_envs = dict((previous.get("extra_envs") or {}) if isinstance(previous, dict) else {})
        _new_envs = dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
        _merged_envs = dict(_prev_envs)
        for _key in to_str_list(bv.get("unset_envs") if isinstance(bv, dict) else None):
            _merged_envs.pop(_key, None)
        _merged_envs.update(_new_envs)
        current_best = {
            "action": task_kind,
            "tput": float(best_tput),
            "variant_name": variant_name,
            "extra_server_args": full_args,
            "extra_envs": _merged_envs,
            "final_overlay": _overlay,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": bv.get("ttft_mean_ms") if isinstance(bv, dict) else None,
            "e2el_mean_ms": bv.get("e2el_mean_ms") if isinstance(bv, dict) else None,
            "tpot_mean_ms": bv.get("tpot_mean_ms") if isinstance(bv, dict) else None,
            "workspace": bv.get("workspace") if isinstance(bv, dict) else None,
        }
        if isinstance(bv, dict):
            for _ctrl_key in ("remove_args", "unset_envs", "args_mode"):
                if bv.get(_ctrl_key):
                    current_best[_ctrl_key] = bv.get(_ctrl_key)
            if bv.get("effective_extra_server_args"):
                current_best["effective_extra_server_args"] = _dedupe_extra_server_args(
                    strip_benchmark_harness_flags(bv.get("effective_extra_server_args"))
                )
            if (bv.get("remove_args") or bv.get("unset_envs")) and not current_best.get("args_mode"):
                current_best["args_mode"] = "replace"
        self.shared_state.current_best = current_best
        self._stamp_current_best_measurement(bv)
        return True

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

    async def _promote_to_shared_state(
        self,
        task_kind: str,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Lift specific action-result fields into the persistent SharedState (baseline/profile/roofline/grid).

        Args:
            task_kind: The settled task's kind, selecting the promote branch.
            result: The task result dict; non-dict results are ignored.
            task: The originating task, used for audit fingerprints and
                pending-roofline gating.
        """
        if not isinstance(result, dict):
            return
        # ``replay_warm_recipe`` is mirrored by _promote_replay_warm_recipe
        # instead: its executor settles on "succeeded" and the keep decision is
        # only reached further down this call, so mirroring it here published
        # every replay as discarded -- including the ones that went on to be
        # pushed onto the stack.
        if task_kind in {"conc_sweep", "integrate_patch"}:
            try:
                from hyperloom.inference_optimizer.breakdown.recorder import instrument

                result_status = str(result.get("status") or "succeeded")
                # Every promotable kind reports its own verdict; hardcoding
                # "discarded" for the rest made kept integrate_patch work look
                # rejected in the breakdown and stripped its attribution.
                kept = result_status.lower() in {"kept", "kept_inert", "promoted", "adopted"}
                v4_result = dict(result)
                v4_result.setdefault("workload", self._v4_workload())
                instrument.record_action_operation(
                    self.session_dir,
                    action=task_kind,
                    task_id=getattr(task, "task_id", "") if task is not None else "",
                    status=result_status,
                    decision="promoted" if kept else "discarded",
                    result=v4_result,
                    extras={
                        "candidate_id": self._framework_candidate_key(result.get("candidate"))
                        if isinstance(result.get("candidate"), dict)
                        else ""
                    },
                    phase=str(getattr(self.shared_state, "phase", "") or ""),
                    macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    tick=int(getattr(self.shared_state, "tick", 0) or 0),
                )
            except Exception:  # noqa: BLE001
                log.debug("v4 action result capture failed", exc_info=True)
        outcome = _PromoteOutcome()
        handler_name = self._PROMOTE_HANDLERS.get(task_kind)
        if handler_name is not None:
            await getattr(self, handler_name)(result, task, outcome)
        # sweep / conc_sweep already recorded + saved + returned via their handler.
        if outcome.early_return:
            return
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

    _PROMOTE_HANDLERS: dict[str, str] = {
        "baseline": "_promote_baseline",
        "replay_warm_recipe": "_promote_replay_warm_recipe",
        "profile": "_promote_profile",
        "roofline": "_promote_roofline",
        "explore": "_promote_explore",
        "integrate_patch": "_promote_integrate_patch",
        "sweep": "_promote_sweep",
        "conc_sweep": "_promote_conc_sweep",
    }

    async def _promote_baseline(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a baseline result: anchor tput / accuracy / config and bootstrap PRELUDE."""
        changed = False
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        tput = result.get("output_throughput")
        warmup_anchor = result.get("warmup_round_tput")
        tracked_tid = str(getattr(self.shared_state.enablement, "revalidation_task_id", "") or "").strip()
        promoting_tid = str(getattr(task, "task_id", "") or "").strip() if task is not None else ""
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        is_revalidation = bool(
            task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON
            or (tracked_tid and tracked_tid == promoting_tid)
        )
        # The anchor is the best measurement of the unmodified stack. A later,
        # lower re-baseline must not replace it, or downstream gains end up
        # measured against a different reference than the one they claim. A
        # revalidation is exempt: the enablement patch changed the stack, so the
        # prior anchor no longer describes anything reproducible.
        prior_anchor = float(self.shared_state.baseline_tput or 0.0)
        # A hot pass measured against a recorded COLD anchor is a correction, not
        # a regression, so it lands even when the number is lower. The two are not
        # comparable -- that is what the cold marker says -- and a cold figure can
        # read higher than the hot one that replaces it whenever the "cold" pass
        # was not really cold: weights in page cache and a JIT cache a prior run
        # populated leave it paying none of the startup its depressed reputation
        # assumes. Without this the session cannot escape the marker. PRELUDE
        # refuses to finish while it is set, the retry that would clear it is
        # rejected for measuring lower, and the run re-measures whole baseline
        # rounds until the clock kills it.
        hot_pass_ran = result.get("measure_round_runtime_sec")
        corrects_a_cold_anchor = (
            bool(getattr(self.shared_state, "baseline_measure_round_dropped", False))
            and isinstance(hot_pass_ran, (int, float))
            and hot_pass_ran > 0
        )
        anchor_accepted = bool(
            isinstance(tput, (int, float))
            and tput > 0
            and (prior_anchor <= 0.0 or float(tput) > prior_anchor or is_revalidation or corrects_a_cold_anchor)
        )
        if isinstance(tput, (int, float)) and tput > 0:
            if anchor_accepted:
                # The anchor is the hot measure round; the cold warmup round is
                # discarded so gain math never mixes cold-before with hot-after.
                self.shared_state.baseline_tput = float(tput)
            else:
                log.info(
                    "baseline anchor: keeping %.1f; re-baseline measured %.1f (task=%s)",
                    prior_anchor,
                    float(tput),
                    promoting_tid,
                )
            self.shared_state.baseline_failure_streak = 0
            self.shared_state.baseline_arg_error_streak = 0
            # A genuine baseline may revalidate an eval-origin enablement.
            if bool(getattr(self.shared_state.enablement, "validation_pending", False)):
                if is_revalidation:
                    acc = result.get("accuracy")
                    floor = float(getattr(self.shared_state.enablement, "accuracy_floor", 0.0) or 0.0)
                    if accuracy_meets_floor(acc, floor):
                        self.shared_state.enablement.succeeded = True
                        self.shared_state.enablement.validation_pending = False
                        self.shared_state.enablement.revalidation_task_id = ""
                        self.shared_state.enablement.origin = ""
                        self.shared_state.enablement.pending = False
                    else:
                        # Sub-floor accuracy on the tracked revalidation: rearm the
                        # specialist loop without clearing the frozen trigger identity.
                        log.warning(
                            "enablement revalidation: accuracy %.4f below floor %.4f; rearming",
                            acc if isinstance(acc, (int, float)) else float("nan"),
                            floor,
                        )
                        self.shared_state.enablement.validation_pending = False
                        self.shared_state.enablement.revalidation_task_id = ""
                        self.shared_state.enablement.stall_streak = (
                            int(getattr(self.shared_state.enablement, "stall_streak", 0) or 0) + 1
                        )
                        if (
                            self.shared_state.enablement.stall_streak >= _ENABLEMENT_MAX_STALL
                            and not self.shared_state.stop_reason
                        ):
                            self.shared_state.set_stop_reason("enablement_stalled")
                        else:
                            self.shared_state.enablement.inflight_task_id = ""
                else:
                    # An unrelated baseline promoted while revalidation is pending.
                    # Only anchor tput; do not consume or clear the pending state.
                    log.info(
                        "enablement revalidation pending: unrelated baseline promoted "
                        "(task=%s tracked=%s); not consuming pending state",
                        promoting_tid,
                        tracked_tid,
                    )
                    self.shared_state.enablement.origin = ""
                    self.shared_state.enablement.pending = False
            else:
                self.shared_state.enablement.origin = ""
                self.shared_state.enablement.pending = False
            changed = True
        # Accuracy / config / wall-clock describe the anchor run, so they only
        # move when the anchor itself moves; otherwise the recorded reference
        # tput and the config it was measured with drift apart.
        if anchor_accepted:
            acc = result.get("accuracy")
            if isinstance(acc, (int, float)):
                self.shared_state.baseline_accuracy = float(acc)
                changed = True
            # Persist the materialized YAML so downstream tasks reuse the exact workload contract.
            materialized = result.get("materialized_config")
            if isinstance(materialized, str) and materialized:
                self.shared_state.baseline_config_path = materialized
                changed = True
                # Parse workload-shape extras from the YAML for lesson/pitfall attrs.
                try:
                    parsed = _parse_baseline_workload_extra(materialized)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "baseline workload extra parsing failed for %s",
                        materialized,
                    )
                    parsed = {}
                if parsed:
                    self.shared_state.baseline_workload_extra = parsed
            # Promote baseline wall-clock so ExploreExecutor derives the overtime kill deadline.
            runtime_sec_raw = result.get("subprocess_runtime_sec")
            if isinstance(runtime_sec_raw, (int, float)) and runtime_sec_raw > 0:
                self.shared_state.baseline_runtime_sec = float(runtime_sec_raw)
                changed = True
            # Promote the warm measure-round wall-clock as the anchor for the
            # explore decision-round overtime kill (present only on the
            # double-run baseline path; else explore uses the cold anchor).
            warm_runtime_raw = result.get("measure_round_runtime_sec")
            if isinstance(warm_runtime_raw, (int, float)) and warm_runtime_raw > 0:
                self.shared_state.baseline_warm_runtime_sec = float(warm_runtime_raw)
                changed = True
            elif float(getattr(self.shared_state, "baseline_warm_runtime_sec", 0.0) or 0.0) != 0.0:
                self.shared_state.baseline_warm_runtime_sec = 0.0
                changed = True
            # Whether this baseline had to keep its cold figure because the budget
            # could not fund a hot pass and a variant to use it. Carried on the
            # session because the decision it drives is the session's: PRELUDE
            # routes to CLOSE rather than optimizing against a denominator that
            # was never the baseline. Cleared by a baseline that does land a hot
            # figure, so a resumed session with a fresh clock is not held to the
            # earlier leg's shortfall.
            measure_round_dropped = bool(result.get("measure_round_dropped"))
            if measure_round_dropped != bool(getattr(self.shared_state, "baseline_measure_round_dropped", False)):
                self.shared_state.baseline_measure_round_dropped = measure_round_dropped
                changed = True
            # Promote the cold round's boot/benchmark split. Cleared the same way
            # the warm figure is when a later baseline does not carry one, so a
            # stale split can never be subtracted from a fresh total and reported
            # as this workload's boot.
            post_ready_raw = result.get("post_ready_runtime_sec")
            if isinstance(post_ready_raw, (int, float)) and post_ready_raw > 0:
                self.shared_state.baseline_post_ready_runtime_sec = float(post_ready_raw)
                changed = True
            elif float(getattr(self.shared_state, "baseline_post_ready_runtime_sec", 0.0) or 0.0) != 0.0:
                self.shared_state.baseline_post_ready_runtime_sec = 0.0
                changed = True
        # current_best.tput follows the same hot baseline contract so the
        # gain numerator and denominator stay aligned. Once the stack carries a
        # validated layer, current_best belongs to the stack top and a baseline
        # must not reset it back to the bare reference config.
        if anchor_accepted and not (getattr(self.shared_state, "optimization_stack", None) or []):
            anchor_tput = float(self.shared_state.baseline_tput or 0.0)
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": (anchor_tput if anchor_tput > 0 else (float(tput) if isinstance(tput, (int, float)) else None)),
                "hot_tput": (float(tput) if isinstance(tput, (int, float)) else None),
                "cold_tput": (
                    float(warmup_anchor) if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0 else None
                ),
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
                "tpot_mean_ms": result.get("tpot_mean_ms"),
                "workspace": result.get("workspace"),
            }
            self._stamp_current_best_measurement(result)
            changed = True
        if anchor_accepted:
            audit_decision = "promoted"
        elif isinstance(tput, (int, float)) and tput > 0:
            audit_decision = "no_promote"
        else:
            audit_decision = "discarded"
        audit_extras = {
            "materialized_config": result.get("materialized_config"),
            "accuracy": result.get("accuracy"),
            "baseline_tput": (float(tput) if isinstance(tput, (int, float)) else None),
            # Stamp canonical params fingerprint for the self-loop denial helper.
            "fingerprint": _baseline_params_fingerprint(task_params),
            # Record revalidation context for history.
            "is_revalidation": bool(task_params.get("reason") == ENABLEMENT_REVALIDATION_REASON),
            "enablement_succeeded": bool(getattr(self.shared_state.enablement, "succeeded", False)),
            "enablement_accuracy_floor": float(getattr(self.shared_state.enablement, "accuracy_floor", 0.0) or 0.0),
        }
        if not anchor_accepted and isinstance(tput, (int, float)) and tput > 0:
            audit_extras["anchor_kept_tput"] = prior_anchor
        # Present only when the probe cut a runaway eval short; explains a ~0 accuracy.
        if result.get("eval_probe"):
            audit_extras["eval_probe"] = result["eval_probe"]
        # seed the gaps[] ledger from baseline (best-effort).
        await self._refresh_gaps(reason="baseline_done")
        if self.shared_state.baseline_tput > 0:
            await self._drain_queued_baselines(reason="baseline_established")
        # Standalone baseline-arm roofline ceiling (pure CPU): backs up the
        # snapshot ceiling in case the later roofline step fails.
        if isinstance(tput, (int, float)) and tput > 0:
            try:
                self.shared_state.record_baseline_roofline_ceiling()
            except Exception as exc:  # noqa: BLE001 — best-effort backup
                log.warning(
                    "baseline roofline-ceiling backup failed: %r",
                    exc,
                )
        # PRELUDE bootstrap (post-baseline), ordering mandatory: (1) inject warm-recipe history, (2) warm-replay, (3) auto-analysis, (4) research scout.
        # Only the run that first establishes the anchor bootstraps; a later
        # re-baseline must not re-fire replay / scout / recon.
        if prior_anchor <= 0.0 and self._should_run_prelude_bootstrap(tput):
            # History injection (fires regardless of --no-warm-replay).
            try:
                self._inject_warm_recipe_history_into_ledger()
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception(
                    "PRELUDE: warm-recipe history injection failed: %r",
                    exc,
                )
            # Warm-recipe replay, anchored on the hot baseline_tput contract.
            try:
                await self._maybe_enqueue_warm_replay(
                    baseline_tput=float(self.shared_state.baseline_tput or tput),
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception(
                    "PRELUDE: failed to enqueue warm-replay task: %r",
                    exc,
                )
            # Auto-analysis (roofline / profile); may defer.
            await self._maybe_enqueue_prelude_initial_analysis_after_baseline(
                baseline_tput=float(tput),
            )
            # Research scout (parallel, read-only, CPU-only).
            await self._maybe_enqueue_prelude_research_scout()
            # Static-recon (parallel, read-only, CPU-only): seed bridge
            # candidates as gaps[] before the optimisation phase starts.
            await self._maybe_enqueue_prelude_static_recon()
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _drain_queued_baselines(self, *, reason: str) -> list[str]:
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
        except Exception:  # noqa: BLE001 — draining is best-effort
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
        await self._record_observation(
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
        try:
            for task in await self.tasks.queued():
                if str(getattr(task, "kind", "") or "") != "baseline":
                    continue
                if (getattr(task, "params", None) or {}).get("reason") == ENABLEMENT_REVALIDATION_REASON:
                    spared.add(str(getattr(task, "task_id", "") or ""))
        except Exception:  # noqa: BLE001 — fall back to the tracked id alone
            log.exception("baseline drain: queued-task scan failed")
        return {t for t in spared if t}

    def _v4_workload(self) -> dict[str, Any]:
        """The workload identity a mirrored action result is measured under."""
        return {
            "framework": str(getattr(self.shared_state, "framework", "") or ""),
            "model_name": str(getattr(self.shared_state, "model_name", "") or ""),
            "gpu_type": str(getattr(self.shared_state, "gpu_type", "") or ""),
            "precision": str(getattr(self.shared_state, "precision", "") or ""),
            "tp": int(getattr(self.shared_state, "tp", 0) or 0),
            "conc": int(getattr(self.shared_state, "conc", 0) or 0),
            "isl": int(getattr(self.shared_state, "isl", 0) or 0),
            "osl": int(getattr(self.shared_state, "osl", 0) or 0),
        }

    def _mirror_warm_replay_verdict(
        self,
        result: dict,
        task: "Task | None",
    ) -> None:
        """Mirror the replay under the verdict the promote path just reached.

        A replay's keep decision belongs to the promote path, not to its
        executor: the action settles on ``succeeded`` whether or not the recipe
        reproduced, and only ``_promote_warm_replay`` compares the measured
        throughput against the bar. Mirroring it alongside the other actions,
        ahead of that ruling, recorded every replay as discarded -- so a
        reproduced one was pushed onto ``optimization_stack``, moved
        ``cumulative_gain_validated`` and journalled a KEEP while the canonical
        streams held no adoption for it. Its gain then had no adopted step to
        hang on: ``optimizations.entries`` came back empty on a session that had
        measurably gained, and the whole gain surfaced as a reconciliation gap.
        """
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        state = self.shared_state
        outcome = dict(getattr(state, "warm_replay_outcome", None) or {})
        reproduced = str(outcome.get("status") or "") == "reproduced"
        mirrored = dict(result)
        mirrored.setdefault("workload", self._v4_workload())

        # Evidence that explains the verdict on either side: the measured gain,
        # the bar it was judged against, and why it landed there. Kept outside
        # the reproduced branch so a rejected replay's attempt row can still
        # state why it was dropped -- the case that most needs an audit trail.
        gain = to_float(outcome.get("actual_gain_pct"))
        keep_threshold = to_float(outcome.get("keep_threshold_pct"))
        reason = str(outcome.get("reason") or "")
        if gain is not None:
            mirrored.setdefault("delta_pct", gain)
        if keep_threshold is not None:
            mirrored.setdefault("keep_threshold_pct", keep_threshold)
        if reason:
            mirrored.setdefault("decision_reason", reason)

        if reproduced:
            mirrored["provenance"] = "warm_replay"
            mirrored.setdefault(
                "decision_reason",
                f"warm replay reproduced {gain:+.2f}% over baseline" if gain is not None else "warm replay reproduced",
            )
            # Only a scored, passing verdict is "validated". A replay admitted
            # when its eval ran but returned no usable score (``eval_ran`` true,
            # ``replay_accuracy`` None) is adopted on the keep verdict alone and
            # must read ``keep_verdict_unscored``, not a passed accuracy gate.
            accuracy = outcome.get("replay_accuracy")
            mirrored["validated"] = True if accuracy is not None else None
            if accuracy is not None:
                # Carry the score into the measurement stream so accuracy_pass
                # has evidence standing behind it.
                mirrored.setdefault("accuracy", accuracy)
            # ``attribution_eligible`` is intentionally left to the recorder
            # default (instrument excludes enablement / inert keeps), so an
            # enablement replay is not force-credited its delta as its own gain.
            # The ledger chains this keep from the recorded session baseline,
            # not an enqueue-time anchor, keeping the ledger and
            # ``cumulative_gain_validated`` a single number.
            #
            # The executor's real status is preserved: the keep rides on
            # ``decision``, not on rewriting the status to "kept".
            status = str(result.get("status") or "succeeded")
            decision = "promoted"
        else:
            # A non-reproduced replay is not adopted. Force the recorded status
            # outside the executor-adoption verdict set so a shared executor
            # that ever reports "kept" cannot lift a drifted replay into an
            # adoption.
            status = str(outcome.get("status") or "discarded")
            decision = "discarded"
        mirrored["status"] = status

        instrument.record_action_operation(
            self.session_dir,
            action="replay_warm_recipe",
            task_id=str(getattr(task, "task_id", "") or "") if task is not None else "",
            status=status,
            decision=decision,
            result=mirrored,
            phase=str(getattr(state, "phase", "") or ""),
            macro_cycle=int(getattr(state, "macro_cycle", 0) or 0),
            tick=int(getattr(state, "tick", 0) or 0),
        )

    async def _promote_replay_warm_recipe(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Separate promote path so replay doesn't overwrite baseline_tput/current_best."""
        try:
            self._promote_warm_replay(result, task=task)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("warm-replay promote failed")
        # Mirrored after the ruling, so the canonical streams carry the verdict
        # the run actually acted on. This record is the whole point of the fix,
        # so a failure here is surfaced rather than swallowed at debug: the
        # original bug was found only by hand-scanning sessions.
        try:
            self._mirror_warm_replay_verdict(result, task)
        except Exception:  # noqa: BLE001 — best-effort recording
            log.exception("warm-replay v4 verdict capture failed")
        # PRELUDE initial roofline was deferred while replay ran.
        await self._maybe_enqueue_prelude_initial_analysis_after_baseline()

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
            if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        else:
            audit_decision = "promoted"
            audit_extras = {
                "trace_path": None,
                "profile_args": None,
                "output_throughput": result.get("output_throughput"),
            }
        # Surface the trace path so Orch passes a real path to trace_analyze.
        trace_path = result.get("main_trace_path") or (result.get("trace_files") or [None])[0]
        profile_status = str(result.get("status") or "")
        if profile_status == "failed" or result.get("error_class") == "no_trace_files":
            self.shared_state.last_profile_status = "failed"
            self.shared_state.last_profile_workload = {}
            if not trace_path:
                self.shared_state.last_profile_trace = ""
            self.shared_state.last_profile_args = ""
            self.shared_state.last_profile_workload_action = ""
            changed = True
        elif trace_path:
            self.shared_state.last_profile_trace = str(trace_path)
            self.shared_state.last_profile_status = "succeeded"
            # Record the server config in effect for this trace, tagged with the
            # arm it measured so a later same-arm check can trust it.
            profile_args = ""
            if task is not None:
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
            # New trace invalidates the stale trace_analyze cache.
            self.shared_state.last_trace_analyze = {}
            changed = True
            audit_extras["trace_path"] = str(trace_path)
            audit_extras["profile_args"] = profile_args
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
        if profile_status == "succeeded":
            anchor_tput = self._current_tput_from_validated_gain()
            if anchor_tput > 0:
                self.shared_state.last_roofline_tput = float(anchor_tput)
                changed = True
        if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
            self.shared_state.auto_roofline_pending_task_id = ""
            changed = True
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

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
            # Still clear the pending pointer so the watermark check can re-arm.
            if task is not None and self.shared_state.auto_roofline_pending_task_id == task.task_id:
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        elif status == "succeeded":
            audit_decision = "promoted"
            # Prefer the executor's last_trace_analyze snapshot over the result dict.
            _last_ta = self.shared_state.last_trace_analyze or {}
            audit_extras = {
                "snapshot_id": (
                    _last_ta.get("roofline_snapshot_id")
                    if _last_ta.get("roofline_snapshot_id") is not None
                    else result.get("snapshot_id")
                ),
                "last_profile_trace": (self.shared_state.last_profile_trace or result.get("last_profile_trace")),
                "analysis_md_path": (_last_ta.get("analysis_md_path") or result.get("analysis_md_path")),
                "profile_workspace": result.get("profile_workspace"),
                "degraded": bool(result.get("degraded", False)),
            }
            # Reset the roofline failure streak on a successful snapshot.
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak = 0
            # Re-anchor the 10% watermark step on the projected current tput --
            # but only for a roofline that actually produced an analysis. The
            # anchor is what stops the watermark firing again until throughput
            # climbs another 10%, so anchoring on an empty one buys a whole
            # cycle of silence for a snapshot that says nothing: the specialist
            # keeps reading "(none)" while the anchor insists a roofline was
            # taken here. Leaving the anchor alone lets the watermark re-arm and
            # take a real one.
            if str((self.shared_state.last_trace_analyze or {}).get("analysis_md_text") or ""):
                anchor_tput = self._current_tput_from_validated_gain()
                if anchor_tput > 0:
                    self.shared_state.last_roofline_tput = float(anchor_tput)
            else:
                log.warning(
                    "roofline %s produced no analysis; leaving the watermark "
                    "anchor at %.4f so a real one can still be taken",
                    task.task_id if task else "?",
                    float(self.shared_state.last_roofline_tput or 0.0),
                )
            changed = True
        else:
            audit_decision = "discarded"
            audit_extras = {
                "phase": result.get("phase"),
                "error_class": result.get("error_class"),
                "error": result.get("error"),
            }
            # Bump the failure streak (mirrors the audit ledger for prompt renderers).
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            changed = True
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
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        # Winners arrive already graded by the executor, on the decision round
        # that judged them; Coordinator is single-writer for explore_search.accepted +
        # current_best + optimization_stack. The lift still refuses a winner that
        # no longer beats the live anchor.
        # 1. Apply the executor's ledger increment.
        update = result.get("explore_search_update")
        if isinstance(update, dict):
            self.shared_state.apply_explore_search_update(update)
            changed = True
        # 2. Search-space expansion bookkeeping (honoured defensively when an update is present).
        disc_update = result.get("discovered_flags_update")
        if isinstance(disc_update, dict):
            self.shared_state.record_discovered_flags(
                framework=str(disc_update.get("framework") or ""),
                backend_flags=disc_update.get("backend_flags"),
                param_flags=disc_update.get("param_flags"),
                source_path=str(disc_update.get("source_path") or ""),
            )
            err = disc_update.get("discovery_error")
            if err:
                self.shared_state.discovered_flags_error = str(err)
            changed = True
        # Per-lever attribution from this round. Recorded regardless of whether
        # the lever variant won: a rewrite measured at +0.1% is as useful to know
        # as one measured at +8%, and without the number the lever would be
        # re-benched every round.
        for attribution in result.get("framework_lever_attributions") or []:
            if not isinstance(attribution, dict):
                continue
            if self.shared_state.record_framework_lever_attribution(
                str(attribution.get("switch") or ""),
                gain_pct=attribution.get("gain_pct"),
                source=str(attribution.get("source") or ""),
            ):
                changed = True
        # 3. Per-winner record_explore_accepted (Coordinator is sole writer).
        winners = result.get("winners") or []
        round_id = str(result.get("round_id") or "")
        best_winner = result.get("best_variant")
        best_tput = result.get("output_throughput")
        promoted = False
        # A post-resume revalidation task confirms the EXISTING stack/current
        # best rather than adding a variant, so it never "promotes".
        # Reconcile the validation watermark + clear the
        # ``resume_pending_revalidation`` flag from the measured tput — but
        # ONLY when the rebench actually produced a valid measurement, so a
        # failed/empty rebench leaves the flag set and reports keep warning.
        is_revalidation_task = task is not None and str((task.params or {}).get("source") or "") in {
            "resume_stack_revalidate",
            "resume_reverify_best",
        }
        if is_revalidation_task:
            measured = result.get("output_throughput")
            measured_ok = isinstance(measured, (int, float)) and measured > 0
            # A GEAK revalidation (2b) must assert config identity + that the
            # optimization engaged before stamping validated, else replay via
            # the GEAK harness (2a). Native revalidations keep the
            # unconditional watermark reconciliation below.
            if bool((task.params or {}).get("geak_fallback")):
                got_hash = ""
                if isinstance(best_winner, dict):
                    got_hash = str(best_winner.get("fingerprint") or "")
                if not got_hash and isinstance(winners, list) and winners and isinstance(winners[0], dict):
                    got_hash = str(winners[0].get("fingerprint") or "")
                cb_now = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
                cb_tput = cb_now.get("tput")
                decision = _geak_revalidation_decision(
                    measured=measured,
                    baseline=self.shared_state.baseline_tput,
                    got_hash=got_hash,
                    expected_hash=str((task.params or {}).get("expected_cfg_hash") or ""),
                    min_engaged_gain_pct=_MIN_KERNEL_ENGAGED_GAIN_PCT,
                    current_best=cb_tput,
                )
                # ``expected_cfg_hash`` fingerprints (args, envs) only, so it
                # cannot see the overlay drop out between dispatch and launch —
                # ``run_grid`` skips an overlay whose dir has gone away and logs
                # a warning, and the run then measures plain flags while the
                # credit still reads as a kernel win. Re-check the overlay's own
                # identity here; a miss is inconclusive, not validated.
                expected_overlay = str((task.params or {}).get("expected_overlay") or "")
                overlay_loaded = True
                if expected_overlay:
                    expected_digest = str((task.params or {}).get("expected_overlay_digest") or "")
                    got_digest = _geak_overlay_digest(expected_overlay)
                    overlay_loaded = _geak_overlay_is_loadable(expected_overlay) and (got_digest == expected_digest)
                    if not overlay_loaded and decision == "validated":
                        log.warning(
                            "geak 2b: overlay %r did not survive the run "
                            "(loadable=%r digest expected=%r got=%r) -> 2a fallback",
                            expected_overlay,
                            _geak_overlay_is_loadable(expected_overlay),
                            expected_digest,
                            got_digest,
                        )
                        decision = "fallback"
                ps = (
                    self.shared_state.geak_result
                    if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                    else {}
                )
                # A rebench that beats current_best is only a KERNEL gain when
                # GEAK actually produced something. Without a material product
                # (kernel/head/overlay/patch or a config delta vs the pre-KERNEL
                # best) the win is same-config measurement noise; drop it.
                # An empty geak_result cannot be judged by the helper, so
                # disambiguate here: a pre-existing ``geak_e2e`` stack entry
                # means this is a resume revalidation of an already-material win
                # (let it through); otherwise there is no material to validate.
                if decision == "validated":
                    stack_now = self.shared_state.optimization_stack or []
                    has_prior_geak_e2e = any(isinstance(e, dict) and e.get("action") == "geak_e2e" for e in stack_now)
                    # Escape hatch first: a pre-existing geak_e2e stack entry is
                    # an already-proven win, so this is a resume revalidation.
                    # It must short-circuit the material check regardless of
                    # geak_result (which is persisted and thus non-empty on
                    # resume) — by now current_best already carries the GEAK
                    # accepted_config, so the fingerprint would match and be
                    # mis-judged no_material, reverting a real win. Only apply
                    # the material gate on the FIRST validation (no prior entry),
                    # where current_best is still the pre-KERNEL config.
                    if not has_prior_geak_e2e:
                        if not ps:
                            decision = "no_material"
                        elif not _geak_result_has_material(
                            ps,
                            prev_best_flags=str(cb_now.get("extra_server_args") or ""),
                            prev_best_envs=cb_now.get("extra_envs") or {},
                        ):
                            decision = "no_material"
                pending = getattr(self.shared_state, "geak_pending", None) or {}
                pending_tid = str(pending.get("revalidation_task_id") or "") if isinstance(pending, dict) else ""
                from ..phases.geak_rebench import geak_rebench_should_apply_result

                macro_cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
                pending_status = str(pending.get("status") or "") if isinstance(pending, dict) else ""
                if not geak_rebench_should_apply_result(self.shared_state, task, macro_cycle=macro_cycle):
                    # The slot either names another task or already carries a
                    # verdict, so this result is orphaned or late. Record it:
                    # silently dropping a measured rebench is hard to diagnose.
                    log.warning(
                        "geak 2b: ignoring %s result from rebench task %s not tracked by "
                        "geak_pending (pending_task=%s status=%s)",
                        decision,
                        task.task_id,
                        pending_tid or "<unset>",
                        pending_status or "<unset>",
                    )
                    try:
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "geak_rebench_result_ignored",
                                "decision": decision,
                                "task_id": task.task_id,
                                "idempotency_key": str(task.idempotency_key or ""),
                                "pending_task_id": pending_tid,
                                "pending_status": pending_status,
                                "measured_tput": (float(measured) if isinstance(measured, (int, float)) else None),
                            },
                        )
                    except Exception:  # noqa: BLE001 - observation is best-effort
                        log.exception("geak orphan rebench: observation emit failed")
                elif decision == "validated":
                    # Write the headline from the measured orchestrator-harness
                    # rebench: lift current_best + optimization_stack + the
                    # validated gain and clear geak_pending.
                    rebench_measurement = (
                        best_winner
                        if isinstance(best_winner, Mapping)
                        else next(
                            (winner for winner in winners if isinstance(winner, Mapping)),
                            None,
                        )
                    )
                    self._promote_geak_from_candidate(
                        ps,
                        measured_tput=float(measured),
                        provenance="geak_orch_harness_validated",
                        # Only an overlay that was dispatched AND still matches
                        # its manifest proves a kernel was in the measurement.
                        overlay_loaded=bool(expected_overlay) and overlay_loaded,
                        measurement_provenance=rebench_measurement,
                    )
                elif decision == "no_material":
                    # No material GEAK product; the rebench beating current_best
                    # is same-config measurement noise. Do not touch the
                    # headline / stack / gain; record + clear the candidate.
                    log.info(
                        "geak 2b rebench beat current_best but GEAK shipped no "
                        "material product (measured=%r current_best=%r) -> "
                        "no_material drop",
                        measured,
                        cb_tput,
                    )
                    try:
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "geak_no_material",
                                "measured_tput": float(measured),
                                "current_best_tput": (float(cb_tput) if isinstance(cb_tput, (int, float)) else None),
                                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
                            },
                        )
                    except Exception:  # noqa: BLE001 - observation is best-effort
                        log.exception("geak no_material: observation emit failed")
                    # Stamp the drop on geak_result (always, so an empty {} is
                    # distinguishable from never-populated on resume/debug and
                    # acts as a tombstone against KERNEL crash-recovery
                    # re-enqueue) and reject any provisional KEEP in
                    # kernel_journey so a session audit does not read a dropped
                    # candidate as an accepted kernel (no-op when the journey
                    # has no KEEP / the file is absent).
                    ps_stamped = dict(ps) if isinstance(ps, dict) else {}
                    ps_stamped["revalidation_status"] = "no_material"
                    self.shared_state.geak_result = ps_stamped
                    try:
                        self.phase_kernel._reject_geak_kernel_journey(
                            ps_stamped,
                            measured_tput=float(measured),
                            current_best_tput=(float(cb_tput) if isinstance(cb_tput, (int, float)) else 0.0),
                            provenance="geak_no_material",
                            rejection_reason="geak_no_material_product",
                        )
                    except Exception:  # noqa: BLE001 - journey reject is best-effort
                        log.exception("geak no_material: journey rejection failed")
                    self.shared_state.geak_pending = {}
                    self.shared_state.resume_pending_revalidation = False
                elif decision == "no_promote":
                    # Well-measured + engaged over baseline, but does not beat
                    # current_best. This is a real result, NOT inconclusive, so
                    # do not replay via the GEAK harness (2a); clear the pending
                    # candidate without touching the headline / stack / gain.
                    log.info(
                        "geak 2b rebench did not beat current_best (measured=%r current_best=%r) -> no_promote",
                        measured,
                        cb_tput,
                    )
                    try:
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "geak_no_promote",
                                "measured_tput": float(measured),
                                "current_best_tput": (float(cb_tput) if isinstance(cb_tput, (int, float)) else None),
                                "baseline_tput": float(self.shared_state.baseline_tput or 0.0),
                            },
                        )
                    except Exception:  # noqa: BLE001 - observation is best-effort
                        log.exception("geak no_promote: observation emit failed")
                    # Persist the closed verdict so a later KERNEL entry does
                    # not recover stale result.json and re-enqueue this already
                    # adjudicated candidate (#1240).
                    ps_stamped = dict(ps) if isinstance(ps, dict) else {}
                    ps_stamped["revalidation_status"] = "no_promote"
                    self.shared_state.geak_result = ps_stamped
                    self.shared_state.geak_pending = {}
                    self.shared_state.resume_pending_revalidation = False
                else:
                    # 2b inconclusive -> GEAK harness replay (2a), which
                    # clears the pending flag on success. Best-effort.
                    log.warning(
                        "geak 2b revalidation inconclusive "
                        "(measured=%r got_hash=%r expected=%r) -> GEAK-harness 2a fallback",
                        measured,
                        got_hash,
                        (task.params or {}).get("expected_cfg_hash"),
                    )
                    fallback_result: dict[str, Any]
                    try:
                        # Routed via ``_coord`` so a test / caller that overrides
                        # ``coordinator._validate_geak_via_geak_harness`` still wins
                        # (bare-name delegation resolves it back onto this class).
                        fallback_result = await self._coord._validate_geak_via_geak_harness(reason="2b_inconclusive")
                    except Exception as exc:  # noqa: BLE001 - defensive
                        log.exception("geak 2a GEAK-harness fallback failed")
                        fallback_result = {
                            "validated": False,
                            "reason": repr(exc),
                        }
                        try:
                            from hyperloom.inference_optimizer.breakdown.recorder import instrument

                            geak_result = (
                                self.shared_state.geak_result
                                if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                                else {}
                            )
                            instrument.record_geak_operation(
                                self.session_dir,
                                stage="final_validation_failed",
                                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                                result={
                                    **geak_result,
                                    "failure_reason": "geak_harness_fallback_exception",
                                    "error": repr(exc),
                                },
                                status="failed",
                                validation_source="geak_same_harness_geak",
                            )
                        except Exception:  # noqa: BLE001
                            log.debug("geak v4 fallback-exception recording failed", exc_info=True)
                    if not bool(fallback_result.get("validated")):
                        geak_result = (
                            dict(self.shared_state.geak_result)
                            if isinstance(getattr(self.shared_state, "geak_result", None), dict)
                            else {}
                        )
                        geak_result["revalidation_status"] = "fallback_failed"
                        geak_result["revalidation_error"] = str(
                            fallback_result.get("reason")
                            or fallback_result.get("status")
                            or "GEAK harness fallback did not validate"
                        )[:500]
                        self.shared_state.geak_result = geak_result
                        self.shared_state.geak_pending = {}
                        self.shared_state.resume_pending_revalidation = False
                changed = True
            else:
                if measured_ok and self.shared_state.baseline_tput > 0:
                    self._update_cumulative_gain_validated(measured)
                    cb_rec = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
                    recorded = cb_rec.get("tput")
                    floor = _DEFAULT_RESUME_DRIFT_FLOOR_PCT
                    if (
                        isinstance(recorded, (int, float))
                        and recorded > 0
                        and float(measured) < float(recorded) * floor / 100.0
                    ):
                        await self._record_observation(
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
                if measured_ok:
                    self.shared_state.resume_pending_revalidation = False
                changed = True
        # A revalidation task only CONFIRMS the existing stack/current_best; its
        # winner is not a new discovery. Skip the accept/lift path so a rebench
        # (e.g. geak_revalidate) never appends a duplicate optimization_stack
        # entry or re-lifts current_best.
        if isinstance(winners, list) and winners and not is_revalidation_task:
            for winner in winners:
                if not isinstance(winner, dict):
                    continue
                accepted = dict(winner)
                accepted.setdefault("accepted_at_round", round_id)
                accepted.setdefault("provenance", winner.get("provenance") or "llm_direct")
                self.shared_state.record_explore_accepted(accepted)
                # A specialist-provenance KEEP zeroes that domain's rounds_since_last_keep counter.
                prov = str(accepted.get("provenance") or "")
                if prov.startswith("specialist:"):
                    try:
                        self.shared_state.note_domain_keep(prov.split(":", 1)[1].strip())
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "depth: note_domain_keep failed for provenance=%r",
                            prov,
                        )
                changed = True
            # 4. Lift every applied winner into current_best / optimization_stack in
            # application order.  Winners are applied cumulatively inside the executor
            # (running_base_tput advances with each in-batch KEEP), so output_throughput
            # reflects the full stack.  Lifting only the highest-gain winner credited
            # that stacked throughput to a config missing the others' args, and the
            # missed winners' recipe_deltas never reached the ledger at all.
            # Each winner carries its own tput from the round that graded it.
            # Because KEEP requires a positive gain over the advancing running base, each
            # winner's tput is strictly greater than the previous one, so the anchor
            # check inside _lift_to_current_best clears for every in-round winner.
            explore_gap_cid = str((task.params or {}).get("gap_canonical_id") or "").strip() if task is not None else ""
            for winner in winners:
                if not isinstance(winner, dict):
                    continue
                winner_tput = winner.get("tput")
                if not isinstance(winner_tput, (int, float)) or float(winner_tput) <= 0:
                    continue
                entry = dict(winner)
                if task is not None:
                    entry["task_id"] = str(task.task_id or "")
                if self._lift_to_current_best(
                    "explore",
                    float(winner_tput),
                    entry,
                    gap_canonical_id=explore_gap_cid,
                ):
                    promoted = True
            changed = True
        try:
            self.shared_state.note_explore_outcome(promoted=promoted)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("depth: note_explore_outcome failed")
        if promoted:
            # A KEEP's own measurement promotes into cumulative_gain_validated and
            # advances validated_stack_len so the unvalidated-stack guard clears.
            # An explore round grades a variant on its decision round and reports
            # that, so the basis names the round the number came from.
            if self.shared_state.baseline_tput > 0 and isinstance(best_tput, (int, float)) and best_tput > 0:
                self._update_cumulative_gain_validated(best_tput, measurement_basis="e2e_decision_round")
                # Watermark refresh: enqueue a fresh roofline once projected tput crosses +10%.
                await self._maybe_enqueue_watermark_roofline(
                    reason="explore_keep_watermark",
                )
        else:
            changed = True
        if promoted:
            audit_decision = "promoted"
        elif winners and not is_revalidation_task:
            audit_decision = "no_promote"
        else:
            audit_decision = "discarded"
        audit_extras = {
            "round_id": round_id,
            "winners_count": (len(winners) if isinstance(winners, list) else 0),
            "losers_count": len(result.get("losers") or []),
            "skipped_dup_count": len(result.get("skipped_dup") or []),
            "best_variant_name": (best_winner.get("name") if isinstance(best_winner, dict) else None),
            "best_gain_pct_vs_base": result.get("best_gain_pct"),
            "output_throughput": best_tput,
            "explore_grid_exhausted": bool(result.get("explore_grid_exhausted")),
        }
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_integrate_patch(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote an integrate_patch result: on KEEP lift current_best; clear pending_integrate."""
        changed = False
        stack_len_before = len(self.shared_state.optimization_stack or [])
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        status = str(result.get("status") or "")
        new_tput = result.get("output_throughput")
        kept_flag = status == "kept" and isinstance(new_tput, (int, float)) and float(new_tput) > 0
        # Register framework-rewrite switches as search levers. Done for both KEEP
        # verdicts: a bundle that cleared the gate is on and gets leave-one-out
        # attribution, while an inert KEEP is dormant and gets additive
        # attribution. ``kept_inert`` deliberately does not lift current_best —
        # the code is applied but every switch is off, so the running
        # configuration is unchanged.
        levers = result.get("framework_levers")
        if isinstance(levers, list) and levers:
            lever_outcome = str(result.get("framework_lever_outcome") or "")
            if self.shared_state.record_authored_framework_levers(
                levers,
                default_on=(lever_outcome == "default_on"),
                specialist_task_id=str(result.get("specialist_task_id") or ""),
                stack_delta_pct=result.get("delta_pct"),
            ):
                changed = True
            audit_extras["framework_levers"] = [str(row.get("switch") or "") for row in levers]
            audit_extras["framework_lever_outcome"] = lever_outcome
        task_params = (getattr(task, "params", None) or {}) if task is not None else {}
        enablement_landing = bool(
            result.get("enablement") or task_params.get("enablement") or task_params.get("enablement_landing")
        )
        prebaseline_enablement = bool(
            kept_flag and float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0) <= 0.0 and enablement_landing
        )
        lifted = False
        if kept_flag:
            specialist_task_id = str(result.get("specialist_task_id") or task_params.get("specialist_task_id") or "")
            origin_domain = str(
                task_params.get("domain") or task_params.get("source_domain") or result.get("domain") or ""
            ).strip()
            origin_provenance = str(task_params.get("provenance") or result.get("provenance") or "").strip()
            if origin_domain and not origin_provenance.startswith("specialist:"):
                origin_provenance = f"specialist:{origin_domain}"
            source_phase = str(task_params.get("source_phase") or result.get("source_phase") or "").strip()
            gap_canonical_id = str(task_params.get("gap_canonical_id") or result.get("gap_canonical_id") or "").strip()
            lift = {
                "name": specialist_task_id or "integrate_patch_keep",
                "task_id": getattr(task, "task_id", "") if task is not None else "",
                "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
                "candidate_extra_envs": dict(
                    result.get("extra_envs_applied") or result.get("config_changes_applied") or {}
                ),
                "recipe_delta": {
                    "extra_server_args": str(result.get("extra_server_args_applied") or ""),
                    "extra_envs": dict(result.get("extra_envs_applied") or result.get("config_changes_applied") or {}),
                    "remove_args": to_str_list(result.get("remove_args_applied") or task_params.get("remove_args")),
                    "unset_envs": to_str_list(result.get("unset_envs_applied") or task_params.get("unset_envs")),
                    "args_mode": str(result.get("args_mode") or task_params.get("args_mode") or "append")
                    .strip()
                    .lower(),
                },
                "extra_envs": dict(result.get("extra_envs_applied") or result.get("config_changes_applied") or {}),
                "tput": float(new_tput),
                "workspace": result.get("workspace"),
                "provenance": origin_provenance or "integrate_patch",
                "scope": "source_patch",
                # Durable source-layer handles so current_best stays relaunchable
                # and reproducible in the GEAK baseline.
                **_source_layer_handles(result),
            }
            # ``IntegratePatchExecutor`` nests benchmark output in
            # ``bench_result``. Preserve its launch evidence when lifting the
            # winning measurement; otherwise the handoff is correctly marked
            # unverified despite the grid runner having captured proof.
            bench_result = result.get("bench_result")
            if not isinstance(bench_result, Mapping):
                bench_result = {}
            for evidence_field in (
                "launch_evidence",
                "launch_evidence_path",
                "server_log_path",
            ):
                value = result.get(evidence_field) or bench_result.get(evidence_field)
                if value:
                    lift[evidence_field] = value
            # The mandate's stamp and the deliverable's markers together;
            # the result wins on a collision.
            lever_kind = _lever_for_keep(task_params, result)
            if lever_kind:
                lift["lever_kind"] = lever_kind
            if source_phase:
                lift["source_phase"] = source_phase
            if origin_domain:
                lift["domain"] = origin_domain
            if task_params.get("gap_layer"):
                lift["gap_layer"] = str(task_params.get("gap_layer"))
            if task_params.get("framework_agent_authoring"):
                lift["framework_agent_authoring"] = True
            if enablement_landing:
                lift["recipe_publishable"] = False
            if prebaseline_enablement:
                lift["baseline_enablement"] = True
                lift["attribution_eligible"] = False
                # This patch establishes the runnable baseline environment. Keep
                # it in the configuration stack for reproducibility, but mark it
                # ineligible for gain attribution because no runnable before
                # measurement exists.
                lifted = self._lift_to_current_best(
                    "integrate_patch",
                    float(new_tput),
                    lift,
                    gap_canonical_id=gap_canonical_id,
                )
                log.info(
                    "integrate_patch KEEP accepted as pre-baseline enablement; "
                    "retained in config stack without gain attribution "
                    "(task=%s specialist=%s)",
                    str(getattr(task, "task_id", "") or ""),
                    specialist_task_id,
                )
            else:
                lifted = self._lift_to_current_best(
                    "integrate_patch",
                    float(new_tput),
                    lift,
                    gap_canonical_id=gap_canonical_id,
                )
                if lifted and self.shared_state.baseline_tput > 0:
                    self._update_cumulative_gain_validated(new_tput)
                    self.shared_state.resume_pending_revalidation = False
                    await self._maybe_enqueue_watermark_roofline(
                        reason="integrate_keep_watermark",
                    )
            changed = True
        # Clear the pending_integrate sentinel after the task outcome is observed.
        if isinstance(getattr(self.shared_state, "pending_integrate", None), dict):
            pending = self.shared_state.pending_integrate
            if not pending or str(pending.get("task_id") or "") in {
                "",
                str(getattr(task, "task_id", "") or ""),
            }:
                self.shared_state.pending_integrate = {}
                changed = True
        if prebaseline_enablement:
            audit_decision = "enablement_accepted" if lifted else "no_promote"
        elif lifted:
            audit_decision = "promoted"
        elif kept_flag:
            audit_decision = "no_promote"
            result[PROMOTION_REFUSED_KEY] = True
        elif status == "kept_inert":
            # Applied but every switch off: nothing was promoted, yet the patch
            # stays on disk as registered levers, so it is not a discard either.
            audit_decision = "kept_inert"
        else:
            audit_decision = "discarded"
        audit_extras = {
            **audit_extras,
            "status": status,
            "specialist_task_id": result.get("specialist_task_id"),
            "output_throughput": new_tput,
            "delta_pct": result.get("delta_pct"),
            "prebaseline_enablement": prebaseline_enablement,
            "accuracy_pass": result.get("accuracy_pass"),
            "patches_applied": result.get("patches_applied") or [],
            "patches_reverted": result.get("patches_reverted") or [],
            # Enablement eval-origin verdict fields for history.
            "correctness_verified": result.get("correctness_verified"),
            "enablement_eval_failure_kind": result.get("enablement_eval_failure_kind"),
            "enablement_observed_accuracy": result.get("enablement_observed_accuracy"),
            "provisional": result.get("provisional"),
        }
        if lifted and not enablement_landing and len(self.shared_state.optimization_stack or []) > stack_len_before:
            owner = _keep_owner_section(task_params, result)
            if owner in {"EXPLORE", "FRAMEWORK_AGENT"}:
                self._enqueue_agent_keep_outbox(
                    owner=owner,
                    stack_index=len(self.shared_state.optimization_stack) - 1,
                    result=result,
                    task=task,
                    include_patches=True,
                )
        outcome.changed = changed
        outcome.audit_decision = audit_decision
        outcome.audit_extras = audit_extras

    async def _promote_sweep(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a sweep result: self-audit + record_sweep + save; discovery-only, never promotes."""
        outcome.early_return = True
        pareto = result.get("pareto_front") or []
        self.shared_state.record_action_attempt(
            action="sweep",
            task_id=getattr(task, "task_id", "") if task is not None else "",
            status="succeeded",
            decision="discarded",
            result=result,
            extras={
                "grid_size": result.get("grid_size"),
                "best_overall": result.get("best_overall"),
                "best_for_each_conc": result.get("best_for_each_conc"),
                "pareto_front_size": (len(pareto) if isinstance(pareto, list) else None),
            },
        )
        self.shared_state.record_sweep(result)
        # Sweep is discovery-only (never promotes) and must not mutate params_no_promote_streak.
        self.shared_state.save(self.session_dir)
        # SWEEP post-hook: chain conc_sweep after a succeeded sweep when opted in.
        if getattr(self.shared_state, "conc_sweep_enabled", False) and result.get("status") == "succeeded":
            try:
                await self._enqueue_internal_conc_sweep_task(
                    reason="post_sweep",
                )
            except Exception:  # noqa: BLE001
                log.exception("conc_sweep: post-sweep enqueue raised (non-fatal)")

    async def _promote_conc_sweep(
        self,
        result: dict,
        task: "Task | None",
        outcome: _PromoteOutcome,
    ) -> None:
        """Promote a conc_sweep result: self-audit + record_conc_sweep + save; discovery-only."""
        outcome.early_return = True
        self.shared_state.record_action_attempt(
            action="conc_sweep",
            task_id=getattr(task, "task_id", "") if task is not None else "",
            status=str(result.get("status") or "succeeded"),
            decision="discarded",
            result=result,
            extras={
                "was_skipped": bool(result.get("was_skipped", False)),
                "skip_reason": result.get("skip_reason"),
                "budget_exhausted": bool(result.get("budget_exhausted", False)),
                "total_budget_sec": result.get("total_budget_sec"),
                "elapsed_sec": result.get("elapsed_sec"),
                "best_speedup": ((result.get("summary") or {}).get("best_speedup")),
                "best_conc": ((result.get("summary") or {}).get("best_conc")),
                "successful_pairs": ((result.get("summary") or {}).get("successful_pairs")),
                "report_path": result.get("report_json_path"),
            },
        )
        # Write last_conc_sweep so exit_normal_sweep can fire conc_sweep_done.
        self.shared_state.record_conc_sweep(result)
        self.shared_state.save(self.session_dir)

    # ------------------------------------------------------------------
    # Resume / replay (folded in from the former ResumeCollaborator).
    # Three semantic boundaries live below: the live-promote / replay path
    # (``_replay_keep_from_result``), the resume-reconcile path
    # (``_resume_consistency_pass`` + its recover helpers), and the
    # current_best lift path (``_current_best_launch_config`` /
    # ``build_env_spec``). Methods keep bare ``self.<name>`` access; tests
    # monkeypatch them via ``coord.writeback.<name>`` (or bare-name
    # ``_DELEGATED`` on the coordinator).
    # ------------------------------------------------------------------
    def _detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking).

        Returns:
            A dict with ``is_resume``, ``event_count``, ``state_json_present``
            and ``rebuilt`` (the last set later by :meth:`replay_for_resume`).
        """
        ev_count = self.bus.db.fetchone_sync("SELECT COUNT(*) AS c FROM events")
        events_present = (int(ev_count["c"]) if ev_count else 0) > 0
        state_path = SharedState.state_path(self.session_dir)
        return {
            "is_resume": events_present or state_path.exists(),
            "event_count": int(ev_count["c"]) if ev_count else 0,
            "state_json_present": state_path.exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }

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

    @staticmethod
    def _handoff_launch_identity(env_spec: Mapping[str, Any]) -> str:
        """Return a stable identity for the complete GEAK baseline launch."""
        config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
        # The recipe PATH is deliberately excluded: a baseline re-run repoints
        # ``baseline_config_path`` at a new timestamped YAML without re-stamping
        # current_best, so hashing the path drops the same-config reference for a
        # byte-identical recipe. The digest already carries the recipe content.
        payload = {
            "base_launch_recipe_digest": str(env_spec.get("base_launch_recipe_digest") or ""),
            "extra_server_args": str(config.get("extra_server_args") or ""),
            "extra_envs": dict(config.get("extra_envs") or {}),
            "server_launch_flags": str(config.get("server_launch_flags") or ""),
            "source_snapshots": list(env_spec.get("source_snapshots") or []),
            "overlay_pythonpath": str(env_spec.get("overlay_pythonpath") or ""),
            "overlay_digest": str(env_spec.get("overlay_digest") or ""),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @staticmethod
    def _launch_recipe_digest(path: str) -> str:
        """Hash the recipe content so an in-place recipe edit invalidates tput."""
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return ""

    def _measurement_launch_flags(self, measurement: Mapping[str, Any]) -> str:
        """Read resolved launch flags only from the promoted measurement's evidence."""
        existing = str(measurement.get("resolved_server_launch_flags") or "").strip()
        if existing:
            return existing
        evidence = measurement.get("launch_evidence")
        if isinstance(evidence, Mapping):
            observed = str(evidence.get("observed_server_launch_flags") or "").strip()
            if observed:
                return observed
        framework = (
            str(
                (evidence.get("framework") if isinstance(evidence, Mapping) else "")
                or os.environ.get("FRAMEWORK", "")
                or "sglang"
            )
            .strip()
            .lower()
        )
        paths = [str(measurement.get("server_log_path") or "").strip()]
        if isinstance(evidence, Mapping):
            paths.append(str(evidence.get("actual_server_log_path") or "").strip())
        for key in ("benchmark_workspace", "workspace"):
            workspace = str(measurement.get(key) or "").strip()
            if workspace:
                root = Path(workspace)
                paths.append(str(root / "server.log"))
        for path in dict.fromkeys(path for path in paths if path):
            flags = launch_argv_from_log(path, framework)
            if flags:
                return flags
        return ""

    @staticmethod
    def _resolved_sglang_server_config(launch_evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve selected SGLang ``ServerArgs`` fields from declared args.

        A captured command line is preferred evidence. This fallback makes the
        declaration inspectable when a reused ready server did not emit a fresh
        CLI line; it is intentionally not treated as observed launch evidence.
        """
        if str(launch_evidence.get("framework") or "").strip().lower() != "sglang":
            return {}
        raw_args = str(
            launch_evidence.get("requested_server_args") or launch_evidence.get("requested_server_flags") or ""
        ).strip()
        model_path = str(launch_evidence.get("model_path") or "").strip()
        try:
            from sglang.srt.server_args import ServerArgs

            parser = argparse.ArgumentParser(add_help=False)
            ServerArgs.add_cli_args(parser)
            tokens = shlex.split(raw_args)
            if model_path and not any(token == "--model-path" or token.startswith("--model-path=") for token in tokens):
                tokens = ["--model-path", model_path, *tokens]
            namespace, _unknown = parser.parse_known_args(tokens)
        except (Exception, SystemExit):  # noqa: BLE001 - optional declared-only evidence
            return {}
        fields = (
            "model_path",
            "tokenizer_path",
            "host",
            "port",
            "tp_size",
            "dp_size",
            "pp_size",
            "mem_fraction_static",
            "context_length",
            "max_total_tokens",
            "max_running_requests",
            "attention_backend",
            "quantization",
            "dtype",
            "disable_cuda_graph",
            "enable_dp_attention",
            "enable_ep_moe",
            "chunked_prefill_size",
            "schedule_policy",
        )
        out: dict[str, Any] = {}
        for field_name in fields:
            value = getattr(namespace, field_name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[field_name] = value
        return out

    @staticmethod
    def _measurement_observed_server_identity(measurement: Mapping[str, Any]) -> dict[str, Any]:
        """Return archived SGLang ``ServerArgs`` evidence from this measurement only."""
        evidence = measurement.get("launch_evidence")
        if isinstance(evidence, Mapping):
            identity = evidence.get("observed_server_identity")
            if isinstance(identity, Mapping) and identity:
                return {str(key): value for key, value in sorted(identity.items())}
        if str((evidence or {}).get("framework") if isinstance(evidence, Mapping) else "").lower() != "sglang":
            return {}
        for path in (
            str(measurement.get("server_log_path") or ""),
            str((evidence or {}).get("actual_server_log_path") if isinstance(evidence, Mapping) else ""),
        ):
            if path:
                identity = observed_sglang_server_identity_from_log(path)
                if identity:
                    return identity
        return {}

    @staticmethod
    def _observed_launch_identity(
        declared_identity: str,
        observed_flags: str,
        observed_server_identity: Mapping[str, Any] | None = None,
    ) -> str:
        """Hash actual captured launch data under the declared identity."""
        if not declared_identity or (not observed_flags and not observed_server_identity):
            return ""
        payload = json.dumps(
            {
                "declared_launch_identity": declared_identity,
                "observed_server_launch_flags": observed_flags,
                "observed_server_identity": dict(observed_server_identity or {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _identity_verification_status(
        *,
        expected_identity: str,
        measurement: Mapping[str, Any],
    ) -> str:
        """Classify whether identity rests on observed or declared evidence."""
        declared_identity = str(measurement.get("declared_launch_identity") or measurement.get("launch_identity") or "")
        if not expected_identity or declared_identity != expected_identity:
            return "unverified"
        evidence = measurement.get("launch_evidence")
        observed = str(measurement.get("resolved_server_launch_flags") or "").strip()
        if not observed and isinstance(evidence, Mapping):
            observed = str(evidence.get("observed_server_launch_flags") or "").strip()
        observed_server_identity = WritebackCollaborator._measurement_observed_server_identity(measurement)
        if observed or observed_server_identity:
            return "verified_observed"
        if isinstance(evidence, Mapping) and (
            str(evidence.get("requested_server_args") or "").strip()
            or bool(evidence.get("requested_server_env"))
            or str(evidence.get("recipe_digest") or "").strip()
        ):
            return "verified_declared_only"
        return "unverified"

    def _stamp_current_best_measurement(self, evidence: Mapping[str, Any] | None = None) -> None:
        """Attach the exact promoted config identity to ``current_best``."""
        cb = self.shared_state.current_best
        if not isinstance(cb, dict):
            return
        evidence = evidence if isinstance(evidence, Mapping) else {}
        launch_evidence = evidence.get("launch_evidence") or {}
        launch_evidence = dict(launch_evidence) if isinstance(launch_evidence, Mapping) else {}
        measurement = {
            "schema_version": 2,
            "tput": float(cb.get("tput") or 0.0),
            "benchmark_workspace": str(
                evidence.get("workspace") or evidence.get("single_workspace") or cb.get("workspace") or ""
            ),
            "server_log_path": str(
                evidence.get("server_log_path") or launch_evidence.get("actual_server_log_path") or ""
            ),
            "launch_evidence": launch_evidence,
            "launch_evidence_path": str(evidence.get("launch_evidence_path") or ""),
        }
        measurement["resolved_server_launch_flags"] = self._measurement_launch_flags(measurement)
        observed_server_identity = self._measurement_observed_server_identity(measurement)
        if observed_server_identity:
            launch_evidence["observed_server_identity"] = observed_server_identity
        measurement["observed_server_identity"] = observed_server_identity
        measurement["resolved_server_config"] = observed_server_identity or self._resolved_sglang_server_config(
            launch_evidence
        )
        declared_identity = str(
            self.build_env_spec(
                measurement=measurement,
                server_launch_flags=measurement["resolved_server_launch_flags"],
            )["launch_identity"]
        )
        measurement["launch_identity"] = declared_identity  # Legacy alias.
        measurement["declared_launch_identity"] = declared_identity
        measurement["observed_launch_identity"] = self._observed_launch_identity(
            declared_identity,
            measurement["resolved_server_launch_flags"],
            observed_server_identity,
        )
        measurement["identity_verification_status"] = self._identity_verification_status(
            expected_identity=declared_identity,
            measurement=measurement,
        )
        # Do not publish a partial measurement. All parsing and identity
        # preparation above is local; these paired assignments are the commit.
        cb["measurement"] = measurement
        self.shared_state.current_best_measurement = measurement

    def _current_best_launch_config(self) -> dict[str, Any]:
        """The launch config ``current_best`` was measured on.

        Returns:
            ``extra_server_args`` / ``extra_envs`` / ``final_overlay``.
        """
        cb = self.shared_state.current_best if isinstance(self.shared_state.current_best, Mapping) else {}
        args = str(cb.get("extra_server_args") or "").strip()
        envs: dict[str, str] = {}
        raw_envs = cb.get("extra_envs")
        if isinstance(raw_envs, Mapping):
            for key, value in raw_envs.items():
                name = str(key)
                # A ``-``-prefixed key is a server arg; exported as an env the
                # backend ignores it and the flag is silently lost.
                if name.startswith("-"):
                    token = name if value in ("", None) else f"{name}={value}"
                    args = _merge_cumulative_extra_server_args(args, token, "")
                else:
                    envs[name] = str(value)
        return {
            "extra_server_args": args,
            "extra_envs": envs,
            "final_overlay": str(cb.get("final_overlay") or "").strip(),
        }

    def build_env_spec(
        self,
        *,
        measurement: Mapping[str, Any] | None = None,
        server_launch_flags: str | None = None,
    ) -> dict[str, Any]:
        """Fully-reproducible descriptor of ``current_best``'s launch environment.

        Layers, in the order a consumer must apply them to reconstruct the exact
        stack ``current_best`` was measured on:

          * ``config``  — cumulative server args + env vars (the reversible layer).
          * ``source_snapshots`` — ordered durable source-layer snapshots
            (``scope=source_patch`` entries), each a self-contained directory
            (see :mod:`source_snapshot`) that reconstructs the patched framework
            tree independent of the mutable live checkout.
          * ``overlay_pythonpath`` — the authored-kernel overlay prefix.
          * ``base_launch_recipe`` — the baseline Magpie recipe to launch from.

        This is the single source of truth the GEAK handoff forwards so the
        baseline ref is materialized from the SAME layers as ``current_best``
        (not just its flags/env), closing the cross-harness baseline gap.
        """
        from ..source_snapshot import source_layer_overlay_dir, source_layer_reproducible

        cb = self.shared_state.current_best if isinstance(self.shared_state.current_best, Mapping) else {}
        materialized = self._current_best_launch_config()
        # current_best's embedded stack was promoted with its tput. Reading the
        # mutable global stack here could attach an unrelated source patch after
        # a resume or partial state write.
        stack = [e for e in (cb.get("optimization_stack") or []) if isinstance(e, dict)]
        source_snapshots: list[dict[str, Any]] = []
        for entry in stack:
            if entry.get("scope") != "source_patch":
                continue
            snap = str(entry.get("source_snapshot") or "").strip()
            if not snap:
                # A source_patch with no durable snapshot (e.g. a pre-fix legacy
                # KEEP) is surfaced so the consumer can flag an unreproducible
                # baseline rather than silently launch a weaker stock tree.
                source_snapshots.append(
                    {
                        "id": str(entry.get("variant_name") or entry.get("name") or ""),
                        "snapshot_dir": "",
                        "framework_root": str(entry.get("framework_root") or ""),
                        "base_sha": str(entry.get("base_sha") or ""),
                        "reproducible": False,
                    }
                )
                continue
            source_snapshots.append(
                {
                    "id": str(entry.get("variant_name") or entry.get("name") or ""),
                    # GEAK PYTHONPATHs this value; it is the import root inside
                    # the snapshot, not the snapshot's own top level.
                    "snapshot_dir": source_layer_overlay_dir(entry),
                    "framework_root": str(entry.get("framework_root") or ""),
                    "base_sha": str(entry.get("base_sha") or ""),
                    "reproducible": source_layer_reproducible(entry),
                }
            )
        # ``extra_server_args``/``extra_envs`` remain the current_best delta.
        # Full engine flags must come from the same promoted measurement, not a
        # search over historical benchmarks with a similar throughput.
        if not isinstance(measurement, Mapping):
            state_measurement = getattr(self.shared_state, "current_best_measurement", None)
            measurement = (
                state_measurement
                if isinstance(state_measurement, Mapping) and state_measurement
                else (cb.get("measurement") if isinstance(cb.get("measurement"), Mapping) else {})
            )
        if server_launch_flags is None:
            server_launch_flags = self._measurement_launch_flags(measurement)
        env_spec = {
            "schema_version": 1,
            "config": {
                "extra_server_args": materialized.get("extra_server_args") or "",
                "extra_envs": dict(materialized.get("extra_envs") or {}),
                # Authoritative, COMPLETE engine flags (run-specific stripped);
                # empty => consumer keeps its own adapter defaults (prior behavior).
                "server_launch_flags": server_launch_flags,
            },
            "source_snapshots": source_snapshots,
            "overlay_pythonpath": materialized.get("final_overlay") or "",
            "overlay_digest": _geak_overlay_digest(str(materialized.get("final_overlay") or "")),
            "base_launch_recipe": str(getattr(self.shared_state, "baseline_config_path", "") or ""),
            "base_launch_recipe_digest": self._launch_recipe_digest(
                str(getattr(self.shared_state, "baseline_config_path", "") or "")
            ),
            # Backward-compatible alias for existing GEAK v2 consumers.
            "launch_recipe": str(getattr(self.shared_state, "baseline_config_path", "") or ""),
            # Additive evidence record. Consumers can distinguish a captured
            # launch from a declaration resolved without a new CLI line.
            "measurement_evidence": dict(measurement.get("launch_evidence") or {}),
            "measurement_identity": {
                "declared_launch_identity": str(
                    measurement.get("declared_launch_identity") or measurement.get("launch_identity") or ""
                ),
                "observed_launch_identity": str(measurement.get("observed_launch_identity") or ""),
                "verification_status": str(measurement.get("identity_verification_status") or "unverified"),
                "observed_server_identity": dict(measurement.get("observed_server_identity") or {}),
                "resolved_server_config": dict(measurement.get("resolved_server_config") or {}),
            },
        }
        env_spec["launch_identity"] = self._handoff_launch_identity(env_spec)
        return env_spec

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
        active_inferencex = str(getattr(state, "active_inferencex_path", "") or "").strip()
        if active_inferencex:
            if Path(active_inferencex).is_dir():
                os.environ["INFERENCEX_PATH"] = active_inferencex
            else:
                report["warnings"].append(
                    {
                        "kind": "active_inferencex_checkout_missing",
                        "path": active_inferencex,
                    }
                )
                if hasattr(state, "set_stop_reason"):
                    state.set_stop_reason("active_inferencex_checkout_missing")
        # (1) Half-applied integrate window: replay the
        # missing stack append or roll back the partial patch BEFORE anything
        # reads the stack, so the rest of the pass sees the recovered truth.
        await self._resume_recover_pending_integrate(report)
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

        # Persist recovered stack/current_best before materializing their KB
        # sections. The outbox must never publish a config that the state file
        # has not made authoritative yet.
        resume_state_durable = True
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            resume_state_durable = False
            log.exception("Coordinator: pre-outbox resume save failed")
            report["warnings"].append({"kind": "resume_pre_outbox_save_failed"})
        pending_kb_before = len(getattr(state, "kb_stage_outbox", []) or [])
        if pending_kb_before and resume_state_durable:
            self._drain_agent_keep_outbox()
            pending_kb_after = len(getattr(state, "kb_stage_outbox", []) or [])
            if pending_kb_after:
                report["warnings"].append(
                    {
                        "kind": "kb_stage_outbox_incomplete",
                        "pending": pending_kb_after,
                    }
                )
            else:
                report["fixes"].append(
                    {
                        "kind": "reconciled_kb_stage_outbox",
                        "count": pending_kb_before,
                    }
                )

        # (4) Validation-watermark compensation: unvalidated
        # KEEPs (claimed gain not yet end-to-end confirmed) → flag + enqueue ONE
        # full-stack rebench. The flag + watermark are reconciled from the
        # measured tput when that rebench promotes (see _promote_to_shared_state).
        stack = [e for e in (getattr(state, "optimization_stack", []) or []) if isinstance(e, dict)]
        vlen = int(getattr(state, "cumulative_gain_validated_stack_len", 0) or 0)
        if vlen < len(stack):
            state.resume_pending_revalidation = True
            report["warnings"].append(
                {
                    "kind": "resume_unvalidated_keeps",
                    "validated_stack_len": vlen,
                    "stack_len": len(stack),
                }
            )
            try:
                fix = await self._enqueue_internal_stack_rebench(reason="resume_unvalidated_keeps")
                report["fixes"].append({"kind": "queued_resume_stack_rebench", **fix})
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: failed to enqueue resume stack rebench")
                report["warnings"].append({"kind": "resume_stack_rebench_enqueue_failed"})

        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: resume consistency save failed")
        await self._record_observation("coordinator", "observation", {"kind": "resume_consistency", **report})
        return report

    def _replay_keep_from_result(self, kind: str, result: dict[str, Any]) -> bool:
        """Replay a recorded KEEP delegated-result into current_best/stack.

        Reconstructs the winning-variant dict from a persisted ``delegated_result``
        and routes it through :meth:`_lift_to_current_best`, which dedupes by
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
            bv_src = result.get("best_variant")
            if not isinstance(bv_src, dict) or not bv_src.get("name"):
                return False
            bv = dict(bv_src)
        elif kind == "integrate_patch":
            sid = str(result.get("specialist_task_id") or "")
            if not sid:
                return False
            domain = str(result.get("domain") or result.get("source_domain") or "").strip()
            provenance = str(result.get("provenance") or "").strip()
            if domain and not provenance.startswith("specialist:"):
                provenance = f"specialist:{domain}"
            bv = {
                "name": sid,
                "candidate_extra_server_args": str(result.get("extra_server_args_applied") or ""),
                "candidate_extra_envs": dict(
                    result.get("extra_envs_applied") or result.get("config_changes_applied") or {}
                ),
                "recipe_delta": {
                    "extra_server_args": str(result.get("extra_server_args_applied") or ""),
                    "extra_envs": dict(result.get("extra_envs_applied") or result.get("config_changes_applied") or {}),
                    "remove_args": to_str_list(result.get("remove_args_applied") or result.get("remove_args")),
                    "unset_envs": to_str_list(result.get("unset_envs_applied") or result.get("unset_envs")),
                    "args_mode": str(result.get("args_mode") or "append").strip().lower(),
                },
                "extra_envs": dict(result.get("extra_envs_applied") or result.get("config_changes_applied") or {}),
                "tput": float(tput),
                "workspace": result.get("workspace"),
                "provenance": provenance or "integrate_patch",
                "scope": "source_patch",
                # Same durable source-layer handles as the primary KEEP lift so a
                # source_patch recovered on THIS path is equally reproducible in
                # the GEAK baseline (no path is left snapshot-less).
                **_source_layer_handles(result),
            }
            replay_lever = patch_lever_kind(result)
            if replay_lever:
                bv["lever_kind"] = replay_lever
            source_phase = patch_owner_phase(result)
            gap_layer = str(result.get("gap_layer") or "").strip()
            if source_phase:
                bv["source_phase"] = source_phase
            else:
                bv["recipe_publishable"] = False
            if domain:
                bv["domain"] = domain
            if gap_layer:
                bv["gap_layer"] = gap_layer
            if result.get("framework_agent_authoring"):
                bv["framework_agent_authoring"] = True
            if result.get("enablement") or result.get("enablement_landing"):
                bv["recipe_publishable"] = False
        else:
            return False
        before = len(self.shared_state.optimization_stack or [])
        if not self._lift_to_current_best(
            kind,
            float(tput),
            bv,
            gap_canonical_id=str(result.get("gap_canonical_id") or ""),
        ):
            return False
        return len(self.shared_state.optimization_stack or []) > before

    def _resume_rollback_pending_integrate(self, pending: dict[str, Any]) -> dict[str, Any]:
        """Reverse-apply a half-applied integrate patch set (Gap C rollback).

        Best-effort ``git apply -R`` of every patch recorded on the
        ``pending_integrate`` sentinel into the framework source tree, so a
        crash AFTER ``git apply`` but BEFORE the bench/KEEP cannot leak a partial
        change into later launches. A patch that is not currently applied simply
        fails the reverse ``--check`` and is reported, not retried.

        Args:
            pending: The ``pending_integrate`` sentinel dict.

        Returns:
            A summary ``{"reversed": [...], "failed": [...]}``.
        """
        from ..actions.executors.integrate_patch import _git_apply_reverse

        summary: dict[str, Any] = {"reversed": [], "failed": []}
        # Discard a half-provisioned attempt venv so a crash mid-provision
        # cannot leak a multi-GB dir. Independent of the patch rollback below.
        attempt_venv_root = str(pending.get("attempt_venv_root") or "").strip()
        if attempt_venv_root:
            gc_root = str(Path(attempt_venv_root).parent)
            if self._gc_attempt_runtime(gc_root):
                summary["attempt_runtime_gc"] = gc_root
        root = str(pending.get("framework_source_root") or "").strip()
        patches = [str(p) for p in (pending.get("patches") or []) if str(p).strip()]
        if not root or not patches:
            return summary
        root_path = Path(root)
        for patch in patches:
            try:
                ok, err = _git_apply_reverse(root_path, Path(patch))
            except Exception as exc:  # noqa: BLE001 — rollback is best-effort
                summary["failed"].append({"patch": patch, "error": repr(exc)})
                continue
            if ok:
                summary["reversed"].append(patch)
            else:
                summary["failed"].append({"patch": patch, "error": err})
        return summary

    @staticmethod
    def _gc_attempt_runtime(attempt_dir: str) -> bool:
        """Remove an attempt-runtime dir (best-effort).

        Returns True when a directory was present and removal was attempted.
        """
        import shutil

        path = Path(str(attempt_dir or "").strip())
        if not attempt_dir or not path.exists():
            return False
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception:  # noqa: BLE001 — GC is best-effort
            return False

    async def _resume_recover_pending_integrate(self, report: dict[str, Any]) -> None:
        """Recover a crashed integrate_patch window from the sentinel.

        Three-way decision keyed on whether a ``kept`` delegated-result exists
        for the sentinel's task: replay the missing append (crashed after KEEP),
        roll back the half-applied patch (crashed after apply, before KEEP), or
        clear a stale sentinel. A scan that could not read the event log reaches
        none of the three, so it must neither roll back nor clear the sentinel.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        pending = getattr(state, "pending_integrate", {}) or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        kept_res: dict[str, Any] | None = None
        scanned = False
        try:
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                if task_id and str(payload.get("task_id") or "") != task_id:
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
                    kept_res = res
                    break
            scanned = True
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: pending_integrate kept-result scan failed")
        if not scanned:
            report["warnings"].append({"kind": "pending_integrate_scan_failed", "task_id": task_id})
            return
        if kept_res is not None:
            appended = self._replay_keep_from_result("integrate_patch", kept_res)
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
        else:
            summary = self._resume_rollback_pending_integrate(pending)
            if summary.get("reversed"):
                report["fixes"].append({"kind": "rolled_back_pending_integrate", "task_id": task_id, **summary})
            elif summary.get("failed"):
                report["warnings"].append({"kind": "pending_integrate_rollback_failed", "task_id": task_id, **summary})
            else:
                report["fixes"].append({"kind": "cleared_stale_pending_integrate", "task_id": task_id})
        state.pending_integrate = {}

    async def _resume_recover_pending_warm_replay(
        self,
        report: dict[str, Any],
    ) -> None:
        """Rollback a combined PRELUDE set whose verdict was lost to a crash."""
        state = self.shared_state
        pending = getattr(state, "warm_replay_pending", {}) or {}
        if not isinstance(pending, dict) or not pending:
            return
        rollback = self.phase_prelude._rollback_combined_warm({}, None)
        errors = list(rollback.get("errors") or [])
        if errors:
            report["warnings"].append(
                {
                    "kind": "resume_warm_rollback_failed",
                    "task_id": pending.get("task_id"),
                    "errors": errors,
                }
            )
            if hasattr(state, "set_stop_reason"):
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
                if hasattr(state, "set_stop_reason"):
                    state.set_stop_reason("warm_replay_rollback_failed")
                state.save(self.session_dir)
                return
        state.warm_replay_outcome = {
            **dict(getattr(state, "warm_replay_outcome", {}) or {}),
            "status": "failed",
            "reason": "interrupted_combined_validation_rolled_back",
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

        from ..framework.targeted_build import kill_build_pgroup

        state = self.shared_state
        pending = getattr(state, "pending_targeted_build", {}) or {}
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
            except Exception:  # noqa: BLE001 — sweep is best-effort
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
            except Exception:  # noqa: BLE001 — reclaim backstop still applies
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
        chance: the window closes and the round is charged to the stall streak.
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
                self._reopen_revalidation_window()
                report["fixes"].append({"kind": "reopened_revalidation_the_run_cancelled", "task_id": tracked_tid})
                log.info(
                    "resume: revalidation task %s was cancelled by the run; window left open "
                    "at generation %d without charging the stall streak",
                    tracked_tid,
                    int(state.enablement.revalidation_generation or 0),
                )
                return
            state.enablement.validation_pending = False
            state.enablement.revalidation_task_id = ""
            state.enablement.stall_streak = int(state.enablement.stall_streak or 0) + 1
            state.enablement.inflight_task_id = ""
            report["fixes"].append({"kind": "cleared_orphaned_revalidation_pending", "task_id": tracked_tid})
            log.info(
                "resume: cleared stale enablement_validation_pending for terminal revalidation task %s",
                tracked_tid,
            )
        except Exception:  # noqa: BLE001 — best-effort
            log.debug("resume: revalidation pending recovery check failed", exc_info=True)

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
        state = self.shared_state
        try:
            stack_keys = {
                (str(e.get("action") or ""), str(e.get("variant_name") or ""))
                for e in (state.optimization_stack or [])
                if isinstance(e, dict)
            }
            seen: set[tuple[str, str]] = set()
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                kind = str(payload.get("kind") or "")
                res = payload.get("result") or {}
                if not isinstance(res, dict) or str(res.get("status") or "").lower() != "kept":
                    continue
                stack_action = kind
                if kind == "integrate_patch":
                    variant = str(res.get("specialist_task_id") or "")
                elif kind == "framework_agent":
                    # This kind stacks under the framework family label, keyed
                    # by the canonical candidate key, so reconcile on both.
                    stack_action = _FRAMEWORK_STACK_ACTION
                    cand = res.get("candidate")
                    variant = self._framework_candidate_key(cand if isinstance(cand, dict) else None)
                elif kind == "explore":
                    bv = res.get("best_variant") or {}
                    variant = str((bv.get("name") if isinstance(bv, dict) else "") or "")
                else:
                    continue
                key = (stack_action, variant)
                if not variant or key in stack_keys or key in seen:
                    continue
                seen.add(key)
                if kind == "integrate_patch":
                    workspace = str(res.get("workspace") or "").strip()
                    if workspace and not Path(workspace).exists():
                        report["warnings"].append(
                            {
                                "kind": "orphaned_keep_discarded",
                                "orphan_kind": kind,
                                "variant": variant,
                                "task_id": payload.get("task_id"),
                                "reason": "workspace_missing",
                            }
                        )
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "orphaned_keep_discarded",
                                "severity": "medium",
                                "orphan_kind": kind,
                                "variant": variant,
                            },
                        )
                    elif self._replay_keep_from_result(kind, res):
                        stack_keys.add(key)
                        report["fixes"].append(
                            {"kind": "replayed_orphaned_keep", "orphan_kind": kind, "variant": variant}
                        )
                    else:
                        report["warnings"].append(
                            {"kind": "orphaned_keep_replay_noop", "orphan_kind": kind, "variant": variant}
                        )
                else:
                    # explore / framework: ambiguous vs eviction — never
                    # resurrect; surface for the operator.
                    report["warnings"].append(
                        {
                            "kind": "orphaned_keep",
                            "orphan_kind": kind,
                            "variant": variant,
                            "task_id": payload.get("task_id"),
                        }
                    )
                    await self._record_observation(
                        "coordinator",
                        "observation",
                        {
                            "kind": "orphaned_keep",
                            "severity": "medium",
                            "orphan_kind": kind,
                            "variant": variant,
                        },
                    )
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: orphaned KEEP resume recovery failed")

    async def _enqueue_internal_stack_rebench(self, *, reason: str) -> dict[str, Any]:
        """Enqueue one full-stack end-to-end rebench of the cumulative config.

        Builds a single-variant ``explore`` task from ``current_best``'s launch
        args/envs, benched against ``baseline_tput`` so the measured
        delta becomes the validated cumulative gain. Tagged
        ``source=resume_stack_revalidate`` so ``_promote_to_shared_state``
        reconciles ``cumulative_gain_validated_stack_len`` + clears
        ``resume_pending_revalidation`` from the measured throughput. GEAK 2b
        revalidations are idempotent per macro-cycle via
        ``geak_revalidate_idempotency_key``.

        Args:
            reason: Human-readable reason stamped on the task params.

        Returns:
            A summary ``{"task_id", "existing"}`` or ``{"skipped", "reason"}``.
        """
        # fix-point 7 (2b) — when the win is a GEAK e2e result, source the
        # revalidation config from result.json (the SINGLE source of truth), NOT
        # from stack materialization. This guarantees the same-harness rebench
        # launches byte-for-byte the config GEAK optimized (flags + parsed env +
        # authored overlay), independent of whether the optimization is a MoE
        # tuned-config / kernel / flag winner — no case-by-case markers. The
        # consumer (_promote_to_shared_state) asserts config identity + effect
        # before stamping validated, and falls back to 2a (GEAK harness) on miss.
        ps = self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        ps_cfg = ps.get("accepted_config") or {}
        ps_overlay = _normalize_geak_overlay_dir(str(ps.get("final_overlay") or "").strip())
        # ``no_gain`` is a verdict on GEAK's headline basis, not on its kernels;
        # a result carrying an accepted, positive-delta kernel is revalidated
        # too, so the kernel gets an orchestrator-measured number.
        ps_admissible = str(ps.get("status") or "") == "ok" or _geak_has_accepted_kernel(ps)
        if ps_admissible and (ps_cfg.get("flags") or ps_cfg.get("env") or ps_overlay):
            from ..actions.executors._canonical_fingerprint import canonical_fingerprint

            ps_flags, ps_envs = _accepted_config_as_variant(ps_cfg)
            # An overlay that cannot load installs nothing: the server launches
            # as plain baseline and any delta measured against it belongs to the
            # flags alone. Resolve that BEFORE dispatch so the task never carries
            # a dead path, and so the row cannot be read as a kernel win.
            ps_overlay_loadable = _geak_overlay_is_loadable(ps_overlay)
            if ps_overlay and not ps_overlay_loadable:
                log.warning(
                    "geak 2b: overlay %r is not loadable (no sitecustomize.py); "
                    "revalidating the config WITHOUT the authored kernel",
                    ps_overlay,
                )
                ps_overlay = ""
            if not (ps_flags or ps_envs or ps_overlay):
                # The overlay was the only material and it is dead. A rebench
                # here would measure plain baseline and credit GEAK for the
                # noise. Hand it to the GEAK harness (2a), which reproduces the
                # optimized config from result.json and so engages by construction.
                return {
                    "skipped": True,
                    "reason": "geak_overlay_unloadable",
                    "fallback": "geak_harness",
                }
            if ps_flags or ps_envs or ps_overlay:
                # Identity hash uses the SAME (args, envs) contract the grid
                # executor fingerprints with (overlay is NOT part of the hash,
                # matching canonical_fingerprint) so expected == the
                # ran variant's fingerprint by construction, and any executor-side
                # drop/alter of config is caught downstream.
                expected_cfg_hash = canonical_fingerprint(ps_flags, ps_envs)
                # ``expected_cfg_hash`` cannot see the overlay, so carry the
                # overlay's own identity beside it. The consumer re-checks both
                # after the run: a dropped or altered overlay then reads as
                # inconclusive instead of as a validated kernel win.
                expected_overlay_digest = _geak_overlay_digest(ps_overlay)
                # Name what ran. Without this the decision row inherits the flag
                # string as its whole identity and the kernel rides along unnamed.
                ps_kernels = [_geak_spec_name(k) for k in _geak_accepted_kernel_specs(ps)]
                params_ps: dict[str, Any] = {
                    "source": "resume_stack_revalidate",
                    "reason": reason,
                    "geak_fallback": True,
                    "expected_cfg_hash": expected_cfg_hash,
                    "expected_overlay": ps_overlay,
                    "expected_overlay_digest": expected_overlay_digest,
                    "grid": [
                        {
                            "name": "geak_revalidate",
                            "extra_args": ps_flags,
                            "extra_envs": dict(ps_envs),
                            "overlay_pythonpath": ps_overlay,
                            "provenance": "geak_revalidate",
                            # Only claim kernels when an overlay is actually
                            # being loaded; a flags-only rebench carries none.
                            "accepted_kernels": ps_kernels if ps_overlay else [],
                            "note": "same-harness config-identity revalidation of the geak e2e win",
                        }
                    ],
                    # Revalidation reproduces the whole stack, so its gain is
                    # cumulative-vs-baseline, not a delta over current_best.
                    "base_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
                }
                if self.shared_state.baseline_config_path:
                    params_ps["config_path"] = self.shared_state.baseline_config_path
                from ..phases.geak_rebench import resolve_geak_revalidate_idempotency_key

                idempotency_key = await resolve_geak_revalidate_idempotency_key(
                    self.tasks,
                    int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                )
                lanes, ttl = self._registry_lanes_ttl("explore")
                task, existing = await self.tasks.create_or_return_existing(
                    kind="explore",
                    params=params_ps,
                    idempotency_key=idempotency_key,
                    # Both halves of the catalogue contract: without lanes the row
                    # launches a server unserialised; without a TTL it is invisible
                    # to ``reclaim_expired_running``.
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                )
                try:
                    from hyperloom.inference_optimizer.breakdown.recorder import instrument

                    instrument.record_geak_operation(
                        self.session_dir,
                        stage="rebench_started",
                        macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                        result={
                            **ps,
                            "rebench": {
                                "task_id": task.task_id,
                                "existing": bool(existing),
                                "mode": "orchestrator_same_harness",
                                "expected_cfg_hash": expected_cfg_hash,
                            },
                        },
                        status="running",
                    )
                except Exception:  # noqa: BLE001
                    log.debug("geak v4 rebench recording failed", exc_info=True)
                return {
                    "task_id": task.task_id,
                    "task_state": task.state,
                    "existing": bool(existing),
                    "mode": "geak_2b",
                }

        launch = self._current_best_launch_config()
        args = launch["extra_server_args"]
        envs = launch["extra_envs"]
        overlay = launch["final_overlay"]
        cb_now = self.shared_state.current_best if isinstance(self.shared_state.current_best, dict) else {}
        cb_remove = cb_now.get("remove_args")
        cb_unset = cb_now.get("unset_envs")
        cb_replace = str(cb_now.get("args_mode") or "").strip().lower() == "replace"
        if not (args or envs or cb_remove or cb_unset or cb_replace):
            return {"skipped": True, "reason": "empty_config"}
        params: dict[str, Any] = {
            "source": "resume_stack_revalidate",
            "reason": reason,
            "grid": [
                {
                    "name": "resume_stack_revalidate",
                    "extra_args": args,
                    "extra_envs": dict(envs),
                    # Carry the overlay so an authored-kernel native stack rebuild
                    # loads the built kernels (inert when empty).
                    "overlay_pythonpath": overlay,
                    "provenance": "resume_stack_revalidate",
                    "note": "post-resume full-stack end-to-end revalidation",
                }
            ],
            # Cumulative-vs-baseline, same as the geak revalidation above.
            "base_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
        }
        if cb_remove:
            params["base_remove_args"] = [cb_remove] if isinstance(cb_remove, str) else list(cb_remove or [])
        if cb_unset:
            params["base_unset_envs"] = [cb_unset] if isinstance(cb_unset, str) else list(cb_unset or [])
        if cb_replace:
            params["base_args_mode"] = "replace"
        if self.shared_state.baseline_config_path:
            params["config_path"] = self.shared_state.baseline_config_path
        lanes, ttl = self._registry_lanes_ttl("explore")
        task, existing = await self.tasks.create_or_return_existing(
            kind="explore",
            params=params,
            idempotency_key="resume-stack-revalidate",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        return {"task_id": task.task_id, "existing": bool(existing)}

    async def _validate_geak_via_geak_harness(self, *, reason: str) -> dict[str, Any]:
        """2a fallback - validate the geak win by REPLAYING it through
        GEAK's own ``bench_e2e.sh`` (the harness that produced the headline
        result), so the optimized config engages BY CONSTRUCTION regardless of
        winner kind (tuned-config / kernel / overlay / flag). Because the replay
        reproduces the optimized config from ``result.json`` directly, a
        ``succeeded`` status is itself the engagement proof. The validated gain
        is the MEASURED replay throughput over the orchestrator's raw baseline
        (``(measured - baseline) / baseline``); GEAK's own ``throughput_speedup``
        / ``hot_geak_speedup`` serves only as a ``> 1.0`` sanity gate, not as the
        reported number. Recorded under a distinct provenance
        (``geak_same_harness_geak``) because the measurement came from GEAK's
        harness rather than the orchestrator's. Used only when 2b (orchestrator
        harness) is inconclusive.

        Args:
            reason: Human-readable reason stamped in logs/return.

        Returns:
            A summary dict describing whether validation succeeded.
        """
        ps = self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        if str(ps.get("status") or "") != "ok" and not _geak_has_accepted_kernel(ps):
            return {"validated": False, "skipped": True, "reason": "no_geak_result"}
        # Overlay identity, captured BEFORE the replay so it can be compared
        # after. 2a replays GEAK's own launch script, which is why a
        # ``succeeded`` status proves the *config* engaged -- but the overlay is
        # a separate artifact on a path in ``result.json``, and it can be gone
        # or inert by the time the replay runs. Measured over
        # ``/shared_nfs/hyperloom-claw``: of 64 runs declaring a
        # ``final_overlay``, 25 name a directory that does not exist and 30 name
        # one holding no ``sitecustomize.py``. Only 9 can install a kernel. A
        # non-empty string is therefore not evidence the kernel ran, and using
        # it as evidence would stamp 55 flag-only measurements as kernel wins.
        # Same check 2b runs; see ``_geak_overlay_is_loadable``.
        ps_overlay_2a = _normalize_geak_overlay_dir(str(ps.get("final_overlay") or "").strip())
        overlay_digest_before = _geak_overlay_digest(ps_overlay_2a) if ps_overlay_2a else ""
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="geak_harness_fallback",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result={**ps, "fallback_reason": reason},
                status="running",
                validation_source="geak_same_harness_geak",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 fallback recording failed", exc_info=True)
        am = ps.get("alignment_metrics") or {}
        # Read GEAK's OWN within-harness speedup on the SAME basis it promoted
        # (result.throughput_speedup == cold_geak_speedup when final_basis=="cold",
        # else the hot within-GEAK ratio; see run_e2e final-basis selection). It is
        # used solely to sanity-check that GEAK claimed a win on its promoted basis
        # before the replay measurement is promoted as the headline.
        # Falls back to the explicit within-GEAK ratios when throughput_speedup is
        # missing (older result.json), preferring the promoted basis.
        try:
            geak_sp = float(ps.get("throughput_speedup") or 0.0)
        except (TypeError, ValueError):
            geak_sp = 0.0
        if geak_sp <= 0:
            basis = str(am.get("final_basis") or ps.get("final_throughput_basis") or "hot")
            fallback_key = "cold_geak_speedup" if basis == "cold" else "hot_geak_speedup"
            try:
                geak_sp = float(am.get(fallback_key) or am.get("hot_geak_speedup") or 0.0)
            except (TypeError, ValueError):
                geak_sp = 0.0
        regimes = ps.get("validated_regimes") or []
        reg = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
        try:
            conc = int(reg.get("conc") or 64)
            isl = int(reg.get("isl") or 1024)
            osl = int(reg.get("osl") or 1024)
        except (TypeError, ValueError):
            conc, isl, osl = 64, 1024, 1024
        from hyperloom.inference_optimizer.session.session_paths import unique_runs_dir
        from ..actions.executors._geak_sweep import sweep_via_geak

        try:
            timeout = int(os.environ.get("SWEEP_VARIANT_TIMEOUT_SEC", "").strip() or "2400")
        except (TypeError, ValueError):
            timeout = 2400
        res = await sweep_via_geak(
            result=ps,
            conc_values=[conc],
            isl_osl_configs=[f"{isl}:{osl}"],
            output_root=unique_runs_dir(self.session_dir, "sweep", "revalidate_geak"),
            variant_timeout_sec=timeout,
            repeats=3,
            # Single-point validated replay pins the headline protocol (num_prompts
            # etc.) so it is protocol-identical to the reported result.
            pin_num_prompts=True,
        )
        if str(res.get("status") or "") == "succeeded" and geak_sp > 1.0:
            # Rebench-first: write the headline from the GEAK-harness MEASURED
            # throughput (engages by construction via the launch-script replay),
            # keeping the leaderboard number a same-harness total rather than a
            # self-reported speedup.
            measured = _geak_sweep_measured_tput(res)
            if measured is None:
                log.warning("geak 2a: succeeded sweep but no measurable throughput; candidate stays pending")
                try:
                    from hyperloom.inference_optimizer.breakdown.recorder import instrument

                    instrument.record_geak_operation(
                        self.session_dir,
                        stage="final_validation_failed",
                        macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                        result={**ps, "fallback_result": res, "failure_reason": "missing_measured_throughput"},
                        status="failed",
                        validation_source="geak_same_harness_geak",
                    )
                except Exception:  # noqa: BLE001
                    log.debug("geak v4 missing-measurement recording failed", exc_info=True)
                return {"validated": False, "status": res.get("status"), "reason": reason}
            # The replay proves the config engaged. It does not prove the
            # overlay did: the overlay has to still be loadable, and still be
            # the same overlay, at the moment the replay ran.
            overlay_loaded_2a = bool(ps_overlay_2a) and _geak_overlay_is_loadable(ps_overlay_2a)
            if overlay_loaded_2a and overlay_digest_before:
                overlay_loaded_2a = _geak_overlay_digest(ps_overlay_2a) == overlay_digest_before
            if ps_overlay_2a and not overlay_loaded_2a:
                log.warning(
                    "geak 2a: overlay %r is not loadable evidence "
                    "(loadable=%r digest before=%r after=%r) -> gain credited "
                    "to config, not to a kernel",
                    ps_overlay_2a,
                    _geak_overlay_is_loadable(ps_overlay_2a),
                    overlay_digest_before,
                    _geak_overlay_digest(ps_overlay_2a),
                )
            self._promote_geak_from_candidate(
                ps,
                measured_tput=measured,
                provenance="geak_same_harness_geak",
                overlay_loaded=overlay_loaded_2a,
                measurement_provenance=(
                    res.get("promotion_measurement") if isinstance(res.get("promotion_measurement"), Mapping) else res
                ),
            )
            base = float(self.shared_state.baseline_tput or 0.0)
            gain_out = ((measured - base) / base * 100.0) if base > 0 else 0.0
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 - defensive
                log.exception("geak 2a: SharedState.save failed")
            return {"validated": True, "gain": gain_out, "reason": reason}
        log.warning(
            "geak 2a fallback did not validate (status=%r geak_speedup=%r reason=%s)",
            res.get("status"),
            geak_sp,
            reason,
        )
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_geak_operation(
                self.session_dir,
                stage="final_validation_failed",
                macro_cycle=int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                result={
                    **ps,
                    "fallback_result": res,
                    "failure_reason": reason,
                    "geak_speedup": geak_sp,
                },
                status="failed",
                validation_source="geak_same_harness_geak",
            )
        except Exception:  # noqa: BLE001
            log.debug("geak v4 failed-validation recording failed", exc_info=True)
        return {"validated": False, "status": res.get("status"), "reason": reason}

    async def _resume_reenter_kernel_if_needed(self) -> None:
        """Idempotently re-fire the KERNEL_AGENT entry hook on resume.

        Phase-entry side effects (the GEAK delegation + its ``result.json``
        crash-recovery, and the collective lane's pending integration) are bound
        to a phase *transition* via ``_on_phase_entered``; a resume only restores
        ``phase`` from state.json and never re-enters the current phase. Without
        this, a session that crashed mid ``KERNEL_AGENT`` sits idle until the
        phase budget cap fires, then hands SWEEP an empty result — the whole
        delegation is silently lost.

        General across every crash timing (not case-by-case): the decision is
        driven purely by whether THIS KERNEL phase's history row already carries
        a ``geak`` completion record, so it self-classifies:

          * completed-this-phase -> only re-arm (+persist) the ``skip_to_sweep``
            hint the delegation sets, so the phase machine winds down to SWEEP
            with no e2e re-run;
          * not-completed -> re-enter ``_on_enter_kernel``; its own entry guard
            promotes an existing OK ``result.json`` (crash-before-handback) and
            re-runs the e2e only when there is genuinely nothing to recover
            (run_e2e itself then continues from the pinned eval_dir on disk).

        No-op unless resumed while parked in ``KERNEL_AGENT`` with a pending
        collective integration or the GEAK backend selected.
        """
        from ..phases.machine_state import (
            ESCALATE_HINT_SKIP_TO_SWEEP,
            PHASE_KERNEL_AGENT,
            collective_integration_pending,
        )

        if not self._resumed_from.get("is_resume"):
            return
        state = self.shared_state
        if (state.phase or "").strip().upper() != PHASE_KERNEL_AGENT:
            return
        kernel_enabled = self._kernel_enabled()
        collective_only = bool(getattr(state, "collective_only_mode", False))
        # Mirror _on_enter_kernel's precedence: GEAK owns the phase unless
        # collective-only mode turned it off, and the collective lane is only
        # reachable when GEAK does not own it. Checking collective state ahead
        # of an owning GEAK would re-run its whole e2e instead of re-arming the
        # wind-down hint.
        geak_enabled = kernel_enabled and not collective_only and self._geak_enabled()
        if kernel_enabled and not geak_enabled:
            try:
                collective_required = bool(
                    collective_integration_pending(state) or self._collective_required_before_kernel_opt()
                )
            except Exception:  # noqa: BLE001
                # A malformed collective record must not strand the GEAK
                # crash-recovery below: without it the session idles to its
                # phase budget and hands SWEEP an empty result.
                log.exception(
                    "resume: Collective state check failed; continuing without it",
                )
                collective_required = False
            if collective_required:
                log.info("resume: re-entering unfinished Collective work")
                try:
                    await self._on_enter_kernel(from_phase="resume")
                except Exception:  # noqa: BLE001
                    log.exception("resume: Collective re-entry failed")
                return
            if collective_only:
                state.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
                state.save(self.session_dir)
                return
        if not geak_enabled:
            return
        history = state.phase_history or []
        row = history[-1] if history else {}
        evidence = row.get("evidence") if isinstance(row, dict) else {}
        completed_this_phase = isinstance(evidence, dict) and isinstance(evidence.get("geak"), dict)
        if completed_this_phase:
            # The delegation landed during this phase but the SWEEP transition
            # never persisted (crash between the hook and the next tick). Re-arm
            # the wind-down hint + persist so the phase machine advances.
            cur = str(getattr(state, "pending_escalate_hint", "") or "").strip()
            if cur != ESCALATE_HINT_SKIP_TO_SWEEP:
                state.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
                try:
                    state.save(self.session_dir)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception("resume: save after re-arming skip_to_sweep failed")
                log.info(
                    "resume: KERNEL GEAK already completed this phase; "
                    "re-armed skip_to_sweep hint (lost before SWEEP transition)."
                )
            return
        log.info(
            "resume: re-entering KERNEL GEAK delegation (no completion "
            "evidence on the current phase row); recover-from-disk or re-run."
        )
        try:
            await self._on_enter_kernel(from_phase="resume")
        except Exception:  # noqa: BLE001 — resume re-entry must never kill the session
            log.exception("resume: KERNEL re-entry hook failed")

    @property
    def resumed_from(self) -> dict[str, Any]:
        """Read-only snapshot of resume detection (set by ``__init__``).

        Returns:
            A copy of the resume-detection dict so callers cannot mutate
            internal state.
        """
        return dict(self._resumed_from)

    # Bounded test interface
    async def _replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session (replay log + abandon orphan dispatches)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()
        await self._resume_consistency_pass()
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT
        # (phase-entry side effects are bound to transitions, not resume). Runs
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
