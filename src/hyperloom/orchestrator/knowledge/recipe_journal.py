# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization journal, Recipe KB facts and the session's final Recipe, written from settled results."""

from __future__ import annotations

import logging as _logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Collection, Mapping

from hyperloom.common.coerce import to_float, to_int
from hyperloom.common.io import append_jsonl
from hyperloom.inference_optimizer.breakdown.recorder import close_out as _close_out
from hyperloom.inference_optimizer.recipe_snapshot_constants import detect_framework_version
from hyperloom.inference_optimizer.session.optimization_journal import (
    OUTCOME_KEEP,
    OUTCOME_REVERT,
    Journal,
    JournalEntry,
    Verdict,
    classify_change_kind,
    derive_journal_outcome,
    summarize_change,
)
from hyperloom.orchestrator.knowledge.recipe_kb import recipe_canonical_id
from hyperloom.orchestrator.lever import lever_kind_for_task

from ..collaborator import CoordinatorCollaborator
from ..state.attempt_ledger import record_config_attempt
from ..state.failure_evidence import UNMEASURED_OUTCOMES, classify_failure_attribution
from ..state.task_registry import Task

if TYPE_CHECKING:
    from ..loop.coordinator import Coordinator

log = _logging.getLogger(__name__)


# Recipe snapshot severity tags (schema has no fixed enum).
_SEVERITY_CRASH: str = "crash"
_SEVERITY_REGRESS: str = "regress"


# Stable ``result_type`` codes for the reasons the remote KB Store returns.
# Use an exact lookup so a reason token cannot collide with the same text inside
# an exception name or explanatory message.
_REMOTE_RESULT_TYPES: dict[str, str] = {
    "KB_STORE_URL/TOKEN not configured": _close_out.RESULT_KB_DISABLED,
    "no_new_keep_or_pure_warm_replay": _close_out.RESULT_NO_NEW_KEEP,
    "nonfinite_optimized_throughput": _close_out.RESULT_INVALID_THROUGHPUT,
    "missing_optimized_throughput": _close_out.RESULT_MISSING_THROUGHPUT,
    "invalid_recipe_scope": _close_out.RESULT_INVALID_SCOPE,
    "invalid_recipe_selection_profile": _close_out.RESULT_INVALID_SELECTION_PROFILE,
    "empty_replay_material": _close_out.RESULT_EMPTY_REPLAY_MATERIAL,
    "not_better_than_champion": _close_out.RESULT_NOT_BETTER,
    "champion_not_promoted": _close_out.RESULT_CHAMPION_NOT_PROMOTED,
}

# The one exception class meaning the store rejected the bundle, rather than
# that we never reached the store at all.
_REMOTE_VALIDATION_ERROR = "RemoteRecipeValidationError"


def _remote_result_type(status: str, reason: str) -> str:
    """The ``close_out.RESULT_*`` code for a remote KB Store write result.

    ``status`` is consulted only when the store's own reason token is
    unrecognized.
    """
    known = _REMOTE_RESULT_TYPES.get(str(reason or "").strip())
    if known:
        return known
    verdict = str(status or "").strip().lower()
    if verdict == "written":
        return _close_out.RESULT_WRITTEN
    if verdict in {"skipped", "disabled"}:
        return _close_out.RESULT_SKIPPED_OTHER
    return _close_out.RESULT_TRANSPORT_FAILED


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


#: Explore outcomes that are not an attempt at anything: a variant skipped as a
#: duplicate was never measured, so no measurement would back its funnel row.
_NON_ATTEMPT_OUTCOMES = frozenset({"SKIPPED_DEDUP"})


def _record_config_run(coord: Any, *, task: Any, result_dict: Mapping[str, Any]) -> None:
    """Record the configuration arm's grid dispatch on the framework event.

    The arm's dispatch is one ``explore`` task, and it earns a run row for the
    same reason a specialist dispatch does: the event reduces its status over
    what it dispatched, so an arm whose grid came back with nothing must not
    read as an arm that was never tried. Recorded at completion, where the
    outcome is known, with the dispatch time taken off the task row; a grid
    that measured nothing still lands, which is the case
    :func:`_record_config_attempts` never sees.
    """
    recorder = coord._coord.phase_framework.timeline()
    if recorder is None:
        return
    from hyperloom.common.timeutil import now_iso
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ARM_CONFIG, ROLE_CONFIG

    task_id = str(getattr(task, "task_id", "") or "")
    if not task_id:
        return
    variants = result_dict.get("per_variant_outcomes")
    measured = [row for row in variants if isinstance(row, dict)] if isinstance(variants, list) else []
    status = str(result_dict.get("status") or "") or "succeeded"
    recorder.record_run(
        task_id,
        role=ROLE_CONFIG,
        arm=ARM_CONFIG,
        status=status,
        dispatched_at=str(getattr(task, "created_at", "") or ""),
        completed_at=now_iso("seconds"),
        reason=str(result_dict.get("error") or "")[:200],
        # A grid the run stopped before it measured anything is not a grid
        # whose variants were measured and lost.
        empty=not measured,
        workspace=str(result_dict.get("workspace") or ""),
    )


def _record_config_attempts(
    coord: Any,
    *,
    task: Any,
    per_variant: list[dict[str, Any]],
    result_dict: Mapping[str, Any],
    adopted_variants: Collection[str],
) -> None:
    """Record the configuration arm's measured attempts on the framework event.

    Timeline only: the ledger row is written by ``fact_write_hook``, which does
    not depend on a recorder being open. The outcome is recorded verbatim here,
    unlike the journal beside it, which collapses ``KEEP_UNSTABLE`` and
    ``KILLED_OVERTIME`` into a plain revert.
    """
    recorder = coord._coord.phase_framework.timeline()
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ARM_CONFIG

    task_id = str(getattr(task, "task_id", "") or "")
    params = getattr(task, "params", None) or {}
    round_id = str(result_dict.get("round_id") or "")
    recorded = 0
    for row in per_variant:
        if not isinstance(row, dict):
            continue
        outcome = str(row.get("outcome") or "")
        if outcome in _NON_ATTEMPT_OUTCOMES:
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
        fingerprint = str(row.get("fingerprint") or "")
        adopted = fingerprint in adopted_variants
        # The fingerprint identifies the variant within the round and the round
        # within the task, so the three together identify the attempt.
        attempt_id = ":".join(part for part in (task_id, round_id, fingerprint) if part) or task_id
        if not attempt_id:
            continue
        gates = [gate for gate in (row.get("gates") or []) if isinstance(gate, dict)]
        # This arm gates accuracy rather than reporting it, so the block both
        # arms carry is projected from the gate that ruled. No gate row means
        # nothing gated the variant, which is not a gate that refused it.
        accuracy_gate = next((gate for gate in gates if str(gate.get("gate") or "") == "accuracy"), {})
        error_class = str(row.get("error_class") or "")
        error_excerpt = str(row.get("error_excerpt") or "")
        failure_attribution = ""
        if outcome in UNMEASURED_OUTCOMES:
            failure_attribution = classify_failure_attribution(
                error_class=error_class,
                error_excerpt=error_excerpt,
                reason=row.get("reason"),
                explicit=row.get("failure_attribution"),
            )
        recorder.record_attempt(
            attempt_id,
            arm=ARM_CONFIG,
            round_id=round_id,
            task_id=task_id,
            proposal_ref=str(params.get("proposal_msg_id") or ""),
            provenance=str(row.get("provenance") or ""),
            outcome=outcome,
            reason=str(row.get("reason") or ""),
            reasoning=str(variant.get("note") or ""),
            reasoning_origin=str(variant.get("reasoning_origin") or ""),
            experience_citations=variant.get("experience_citations") or [],
            stage=str(row.get("stage") or ""),
            fingerprint=fingerprint,
            # The fingerprint is the join key; the name is what a reader
            # recognises the variant by.
            variant_name=str(row.get("variant_name") or ""),
            measurement={
                # Both ends of the pair: the anchor advances on every KEEP,
                # so a percentage without its denominator adds to nothing.
                "before_tput": metrics.get("base_tput"),
                "after_tput": metrics.get("tput"),
                "gain_pct": metrics.get("gain_pct"),
                "runtime_sec": metrics.get("runtime_sec"),
                "estimated_output_throughput": metrics.get("estimated_output_throughput"),
            },
            config_delta={
                "extra_server_args": variant.get("extra_server_args"),
                "extra_envs": variant.get("extra_envs"),
                "remove_args": variant.get("remove_args"),
                "unset_envs": variant.get("unset_envs"),
                "args_mode": variant.get("args_mode"),
            },
            accuracy={
                "required": True if accuracy_gate else None,
                "value": accuracy_gate.get("observed"),
                "reference": accuracy_gate.get("threshold"),
                "passed": accuracy_gate.get("passed"),
            },
            failure={
                "error_class": error_class,
                "error_excerpt": error_excerpt,
                "attribution": failure_attribution,
            },
            artifacts={
                "workspace": str(row.get("workspace") or ""),
                "server_log_path": str(row.get("server_log_path") or ""),
                "raw_result_path": str(row.get("raw_result_path") or ""),
            },
            decision=outcome,
            adopted=adopted,
            # Recorded rather than referenced: every KEEP advances the
            # stack, so the session's current config is not what this
            # variant was measured on top of.
            measured_against=row.get("measured_against") or {},
            # What stood behind the verdict. Absent when nothing ruled.
            validation_basis=str(row.get("validation_basis") or ""),
            # A pair is what makes a gain addable, so eligibility follows
            # the pair being present rather than the outcome being a KEEP.
            attribution_eligible=(adopted and metrics.get("base_tput") is not None and metrics.get("tput") is not None),
        )
        for gate in gates:
            if not str(gate.get("gate") or ""):
                continue
            recorder.record_attempt_gate(
                attempt_id,
                str(gate.get("gate")),
                passed=gate.get("passed"),
                reason=str(gate.get("reason") or ""),
                observed=gate.get("observed"),
                threshold=gate.get("threshold"),
            )
        recorded += 1
    proposal_ref = str(params.get("proposal_msg_id") or "")
    if not (recorded and proposal_ref):
        return
    # Settled here rather than at materialization: a grid whose task dies before
    # any variant runs has no attempt to stand behind an ``attempted`` reading,
    # and is more honestly left unsettled.
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        DISPOSITION_ATTEMPTED,
        STEP_ATTEMPTED,
    )

    recorder.record_proposal_step(proposal_ref, step=STEP_ATTEMPTED, outcome=str(recorded))
    recorder.settle_proposal(proposal_ref, disposition=DISPOSITION_ATTEMPTED)


def _extra_server_args(payload: Mapping[str, Any]) -> str:
    """Read canonical ``extra_server_args`` from a payload."""
    value = payload.get("extra_server_args")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


class RecipeJournalCollaborator(CoordinatorCollaborator):
    """Records settled results into the optimization journal, the Recipe KB facts and the final Recipe."""

    PITFALL_REGRESS_THRESHOLD_PCT: float = -5.0

    def __init__(self, coordinator: "Coordinator") -> None:
        super().__init__(coordinator)
        self._journal: Journal | None = None
        self._local_recipe_cache: tuple[int, dict[str, Any]] | None = None

    def journal_integrate_keep(self, result: dict[str, Any], *, lift_kind: str, new_tput: float) -> None:
        """Mirror an adopted kernel-recipe-lane (forge-loop/fusion) KEEP as an ``optimization_journal`` row.

        ``lift_to_current_best`` promotes this KEEP into ``optimization_stack`` directly; it never
        goes through the generic ``fact_write_hook`` -> ``_record_fact_per_task`` path every
        dispatched ``Task`` uses to append its own journal row. Without this, ``final_throughput`` /
        ``total_gain_pct`` in the journal's header name a KEEP the journal's own ``entries`` list
        never records (see ``_journal_gemm_tuning_keep`` for the sibling gap on the gemm_tuning lane).
        """
        try:
            journal = self.ensure_journal()
            # optimization_journal.py has no dedicated fusion bucket; a fusion KEEP is a kernel
            # integration by the same lever (see lever_kind_for_task), so it uses the same kind. The
            # raw lift_kind still reaches the row through provenance below.
            kind = classify_change_kind("integrate")
            change = str(result.get("kernel_id") or lift_kind)
            backend_or_engine = str(result.get("backend") or result.get("engine") or "")
            journal.append_entry(
                JournalEntry(
                    phase=self.journal_entry_phase(),
                    lever_kind=lever_kind_for_task(lift_kind, result if isinstance(result, dict) else None),
                    iter=int(self.shared_state.tick or 0),
                    kind=kind,
                    change=change,
                    outcome=OUTCOME_KEEP,
                    gain_pct=to_float(result.get("gain_pct")),
                    throughput_after=new_tput,
                    task_id=str(result.get("integration_id") or ""),
                    variant_name=str(result.get("kernel_id") or lift_kind),
                    provenance=f"{lift_kind}:{backend_or_engine}" if backend_or_engine else lift_kind,
                    tick=int(self.shared_state.tick or 0),
                )
            )
        except Exception:
            log.exception("integrate journal append failed")

    def _source_session_id(self) -> str:
        """Return the hyperloom-local session id used as source_session_id on KB fact writes.

        NOT a KB-side session id; prefers recipe_kb_session_id, falls back to session_dir.name.

        Returns:
            The hyperloom-local session id (recipe_kb_session_id when set, else
            ``session_dir.name``).
        """
        return str(self.shared_state.recipe_kb_session_id or "") or self.session_dir.name

    async def fact_write_hook(
        self,
        *,
        task: "Task",
        result: Any,
        verdict: Verdict,
        adopted_variants: Collection[str] = (),
    ) -> None:
        """Per-task fact-write entry point (per_variant for explore grids, else per-task).

        Args:
            task: The completed task being recorded.
            result: The task's :class:`SubAgentResult` (or result dict).
            verdict: The settlement verdict.
            adopted_variants: Fingerprints of the explore winners whose lift
                landed; a per-variant KEEP outside it did not adopt anything.
        """
        result_dict = result.result if hasattr(result, "result") else (result or {})
        if not isinstance(result_dict, dict):
            result_dict = {}
        source_session_id = self._source_session_id()
        per_variant = result_dict.get("per_variant_outcomes")
        if task.kind == "explore":
            _record_config_run(self, task=task, result_dict=result_dict)
        if task.kind == "explore" and isinstance(per_variant, list) and per_variant:
            _record_config_attempts(
                self,
                task=task,
                per_variant=per_variant,
                result_dict=result_dict,
                adopted_variants=adopted_variants,
            )
            round_id = str(result_dict.get("round_id") or "")
            for vo in per_variant:
                adopted = str(vo.get("fingerprint") or "") in adopted_variants
                outcome = str(vo.get("outcome") or "") if isinstance(vo, dict) else ""
                if outcome and outcome not in _NON_ATTEMPT_OUTCOMES:
                    metrics = vo.get("metrics") if isinstance(vo.get("metrics"), dict) else {}
                    record_config_attempt(
                        self.shared_state,
                        task_id=str(task.task_id or ""),
                        round_id=round_id,
                        fingerprint=str(vo.get("fingerprint") or ""),
                        variant_name=str(vo.get("variant_name") or ""),
                        outcome=outcome,
                        adopted=adopted,
                        gain_pct=metrics.get("gain_pct"),
                        before_tput=metrics.get("base_tput"),
                        after_tput=metrics.get("tput"),
                        error_class=str(vo.get("error_class") or ""),
                        provenance=str(vo.get("provenance") or ""),
                    )
                self._record_fact_per_variant(
                    task=task,
                    source_session_id=source_session_id,
                    variant_outcome=vo,
                    adopted=adopted,
                )
        else:
            self._record_fact_per_task(
                task=task,
                source_session_id=source_session_id,
                result_dict=result_dict,
                verdict=verdict,
            )
        self.shared_state.save(self.session_dir)

    def ensure_journal(self) -> Journal:
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume).

        Returns:
            The per-session :class:`Journal` instance (created on first call,
            with the baseline backfilled on subsequent calls).
        """
        existing = self._journal
        if existing is None:
            ss = self.shared_state
            self._journal = Journal.load_or_create(
                self.session_dir,
                session_id=str(ss.recipe_kb_session_id or "") or str(ss.session_id or "") or self.session_dir.name,
                model=str(ss.model_name or ""),
                hardware=str(ss.gpu_type or ""),
                framework=str(ss.framework or ""),
                baseline_throughput=float(ss.baseline_tput or 0.0),
            )
        else:
            # Backfill baseline once the baseline executor finishes.
            existing.update_baseline(float(self.shared_state.baseline_tput or 0.0))
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

    def journal_entry_phase(self) -> str:
        """Return the current phase label for journal entries.

        Returns:
            str: The uppercased phase name, or ``"UNKNOWN"`` when unset.
        """
        return str(self.shared_state.phase or "").strip().upper() or "UNKNOWN"

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
        workload_tags = self._collect_workload_tags()
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
            statement = self._build_statement(
                change=change,
                kind="lesson",
            )
            impact = self._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(self.shared_state.optimization_stack or []),
                measured_at=now_iso,
            )
            live = self.read_local_recipe_row()
            recipe_overrides = self.kb_best_config_overrides_for_keep(
                live=live,
                best_config_candidate=best_config_candidate,
                throughput_after=throughput_after,
            )
            self.kb_amend_recipe(
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
            description = self._build_statement(
                change=change,
                severity=severity,
                kind="pitfall",
            )
            self.kb_amend_recipe(
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
        verdict: Verdict,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact (source_session_id is hyperloom-local).

        Args:
            task: The completed task being recorded.
            source_session_id: The hyperloom-local session id stamped on the
                fact provenance.
            result_dict: The task result dict.
            verdict: The settlement verdict (ADOPTED → KEEP and a lesson, else
                a non-KEEP row and maybe a pitfall).
        """
        journal = self.ensure_journal()
        # integrate_patch reports its delta under ``delta_pct``;
        # fall back to it so a reverted/kept patch shows its REAL measured delta
        # in the journal instead of a null gain.
        gain_pct = to_float(result_dict.get("gain_pct"))
        if gain_pct is None:
            gain_pct = to_float(result_dict.get("delta_pct"))
        throughput_after = to_float(result_dict.get("output_throughput"))
        kind = classify_change_kind(task.kind, None)
        change = summarize_change(task.kind, None, result_dict)
        outcome = derive_journal_outcome(verdict, result_dict)
        is_keep = verdict is Verdict.ADOPTED
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
                phase=self.journal_entry_phase(),
                lever_kind=lever_kind_for_task(kind, result_dict if isinstance(result_dict, dict) else None),
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
            best_config_candidate=self.extract_kept_best_config(
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
        framework = str(self.shared_state.framework or "").strip()
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
        adopted: bool,
    ) -> None:
        """Per-variant fact write — mirror of _record_fact_per_task for explore per-variant decisions.

        Args:
            task: The completed explore task.
            source_session_id: The hyperloom-local session id stamped on the
                fact provenance.
            variant_outcome: One per-variant outcome row (name, outcome,
                metrics).
            adopted: Whether this variant's lift landed; an executor KEEP that
                was not adopted journals as ``no_promote``.
        """
        journal = self.ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw == "SKIPPED_DEDUP":
            return
        if outcome_raw == "KEEP":
            verdict = Verdict.ADOPTED if adopted else Verdict.REFUSED
        elif outcome_raw in ("REVERT", "FAILED"):
            verdict = Verdict.REVERTED
        else:
            verdict = Verdict.REFUSED
        outcome = derive_journal_outcome(verdict, variant_outcome)
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
                phase=self.journal_entry_phase(),
                lever_kind=lever_kind_for_task(kind, variant_outcome if isinstance(variant_outcome, dict) else None),
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
            best_config_candidate=self.extract_kept_best_config(
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
        framework = str(ss.framework or "").strip()
        if framework:
            out["framework"] = framework
        model_class = str(ss.model_class or "").strip()
        if model_class:
            out["model_class"] = model_class
        # model_family is not part of the seven-dimension Recipe identity.
        model_name = str(ss.model_name or "").strip()
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
        fp_meta = ss.stack_fingerprint_meta or {}
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
        wl_extra = ss.baseline_workload_extra or {}
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
        opt_attempts = ss.kernel_opt_task_attempts or {}
        integ_attempts = ss.kernel_integrate_attempts or {}
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
        gaps = self.shared_state.gaps or []
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
        current_best = ss.current_best or {}
        opt_stack = ss.optimization_stack or []
        gain_per_stack = ss.gain_per_stack_entry or []
        last_failures = ss.last_action_failures or []
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
        sediment_on = bool(ss.recipe_sediment_enabled)
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
        kernel_optimizations = self._build_kernel_optimizations_from_state()
        cumulative_validated = float(ss.cumulative_gain_validated or 0.0)
        validated_stack_len = int(ss.cumulative_gain_validated_stack_len or 0)
        # Workload-shape tags for shape-filtered warm-start queries (shared via _collect_workload_tags).
        workload_tags = self._collect_workload_tags()
        # framework_version left unset here (manifest-derived); the T0 backfill writes it.
        return {
            "best_config": best_config,
            "best_throughput": float(current_best.get("tput", 0.0)) if isinstance(current_best, dict) else 0.0,
            "what_worked": what_worked,
            "what_failed": what_failed,
            "kernel_optimizations": kernel_optimizations,
            "last_profiled": str(ss.cumulative_gain_validated_ts or ""),
            "workload": workload_tags,
            "sessions": [
                {
                    "session_id": str(ss.recipe_kb_session_id or self.session_dir.name),
                    "gain_pct": cumulative_validated,
                    "stack_len": validated_stack_len or len(opt_stack),
                    # arbor-shape provenance so the session row is self-describing (before/after tput + knobs).
                    "throughput_before": float(ss.baseline_tput or 0.0),
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
        primary_metric: str = "",
        primary_value: float = 0.0,
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
                },
                "provenance": {
                    "component": "remote_recipe",
                    "source": source,
                },
            }
            if primary_metric:
                row["result"].update(
                    {
                        "primary_metric": primary_metric,
                        "primary_value": primary_value,
                    }
                )
            if primary_metric == "optimized_throughput":
                row["result"]["best_throughput"] = primary_value
            if error_type:
                row["error"] = {"type": error_type}
            append_jsonl(
                recipe_snapshot_audit_jsonl(self.session_dir),
                row,
                make_parents=True,
                sort_keys=True,
            )
        except Exception:
            log.debug("Remote Recipe KB audit append failed", exc_info=True)

    def ensure_recipe_finalized(
        self,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Idempotently publish terminal Recipe state and persist its outcome."""
        state = self.shared_state
        prior = dict(state.recipe_finalize_outcome or {})
        prior_status = str(state.recipe_finalize_status or prior.get("status") or "")
        if prior_status in {"written", "skipped", "disabled"}:
            return prior

        attempts = int(state.recipe_finalize_attempts or 0) + 1
        state.recipe_finalize_attempts = attempts
        state.recipe_finalize_status = "pending"
        # Opened before the write is tried, so an attempt that never settles
        # reads as unsettled rather than as a refusal the store never issued.
        _close_out.record_write_back_opened(self.session_dir, attempt=attempts, source=source)
        try:
            state.save(self.session_dir)
        except Exception:
            log.exception("Recipe finalize pending-state save failed")

        try:
            raw = self.finalize_recipe_and_journal(source=source) or {}
            outcome = {
                **dict(raw),
                "source": source,
                "attempt": attempts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            log.exception("Recipe finalize raised")
            outcome = {
                "status": "error",
                "reason": type(exc).__name__,
                "source": source,
                "attempt": attempts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result_type": _close_out.RESULT_TRANSPORT_FAILED,
                "error_class": type(exc).__name__,
            }

        raw_status = str(outcome.get("status") or "error")
        state.recipe_finalize_status = "failed" if raw_status == "error" else raw_status
        state.recipe_finalize_outcome = outcome
        self._record_write_back_settled(outcome, attempt=attempts, source=source)
        try:
            state.save(self.session_dir)
        except Exception:
            log.exception("Recipe finalize outcome save failed")
        return outcome

    def _record_write_back_settled(
        self,
        outcome: dict[str, Any],
        *,
        attempt: int,
        source: str,
    ) -> None:
        """Record a settled publication attempt into the breakdown's close-out.

        The publisher already stamped the outcome with a stable ``result_type``
        at whichever exit it took, so this only adds the workload facts the
        recipe was scoped to. ``source`` is ``close`` or ``t4_fallback``.
        """
        state = self.shared_state
        current_best = state.current_best or {}
        tput = current_best.get("tput") if isinstance(current_best, dict) else None
        validated_gain = state.cumulative_gain_validated
        result_type = str(outcome.get("result_type") or "")
        if result_type == _close_out.RESULT_UNVALIDATED_RECIPE:
            # The throughput and the gain belong to different Recipes here.
            tput = validated_gain = None
        _close_out.record_write_back_settled(
            self.session_dir,
            attempt=attempt,
            source=source,
            status=str(outcome.get("status") or ""),
            result_type=result_type,
            raw_reason=str(outcome.get("reason") or ""),
            error_class=str(outcome.get("error_class") or ""),
            backend=str(outcome.get("backend") or ""),
            canonical_id=str(outcome.get("canonical_id") or ""),
            session_id=str(outcome.get("session_id") or ""),
            scope=self._write_back_scope(),
            optimized_throughput=to_float(tput, None),
            validated_gain_pct=to_float(validated_gain, None),
            ts=str(outcome.get("updated_at") or ""),
        )

    def _write_back_scope(self) -> dict[str, Any]:
        """The workload dimensions the published recipe is partitioned by.

        Matches the ``RecipeScope`` shape the section already published.
        """
        state = self.shared_state
        return {
            "kernel_optimizer": str(state.kernel_optimizer or ""),
            "tp": to_int(state.tp, None),
            "conc": to_int(state.conc, None),
            "isl": to_int(state.isl, None),
            "osl": to_int(state.osl, None),
        }

    def finalize_recipe_and_journal(
        self,
        *,
        source: str = "close",
    ) -> dict[str, Any]:
        """Finalize Recipe state and return a secret-free observable outcome."""
        # ``current_best`` and ``cumulative_gain_validated`` are only one Recipe's
        # figures when nothing was lifted since the last validation. Otherwise
        # every sink below -- journal, local KB, remote KB -- would pair the
        # newer config with the older gain, so none of them runs.
        if self.shared_state.optimization_stack_has_unvalidated_keeps():
            ss = self.shared_state
            log.warning(
                "Recipe finalize skipped: working recipe is not validated (stack=%s/%s generation=%s/%s)",
                len(ss.optimization_stack or []),
                ss.cumulative_gain_validated_stack_len,
                ss.working_recipe_generation,
                ss.validated_recipe_generation,
            )
            return {
                "status": "skipped",
                "reason": "unvalidated_recipe_stack",
                "backend": "none",
                "result_type": _close_out.RESULT_UNVALIDATED_RECIPE,
            }
        journal = self.ensure_journal()
        ss = self.shared_state
        cb = ss.current_best or {}
        final_tput = float(cb.get("tput", 0.0)) if isinstance(cb, dict) else 0.0
        total_gain = float(ss.cumulative_gain_validated or 0.0)
        journal.finalize(
            final_throughput=final_tput if final_tput > 0 else None,
            total_gain_pct=total_gain,
        )

        kp = self.knowledge_plane
        if bool(getattr(kp, "kb_disabled", False)):
            log.info("Recipe KB finalize skipped (--degraded-kb)")
            return {
                "status": "skipped",
                "reason": "degraded_kb",
                "backend": "disabled",
                "result_type": _close_out.RESULT_KB_DISABLED,
            }

        from ..knowledge.config import KnowledgeConfig, KnowledgeStoreMode

        try:
            config = getattr(kp, "config", None) or KnowledgeConfig.from_env()
        except Exception as exc:
            log.exception("Recipe KB finalize configuration failed (non-fatal)")
            return {
                "status": "error",
                "reason": f"configuration:{type(exc).__name__}",
                "backend": "unknown",
                "result_type": _close_out.RESULT_CONFIGURATION_FAILED,
                "error_class": type(exc).__name__,
            }
        from hyperloom.common.perf_metric import agentx_active

        if (
            agentx_active(benchmark_mode=self.shared_state.benchmark_mode)
            and config.mode is not KnowledgeStoreMode.REMOTE
        ):
            return {
                "status": "skipped",
                "reason": "agentx_local_store_unsupported",
                "backend": str(getattr(config, "mode", "") or "unknown"),
                "result_type": _close_out.RESULT_AGENTX_BLOCKED,
            }
        if config.mode is KnowledgeStoreMode.REMOTE:
            # Remote mode has one Recipe sink: the KB Store final session
            # writer. T0 and runtime amendment are intentionally absent.
            remote_cid = ""
            remote_sid = str(
                self.shared_state.recipe_kb_session_id or self.shared_state.session_id or self.session_dir.name
            )
            try:
                from ..knowledge.remote_recipe import HyperloomRemoteKB

                remote_cid = self.workload_canonical_id()
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
                    primary_metric=remote_result.primary_metric,
                    primary_value=remote_result.primary_value,
                    reason=remote_result.reason,
                )
                return {
                    "status": remote_result.status,
                    "reason": remote_result.reason,
                    "backend": "kb-store",
                    "canonical_id": str(getattr(remote_result, "canonical_id", "") or remote_cid),
                    "session_id": str(getattr(remote_result, "session_id", "") or remote_sid),
                    "result_type": _remote_result_type(remote_result.status, remote_result.reason),
                }
            except Exception as exc:
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
                    # A validation error means the store rejected the bundle we
                    # built; anything else failed on the way there.
                    "result_type": (
                        _close_out.RESULT_BUNDLE_BUILD_FAILED
                        if type(exc).__name__ == _REMOTE_VALIDATION_ERROR
                        else _close_out.RESULT_TRANSPORT_FAILED
                    ),
                    "error_class": type(exc).__name__,
                }

        # Local mode never consults ambient KB_STORE_* credentials.
        if self.recipe_kb is None:
            return {
                "status": "skipped",
                "reason": "no_recipe_backend",
                "backend": "local",
                "result_type": _close_out.RESULT_KB_DISABLED,
            }
        ss = self.shared_state
        model_name = ss.model_name or ""
        gpu_type = ss.gpu_type or ""
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
                "result_type": _close_out.RESULT_INVALID_SCOPE,
            }
        try:
            attrs = self._build_recipe_attrs_from_state()
            # Hoist workload tags flat into top-level recipe attrs (shallow-merged) for warm-start filters.
            workload_tags = attrs.get("workload") or {}

            # sessions[] read-modify-write: read anchor, drop prior entry with our session_id (resume safety), append ours, write back.
            my_sessions = list(attrs["sessions"] or [])
            my_session_ids = {str((s or {}).get("session_id") or "") for s in my_sessions if isinstance(s, dict)}
            # v2: read-modify-write the recipe row; sessions[] merged in-process under the cid flock so concurrent finalises don't tear.
            merged_sessions: list[dict[str, Any]] = list(my_sessions)
            existing_row: dict[str, Any] = {}
            if self.recipe_kb is not None:
                cid = self.workload_canonical_id()
                # Read exactly the local store's authority row.
                try:
                    existing_row = self.recipe_kb.get_authoritative_recipe(canonical_id=cid) or {}
                except Exception as exc:  # noqa: BLE001 - the recipe store may be remote
                    log.info("recipe read failed (%s); finalize appends the current session only", exc)
                existing_sessions: list[dict[str, Any]] = []
                for row in existing_row.get("sessions") or []:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("session_id") or "") in my_session_ids:
                        # Resume/retry of the same session — our new entry supersedes the prior one.
                        continue
                    existing_sessions.append(dict(row))
                merged_sessions = existing_sessions + my_sessions

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
            cb_now = ss.current_best or {}
            cb_args_now = str(cb_now.get("extra_server_args") or "").strip() if isinstance(cb_now, dict) else ""
            validated_gain = float(ss.cumulative_gain_validated or 0.0)
            has_validated_win = bool((ss.optimization_stack or []) or validated_gain > 0.0 or cb_args_now)
            try:
                live_tput = float(existing_row.get("best_throughput") or 0.0)
            except (TypeError, ValueError):
                live_tput = 0.0
            if has_validated_win and my_tput > live_tput:
                overrides["best_config"] = attrs["best_config"]
                overrides["best_throughput"] = my_tput
            self.kb_amend_recipe(
                recipe_overrides=overrides,
                provenance_details={
                    "phase": "close_finalize",
                    "evidence": [
                        f"log:session-{ss.recipe_kb_session_id or self.session_dir.name}",
                    ],
                },
            )
            return {
                "status": "written",
                "reason": "",
                "backend": "local",
                "result_type": _close_out.RESULT_WRITTEN,
            }
        # Catch-all keeps CLOSE step 2.5 defensive against programmer bugs.
        except Exception as exc:
            log.exception("update_recipe raised unexpectedly")
            return {
                "status": "error",
                "reason": type(exc).__name__,
                "backend": "local",
                "result_type": _close_out.RESULT_TRANSPORT_FAILED,
                "error_class": type(exc).__name__,
            }

    def _kb_hardware_slug(self) -> str:
        """Topology-aware hardware dimension for the recipe ``canonical_id``."""
        from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology
        from hyperloom.inference_optimizer.recipe_snapshot_constants import kb_hardware_slug

        ss = self.shared_state
        return kb_hardware_slug(ss.gpu_type or "unknown_gpu", **resolve_kb_topology())

    def workload_canonical_id(self) -> str:
        """Return the workload's canonical seven-dimension Recipe identity."""
        ss = self.shared_state
        workload = ss.model_name or "unknown_model"
        hw = self._kb_hardware_slug()
        framework = str(ss.framework or "")
        framework_version = str(ss.framework_version or "")
        if not framework_version and framework:
            framework_version = detect_framework_version(framework)
        precision = str(ss.precision or "")
        model_type = str(ss.model_type or "")
        architectures = ss.model_architectures or []
        from hyperloom.common.perf_metric import agentx_active

        return recipe_canonical_id(
            model=workload,
            hardware=hw,
            framework_name=framework,
            framework_version=framework_version,
            precision=precision,
            model_type=model_type,
            architectures=architectures,
            scheme=("agentx" if agentx_active(benchmark_mode=ss.benchmark_mode) else "inference"),
        )

    def read_local_recipe_row(self) -> dict[str, Any]:
        """Load the selected store's exact authority row for writes."""
        if self.recipe_kb is None:
            return {}
        tick = int(self.shared_state.tick or 0)
        cache = self._local_recipe_cache
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == tick:
            return cache[1]
        try:
            row = (
                self.recipe_kb.get_authoritative_recipe(
                    canonical_id=self.workload_canonical_id(),
                )
                or {}
            )
        except Exception:  # noqa: BLE001 - the recipe store may be remote
            row = {}
        self._local_recipe_cache = (tick, row)
        return row

    @staticmethod
    def extract_kept_best_config(
        *,
        task: "Task",
        variant_attrs: dict[str, Any] | None = None,
        result_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a replayable ``best_config`` from a KEEP'd task or explore variant."""
        params = task.params if isinstance(getattr(task, "params", None), dict) else {}
        attrs = variant_attrs if isinstance(variant_attrs, dict) else {}

        args = _extra_server_args(attrs)
        if not args.strip():
            args = _extra_server_args(params)
        if not args.strip() and isinstance(result_dict, dict):
            args = _extra_server_args(result_dict)

        envs_raw = attrs.get("extra_envs") or params.get("extra_envs") or {}
        if not envs_raw and isinstance(result_dict, dict):
            envs_raw = result_dict.get("extra_envs") or {}
        envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}

        if not args.strip() and not envs:
            return {}

        best_config: dict[str, Any] = {}
        if args.strip():
            best_config["extra_server_args"] = args.strip()
        if envs:
            best_config["extra_envs"] = envs
        return best_config

    @staticmethod
    def kb_best_config_overrides_for_keep(
        *,
        live: Mapping[str, Any],
        best_config_candidate: Mapping[str, Any],
        throughput_after: float | None,
    ) -> dict[str, Any]:
        """Decide whether a KEEP amend should also stamp ``best_config`` on the recipe row."""
        if not best_config_candidate:
            return {}

        live_bc = live.get("best_config") if isinstance(live.get("best_config"), Mapping) else {}
        live_has_config = bool(
            _extra_server_args(live_bc).strip()
            or (isinstance(live_bc.get("extra_envs"), Mapping) and live_bc.get("extra_envs"))
        )
        try:
            live_tput = float(live.get("best_throughput") or 0.0)
        except (TypeError, ValueError):
            live_tput = 0.0
        try:
            new_tput = float(throughput_after or 0.0)
        except (TypeError, ValueError):
            new_tput = 0.0

        if not live_has_config or (new_tput > 0.0 and new_tput >= live_tput):
            overrides: dict[str, Any] = {
                "best_config": dict(best_config_candidate),
            }
            if new_tput > 0.0:
                overrides["best_throughput"] = new_tput
            return overrides
        return {}

    def kb_amend_recipe(
        self,
        *,
        append_lesson: dict[str, Any] | None = None,
        append_pitfall: dict[str, Any] | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        provenance_details: dict[str, Any] | None = None,
    ) -> None:
        """Read-modify-write helper for the recipe-snapshot KB: load live row, append lesson/pitfall, merge recipe_overrides (unset fields preserved), write back. Best-effort; lesson/pitfall appended without dedup."""
        config = getattr(self.knowledge_plane, "config", None)
        if getattr(getattr(config, "mode", None), "value", None) == "remote" or self.recipe_kb is None:
            return
        from hyperloom.common.perf_metric import agentx_active

        if agentx_active(benchmark_mode=self.shared_state.benchmark_mode):
            return
        cid = self.workload_canonical_id()

        ss = self.shared_state
        framework = str(ss.framework or "")
        framework_version = str(ss.framework_version or "")
        if not framework_version and framework:
            framework_version = detect_framework_version(framework)
        precision = str(ss.precision or "")

        # Local mode reads the exact authority row before amending it.
        try:
            live = self.recipe_kb.get_authoritative_recipe(canonical_id=cid) or {}
        except Exception as exc:  # noqa: BLE001
            log.info(
                "kb_amend_recipe: authority get_recipe failed (%s); proceeding with empty live",
                exc,
            )
            live = {}

        lessons = list(live.get("lessons") or [])
        if append_lesson is not None:
            lessons.append(append_lesson)
        pitfalls = list(live.get("pitfalls") or [])
        if append_pitfall is not None:
            pitfalls.append(append_pitfall)

        # Build put_recipe kwargs, preserving live fields the caller didn't override.
        overrides = dict(recipe_overrides or {})
        _reserved = {
            "canonical_id",
            "version",
            "created_at",
            "updated_at",
            "model",
            "hardware",
            "framework",
            "framework_name",
            "framework_version",
            "precision",
            "best_config",
            "best_throughput",
            "what_worked",
            "what_failed",
            "remaining_gaps",
            "pitfalls",
            "lessons",
            "last_profiled",
            "stack_fingerprint",
            "sessions",
            "authority",
            "confidence",
            "evidence_refs",
            "provenance",
        }
        prior_extras = {k: v for k, v in live.items() if k not in _reserved}
        merged_extras = {**prior_extras, **(overrides.get("extras") or {})}
        # Re-stamp config.json architecture-identity tags; skipped when unset.
        _arch = ss.model_architectures or []
        if isinstance(_arch, list):
            _arch_list = [str(a).strip() for a in _arch if str(a or "").strip()]
            if _arch_list:
                merged_extras["architectures"] = _arch_list
        _mtype = str(ss.model_type or "").strip()
        if _mtype:
            merged_extras["model_type"] = _mtype
        put_kwargs: dict[str, Any] = {
            "canonical_id": cid,
            "model": ss.model_name or "unknown_model",
            "hardware": self._kb_hardware_slug(),
            "framework_name": framework,
            "framework_version": framework_version,
            "precision": precision,
            "best_config": overrides.get("best_config")
            if "best_config" in overrides
            else dict(live.get("best_config") or {}),
            "best_throughput": overrides.get("best_throughput")
            if "best_throughput" in overrides
            else float(live.get("best_throughput") or 0.0),
            "what_worked": overrides.get("what_worked")
            if "what_worked" in overrides
            else list(live.get("what_worked") or []),
            "what_failed": overrides.get("what_failed")
            if "what_failed" in overrides
            else list(live.get("what_failed") or []),
            "remaining_gaps": overrides.get("remaining_gaps")
            if "remaining_gaps" in overrides
            else list(live.get("remaining_gaps") or []),
            "pitfalls": pitfalls,
            "lessons": lessons,
            "last_profiled": overrides.get("last_profiled")
            if "last_profiled" in overrides
            else str(live.get("last_profiled") or ""),
            "stack_fingerprint": overrides.get("stack_fingerprint")
            if "stack_fingerprint" in overrides
            else dict(live.get("stack_fingerprint") or {}),
            "sessions": overrides.get("sessions") if "sessions" in overrides else list(live.get("sessions") or []),
            "extras": merged_extras,
            # Preserve audit fields across the amend (else put_recipe resets them to defaults).
            "authority": overrides.get("authority")
            if "authority" in overrides
            else str(live.get("authority") or "EXPERIENTIAL"),
            "confidence": overrides.get("confidence")
            if "confidence" in overrides
            else float(live.get("confidence") or 0.85),
            "evidence_refs": overrides.get("evidence_refs")
            if "evidence_refs" in overrides
            else list(live.get("evidence_refs") or []),
            "provenance": {
                "source": "hyperloom-inference-optimizer",
                "generator": "coordinator",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="microseconds",
                ),
                "details": dict(provenance_details or {}),
            },
        }
        try:
            self.recipe_kb.put_recipe(**put_kwargs)
            self._local_recipe_cache = None
        except Exception:
            log.exception(
                "kb_amend_recipe: put_recipe failed for cid=%s",
                cid,
            )


__all__ = ["RecipeJournalCollaborator"]
