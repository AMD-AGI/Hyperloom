# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization journal, Recipe KB facts and the session's final Recipe, written from settled results."""

from __future__ import annotations

import logging as _logging
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection, Mapping, NamedTuple

from hyperloom.common.coerce import first_float, recorded_float, to_float, to_int, to_mapping, to_str_list
from hyperloom.common.io import JsonLineFormat, append_jsonl
from hyperloom.inference_optimizer.breakdown.recorder import close_out as _close_out
from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
    ARM_CONFIG,
)
from hyperloom.inference_optimizer.recipe_snapshot_constants import RecipeIdentity, detect_framework_version
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
from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStoreError, RecipeUpsert, recipe_canonical_id
from hyperloom.orchestrator.lever import LEVER_CONFIG, lever_kind_for_task

from ..collaborator import CoordinatorCollaborator
from ..state.attempt_ledger import AttemptRow, record_attempt_row
from .config import KnowledgeStoreMode
from ._config_timeline import _NON_ATTEMPT_OUTCOMES, _record_config_attempts
from ._recipe_attrs import attempt_provenance, recipe_attrs_from_state, session_best_config, workload_kb_tags

if TYPE_CHECKING:
    from ..loop.coordinator import Coordinator
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)


# Recipe snapshot severity tags (schema has no fixed enum).
_SEVERITY_CRASH: str = "crash"
_SEVERITY_REGRESS: str = "regress"


# Stable ``result_type`` codes for the reasons the remote KB Store returns.
# Use an exact lookup so a reason token cannot collide with the same text inside
# an exception name or explanatory message.
_REMOTE_RESULT_TYPES: dict[str, str] = {
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


#: Explore outcomes the journal collapses into a plain revert.
_MEASURED_LOSS_OUTCOMES: frozenset[str] = frozenset({OUTCOME_REVERT, "FAILED"})


#: Recipe fields an amend carries forward when the caller does not override
#: them, each with the empty value ``put_recipe`` falls back to.
PRESERVED_RECIPE_FIELDS: Mapping[str, Any] = MappingProxyType(
    {
        "best_config": {},
        "best_throughput": 0.0,
        "what_worked": [],
        "what_failed": [],
        "remaining_gaps": [],
        "last_profiled": "",
        "stack_fingerprint": {},
        "sessions": [],
        "authority": "EXPERIENTIAL",
        "confidence": 0.85,
        "evidence_refs": [],
    }
)

#: Row keys ``put_recipe`` owns; everything else on a live row is a session
#: extra and survives the amend untouched.
_RECIPE_OWNED_KEYS: frozenset[str] = frozenset(PRESERVED_RECIPE_FIELDS) | frozenset(
    {
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
        "pitfalls",
        "lessons",
        "provenance",
    }
)

#: Per-variant measurement detail the journal row carries beyond the headline
#: gain and throughput.
_VARIANT_DETAIL_METRICS: tuple[str, ...] = (
    "runtime_sec",
    "wall_clock_ratio_vs_baseline",
    "estimated_output_throughput",
)


@dataclass(frozen=True)
class FactRecord:
    """One settled outcome to write to the Recipe KB as a lesson or a pitfall.

    ``pitfall_severity_dict`` is the result dict per-task, and the variant's merged
    metrics + outcome per-variant. ``variant_name`` is set only by the per-variant
    caller, and reaching ``provenance_details`` is what it is for.
    """

    task: "Task"
    source_session_id: str
    is_keep: bool
    change: str
    gain_pct: float | None
    throughput_after: float | None
    evidence_refs: list[str]
    pitfall_severity_dict: dict[str, Any]
    variant_name: str | None = None


@dataclass(frozen=True)
class RemoteRecipeAudit:
    """The outcome of one remote Recipe KB publish attempt, as an audit row records it."""

    status: str
    canonical_id: str
    session_id: str
    primary_metric: str = ""
    primary_value: float = 0.0
    reason: str = ""
    error: Exception | None = None


class _FrameworkIdentity(NamedTuple):
    name: str
    version: str
    precision: str


class _VariantMeasurement(NamedTuple):
    gain_pct: float | None
    throughput_after: float | None
    detail: dict[str, Any]


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
    """First non-zero ``predicted_gain_pct`` across ordered sources.

    ``None`` when none carry one, so the journal row for an unpredicted
    (default-grid) change stays ``predicted``-free rather than recording a fake 0.
    """
    for src in sources:
        if not isinstance(src, dict):
            continue
        val = to_float(src.get("predicted_gain_pct"))
        if val is not None and val != 0.0:
            return val
    return None


def _extra_server_args(payload: Mapping[str, Any]) -> str:
    return " ".join(to_str_list(payload.get("extra_server_args")))


def _record_config_run(coord: Any, *, task: Any, result_dict: Mapping[str, Any]) -> None:
    """Record the configuration arm's grid dispatch on the framework event.

    The arm's dispatch is one ``explore`` task, and it earns a run row for the
    same reason a specialist dispatch does: the event reduces its status over
    what it dispatched, so an arm whose grid came back with nothing must not
    read as an arm that was never tried. Recorded at completion, where the
    outcome is known, with the dispatch time taken off the task row; a grid
    that measured nothing still lands, which is the case
    :func:`record_config_attempts` never sees.
    """
    recorder = coord._coord.phase_framework.timeline()
    if recorder is None:
        return
    from hyperloom.common.timeutil import now_iso
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ROLE_CONFIG

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


def rows_not_superseded(live_rows: Any, superseding: Any, key: str) -> list[dict[str, Any]]:
    claimed = {str(row.get(key) or "") for row in superseding or [] if isinstance(row, dict)}
    kept: list[dict[str, Any]] = []
    for row in live_rows or []:
        if not isinstance(row, dict):
            continue
        if str(row.get(key) or "") in claimed:
            continue
        kept.append(dict(row))
    return kept


def appended(live_facts: Any, addition: Mapping[str, Any] | None) -> list[Any]:
    facts = list(live_facts or [])
    if addition is not None:
        facts.append(addition)
    return facts


def amended_recipe_fields(live: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    # An absent or empty live value falls back to the field's own default, so
    # ``put_recipe`` never sees ``None`` where it declares a container.
    merged: dict[str, Any] = {}
    for field, default in PRESERVED_RECIPE_FIELDS.items():
        if field in overrides:
            merged[field] = overrides[field]
            continue
        live_value = live.get(field)
        merged[field] = type(default)(live_value) if live_value else copy(default)
    return merged


def amended_recipe_extras(live: Mapping[str, Any], overrides: Mapping[str, Any], state: Any) -> dict[str, Any]:
    # The architecture tags are re-stamped from config.json each time because a
    # row written before the architecture was read carries none.
    extras = {key: value for key, value in live.items() if key not in _RECIPE_OWNED_KEYS}
    extras.update(overrides.get("extras") or {})
    architectures = state.model_architectures or []
    if isinstance(architectures, list):
        named = [str(name).strip() for name in architectures if str(name or "").strip()]
        if named:
            extras["architectures"] = named
    model_type = str(state.model_type or "").strip()
    if model_type:
        extras["model_type"] = model_type
    return extras


def _framework_identity(state: Any) -> _FrameworkIdentity:
    # Probed from the installed framework when the session never recorded a
    # version, so a recipe is never filed under a blank one.
    name = str(state.framework or "")
    version = str(state.framework_version or "")
    if not version and name:
        version = detect_framework_version(name)
    return _FrameworkIdentity(name=name, version=version, precision=str(state.precision or ""))


def _result_payload(result: Any) -> dict[str, Any]:
    payload = result.result if hasattr(result, "result") else (result or {})
    return payload if isinstance(payload, dict) else {}


def _grid_attempt_row(
    variant_outcome: Any,
    *,
    task_id: str,
    round_id: str,
    adopted: bool,
) -> AttemptRow | None:
    outcome = str(variant_outcome.get("outcome") or "") if isinstance(variant_outcome, dict) else ""
    if not outcome or outcome in _NON_ATTEMPT_OUTCOMES:
        return None
    metrics = variant_outcome.get("metrics") if isinstance(variant_outcome.get("metrics"), dict) else {}
    return AttemptRow(
        lever_kind=LEVER_CONFIG,
        task_id=task_id,
        round_id=round_id,
        fingerprint=str(variant_outcome.get("fingerprint") or ""),
        variant_name=str(variant_outcome.get("variant_name") or ""),
        outcome=outcome,
        adopted=adopted,
        gain_pct=metrics.get("gain_pct"),
        before_tput=metrics.get("base_tput"),
        after_tput=metrics.get("tput"),
        error_class=str(variant_outcome.get("error_class") or ""),
        provenance=str(variant_outcome.get("provenance") or ""),
    )


def _variant_verdict(outcome: str, adopted: bool) -> Verdict:
    if outcome == OUTCOME_KEEP:
        return Verdict.ADOPTED if adopted else Verdict.REFUSED
    if outcome in _MEASURED_LOSS_OUTCOMES:
        return Verdict.REVERTED
    return Verdict.REFUSED


def _variant_measurement(metrics: Mapping[str, Any]) -> _VariantMeasurement:
    return _VariantMeasurement(
        gain_pct=to_float(metrics.get("gain_pct")),
        throughput_after=to_float(metrics.get("output_throughput")),
        detail={key: metrics[key] for key in _VARIANT_DETAIL_METRICS if metrics.get(key) is not None},
    )


def _variant_change_attrs(variant_attrs: Any, variant_name: str) -> dict[str, Any]:
    attrs = dict(variant_attrs) if isinstance(variant_attrs, dict) else {}
    named_knob = attrs.get("extra_server_args") or attrs.get("extra_envs") or attrs.get("name")
    if not named_knob and variant_name:
        attrs["name"] = variant_name
    return attrs


def _finalize_outcome(*, status: str, reason: str, backend: str, result_type: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "backend": backend, "result_type": result_type}


def _finalize_error(exc: BaseException, *, reason: str, backend: str, result_type: str) -> dict[str, Any]:
    failed = _finalize_outcome(status="error", reason=reason, backend=backend, result_type=result_type)
    return {**failed, "error_class": type(exc).__name__}


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
                    lever_kind=lever_kind_for_task(lift_kind, result),
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
        except (OSError, ValueError, TypeError):
            log.exception("integrate journal append failed")

    def _source_session_id(self) -> str:
        """The hyperloom-local session id stamped on KB fact writes, never a KB-side one."""
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

        ``adopted_variants`` holds the fingerprints of the explore winners whose lift
        landed; a per-variant KEEP outside it did not adopt anything.
        """
        result_dict = _result_payload(result)
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
            task_id = str(task.task_id or "")
            for vo in per_variant:
                adopted = str(vo.get("fingerprint") or "") in adopted_variants
                row = _grid_attempt_row(vo, task_id=task_id, round_id=round_id, adopted=adopted)
                if row is not None:
                    record_attempt_row(self.shared_state, row)
                self._record_fact_per_variant(
                    task=task, source_session_id=source_session_id, variant_outcome=vo, adopted=adopted
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
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume)."""
        existing = self._journal
        if existing is None:
            ss = self.shared_state
            fresh = Journal(
                session_id=str(ss.recipe_kb_session_id or "") or str(ss.session_id or "") or self.session_dir.name,
                model=str(ss.model_name or ""),
                hardware=str(ss.gpu_type or ""),
                framework=str(ss.framework or ""),
                baseline_throughput=float(ss.baseline_tput or 0.0),
            )
            self._journal = Journal.load_or_create(self.session_dir, fresh)
        else:
            # Backfill baseline once the baseline executor finishes.
            existing.update_baseline(float(self.shared_state.baseline_tput or 0.0))
        return self._journal

    def _pitfall_severity_for(
        self,
        result_dict: dict[str, Any] | None,
    ) -> str | None:
        """The severity of the pitfall a failed result warrants, or ``None`` for no pitfall."""
        if not result_dict:
            return None
        error_class = str(result_dict.get("error_class") or "").lower()
        # ``detokenizer_stall`` is a hang in all but name; record it as a
        # crash-severity pitfall so the offending config is not re-proposed.
        if error_class in ("crash", "oom", "hang", "detokenizer_stall"):
            return _SEVERITY_CRASH
        status = str(result_dict.get("status") or "").lower()
        if status in ("crash", "oom", "hang"):
            return _SEVERITY_CRASH
        gain_pct = to_float(result_dict.get("gain_pct"))
        if gain_pct is not None and gain_pct <= self.PITFALL_REGRESS_THRESHOLD_PCT:
            return _SEVERITY_REGRESS
        return None

    def journal_entry_phase(self) -> str:
        """The uppercased phase name journal entries are stamped with, ``"UNKNOWN"`` when unset."""
        return str(self.shared_state.phase or "").strip().upper() or "UNKNOWN"

    def _record_fact_impl(self, fact: FactRecord) -> None:
        """Write one KB lesson (KEEP with positive gain) or one KB pitfall (REVERT/failure).

        Call only after the journal entry has been appended and ``recipe_kb`` is
        confirmed non-None by the caller.
        """
        state = self.shared_state
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        provenance_base: dict[str, Any] = {
            "source_session_id": fact.source_session_id,
            "source_task_id": fact.task.task_id,
            "evidence": list(fact.evidence_refs or []),
            "applicable_models": [str(state.model_name)] if state.model_name else [],
            "applicable_hardware": [str(state.gpu_type)] if state.gpu_type else [],
            "extra": dict(workload_kb_tags(state)),
            "now": now_iso,
        }
        if fact.variant_name is not None:
            provenance_base["source_variant_name"] = fact.variant_name

        lesson = None
        pitfall = None
        if fact.is_keep and fact.gain_pct is not None and fact.gain_pct > 0:
            lesson = {
                "statement": self._build_statement(change=fact.change, kind="lesson"),
                "measured_impact": self._build_measured_impact(
                    gain_pct=fact.gain_pct,
                    throughput_after=fact.throughput_after,
                    stack_depth=len(self.shared_state.optimization_stack or []),
                    measured_at=now_iso,
                ),
            }
        else:
            severity = self._pitfall_severity_for(fact.pitfall_severity_dict)
            if severity is not None:
                pitfall = {
                    "description": self._build_statement(change=fact.change, severity=severity, kind="pitfall"),
                    "severity": severity,
                }

        try:
            if lesson is not None:
                # Promotion has already folded this KEEP into the session best; the task's own
                # args are only its layer, and an integrate_patch KEEP has none at all.
                recipe_overrides = self.kb_best_config_overrides_for_keep(
                    live=self.read_local_recipe_row(),
                    best_config=session_best_config(state.current_best or {}, state.optimization_stack or []),
                )
                self.kb_amend_recipe(
                    append_lesson=lesson,
                    recipe_overrides=recipe_overrides or None,
                    provenance_details=provenance_base,
                )
            elif pitfall is not None:
                self.kb_amend_recipe(append_pitfall=pitfall, provenance_details=provenance_base)
        except (LocalRecipeStoreError, OSError, ValueError) as error:
            log.warning(
                "recipe KB write for task %s%s failed: %s: %s",
                fact.task.task_id,
                f" variant {fact.variant_name}" if fact.variant_name is not None else "",
                type(error).__name__,
                error,
            )

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        verdict: Verdict,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact.

        ``ADOPTED`` journals a KEEP and may write a lesson; any other verdict
        journals a non-KEEP row and may write a pitfall.
        """
        journal = self.ensure_journal()
        # integrate_patch reports its delta under ``delta_pct``;
        # fall back to it so a reverted/kept patch shows its REAL measured delta
        # in the journal instead of a null gain.
        gain_pct = first_float(result_dict.get("gain_pct"), result_dict.get("delta_pct"))
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
                lever_kind=lever_kind_for_task(kind, result_dict),
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
            FactRecord(
                task=task,
                source_session_id=source_session_id,
                is_keep=is_keep,
                change=change,
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                # evidence_refs (log:task-...) gives traceability since source_session_id lands in attrs.
                evidence_refs=[f"log:task-{task.task_id}"],
                pitfall_severity_dict=result_dict,
            )
        )

    def _build_statement(
        self,
        *,
        change: str,
        kind: str,
        severity: str | None = None,
    ) -> str:
        """Build the lesson statement / pitfall description hashed into the KB canonical_id.

        Identity is framework + change + model/hw. It MUST exclude volatile fields
        (``gain_pct`` and the like) so N sessions merge instead of producing N rows.
        """
        framework = str(self.shared_state.framework or "").strip()
        fw_tag = f"[{framework or '?'}] "
        model = self.shared_state.model_name or "?"
        hw = self.shared_state.gpu_type or "?"
        if kind == "lesson":
            return f"{fw_tag}{change} on {model}/{hw}"
        # Remaining kind is "pitfall"
        return f"{fw_tag}{change} → {severity or '?'} on {model}/{hw}"

    @staticmethod
    def _build_measured_impact(
        *,
        gain_pct: float | None,
        throughput_after: float | None,
        stack_depth: int,
        measured_at: str,
    ) -> dict[str, Any]:
        """Structured ``measured_impact`` payload, a dict rather than a legacy string so consumers parse without regex.

        ``stack_depth`` is the optimization-stack length before this lesson lands.
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

        An executor KEEP that was not adopted journals as ``no_promote``.
        """
        journal = self.ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw in _NON_ATTEMPT_OUTCOMES:
            return
        outcome = derive_journal_outcome(_variant_verdict(outcome_raw, adopted), variant_outcome)
        variant_name = str(variant_outcome.get("variant_name") or "")
        metrics = to_mapping(variant_outcome.get("metrics"))
        measurement = _variant_measurement(metrics)
        variant_attrs = variant_outcome.get("variant") or {}
        attrs = variant_attrs if isinstance(variant_attrs, dict) else None
        kind = classify_change_kind(task.kind, attrs)
        change_attrs = _variant_change_attrs(variant_attrs, variant_name)
        change = summarize_change(task.kind, change_attrs, None)
        # Only a measured loss carries a failure; a refusal has nothing to report.
        reverted = outcome == OUTCOME_REVERT
        tick = int(self.shared_state.tick or 0)
        journal.append_entry(
            JournalEntry(
                phase=self.journal_entry_phase(),
                lever_kind=lever_kind_for_task(kind, variant_outcome),
                iter=tick,
                kind=kind,
                change=change,
                outcome=outcome,
                gain_pct=measurement.gain_pct,
                throughput_after=measurement.throughput_after,
                error_class=(str(variant_outcome.get("error_class") or "") or None) if reverted else None,
                reason=(str(variant_outcome.get("reason") or "") or None) if reverted else None,
                task_id=task.task_id,
                variant_name=variant_name,
                provenance=str(variant_outcome.get("provenance") or ""),
                scope=str(variant_outcome.get("scope") or ""),
                fingerprint=str(variant_outcome.get("fingerprint") or ""),
                # Proposer attribution and the measurement detail beyond the
                # headline figures, so the row records how the change measured
                # and who proposed it.
                metrics=measurement.detail,
                tick=tick,
                predicted_gain_pct=_predicted_gain(variant_outcome, attrs, getattr(task, "params", None)),
            )
        )

        if self.recipe_kb is None:
            return

        self._record_fact_impl(
            FactRecord(
                task=task,
                source_session_id=source_session_id,
                is_keep=outcome == OUTCOME_KEEP,
                change=change,
                gain_pct=measurement.gain_pct,
                throughput_after=measurement.throughput_after,
                # Workload-shape tags — see _record_fact_per_task.
                evidence_refs=[f"log:task-{task.task_id}", f"variant:{variant_name}"],
                pitfall_severity_dict={
                    **metrics,
                    "error_class": variant_outcome.get("error_class"),
                    "status": variant_outcome.get("outcome"),
                },
                variant_name=variant_name,
            )
        )

    def _record_remote_recipe_audit(self, source: str, audit: RemoteRecipeAudit) -> None:
        """Append one best-effort, secret-free KB Store publish audit row."""
        try:
            from .remote_recipe.audit import publish_error_fields
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
                "resolution": audit.status,
                "success": audit.status in {"written", "skipped"},
                "generator": source,
                "phase": "CLOSE",
                "status": audit.status,
                "reason": audit.reason,
                "request": {
                    "canonical_id": audit.canonical_id or None,
                    "session_id": audit.session_id or None,
                },
                "result": {
                    "canonical_id": audit.canonical_id,
                    "session_id": audit.session_id,
                    "created": audit.status == "written",
                },
                "provenance": {
                    "component": "remote_recipe",
                    "source": source,
                },
            }
            if audit.primary_metric:
                row["result"].update(
                    {
                        "primary_metric": audit.primary_metric,
                        "primary_value": audit.primary_value,
                    }
                )
            if audit.primary_metric == "optimized_throughput":
                row["result"]["best_throughput"] = audit.primary_value
            if audit.error is not None:
                row["error"] = publish_error_fields(audit.error)
            append_jsonl(
                recipe_snapshot_audit_jsonl(self.session_dir),
                row,
                make_parents=True,
                fmt=JsonLineFormat(sort_keys=True),
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
        except OSError:
            log.exception("Recipe finalize pending-state save failed")

        try:
            raw = self.finalize_recipe_and_journal(source=source) or {}
            outcome = {
                **dict(raw),
                "source": source,
                "attempt": attempts,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # isolation boundary: one Recipe finalize attempt, settled as transport_failed
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
        except OSError:
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
        tput = state.current_best.get("tput")
        validated_gain = state.cumulative_gain_validated
        result_type = str(outcome.get("result_type") or "")
        if result_type == _close_out.RESULT_UNVALIDATED_RECIPE:
            # The throughput and the gain belong to different Recipes here.
            tput = validated_gain = None
        _close_out.record_write_back_settled(
            self.session_dir,
            _close_out.WriteBackSettlement(
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
            ),
        )

    def _write_back_scope(self) -> dict[str, Any]:
        """The workload dimensions the published recipe is partitioned by, in ``RecipeScope`` shape."""
        state = self.shared_state
        return {
            "kernel_optimizer": str(state.kernel_optimizer or ""),
            "tp": to_int(state.tp, None),
            "conc": to_int(state.conc, None),
            "isl": to_int(state.isl, None),
            "osl": to_int(state.osl, None),
        }

    def _unvalidated_stack_outcome(self) -> dict[str, Any] | None:
        # ``current_best`` and ``cumulative_gain_validated`` are only one
        # Recipe's figures when nothing was lifted since the last validation;
        # otherwise every sink below would pair the newer config with the
        # older gain.
        state = self.shared_state
        if not state.optimization_stack_has_unvalidated_keeps():
            return None
        log.warning(
            "Recipe finalize skipped: working recipe is not validated (stack=%s/%s generation=%s/%s)",
            len(state.optimization_stack or []),
            state.cumulative_gain_validated_stack_len,
            state.working_recipe_generation,
            state.validated_recipe_generation,
        )
        return _finalize_outcome(
            status="skipped",
            reason="unvalidated_recipe_stack",
            backend="none",
            result_type=_close_out.RESULT_UNVALIDATED_RECIPE,
        )

    def finalize_recipe_and_journal(
        self,
        *,
        source: str = "close",
    ) -> dict[str, Any]:
        """Finalize Recipe state and return a secret-free observable outcome."""
        unvalidated = self._unvalidated_stack_outcome()
        if unvalidated is not None:
            return unvalidated
        state = self.shared_state
        final_tput = float(state.current_best.get("tput", 0.0))
        self.ensure_journal().finalize(
            final_throughput=final_tput if final_tput > 0 else None,
            total_gain_pct=float(state.cumulative_gain_validated or 0.0),
        )

        knowledge_plane = self.knowledge_plane
        if bool(getattr(knowledge_plane, "kb_disabled", False)):
            log.info("Recipe KB finalize skipped (--degraded-kb)")
            return _finalize_outcome(
                status="skipped",
                reason="degraded_kb",
                backend="disabled",
                result_type=_close_out.RESULT_KB_DISABLED,
            )

        from ..knowledge.config import KnowledgeConfig

        try:
            config = getattr(knowledge_plane, "config", None) or KnowledgeConfig.from_env()
        except ValueError as exc:
            log.exception("Recipe KB finalize configuration failed (non-fatal)")
            return _finalize_error(
                exc,
                reason=f"configuration:{type(exc).__name__}",
                backend="unknown",
                result_type=_close_out.RESULT_CONFIGURATION_FAILED,
            )
        from hyperloom.common.perf_metric import agentx_active

        remote = config.mode is KnowledgeStoreMode.REMOTE
        if agentx_active(benchmark_mode=self.shared_state.benchmark_mode) and not remote:
            return _finalize_outcome(
                status="skipped",
                reason="agentx_local_store_unsupported",
                backend=str(getattr(config, "mode", "") or "unknown"),
                result_type=_close_out.RESULT_AGENTX_BLOCKED,
            )
        if remote:
            return self._publish_remote_recipe(source)
        return self._write_local_recipe()

    def _publish_remote_recipe(self, source: str) -> dict[str, Any]:
        # Remote mode has one Recipe sink: the KB Store final session writer.
        # T0 and runtime amendment are intentionally absent.
        state = self.shared_state
        session_id = str(state.recipe_kb_session_id or state.session_id or self.session_dir.name)
        canonical_id = ""
        from ..knowledge.remote_recipe import KBStoreError, RemoteRecipeClient, write_final_remote_recipe

        try:
            canonical_id = self.workload_canonical_id()
            written = write_final_remote_recipe(state, canonical_id, session_id, client=RemoteRecipeClient.from_env())
        except (KBStoreError, OSError, ValueError) as exc:
            self._record_remote_recipe_audit(
                source,
                RemoteRecipeAudit(
                    status="error",
                    canonical_id=canonical_id,
                    session_id=session_id,
                    error=exc,
                ),
            )
            log.exception("Remote Recipe KB finalize failed (non-fatal)")
            return {
                **_finalize_error(
                    exc,
                    reason=type(exc).__name__,
                    backend="kb-store",
                    result_type=(
                        _close_out.RESULT_BUNDLE_BUILD_FAILED
                        if type(exc).__name__ == _REMOTE_VALIDATION_ERROR
                        else _close_out.RESULT_TRANSPORT_FAILED
                    ),
                ),
                "canonical_id": canonical_id,
                "session_id": session_id,
            }
        log.info(
            "Remote Recipe KB finalize: status=%s reason=%s cid=%s sid=%s",
            written.status,
            written.reason,
            canonical_id,
            written.session_id,
        )
        self._record_remote_recipe_audit(
            source,
            RemoteRecipeAudit(
                status=written.status,
                canonical_id=canonical_id,
                session_id=written.session_id,
                primary_metric=written.primary_metric,
                primary_value=written.primary_value,
                reason=written.reason,
            ),
        )
        return {
            **_finalize_outcome(
                status=written.status,
                reason=written.reason,
                backend="kb-store",
                result_type=_remote_result_type(written.status, written.reason),
            ),
            "canonical_id": str(getattr(written, "canonical_id", "") or canonical_id),
            "session_id": str(getattr(written, "session_id", "") or session_id),
        }

    def _final_recipe_overrides(self) -> dict[str, Any]:
        # Read-modify-write under the cid flock, so two finalises of the same
        # session leave one row rather than tearing the lists they both append to.
        state = self.shared_state
        attrs = recipe_attrs_from_state(state, self.session_dir.name)
        live = self.recipe_kb.get_authoritative_recipe(canonical_id=self.workload_canonical_id()) or {}
        my_sessions = list(attrs["sessions"] or [])
        extras = dict(attrs.get("workload") or {})
        overrides: dict[str, Any] = {
            "what_worked": attrs["what_worked"],
            "what_failed": attrs["what_failed"],
            "last_profiled": attrs["last_profiled"],
            "sessions": rows_not_superseded(live.get("sessions"), my_sessions, "session_id") + my_sessions,
            "extras": extras,
        }
        # The row's own figures survive a session that never validated a win,
        # or whose win does not beat what the row already advertises.
        my_throughput = float(attrs.get("best_throughput") or 0.0)
        won = bool(state.optimization_stack or float(state.cumulative_gain_validated or 0.0) > 0.0)
        won = won or bool(str(state.current_best.get("extra_server_args") or "").strip())
        if won and my_throughput > recorded_float(live.get("best_throughput")):
            overrides["best_config"] = attrs["best_config"]
            overrides["best_throughput"] = my_throughput
        return overrides

    def _write_local_recipe(self) -> dict[str, Any]:
        # Local mode never consults ambient KB_STORE_* credentials.
        if self.recipe_kb is None:
            return _finalize_outcome(
                status="skipped",
                reason="no_recipe_backend",
                backend="local",
                result_type=_close_out.RESULT_KB_DISABLED,
            )
        state = self.shared_state
        model_name = state.model_name or ""
        gpu_type = state.gpu_type or ""
        if not model_name or not gpu_type:
            log.info(
                "recipe KB finalize_recipe: missing model/hardware (model=%r hardware=%r); skipping update_recipe",
                model_name,
                gpu_type,
            )
            return _finalize_outcome(
                status="skipped",
                reason="missing_model_or_hardware",
                backend="local",
                result_type=_close_out.RESULT_INVALID_SCOPE,
            )
        self.kb_amend_recipe(
            recipe_overrides=self._final_recipe_overrides(),
            provenance_details={
                "phase": "close_finalize",
                "evidence": [
                    f"log:session-{state.recipe_kb_session_id or self.session_dir.name}",
                ],
            },
        )
        return _finalize_outcome(
            status="written",
            reason="",
            backend="local",
            result_type=_close_out.RESULT_WRITTEN,
        )

    def _kb_hardware_slug(self) -> str:
        """Topology-aware hardware dimension for the recipe ``canonical_id``."""
        from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology
        from hyperloom.inference_optimizer.recipe_snapshot_constants import kb_hardware_slug

        ss = self.shared_state
        return kb_hardware_slug(ss.gpu_type or "unknown_gpu", resolve_kb_topology())

    def workload_canonical_id(self) -> str:
        """Return the workload's canonical seven-dimension Recipe identity."""
        state = self.shared_state
        identity = _framework_identity(state)
        from hyperloom.common.perf_metric import agentx_active

        rid = RecipeIdentity(
            model=state.model_name or "unknown_model",
            hardware=self._kb_hardware_slug(),
            framework_name=identity.name,
            framework_version=identity.version,
            precision=identity.precision,
            model_type=str(state.model_type or ""),
            architectures=state.model_architectures or [],
        )
        return recipe_canonical_id(rid, "agentx" if agentx_active(benchmark_mode=state.benchmark_mode) else "inference")

    def read_local_recipe_row(self) -> dict[str, Any]:
        """Load the selected store's exact authority row for writes."""
        if self.recipe_kb is None:
            return {}
        tick = int(self.shared_state.tick or 0)
        cache = self._local_recipe_cache
        if cache is not None and cache[0] == tick:
            return cache[1]
        row = self.recipe_kb.get_authoritative_recipe(canonical_id=self.workload_canonical_id()) or {}
        self._local_recipe_cache = (tick, row)
        return row

    @staticmethod
    def kb_best_config_overrides_for_keep(
        *,
        live: Mapping[str, Any],
        best_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Decide whether a KEEP amend should also stamp the session ``best_config`` on the recipe row."""
        if not best_config:
            return {}

        live_bc = live.get("best_config") if isinstance(live.get("best_config"), Mapping) else {}
        live_has_config = bool(
            _extra_server_args(live_bc).strip()
            or (isinstance(live_bc.get("extra_envs"), Mapping) and live_bc.get("extra_envs"))
        )
        live_tput = recorded_float(live.get("best_throughput"))
        new_tput = recorded_float(best_config.get("tput"))
        if not live_has_config or (new_tput > 0.0 and new_tput >= live_tput):
            overrides: dict[str, Any] = {
                "best_config": dict(best_config),
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
        """Read-modify-write the local recipe row: append the fact, merge the overrides, write it back.

        Read or write failures propagate. A lesson or pitfall is appended
        without dedup, and a field the caller did not override keeps the value
        the live row carries.
        """
        config = getattr(self.knowledge_plane, "config", None)
        mode = getattr(getattr(config, "mode", None), "value", None)
        if mode == KnowledgeStoreMode.REMOTE.value or self.recipe_kb is None:
            return
        from hyperloom.common.perf_metric import agentx_active

        state = self.shared_state
        if agentx_active(benchmark_mode=state.benchmark_mode):
            return
        canonical_id = self.workload_canonical_id()
        identity = _framework_identity(state)
        live = self.recipe_kb.get_authoritative_recipe(canonical_id=canonical_id) or {}
        overrides = dict(recipe_overrides or {})
        upsert = RecipeUpsert(
            canonical_id=canonical_id,
            model=state.model_name or "unknown_model",
            hardware=self._kb_hardware_slug(),
            framework_name=identity.name,
            framework_version=identity.version,
            precision=identity.precision,
            pitfalls=appended(live.get("pitfalls"), append_pitfall),
            lessons=appended(live.get("lessons"), append_lesson),
            extras=amended_recipe_extras(live, overrides, state),
            provenance={
                "source": "hyperloom-inference-optimizer",
                "generator": "coordinator",
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "details": dict(provenance_details or {}),
            },
            **amended_recipe_fields(live, overrides),
        )
        self.recipe_kb.put_recipe(upsert)
        self._local_recipe_cache = None


__all__ = ["RecipeJournalCollaborator", "attempt_provenance", "recipe_attrs_from_state", "workload_kb_tags"]
