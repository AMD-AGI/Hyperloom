# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization journal, Recipe KB facts and the session's final Recipe, written from settled results."""

from __future__ import annotations

import logging as _logging
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection, Mapping, NamedTuple, Sequence

from hyperloom.common.coerce import recorded_float, to_float, to_int
from hyperloom.common.io import JsonLineFormat, append_jsonl
from hyperloom.inference_optimizer.breakdown.recorder import close_out as _close_out
from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
    ARM_CONFIG,
    DISPOSITION_ATTEMPTED,
    STEP_ATTEMPTED,
    GateVerdict,
)
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
from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStoreError, recipe_canonical_id
from hyperloom.orchestrator.lever import LEVER_CONFIG, lever_kind_for_task

from ..collaborator import CoordinatorCollaborator
from ..state.attempt_ledger import AttemptRow, record_attempt_row
from ..state.failure_evidence import UNMEASURED_OUTCOMES, classify_failure_attribution
from .config import KnowledgeStoreMode
from .workload_tags import workload_shape_tags

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


#: Explore outcomes the journal collapses into a plain revert.
_MEASURED_LOSS_OUTCOMES: frozenset[str] = frozenset({OUTCOME_REVERT, "FAILED"})

#: Explore outcomes that are not an attempt at anything: a variant skipped as a
#: duplicate was never measured, so no measurement would back its funnel row.
_NON_ATTEMPT_OUTCOMES: frozenset[str] = frozenset({"SKIPPED_DEDUP"})

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

    Attributes:
        task: The completed task (provides task_id).
        source_session_id: Hyperloom-local session id stamped on provenance.
        is_keep: True when the outcome is a validated KEEP.
        change: Summarized change string (used in the statement).
        gain_pct: Measured gain percentage, or ``None``.
        throughput_after: Measured throughput after the change, or ``None``.
        best_config_candidate: Pre-extracted best-config dict (differs between
            per-task and per-variant callers).
        evidence_refs: Evidence reference strings stamped on the provenance
            (task-only or task+variant refs).
        pitfall_severity_dict: The dict passed to ``_pitfall_severity_for``
            (per-task passes the result dict; per-variant passes a merged
            metrics + outcome dict).
        variant_name: Variant name, present only for per-variant calls; added to
            ``provenance_details`` when non-None.
    """

    task: "Task"
    source_session_id: str
    is_keep: bool
    change: str
    gain_pct: float | None
    throughput_after: float | None
    best_config_candidate: dict[str, Any] | None
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


class _VariantFailure(NamedTuple):
    error_class: str | None
    reason: str | None


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


def _attempt_failure(row: Mapping[str, Any], outcome: str) -> dict[str, Any]:
    error_class = str(row.get("error_class") or "")
    error_excerpt = str(row.get("error_excerpt") or "")
    attribution = ""
    if outcome in UNMEASURED_OUTCOMES:
        attribution = classify_failure_attribution(
            error_class=error_class,
            error_excerpt=error_excerpt,
            reason=row.get("reason"),
            explicit=row.get("failure_attribution"),
        )
    return {"error_class": error_class, "error_excerpt": error_excerpt, "attribution": attribution}


def _attempt_fields(
    row: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    shared: Mapping[str, str],
    adopted: bool,
) -> dict[str, Any]:
    variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    outcome = str(row.get("outcome") or "")
    # This arm gates accuracy rather than reporting it, so the block both arms
    # carry is projected from the gate that ruled. No gate row means nothing
    # gated the variant, which is not a gate that refused it.
    ruling = next((gate for gate in gates if str(gate.get("gate") or "") == "accuracy"), {})
    return {
        "arm": ARM_CONFIG,
        **shared,
        "provenance": row.get("provenance"),
        "outcome": outcome,
        "reason": row.get("reason"),
        "reasoning": variant.get("note"),
        "reasoning_origin": variant.get("reasoning_origin"),
        "experience_citations": variant.get("experience_citations"),
        "stage": row.get("stage"),
        "fingerprint": row.get("fingerprint"),
        # The fingerprint is the join key; the name is what a reader
        # recognises the variant by.
        "variant_name": row.get("variant_name"),
        "measurement": {
            # Both ends of the pair: the anchor advances on every KEEP, so a
            # percentage without its denominator adds to nothing.
            "before_tput": metrics.get("base_tput"),
            "after_tput": metrics.get("tput"),
            "gain_pct": metrics.get("gain_pct"),
            "runtime_sec": metrics.get("runtime_sec"),
            "estimated_output_throughput": metrics.get("estimated_output_throughput"),
        },
        "config_delta": {
            "extra_server_args": variant.get("extra_server_args"),
            "extra_envs": variant.get("extra_envs"),
            "remove_args": variant.get("remove_args"),
            "unset_envs": variant.get("unset_envs"),
            "args_mode": variant.get("args_mode"),
        },
        "accuracy": {
            "required": True if ruling else None,
            "value": ruling.get("observed"),
            "reference": ruling.get("threshold"),
            "passed": ruling.get("passed"),
        },
        "failure": _attempt_failure(row, outcome),
        "artifacts": {
            "workspace": row.get("workspace"),
            "server_log_path": row.get("server_log_path"),
            "raw_result_path": row.get("raw_result_path"),
        },
        "decision": outcome,
        "adopted": adopted,
        # Every variant recorded here ran: the deduped one is skipped above.
        "benched": True,
        # Recorded rather than referenced: every KEEP advances the stack, so
        # the session's current config is not what this variant was measured
        # on top of.
        "measured_against": row.get("measured_against"),
        # What stood behind the verdict. Absent when nothing ruled.
        "validation_basis": row.get("validation_basis"),
        # A pair is what makes a gain addable, so eligibility follows the pair
        # being present rather than the outcome being a KEEP.
        "attribution_eligible": (adopted and metrics.get("base_tput") is not None and metrics.get("tput") is not None),
    }


def _record_variant_attempt(
    recorder: Any,
    row: Any,
    shared: Mapping[str, str],
    adopted_variants: Collection[str],
) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("outcome") or "") in _NON_ATTEMPT_OUTCOMES:
        return False
    # The fingerprint identifies the variant within the round and the round
    # within the task, so the three together identify the attempt.
    fingerprint = str(row.get("fingerprint") or "")
    parts = (shared["task_id"], shared["round_id"], fingerprint)
    attempt_id = ":".join(part for part in parts if part) or shared["task_id"]
    if not attempt_id:
        return False
    gates = [gate for gate in (row.get("gates") or []) if isinstance(gate, dict)]
    recorder.record_attempt(attempt_id, **_attempt_fields(row, gates, shared, fingerprint in adopted_variants))
    for gate in gates:
        recorder.record_attempt_gate(
            attempt_id,
            GateVerdict(
                gate=str(gate.get("gate") or ""),
                passed=gate.get("passed"),
                reason=str(gate.get("reason") or ""),
                observed=gate.get("observed"),
                threshold=gate.get("threshold"),
            ),
        )
    return True


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
    params = getattr(task, "params", None) or {}
    shared = {
        "task_id": str(getattr(task, "task_id", "") or ""),
        "round_id": str(result_dict.get("round_id") or ""),
        "proposal_ref": str(params.get("proposal_msg_id") or ""),
    }
    recorded = 0
    for row in per_variant:
        if _record_variant_attempt(recorder, row, shared, adopted_variants):
            recorded += 1
    if not (recorded and shared["proposal_ref"]):
        return
    # Settled here rather than at materialization: a grid whose task dies before
    # any variant runs has no attempt to stand behind an ``attempted`` reading,
    # and is more honestly left unsettled.
    recorder.record_proposal_step(shared["proposal_ref"], step=STEP_ATTEMPTED, outcome=str(recorded))
    recorder.settle_proposal(shared["proposal_ref"], disposition=DISPOSITION_ATTEMPTED)


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


def _variant_measurement(metrics: Any) -> _VariantMeasurement:
    if not isinstance(metrics, dict):
        return _VariantMeasurement(gain_pct=None, throughput_after=None, detail={})
    return _VariantMeasurement(
        gain_pct=to_float(metrics.get("gain_pct")),
        throughput_after=to_float(metrics.get("output_throughput")),
        detail={key: metrics[key] for key in _VARIANT_DETAIL_METRICS if metrics.get(key) is not None},
    )


def _variant_failure(variant_outcome: Mapping[str, Any], outcome: str) -> _VariantFailure:
    if outcome != OUTCOME_REVERT:
        return _VariantFailure(error_class=None, reason=None)
    return _VariantFailure(
        error_class=str(variant_outcome.get("error_class") or "") or None,
        reason=str(variant_outcome.get("reason") or "") or None,
    )


def _variant_change_attrs(variant_attrs: Any, variant_name: str) -> dict[str, Any]:
    attrs = dict(variant_attrs) if isinstance(variant_attrs, dict) else {}
    named_knob = attrs.get("extra_server_args") or attrs.get("extra_envs") or attrs.get("name")
    if not named_knob and variant_name:
        attrs["name"] = variant_name
    return attrs


def _finalize_outcome(*, status: str, reason: str, backend: str, result_type: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "backend": backend,
        "result_type": result_type,
    }


def _finalize_error(exc: BaseException, *, reason: str, backend: str, result_type: str) -> dict[str, Any]:
    return {
        "status": "error",
        "reason": reason,
        "backend": backend,
        "result_type": result_type,
        "error_class": type(exc).__name__,
    }


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
        except (OSError, ValueError, TypeError):
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
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume).

        Returns:
            The per-session :class:`Journal` instance (created on first call,
            with the baseline backfilled on subsequent calls).
        """
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

    def _record_fact_impl(self, fact: FactRecord) -> None:
        """Shared KB write for _record_fact_per_task and _record_fact_per_variant.

        Writes one KB lesson (on KEEP with positive gain) or one KB pitfall
        (on REVERT/failure) to the recipe row, then returns.  Call only after
        the journal entry has been appended and ``recipe_kb`` is confirmed
        non-None by the caller.

        Args:
            fact: The completed task, its outcome and the evidence to stamp on
                the provenance.
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
                live = self.read_local_recipe_row()
                recipe_overrides = self.kb_best_config_overrides_for_keep(
                    live=live,
                    best_config_candidate=fact.best_config_candidate,
                    throughput_after=fact.throughput_after,
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
            FactRecord(
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

        An executor KEEP that was not adopted journals as ``no_promote``.
        """
        journal = self.ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw in _NON_ATTEMPT_OUTCOMES:
            return
        outcome = derive_journal_outcome(_variant_verdict(outcome_raw, adopted), variant_outcome)
        variant_name = str(variant_outcome.get("variant_name") or "")
        metrics = variant_outcome.get("metrics") or {}
        measurement = _variant_measurement(metrics)
        variant_attrs = variant_outcome.get("variant") or {}
        attrs = variant_attrs if isinstance(variant_attrs, dict) else None
        kind = classify_change_kind(task.kind, attrs)
        change_attrs = _variant_change_attrs(variant_attrs, variant_name)
        change = summarize_change(task.kind, change_attrs, None)
        failure = _variant_failure(variant_outcome, outcome)
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
                error_class=failure.error_class,
                reason=failure.reason,
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
        from ..knowledge.remote_recipe import HyperloomRemoteKB, KBStoreError

        try:
            canonical_id = self.workload_canonical_id()
            written = HyperloomRemoteKB.from_env().write(canonical_id, state, session_id=session_id)
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
        my_kernels = list(attrs.get("kernel_optimizations") or [])
        # KEEP'd kernel optimizations ride the extras channel.
        extras = dict(attrs.get("workload") or {})
        kernels = my_kernels + rows_not_superseded(live.get("kernel_optimizations"), my_kernels, "kernel_id")
        if kernels:
            extras["kernel_optimizations"] = kernels
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
        return kb_hardware_slug(ss.gpu_type or "unknown_gpu", **resolve_kb_topology())

    def workload_canonical_id(self) -> str:
        """Return the workload's canonical seven-dimension Recipe identity."""
        state = self.shared_state
        identity = _framework_identity(state)
        from hyperloom.common.perf_metric import agentx_active

        return recipe_canonical_id(
            model=state.model_name or "unknown_model",
            hardware=self._kb_hardware_slug(),
            framework_name=identity.name,
            framework_version=identity.version,
            precision=identity.precision,
            model_type=str(state.model_type or ""),
            architectures=state.model_architectures or [],
            scheme=("agentx" if agentx_active(benchmark_mode=state.benchmark_mode) else "inference"),
        )

    def read_local_recipe_row(self) -> dict[str, Any]:
        """Load the selected store's exact authority row for writes."""
        if self.recipe_kb is None:
            return {}
        tick = int(self.shared_state.tick or 0)
        cache = self._local_recipe_cache
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == tick:
            return cache[1]
        row = self.recipe_kb.get_authoritative_recipe(canonical_id=self.workload_canonical_id()) or {}
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
        put_kwargs: dict[str, Any] = {
            "canonical_id": canonical_id,
            "model": state.model_name or "unknown_model",
            "hardware": self._kb_hardware_slug(),
            "framework_name": identity.name,
            "framework_version": identity.version,
            "precision": identity.precision,
            "pitfalls": appended(live.get("pitfalls"), append_pitfall),
            "lessons": appended(live.get("lessons"), append_lesson),
            "extras": amended_recipe_extras(live, overrides, state),
            "provenance": {
                "source": "hyperloom-inference-optimizer",
                "generator": "coordinator",
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "details": dict(provenance_details or {}),
            },
            **amended_recipe_fields(live, overrides),
        }
        self.recipe_kb.put_recipe(**put_kwargs)
        self._local_recipe_cache = None


__all__ = ["RecipeJournalCollaborator"]


#: The probe's own word for a version it could not read.
_UNIDENTIFIED_VERSION: str = "unknown"


class AttemptProvenance(NamedTuple):
    """Where a session's proven optimizations came from, and what it reverted.

    Keepers are indexed twice because a stack layer may identify itself either
    by name or by the research gap it closed.
    """

    kept_by_name: dict[str, str]
    kept_by_gap: dict[str, str]
    reverted: list[dict[str, Any]]


def _known_version(raw: Any) -> str:
    version = str(raw or "").strip()
    return "" if version == _UNIDENTIFIED_VERSION else version


def _runtime_version_tags(fingerprint_meta: Mapping[str, Any], framework: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    framework_key = framework.lower()
    if framework_key in {"sglang", "vllm"}:
        active = _known_version(fingerprint_meta.get(framework_key))
        if active:
            tags["framework_version"] = active
    for source_key, tag in (("rocm", "rocm_version"), ("aiter", "aiter_version"), ("image_digest", "image_digest")):
        version = _known_version(fingerprint_meta.get(source_key))
        if version:
            tags[tag] = version
    return tags


def _baseline_workload_tags(workload_extra: Mapping[str, Any]) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    for key in ("max_running_requests", "max_num_seqs"):
        value = workload_extra.get(key)
        if isinstance(value, int) and value > 0:
            tags[key] = value
    for key in ("chunked_prefill_enabled", "enable_torch_compile"):
        value = workload_extra.get(key)
        if isinstance(value, bool):
            tags[key] = value
    for key in ("quant_scheme", "workload_mode"):
        value = workload_extra.get(key)
        if isinstance(value, str) and value.strip():
            tags[key] = value.strip()
    return tags


def workload_kb_tags(state: Any) -> dict[str, Any]:
    """The workload-shape tags describing the session this recipe came from."""
    tags: dict[str, Any] = {}
    # ``model_family`` is deliberately absent, being no part of the
    # seven-dimension Recipe identity.
    for attr in ("framework", "model_class", "model_name"):
        text = str(getattr(state, attr, "") or "").strip()
        if text:
            tags[attr] = text
    precision = getattr(state, "precision", None)
    if precision not in (None, "", 0):
        tags["precision"] = precision
    tags.update(workload_shape_tags(state))
    fingerprint_meta = state.stack_fingerprint_meta or {}
    if isinstance(fingerprint_meta, dict):
        tags.update(_runtime_version_tags(fingerprint_meta, str(tags.get("framework") or "")))
    workload_extra = state.baseline_workload_extra or {}
    if isinstance(workload_extra, dict):
        tags.update(_baseline_workload_tags(workload_extra))
    return tags


def _index_integrate_attempts(attempts: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_kernel: dict[str, dict[str, Any]] = {}
    by_task_group: dict[str, dict[str, Any]] = {}
    for entry in attempts.values():
        if not isinstance(entry, dict):
            continue
        kernel_id = str(entry.get("kernel_id") or "")
        if kernel_id:
            by_kernel[kernel_id] = entry
        task_group_key = str(entry.get("task_group_key") or "")
        if task_group_key:
            by_task_group[task_group_key] = entry
    return by_kernel, by_task_group


def _last_rebenched_throughput(integration: Mapping[str, Any]) -> float:
    for attempt in reversed(list(integration.get("attempts") or [])):
        if isinstance(attempt, dict) and attempt.get("new_tput") is not None:
            return recorded_float(attempt.get("new_tput"))
    return 0.0


def _kernel_optimization_row(
    entry: Mapping[str, Any],
    ledger_id: str,
    by_kernel: Mapping[str, Any],
    by_task_group: Mapping[str, Any],
) -> dict[str, Any]:
    kernel_id = str(entry.get("current_kernel_id") or entry.get("kernel_id") or ledger_id)
    task_group_key = str(entry.get("task_group_key") or "")
    integration = by_task_group.get(task_group_key) if task_group_key else by_kernel.get(kernel_id)
    integrated = isinstance(integration, dict)
    return {
        "kernel_id": kernel_id,
        "source_file": str(entry.get("last_source_file") or ""),
        "artifact_path": str(entry.get("last_artifact_path") or ""),
        "micro_speedup": recorded_float(entry.get("last_micro_speedup")),
        "decision": OUTCOME_KEEP,
        "e2e_gain_pct": recorded_float(integration.get("best_gain_pct")) if integrated else 0.0,
        "e2e_tput": _last_rebenched_throughput(integration) if integrated else 0.0,
        "e2e_decision": str(integration.get("last_decision") or "").upper() if integrated else "",
        "integrated": integrated,
        "ts": str(entry.get("last_ts") or ""),
    }


def kept_kernel_optimizations(state: Any) -> list[dict[str, Any]]:
    """The KEEP'd kernel optimizations, each joined to its end-to-end integrate verdict."""
    optimizations = state.kernel_opt_task_attempts or {}
    if not isinstance(optimizations, dict):
        return []
    integrations = state.kernel_integrate_attempts or {}
    by_kernel, by_task_group = _index_integrate_attempts(integrations if isinstance(integrations, dict) else {})
    rows: list[dict[str, Any]] = []
    for ledger_id, entry in optimizations.items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("last_decision", "")).upper() != OUTCOME_KEEP:
            continue
        rows.append(_kernel_optimization_row(entry, ledger_id, by_kernel, by_task_group))
    return rows


def _reverted_attempt_row(name: str, attempt: Mapping[str, Any], source: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "reason": "reverted",
        "gain_pct": attempt.get("gain_pct"),
    }
    if source:
        row["source"] = source
    return row


def _gap_attempt_provenance(gap: Mapping[str, Any]) -> AttemptProvenance:
    source = str(gap.get("provenance") or "").strip()
    canonical_id = str(gap.get("canonical_id") or "").strip()
    kept_by_name: dict[str, str] = {}
    kept_by_gap: dict[str, str] = {}
    reverted: list[dict[str, Any]] = []
    for attempt in gap.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        variant = str(attempt.get("variant_name") or "").strip()
        kernel = str(attempt.get("kernel_id") or "").strip()
        outcome = str(attempt.get("outcome") or "").strip().upper()
        if outcome == OUTCOME_KEEP and source:
            for name in (variant, kernel):
                if name:
                    kept_by_name.setdefault(name, source)
            if canonical_id:
                kept_by_gap.setdefault(canonical_id, source)
        elif outcome == OUTCOME_REVERT and (variant or kernel):
            reverted.append(_reverted_attempt_row(variant or kernel, attempt, source))
    return AttemptProvenance(kept_by_name=kept_by_name, kept_by_gap=kept_by_gap, reverted=reverted)


def attempt_provenance(state: Any) -> AttemptProvenance:
    """Trace each proven optimization back to the research hint it came from."""
    kept_by_name: dict[str, str] = {}
    kept_by_gap: dict[str, str] = {}
    reverted: list[dict[str, Any]] = []
    for gap in state.gaps or []:
        if not isinstance(gap, dict):
            continue
        from_gap = _gap_attempt_provenance(gap)
        for name, source in from_gap.kept_by_name.items():
            kept_by_name.setdefault(name, source)
        for canonical_id, source in from_gap.kept_by_gap.items():
            kept_by_gap.setdefault(canonical_id, source)
        reverted.extend(from_gap.reverted)
    return AttemptProvenance(kept_by_name=kept_by_name, kept_by_gap=kept_by_gap, reverted=reverted)


def _best_config(current_best: Mapping[str, Any], optimization_stack: Sequence[Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_args = current_best.get("extra_server_args")
    if current_args:
        config["extra_server_args"] = str(current_args)
    for key in ("extra_envs", "name", "tput", "accuracy"):
        if key in current_best:
            config[key] = current_best[key]
    last = optimization_stack[-1] if optimization_stack else None
    stack_args = (
        str(last.get("candidate_extra_server_args") or last.get("extra_server_args") or "").strip()
        if isinstance(last, dict)
        else ""
    )
    if stack_args:
        config["extra_server_args"] = stack_args
    return config


def _what_worked(
    optimization_stack: Sequence[Any],
    gain_per_entry: Sequence[Any],
    provenance: AttemptProvenance,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(optimization_stack):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("variant_name") or entry.get("name") or entry.get("kernel_id") or "")
        row: dict[str, Any] = {
            "name": name,
            "extra_server_args": str(entry.get("extra_server_args") or ""),
            "extra_envs": dict(entry.get("extra_envs") or {}),
            "gain_pct": gain_per_entry[index] if index < len(gain_per_entry) else None,
        }
        # The layer's gap id wins; the variant or kernel name answers for
        # layers recorded before gap ids were stamped.
        gap_id = str(entry.get("gap_canonical_id") or "").strip()
        by_name = provenance.kept_by_name
        source = (provenance.kept_by_gap.get(gap_id) if gap_id else "") or by_name.get(name)
        source = source or by_name.get(str(entry.get("kernel_id") or "")) or ""
        if source:
            row["source"] = source
        rows.append(row)
    return rows


def _what_failed(recent_failures: Sequence[Any], reverted: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for failure in recent_failures[-10:]:
        if isinstance(failure, dict):
            rows.append(
                {
                    "name": str(failure.get("name") or failure.get("action") or ""),
                    "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                }
            )
    rows.extend(reverted)
    return rows


def _session_row(
    state: Any,
    session_name: str,
    optimization_stack: Collection[Any],
    final_throughput: float,
) -> dict[str, Any]:
    validated_stack_len = int(state.cumulative_gain_validated_stack_len or 0)
    actions = [
        str(entry.get("variant_name") or entry.get("name") or entry.get("action") or "").strip()
        for entry in optimization_stack
        if isinstance(entry, dict)
    ]
    return {
        "session_id": str(state.recipe_kb_session_id or session_name),
        "gain_pct": float(state.cumulative_gain_validated or 0.0),
        "stack_len": validated_stack_len or len(optimization_stack),
        "throughput_before": float(state.baseline_tput or 0.0),
        "throughput_after": final_throughput,
        "date": datetime.now(timezone.utc).isoformat(),
        "actions_taken": [name for name in actions if name],
    }


def recipe_attrs_from_state(state: Any, session_name: str) -> dict[str, Any]:
    """Materialise the recipe-shaped view of one session's settled state.

    ``framework_version`` is left unset here because it is manifest-derived; the
    T0 backfill writes it.
    """
    optimization_stack = state.optimization_stack or []
    # A session that does not sediment its attempts reports no provenance.
    provenance = attempt_provenance(state) if state.recipe_sediment_enabled else AttemptProvenance({}, {}, [])
    final_throughput = float(state.current_best.get("tput", 0.0))
    return {
        "best_config": _best_config(state.current_best, optimization_stack),
        "best_throughput": final_throughput,
        "what_worked": _what_worked(optimization_stack, state.gain_per_stack_entry or [], provenance),
        "what_failed": _what_failed(state.last_action_failures or [], provenance.reverted),
        "kernel_optimizations": kept_kernel_optimizations(state),
        "last_profiled": str(state.cumulative_gain_validated_ts or ""),
        "workload": workload_kb_tags(state),
        "sessions": [_session_row(state, session_name, optimization_stack, final_throughput)],
    }
