# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Writes onto the FRAMEWORK breakdown timeline: specialist runs, candidate lifecycle steps, Critic rulings and measured attempts.

The phase decides; this module records what it decided. Nothing here reads back,
and a phase entry with no open recorder records nothing rather than failing.
"""

from __future__ import annotations

import hashlib
import logging as _logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.env_safety import redact_secret_values

from ._payload import as_entries, as_mapping, as_text, params_of, task_id_of
from ..state.failure_evidence import UNMEASURED_OUTCOMES, classify_failure_attribution

log = _logging.getLogger(__name__)


# These helpers are module-level because the phase's methods get borrowed onto
# lightweight test stand-ins, where a ``self._record_*`` call would break every
# stand-in that does not implement it.
def _recorder(coord: Any):
    """Return the framework recorder for this phase entry, or ``None`` outside an open entry."""
    return coord._framework_timeline_recorder


def _record_run(coord: Any, task: Any, *, role: str, status: str, **fields: Any) -> None:
    """Record one specialist dispatch, or its completion.

    ``role`` is one of ``discovery``, ``authoring`` or ``config`` -- not
    derivable from the arm, since the source arm dispatches twice per
    candidate. Dispatch and terminal status land on one row keyed by the task
    id; a falsy id records nothing rather than keying a row on the empty
    string.
    """
    recorder = _recorder(coord)
    if recorder is None:
        return
    from hyperloom.common.timeutil import now_iso
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        ARM_CONFIG,
        ARM_SOURCE,
        ROLE_CONFIG,
    )

    task_id = task if isinstance(task, str) else task_id_of(task)
    if not task_id:
        return
    fields.setdefault("dispatched_at" if status == "dispatched" else "completed_at", now_iso("seconds"))
    recorder.record_run(
        task_id,
        role=role,
        arm=ARM_CONFIG if role == ROLE_CONFIG else ARM_SOURCE,
        status=status,
        **fields,
    )


class ProposalStep(NamedTuple):
    """One step of a candidate's lifecycle: a ``STEP_*`` value and what it ran and decided."""

    step: str
    run_ref: str = ""
    outcome: str = ""
    reason: str = ""


def _record_step(coord: Any, proposal_id: str, step: ProposalStep) -> None:
    """Record one step of a candidate's lifecycle, keyed by ``STEP_*`` value.

    Steps are recorded as they happen rather than derived from counters: a
    candidate re-authored twice then retried once is three rows a reader can
    follow.
    """
    recorder = _recorder(coord)
    if recorder is None or not proposal_id:
        return
    recorder.record_proposal_step(
        proposal_id,
        step=step.step,
        run_ref=step.run_ref,
        outcome=step.outcome,
        reason=step.reason,
    )


def _record_review_outcome(
    coord: Any,
    proposal_id: str,
    *,
    verdict: str = "",
    reason: str = "",
    **outcome: Any,
) -> None:
    """Record what the phase did with the Critic's ruling on a candidate.

    The consequence is recorded onto the ruling because on its own a routed
    candidate does not say what let it through. ``verdict`` is only passed on
    the deny path, which is the only one carrying the rationale.
    """
    recorder = _recorder(coord)
    if recorder is None or not proposal_id:
        return
    if verdict:
        recorder.record_proposal_review(proposal_id, verdict=verdict, reason=reason)
        recorder.record_proposal_step(proposal_id, step="reviewed", outcome=verdict, reason=reason)
    recorder.record_proposal_review_outcome(proposal_id, **outcome)


def _settle(coord: Any, proposal_id: str, *, disposition: str, reason: str = "") -> None:
    """Record where a candidate ended up: ``attempted``, ``dropped`` or ``pending``."""
    recorder = _recorder(coord)
    if recorder is None or not proposal_id:
        return
    recorder.settle_proposal(proposal_id, disposition=disposition, reason=reason)


def _source_action_reasoning(
    params: Mapping[str, Any],
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve source-attempt reasoning without disguising fallback context as authored rationale."""
    for owner, values, fields in (
        ("action_params", params, ("reasoning", "rationale")),
        ("candidate", candidate, ("reasoning", "rationale", "why")),
        ("context", params, ("gap_symptom",)),
        ("post_action_result", result, ("reasoning",)),
    ):
        for field in fields:
            value = str(values.get(field) or "").strip()
            if value:
                return value, f"{owner}.{field}"
    return "", ""


def _patch_material(session_dir: Path, paths: Iterable[str]) -> list[dict[str, str]]:
    """Each distinct session-local patch as ``{path, sha256, content}``, in the order given.

    A patch outside the session, not UTF-8, or carrying a credential is left
    out, so the session breakdown only ever holds patch text that is safe to
    publish whole.
    """
    root = Path(session_dir).resolve()
    material: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        candidate = Path(raw)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            continue
        if relative in seen:
            continue
        try:
            if not resolved.is_file():
                continue
            payload = resolved.read_bytes()
        except OSError:
            continue
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if redact_secret_values(content) != content:
            continue
        seen.add(relative)
        material.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "content": content})
    return material


def _record_kb_exposure(recorder: Any, proposal_id: str, params: Mapping[str, Any]) -> None:
    """Add the Experience KB read a specialist was shown to the proposal it produced.

    The proposal row is an upsert whose lists merge, so a candidate shaped by
    both a discovery and an authoring specialist keeps both reads' refs.
    """
    read_id = as_text(params.get("kb_read_id"))
    refs = as_entries(params.get("kb_rendered_refs"))
    if read_id or refs:
        recorder.record_proposal(proposal_id, kb_read_id=read_id, rendered_refs=refs)


class PatchTrail(NamedTuple):
    """The patch files one attempt touched: the one it is named for, and everything it applied or rolled back."""

    lead: str
    applied: list[str]
    reverted: list[str]


def _patch_trail(result: Mapping[str, Any]) -> PatchTrail:
    """The patches an attempt touched, falling back to the first file it moved when no patch is named."""
    applied = [str(path) for path in as_entries(result.get("patches_applied")) if str(path)]
    reverted = [str(path) for path in as_entries(result.get("patches_reverted")) if str(path)]
    lead = as_text(result.get("source_realized_patch"), result.get("patch_path"))
    if not lead:
        lead = next(iter(applied or reverted), "")
    return PatchTrail(lead=lead, applied=applied, reverted=reverted)


def _attempt_measurement(result: Mapping[str, Any], base: Any) -> dict[str, Any]:
    """What the attempt measured, against the anchor it was graded on."""
    return {
        "before_tput": base,
        "after_tput": result.get("output_throughput"),
        "gain_pct": result.get("delta_pct"),
        "runtime_sec": result.get("runtime_sec"),
    }


def _attempt_accuracy(result: Mapping[str, Any]) -> dict[str, Any]:
    """What the accuracy gate observed; ``required`` is ``None`` for a gate that never ran, not one that failed."""
    accuracy_pass = result.get("accuracy_pass")
    return {
        "required": None if accuracy_pass is None else True,
        "value": result.get("accuracy_value"),
        "reference": result.get("accuracy_reference"),
        "passed": accuracy_pass,
    }


def _attempt_failure(result: Mapping[str, Any], status: str) -> dict[str, str]:
    """How an attempt failed, and whom the failure is attributed to once it never reached a measurement."""
    error_class = as_text(result.get("error_class"))
    error_excerpt = as_text(result.get("error"))[:600]
    attribution = ""
    if status.upper() in UNMEASURED_OUTCOMES:
        attribution = classify_failure_attribution(
            error_class=error_class,
            error_excerpt=error_excerpt,
            reason=result.get("reason"),
            explicit=result.get("failure_attribution"),
        )
    return {"error_class": error_class, "error_excerpt": error_excerpt, "attribution": attribution}


def _adoption_basis(accuracy_pass: Any) -> str:
    """What stood behind an adoption, on the same rule the config arm writes it under.

    A KEEP no accuracy gate ruled on rests on throughput alone, a weaker claim
    that must not read alike. Only an adoption carries a basis -- on a reverted
    row ``accuracy_pass`` would name the gate that refused it.
    """
    return "accuracy_pass" if accuracy_pass is not None else "keep_verdict_unscored"


def _record_attempt_gates(recorder: Any, task_id: str, result: Mapping[str, Any], params: Mapping[str, Any]) -> None:
    """Record every gate that ruled on one attempt: the keep threshold, the accuracy reference and switch-off parity."""
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import GateVerdict

    delta_pct = result.get("delta_pct")
    keep_threshold = result.get("keep_threshold_pct")
    if keep_threshold is None:
        keep_threshold = params.get("keep_threshold_pct")
    if isinstance(delta_pct, (int, float)) and isinstance(keep_threshold, (int, float)):
        recorder.record_attempt_gate(
            task_id,
            GateVerdict(
                gate="keep_threshold",
                passed=float(delta_pct) >= float(keep_threshold),
                observed=delta_pct,
                threshold=keep_threshold,
            ),
        )
    accuracy_pass = result.get("accuracy_pass")
    if accuracy_pass is not None:
        recorder.record_attempt_gate(
            task_id,
            GateVerdict(
                gate="accuracy",
                passed=bool(accuracy_pass),
                observed=result.get("accuracy_value"),
                threshold=result.get("accuracy_reference"),
            ),
        )
    parity = as_mapping(result.get("switch_off_parity"))
    if parity.get("ran"):
        recorder.record_attempt_gate(
            task_id,
            GateVerdict(gate="switch_off_parity", passed=bool(parity.get("ok")), reason=as_text(parity.get("reason"))),
        )


class AuthoredAttempt(NamedTuple):
    """One authored deliverable's measured attempt: who ran it and how it was decided."""

    task: Any
    candidate_id: str
    status: str
    adopted: bool
    specialist_task_id: str = ""


def _record_source_attempt(
    coord: Any,
    attempt: AuthoredAttempt,
    *,
    result: Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    """Record one authored deliverable's measured attempt -- a patch, or server args and envs -- on the framework event.

    A pure timeline recorder: the control-plane ledger write sits beside the call
    to this function, not inside it, so a phase with no open recorder still
    records the attempt.
    """
    recorder = _recorder(coord)
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ARM_SOURCE

    task_id = task_id_of(attempt.task)
    if not task_id:
        return
    base = result.get("base_tput") if result.get("base_tput") is not None else params.get("base_tput")
    accuracy_pass = result.get("accuracy_pass")
    # Only the executor knows the stack the patch was measured on, since it
    # rebinds onto the live stack top before running; the task's params are the
    # stack as of dispatch. Absent on a row that never reached a measurement.
    stack = as_mapping(result.get("measured_against"))
    measured_against = {"measured_against": stack} if stack else {}
    # The levers the deliverable layered onto that stack: the whole change when the specialist returned no patch.
    levers = {"extra_server_args": params.get("extra_server_args"), "extra_envs": params.get("extra_envs")}
    config_delta = {"config_delta": levers} if any(levers.values()) else {}
    reasoning, reasoning_origin = _source_action_reasoning(params, as_mapping(params.get("candidate")), result)
    trail = _patch_trail(result)
    recorder.record_attempt(
        task_id,
        arm=ARM_SOURCE,
        task_id=task_id,
        proposal_ref=attempt.candidate_id,
        candidate_id=attempt.candidate_id,
        provenance=as_text(params.get("lever_kind")),
        outcome=attempt.status,
        reason=as_text(result.get("reason")),
        reasoning=reasoning,
        reasoning_origin=reasoning_origin,
        experience_citations=as_entries(params.get("experience_citations")),
        stage=as_text(result.get("stage")),
        route=as_text(params.get("audit_step")),
        patch_source=attempt.specialist_task_id,
        patch_path=trail.lead,
        fingerprint=as_text(result.get("patch_sha256"), result.get("fingerprint")),
        # An attempt can apply several patches, and which ones landed is
        # not recoverable from the single primary path.
        patches_applied=trail.applied,
        patches_reverted=trail.reverted,
        patch_material=_patch_material(Path(coord._coord.session_dir), [trail.lead, *trail.applied, *trail.reverted]),
        target_files=as_entries(result.get("target_files")),
        source_ref=as_text(params.get("framework_agent_candidate_id")) or attempt.candidate_id,
        measurement=_attempt_measurement(result, base),
        accuracy=_attempt_accuracy(result),
        failure=_attempt_failure(result, attempt.status),
        artifacts={
            "workspace": as_text(result.get("workspace")),
            "server_log_path": as_text(result.get("server_log_path")),
        },
        decision=attempt.status,
        adopted=attempt.adopted,
        validation_basis=_adoption_basis(accuracy_pass) if attempt.adopted else "",
        attribution_eligible=(attempt.adopted and base is not None and result.get("output_throughput") is not None),
        **measured_against,
        **config_delta,
    )
    _record_kb_exposure(recorder, attempt.candidate_id, params)
    _record_attempt_gates(recorder, task_id, result, params)
    _record_step(
        coord,
        attempt.candidate_id,
        ProposalStep(step="attempted", run_ref=attempt.specialist_task_id, outcome=attempt.status),
    )


def _record_discovered(coord: Any, task: Any, *, raw: Any, candidates: list[dict[str, Any]]) -> None:
    """Record what one discovery round produced, including what it dropped.

    Walks ``raw`` -- the specialist's ``proposal_set`` before auditing --
    rather than only the surviving ``candidates``: a round that found five
    upstream PRs and judged all five already landed is a very different result
    from one that found nothing.
    """
    recorder = _recorder(coord)
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        ARM_SOURCE,
        DISPOSITION_DROPPED,
        PRODUCER_SPECIALIST,
        STEP_PROPOSED,
    )

    run_id = task_id_of(task)
    discovery_params = params_of(task)
    domain = as_text(discovery_params.get("domain"))
    kept = {coord._framework_candidate_key(cand): cand for cand in candidates}
    for cand_id, cand in kept.items():
        if not cand_id:
            continue
        _record_kb_exposure(recorder, cand_id, discovery_params)
        recorder.record_proposal(
            cand_id,
            arm=ARM_SOURCE,
            producer=PRODUCER_SPECIALIST,
            producer_ref=domain,
            run_ref=run_id,
            source_ref=as_text(cand.get("pr_url"), cand.get("head_sha")),
            repo=as_text(cand.get("repo")),
            title=as_text(cand.get("title")),
            reasoning=as_text(cand.get("reasoning"), cand.get("rationale"), cand.get("why")),
            changed_files=as_entries(cand.get("changed_files")),
            gap_canonical_id=as_text(cand.get("gap_canonical_id")),
            route=as_text(cand.get("route")),
            verdict=as_text(as_mapping(cand.get("audit")).get("verdict")),
        )
        recorder.record_proposal_step(cand_id, step=STEP_PROPOSED, run_ref=run_id)
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        verdict = as_text(entry.get("verdict")).strip().lower()
        if verdict not in {"already_present", "not_applicable"}:
            continue
        ref = as_text(entry.get("pr_url"), entry.get("url"), entry.get("head_sha")).strip()
        if not ref or ref in kept:
            continue
        recorder.record_proposal(
            ref,
            arm=ARM_SOURCE,
            producer=PRODUCER_SPECIALIST,
            producer_ref=domain,
            run_ref=run_id,
            source_ref=ref,
            repo=as_text(entry.get("repo")),
            title=as_text(entry.get("title")),
            verdict=verdict,
        )
        recorder.settle_proposal(ref, disposition=DISPOSITION_DROPPED, reason=verdict)
