# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT deliverable readers, timeline record helpers, and the authored-outcome progress bridge."""

from __future__ import annotations

import hashlib
import logging as _logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from hyperloom.common.coerce import first_str, first_text, to_list, to_mapping
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json
from hyperloom.orchestrator.lever import LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR, patch_lever_kind
from ..actions.executors import _integrate_vocabulary
from ..state.attempt_ledger import AttemptRow, record_attempt_row
from ..state.failure_evidence import UNMEASURED_OUTCOMES, classify_failure_attribution
from ..state.task_registry import TaskNotFound

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)


def _config_lever_from_proposal(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The server args and env overrides one lever proposal carries, dropping args Magpie cannot transport."""
    extra_envs = {str(k).strip(): str(v) for k, v in to_mapping(entry.get("extra_envs")).items() if str(k).strip()}
    args = entry.get("extra_args")
    tokens = [str(a) for a in args if str(a).strip()] if isinstance(args, (list, tuple)) else []
    if any(ch.isspace() for token in tokens for ch in token):
        text, fault = "", "has a whitespace-bearing argv token"
    else:
        text, fault = (args if isinstance(args, str) else " ".join(tokens)), ""
    parsed_args = tokenize_server_args_preserving_json(text)
    if parsed_args is None:
        fault = (
            "has server args unsupported by Magpie's unquoted argv transport"
            if isinstance(args, str)
            else "has unparsable server args"
        )
    if fault:
        log.warning(
            "FRAMEWORK config lever %r %s; dropping the args%s",
            entry.get("name"),
            fault,
            " while preserving its environment overrides" if extra_envs else "",
        )
        if not extra_envs:
            return {}
    extra_server_args = "" if parsed_args is None else parsed_args[0]
    if not extra_server_args and not extra_envs:
        return {}
    return {"extra_server_args": extra_server_args, "extra_envs": extra_envs}


def _framework_config_levers_from_done(
    done_payload: dict[str, Any] | None,
    *,
    levers_ride_with_patches: bool = False,
) -> dict[str, Any]:
    """Extract a config-lever set from a FRAMEWORK specialist deliverable.

    Args:
        done_payload: The specialist's ``specialist_done`` payload.
        levers_ride_with_patches: Whether a lever delivered alongside a patch
            belongs to the patch's round. True for ENABLEMENT, where the pair is
            jointly what makes the model boot; False while optimizing, where a
            patch is its own outcome and a lever is judged on its own.
    """
    if not isinstance(done_payload, dict):
        return {}
    proposals = done_payload.get("proposal_set") or []
    if not isinstance(proposals, list):
        return {}
    # A patch deliverable otherwise takes precedence: a lever that merely
    # *accompanies* a patch is not a config-only outcome. ``atomic`` remains the
    # specialist's own way to say the two are inseparable, but it cannot be the
    # only way -- it is a model-authored boolean, and the same specialist has
    # emitted ``atomic: false`` on a lever whose own reason read "required to
    # boot at all once the patch lands". Enablement therefore decides this from
    # the lane it is running, not from the deliverable's self-description.
    patches = done_payload.get("patches_written") or []
    if isinstance(patches, list) and patches and not levers_ride_with_patches:
        proposals = [e for e in proposals if isinstance(e, dict) and e.get("atomic") is True]
        if not proposals:
            return {}
    for entry in proposals:
        lever = _config_lever_from_proposal(entry) if isinstance(entry, dict) else {}
        if lever:
            return lever
    return {}


def _patches_on_disk(patch: Path, resolve_bases: list[Path]) -> bool:
    """True when a specialist's claimed patch path resolves to a real file under one of its worktree roots."""
    return (patch.is_absolute() and patch.is_file()) or any((base / patch).is_file() for base in resolve_bases)


def _resolvable_artifacts_from_done(
    done_payload: dict[str, Any] | None,
    resolve_bases: list[Path],
) -> list[dict[str, Any]]:
    """Return ``artifacts_written`` entries whose ``source`` file exists on disk."""
    if not isinstance(done_payload, dict):
        return []
    arts = done_payload.get("artifacts_written")
    if not isinstance(arts, list):
        return []
    out: list[dict[str, Any]] = []
    # Sandbox bases are invariant across entries — resolve once.
    bases_resolved = [base.resolve() for base in resolve_bases]
    for entry in arts:
        if not isinstance(entry, dict):
            continue
        src = str(entry.get("source") or "").strip()
        tgt = str(entry.get("target") or "").strip()
        if not src or not tgt:
            continue
        raw = Path(src)
        # An absolute ``source`` is checked as-is; a relative one is resolved under each base.
        cands = [raw] if raw.is_absolute() else [base / raw for base in resolve_bases]
        for cand in cands:
            resolved = cand.resolve()
            if not resolved.is_file():
                continue
            contained = False
            for base in bases_resolved:
                try:
                    resolved.relative_to(base)
                except ValueError:
                    continue
                contained = True
                break
            if contained:
                out.append(entry)
                break
    return out


# These helpers are module-level because the phase's methods get borrowed onto
# lightweight test stand-ins, where a ``self._record_*`` call would break every
# stand-in that does not implement it.
def _recorder(coord: Any):
    """Return the framework recorder for this phase entry, or ``None`` outside an open entry."""
    return coord._framework_timeline_recorder


def _record_run(coord: Any, task: "Task | str", *, role: str, status: str, **fields: Any) -> None:
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

    task_id = task if isinstance(task, str) else task.task_id
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


class ProgressStamp(NamedTuple):
    """The terminal ``framework_agent_phase_progress`` row a candidate ends with."""

    candidate_id: str
    status: str
    batch_id: str = ""
    kept: bool = False
    rationale: str = ""
    provenance: str = ""
    gain_pct: float | None = None
    extra: dict[str, Any] | None = None


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
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import CriticReview

    if verdict:
        recorder.record_proposal_review(proposal_id, CriticReview(verdict=verdict, reason=reason))
        recorder.record_proposal_step(proposal_id, step="reviewed", outcome=verdict, reason=reason)
    recorder.record_proposal_review_outcome(proposal_id, **outcome)


def _settle(coord: Any, proposal_id: str, *, disposition: str, reason: str = "") -> None:
    """Record where a candidate ended up: ``attempted`` or ``dropped``."""
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
    read_id = str(params.get("kb_read_id") or "")
    refs = params.get("kb_rendered_refs") or []
    if read_id or refs:
        recorder.record_proposal(proposal_id, kb_read_id=read_id, rendered_refs=refs)


class AuthoredAttempt(NamedTuple):
    """One authored deliverable's measured attempt: who ran it and how it was decided."""

    task: Any
    candidate_id: str
    status: str
    adopted: bool
    specialist_task_id: str = ""


def _attempt_outcome_fields(
    attempt: AuthoredAttempt, *, result: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    """The measurement, accuracy, failure and artifact blocks an attempt's record carries, and what its adoption rests on."""
    base = result.get("base_tput") if result.get("base_tput") is not None else params.get("base_tput")
    accuracy_pass = result.get("accuracy_pass")
    attribution = ""
    if attempt.status.upper() in UNMEASURED_OUTCOMES:
        attribution = classify_failure_attribution(
            error_class=first_str(result.get("error_class")),
            error_excerpt=first_str(result.get("error"))[:600],
            reason=result.get("reason"),
            explicit=result.get("failure_attribution"),
        )
    return {
        "measurement": {
            "before_tput": base,
            "after_tput": result.get("output_throughput"),
            "gain_pct": result.get("delta_pct"),
            "runtime_sec": result.get("runtime_sec"),
        },
        # A ``required`` of ``None`` is an accuracy gate that did not run, not one that ran and failed.
        "accuracy": {
            "required": None if accuracy_pass is None else True,
            "value": result.get("accuracy_value"),
            "reference": result.get("accuracy_reference"),
            "passed": accuracy_pass,
        },
        "failure": {
            "error_class": first_str(result.get("error_class")),
            "error_excerpt": first_str(result.get("error"))[:600],
            "attribution": attribution,
        },
        "artifacts": {
            "workspace": first_str(result.get("workspace")),
            "server_log_path": first_str(result.get("server_log_path")),
        },
        # On the same rule the config arm writes it under: a KEEP no accuracy gate ruled on rests on throughput
        # alone, a weaker claim that must not read alike. Only an adoption carries it -- on a reverted row
        # "accuracy_pass" would name the gate that refused it.
        "validation_basis": (
            ("accuracy_pass" if accuracy_pass is not None else "keep_verdict_unscored") if attempt.adopted else ""
        ),
        "attribution_eligible": (attempt.adopted and base is not None and result.get("output_throughput") is not None),
    }


def _record_attempt_gates(recorder: Any, task_id: str, *, result: Mapping[str, Any], params: Mapping[str, Any]) -> None:
    """Record the gate verdicts an attempt's measurement settled: keep threshold, accuracy, switch-off parity."""
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import GateVerdict

    delta_pct = result.get("delta_pct")
    keep_threshold = result.get("keep_threshold_pct")
    if keep_threshold is None:
        keep_threshold = params.get("keep_threshold_pct")
    accuracy_pass = result.get("accuracy_pass")
    parity = to_mapping(result.get("switch_off_parity"))
    verdicts: list[Any] = []
    if isinstance(delta_pct, (int, float)) and isinstance(keep_threshold, (int, float)):
        verdicts.append(
            GateVerdict(
                gate="keep_threshold",
                passed=float(delta_pct) >= float(keep_threshold),
                observed=delta_pct,
                threshold=keep_threshold,
            )
        )
    if accuracy_pass is not None:
        verdicts.append(
            GateVerdict(
                gate="accuracy",
                passed=bool(accuracy_pass),
                observed=result.get("accuracy_value"),
                threshold=result.get("accuracy_reference"),
            )
        )
    if parity.get("ran"):
        verdicts.append(
            GateVerdict(gate="switch_off_parity", passed=bool(parity.get("ok")), reason=first_str(parity.get("reason")))
        )
    for verdict in verdicts:
        recorder.record_attempt_gate(task_id, verdict)


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

    task_id = first_str(getattr(attempt.task, "task_id", ""))
    if not task_id:
        return
    # Only the executor knows the stack the patch was measured on, since it rebinds onto the live stack top before
    # running; the task's params are the stack as of dispatch. Absent on a row that never reached a measurement.
    stack = result.get("measured_against")
    measured_against = {"measured_against": stack} if isinstance(stack, Mapping) and stack else {}
    # The levers the deliverable layered onto that stack: the whole change when the specialist returned no patch.
    levers = {"extra_server_args": params.get("extra_server_args"), "extra_envs": params.get("extra_envs")}
    config_delta = {"config_delta": levers} if any(levers.values()) else {}
    reasoning, reasoning_origin = _source_action_reasoning(params, to_mapping(params.get("candidate")), result)
    patches_applied = [str(path) for path in to_list(result.get("patches_applied")) if str(path)]
    patches_reverted = [str(path) for path in to_list(result.get("patches_reverted")) if str(path)]
    patch_path = first_str(result.get("source_realized_patch"), result.get("patch_path")) or next(
        iter(patches_applied or patches_reverted), ""
    )
    recorder.record_attempt(
        task_id,
        arm=ARM_SOURCE,
        task_id=task_id,
        proposal_ref=attempt.candidate_id,
        candidate_id=attempt.candidate_id,
        provenance=first_str(params.get("lever_kind")),
        outcome=attempt.status,
        reason=first_str(result.get("reason")),
        reasoning=reasoning,
        reasoning_origin=reasoning_origin,
        experience_citations=params.get("experience_citations") or [],
        stage=first_str(result.get("stage")),
        route=first_str(params.get("audit_step")),
        patch_source=attempt.specialist_task_id,
        patch_path=patch_path,
        fingerprint=first_str(result.get("patch_sha256"), result.get("fingerprint")),
        # Which patches landed is not recoverable from the single primary path.
        patches_applied=patches_applied,
        patches_reverted=patches_reverted,
        patch_material=_patch_material(
            Path(coord._coord.session_dir), [patch_path, *patches_applied, *patches_reverted]
        ),
        target_files=result.get("target_files") or [],
        source_ref=first_str(params.get("framework_agent_candidate_id"), attempt.candidate_id),
        decision=attempt.status,
        adopted=attempt.adopted,
        benched=bool(result.get(_integrate_vocabulary.RESULT_BENCHED)),
        **_attempt_outcome_fields(attempt, result=result, params=params),
        **measured_against,
        **config_delta,
    )
    _record_kb_exposure(recorder, attempt.candidate_id, params)
    _record_attempt_gates(recorder, task_id, result=result, params=params)
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

    run_id = str(getattr(task, "task_id", "") or "")
    discovery_params = getattr(task, "params", None) or {}
    domain = str(discovery_params.get("domain") or "")
    # Candidates are the rows _candidates_from_discovery_proposals built: keyed, with every field normalized.
    kept = {coord._framework_candidate_key(cand): cand for cand in candidates}
    for cand_id, cand in kept.items():
        _record_kb_exposure(recorder, cand_id, discovery_params)
        recorder.record_proposal(
            cand_id,
            arm=ARM_SOURCE,
            producer=PRODUCER_SPECIALIST,
            producer_ref=domain,
            run_ref=run_id,
            source_ref=cand["pr_url"] or cand["head_sha"],
            repo=cand["repo"],
            title=cand["title"],
            reasoning=cand["reasoning"],
            changed_files=cand["changed_files"],
            gap_canonical_id=cand["gap_canonical_id"],
            route=cand["route"],
            verdict=cand["audit"]["verdict"],
        )
        recorder.record_proposal_step(cand_id, step=STEP_PROPOSED, run_ref=run_id)
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        verdict = first_text(entry.get("verdict")).lower()
        if verdict not in {"already_present", "not_applicable"}:
            continue
        ref = first_text(entry.get("pr_url"), entry.get("url"), entry.get("head_sha"))
        if not ref or ref in kept:
            continue
        recorder.record_proposal(
            ref,
            arm=ARM_SOURCE,
            producer=PRODUCER_SPECIALIST,
            producer_ref=domain,
            run_ref=run_id,
            source_ref=ref,
            repo=str(entry.get("repo") or ""),
            title=str(entry.get("title") or ""),
            verdict=verdict,
        )
        recorder.settle_proposal(ref, disposition=DISPOSITION_DROPPED, reason=verdict)


class _AuthoredOutcomeMixin:
    """Bridges authored-deliverable outcomes into the FRAMEWORK progress ledger, the attempt ledger and the timeline."""

    def _record_framework_agent_authored_outcome(
        self,
        *,
        task: "Task",
        result: Any,
        adopted: bool,
    ) -> None:
        """Bridge an authored-patch ``integrate_patch`` outcome into the FRAMEWORK progress ledger.

        Attributed to the latest batch; every terminal status is recorded (empty/in-progress
        statuses and lane-owned ``apply_failed`` retries are skipped). The row is ``kept``
        only when the patch was adopted.
        """
        params = to_mapping(getattr(task, "params", None))
        if not params.get("framework_agent_authoring"):
            return
        res = result if isinstance(result, dict) else getattr(result, "result", None)
        if not isinstance(res, dict):
            return
        # Every terminal integrate_patch outcome is recorded, not just keep/revert; an apply_failed carrying a lane
        # is the unified retry loop's to settle (re-dispatch, or a terminal row at the cap).
        status = first_str(res.get("status"))
        if not status or (status == "apply_failed" and res.get("lane") in ("perf_framework", "perf_explore")):
            return
        # The FRAMEWORK candidate id an authored patch belongs to is a PR URL.
        spec_tid = first_str(params.get("specialist_task_id"))
        cand_map = to_mapping(self.shared_state.framework_agent_specialist_candidate_map)
        cand_id = first_str(
            params.get("framework_agent_candidate_id"), cand_map.get(spec_tid), spec_tid, getattr(task, "task_id", "")
        )
        batches = to_list(self.shared_state.framework_agent_batches)
        batch_id = first_str(
            params.get("framework_batch_id"), to_mapping(batches[-1] if batches else None).get("batch_id")
        )
        delta_pct = res.get("delta_pct")
        new_tput = res.get("output_throughput")
        gain = float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
        if not self._clear_superseded_progress(cand_id):
            return
        recorded = self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status=status,
                kept=adopted,
                rationale=first_str(res.get("reason")),
                provenance="authored",
                gain_pct=gain,
                extra={
                    # The anchor the executor graded against, so post/pre and gain_pct in the same row agree once a stack
                    # has formed.
                    "pre_tput": float(res.get("base_tput") or params.get("base_tput") or 0.0),
                    "post_tput": float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
                    "accuracy_pass": res.get("accuracy_pass"),
                    "specialist_task_id": spec_tid,
                    "integrate_task_id": first_str(getattr(task, "task_id", "")),
                    "reauthor_attempt": res.get("reauthor_attempt", params.get("reauthor_attempt")),
                },
            )
        )
        if not recorded:
            return
        attempt = AuthoredAttempt(
            task=task, candidate_id=cand_id, status=status, adopted=adopted, specialist_task_id=spec_tid
        )
        self._record_authored_attempt(attempt, result=res, params=params)
        log.info(
            "FRAMEWORK: authored patch outcome candidate=%s batch=%s status=%s gain=%.2f%%",
            cand_id,
            batch_id,
            status,
            gain,
        )

    def _clear_superseded_progress(self, cand_id: str) -> bool:
        """Drop the progress rows earlier attempts on ``cand_id`` left; False when one of them was adopted.

        An adopted patch is the last word on a candidate; any other row is an outcome a later attempt may better.
        """
        progress = to_list(self.shared_state.framework_agent_phase_progress)
        matching = [row for row in progress if isinstance(row, dict) and self._framework_candidate_key(row) == cand_id]
        if any(row.get("kept") for row in matching):
            return False
        if matching:
            self.shared_state.framework_agent_phase_progress = [row for row in progress if row not in matching]
        return True

    def _record_authored_attempt(
        self, attempt: AuthoredAttempt, *, result: Mapping[str, Any], params: Mapping[str, Any]
    ) -> None:
        """Record one authored deliverable's measured attempt in the attempt ledger and on the framework timeline."""
        evidence = {**result, **params}  # A lever the dispatch named outranks the one the deliverable names.
        record_attempt_row(
            self.shared_state,
            AttemptRow(
                lever_kind=patch_lever_kind(evidence),
                task_id=first_str(getattr(attempt.task, "task_id", "")),
                candidate_id=first_str(evidence.get("framework_agent_candidate_id")),
                specialist_task_id=attempt.specialist_task_id,
                outcome=attempt.status,
                adopted=attempt.adopted,
                gain_pct=result.get("delta_pct"),
                before_tput=result.get("base_tput") if result.get("base_tput") is not None else params.get("base_tput"),
                after_tput=result.get("output_throughput"),
                error_class=first_str(result.get("error_class")),
            ),
        )
        _record_source_attempt(self, attempt, result=result, params=params)

    async def _recover_framework_agent_authoring_outcome(
        self,
        *,
        specialist_task: "Task",
    ) -> bool:
        """Recover a missed authoring outcome from persisted delegated results."""
        specialist_task_id = first_str(getattr(specialist_task, "task_id", ""))
        cand_id = first_str(to_mapping(getattr(specialist_task, "params", None)).get("framework_agent_candidate_id"))
        if not specialist_task_id or not cand_id:
            return False
        messages = await self.bus.tail(n=10000, topic="delegated_result")
        if self._replay_specialist_deliverable(specialist_task, specialist_task_id, messages):
            if cand_id in self._framework_processed_candidate_keys():
                return True
        return await self._replay_authored_integrates(specialist_task_id, cand_id, messages)

    def _replay_specialist_deliverable(self, specialist_task: "Task", specialist_task_id: str, messages: Any) -> bool:
        """Re-record the deliverable the specialist's own bus entry carries; False when the bus holds none."""
        for message in messages:
            payload = to_mapping(getattr(message, "payload", None))
            if (
                first_str(payload.get("task_id")) != specialist_task_id
                or first_str(payload.get("kind")) != "specialist"
            ):
                continue
            done_payload = to_mapping(payload.get("result")).get("specialist_done")
            # A run that failed before delivering carries its error on the bus entry rather than in the result, and
            # must reach the recorder for the same reason it does on the live path.
            run_error = first_str(payload.get("error"))
            if not isinstance(done_payload, dict) and not run_error:
                return False
            self._record_framework_agent_authoring_empty_outcome(
                task=specialist_task, done_payload=to_mapping(done_payload), run_error=run_error
            )
            return True
        return False

    async def _replay_authored_integrates(self, specialist_task_id: str, cand_id: str, messages: Any) -> bool:
        """Re-record every integrate_patch outcome the specialist's patches reached; True once ``cand_id`` is settled."""
        for message in messages:
            payload = to_mapping(getattr(message, "payload", None))
            task_id = first_str(payload.get("task_id"))
            if first_str(payload.get("kind")) != "integrate_patch" or not task_id:
                continue
            try:
                integrate_task = await self.tasks.get(task_id)
            except TaskNotFound:
                continue
            integrate_params = to_mapping(integrate_task.params)
            if first_str(integrate_params.get("specialist_task_id")) != specialist_task_id or not integrate_params.get(
                "framework_agent_authoring"
            ):
                continue
            # The bus carries the executor's result, not the settlement verdict; the stack entry the
            # lift wrote for this task is the durable record of the adoption.
            self._record_framework_agent_authored_outcome(
                task=integrate_task,
                result=to_mapping(payload.get("result")),
                adopted=any(
                    first_str(to_mapping(entry).get("task_id")) == task_id
                    for entry in to_list(self.shared_state.optimization_stack)
                ),
            )
            if cand_id in self._framework_processed_candidate_keys():
                return True
        return False

    def _record_framework_agent_dispatch_failure(self, *, task: "Task", run_error: str) -> None:
        """Stamp a terminal row for a candidate whose specialist never ran."""
        params = getattr(task, "params", None) or {}
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not cand_id:
            return
        status = "dispatch_failed"
        recorded = self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=str(params.get("framework_batch_id") or ""),
                status=status,
                rationale=run_error[:500],
                provenance=status,
                gain_pct=0.0,
                extra={
                    "specialist_task_id": str(getattr(task, "task_id", "") or ""),
                    "reauthor_attempt": params.get("reauthor_attempt"),
                },
            )
        )
        if not recorded:
            return
        self._record_undelivered_authoring_attempt(task=task, candidate_id=cand_id, status=status)
        log.warning(
            "FRAMEWORK: authoring specialist never delivered candidate=%s: %s",
            cand_id,
            run_error[:300],
        )

    def _record_framework_agent_authoring_empty_outcome(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any] | None,
        run_error: str = "",
    ) -> None:
        """Record a terminal FRAMEWORK row when an authoring specialist finishes WITHOUT a patch."""
        params = getattr(task, "params", None) or {}
        if not bool(params.get("framework_agent_authoring")):
            return
        payload = done_payload if isinstance(done_payload, dict) else {}
        # No payload at all plus an error means the run never delivered: there is no deliverable to judge, so this is
        # infrastructure, not a search result.
        if run_error and not payload:
            self._record_framework_agent_dispatch_failure(task=task, run_error=run_error)
            return
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        # A downstream integrate_patch (owned by the authored-outcome bridge that writes the terminal row) is created
        # by ``maybe_autosubmit_specialist_patches`` when the deliverable is routable: ``patches_written`` (post
        # safety-vetting) non-empty, OR a config-lever deliverable, OR a non-diff tuned artifact.
        patches = inner.get("patches_written") or []
        if isinstance(patches, list) and patches:
            return
        # Relaxed FRAMEWORK rule: a config-lever deliverable (proposal_set carrying extra_args / extra_envs) is a FULL
        # result, not "empty".
        if _framework_config_levers_from_done(inner):
            return
        # Relaxed FRAMEWORK rule (parity with autosubmit): a non-diff tuned artifact deliverable
        # (``artifacts_written`` with a real source file) is a FULL result — autosubmit routes it to integrate_patch,
        # which owns the terminal row.
        task_id = str(getattr(task, "task_id", "") or "")
        try:
            from hyperloom.inference_optimizer.session.session_paths import (
                runs_dir as _runs_dir,
            )
            from pathlib import Path as _Path

            if self.session_dir is not None and task_id:
                _spec_root = _runs_dir(_Path(self.session_dir), "specialist", task_id)
                if _resolvable_artifacts_from_done(inner, [_spec_root / "worktree", _spec_root]):
                    return
        except (OSError, ValueError, RuntimeError):
            log.debug("FRAMEWORK: artifacts routable-check failed", exc_info=True)
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not cand_id:
            return
        batch_id = str(params.get("framework_batch_id") or "")
        reason = first_text(inner.get("summary"))[:500]
        recorded = self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="author_empty",
                rationale=reason,
                provenance="authored_empty",
                gain_pct=0.0,
                extra={
                    "specialist_task_id": task_id,
                    "reauthor_attempt": params.get("reauthor_attempt"),
                },
            )
        )
        if not recorded:
            return
        self._record_undelivered_authoring_attempt(task=task, candidate_id=cand_id, status="author_empty")
        log.info("FRAMEWORK: authoring specialist empty deliverable candidate=%s batch=%s", cand_id, batch_id)

    def _record_undelivered_authoring_attempt(self, *, task: "Task", candidate_id: str, status: str) -> None:
        """Charge a newly stamped round that delivered nothing to its PR lead's lever, else to the source lever."""
        lever = LEVER_UPSTREAM_PR if patch_lever_kind(task.params) == LEVER_UPSTREAM_PR else LEVER_SOURCE_PATCH
        record_attempt_row(self.shared_state, AttemptRow.undelivered_round(lever, status, task.task_id, candidate_id))
