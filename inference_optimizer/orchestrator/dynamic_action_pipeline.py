"""End-to-end pipeline glue for ``dynamic_action`` (pure helpers).

Every dynamic-only piece of logic lives here so the Coordinator
wiring stays thin and the specialist / integrate_patch / grid main
chain is reused unchanged. All functions are side-effect-isolated
(filesystem only) and import-cycle-free.

Public surface:

* :func:`runner_status_to_lifecycle` — runner terminal state → lifecycle.
* :func:`integrate_status_to_lifecycle` — ``IntegratePatchExecutor``
  status → lifecycle.
* :data:`DYNAMIC_SPECIALIST_TASK_ID_PREFIX` — prefix the pipeline
  uses when synthesising a specialist-shaped workspace; the verdict
  router keys off it to attribute the result back to the dyn_id.
* :func:`materialize_dynamic_patch_workspace` — write the runner's
  proposal into a specialist-shaped layout so the existing
  ``IntegratePatchExecutor`` consumes it without contract changes.
* :func:`build_integrate_patch_proposal_payload` — the payload the
  Coordinator pushes to the bus for the Critic. Carries
  ``provenance="dynamic"`` so the backend enrichment helper flips
  ``review_constraints.cross_domain``.
* :func:`compose_critic_verdict_envelope` — combine the mechanical
  pre-verdict with the LLM-critic verdict using the "strictest wins"
  rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..session_paths import dynamic_action_artifact_dir
from .dynamic_action_critic import (
    CrossDomainPreverdict,
    build_critic_verdict_envelope,
    run_mechanical_cross_domain_checks,
)
from .dynamic_action_proposal import (
    DynamicActionStatus,
    DynamicRunnerTerminalState,
    RUNNER_STATE_TO_STATUS,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DYNAMIC_SPECIALIST_TASK_ID_PREFIX: str = "dyn-"


# ---------------------------------------------------------------------------
# Status mapping helpers
# ---------------------------------------------------------------------------
def runner_status_to_lifecycle(
    terminal_state: str | DynamicRunnerTerminalState,
) -> DynamicActionStatus:
    """Translate :class:`DynamicRunnerTerminalState` to lifecycle.

    Accepts both the enum and its string value. Unknown / empty input
    maps to ``FAILED`` so an unrecognised value never silently lands
    in a non-terminal state.

    Args:
        terminal_state (str | DynamicRunnerTerminalState): Runner
            terminal state as an enum member or its string value.

    Returns:
        DynamicActionStatus: The mapped lifecycle status, or ``FAILED``
        for unknown/empty input.
    """
    if isinstance(terminal_state, DynamicRunnerTerminalState):
        ts = terminal_state
    else:
        try:
            ts = DynamicRunnerTerminalState((terminal_state or "").strip())
        except ValueError:
            return DynamicActionStatus.FAILED
    return RUNNER_STATE_TO_STATUS.get(ts, DynamicActionStatus.FAILED)


def integrate_status_to_lifecycle(
    integrate_status: str,
) -> DynamicActionStatus:
    """Map ``IntegratePatchExecutor.status`` to lifecycle.

    * ``kept``                  → ``KEPT``
    * ``reverted``              → ``REVERTED`` (gain < KEEP threshold
                                  or accuracy gate fails)
    * ``apply_failed`` /
      ``no_patches`` /
      ``failed``                → ``INTEGRATE_FAILED``
    * ``applied_no_bench``      → ``KEPT`` (apply_only mode)
    * everything else           → ``INTEGRATE_FAILED``

    Args:
        integrate_status (str): Status string from
            ``IntegratePatchExecutor`` (case-insensitive).

    Returns:
        DynamicActionStatus: The mapped lifecycle status.
    """
    s = (integrate_status or "").strip().lower()
    if s == "kept" or s == "applied_no_bench":
        return DynamicActionStatus.KEPT
    if s == "reverted":
        return DynamicActionStatus.REVERTED
    return DynamicActionStatus.INTEGRATE_FAILED


# ---------------------------------------------------------------------------
# Specialist-shaped workspace materialisation
# ---------------------------------------------------------------------------
def make_dynamic_specialist_task_id(dyn_id: str) -> str:
    """Canonical synthesised specialist_task_id for ``dyn_id``.

    A ``dyn_id`` already starts with ``dyn-``, so the id is itself.
    Kept as a single point of truth so future renames stay safe.

    Args:
        dyn_id (str): Dynamic-action identifier.

    Returns:
        str: The synthesised specialist task id (the stripped dyn_id).
    """
    return str(dyn_id or "").strip()


def is_dynamic_specialist_task_id(specialist_task_id: str) -> bool:
    """Return whether a specialist task id was synthesised from a dyn_id.

    Used to route integrate completions back to the dyn_id summary.

    Args:
        specialist_task_id (str): Specialist task id to test.

    Returns:
        bool: ``True`` when the id starts with
        :data:`DYNAMIC_SPECIALIST_TASK_ID_PREFIX`.
    """
    sid = (specialist_task_id or "").strip()
    return sid.startswith(DYNAMIC_SPECIALIST_TASK_ID_PREFIX)


def materialize_dynamic_patch_workspace(
    *,
    session_dir: Path,
    dyn_id: str,
    proposal: dict[str, Any],
) -> tuple[str, list[str]]:
    """Write the runner's patch into a specialist-shaped layout.

    ``IntegratePatchExecutor`` discovers patches under
    ``runs/specialist/<sid>/worktree/patches/`` and optionally reads
    ``runs/specialist/<sid>/specialist_done.json`` for
    ``patches_written``.

    Args:
        session_dir (Path): Session directory holding the runs tree.
        dyn_id (str): Dynamic-action identifier for the workspace.
        proposal (dict[str, Any]): Runner proposal carrying ``name`` and
            ``patch_text``.

    Returns:
        tuple[str, list[str]]: ``(specialist_task_id, patches_written)``
        for the integrate_patch task params.
    """
    specialist_task_id = make_dynamic_specialist_task_id(dyn_id)
    workspace = (
        Path(session_dir) / "runs" / "specialist" / specialist_task_id
    )
    patches_dir = workspace / "worktree" / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    name = str(proposal.get("name") or "patch").strip() or "patch"
    safe_name = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in name
    )[:80]
    patch_path = patches_dir / f"001_{safe_name}.patch"
    patch_text = str(proposal.get("patch_text") or "")
    if not patch_text.endswith("\n"):
        patch_text = patch_text + "\n"
    patch_path.write_text(patch_text, encoding="utf-8")
    patches_written = [str(patch_path)]
    done_payload = {
        "domain": "<dynamic-multi>",
        "gap_canonical_id": "",
        "provenance": "dynamic",
        "dyn_id": str(dyn_id),
        "proposal_set": [proposal],
        "patches_written": patches_written,
        "empty": False,
        "summary": (
            "Synthetic specialist_done.json materialised from "
            "dynamic_action proposal_set so integrate_patch finds "
            "the patch via the canonical specialist workspace layout."
        ),
        "confidence": 0.0,
        "new_findings": [],
        "residual_questions": [],
    }
    (workspace / "specialist_done.json").write_text(
        json.dumps(done_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return specialist_task_id, patches_written


# ---------------------------------------------------------------------------
# Critic-bound proposal payload
# ---------------------------------------------------------------------------
def build_integrate_patch_proposal_payload(
    *,
    dyn_id: str,
    specialist_task_id: str,
    proposal: dict[str, Any],
    spec_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``propose_action`` payload pushed to the bus for the
    Critic to review.

    Carries ``provenance="dynamic"`` so the backend enrichment flips
    ``review_constraints.cross_domain``. On approve, the existing
    ``_materialize_approved_proposal`` queues the integrate_patch task
    with the synthesised ``specialist_task_id``.

    Args:
        dyn_id (str): Dynamic-action identifier.
        specialist_task_id (str): Synthesised specialist task id.
        proposal (dict[str, Any]): Runner proposal supplying rationale,
            qualitative argument, and patch name.
        spec_payload (dict[str, Any]): Spec payload supplying
            ``scope_domains``.

    Returns:
        dict[str, Any]: The ``propose_action`` payload for the bus.
    """
    return {
        "action_name": "integrate_patch",
        "provenance": "dynamic",
        "predicted_gain_pct": 0.0,
        "params": {
            "specialist_task_id": specialist_task_id,
            "dyn_id": str(dyn_id),
            "provenance": "dynamic",
            "scope_domains": list(spec_payload.get("scope_domains") or ()),
            "cross_domain_rationale": str(
                proposal.get("cross_domain_rationale") or "",
            ),
            "expected_qualitative_argument": str(
                proposal.get("expected_qualitative_argument") or "",
            ),
            "patch_name": str(proposal.get("name") or ""),
        },
    }


# ---------------------------------------------------------------------------
# Mechanical floor → final verdict envelope
# ---------------------------------------------------------------------------
def compose_critic_verdict_envelope(
    *,
    dyn_id: str,
    proposal: dict[str, Any],
    spec_scope_domains: list[str],
    llm_verdict: str | None = None,
    llm_reason: str | None = None,
) -> tuple[dict[str, Any], DynamicActionStatus]:
    """Combine the mechanical pre-verdict with the LLM-critic verdict.

    Returns ``(envelope_dict, lifecycle_status)``:

    * ``envelope_dict`` matches :data:`CRITIC_VERDICT_FIELDS` exactly.
    * ``lifecycle_status`` is ``CRITIC_REJECTED`` for reject / revise,
      ``INTEGRATING`` for approve.

    Strictest-wins composition: a blocking mechanical pre-verdict
    overrides any LLM verdict. ``revise`` and ``reject`` both land on
    ``CRITIC_REJECTED`` (no sub-agent re-dispatch loop today); the
    verdict label is preserved on ``critic_verdict.json`` for audit.

    Args:
        dyn_id (str): Dynamic-action identifier.
        proposal (dict[str, Any]): Proposal under review.
        spec_scope_domains (list[str]): Authoritative scope domains from
            the spec.
        llm_verdict (str | None): Optional LLM-critic verdict; defaults
            to ``approve`` and is coerced to ``reject`` if unrecognised.
        llm_reason (str | None): Optional LLM-critic rationale recorded
            in the reviewer notes.

    Returns:
        tuple[dict[str, Any], DynamicActionStatus]: The verdict envelope
        and the resulting lifecycle status (``INTEGRATING`` on approve,
        ``CRITIC_REJECTED`` otherwise).
    """
    pre: CrossDomainPreverdict = run_mechanical_cross_domain_checks(
        proposal, spec_scope_domains=list(spec_scope_domains or ()),
    )
    if pre.is_blocking():
        envelope = build_critic_verdict_envelope(
            dyn_id=dyn_id,
            verdict=pre.verdict,
            reason_codes=pre.reason_codes,
            reviewer_notes=pre.reviewer_notes,
            applied_rules=pre.applied_rules,
            cross_domain_flag=pre.cross_domain_flag,
        )
        return envelope, DynamicActionStatus.CRITIC_REJECTED

    verdict = (llm_verdict or "approve").strip().lower()
    if verdict not in {"approve", "reject", "revise"}:
        verdict = "reject"
    reason_codes: list[str] = list(pre.reason_codes)
    reviewer_notes: list[str] = list(pre.reviewer_notes)
    if verdict != "approve":
        reason_codes.append(f"llm_critic_{verdict}")
        if llm_reason and llm_reason.strip():
            reviewer_notes.append(f"llm_critic: {llm_reason.strip()}")
    envelope = build_critic_verdict_envelope(
        dyn_id=dyn_id,
        verdict=verdict,
        reason_codes=reason_codes,
        reviewer_notes=reviewer_notes,
        applied_rules=pre.applied_rules,
        cross_domain_flag=pre.cross_domain_flag,
    )
    if verdict == "approve":
        return envelope, DynamicActionStatus.INTEGRATING
    return envelope, DynamicActionStatus.CRITIC_REJECTED


# ---------------------------------------------------------------------------
# proposal_set on-disk reader (Coordinator hook helper)
# ---------------------------------------------------------------------------
def read_runner_proposal_set(
    session_dir: Path, dyn_id: str,
) -> dict[str, Any] | None:
    """Read the runner's ``proposal_set.json`` from disk.

    Returns ``None`` when the file is absent or unparsable so the caller
    can route the dispatch to ``FAILED`` instead of crashing the tick.

    Args:
        session_dir (Path): Session directory holding the artefact tree.
        dyn_id (str): Dynamic-action identifier for the artefact dir.

    Returns:
        dict[str, Any] | None: Parsed proposal set, or ``None`` when the
        file is missing or invalid JSON.
    """
    artefact = dynamic_action_artifact_dir(session_dir, dyn_id)
    path = artefact / "proposal_set.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "dynamic_action pipeline: failed to read proposal_set for "
            "dyn_id=%s: %r",
            dyn_id, exc,
        )
        return None


__all__ = [
    "DYNAMIC_SPECIALIST_TASK_ID_PREFIX",
    "build_integrate_patch_proposal_payload",
    "compose_critic_verdict_envelope",
    "integrate_status_to_lifecycle",
    "is_dynamic_specialist_task_id",
    "make_dynamic_specialist_task_id",
    "materialize_dynamic_patch_workspace",
    "read_runner_proposal_set",
    "runner_status_to_lifecycle",
]
