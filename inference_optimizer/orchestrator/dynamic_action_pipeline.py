"""dynamic_action.MD P5 — end-to-end pipeline glue (pure helpers).

P5 §8 forbids touching the specialist done / integrate_patch / grid /
promote main chain; every dynamic-only piece of logic lives here so
the Coordinator wiring stays thin. The functions below are
side-effect-isolated (filesystem writes only) and import-cycle-free.

Public surface:

* :func:`runner_status_to_lifecycle` — map P3 terminal state → P5
  lifecycle status (with the P5 §6 attribution table baked in).
* :func:`integrate_status_to_lifecycle` — map the
  ``IntegratePatchExecutor`` status field → P5 lifecycle status.
* :data:`DYNAMIC_SPECIALIST_TASK_ID_PREFIX` — the literal that the
  pipeline prepends to a ``dyn_id`` when synthesising a
  specialist-shaped workspace; ``_handle_single_verdict`` keys off
  this to route the integrate verdict back onto the dyn_id summary.
* :func:`materialize_dynamic_patch_workspace` — copy the runner's
  proposal_set.json patch_text into a specialist-shaped layout
  (``runs/specialist/dyn-<dyn_id>/worktree/patches/`` + a synthetic
  ``specialist_done.json``) so the existing IntegratePatchExecutor
  finds the patch without any input-contract change (P5 §8 #2).
* :func:`build_integrate_patch_proposal_payload` — the payload the
  Coordinator pushes to the bus as a "proposal" that the Critic
  reviews. Carries ``provenance="dynamic"`` so the P4 enrichment
  helper auto-flips ``review_constraints.cross_domain``.
* :func:`compose_critic_verdict_envelope` — combine the P4
  mechanical pre-verdict with the LLM-critic's verdict using the
  "take the strictest" rule.
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
    """Translate :class:`DynamicRunnerTerminalState` (P3) to the
    lifecycle status (P5 §6).

    Accepts both the enum and its string value so the dispatcher can
    feed the runner result dict directly. Unknown / empty input maps
    to FAILED — a defensive choice that keeps an unrecognised value
    from silently slipping the dispatch into a non-terminal state.
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
    """Map the ``IntegratePatchExecutor.status`` literal to P5 §6.

    Mapping (anchored at integrate_patch.py docstring):

    * ``kept``                  → KEPT
    * ``reverted``              → REVERTED (covers gain < KEEP threshold
                                  and accuracy gate fails uniformly)
    * ``apply_failed`` /
      ``no_patches`` /
      ``failed``                → INTEGRATE_FAILED
    * ``applied_no_bench``      → KEPT (apply_only mode; treated as a
                                  successful integrate for lifecycle
                                  audit even though no KEEP gate ran)

    Anything outside the closed set lands in INTEGRATE_FAILED so an
    unrecognised status never reads as a success.
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
    """``dyn-0-1`` → ``dyn-dyn-0-1``-shaped? No — the dyn_id already
    starts with ``dyn-`` (P2 §3.3), so the canonical synthesised id is
    just the dyn_id itself. We keep this helper as the single point of
    truth so future renames stay safe."""
    return str(dyn_id or "").strip()


def is_dynamic_specialist_task_id(specialist_task_id: str) -> bool:
    """Predicate the Coordinator uses to route an integrate_patch
    completion back onto the dyn_id summary."""
    sid = (specialist_task_id or "").strip()
    return sid.startswith(DYNAMIC_SPECIALIST_TASK_ID_PREFIX)


def materialize_dynamic_patch_workspace(
    *,
    session_dir: Path,
    dyn_id: str,
    proposal: dict[str, Any],
) -> tuple[str, list[str]]:
    """Place the runner's patch_text into a specialist-shaped layout.

    The existing ``IntegratePatchExecutor`` discovers patches under
    ``runs/specialist/<sid>/worktree/patches/`` and optionally reads
    ``runs/specialist/<sid>/specialist_done.json`` for
    ``patches_written``. We mimic that layout so the executor (and
    the Critic gate logic on ``record_specialist_patch_verdict``)
    runs unmodified.

    Returns the ``(specialist_task_id, patches_written)`` pair the
    Coordinator passes through to the integrate_patch task params.
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
    )
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
    """Construct the ``propose_action`` payload the Coordinator will
    push to the bus for the Critic to review.

    Carries ``provenance="dynamic"`` so the P4 backend enrichment
    flips ``review_constraints.cross_domain`` automatically. The
    Critic sees a single-proposal integrate_patch shape; on approve
    the existing ``_materialize_approved_proposal`` queues the
    integrate_patch task with the synthesised ``specialist_task_id``.
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
    """Combine the P4 mechanical pre-verdict with the LLM-critic verdict.

    Returns ``(envelope_dict, lifecycle_status)``:

    * ``envelope_dict`` matches :data:`CRITIC_VERDICT_FIELDS` exactly.
    * ``lifecycle_status`` is ``CRITIC_REJECTED`` when the final
      verdict is reject/revise, ``DISPATCHED`` when approve (the
      pipeline advances to integrate_patch; the integrate completion
      hook flips the status to KEPT / REVERTED / INTEGRATE_FAILED).

    Strictest-wins composition: if mechanical pre says
    reject/revise, the LLM verdict is ignored — mechanical layer is a
    *floor* per P4 §7.
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
        return envelope, DynamicActionStatus.DISPATCHED
    return envelope, DynamicActionStatus.CRITIC_REJECTED


# ---------------------------------------------------------------------------
# proposal_set on-disk reader (Coordinator hook helper)
# ---------------------------------------------------------------------------
def read_runner_proposal_set(
    session_dir: Path, dyn_id: str,
) -> dict[str, Any] | None:
    """Read the per-dispatch ``proposal_set.json`` that the runner
    wrote (P3 §6 recovery whitelist). Returns ``None`` when the file
    is absent or unparsable so the caller can route the dispatch to
    FAILED without crashing the dispatcher tick."""
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
