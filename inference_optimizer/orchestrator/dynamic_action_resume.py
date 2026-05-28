"""dynamic_action.MD P8 — resume-time abandoned sweep.

Coordinator startup hook that consolidates every non-terminal
dynamic_action into the ``ABANDONED`` terminal state, persists a
structured ``abandoned_on_resume`` record, and frees the residual
worktree / git branch the runner would have cleaned up on a normal
exit.

The module is **side-effect-isolated**:

* reads / writes the per-dyn_id artefact dir
  (``agents/orchestration/dynamic_actions/<dyn_id>/``),
* runs ``git worktree remove`` + ``git branch -D`` as a best-effort
  cleanup (failures log + continue),
* calls :meth:`SharedState.record_dynamic_action_outcome` to flip
  the lifecycle status.

It does **not** touch the bus, the task registry, or any sub-agent
runner — that machinery has already been reset by Coordinator's
own resume path by the time this hook runs.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_dispatch_history_path,
    dynamic_action_spec_path,
    dynamic_actions_root,
)
from .dynamic_action_proposal import (
    DynamicActionStatus,
    LAST_OUTCOME_BY_STATUS,
    MOTIVATION_GAP_SHORT_MAX_CHARS,
    TERMINAL_LIFECYCLE_STATUSES,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Closed schema for dispatch_history.jsonl ``abandoned_on_resume`` rows
# (P8 §6). Any field outside this set is a design change.
# ---------------------------------------------------------------------------
ABANDONED_HISTORY_FIELDS: frozenset[str] = frozenset({
    "event",
    "ts",
    "previous_status",
    "coordinator_session_id",
    "worktree_cleanup_outcome",
    "artifact_missing",
})

# Cleanup outcome enum surfaced on the abandoned_on_resume row (§6).
WORKTREE_CLEANUP_OUTCOMES: frozenset[str] = frozenset({
    "success",   # worktree was present and removed cleanly
    "partial",   # cleanup attempted; some step failed
    "skipped",   # nothing to clean (no worktree / no base repo)
})


@dataclass
class AbandonedSweepResult:
    """Per-invocation summary returned to the caller for audit.

    Coordinator's resume path can log ``len(abandoned)`` /
    ``len(skipped_terminal)`` so the operator sees a one-line
    sweep summary in the boot log.
    """

    abandoned: list[str] = field(default_factory=list)
    skipped_terminal: list[str] = field(default_factory=list)
    artifact_missing: list[str] = field(default_factory=list)
    summary_missing: list[str] = field(default_factory=list)

    def to_log_line(self) -> str:
        return (
            f"dynamic_action resume sweep: "
            f"abandoned={len(self.abandoned)} "
            f"skipped_terminal={len(self.skipped_terminal)} "
            f"artifact_missing={len(self.artifact_missing)} "
            f"summary_missing={len(self.summary_missing)}"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# git helpers — best-effort cleanup of the per-dyn_id worktree + branch
# ---------------------------------------------------------------------------
def _branch_name_for(dyn_id: str) -> str:
    """Mirror :class:`DynamicActionRunner._setup_worktree`'s ``-b``
    argument (``dynamic-<dyn_id>``)."""
    return f"dynamic-{dyn_id}"


def _resolve_dynamic_worktree(session_dir: Path, dyn_id: str) -> Path:
    """Mirror :meth:`DynamicActionRunner._setup_worktree`'s target path
    (``$SESSION_DIR/runs/dynamic/<dyn_id>/worktree/``)."""
    return (
        Path(session_dir) / "runs" / "dynamic" / dyn_id / "worktree"
    )


def _pick_worktree_base(framework_source_roots: tuple[str, ...]) -> Path | None:
    """Same probe as :func:`specialist_subprocess._pick_worktree_base`
    — we inline the logic so this module has no orchestrator import
    cycle."""
    for r in framework_source_roots:
        p = Path(r)
        if not p.is_dir():
            continue
        if (p / ".git").exists():
            return p
    return None


def _delete_branch(base: Path, branch: str) -> bool:
    """Run ``git branch -D <branch>`` inside ``base``. Returns True on
    success or when the branch did not exist; False on any other
    failure."""
    try:
        cp = subprocess.run(
            ["git", "-C", str(base), "branch", "-D", branch],
            capture_output=True, text=True, timeout=15.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.warning(
            "dynamic_action resume: git branch -D %s failed to spawn "
            "in %s: %r", branch, base, exc,
        )
        return False
    if cp.returncode == 0:
        return True
    # ``not found`` exits non-zero but is the no-op success case
    # we treat as cleanly handled.
    stderr = (cp.stderr or "").lower()
    if "not found" in stderr:
        return True
    log.warning(
        "dynamic_action resume: git branch -D %s rc=%d stderr=%r",
        branch, cp.returncode, (cp.stderr or "").strip()[-400:],
    )
    return False


def _cleanup_worktree_and_branch(
    *,
    session_dir: Path,
    dyn_id: str,
    framework_source_roots: tuple[str, ...] = (),
) -> str:
    """Run the §5 cleanup sequence. Returns one of
    :data:`WORKTREE_CLEANUP_OUTCOMES`."""
    worktree = _resolve_dynamic_worktree(session_dir, dyn_id)
    base = _pick_worktree_base(framework_source_roots)
    if not worktree.exists() and base is None:
        return "skipped"
    outcome = "success"
    if worktree.exists():
        wt_removed = False
        if base is not None and (base / ".git").exists():
            try:
                cp = subprocess.run(
                    ["git", "-C", str(base), "worktree", "remove",
                     "--force", str(worktree)],
                    capture_output=True, text=True, timeout=30.0,
                    check=False,
                )
                wt_removed = cp.returncode == 0
                if not wt_removed:
                    log.warning(
                        "dynamic_action resume: git worktree remove %s "
                        "rc=%d stderr=%r",
                        worktree, cp.returncode,
                        (cp.stderr or "").strip()[-400:],
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                log.warning(
                    "dynamic_action resume: git worktree remove %s "
                    "failed to spawn: %r", worktree, exc,
                )
        if worktree.exists():
            try:
                shutil.rmtree(worktree, ignore_errors=True)
            except OSError:
                log.exception(
                    "dynamic_action resume: rm -rf %s failed", worktree,
                )
        if worktree.exists():
            outcome = "partial"
    if base is not None:
        branch_ok = _delete_branch(base, _branch_name_for(dyn_id))
        if not branch_ok and outcome == "success":
            outcome = "partial"
    return outcome


# ---------------------------------------------------------------------------
# dispatch_history.jsonl writer
# ---------------------------------------------------------------------------
def _append_abandoned_history(
    *,
    session_dir: Path,
    dyn_id: str,
    previous_status: str,
    coordinator_session_id: str,
    worktree_cleanup_outcome: str,
    artifact_missing: bool,
) -> None:
    """Append one ``abandoned_on_resume`` row (§6 closed schema)."""
    if worktree_cleanup_outcome not in WORKTREE_CLEANUP_OUTCOMES:
        raise ValueError(
            f"worktree_cleanup_outcome={worktree_cleanup_outcome!r} not "
            f"in {sorted(WORKTREE_CLEANUP_OUTCOMES)!r}"
        )
    row = {
        "event": "abandoned_on_resume",
        "ts": _now_iso(),
        "previous_status": str(previous_status or ""),
        "coordinator_session_id": str(coordinator_session_id or ""),
        "worktree_cleanup_outcome": worktree_cleanup_outcome,
        "artifact_missing": bool(artifact_missing),
    }
    extra_keys = set(row.keys()) - ABANDONED_HISTORY_FIELDS
    if extra_keys:
        raise ValueError(
            f"abandoned history row carries unknown fields: "
            f"{sorted(extra_keys)!r}"
        )
    target = dynamic_action_dispatch_history_path(session_dir, dyn_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Spec.json reader (used to backfill a missing summary row)
# ---------------------------------------------------------------------------
def _load_spec_for_recovery(
    session_dir: Path, dyn_id: str,
) -> dict[str, Any] | None:
    spec_path = dynamic_action_spec_path(session_dir, dyn_id)
    if not spec_path.is_file():
        return None
    try:
        return json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "dynamic_action resume: failed to load spec.json for "
            "dyn_id=%s: %r", dyn_id, exc,
        )
        return None


def _seed_recovery_summary(
    *,
    shared_state: Any,
    dyn_id: str,
    spec: dict[str, Any],
) -> None:
    """Re-create a missing summary row at status=DISPATCHED so the
    transition validator accepts the subsequent DISPATCHED → ABANDONED
    step. Mirrors :func:`Coordinator._ensure_dynamic_action_dispatched_row`
    but lives here to keep the resume hook decoupled from the
    coordinator class."""
    payload = spec.get("payload") or {}
    motivation = str(payload.get("motivation_gap_text") or "")
    if len(motivation) > MOTIVATION_GAP_SHORT_MAX_CHARS:
        motivation = (
            motivation[: MOTIVATION_GAP_SHORT_MAX_CHARS - 3].rstrip()
            + "..."
        )
    extra = {
        "dyn_id": dyn_id,
        "round_index": spec.get("round_index"),
        "scope_domains": list(payload.get("scope_domains") or ()),
        "motivation_gap_short": motivation,
        "verdict": None,
        "cumulative_gain": None,
        "artifact_path": "",
        "dispatched_at": str(spec.get("dispatched_at") or ""),
        "synthesised_row": True,
    }
    shared_state.record_dynamic_action_outcome(
        dyn_id,
        status=DynamicActionStatus.DISPATCHED.value,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def resume_abandon_dynamic_actions(
    *,
    session_dir: Path,
    shared_state: Any,
    coordinator_session_id: str = "",
    framework_source_roots: tuple[str, ...] = (),
) -> AbandonedSweepResult:
    """Sweep both the artefact dir and the SharedState summary map;
    every non-terminal dyn_id is transitioned to ABANDONED.

    Coordinator calls this from its resume path (after
    :meth:`replay_for_resume`); the side-effects are bounded to the
    per-dyn_id artefact dir + git plumbing, so this helper is safe
    to invoke multiple times (terminal states no-op).
    """
    result = AbandonedSweepResult()
    session_dir = Path(session_dir)
    root = dynamic_actions_root(session_dir)

    artefact_dyn_ids: set[str] = set()
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                artefact_dyn_ids.add(child.name)

    summary_dyn_ids: set[str] = set()
    summaries = getattr(shared_state, "dynamic_actions", None) or {}
    if isinstance(summaries, dict):
        summary_dyn_ids = {str(k) for k in summaries.keys()}

    all_dyn_ids = sorted(artefact_dyn_ids | summary_dyn_ids)
    for dyn_id in all_dyn_ids:
        if not dyn_id:
            continue
        try:
            _process_one(
                dyn_id=dyn_id,
                session_dir=session_dir,
                shared_state=shared_state,
                coordinator_session_id=coordinator_session_id,
                framework_source_roots=framework_source_roots,
                artefact_dyn_ids=artefact_dyn_ids,
                summaries=summaries,
                result=result,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action resume sweep: failed to process dyn_id=%s",
                dyn_id,
            )
    return result


def _process_one(
    *,
    dyn_id: str,
    session_dir: Path,
    shared_state: Any,
    coordinator_session_id: str,
    framework_source_roots: tuple[str, ...],
    artefact_dyn_ids: set[str],
    summaries: dict[str, Any],
    result: AbandonedSweepResult,
) -> None:
    artefact_present = dyn_id in artefact_dyn_ids
    summary = summaries.get(dyn_id) if isinstance(summaries, dict) else None
    previous_status_raw = (
        str(summary.get("status") or "")
        if isinstance(summary, dict) else ""
    )

    # Corner case A — artefact present, summary missing: rebuild a
    # synthetic DISPATCHED row first so the transition validator
    # accepts the subsequent ABANDONED step.
    if artefact_present and not isinstance(summary, dict):
        spec = _load_spec_for_recovery(session_dir, dyn_id)
        if spec is not None:
            _seed_recovery_summary(
                shared_state=shared_state, dyn_id=dyn_id, spec=spec,
            )
            previous_status_raw = DynamicActionStatus.DISPATCHED.value
            result.summary_missing.append(dyn_id)
        else:
            # spec.json missing too → cannot reconstruct anything;
            # log + skip.
            log.warning(
                "dynamic_action resume: artefact dir %s present but "
                "spec.json missing; skipping",
                dynamic_action_artifact_dir(session_dir, dyn_id),
            )
            return

    # Terminal already → no-op (P8 §4.2 末分支 + §8.4 idempotency).
    try:
        previous_status = DynamicActionStatus(previous_status_raw)
    except ValueError:
        # Unknown / empty status — when we have a summary but no
        # parseable status, fall through to ABANDONED. When neither
        # artefact nor summary exists, skip silently.
        if not artefact_present and not isinstance(summary, dict):
            return
        previous_status = DynamicActionStatus.DISPATCHED
    if previous_status in TERMINAL_LIFECYCLE_STATUSES:
        result.skipped_terminal.append(dyn_id)
        return

    artifact_missing = not artefact_present
    if artifact_missing:
        result.artifact_missing.append(dyn_id)

    cleanup_outcome = "skipped"
    if artefact_present:
        cleanup_outcome = _cleanup_worktree_and_branch(
            session_dir=session_dir,
            dyn_id=dyn_id,
            framework_source_roots=framework_source_roots,
        )

    extra = {
        "abandoned_at": _now_iso(),
        "abandoned_previous_status": previous_status.value,
        "abandoned_worktree_cleanup_outcome": cleanup_outcome,
    }
    if artifact_missing:
        extra["artifact_missing"] = True
    shared_state.record_dynamic_action_outcome(
        dyn_id,
        status=DynamicActionStatus.ABANDONED.value,
        last_outcome=LAST_OUTCOME_BY_STATUS[DynamicActionStatus.ABANDONED],
        extra=extra,
    )
    if artefact_present:
        try:
            _append_abandoned_history(
                session_dir=session_dir,
                dyn_id=dyn_id,
                previous_status=previous_status.value,
                coordinator_session_id=coordinator_session_id,
                worktree_cleanup_outcome=cleanup_outcome,
                artifact_missing=artifact_missing,
            )
        except (OSError, ValueError):
            log.exception(
                "dynamic_action resume: dispatch_history append failed "
                "for dyn_id=%s",
                dyn_id,
            )
    result.abandoned.append(dyn_id)


__all__ = [
    "ABANDONED_HISTORY_FIELDS",
    "AbandonedSweepResult",
    "WORKTREE_CLEANUP_OUTCOMES",
    "resume_abandon_dynamic_actions",
]
