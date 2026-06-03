"""Resume-time abandoned sweep for ``dynamic_action``.

Coordinator startup hook that:

* transitions every non-terminal ``dyn_id`` to ``ABANDONED``;
* appends an ``abandoned_on_resume`` row to dispatch_history.jsonl;
* writes a terminal telemetry rollup;
* tears down the residual worktree + git branch (best-effort, log on
  failure).

Side-effects are bounded to the per-dyn_id artefact dir + git
plumbing; the bus, task registry, and sub-agent runners are not
touched (the Coordinator's own resume path has already reset them).
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
from .dynamic_action_history import (
    ABANDONED_FIELDS as _ABANDONED_FIELDS_CANONICAL,
    DispatchHistoryEvent,
    append_dispatch_history_row,
    write_dynamic_action_telemetry,
)
from .dynamic_action_proposal import (
    DynamicActionStatus,
    LAST_OUTCOME_BY_STATUS,
    MOTIVATION_GAP_SHORT_MAX_CHARS,
    TERMINAL_LIFECYCLE_STATUSES,
)


log = logging.getLogger(__name__)


# Backwards-compatible alias for the canonical schema in
# :mod:`dynamic_action_history`.
ABANDONED_HISTORY_FIELDS: frozenset[str] = _ABANDONED_FIELDS_CANONICAL

# Cleanup outcomes surfaced on the abandoned_on_resume row.
WORKTREE_CLEANUP_OUTCOMES: frozenset[str] = frozenset({
    "success",   # worktree removed cleanly
    "partial",   # cleanup attempted; some step failed
    "skipped",   # nothing to clean (no worktree / no base repo)
})


@dataclass
class AbandonedSweepResult:
    """Per-invocation summary for boot-log auditing."""

    abandoned: list[str] = field(default_factory=list)
    skipped_terminal: list[str] = field(default_factory=list)
    artifact_missing: list[str] = field(default_factory=list)
    summary_missing: list[str] = field(default_factory=list)

    def to_log_line(self) -> str:
        """Render the sweep counts as a single boot-log line.

        Returns:
            str: One-line summary of abandoned, skipped-terminal,
            artifact-missing, and summary-missing counts.
        """
        return (
            f"dynamic_action resume sweep: "
            f"abandoned={len(self.abandoned)} "
            f"skipped_terminal={len(self.skipped_terminal)} "
            f"artifact_missing={len(self.artifact_missing)} "
            f"summary_missing={len(self.summary_missing)}"
        )


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        str: Timestamp with microsecond precision in UTC.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# git helpers — best-effort cleanup of the per-dyn_id worktree + branch
# ---------------------------------------------------------------------------
def _branch_name_for(dyn_id: str) -> str:
    """Return the branch name created by the dynamic-action runner.

    Args:
        dyn_id (str): Dynamic-action identifier.

    Returns:
        str: Branch name (``dynamic-<dyn_id>``) used for the worktree.
    """
    return f"dynamic-{dyn_id}"


def _resolve_dynamic_worktree(session_dir: Path, dyn_id: str) -> Path:
    """Return the per-dyn_id worktree path used by the runner.

    Args:
        session_dir (Path): Session directory holding the runs tree.
        dyn_id (str): Dynamic-action identifier.

    Returns:
        Path: ``<session_dir>/runs/dynamic/<dyn_id>/worktree``.
    """
    return (
        Path(session_dir) / "runs" / "dynamic" / dyn_id / "worktree"
    )


def _pick_worktree_base(framework_source_roots: tuple[str, ...]) -> Path | None:
    """Return the first ``framework_source_root`` that is a git checkout.

    Inlined to keep this module free of orchestrator imports.

    Args:
        framework_source_roots (tuple[str, ...]): Candidate root paths.

    Returns:
        Path | None: The first directory containing a ``.git`` entry, or
        ``None`` when none qualifies.
    """
    for r in framework_source_roots:
        p = Path(r)
        if not p.is_dir():
            continue
        if (p / ".git").exists():
            return p
    return None


def _delete_branch(base: Path, branch: str) -> bool:
    """Force-delete a git branch inside a checkout.

    Runs ``git branch -D <branch>`` inside ``base``.

    Args:
        base (Path): Git checkout to run the command in.
        branch (str): Branch name to delete.

    Returns:
        bool: ``True`` on success or when the branch did not exist;
        ``False`` if git failed to spawn or returned another error.
    """
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
    """Drop the per-dyn_id worktree and its branch.

    Attempts ``git worktree remove`` then ``shutil.rmtree`` for the
    worktree, followed by branch deletion, logging each failure.

    Args:
        session_dir (Path): Session directory holding the runs tree.
        dyn_id (str): Dynamic-action identifier.
        framework_source_roots (tuple[str, ...]): Candidate git
            checkouts used to locate the worktree base.

    Returns:
        str: One of :data:`WORKTREE_CLEANUP_OUTCOMES` (``success``,
        ``partial``, or ``skipped``).
    """
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
    """Append one ``abandoned_on_resume`` row via the unified writer.

    Args:
        session_dir (Path): Session directory holding dispatch history.
        dyn_id (str): Dynamic-action identifier.
        previous_status (str): Status the dyn_id held before the sweep.
        coordinator_session_id (str): Identifier of the resuming
            coordinator session.
        worktree_cleanup_outcome (str): One of
            :data:`WORKTREE_CLEANUP_OUTCOMES`.
        artifact_missing (bool): Whether the artefact dir was absent.

    Raises:
        ValueError: If ``worktree_cleanup_outcome`` is not a recognised
            cleanup outcome.
    """
    if worktree_cleanup_outcome not in WORKTREE_CLEANUP_OUTCOMES:
        raise ValueError(
            f"worktree_cleanup_outcome={worktree_cleanup_outcome!r} not "
            f"in {sorted(WORKTREE_CLEANUP_OUTCOMES)!r}"
        )
    append_dispatch_history_row(
        session_dir=session_dir,
        dyn_id=dyn_id,
        event=DispatchHistoryEvent.ABANDONED_ON_RESUME,
        payload={
            "previous_status": str(previous_status or ""),
            "coordinator_session_id": str(coordinator_session_id or ""),
            "worktree_cleanup_outcome": worktree_cleanup_outcome,
            "artifact_missing": bool(artifact_missing),
        },
    )


# ---------------------------------------------------------------------------
# Spec.json reader (used to backfill a missing summary row)
# ---------------------------------------------------------------------------
def _load_spec_for_recovery(
    session_dir: Path, dyn_id: str,
) -> dict[str, Any] | None:
    """Load ``spec.json`` for a dyn_id to backfill recovery state.

    Args:
        session_dir (Path): Session directory holding the spec file.
        dyn_id (str): Dynamic-action identifier.

    Returns:
        dict[str, Any] | None: Parsed spec contents, or ``None`` when
        the file is missing, unreadable, or invalid JSON.
    """
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
    """Synthesise a ``DISPATCHED`` summary row from ``spec.json``.

    Seeds the row so a subsequent ``DISPATCHED → ABANDONED`` write is
    accepted by the transition validator.

    Args:
        shared_state (Any): SharedState whose
            ``record_dynamic_action_outcome`` is called.
        dyn_id (str): Dynamic-action identifier.
        spec (dict[str, Any]): Parsed spec contents used to populate the
            synthesised row.
    """
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
    """Transition every non-terminal dyn_id to ``ABANDONED``.

    Sweeps both the artefact dir and the SharedState summary map.
    Safe to invoke multiple times — terminal statuses are no-ops.

    Args:
        session_dir (Path): Session directory to sweep.
        shared_state (Any): SharedState holding dynamic-action summaries.
        coordinator_session_id (str): Identifier of the resuming
            coordinator session, recorded on each row.
        framework_source_roots (tuple[str, ...]): Candidate git
            checkouts used for worktree cleanup.

    Returns:
        AbandonedSweepResult: Per-invocation summary of the dyn_ids
        abandoned, skipped, or missing artefacts/summaries.
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
    """Process one dyn_id during the abandoned sweep.

    Recovers a missing summary from ``spec.json`` when needed, skips
    terminal statuses, cleans up the worktree/branch, records the
    ABANDONED outcome, and appends history + telemetry. Mutates
    ``result`` to track the outcome category.

    Args:
        dyn_id (str): Dynamic-action identifier to process.
        session_dir (Path): Session directory holding artefacts.
        shared_state (Any): SharedState updated with the outcome.
        coordinator_session_id (str): Identifier of the resuming
            coordinator session.
        framework_source_roots (tuple[str, ...]): Candidate git
            checkouts used for worktree cleanup.
        artefact_dyn_ids (set[str]): Dyn_ids that have an artefact dir.
        summaries (dict[str, Any]): SharedState summary map keyed by
            dyn_id.
        result (AbandonedSweepResult): Accumulator mutated in place with
            the categorised outcome.
    """
    artefact_present = dyn_id in artefact_dyn_ids
    summary = summaries.get(dyn_id) if isinstance(summaries, dict) else None
    previous_status_raw = (
        str(summary.get("status") or "")
        if isinstance(summary, dict) else ""
    )

    # Artefact present but summary missing: rebuild a synthetic
    # DISPATCHED row so the transition validator accepts the
    # subsequent ABANDONED write.
    if artefact_present and not isinstance(summary, dict):
        spec = _load_spec_for_recovery(session_dir, dyn_id)
        if spec is not None:
            _seed_recovery_summary(
                shared_state=shared_state, dyn_id=dyn_id, spec=spec,
            )
            previous_status_raw = DynamicActionStatus.DISPATCHED.value
            result.summary_missing.append(dyn_id)
        else:
            log.warning(
                "dynamic_action resume: artefact dir %s present but "
                "spec.json missing; skipping",
                dynamic_action_artifact_dir(session_dir, dyn_id),
            )
            return

    try:
        previous_status = DynamicActionStatus(previous_status_raw)
    except ValueError:
        # Empty/unknown status: skip silently when there is nothing
        # to recover, otherwise treat as ``DISPATCHED`` so the sweep
        # can still transition.
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
        # ABANDONED counts toward the per-dyn_id terminal-state
        # telemetry tally.
        try:
            write_dynamic_action_telemetry(
                session_dir=session_dir,
                dyn_id=dyn_id,
                lifecycle=DynamicActionStatus.ABANDONED,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action resume: telemetry write failed "
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
