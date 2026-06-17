# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session-end postmortem + decision trace finalizer (L1 + L2).

Fired once per session on the first non-empty ``stop_reason``; idempotent
via a ``.robustness_finalized`` marker under ``<session_dir>/reports/``.

Outputs (under ``<session_dir>/reports/``):

* ``robustness_postmortem.md`` — markdown human-facing summary.
* ``decision_trace.json`` — machine-readable per-task ledger.

Best-effort: per-file errors are logged and skipped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Idempotency marker; lives under ``reports/`` for one-``rm -r`` cleanup.
_FINALIZED_MARKER_NAME: str = ".robustness_finalized"

_POSTMORTEM_FILENAME: str = "robustness_postmortem.md"
_DECISION_TRACE_FILENAME: str = "decision_trace.json"

# Action families scanned under ``runs/`` for L2. Mirrors
# ``inference_optimizer.session_paths.RUN_ACTION_FAMILIES`` (source of truth).
_DECISION_TRACE_ACTION_DIRS: tuple[str, ...] = (
    "baseline",
    "profile",
    "roofline",
    "explore",
    "sweep",
    "integrate",
    "kernel_opt",
    "deep_kernel_analysis",
    "operator_tuning",
    "vendor_kernel_config",
    "recover",
    "target_analysis",
    "report",
)


@dataclass
class PostmortemFinalizerConfig:
    """Tunables for the postmortem writer.

    Defaults match the SKILL.md ``reports/`` convention; no operator setup
    needed for the common case.
    """

    reports_subdir: str = "reports"
    # Where FindingSink writes; the finalizer only reads here.
    findings_subdir: str = "agents/robustness/findings"
    runs_subdir: str = "runs"
    # HIGH-severity findings rendered in the body (cap keeps markdown readable).
    max_findings_in_report: int = 20
    # Recent tasks per action in the trace (keeps JSON small on sweep-heavy runs).
    max_tasks_per_action: int = 50


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

class PostmortemFinalizer:
    """Once-per-session aggregator for L1 (flashpoint) + L2 (decision trace).

    Lifecycle: callers (the reactor) invoke :meth:`finalize` once when
    ``stop_reason`` is first observed non-empty. Re-running has no
    effect after the marker file is in place.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        session_id: str,
        config: PostmortemFinalizerConfig | None = None,
    ) -> None:
        """Initialise the finalizer for one session.

        Args:
            session_dir (Path): Root session directory; outputs land under
                its ``reports/`` subdirectory.
            session_id (str): Session identifier; defaults to ``"default"``
                when empty.
            config (PostmortemFinalizerConfig | None): Optional tunables;
                a default config is used when omitted.
        """
        self.session_dir = Path(session_dir)
        self.session_id = session_id or "default"
        self._config = config or PostmortemFinalizerConfig()

    # ------------------------------------------------------------------
    # marker
    # ------------------------------------------------------------------
    @property
    def reports_dir(self) -> Path:
        """Directory under the session where outputs are written.

        Returns:
            Path: The ``<session_dir>/reports`` directory path.
        """
        return self.session_dir / self._config.reports_subdir

    @property
    def marker_path(self) -> Path:
        """Path of the idempotency marker file.

        Returns:
            Path: The ``.robustness_finalized`` marker under
            :attr:`reports_dir`.
        """
        return self.reports_dir / _FINALIZED_MARKER_NAME

    def is_finalized(self) -> bool:
        """Report whether this session was already finalized.

        Returns:
            bool: True when the marker file exists on disk.
        """
        return self.marker_path.is_file()

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def finalize(self, *, stop_reason: str) -> bool:
        """Run the L1+L2 pipeline. Returns True if we wrote new files.

        Best-effort: any IO error is logged and swallowed. The reactor
        must never crash because the postmortem failed to write.

        Args:
            stop_reason (str): The session's stop reason, recorded in the
                postmortem and marker.

        Returns:
            bool: True if any output file was written; False when already
            finalized or the reports directory could not be created.
        """
        if self.is_finalized():
            log.debug(
                "postmortem already finalized at %s — skipping",
                self.marker_path,
            )
            return False
        try:
            findings = self._load_findings()
        except Exception:  # noqa: BLE001 — best-effort, log and continue
            log.exception("postmortem: failed to load findings")
            findings = []
        try:
            decision_trace = self._build_decision_trace()
        except Exception:  # noqa: BLE001
            log.exception("postmortem: failed to build decision trace")
            decision_trace = {"tasks_by_action": {}, "total_tasks": 0}
        try:
            postmortem_md = self._build_postmortem_md(
                findings=findings,
                decision_trace=decision_trace,
                stop_reason=stop_reason,
            )
        except Exception:  # noqa: BLE001
            log.exception("postmortem: failed to render markdown")
            postmortem_md = self._fallback_postmortem_md(
                stop_reason=stop_reason
            )
        # Best-effort writes; failing to write the trace shouldn't stop
        # us from writing the marker (idempotency wins).
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("postmortem: cannot create %s: %s", self.reports_dir, exc)
            return False
        wrote_any = False
        wrote_any |= self._write_text(_POSTMORTEM_FILENAME, postmortem_md)
        wrote_any |= self._write_json(
            _DECISION_TRACE_FILENAME, decision_trace
        )
        self._write_marker(stop_reason=stop_reason)
        return wrote_any

    # ------------------------------------------------------------------
    # findings — L1 inputs
    # ------------------------------------------------------------------
    def _findings_path(self) -> Path:
        """Path of the FindingSink JSONL file for this session.

        Returns:
            Path: ``<session_dir>/<findings_subdir>/<session_id>.jsonl``.
        """
        return (
            self.session_dir
            / self._config.findings_subdir
            / f"{self.session_id}.jsonl"
        )

    def _load_findings(self) -> list[dict[str, Any]]:
        """Load the session findings from the FindingSink JSONL file.

        Missing files, read errors, and malformed JSON lines are tolerated
        and skipped rather than raised.

        Returns:
            list[dict[str, Any]]: One dict per well-formed finding row;
            empty when the file is absent or unreadable.
        """
        path = self._findings_path()
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("postmortem: cannot read findings %s: %s", path, exc)
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    # ------------------------------------------------------------------
    # decision trace — L2 inputs
    # ------------------------------------------------------------------
    def _build_decision_trace(self) -> dict[str, Any]:
        """Scan ``runs/<action>/<task>/result.json`` into a trace dict.

        Walks each known action directory, collecting recent task results
        per the configured caps. A missing ``runs/`` root yields an empty
        trace.

        Returns:
            dict[str, Any]: Trace with ``session_id``, ``tasks_by_action``
            and ``total_tasks`` keys.
        """
        runs_root = self.session_dir / self._config.runs_subdir
        out: dict[str, Any] = {
            "session_id": self.session_id,
            "tasks_by_action": {},
            "total_tasks": 0,
        }
        if not runs_root.is_dir():
            return out
        cfg = self._config
        for action in _DECISION_TRACE_ACTION_DIRS:
            action_dir = runs_root / action
            if not action_dir.is_dir():
                continue
            tasks = self._collect_action_tasks(action_dir, cfg)
            if not tasks:
                continue
            out["tasks_by_action"][action] = tasks
            out["total_tasks"] += len(tasks)
        return out

    def _collect_action_tasks(
        self, action_dir: Path, cfg: PostmortemFinalizerConfig,
    ) -> list[dict[str, Any]]:
        """Collect normalised task entries for one action directory.

        Reads up to ``cfg.max_tasks_per_action`` most-recent task dirs,
        recording an error entry for any missing / unparsable / non-dict
        ``result.json`` instead of dropping the task.

        Args:
            action_dir (Path): The ``runs/<action>`` directory to scan.
            cfg (PostmortemFinalizerConfig): Config supplying the
                per-action task cap.

        Returns:
            list[dict[str, Any]]: One entry per task (normalised result or
            error marker).
        """
        try:
            task_dirs = sorted(
                (p for p in action_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[: cfg.max_tasks_per_action]
        except OSError as exc:
            log.debug("postmortem: action dir %s scan failed: %s", action_dir, exc)
            return []
        out: list[dict[str, Any]] = []
        for task_dir in task_dirs:
            result_path = task_dir / "result.json"
            if not result_path.is_file():
                out.append({
                    "task_id": task_dir.name,
                    "workspace": str(task_dir),
                    "result_path": None,
                    "error": "result.json_missing",
                })
                continue
            try:
                raw = result_path.read_text(encoding="utf-8")
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                out.append({
                    "task_id": task_dir.name,
                    "workspace": str(task_dir),
                    "result_path": str(result_path),
                    "error": f"result.json_parse_failed: {exc}",
                })
                continue
            if not isinstance(payload, dict):
                out.append({
                    "task_id": task_dir.name,
                    "workspace": str(task_dir),
                    "result_path": str(result_path),
                    "error": "result.json_not_dict",
                })
                continue
            out.append(_normalise_task_entry(task_dir, result_path, payload))
        return out

    # ------------------------------------------------------------------
    # markdown rendering
    # ------------------------------------------------------------------
    def _build_postmortem_md(
        self,
        *,
        findings: list[dict[str, Any]],
        decision_trace: dict[str, Any],
        stop_reason: str,
    ) -> str:
        """Render the human-facing postmortem markdown.

        Composes a header, flashpoint section, findings catalogue and
        decision-trace summary table from the prepared inputs.

        Args:
            findings (list[dict[str, Any]]): Loaded findings rows.
            decision_trace (dict[str, Any]): Trace built by
                :meth:`_build_decision_trace`.
            stop_reason (str): The session's stop reason.

        Returns:
            str: The full postmortem markdown document.
        """
        cfg = self._config
        lines: list[str] = []
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines.append(f"# Robustness postmortem — `{self.session_id}`")
        lines.append("")
        lines.append(f"- **stop_reason**: `{stop_reason or '(unspecified)'}`")
        lines.append(f"- **finalized_at_utc**: `{now_iso}`")
        lines.append(f"- **findings_count**: {len(findings)}")
        lines.append(
            f"- **tasks_count**: {decision_trace.get('total_tasks', 0)}"
        )
        lines.append("")

        # ----- Flashpoint -----
        flashpoint = _pick_flashpoint(findings)
        lines.append("## Flashpoint")
        lines.append("")
        if flashpoint is None:
            lines.append(
                "_No HIGH-severity finding recorded this session._ The "
                "stop_reason above is the only signal."
            )
        else:
            lines.append(
                f"- **symptom**: `{flashpoint.get('symptom_name', '(unknown)')}`"
            )
            lines.append(
                f"- **severity**: `{flashpoint.get('severity', '?')}`"
            )
            lines.append(
                f"- **tick_index**: `{flashpoint.get('tick_index', '?')}`  "
                f"**timestamp_unix**: `{flashpoint.get('timestamp_unix', '?')}`"
            )
            summary = str(flashpoint.get("summary", "")).strip()
            if summary:
                lines.append("")
                lines.append(f"> {summary}")
            evidence = flashpoint.get("evidence")
            if isinstance(evidence, dict) and evidence:
                lines.append("")
                lines.append("**Evidence:**")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(evidence, indent=2, sort_keys=True))
                lines.append("```")
            rca = str(flashpoint.get("rca_text") or "").strip()
            if rca:
                lines.append("")
                lines.append("**RCA:**")
                lines.append("")
                lines.append("> " + rca.replace("\n", "\n> "))
            intents = flashpoint.get("intents") or []
            if isinstance(intents, list) and intents:
                lines.append("")
                lines.append("**Robustness emitted:**")
                lines.append("")
                for intent in intents:
                    if not isinstance(intent, dict):
                        continue
                    lines.append(
                        f"- `{intent.get('intent_type', '?')}` "
                        f"→ `{intent.get('payload', {})}`"
                    )
        lines.append("")

        # ----- Findings catalogue -----
        lines.append("## Findings catalogue")
        lines.append("")
        if not findings:
            lines.append("_No findings recorded._")
        else:
            high = [f for f in findings if str(f.get("severity")) == "high"]
            medium = [f for f in findings if str(f.get("severity")) == "medium"]
            low = [f for f in findings if str(f.get("severity")) == "low"]
            lines.append(
                f"Totals: HIGH={len(high)} / MEDIUM={len(medium)} / LOW={len(low)}"
            )
            lines.append("")
            # Render the most recent N HIGH-severity for the operator;
            # MEDIUM/LOW go to decision_trace.json (full corpus).
            ordered = sorted(
                high, key=lambda f: f.get("tick_index") or 0, reverse=True,
            )[: cfg.max_findings_in_report]
            if ordered:
                lines.append("**HIGH findings (most recent first):**")
                lines.append("")
                for f in ordered:
                    lines.append(
                        f"- `tick={f.get('tick_index','?')}` "
                        f"`{f.get('symptom_name','?')}` — "
                        f"{str(f.get('summary',''))[:200]}"
                    )
        lines.append("")

        # ----- Decision-trace summary -----
        lines.append("## Decision-trace summary")
        lines.append("")
        tasks_by_action = decision_trace.get("tasks_by_action") or {}
        if not tasks_by_action:
            lines.append(
                "_No ``runs/<action>/<task_id>/result.json`` found. "
                "The Coordinator did not persist decision artefacts._"
            )
        else:
            lines.append("| action | tasks | KEEP | REVERT | other |")
            lines.append("|---|---:|---:|---:|---:|")
            for action, tasks in sorted(tasks_by_action.items()):
                keep = sum(
                    1 for t in tasks
                    if isinstance(t, dict) and str(t.get("decision") or "") == "KEEP"
                )
                revert = sum(
                    1 for t in tasks
                    if isinstance(t, dict) and str(t.get("decision") or "") == "REVERT"
                )
                other = len(tasks) - keep - revert
                lines.append(
                    f"| `{action}` | {len(tasks)} | {keep} | {revert} | {other} |"
                )
        lines.append("")
        lines.append(
            "_See `decision_trace.json` for the full per-task ledger._"
        )
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _fallback_postmortem_md(self, *, stop_reason: str) -> str:
        """Render a minimal postmortem when full rendering fails.

        Args:
            stop_reason (str): The session's stop reason.

        Returns:
            str: A short markdown stub pointing the reader to the logs.
        """
        return (
            f"# Robustness postmortem — `{self.session_id}`\n\n"
            f"stop_reason: `{stop_reason}`\n\n"
            f"_(finalizer encountered an error rendering the full body; "
            f"see logs)_\n"
        )

    # ------------------------------------------------------------------
    # write helpers
    # ------------------------------------------------------------------
    def _write_text(self, filename: str, body: str) -> bool:
        """Write a text file into the reports directory.

        Args:
            filename (str): Name of the file under :attr:`reports_dir`.
            body (str): Text content to write.

        Returns:
            bool: True on success; False when the write raised ``OSError``.
        """
        target = self.reports_dir / filename
        try:
            target.write_text(body, encoding="utf-8")
            return True
        except OSError as exc:
            log.warning(
                "postmortem: cannot write %s: %s", target, exc,
            )
            return False

    def _write_json(self, filename: str, payload: dict[str, Any]) -> bool:
        """Write a JSON file into the reports directory.

        Args:
            filename (str): Name of the file under :attr:`reports_dir`.
            payload (dict[str, Any]): JSON-serialisable content to write.

        Returns:
            bool: True on success; False when serialisation or the write
            failed.
        """
        target = self.reports_dir / filename
        try:
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return True
        except (OSError, TypeError, ValueError) as exc:
            log.warning(
                "postmortem: cannot write %s: %s", target, exc,
            )
            return False

    def _write_marker(self, *, stop_reason: str) -> None:
        """Write the idempotency marker recording this finalization.

        Failures are logged and swallowed; a missing marker only risks a
        redundant re-run on resume, never data loss.

        Args:
            stop_reason (str): The session's stop reason to record.
        """
        marker = self.marker_path
        try:
            marker.write_text(
                json.dumps(
                    {
                        "stop_reason": stop_reason,
                        "session_id": self.session_id,
                        "finalized_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(timespec="seconds"),
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("postmortem: cannot write marker %s: %s", marker, exc)


# ---------------------------------------------------------------------------
# Standalone entry point for the operator CLI / post-hoc runs
# ---------------------------------------------------------------------------

def finalize_session(
    session_dir: Path,
    *,
    session_id: str,
    stop_reason: str = "manual_finalize",
    config: PostmortemFinalizerConfig | None = None,
) -> bool:
    """Convenience wrapper for non-reactor callers (e.g. post-hoc re-runs).

    Args:
        session_dir: Session directory to finalize.
        session_id: Identifier of the session.
        stop_reason: Reason recorded for finalization.
        config: Optional finalizer configuration.

    Returns:
        The :meth:`PostmortemFinalizer.finalize` boolean.
    """
    finalizer = PostmortemFinalizer(
        session_dir=session_dir,
        session_id=session_id,
        config=config,
    )
    return finalizer.finalize(stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_flashpoint(
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the earliest HIGH-severity finding as the flashpoint.

    Ordered by ``tick_index`` then ``timestamp_unix`` (tie-break).

    Args:
        findings: Candidate finding dicts.

    Returns:
        The first HIGH-severity finding, or ``None`` if none crossed HIGH.
    """
    high = [
        f for f in findings
        if isinstance(f, dict) and str(f.get("severity")) == "high"
    ]
    if not high:
        return None
    high.sort(
        key=lambda f: (
            int(f.get("tick_index") or 0),
            float(f.get("timestamp_unix") or 0.0),
        )
    )
    return high[0]


def _normalise_task_entry(
    task_dir: Path,
    result_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Project an executor ``result.json`` into the trace entry shape.

    Captures the union of action-specific fields (e.g. ``output_throughput``
    vs ``gain_pct``) so dashboards need not re-read each file.

    Args:
        task_dir: The task's workspace directory.
        result_path: Path to the task's ``result.json``.
        payload: Parsed result payload.

    Returns:
        A normalized trace entry dict.
    """
    entry: dict[str, Any] = {
        "task_id": task_dir.name,
        "workspace": str(task_dir),
        "result_path": str(result_path),
        "decision": payload.get("decision"),
        "status": payload.get("status"),
        "error_class": payload.get("error_class"),
        "ts": payload.get("ts"),
    }
    # Common executor outputs — only include when non-None to keep
    # the JSON narrow.
    for key in (
        "gain_pct", "validated_gain_pct", "output_throughput",
        "base_tput", "new_tput", "kernel_id", "patch_path",
        "report_path", "variant_name",
    ):
        if key in payload and payload[key] is not None:
            entry[key] = payload[key]
    return entry


__all__ = [
    "PostmortemFinalizer",
    "PostmortemFinalizerConfig",
    "finalize_session",
]
