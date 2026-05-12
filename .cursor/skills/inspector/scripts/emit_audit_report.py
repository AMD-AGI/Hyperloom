#!/usr/bin/env python3
"""emit_audit_report.py

Implicit-mode emitter for inspector audits. Combines the verdict JSON produced
by `compute_verdict.py` with the audit_window / observations / next_checkpoint
metadata, writes the full audit_report.json to disk under
`$RESULT_DIR/.audit/<PHASE>_<ts>.json`, updates the sentinel state file
`$RESULT_DIR/.audit/_state.json` (used by `find_transcript.py` to compute the
next audit window), and prints exactly **one** line on stdout that the agent
echoes verbatim into the chat.

The on-disk audit_report.json is the single canonical artifact. The chat ack
is the minimal user-visible footprint that prevents the user from believing
"no audit happened" without exposing the full audit machinery.

Inputs:
  --verdict-json     Path to compute_verdict.py stdout (the canonical verdict
                     block). Required.
  --observations     Path to /tmp/inspector_obs_<PHASE>.json (S5a output).
                     Required; embedded into observations field.
  --manifest         Path to /tmp/inspector_manifest_<PHASE>.json. Optional;
                     used to compute extraction_diagnostics if present.
  --result-dir       Path to $RESULT_DIR (the run's result directory). The
                     audit reports go to <result-dir>/.audit/. Required.
  --transcript-path  Absolute transcript JSONL path (from find_transcript.py).
                     Required; recorded in audit_window.
  --audit-from-line  First line of audit window. Required.
  --audit-to-line    Last line of audit window. Required.
  --target-skill-dir   Path to the audited skill (echoed into report). Required.
  --phase-action-files Comma-separated relative paths under target-skill-dir.
                       Required; written into phase_action_files[].
  --next-phase       Symbolic next phase name (or empty for terminal phase).
                     Required.

Outputs:
  - Writes <result-dir>/.audit/<PHASE>_<utc_ts>.json (the full audit_report).
  - Updates <result-dir>/.audit/_state.json (sentinel; see schema in
    audit-report-schema.md §4).
  - Prints exactly one line to stdout, e.g.
      [Inspection] phase=BASELINE verdict=PASS passes=8 unverified=2  -> /shared/.audit/BASELINE_2026-04-21T10-34-00Z.json
    The agent echoes this verbatim. No other stdout output.
  - Prints diagnostic lines to stderr only on error (non-fatal informational
    notes also go to stderr).

Stdlib only. Exits non-zero on hard errors (missing inputs, malformed JSON,
RESULT_DIR not writable).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any


SENTINEL_HISTORY_CAP = 50  # keep last N audits in sentinel; older are pruned


def _read_json(path: str) -> Any:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"emit_audit_report: input not found: {path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"emit_audit_report: invalid JSON in {path}: {e}")


def _utc_ts_filename() -> str:
    """ISO-8601 with `Z` suffix and `:` replaced by `-` for filesystem safety."""
    now = _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z").replace(":", "-")


def _utc_ts_iso() -> str:
    now = _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _build_extraction_diagnostics(manifest: dict | None) -> dict:
    if not isinstance(manifest, dict):
        return {
            "candidates_from_regex": 0,
            "candidates_kept_after_classification": 0,
            "modality_promotions": 0,
            "modality_demotions": 0,
            "regex_anchors_diff_summary": "manifest not provided",
        }
    diag = manifest.get("extraction_diagnostics")
    if isinstance(diag, dict):
        return diag
    # Best-effort fallback: count buckets.
    total = (len(manifest.get("expected_tool_calls", []))
             + len(manifest.get("expected_artifacts", []))
             + len(manifest.get("expected_state_assertions", [])))
    return {
        "candidates_from_regex": total,
        "candidates_kept_after_classification": total,
        "modality_promotions": 0,
        "modality_demotions": 0,
        "regex_anchors_diff_summary": "diagnostics not present in manifest",
    }


def _short_path(p: str | Path, limit: int = 80) -> str:
    s = str(p)
    if len(s) <= limit:
        return s
    return "..." + s[-(limit - 3):]


def _ack_line(verdict: str, phase: str, summary: str,
              report_path: Path, top_violation_id: str | None) -> str:
    """One-line chat acknowledgement. Format chosen to be:
    - parseable by future tooling (stable prefix `[Inspection]` and key=value pairs)
    - terse enough that it does not dominate the chat
    - informative enough to satisfy 'user must know an audit ran'
    """
    parts = [f"[Inspection] phase={phase}", f"verdict={verdict}"]
    if summary:
        # summary is the verdict_summary like "passes=8 fatal=0 ..."; copy as-is
        parts.append(summary)
    if verdict in ("BLOCK", "FATAL") and top_violation_id:
        parts.append(f"top={top_violation_id}")
    parts.append(f"-> {_short_path(report_path)}")
    return " ".join(parts)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _update_sentinel(audit_dir: Path, *, phase: str, verdict: str,
                     transcript_path: str, audit_to_line: int,
                     report_filename: str, ts_iso: str,
                     next_phase: str | None) -> Path:
    sentinel = audit_dir / "_state.json"
    if sentinel.is_file():
        try:
            state = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    else:
        state = {}
    history = state.get("history") or []
    if not isinstance(history, list):
        history = []
    history.append({
        "phase": phase,
        "ts": ts_iso,
        "verdict": verdict,
        "to_line": audit_to_line,
        "report_file": report_filename,
    })
    if len(history) > SENTINEL_HISTORY_CAP:
        history = history[-SENTINEL_HISTORY_CAP:]
    new_state = {
        "transcript_path": transcript_path,
        "last_audit_to_line": audit_to_line,
        "last_phase": phase,
        "last_verdict": verdict,
        "last_ts": ts_iso,
        "next_phase_hint": next_phase or None,
        "history": history,
    }
    _atomic_write_json(sentinel, new_state)
    return sentinel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict-json", required=True)
    ap.add_argument("--observations", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--transcript-path", required=True)
    ap.add_argument("--audit-from-line", type=int, required=True)
    ap.add_argument("--audit-to-line", type=int, required=True)
    ap.add_argument("--target-skill-dir", required=True)
    ap.add_argument("--phase-action-files", required=True,
                    help="Comma-separated relative paths.")
    ap.add_argument("--next-phase", default="",
                    help="Empty if this was the terminal phase.")
    ap.add_argument("--phase-index", type=int, default=None)
    args = ap.parse_args()

    verdict_block = _read_json(args.verdict_json)
    observations = _read_json(args.observations)
    manifest = _read_json(args.manifest) if args.manifest else None

    if not isinstance(verdict_block, dict):
        raise SystemExit("emit_audit_report: --verdict-json must be a JSON object")

    phase = str(verdict_block.get("phase") or "UNKNOWN")
    verdict = str(verdict_block.get("verdict") or "BLOCK")
    summary = str(verdict_block.get("verdict_summary") or "")

    # Build full audit_report.json by enveloping the verdict block with
    # audit_window, observations, extraction_diagnostics, next_checkpoint, and
    # the run_env metadata if available from the manifest.
    action_files = [p for p in args.phase_action_files.split(",") if p]
    audit_window = {
        "transcript": args.transcript_path,
        "from_line": args.audit_from_line,
        "to_line": args.audit_to_line,
    }
    next_phase = args.next_phase.strip() or None
    if next_phase:
        next_checkpoint = {
            "should_invoke_inspector_after": next_phase,
            "reminder_text": (
                f"After completing {next_phase} phase, run the inspector "
                f"audit again."
            ),
        }
    else:
        next_checkpoint = {
            "should_invoke_inspector_after": None,
            "reminder_text": "Run complete. No further inspector audits required.",
        }

    ts_iso = _utc_ts_iso()
    report = {
        **verdict_block,
        "phase_action_files": action_files,
        "phase_index": args.phase_index,
        "target_skill": args.target_skill_dir,
        "audited_at_utc": ts_iso,
        "audit_window": audit_window,
        "run_env_resolved": (manifest or {}).get("run_env_resolved", {}),
        "run_env_unresolved": (manifest or {}).get("run_env_unresolved", []),
        "extraction_diagnostics": _build_extraction_diagnostics(manifest),
        "observations": observations,
        "next_checkpoint": next_checkpoint,
    }

    audit_dir = Path(args.result_dir) / ".audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"{phase}_{_utc_ts_filename()}.json"
    report_path = audit_dir / report_filename
    _atomic_write_json(report_path, report)

    _update_sentinel(
        audit_dir,
        phase=phase, verdict=verdict,
        transcript_path=args.transcript_path,
        audit_to_line=args.audit_to_line,
        report_filename=report_filename, ts_iso=ts_iso,
        next_phase=next_phase,
    )

    top_violation = None
    violations = verdict_block.get("violations") or []
    if isinstance(violations, list) and violations:
        # Pick the highest-severity violation as the headline; ties broken by
        # array order so the report stays stable.
        sev_rank = {"info": 0, "warn": 1, "block": 2, "fatal": 3}
        v_sorted = sorted(
            (v for v in violations if isinstance(v, dict)),
            key=lambda v: -sev_rank.get(str(v.get("severity", "info")), 0),
        )
        if v_sorted:
            top_violation = str(v_sorted[0].get("id") or "")

    sys.stdout.write(_ack_line(verdict, phase, summary, report_path, top_violation) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
