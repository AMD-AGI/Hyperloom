#!/usr/bin/env python3
"""find_transcript.py

Locate the current Cursor agent conversation's transcript JSONL file and
compute the starting line for the next inspector audit window.

Layout assumed (per Cursor 2026-04 builds):
    ~/.cursor/projects/<project-slug>/agent-transcripts/<uuid>/<uuid>.jsonl

Project slug = absolute cwd with leading "/" removed and remaining "/" replaced
with "-". Example: cwd=/root/Hyperloom -> slug=root-Hyperloom.

Audit-window discovery:
  The on-disk sentinel `<RESULT_DIR>/.audit/_state.json` (written by
  `emit_audit_report.py`) records `last_audit_to_line` and the
  `transcript_path` it was produced against. If the sentinel exists AND its
  `transcript_path` matches the chosen transcript, the next audit starts at
  `last_audit_to_line + 1`. Otherwise (first audit of the run, or sentinel
  was produced for a different transcript) the audit starts at line 1.

Selection rule (which transcript file):
  1. Optional --marker-sentence: prefer the JSONL whose content contains the
     given sentence (matches the user prompt that started this run). If exactly
     one match -> pick it. If multiple -> fall through to mtime. If zero ->
     fall through to mtime.
  2. Otherwise pick the JSONL inside the most-recently-mtime UUID directory.

Output: a single JSON object on stdout, e.g.
  {"transcript_path": "...", "audit_from_line": 412,
   "selection_method": "mtime", "window_source": "sentinel",
   "ts": "2026-04-21T10:34:00Z"}

`window_source` is one of `sentinel` or `start_of_file` and documents which
mechanism produced `audit_from_line`.

Read-only. Stdlib only. Exits non-zero on hard errors (no transcripts dir).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path


def _slug_from_cwd() -> str:
    cwd = Path.cwd().resolve()
    return str(cwd).lstrip("/").replace("/", "-")


def _candidate_transcripts(transcripts_root: Path) -> list[Path]:
    if not transcripts_root.is_dir():
        return []
    out: list[Path] = []
    for uuid_dir in transcripts_root.iterdir():
        if not uuid_dir.is_dir():
            continue
        for jsonl in uuid_dir.glob("*.jsonl"):
            out.append(jsonl)
    return out


def _file_contains(path: Path, sentence: str, max_bytes: int = 4_000_000) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return sentence in f.read(max_bytes)
    except OSError:
        return False


def _read_sentinel(result_dir: Path) -> dict | None:
    sentinel = result_dir / ".audit" / "_state.json"
    if not sentinel.is_file():
        return None
    try:
        data = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projects-root", default=str(Path.home() / ".cursor" / "projects"),
                   help="Override Cursor projects root (for tests).")
    p.add_argument("--slug", default=None,
                   help="Override project slug (default: derived from cwd).")
    p.add_argument("--marker-sentence", default=None,
                   help="Sentence to grep inside JSONL to disambiguate parallel sessions.")
    p.add_argument("--result-dir", default=os.environ.get("RESULT_DIR"),
                   help="$RESULT_DIR for the run. The on-disk sentinel "
                        "<result-dir>/.audit/_state.json supplies the next "
                        "audit's from-line.")
    args = p.parse_args()

    slug = args.slug or _slug_from_cwd()
    transcripts_root = Path(args.projects_root) / slug / "agent-transcripts"
    candidates = _candidate_transcripts(transcripts_root)
    if not candidates:
        json.dump(
            {"error": "no_transcripts_found",
             "transcripts_root": str(transcripts_root), "slug": slug},
            sys.stdout)
        sys.stdout.write("\n")
        return 2

    selection_method = "mtime"
    chosen: Path | None = None
    if args.marker_sentence:
        matches = [c for c in candidates if _file_contains(c, args.marker_sentence)]
        if len(matches) == 1:
            chosen = matches[0]
            selection_method = "marker_sentence"
        # else fall through to mtime
    if chosen is None:
        chosen = max(candidates, key=lambda c: c.stat().st_mtime)

    audit_from_line = 1
    window_source = "start_of_file"

    if args.result_dir:
        sentinel = _read_sentinel(Path(args.result_dir))
        if sentinel:
            sent_path = sentinel.get("transcript_path")
            sent_line = sentinel.get("last_audit_to_line")
            # Only trust the sentinel if it was written for the same transcript;
            # otherwise we are likely in a different session and the line
            # numbers do not correspond. Normalise both sides to absolute paths
            # so a sentinel written with an absolute path still matches a
            # transcript resolved from a relative --projects-root.
            chosen_abs = str(chosen.resolve()) if chosen else ""
            try:
                sent_abs = str(Path(sent_path).resolve()) if sent_path else ""
            except OSError:
                sent_abs = sent_path or ""
            if (isinstance(sent_line, int)
                    and (not sent_abs or sent_abs == chosen_abs)):
                audit_from_line = sent_line + 1
                window_source = "sentinel"

    out = {
        "transcript_path": str(chosen),
        "audit_from_line": audit_from_line,
        "selection_method": selection_method,
        "window_source": window_source,
        "candidates_considered": len(candidates),
        "slug": slug,
        "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
