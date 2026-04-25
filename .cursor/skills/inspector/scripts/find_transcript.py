#!/usr/bin/env python3
"""find_transcript.py

Locate the current Cursor agent conversation's transcript JSONL file and find
the line number of the most recent INSPECTOR_END marker so the next audit can
scope its grep window incrementally.

Layout assumed (per Cursor 2026-04 builds):
    ~/.cursor/projects/<project-slug>/agent-transcripts/<uuid>/<uuid>.jsonl

Project slug = absolute cwd with leading "/" removed and remaining "/" replaced
with "-". Example: cwd=/root/Hyperloom -> slug=root-Hyperloom.

Selection rule:
  1. Optional --marker-sentence: prefer the JSONL whose content contains the
     given sentence (matches the user prompt that started this run). If exactly
     one match -> pick it. If multiple -> fall through to mtime. If zero ->
     fall through to mtime.
  2. Otherwise pick the JSONL inside the most-recently-mtime UUID directory.

Output: a single JSON object on stdout, e.g.
  {"transcript_path": "...", "audit_from_line": 412,
   "previous_inspector_end_line": 411, "selection_method": "mtime",
   "ts": "2026-04-21T10:34:00Z"}

Read-only. Stdlib only. Exits non-zero on hard errors (no transcripts dir).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

INSPECTOR_END_RE = re.compile(r"^=== INSPECTOR_END phase=\S+ ts=\S+ verdict=\S+ ===\s*$")


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


def _last_inspector_end_line(path: Path) -> int | None:
    last: int | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if INSPECTOR_END_RE.search(line):
                    last = i
    except OSError:
        return None
    return last


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projects-root", default=str(Path.home() / ".cursor" / "projects"),
                   help="Override Cursor projects root (for tests).")
    p.add_argument("--slug", default=None,
                   help="Override project slug (default: derived from cwd).")
    p.add_argument("--marker-sentence", default=None,
                   help="Sentence to grep inside JSONL to disambiguate parallel sessions.")
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

    last_end = _last_inspector_end_line(chosen)
    audit_from_line = (last_end + 1) if last_end is not None else 1
    out = {
        "transcript_path": str(chosen),
        "audit_from_line": audit_from_line,
        "previous_inspector_end_line": last_end,
        "selection_method": selection_method,
        "candidates_considered": len(candidates),
        "slug": slug,
        "ts": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
