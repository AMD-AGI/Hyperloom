#!/usr/bin/env python3
"""Ingest a new entry into the knowledge base with conflict detection.

Usage:
    # From JSON string
    python3 kb_ingest.py --entry '{"category":"backend_exploration", ...}'

    # From JSON file
    python3 kb_ingest.py --file entry.json

    # Minimal entry (auto-fills id, timestamp, defaults)
    python3 kb_ingest.py \
        --category backend_exploration \
        --model "GLM-5-FP8" \
        --action "enable --nsa-decode-backend aiter" \
        --lesson "Switches NSA decode from tilelang to CK, +3.1% on GLM-5" \
        --tags nsa,aiter,decode-backend \
        --gain 3.1 --status KEEP
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from kb_schema import new_entry, validate, detect_conflict, resolve_conflict

KB_FILE = SCRIPT_DIR / "entries.jsonl"
CONFLICTS_FILE = SCRIPT_DIR / "conflicts.jsonl"


def load_entries() -> list[dict]:
    if not KB_FILE.exists():
        return []
    entries = []
    for line in KB_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def append_entry(entry: dict) -> None:
    with open(KB_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_conflict(resolution: dict) -> None:
    with open(CONFLICTS_FILE, "a") as f:
        f.write(json.dumps(resolution, ensure_ascii=False) + "\n")


def update_entry_confidence(entry_id: str, new_confidence: float) -> None:
    """Update an existing entry's confidence in-place."""
    if not KB_FILE.exists():
        return
    lines = KB_FILE.read_text().splitlines()
    updated = []
    for line in lines:
        if not line.strip():
            continue
        e = json.loads(line)
        if e["id"] == entry_id:
            e["confidence"] = new_confidence
        updated.append(json.dumps(e, ensure_ascii=False))
    KB_FILE.write_text("\n".join(updated) + "\n")


def ingest(entry: dict, force: bool = False) -> dict:
    """Ingest an entry, handling conflicts. Returns status dict."""
    validate(entry)
    existing = load_entries()

    conflicts = detect_conflict(entry, existing)
    if conflicts and not force:
        resolutions = []
        for conflict in conflicts:
            res = resolve_conflict(entry, conflict)
            resolutions.append(res)
            log_conflict(res)

            if res["action"] == "SUPERSEDE":
                update_entry_confidence(
                    conflict["existing_entry"]["id"],
                    conflict["existing_entry"]["confidence"],
                )
                print(f"  SUPERSEDE: old entry {res['old_id'][:8]}... "
                      f"confidence reduced. Reason: {res['reason']}")
            elif res["action"] == "KEEP_OLD":
                print(f"  KEEP_OLD: new entry confidence reduced. "
                      f"Reason: {res['reason']}")
            elif res["action"] == "FLAG_REVIEW":
                print(f"  FLAG_REVIEW: conflict with {res['old_id'][:8]}... "
                      f"Reason: {res['reason']}")
            elif res["action"] == "KEEP_BOTH":
                print(f"  KEEP_BOTH: {res['reason']}")

    append_entry(entry)
    return {
        "status": "ingested",
        "id": entry["id"],
        "conflicts": len(conflicts),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest a knowledge base entry")
    parser.add_argument("--entry", help="JSON string of the full entry")
    parser.add_argument("--file", help="Path to JSON file with the entry")
    parser.add_argument("--category", help="Entry category")
    parser.add_argument("--model", default="", help="Model name")
    parser.add_argument("--gpu", default="MI355X", help="GPU type")
    parser.add_argument("--framework", default="", help="Framework (sglang/vllm)")
    parser.add_argument("--action", help="What was done")
    parser.add_argument("--lesson", help="Key takeaway")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--gain", type=float, default=None, help="Throughput gain (percent)")
    parser.add_argument("--status", default="", help="KEEP/REVERT/DISCARD")
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--source", default="", help="Source identifier")
    parser.add_argument("--context", default="", help="Additional context")
    parser.add_argument("--force", action="store_true", help="Skip conflict check")
    args = parser.parse_args()

    if args.entry:
        entry_data = json.loads(args.entry)
        if "id" not in entry_data:
            entry = new_entry(**entry_data)
        else:
            validate(entry_data)
            entry = entry_data
    elif args.file:
        entry_data = json.loads(Path(args.file).read_text())
        if "id" not in entry_data:
            entry = new_entry(**entry_data)
        else:
            validate(entry_data)
            entry = entry_data
    elif args.category and args.action and args.lesson:
        result = {}
        if args.gain is not None:
            result["gain_pct"] = args.gain
        if args.status:
            result["status"] = args.status
        entry = new_entry(
            category=args.category,
            model=args.model,
            gpu=args.gpu,
            framework=args.framework,
            action=args.action,
            lesson=args.lesson,
            tags=[t.strip() for t in args.tags.split(",") if t.strip()],
            result=result,
            confidence=args.confidence,
            source=args.source,
            context=args.context,
        )
    else:
        parser.error("Provide --entry JSON, --file path, or --category/--action/--lesson")

    result = ingest(entry, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
