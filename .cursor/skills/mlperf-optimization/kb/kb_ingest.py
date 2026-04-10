#!/usr/bin/env python3
"""Ingest a new entry into the MLPerf optimization knowledge base.

Usage:
    python3 kb_ingest.py --entry '{"category":"fusion_flags", ...}'
    python3 kb_ingest.py --file entry.json
    python3 kb_ingest.py \
        --category fusion_flags \
        --model "GPT-OSS-20B" \
        --action "moe_permute_fusion=true" \
        --lesson "Already enabled by default in GPT-OSS-20B config" \
        --tags mlperf,moe,permute,fusion \
        --gain 0 --status baseline
"""

import argparse
import json
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


def ingest(entry: dict, force: bool = False) -> dict:
    validate(entry)
    existing = load_entries()

    conflicts = detect_conflict(entry, existing)
    if conflicts and not force:
        for conflict in conflicts:
            res = resolve_conflict(entry, conflict)
            log_conflict(res)
            print(f"  {res['action']}: {res['reason']}")

    append_entry(entry)
    return {
        "status": "ingested",
        "id": entry["id"],
        "conflicts": len(conflicts),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest a MLPerf optimization KB entry")
    parser.add_argument("--entry", help="JSON string of the full entry")
    parser.add_argument("--file", help="Path to JSON file with the entry")
    parser.add_argument("--category", help="Entry category")
    parser.add_argument("--model", default="GPT-OSS-20B")
    parser.add_argument("--gpu", default="MI355X")
    parser.add_argument("--framework", default="primus")
    parser.add_argument("--action", help="What was done")
    parser.add_argument("--lesson", help="Key takeaway")
    parser.add_argument("--tags", default="")
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--status", default="")
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--source", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--force", action="store_true")
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
