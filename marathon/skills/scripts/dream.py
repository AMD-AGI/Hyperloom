#!/usr/bin/env python3
"""Dream — KB consolidation engine for cross-run learning.

Usage:
    python3 dream.py consolidate --kb-path KB --model MODEL [--run-id RUN_ID] [--output OUT]
    python3 dream.py prune --kb-path KB [--min-confidence 0.3] [--dry-run]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kb"))
from kb_schema import text_similarity, detect_conflict, resolve_conflict


def load_entries(kb_path: str) -> list[dict]:
    entries = []
    with open(kb_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def atomic_write_entries(kb_path: str, entries: list[dict]) -> None:
    """Write entries atomically — write to temp, then rename."""
    dir_path = os.path.dirname(os.path.abspath(kb_path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp_path, kb_path)
    except Exception:
        os.unlink(tmp_path)
        raise


def group_entries(entries: list[dict]) -> dict:
    """Group entries by (model, category, action_pattern)."""
    groups = defaultdict(list)
    for entry in entries:
        model = entry.get("model", "")
        category = entry.get("category", "")
        action_words = entry.get("action", "").lower().split()[:3]
        pattern = " ".join(action_words)
        key = (model, category, pattern)
        groups[key].append(entry)
    return dict(groups)


def find_contradictions(group: list[dict]) -> list[tuple[dict, dict, str]]:
    """Find contradictions within a group of related entries."""
    contradictions = []
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            a_gain = a.get("result", {}).get("gain_pct")
            b_gain = b.get("result", {}).get("gain_pct")
            a_status = a.get("result", {}).get("status", "")
            b_status = b.get("result", {}).get("status", "")

            if a_status and b_status and a_status != b_status:
                contradictions.append((a, b, f"status: {a_status} vs {b_status}"))
            elif a_gain is not None and b_gain is not None and abs(a_gain - b_gain) > 5.0:
                contradictions.append((a, b, f"gain: {a_gain}% vs {b_gain}%"))
    return contradictions


def merge_duplicates(group: list[dict], sim_threshold: float = 0.8) -> list[dict]:
    """Merge near-duplicate entries, keeping the higher-confidence one."""
    if len(group) <= 1:
        return group

    merged = []
    skip_ids = set()

    for i, a in enumerate(group):
        if a["id"] in skip_ids:
            continue
        best = a
        for b in group[i + 1:]:
            if b["id"] in skip_ids:
                continue
            sim = text_similarity(a.get("lesson", ""), b.get("lesson", ""))
            if sim >= sim_threshold:
                if b.get("confidence", 0.9) > best.get("confidence", 0.9):
                    skip_ids.add(best["id"])
                    best = b
                else:
                    skip_ids.add(b["id"])
                    best["confidence"] = min(1.0, best.get("confidence", 0.9) + 0.05)
        merged.append(best)

    return merged


def consolidate(kb_path: str, model: str, run_id: str = "", output: str = "") -> dict:
    """Run full consolidation: group, detect contradictions, merge, resolve."""
    entries = load_entries(kb_path)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "run_id": run_id,
        "total_entries_before": len(entries),
        "contradictions_found": 0,
        "contradictions_resolved": 0,
        "duplicates_merged": 0,
        "entries_updated": 0,
    }

    model_entries = [e for e in entries if e.get("model", "") == model or not model]
    other_entries = [e for e in entries if e not in model_entries]

    groups = group_entries(model_entries)

    resolved_entries = []
    for key, group in groups.items():
        contradictions = find_contradictions(group)
        report["contradictions_found"] += len(contradictions)

        for a, b, reason in contradictions:
            conflict = {"existing_entry": a, "similarity": 0.9, "reason": reason}
            resolution = resolve_conflict(b, conflict)
            report["contradictions_resolved"] += 1

        merged = merge_duplicates(group)
        report["duplicates_merged"] += len(group) - len(merged)
        resolved_entries.extend(merged)

    all_entries = other_entries + resolved_entries
    report["total_entries_after"] = len(all_entries)
    report["entries_updated"] = report["total_entries_before"] - report["total_entries_after"]

    atomic_write_entries(kb_path, all_entries)

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)

    return report


def prune(kb_path: str, min_confidence: float = 0.3, dry_run: bool = True) -> dict:
    """Remove entries below confidence threshold that have been superseded."""
    entries = load_entries(kb_path)
    superseded_ids = {e.get("supersedes") for e in entries if e.get("supersedes")}

    to_prune = []
    to_keep = []
    for entry in entries:
        if (entry["id"] in superseded_ids and
                entry.get("confidence", 0.9) < min_confidence):
            to_prune.append(entry)
        else:
            to_keep.append(entry)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(entries),
        "pruned": len(to_prune),
        "kept": len(to_keep),
        "dry_run": dry_run,
        "pruned_ids": [e["id"] for e in to_prune],
    }

    if not dry_run and to_prune:
        atomic_write_entries(kb_path, to_keep)

    return report


def main():
    parser = argparse.ArgumentParser(description="Dream — KB consolidation engine")
    sub = parser.add_subparsers(dest="command", required=True)

    cons = sub.add_parser("consolidate")
    cons.add_argument("--kb-path", required=True)
    cons.add_argument("--model", required=True)
    cons.add_argument("--run-id", default="")
    cons.add_argument("--output", default="")

    pr = sub.add_parser("prune")
    pr.add_argument("--kb-path", required=True)
    pr.add_argument("--min-confidence", type=float, default=0.3)
    pr.add_argument("--dry-run", action="store_true", default=True)
    pr.add_argument("--apply", action="store_true", help="Actually prune (disables dry-run)")

    args = parser.parse_args()

    if args.command == "consolidate":
        report = consolidate(args.kb_path, args.model, args.run_id, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "prune":
        dry_run = not args.apply
        report = prune(args.kb_path, args.min_confidence, dry_run)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
