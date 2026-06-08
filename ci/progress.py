#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ci/progress.py — promote / dedup batch results.

After a batch finishes, this script promotes Succeeded models from the
batch's ``ci_summary.json`` into ``ci/candidates/already_done.json`` so the
next batch dispatch automatically skips them.

It also supports listing pending candidates for retry or for sanity-checking
how many of the candidates pool remain unrun.

Commands:
  promote <ci_summary.json> [--already-done <path>] [--write]
      Read ci_summary.json (from build_summary.py output), extract
      Succeeded rows + Failed rows, merge into already_done.json.
      Without --write, prints what would change to stdout (dry-run).

  list-remaining <candidates.json> [--already-done <path>]
      Print repo_ids in candidates.json that are NOT in already_done.json.
      Useful for retry batches.

  stats [--already-done <path>] [--candidates <path>]
      Summarize already_done.json (counts by status, success rate).

The schema produced by ``promote`` matches the existing already_done.json:
  {"models": [{"repo_id", "status", "framework", "precision", "tp",
               "params_b", "gain_pct"?, "vs_infx_pct"?, "phase"?, ...}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_ALREADY = Path(__file__).parent / "candidates" / "already_done.json"


def _load_json(path: Path) -> dict:
    """Load a JSON object from disk, tolerating missing or invalid files.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: The parsed object, or an empty dict if the file is missing or
        cannot be parsed (a warning is printed to stderr in the latter case).
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARN: could not parse {path}: {e}", file=sys.stderr)
        return {}


def _summary_rows(summary_path: Path) -> list[dict]:
    """Extract rows from a ci_summary.json (whatever shape build_summary uses).

    Accepts both `{"rows": [...]}` and `{"models": [...]}` and a bare list.

    Args:
        summary_path (Path): Path to the ci_summary.json file.

    Returns:
        list[dict]: The extracted row dicts, or an empty list if none are found.
    """
    data = _load_json(summary_path)
    if isinstance(data, list):
        return data
    return data.get("rows") or data.get("models") or []


def _classify_status(row: dict) -> tuple[str, str | None]:
    """Return (status, reason) for an already_done entry.

    Status taxonomy:
      completed — task hit a final phase AND we have baseline+optimized
      partial   — task ran but only optimized (or only baseline) recorded
      failed    — submit/sandbox/register error, no usable data at all

    Args:
        row (dict): A summary row with status and throughput fields.

    Returns:
        tuple[str, str | None]: The status label and an optional reason string
        explaining a ``partial``/``failed`` classification.
    """
    fs = row.get("final_status")
    submit = row.get("submit_status")
    has_opt = row.get("optimized_tok_per_gpu") is not None
    has_baseline = row.get("baseline_tok_per_gpu") is not None
    has_gain = row.get("gain_pct") is not None

    if has_baseline and has_opt and has_gain:
        return "completed", None
    if has_opt or has_baseline:
        return "partial", "single-data point only (no gain comparable)"
    if fs in ("Failed", "Canceled", "Cancelled"):
        return "failed", f"final_status={fs}"
    if submit in ("failed", "error"):
        return "failed", f"submit_status={submit}"
    return "failed", "no data produced"


def _row_to_entry(row: dict) -> dict:
    """Convert a summary row into an already_done.json entry.

    Args:
        row (dict): A summary row from a ci_summary.json file.

    Returns:
        dict: An entry with ``repo_id``, ``status``, framework/precision/tp
        fields, and optional ``reason``, ``gain_pct``, ``vs_infx_pct``, and
        ``task_id`` keys.
    """
    status, reason = _classify_status(row)
    e = {
        "repo_id":   row.get("model"),
        "status":    status,
        "framework": row.get("framework"),
        "precision": row.get("precision"),
        "tp":        row.get("tp"),
        "params_b":  row.get("params_b"),
    }
    if reason:
        e["reason"] = reason
    if row.get("gain_pct") is not None:
        e["gain_pct"] = round(row["gain_pct"], 2)
    if row.get("vs_inferenceX_pct") is not None:
        e["vs_infx_pct"] = round(row["vs_inferenceX_pct"], 2)
    if row.get("task_id"):
        e["task_id"] = row["task_id"]
    return e


def cmd_promote(args: argparse.Namespace) -> int:
    """Promote batch summary rows into already_done.json.

    Merges new entries and upgrades existing ones (never downgrading
    completed > partial > failed). Without ``--write`` the planned changes are
    printed as a dry-run.

    Args:
        args (argparse.Namespace): Parsed CLI args with ``summary``,
            ``already_done``, and ``write`` attributes.

    Returns:
        int: ``0`` on success or no-op, ``1`` if the summary contains no rows.
    """
    summary = _summary_rows(Path(args.summary))
    if not summary:
        print(f"no rows in {args.summary}", file=sys.stderr)
        return 1

    already_path = Path(args.already_done)
    already = _load_json(already_path)
    existing = {m["repo_id"]: m for m in already.get("models", [])}

    new_entries: list[dict] = []
    updates: list[dict] = []
    for row in summary:
        repo = row.get("model")
        if not repo:
            continue
        entry = _row_to_entry(row)
        if repo in existing:
            old = existing[repo]
            # Don't downgrade: completed > partial > failed
            rank = {"completed": 0, "partial": 1, "failed": 2}
            if rank.get(entry["status"], 9) < rank.get(old.get("status", "failed"), 9):
                updates.append({"old": old, "new": entry})
                existing[repo] = entry
        else:
            new_entries.append(entry)

    if not new_entries and not updates:
        print("nothing to promote (all batch rows already in already_done)")
        return 0

    print(f"=== promote summary ({Path(args.summary).name}) ===")
    print(f"new entries:  {len(new_entries)}")
    print(f"upgrades:     {len(updates)}")
    print()
    if new_entries:
        print("New:")
        for e in new_entries[:15]:
            print(f"  + {e['status']:9s} {e['repo_id']:60s} "
                  f"{(e.get('gain_pct') or '—'):>7}")
        if len(new_entries) > 15:
            print(f"  ... and {len(new_entries) - 15} more")
    if updates:
        print("Upgrades:")
        for u in updates[:10]:
            print(f"  ~ {u['old']['status']} → {u['new']['status']}: "
                  f"{u['new']['repo_id']}")

    if not args.write:
        print("\n(dry-run; pass --write to persist)")
        return 0

    merged_models = list(existing.values()) + new_entries
    meta = already.get("_meta", {})
    meta["last_promoted_at"] = datetime.utcnow().isoformat() + "Z"
    meta["last_promoted_from"] = str(args.summary)
    meta["count"] = len(merged_models)

    already_path.parent.mkdir(parents=True, exist_ok=True)
    already_path.write_text(
        json.dumps({"_meta": meta, "models": merged_models}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {len(merged_models)} models to {already_path}")
    return 0


def cmd_list_remaining(args: argparse.Namespace) -> int:
    """Print candidate repo_ids that are not yet in already_done.json.

    Args:
        args (argparse.Namespace): Parsed CLI args with ``candidates`` and
            ``already_done`` attributes.

    Returns:
        int: ``0`` on success, ``1`` if the candidates file has no entries.
    """
    cands = _load_json(Path(args.candidates))
    cand_repos = [c["repo_id"] for c in cands.get("candidates", [])]
    if not cand_repos:
        print(f"no candidates in {args.candidates}", file=sys.stderr)
        return 1
    already = _load_json(Path(args.already_done))
    done = {m["repo_id"] for m in already.get("models", [])}

    remaining = [r for r in cand_repos if r not in done]
    print(f"# {len(remaining)} remaining of {len(cand_repos)} candidates "
          f"(already_done excludes {len(done)})")
    for r in remaining:
        print(r)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Summarize already_done.json counts and gain statistics.

    Prints totals by status, framework, and precision, plus gain stats, and
    optionally compares against a candidates pool.

    Args:
        args (argparse.Namespace): Parsed CLI args with ``already_done`` and
            optional ``candidates`` attributes.

    Returns:
        int: Always ``0``.
    """
    already = _load_json(Path(args.already_done))
    models = already.get("models", [])
    by_status: dict[str, int] = {}
    by_framework: dict[str, int] = {}
    by_precision: dict[str, int] = {}
    gains = []
    for m in models:
        s = m.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
        if m.get("framework"):
            by_framework[m["framework"]] = by_framework.get(m["framework"], 0) + 1
        if m.get("precision"):
            by_precision[m["precision"]] = by_precision.get(m["precision"], 0) + 1
        if m.get("gain_pct") is not None:
            gains.append(m["gain_pct"])

    print(f"=== already_done.json stats ({args.already_done}) ===")
    print(f"Total models: {len(models)}")
    print(f"By status:    {by_status}")
    print(f"By framework: {by_framework}")
    print(f"By precision: {by_precision}")
    if gains:
        gains.sort(reverse=True)
        avg = sum(gains) / len(gains)
        print(f"With gain%:   {len(gains)} (avg {avg:+.2f}%, "
              f"top {gains[0]:+.2f}%, beat-0%: {sum(1 for g in gains if g > 0)})")

    if args.candidates:
        cands = _load_json(Path(args.candidates))
        cand_repos = {c["repo_id"] for c in cands.get("candidates", [])}
        done_repos = {m["repo_id"] for m in models}
        intersect = cand_repos & done_repos
        remaining = cand_repos - done_repos
        print(f"\n=== vs candidates pool ({args.candidates}) ===")
        print(f"Pool size:    {len(cand_repos)}")
        print(f"Already done: {len(intersect)} (= {len(intersect)*100/len(cand_repos):.0f}%)")
        print(f"Remaining:    {len(remaining)}")
    return 0


def main() -> int:
    """Parse CLI arguments and dispatch to the selected subcommand.

    Supports the ``promote``, ``list-remaining``, and ``stats`` subcommands.

    Returns:
        int: The exit code returned by the dispatched subcommand.
    """
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("promote", help="merge ci_summary.json into already_done.json")
    pp.add_argument("summary", help="path to ci_summary.json")
    pp.add_argument("--already-done", default=str(DEFAULT_ALREADY))
    pp.add_argument("--write", action="store_true",
                    help="actually persist changes (without it, dry-run only)")

    pl = sub.add_parser("list-remaining",
                        help="print candidates not yet in already_done")
    pl.add_argument("candidates", help="path to candidates JSON")
    pl.add_argument("--already-done", default=str(DEFAULT_ALREADY))

    ps = sub.add_parser("stats", help="counters from already_done.json")
    ps.add_argument("--already-done", default=str(DEFAULT_ALREADY))
    ps.add_argument("--candidates", default=None,
                    help="optional pool to compare against")

    args = p.parse_args()

    if args.cmd == "promote":
        return cmd_promote(args)
    if args.cmd == "list-remaining":
        return cmd_list_remaining(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
