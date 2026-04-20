#!/usr/bin/env python3
"""Fix marathon state after a Claw outage.

Adjusts start_time so dead hours don't count against the 24h budget,
and removes completed actions that were reverted or failed during the outage
so the DFS can retry them.

Usage:
    python3 fix_state_after_outage.py <session_dir> [--dry-run]
"""

import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fix state after Claw outage")
    parser.add_argument("session_dir", help="Path to session directory")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    state_path = session_dir / "state.json"
    ckpt_latest = session_dir / "checkpoints" / "latest"

    if not state_path.exists():
        print(f"ERROR: {state_path} not found")
        return 1

    # Load from latest checkpoint (the cleanest state)
    if ckpt_latest.exists():
        ckpt_path = ckpt_latest.resolve()
        print(f"Loading from checkpoint: {ckpt_path}")
        state = json.loads(ckpt_path.read_text())
    else:
        print(f"Loading from state.json (no checkpoint)")
        state = json.loads(state_path.read_text())

    now = time.time()
    old_start = state.get("start_time", now)

    # The last checkpoint timestamp = last time productive work happened
    ckpt_epoch = int(ckpt_path.name.replace("checkpoint_", "").replace(".json", ""))
    productive_seconds = ckpt_epoch - old_start
    productive_hours = productive_seconds / 3600
    dead_seconds = now - ckpt_epoch
    dead_hours = dead_seconds / 3600

    print(f"\n=== TIMING ANALYSIS ===")
    print(f"Original start:     {time.strftime('%H:%M:%S', time.localtime(old_start))}")
    print(f"Last checkpoint:    {time.strftime('%H:%M:%S', time.localtime(ckpt_epoch))}")
    print(f"Now:                {time.strftime('%H:%M:%S', time.localtime(now))}")
    print(f"Productive time:    {productive_hours:.1f}h")
    print(f"Dead time (outage): {dead_hours:.1f}h")

    # Shift start_time forward by dead_seconds so marathon sees productive_hours elapsed
    new_start = old_start + dead_seconds
    print(f"\nAdjusting start_time: {old_start:.0f} -> {new_start:.0f}")
    print(f"  Marathon will now see ~{productive_hours:.1f}h elapsed instead of {(now - old_start)/3600:.1f}h")

    state["start_time"] = new_start

    # Fix total_wall_minutes (was stuck at 0.0)
    state["total_wall_minutes"] = productive_seconds / 60
    print(f"Fixed total_wall_minutes: 0.0 -> {productive_seconds / 60:.1f}")

    # Find actions that were popped from stack during outage but got no real LLM work
    # The current marathon resume popped action_amdgcn_buffer_ops_enable from stack.
    # We'll re-check completed_actions for any with "reverted" or "pending bench" status
    completed = state.get("completed_actions", [])
    original_count = len(completed)
    keep = []
    requeue = []
    for a in completed:
        name = a.get("name", a.get("id", "?"))
        result = a.get("result", a.get("outcome", ""))
        if isinstance(result, str) and any(kw in result.lower() for kw in
                ["reverted", "pending bench", "claw error", "connection", "no output"]):
            requeue.append(a)
        else:
            keep.append(a)

    if requeue:
        print(f"\n=== REQUEUEING {len(requeue)} FAILED ACTIONS ===")
        for a in requeue:
            name = a.get("name", a.get("id", "?"))
            result = a.get("result", a.get("outcome", "?"))
            score = a.get("score", 5.0)
            print(f"  Re-adding to stack: {name} (was: {str(result)[:60]})")
            # Push back onto action stack with original score
            stack_entry = {
                "id": a.get("id", name),
                "name": name,
                "action": a.get("action", a.get("strategy", "?")),
                "score": score,
                "description": a.get("description", f"RETRY after Claw outage: {name}"),
            }
            state.setdefault("action_stack", []).append(stack_entry)

        state["completed_actions"] = keep
        # Re-sort stack by score descending
        state["action_stack"].sort(key=lambda x: x.get("score", 0), reverse=True)
        print(f"Completed: {original_count} -> {len(keep)}")
        print(f"Stack size: {len(state.get('action_stack', []))}")
    else:
        print(f"\nNo completed actions need requeueing ({original_count} all clean)")

    if args.dry_run:
        print("\n[DRY RUN] No changes written")
        return 0

    # Backup current state
    backup = state_path.with_suffix(".json.bak")
    shutil.copy2(state_path, backup)
    print(f"\nBackup: {backup}")

    # Write fixed state
    state_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"Written: {state_path}")

    # Also update the checkpoint
    ckpt_path.write_text(json.dumps(state, indent=2, default=str))
    print(f"Updated checkpoint: {ckpt_path}")

    print(f"\n=== DONE ===")
    print(f"Productive time preserved: ~{productive_hours:.1f}h")
    print(f"Time budget remaining:     ~{24 - productive_hours:.1f}h")
    return 0


if __name__ == "__main__":
    exit(main())
