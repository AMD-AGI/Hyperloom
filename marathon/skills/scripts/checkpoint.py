#!/usr/bin/env python3
"""Checkpoint — state save/load with atomic writes for multi-day optimization runs.

Usage:
    python3 checkpoint.py save --state-json STATE --output OUT [--metadata KEY=VAL]
    python3 checkpoint.py restore --checkpoint CKPT --output STATE
    python3 checkpoint.py list --checkpoint-dir DIR
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_write(path: str, data: dict) -> None:
    """Write JSON atomically — temp file + rename."""
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def save_checkpoint(state_json: str, output: str, metadata: str = "") -> dict:
    """Save state to a checkpoint file."""
    with open(state_json) as f:
        state = json.load(f)

    meta = {}
    if metadata:
        for pair in metadata.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                meta[k.strip()] = v.strip()

    checkpoint = {
        "version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": meta,
        "state": state,
    }

    atomic_write(output, checkpoint)

    latest_link = os.path.join(os.path.dirname(output), "latest.json")
    atomic_write(latest_link, checkpoint)

    return {
        "saved": output,
        "timestamp": checkpoint["timestamp"],
        "state_keys": list(state.keys()),
        "metadata": meta,
    }


def restore_checkpoint(checkpoint_path: str, output: str) -> dict:
    """Restore state from a checkpoint file."""
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    version = checkpoint.get("version", 1)
    state = checkpoint.get("state", {})

    if not state:
        raise ValueError(f"Checkpoint {checkpoint_path} has no state data")

    atomic_write(output, state)

    return {
        "restored_from": checkpoint_path,
        "version": version,
        "timestamp": checkpoint.get("timestamp", "unknown"),
        "metadata": checkpoint.get("metadata", {}),
        "state_keys": list(state.keys()),
    }


def list_checkpoints(checkpoint_dir: str) -> list[dict]:
    """List all checkpoints in directory, sorted by timestamp."""
    checkpoints = []
    dir_path = Path(checkpoint_dir)
    if not dir_path.exists():
        return []

    for f in dir_path.glob("checkpoint_*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            checkpoints.append({
                "path": str(f),
                "timestamp": data.get("timestamp", "unknown"),
                "metadata": data.get("metadata", {}),
                "size_kb": f.stat().st_size / 1024,
            })
        except (json.JSONDecodeError, IOError):
            checkpoints.append({
                "path": str(f),
                "timestamp": "CORRUPT",
                "metadata": {},
                "size_kb": f.stat().st_size / 1024 if f.exists() else 0,
            })

    return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint — state persistence")
    sub = parser.add_subparsers(dest="command", required=True)

    save = sub.add_parser("save")
    save.add_argument("--state-json", required=True, help="Path to current state.json")
    save.add_argument("--output", required=True, help="Path for checkpoint file")
    save.add_argument("--metadata", default="", help="Comma-separated key=value pairs")

    restore = sub.add_parser("restore")
    restore.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    restore.add_argument("--output", required=True, help="Path to write restored state.json")

    ls = sub.add_parser("list")
    ls.add_argument("--checkpoint-dir", required=True, help="Directory containing checkpoints")

    args = parser.parse_args()

    if args.command == "save":
        result = save_checkpoint(args.state_json, args.output, args.metadata)
        print(json.dumps(result, indent=2))
    elif args.command == "restore":
        result = restore_checkpoint(args.checkpoint, args.output)
        print(json.dumps(result, indent=2))
    elif args.command == "list":
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print("No checkpoints found.")
        else:
            for ckpt in checkpoints:
                meta_str = ", ".join(f"{k}={v}" for k, v in ckpt["metadata"].items())
                print(f"  {ckpt['timestamp']}  {ckpt['size_kb']:.1f}KB  {meta_str}  {ckpt['path']}")


if __name__ == "__main__":
    main()
