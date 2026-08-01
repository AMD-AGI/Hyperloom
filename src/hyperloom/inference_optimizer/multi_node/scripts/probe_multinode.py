"""Report whether every server process launch_multinode.py spawned is still alive.

Read-only counterpart to ``kill_multinode.py``: same node fan-out, same PID
files, but it only ever asks ``os.kill(pid, 0)``. It exists to tell a cold start
apart from a dead cluster, which no job status or HTTP probe can do -- a server
still loading weights answers nothing yet but its process is there, while a
crashed one is gone.

Prints one JSON summary to stdout::

    {"nodes": 2, "total": 2, "alive": 2, "dead": 0, "per_node": {...}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Actor budget: a PID check is a syscall, so this only has to cover Ray
# scheduling the actor onto its node.
_ACTOR_TIMEOUT_S = 120


def _log(msg: str) -> None:
    """Write a progress line to stderr, keeping stdout pure JSON.

    Args:
        msg: Message to log.
    """
    sys.stderr.write(f"[probe_multinode] {msg}\n")
    sys.stderr.flush()


def _server_pid_files(pid_dir: str) -> list[Path]:
    """The PID files that stand for a serving process on this pod.

    ``rank_*_rayjoin.pid`` is excluded: that helper joins the Ray cluster and
    outlives nothing in particular, so counting it would report a dead cluster
    as alive. ``router*.pid`` is excluded too -- in PD the router is submitted
    only after the launch driver returns, so its absence is the normal state
    during the window this probe exists to describe.

    Args:
        pid_dir: Directory the launcher wrote PID files into.

    Returns:
        list[Path]: Sorted PID file paths, empty when the directory is absent.
    """
    root = Path(pid_dir)
    if not root.is_dir():
        return []
    ranks = [p for p in root.glob("rank_*.pid") if not p.name.endswith("_rayjoin.pid")]
    return sorted(ranks + list(root.glob("prefill_*.pid")) + list(root.glob("decode_*.pid")))


def _probe_remote(pid_dir: str) -> dict[str, Any]:
    """Check every recorded server PID on this node.

    A PID file whose process is gone counts as dead; an unreadable or malformed
    one counts as dead too, so a half-written directory can never be mistaken
    for a healthy cluster.

    Args:
        pid_dir: Directory containing the PID files.

    Returns:
        dict[str, Any]: ``{"alive": [...], "dead": [...]}`` keyed by file name.
    """
    alive: list[str] = []
    dead: list[str] = []
    for pid_file in _server_pid_files(pid_dir):
        try:
            raw = pid_file.read_text(encoding="utf-8").strip()
        except OSError:
            dead.append(pid_file.name)
            continue
        if not raw.isdigit() or int(raw) <= 0:
            dead.append(pid_file.name)
            continue
        try:
            os.kill(int(raw), 0)
        except OSError:
            dead.append(pid_file.name)
        else:
            alive.append(pid_file.name)
    return {"alive": alive, "dead": dead}


def main() -> int:
    """Fan out the PID check across every alive node and print the aggregate.

    Returns:
        int: 0 whenever the probe itself ran; the verdict is in the JSON, since
        a cluster with no servers is a valid answer rather than a probe failure.
    """
    parser = argparse.ArgumentParser(
        prog="probe_multinode.py",
        description="Report whether every multi-node server process is still alive.",
    )
    parser.add_argument(
        "--pid-dir",
        required=True,
        help="dir containing rank_*.pid files (same value passed to launch_multinode)",
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    _log(f"pid_dir={args.pid_dir} alive nodes: {len(nodes)}")

    ProbeActor = ray.remote(num_cpus=0, num_gpus=0)(_probe_remote)
    refs = []
    for node in nodes:
        node_id = node["NodeID"]
        ref = ProbeActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(args.pid_dir)
        refs.append((node_id[:16], ref))

    per_node: dict[str, dict] = {}
    alive = 0
    dead = 0
    for short_id, ref in refs:
        try:
            result = ray.get(ref, timeout=_ACTOR_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            # An unreachable node is not evidence of life on it.
            _log(f"node {short_id}: probe FAILED: {type(exc).__name__}: {exc}")
            per_node[short_id] = {"error": str(exc)}
            dead += 1
            continue
        per_node[short_id] = result
        alive += len(result.get("alive") or [])
        dead += len(result.get("dead") or [])

    summary = {
        "nodes": len(nodes),
        "total": alive + dead,
        "alive": alive,
        "dead": dead,
        "per_node": per_node,
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
