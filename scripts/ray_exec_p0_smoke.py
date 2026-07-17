#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P0 acceptance smoke for Ray-managed GPU execution (ray_modify.plan.md §6 P0).

Demonstrates the make-or-break invariant (§4.2) on a real Ray cluster:

  1. A ServingActor holding num_gpus occupies the GPUs while alive.
  2. A second GPU actor requesting a GPU is *pending* (queued) until the
     serving actor releases — proving Ray serializes GPU contention.
  3. Killing the serving actor reaps its subprocess tree (PR_SET_PDEATHSIG) —
     no detached GPU process escapes the lease.

Runs against an already-running Ray cluster (does not start/stop one). Uses a
harmless ``sleep`` as the "GPU process" so no real model is loaded — this
validates Ray resource accounting + process lifetime, not throughput.

Usage:  python scripts/ray_exec_p0_smoke.py [--gpus N]
Exit 0 = all invariants held.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Resolve circular import (executors package) the same way the tests do.
import hyperloom.orchestrator.actions.executors._grid_runner  # noqa: E402,F401
from hyperloom.orchestrator.actions.executors._ray_backend import get_ray_backend  # noqa: E402
from hyperloom.orchestrator.actions.executors._ray_serving import (  # noqa: E402
    make_gpu_specialist_actor,
    make_serving_actor,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpus", type=int, default=0, help="GPUs for the serving actor (0 = all available).")
    args = ap.parse_args(argv)

    import ray

    backend = get_ray_backend()
    backend.ensure()  # connects to the existing cluster (idempotent)

    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus <= 0:
        print("[smoke] no GPUs in the Ray cluster; cannot run the invariant demo.")
        return 1
    serving_gpus = args.gpus or total_gpus
    print(f"[smoke] cluster GPUs={total_gpus}; serving actor will hold num_gpus={serving_gpus}")

    ok = True
    serving = None
    specialist = None
    server_pid = None
    try:
        # (1) Serving actor takes the GPUs and launches a long-lived process.
        serving = make_serving_actor(num_gpus=serving_gpus, serving_slot=False)
        server_pid = ray.get(serving.start.remote(["sleep", "120"]), timeout=60)
        assert ray.get(serving.is_alive.remote(), timeout=10)
        print(f"[smoke] serving actor up; server pid={server_pid}, alive={_pid_alive(server_pid)}")

        time.sleep(2)
        avail = int(ray.available_resources().get("GPU", 0))
        print(f"[smoke] available GPUs after serving lease: {avail} (expected ~0)")
        if avail >= serving_gpus:
            print("[smoke] FAIL: serving actor did not consume GPU resources")
            ok = False

        # (2) A second GPU actor must be PENDING while serving holds the GPUs.
        specialist = make_gpu_specialist_actor(num_gpus=1)
        ready, pending = ray.wait([specialist.pid.remote()], timeout=8)
        if pending and not ready:
            print("[smoke] PASS: gpu-specialist actor is PENDING (queued) while serving holds GPUs")
        else:
            print("[smoke] FAIL: specialist ran despite no free GPU (共卡!)")
            ok = False

        # (3) Kill the serving actor -> subprocess must be reaped (no escape).
        ray.kill(serving)
        serving = None
        deadline = time.time() + 15.0
        while time.time() < deadline and _pid_alive(server_pid):
            time.sleep(0.2)
        if _pid_alive(server_pid):
            print(f"[smoke] FAIL: server pid={server_pid} survived actor kill (detached GPU proc escaped!)")
            ok = False
        else:
            print(f"[smoke] PASS: server pid={server_pid} reaped after actor kill")

        # (4) With serving gone, the specialist should now schedule.
        got = ray.get(specialist.pid.remote(), timeout=30)
        print(f"[smoke] PASS: specialist scheduled after serving released (actor pid method -> {got})")
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] ERROR: {exc!r}")
        ok = False
    finally:
        for handle in (serving, specialist):
            if handle is not None:
                try:
                    ray.kill(handle)
                except Exception:  # noqa: BLE001
                    pass
        # Belt-and-suspenders: ensure the demo sleep is gone.
        if server_pid and _pid_alive(server_pid):
            try:
                os.kill(server_pid, 9)
            except OSError:
                pass

    print("\n[smoke] RESULT:", "ALL INVARIANTS HELD" if ok else "FAILURES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
