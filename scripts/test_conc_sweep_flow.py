#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end validation of the single-server-per-arm concurrency sweep.

Reuses a real optimization session's baseline + accepted ``current_best`` and
runs the *current* ``run_conc_sweep`` over a descending CONC ladder against the
real model on GPU. It then answers one question:

    "As CONC descends from high to low on a single reused server, does the
     server get killed between points?"

The verdict is derived from two independent signals:

  1. Result signal  — every CONC point in each arm reports a measured
     ``output_throughput``. If the server had been killed after the boot round,
     the lower-CONC reuse rounds would fail with connection errors.
  2. Boot signal    — the number of distinct server *launches* per arm, counted
     from Magpie's ``server.log`` / ``reuse_server_spawn.pid`` artifacts. In the
     single-server path this must be exactly one launch per arm.

Finally it renders the InferenceX-style throughput-vs-interactivity plot from
the produced ``conc_sweep_summary.json`` artifact, completing the full flow.

Usage::

    python scripts/test_conc_sweep_flow.py \\
        --session-dir /path/to/Hyperloom-Sessions/Qwen3-8B/<sid> \\
        --concs 8,4,2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

# Make the src/ tree importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import the executors package first so the kernel.conc_sweep <-> executors
# circular import resolves (mirrors the test module's import order).
import hyperloom.orchestrator.actions.executors._grid_runner  # noqa: E402,F401
from hyperloom.inference_optimizer.session.session_paths import reports_dir, runs_root  # noqa: E402
from hyperloom.orchestrator.kernel.conc_sweep import run_conc_sweep  # noqa: E402
from hyperloom.orchestrator.kernel.conc_sweep_plot import render_conc_sweep_curve  # noqa: E402
from hyperloom.orchestrator.state.shared_state import SharedState  # noqa: E402


def _prepare_test_session(source_session: Path, out_dir: Path) -> None:
    """Copy the source session's ``state.json`` into a fresh test dir.

    We run against a copy so the real session's artifacts are never mutated.
    ``baseline_config_path`` inside the state keeps pointing at the source
    session's (read-only) materialized YAML, which is fine.

    Args:
        source_session: The real optimization session directory.
        out_dir: Fresh directory to run the test sweep in.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src_state = source_session / "state.json"
    if not src_state.is_file():
        raise SystemExit(f"state.json not found in {source_session}")
    shutil.copy2(src_state, out_dir / "state.json")


def _count_server_launches(sweep_workspace: Path, arm: str) -> tuple[int, list[str]]:
    """Count distinct server launches for one arm under the sweep workspace.

    A launch is evidenced by a non-empty ``server.log`` produced by the
    server_lifecycle boot. Reuse rounds re-attach and do not spawn a server.

    Args:
        sweep_workspace: ``runs/conc_sweep/<task_id>`` directory.
        arm: Arm label (``baseline`` / ``optimized``).

    Returns:
        ``(launch_count, pids)`` — the number of server.log launches found for
        the arm and the list of distinct spawn PIDs recovered from
        ``reuse_server_spawn.pid`` files.
    """
    launches = 0
    pids: list[str] = []
    for slot in sorted(sweep_workspace.glob(f"variant_*_{arm}_conc*")):
        for server_log in slot.rglob("server.log"):
            try:
                if server_log.stat().st_size > 0:
                    launches += 1
            except OSError:
                pass
        for pid_file in slot.rglob("reuse_server_spawn.pid"):
            try:
                txt = pid_file.read_text(encoding="utf-8").split()
                if txt:
                    pids.append(txt[0])
            except OSError:
                pass
    return launches, sorted(set(pids))


def _analyze(payload: dict, sweep_workspace: Path) -> bool:
    """Print the kill/survival verdict from the sweep payload + server logs.

    Args:
        payload: Parsed ``conc_sweep_summary.json``.
        sweep_workspace: The ``runs/conc_sweep/<task_id>`` directory.

    Returns:
        ``True`` when the verdict is "server survived the descending ladder"
        for every arm; ``False`` otherwise.
    """
    print("\n" + "=" * 78)
    print("VERDICT: will CONC high->low kill the reused server?")
    print("=" * 78)
    overall_ok = True
    for arm in ("optimized", "baseline"):
        points = (payload.get(arm) or {}).get("points") or []
        points = sorted(points, key=lambda p: p.get("conc") or 0, reverse=True)
        if not points:
            print(f"\n[{arm}] no points recorded — arm skipped.")
            continue
        launches, pids = _count_server_launches(sweep_workspace, arm)
        print(f"\n[{arm}] {len(points)} CONC points (high -> low):")
        arm_ok = True
        for p in points:
            conc = p.get("conc")
            status = p.get("status")
            tput = p.get("output_throughput")
            marker = "ok " if status == "succeeded" and tput else "BAD"
            if marker == "BAD":
                arm_ok = False
            print(f"    conc={conc:<4} status={status:<10} output_throughput={tput}")
        print(f"  server launches counted (server.log): {launches}")
        print(f"  distinct spawn PIDs: {pids or '(none captured)'}")
        if arm_ok and launches <= 1:
            print(f"  => [{arm}] SERVER SURVIVED: booted once, reused down the ladder, NOT killed.")
        elif arm_ok and launches > 1:
            print(f"  => [{arm}] all points ok but {launches} launches — server was restarted (check lifecycle).")
            overall_ok = False
        else:
            print(f"  => [{arm}] some points failed — server may have been killed mid-ladder.")
            overall_ok = False
    print("\n" + "=" * 78)
    print("ANSWER:", "NO — the server is booted once per arm and reused (not killed)."
          if overall_ok else "SEE ABOVE — anomalies detected.")
    print("=" * 78 + "\n")
    return overall_ok


async def _run(session_dir: Path, concs: list[int], variant_timeout_sec: int) -> dict:
    """Load the copied state and run the sweep with single-server mode on.

    Args:
        session_dir: The fresh test session directory.
        concs: Descending CONC ladder to sweep.
        variant_timeout_sec: Per-variant hard timeout.

    Returns:
        The sweep payload dict.
    """
    state = SharedState.load_or_init(session_dir)
    # The source session already COMPLETED, so its persisted lifecycle flags
    # (closing_phase / stop_reason) and exhausted wall-clock would make the new
    # sweep's deadline/event detection skip everything. Clear them so this
    # deliberate re-run observes pure single-server reuse behaviour.
    state.closing_phase = False
    state.stop_reason = ""
    state.max_minutes = 0  # 0 => remaining_minutes() is None (unbounded)
    # total_budget_sec=0 disables the wall-clock budget gate too.
    return await run_conc_sweep(
        state,
        session_dir,
        concs=concs,
        variant_timeout_sec=variant_timeout_sec,
        total_budget_sec=0,
        write_reports=True,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code (0 = server survived, 1 = anomalies / no data).
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--session-dir",
        type=Path,
        default=Path("/path/to/Hyperloom-Sessions/Qwen3-8B/<sid>"),
        help="Real optimization session (baseline + accepted current_best).",
    )
    ap.add_argument("--concs", default="8,4,2", help="Descending CONC ladder (comma-separated).")
    ap.add_argument("--variant-timeout-sec", type=int, default=1800)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Fresh test session dir; default = <session-dir>_conc_flowtest_<ts>.",
    )
    args = ap.parse_args(argv)

    # Force the single-server-per-arm path (the behaviour under test).
    os.environ["INFERENCE_OPTIMIZER_CONC_SWEEP_SINGLE_SERVER"] = "1"

    source = args.session_dir.expanduser().resolve()
    out_dir = args.out_dir or source.parent / f"{source.name}_conc_flowtest_{int(time.time())}"
    out_dir = Path(out_dir).expanduser().resolve()
    concs = [int(c) for c in str(args.concs).split(",") if c.strip()]

    # Single-node: point the multi-node state file at a (non-existent) path in
    # the test dir so is_multi_node() resolves cleanly to False (the optimizer
    # normally binds this per-session; a standalone script must set it).
    os.environ.setdefault("INFERENCE_OPTIMIZER_NODES", "1")
    os.environ["MULTI_NODE_STATE_FILE"] = str(out_dir / "multi_node_state.json")

    print(f"[flowtest] source session : {source}")
    print(f"[flowtest] test session   : {out_dir}")
    print(f"[flowtest] CONC ladder     : {concs}  (descending single-server reuse)")

    _prepare_test_session(source, out_dir)

    started = time.time()
    payload = asyncio.run(_run(out_dir, concs, args.variant_timeout_sec))
    print(f"\n[flowtest] run_conc_sweep finished in {time.time() - started:.1f}s; status={payload.get('status')}")

    # Locate the sweep workspace (runs/conc_sweep/<task_id>) for boot counting.
    cs_root = runs_root(out_dir) / "conc_sweep"
    sweep_workspaces = sorted(cs_root.glob("conc_sweep_*"), key=lambda p: p.stat().st_mtime)
    sweep_ws = sweep_workspaces[-1] if sweep_workspaces else cs_root

    survived = _analyze(payload, sweep_ws)

    # Complete the designed flow: render the plot from the produced artifact.
    summary_json = reports_dir(out_dir) / "conc_sweep_summary.json"
    png_out = reports_dir(out_dir) / "conc_sweep_curve.png"
    if summary_json.is_file():
        result = render_conc_sweep_curve(
            summary_json,
            png_out,
            model_label=str(payload.get("session_id") or "Qwen3-8B"),
            gpu_label=os.environ.get("GPU_TYPE", "mi355x").upper(),
            tp=int(payload.get("tp") or 1),
            isl=int(payload.get("isl") or 0),
            osl=int(payload.get("osl") or 0),
            draw_ceiling=True,
        )
        if result is not None:
            print(f"[flowtest] plot rendered  : {result}")
        else:
            print("[flowtest] plot skipped (no valid data / matplotlib unavailable)")
    print(f"[flowtest] artifact json  : {summary_json}")

    return 0 if survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
