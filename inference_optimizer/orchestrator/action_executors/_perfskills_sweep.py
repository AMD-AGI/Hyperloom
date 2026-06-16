"""Workload sweep that REUSES PerfSkills' delivered scripts.

When the KERNEL phase was delegated to PerfSkills (``--kernel-optimizer
perfskills``), the optimized server is reproduced from PerfSkills' own
``bench_e2e.sh`` + the already-built overlay/flags/env recorded in
``result.json`` — never reconstructed by Hyperloom (this removes the overlay
reproduction risk). Each grid point relaunches the optimized server through
``bench_e2e.sh`` (same per-variant-server semantics as the native sweep),
benches at ``(CONC, ISL, OSL)``, and parses ``bench_summary.json``.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _serving_gpus(tp: int) -> str:
    return ",".join(str(i) for i in range(max(tp, 1)))


def _parse_isl_osl(spec: str) -> tuple[int, int]:
    isl_s, _, osl_s = str(spec).partition(":")
    return int(isl_s or 1024), int(osl_s or 1024)


def _pareto_front(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Max output_throughput, min ttft_mean_ms (bench_summary has no e2el)."""
    succ = [
        e for e in entries
        if e["status"] == "succeeded"
        and isinstance(e.get("output_throughput"), (int, float))
        and isinstance(e.get("ttft_mean_ms"), (int, float))
    ]
    front: list[dict[str, Any]] = []
    for cand in succ:
        dominated = False
        for other in succ:
            if other is cand:
                continue
            if (other["output_throughput"] >= cand["output_throughput"]
                    and other["ttft_mean_ms"] <= cand["ttft_mean_ms"]
                    and (other["output_throughput"] > cand["output_throughput"]
                         or other["ttft_mean_ms"] < cand["ttft_mean_ms"])):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return front


async def sweep_via_perfskills(
    *,
    result: dict[str, Any],
    conc_values: list[int],
    isl_osl_configs: list[str],
    output_root: Path,
    variant_timeout_sec: int,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run a CONC × (ISL, OSL) sweep on the PerfSkills-optimized server."""
    bench_script = result.get("bench_script") or result.get("perfskills_bench_script")
    overlay = result.get("final_overlay") or ""
    cfg = result.get("accepted_config") or {}
    flags = str(cfg.get("flags") or "")
    env_str = str(cfg.get("env") or "")

    if not bench_script or not Path(bench_script).is_file():
        return {"status": "failed", "error_class": "missing_bench_script",
                "error": f"PerfSkills bench script not found: {bench_script}"}

    model = os.environ.get("MODEL_PATH", "").strip()
    backend = (os.environ.get("FRAMEWORK", "") or "sglang").strip()
    tp = int(os.environ.get("TP", "1") or 1)
    gpus = _serving_gpus(tp)
    # Reuse the SAME bench client the KERNEL phase measured with (recorded in
    # result.json) so sweep numbers stay 口径-consistent with the headline result.
    bench_client = str(result.get("bench_client") or "native").strip() or "native"

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for conc in conc_values:
        for spec in isl_osl_configs:
            isl, osl = _parse_isl_osl(spec)
            out_dir = output_root / f"conc{conc}_isl{isl}_osl{osl}"
            out_dir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update({
                "BACKEND": backend,
                "OUT_DIR": str(out_dir),
                "GPU": gpus,
                "TP": str(tp),
                "MODEL": model,
                "ISL": str(isl),
                "OSL": str(osl),
                "CONC": str(conc),
                "REPEATS": str(repeats),
                "PROFILE": "0",
                "OVERLAY_PYTHONPATH": overlay,
                "EXTRA_SERVER_ARGS": flags,
                "EXTRA_ENV": env_str,
                "BENCH_CLIENT": bench_client,
            })
            cmd = ["bash", str(bench_script)]

            def _run() -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=variant_timeout_sec, env=env, cwd=str(out_dir),
                )

            entry: dict[str, Any] = {"conc": conc, "isl": isl, "osl": osl,
                                     "workspace": str(out_dir)}
            try:
                proc = await asyncio.to_thread(_run)
                summ = _read_json(out_dir / "bench_summary.json")
                tput = summ.get("output_throughput_tok_s_median")
                if proc.returncode == 0 and isinstance(tput, (int, float)) and tput > 0:
                    entry.update({
                        "status": "succeeded",
                        "output_throughput": tput,
                        "ttft_mean_ms": summ.get("ttft_ms_median"),
                        "tpot_mean_ms": summ.get("tpot_ms_median"),
                    })
                else:
                    entry.update({
                        "status": "failed",
                        "error": (proc.stderr or "")[-500:] or "no throughput",
                    })
            except Exception as exc:  # noqa: BLE001
                entry.update({"status": "failed", "error": repr(exc)})
            entries.append(entry)

    front = _pareto_front(entries)
    best_for_each_conc: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e["status"] != "succeeded":
            continue
        cur = best_for_each_conc.get(str(e["conc"]))
        if cur is None or (
            isinstance(e.get("output_throughput"), (int, float))
            and e["output_throughput"] > cur.get("output_throughput", 0)
        ):
            best_for_each_conc[str(e["conc"])] = e

    succeeded = [e for e in entries if e["status"] == "succeeded"]
    return {
        "status": "succeeded" if succeeded else "failed",
        "grid_size": len(entries),
        "sweep_grid": entries,
        "pareto_front": front,
        "best_for_each_conc": best_for_each_conc,
        "workspace": output_root.as_posix(),
        "source": "perfskills",
    }
