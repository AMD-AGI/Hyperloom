# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workload sweep that relaunches the GEAK-optimized server.

When the KERNEL_AGENT phase is delegated to GEAK
(``KERNEL_OPT_BACKEND_ORDER=geak``), the optimized server is reproduced
from GEAK' own ``bench_e2e.sh`` plus the built overlay/flags/env recorded
in ``result.json``. Each grid point relaunches the optimized server through
``bench_e2e.sh`` (same per-variant-server semantics as the native sweep),
benches at ``(CONC, ISL, OSL)``, and parses ``bench_summary.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.common.jsonio import read_json
from ._grid_base import best_entry_for_each_conc, select_sweep_pareto

log = logging.getLogger(__name__)


def _write_benchmark_report(
    out_dir: Path,
    *,
    conc: int,
    isl: int,
    osl: int,
    success: bool,
    output_throughput_tok_s: float | None,
    mean_ttft_ms: float | None,
    mean_tpot_ms: float | None,
    mean_e2el_ms: float | None,
    error: str | None = None,
) -> None:
    """Write a session-breakdown-compatible ``benchmark_report.json``.

    Field names match what ``breakdown.collectors._benchmark_report_metrics``
    parses (flat ``output_throughput_tok_s`` / ``mean_ttft_ms`` /
    ``mean_tpot_ms`` / ``mean_e2el_ms``) and ``success`` drives the per-variant
    status, so the geak sweep points are auditable through the exact same
    collector path as the native sweep. Best-effort: never raises.
    """
    report = {
        "success": bool(success),
        "conc": conc,
        "isl": isl,
        "osl": osl,
        "output_throughput_tok_s": output_throughput_tok_s,
        "mean_ttft_ms": mean_ttft_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "mean_e2el_ms": mean_e2el_ms,
        "source": "geak",
    }
    if error:
        report["error"] = error
    try:
        (out_dir / "benchmark_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        # Best-effort: a failed write must never break the sweep.
        log.warning("geak_sweep: could not write %s: %s", out_dir / "benchmark_report.json", exc)


def _serving_gpus(tp: int) -> str:
    return ",".join(str(i) for i in range(max(tp, 1)))


def _parse_isl_osl(spec: str) -> tuple[int, int]:
    isl_s, _, osl_s = str(spec).partition(":")
    return int(isl_s or 1024), int(osl_s or 1024)


async def sweep_via_geak(
    *,
    result: dict[str, Any],
    conc_values: list[int],
    isl_osl_configs: list[str],
    output_root: Path,
    variant_timeout_sec: int,
    repeats: int = 3,
    pin_num_prompts: bool = False,
    framework: str | None = None,
    state: Any = None,
) -> dict[str, Any]:
    """Run a CONC × (ISL, OSL) sweep on the GEAK-optimized server.

    Args:
        pin_num_prompts: When True, also forward ``num_prompts`` from the
            protocol onto every point (NUM_PROMPTS). Off by default because a
            multi-conc sweep's prompt count is tied to each concurrency (a fixed
            count would mis-size other concs); a single-point validated replay
            sets it True so the replay matches the headline result's exact
            protocol instead of bench_e2e.sh's per-conc default.
    """
    bench_script = result.get("bench_script") or result.get("geak_bench_script")
    overlay = result.get("final_overlay") or ""
    cfg = result.get("accepted_config") or {}
    flags = str(cfg.get("flags") or "")
    env_str = str(cfg.get("env") or "")

    if not bench_script or not Path(bench_script).is_file():
        return {
            "status": "failed",
            "error_class": "missing_bench_script",
            "error": f"GEAK bench script not found: {bench_script}",
        }

    model = os.environ.get("MODEL_PATH", "").strip()
    backend = (os.environ.get("FRAMEWORK", "") or "sglang").strip()
    tp = int(os.environ.get("TP", "1") or 1)
    gpus = _serving_gpus(tp)
    # Reuse the SAME bench client the KERNEL_AGENT phase measured with.
    bench_client = str(result.get("bench_client") or "native").strip() or "native"

    # Forward the validated measurement config + client trust onto every variant
    # so the sweep measures on the same workload shape the KERNEL_AGENT phase
    # accepted (else bench_e2e.sh falls back to its own defaults). Prefer an
    # explicit bench_protocol block, else the first validated regime. Only
    # concurrency-independent knobs are forwarded; num_prompts is left to
    # bench_e2e.sh's per-conc default.
    _protocol = result.get("bench_protocol")
    if not isinstance(_protocol, dict):
        _regimes = result.get("validated_regimes") or []
        _protocol = _regimes[0] if _regimes and isinstance(_regimes[0], dict) else {}
    protocol_env: dict[str, str] = {}
    _protocol_map = [
        ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        ("num_warmups", "NUM_WARMUPS"),
        ("seed", "SEED"),
    ]
    # Single-point validated replay: also pin num_prompts (see docstring).
    if pin_num_prompts:
        _protocol_map.append(("num_prompts", "NUM_PROMPTS"))
    for _src, _dst in _protocol_map:
        _val = _protocol.get(_src)
        if _val is not None:
            protocol_env[_dst] = str(_val)
    # Mirror the server's --trust-remote-code onto the bench client so its
    # tokenizer load doesn't raise. Keyed on the flags, never a model name.
    if "trust-remote-code" in flags or "trust_remote_code" in flags:
        for _tk in ("BENCH_TRUST_REMOTE_CODE", "HF_HUB_TRUST_REMOTE_CODE", "MAGPIE_TRUST_REMOTE_CODE"):
            protocol_env.setdefault(_tk, "1")

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    variant_idx = 0
    for conc in conc_values:
        for spec in isl_osl_configs:
            isl, osl = _parse_isl_osl(spec)
            # Name matches the sweep collector's scanner regex
            # (``variant_<idx>_conc<c>_isl<i>_osl<o>``) so it is discovered.
            variant_name = f"variant_{variant_idx}_conc{conc}_isl{isl}_osl{osl}"
            variant_idx += 1
            out_dir = output_root / variant_name
            out_dir.mkdir(parents=True, exist_ok=True)
            env = build_benchmark_env(
                {
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
                }
            )
            # setdefault: forwarded config/trust apply unless already pinned.
            for _k, _v in protocol_env.items():
                env.setdefault(_k, _v)
            cmd = ["bash", str(bench_script)]

            def _run() -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=variant_timeout_sec,
                    env=env,
                    cwd=str(out_dir),
                )

            entry: dict[str, Any] = {
                "conc": conc,
                "isl": isl,
                "osl": osl,
                "variant_name": variant_name,
                "workspace": str(out_dir),
            }
            ttft = tpot = e2el = None
            tput = None
            succeeded = False
            err: str | None = None
            try:
                proc = await asyncio.to_thread(_run)
                summ = read_json(out_dir / "bench_summary.json", default={}, require_dict=True)
                tput = summ.get("output_throughput_tok_s_median")
                ttft = summ.get("ttft_ms_median")
                tpot = summ.get("tpot_ms_median")
                e2el = summ.get("e2el_ms_median")
                total_tput = summ.get("total_token_throughput")
                if total_tput is None:
                    total_tput = summ.get("total_throughput_tok_s_median")
                intvty = summ.get("intvty_p90_tok_s_user")
                if intvty is None:
                    intvty = summ.get("intvty_p90")
                input_tput = summ.get("input_throughput")
                if input_tput is None:
                    input_tput = summ.get("input_throughput_tok_s_median")
                if proc.returncode == 0 and isinstance(tput, (int, float)) and tput > 0:
                    succeeded = True
                    entry.update(
                        {
                            "status": "succeeded",
                            "output_throughput": tput,
                            "ttft_mean_ms": ttft,
                            "tpot_mean_ms": tpot,
                        }
                    )
                    if isinstance(total_tput, (int, float)) and total_tput > 0:
                        entry["total_token_throughput"] = float(total_tput)
                    if isinstance(intvty, (int, float)) and intvty > 0:
                        entry["intvty_p90"] = float(intvty)
                    if isinstance(input_tput, (int, float)) and input_tput > 0:
                        entry["input_throughput"] = float(input_tput)
                else:
                    err = (proc.stderr or "")[-500:] or "no throughput"
                    entry.update({"status": "failed", "error": err})
            except Exception as exc:  # noqa: BLE001
                err = repr(exc)
                entry.update({"status": "failed", "error": err})

            # Emit a session-breakdown-compatible benchmark_report.json so the
            # sweep collector parses this point like the native sweep;
            # bench_summary.json is kept as the raw artifact.
            _write_benchmark_report(
                out_dir,
                conc=conc,
                isl=isl,
                osl=osl,
                success=succeeded,
                output_throughput_tok_s=tput if isinstance(tput, (int, float)) else None,
                mean_ttft_ms=ttft if isinstance(ttft, (int, float)) else None,
                mean_tpot_ms=tpot if isinstance(tpot, (int, float)) else None,
                mean_e2el_ms=e2el if isinstance(e2el, (int, float)) else None,
                error=err,
            )
            entries.append(entry)

    front = select_sweep_pareto(
        entries,
        framework=framework,
        fallback_latency_key="ttft_mean_ms",
    )
    best_for_each_conc = best_entry_for_each_conc(
        entries,
        framework=framework,
        state=state,
    )

    succeeded = [e for e in entries if e["status"] == "succeeded"]
    return {
        "status": "succeeded" if succeeded else "failed",
        "grid_size": len(entries),
        "sweep_grid": entries,
        "pareto_front": front,
        "best_for_each_conc": best_for_each_conc,
        "workspace": output_root.as_posix(),
        "source": "geak",
    }
