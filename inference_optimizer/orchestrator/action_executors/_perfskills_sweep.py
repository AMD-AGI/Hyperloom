"""Workload sweep that relaunches the PerfSkills-optimized server.

When the KERNEL phase is delegated to PerfSkills
(``KERNEL_OPT_BACKEND_ORDER=perfskills``), the optimized server is reproduced
from PerfSkills' own ``bench_e2e.sh`` plus the built overlay/flags/env recorded
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

log = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # ValueError covers json.JSONDecodeError; an absent vs malformed file
        # both degrade to {} but are now distinguishable in debug logs.
        log.debug("_read_json: could not read %s: %s", path, exc)
        return {}


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
    status, so the perfskills sweep points are auditable through the exact same
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
        "source": "perfskills",
    }
    if error:
        report["error"] = error
    try:
        (out_dir / "benchmark_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
    except OSError as exc:
        # Best-effort reporting: a failed benchmark_report.json write must never
        # break the sweep, so log and continue instead of propagating.
        log.warning("perfskills_sweep: could not write %s: %s",
                    out_dir / "benchmark_report.json", exc)


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

    # ── Forward the validated measurement 口径 + client trust onto every variant ──
    # The sweep must measure on the SAME workload shape the KERNEL phase
    # accepted, otherwise bench_e2e.sh falls back to its own standalone defaults
    # (notably RANDOM_RANGE_RATIO=1 => fixed full-length prompts) and the grid
    # is no longer 口径-comparable to the baseline/final. Prefer an explicit
    # bench_protocol block, else the first validated regime. Only the
    # concurrency-INDEPENDENT knobs are forwarded; num_prompts is deliberately
    # left to bench_e2e.sh's per-conc default because the regime's count is tied
    # to its own concurrency (forwarding a fixed count would mis-size other concs).
    _protocol = result.get("bench_protocol")
    if not isinstance(_protocol, dict):
        _regimes = result.get("validated_regimes") or []
        _protocol = _regimes[0] if _regimes and isinstance(_regimes[0], dict) else {}
    protocol_env: dict[str, str] = {}
    for _src, _dst in (
        ("random_range_ratio", "RANDOM_RANGE_RATIO"),
        ("num_warmups", "NUM_WARMUPS"),
        ("seed", "SEED"),
    ):
        _val = _protocol.get(_src)
        if _val is not None:
            protocol_env[_dst] = str(_val)
    # Mirror the server's --trust-remote-code onto the bench CLIENT: the accepted
    # flags launch a custom-tokenizer server and bench_e2e.sh's client loads the
    # same tokenizer (transformers raises ValueError without trust). Model-
    # agnostic — keyed on the flags, never on a model name.
    if "trust-remote-code" in flags or "trust_remote_code" in flags:
        for _tk in ("BENCH_TRUST_REMOTE_CODE", "HF_HUB_TRUST_REMOTE_CODE", "MAGPIE_TRUST_REMOTE_CODE"):
            protocol_env.setdefault(_tk, "1")

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    variant_idx = 0
    for conc in conc_values:
        for spec in isl_osl_configs:
            isl, osl = _parse_isl_osl(spec)
            # Name the per-point dir so the session-breakdown sweep collector's
            # on-disk scanner (_scan_sweep_variants / _VARIANT_NAME_RE expects
            # ``variant_<idx>_conc<c>_isl<i>_osl<o>``) discovers it and populates
            # ``sweep.all_variants`` with per-variant detail.
            variant_name = f"variant_{variant_idx}_conc{conc}_isl{isl}_osl{osl}"
            variant_idx += 1
            out_dir = output_root / variant_name
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
            # setdefault: the forwarded 口径/trust apply unless the operator
            # already pinned the knob in the process env (explicit wins).
            for _k, _v in protocol_env.items():
                env.setdefault(_k, _v)
            cmd = ["bash", str(bench_script)]

            def _run() -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=variant_timeout_sec, env=env, cwd=str(out_dir),
                )

            entry: dict[str, Any] = {"conc": conc, "isl": isl, "osl": osl,
                                     "variant_name": variant_name,
                                     "workspace": str(out_dir)}
            ttft = tpot = e2el = None
            tput = None
            succeeded = False
            err: str | None = None
            try:
                proc = await asyncio.to_thread(_run)
                summ = _read_json(out_dir / "bench_summary.json")
                tput = summ.get("output_throughput_tok_s_median")
                ttft = summ.get("ttft_ms_median")
                tpot = summ.get("tpot_ms_median")
                e2el = summ.get("e2el_ms_median")
                if proc.returncode == 0 and isinstance(tput, (int, float)) and tput > 0:
                    succeeded = True
                    entry.update({
                        "status": "succeeded",
                        "output_throughput": tput,
                        "ttft_mean_ms": ttft,
                        "tpot_mean_ms": tpot,
                    })
                else:
                    err = (proc.stderr or "")[-500:] or "no throughput"
                    entry.update({"status": "failed", "error": err})
            except Exception as exc:  # noqa: BLE001
                err = repr(exc)
                entry.update({"status": "failed", "error": err})

            # Emit a session-breakdown-compatible benchmark_report.json so the
            # sweep collector parses this point with the SAME 口径 as the native
            # sweep (output_throughput_tok_s / mean_ttft_ms / mean_tpot_ms /
            # mean_e2el_ms). bench_summary.json is kept as the raw artifact.
            _write_benchmark_report(
                out_dir, conc=conc, isl=isl, osl=osl, success=succeeded,
                output_throughput_tok_s=tput if isinstance(tput, (int, float)) else None,
                mean_ttft_ms=ttft if isinstance(ttft, (int, float)) else None,
                mean_tpot_ms=tpot if isinstance(tpot, (int, float)) else None,
                mean_e2el_ms=e2el if isinstance(e2el, (int, float)) else None,
                error=err,
            )
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
