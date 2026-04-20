"""profiling.py — hardware performance counter integration via rocprof.

Provides causal performance insights (memory-bound, compute-bound, register
pressure) to inform kernel optimization strategy selection.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROCPROF_COUNTERS = [
    "SQ_WAVES",
    "SQ_INSTS_VALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
    "SQ_WAIT_INST_VMEM",
    "SQ_WAIT_INST_LDS",
    "TCC_HIT",
    "TCC_MISS",
    "GRBM_GUI_ACTIVE",
]

BOTTLENECK_THRESHOLDS = {
    "memory_bound": 0.40,
    "lds_bound": 0.30,
    "compute_heavy": 0.70,
    "cache_miss_rate": 0.30,
}


@dataclass
class ProfilingResult:
    kernel_name: str = ""
    counters: dict[str, float] = field(default_factory=dict)
    bottleneck: str = "unknown"
    bottleneck_confidence: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    raw_csv_path: str = ""
    duration_s: float = 0.0


async def profile_kernel(
    host: str,
    port: int,
    output_dir: str,
    prompt_text: str = "Hello, tell me a short joke",
    max_tokens: int = 64,
    timeout_s: float = 120,
) -> ProfilingResult:
    """Run rocprof on a single inference request and analyze counters."""
    if not shutil.which("rocprof"):
        log.warning("rocprof not found — skipping hardware profiling")
        return ProfilingResult(bottleneck="unavailable")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    counters_file = out_dir / f"counters_{ts}.txt"
    counters_file.write_text("\n".join(ROCPROF_COUNTERS) + "\n")

    result_csv = out_dir / f"rocprof_{ts}.csv"
    trigger_script = out_dir / f"trigger_{ts}.py"
    import json as _json
    trigger_script.write_text(
        f'import json, requests\n'
        f'payload = json.loads({_json.dumps(_json.dumps({"text": prompt_text, "sampling_params": {"max_new_tokens": max_tokens}}))})\n'
        f'r = requests.post("http://{host}:{port}/generate", json=payload, timeout=60)\n'
        f'print(f"Status: {{r.status_code}}, length: {{len(r.text)}}")\n'
    )

    cmd = (
        f"rocprof --stats --timestamp on "
        f"-i {counters_file} -o {result_csv} "
        f"python3 {trigger_script}"
    )

    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
    )

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        log.error("rocprof timed out after %ds", timeout_s)
        return ProfilingResult(bottleneck="timeout", duration_s=timeout_s)

    duration_s = time.monotonic() - t0

    if proc.returncode != 0:
        log.error("rocprof failed (exit %d): %s",
                  proc.returncode, (stdout or b"").decode()[:500])
        return ProfilingResult(bottleneck="error", duration_s=duration_s)

    return _analyze_rocprof_csv(str(result_csv), duration_s)


def _analyze_rocprof_csv(csv_path: str, duration_s: float) -> ProfilingResult:
    """Parse rocprof CSV output and classify bottlenecks."""
    p = Path(csv_path)
    if not p.exists():
        return ProfilingResult(bottleneck="no_output", duration_s=duration_s)

    counters: dict[str, float] = {}
    try:
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ROCPROF_COUNTERS:
                    if key in row:
                        counters[key] = counters.get(key, 0) + float(row[key] or 0)
    except Exception as exc:
        log.warning("Failed to parse rocprof CSV %s: %s", csv_path, exc)
        return ProfilingResult(bottleneck="parse_error", duration_s=duration_s,
                               raw_csv_path=csv_path)

    return _classify_bottleneck(counters, csv_path, duration_s)


def _classify_bottleneck(
    counters: dict[str, float],
    csv_path: str,
    duration_s: float,
) -> ProfilingResult:
    """Classify the primary bottleneck from hardware counter data."""
    total_insts = sum(counters.get(k, 0) for k in
                      ["SQ_INSTS_VALU", "SQ_INSTS_VMEM", "SQ_INSTS_SALU",
                       "SQ_INSTS_SMEM", "SQ_INSTS_LDS"])
    if total_insts == 0:
        return ProfilingResult(counters=counters, bottleneck="no_data",
                               raw_csv_path=csv_path, duration_s=duration_s)

    vmem_frac = counters.get("SQ_INSTS_VMEM", 0) / total_insts
    valu_frac = counters.get("SQ_INSTS_VALU", 0) / total_insts
    lds_frac = counters.get("SQ_INSTS_LDS", 0) / total_insts

    tcc_hit = counters.get("TCC_HIT", 0)
    tcc_miss = counters.get("TCC_MISS", 0)
    cache_miss_rate = tcc_miss / (tcc_hit + tcc_miss) if (tcc_hit + tcc_miss) > 0 else 0

    vmem_wait = counters.get("SQ_WAIT_INST_VMEM", 0)
    lds_wait = counters.get("SQ_WAIT_INST_LDS", 0)
    waves = max(counters.get("SQ_WAVES", 1), 1)
    wait_per_wave = (vmem_wait + lds_wait) / waves

    recommendations: list[str] = []
    bottleneck = "balanced"
    confidence = 0.5

    if vmem_frac > BOTTLENECK_THRESHOLDS["memory_bound"]:
        bottleneck = "memory-bound"
        confidence = min(vmem_frac * 1.5, 0.95)
        recommendations.extend([
            "Consider data layout optimization (coalescing, padding)",
            "Try larger tile sizes to improve memory reuse",
            "Explore Interwave scheduling for better memory latency hiding",
        ])
        if cache_miss_rate > BOTTLENECK_THRESHOLDS["cache_miss_rate"]:
            recommendations.append(
                f"High L2 cache miss rate ({cache_miss_rate:.0%}) — "
                "consider tiling strategy changes"
            )
    elif valu_frac > BOTTLENECK_THRESHOLDS["compute_heavy"]:
        bottleneck = "compute-bound"
        confidence = min(valu_frac * 1.2, 0.95)
        recommendations.extend([
            "Consider reduced precision (FP8, INT8) if accuracy allows",
            "Look for algorithmic optimizations (fused operations)",
            "Explore Intrawave scheduling for compute-dense kernels",
        ])
    elif lds_frac > BOTTLENECK_THRESHOLDS["lds_bound"]:
        bottleneck = "lds-bound"
        confidence = min(lds_frac * 2.0, 0.95)
        recommendations.extend([
            "Reduce LDS usage per workgroup",
            "Consider single-LDS-barrier pipeline (Interwave)",
            "Check for bank conflicts in shared memory access patterns",
        ])
    else:
        recommendations.append("No single bottleneck dominates — profile at kernel level")

    if wait_per_wave > 100:
        recommendations.append(
            f"High wait cycles per wave ({wait_per_wave:.0f}) — "
            "latency hiding may help (more waves, prefetching)"
        )

    return ProfilingResult(
        counters=counters,
        bottleneck=bottleneck,
        bottleneck_confidence=confidence,
        recommendations=recommendations,
        raw_csv_path=csv_path,
        duration_s=duration_s,
    )


def format_profiling_for_prompt(result: ProfilingResult) -> str:
    """Format profiling results as context for LLM prompts."""
    if result.bottleneck in ("unavailable", "timeout", "error", "no_output"):
        return f"[Hardware profiling: {result.bottleneck}]\n"

    lines = [
        f"Hardware Profiling Results (rocprof, {result.duration_s:.1f}s):",
        f"  Bottleneck: {result.bottleneck} (confidence: {result.bottleneck_confidence:.0%})",
    ]
    if result.counters:
        lines.append("  Key counters:")
        for k, v in sorted(result.counters.items()):
            if v > 0:
                lines.append(f"    {k}: {v:,.0f}")
    if result.recommendations:
        lines.append("  Recommendations:")
        for r in result.recommendations:
            lines.append(f"    - {r}")
    return "\n".join(lines) + "\n"
