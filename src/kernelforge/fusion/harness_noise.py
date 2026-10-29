# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much one kernel-validation harness varies on this machine: the forge-fuse noise diagnostic."""

from __future__ import annotations

import logging
import statistics

from .models import Recipe
from .validate import HarnessKernelRunner

log = logging.getLogger("forge_fusion")


def measure_harness_noise(
    *,
    harness: str,
    repeat: int = 20,
    gpu: str = "0",
    env_flags: tuple[str, ...] = (),
    workdir: str = ".",
) -> dict[str, object]:
    """Measure how much the same harness varies on this machine."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    flags = dict.fromkeys(env_flags, "1")
    speedups: list[float] = []
    eager: list[float] = []
    fused: list[float] = []
    failures = 0
    for i in range(max(1, repeat)):
        runner = HarnessKernelRunner(
            harness_path=harness,
            workdir=workdir,
            framework_root=workdir,
            gpu=gpu,
            env_flags=flags,
        )
        bench = runner.microbench(
            Recipe(
                pattern_id="noise-probe",
                description="",
                env_flag=" ".join(env_flags),
                source_file="",
                source_hints=[],
                fusion_math="",
                eager_reference_hint="",
                shapes={},
                matched_categories=[],
                trigger_share=0.0,
            )
        )
        if bench.skipped or not bench.eager_us or not bench.fused_us:
            failures += 1
            continue
        eager.append(float(bench.eager_us))
        fused.append(float(bench.fused_us))
        speedups.append(float(bench.eager_us) / float(bench.fused_us))
        log.info("run %d/%d: %.4fx", i + 1, repeat, speedups[-1])

    report: dict[str, object] = {"runs": repeat, "usable": len(speedups), "failed": failures}
    if len(speedups) >= 2:
        mean = statistics.fmean(speedups)
        sd = statistics.stdev(speedups)
        cv = sd / mean if mean else 0.0
        report.update(
            {
                "speedup_mean": round(mean, 4),
                "speedup_stdev": round(sd, 5),
                "speedup_cv": round(cv, 5),
                "speedup_min": round(min(speedups), 4),
                "speedup_max": round(max(speedups), 4),
                "spread_pct": round((max(speedups) - min(speedups)) / mean * 100.0, 2),
                "eager_us_mean": round(statistics.fmean(eager), 3),
                "fused_us_mean": round(statistics.fmean(fused), 3),
                # How far out the 3% improvement bar sits.
                "bar_in_sigmas": round(0.03 / (cv * (2**0.5)), 2) if cv else None,
                "verdict": (
                    "the 3% bar is within noise"
                    if cv and 0.03 / (cv * (2**0.5)) < 2.0
                    else "the 3% bar is outside noise"
                ),
            }
        )
    return report
