# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``target_analysis`` ActionRunner — external baseline comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.env import env_int, env_str
from hyperloom.inference_optimizer.baseline_comparison.target_analyzer import (
    analyze,
    clear_competitor_target,
)
from hyperloom.inference_optimizer.baseline_comparison.types import BenchmarkMode, TargetAnalysisRequest
from ...loop.sub_agent_runner import RunnerContext


log = logging.getLogger(__name__)


class TargetAnalysisExecutor:
    """ActionRunner for the ``target_analysis`` action kind."""

    def __init__(self, *, compare_against_gpu: str):
        """Initialize the executor with the pinned comparison reference."""
        self.compare_against_gpu = (compare_against_gpu or "").strip()

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        """Run the external-baseline comparison and persist report artefacts."""
        params = dict(ctx.task.params or {})

        from hyperloom.common.perf_metric import agentx_active

        state = (getattr(ctx, "extra", None) or {}).get("shared_state")
        benchmark_mode: BenchmarkMode = (
            "agentx" if agentx_active(benchmark_mode=getattr(state, "benchmark_mode", "")) else "synthetic"
        )
        model_path = str(params.get("model_path") or getattr(state, "model_path", "") or env_str("MODEL_PATH"))

        session_dir = Path(ctx.extra["session_dir"])
        compare_against_gpu = str(params.get("compare_against_gpu") or self.compare_against_gpu or "").strip()
        framework = str(params.get("framework") or getattr(state, "framework", "") or env_str("FRAMEWORK"))
        precision = str(params.get("precision") or env_str("PRECISION") or getattr(state, "precision", ""))
        isl = int(params.get("isl") or env_int("ISL", 0))
        osl = int(params.get("osl") or env_int("OSL", 0))

        try:
            summary = analyze(
                session_dir=session_dir,
                request=TargetAnalysisRequest(
                    model_path=model_path,
                    compare_against_gpu=compare_against_gpu,
                    framework=framework,
                    precision=precision,
                    isl=isl,
                    osl=osl,
                    benchmark_mode=benchmark_mode,
                ),
            )
        except Exception:
            clear_competitor_target(session_dir)
            raise
        return self._format_result(ctx, summary, session_dir)

    def _format_result(
        self,
        ctx: RunnerContext,
        summary: Any,
        session_dir: Path,
    ) -> dict[str, Any]:
        """Build the small bus-friendly result payload (pointer + status; the heavy JSON stays on disk)."""
        from hyperloom.inference_optimizer.session.session_paths import target_analysis_report_md, target_baseline_json

        json_path = target_baseline_json(session_dir)
        md_path = target_analysis_report_md(session_dir)
        out = {
            "status": "succeeded",
            "kind": ctx.task.kind,
            "baseline_status": getattr(summary, "status", "unknown"),
            "reason": getattr(summary, "reason", ""),
            "warning": getattr(summary, "warning", ""),
            "row_count": getattr(summary, "row_count", 0),
            "json_path": str(json_path),
            "md_path": str(md_path),
        }
        best = getattr(summary, "best", None)
        if best is not None:
            out["best_tput_per_gpu"] = best.tput_per_gpu
            out["best_conc"] = best.conc
            out["best_decode_tp"] = best.decode_tp
            if getattr(getattr(summary, "query", None), "benchmark_mode", "synthetic") == "agentx":
                out["best_e2e_norm_intvty_p90"] = best.e2e_norm_intvty_p90
                out["best_benchmark_id"] = best.benchmark_id
        log.info(
            "target_analysis_executor: status=%s reason=%s rows=%d (%s)",
            out["baseline_status"],
            out["reason"] or "-",
            out["row_count"],
            out["warning"] or "ok",
        )
        return out


__all__ = ["TargetAnalysisExecutor"]
